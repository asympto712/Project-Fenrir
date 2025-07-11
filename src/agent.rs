#![cfg(feature = "torch")]
#![allow(dead_code)]
#![allow(unused)]

use color_eyre::eyre::OptionExt;
use rand::thread_rng;
use rand::Rng;
use rand::seq::index::sample_weighted;

// external
use tch::Tensor;
use tch::Kind;
use tch::Device;
use color_eyre::eyre::Result;
use color_eyre::eyre::WrapErr;
use color_eyre::eyre::ErrReport;
use rand::prelude::*;
use rv::dist::SymmetricDirichlet;
use rv::traits::Sampleable;

// internal
use game::game::Game;
use game::board::TaflBoardEleven;
use game::game::GameState;
use game::game::Side;
use bitboard::eleven::{ElevenBoardPositionalEncoding, MoveOnBoardEleven, BoardEleven};
use crate::utils::{MoveRepresentation, TBoardEleven, Rotation, TAction, VectorBasedMove};
use crate::model::Evaluation;
use crate::model::PVNet;
use crate::self_play::{Batch, Directer, Request, Query};

// multi-threading related 
use crossbeam::channel::{unbounded, Sender, Receiver};
use std::thread;
use std::sync::Arc;
use rayon::{iter::{IntoParallelIterator, ParallelIterator}};

// std
use std::rc::Rc;

pub trait Oracle {
    fn infer(&self, game: &Game, actions: Option<Vec<&MoveOnBoardEleven>>) -> Result<(Tensor, f32)>;
}

pub struct Actor {
    sender: Arc<Sender<(Request, usize)>>,
    // model id will be passed to the director along with the query to determine which batch to push it into.
    model_id: usize
}

impl Oracle for Actor {

    // returns logits of the valid actions and the value
    fn infer(&self, game: &Game, actions: Option<Vec<&MoveOnBoardEleven>>) -> Result<(Tensor, f32)> {
        let (s, r) = unbounded::<Evaluation>();
        let mut rng = thread_rng();
        let k: u8 = rng.gen_range(0..4);
        let query: Query = TBoardEleven::get_pnet_input(game, Rotation::Do(k), (Kind::Float, Device::Cpu)).get();
        let request: Request = Request::new(query, Arc::new(s));
        self.sender.send((request, self.model_id))
            .map_err(|_| ErrReport::msg("Could not send the inference request"))?;


        let (pre_logits, pre_value) = r.recv()?;
        // Not sure, but the received result should have the shape
        // [1,20,11,11] and [1,1]
        // Calling squeeze just in case
        let value: f32 = pre_value.squeeze().f_double_value(&[0])? as f32;
        
        // mask has shape [20,11,11]
        let mask = if let Some(mbes) = actions {
            let vbms: Vec<VectorBasedMove> = mbes.iter()
                .map(|&mbe| VectorBasedMove::convert_from(mbe))
                .collect::<Result<Vec<_>>>()?;
            TAction::vec_vbm_one_hot_encode(&vbms)
        } else {
            let vbms: Vec<VectorBasedMove> = game.get_possible_actions()
                .into_iter().map(|mbe| VectorBasedMove::convert_from(&mbe))
                .collect::<Result<Vec<_>>>()?;
            TAction::vec_vbm_one_hot_encode(&vbms)
        };

        let pre_logits = pre_logits.squeeze();
        let logits = &pre_logits * mask.get().to_kind(Kind::Float);
        let sum: f64 = logits.sum(Kind::Float).double_value(&[0]);
        let logits = logits.divide_scalar(sum);
        Ok((logits, value))

    }
}

impl Actor {
    pub fn new_with_model_name(directer: &Directer, sender: Arc<Sender<(Request, usize)>>, model_name: String) -> Result<Self> {
        let model_id = directer.lookup_id_from_model(model_name)?;
        let sender = sender.clone();
        Ok(Self{
            sender,
            model_id,
        })
    }

    pub fn new_with_model_id(sender: Arc<Sender<(Request, usize)>>, model_id: usize) -> Self {
        Self { 
            sender,
            model_id
        }
    }

}

pub fn get_vec_priors(logits: &Tensor, ordered_actions: &Vec<MoveOnBoardEleven>) -> Result<Vec<f32>> {
    let flattened_logits = logits.flatten(0, -1);
    let priors: Vec<f32> = ordered_actions.iter()
        .map(|mbe| {
            let vbm = VectorBasedMove::convert_from(mbe)?;
            let index = vbm.to_index();
            let p: f32 = flattened_logits.double_value(&[index]) as f32;
            Ok(p)
        }).collect::<Result<Vec<_>>>()?;
    Ok(priors)
}

#[derive(Debug)]
pub struct Node {
    game: Game,
    actions: Vec<MoveOnBoardEleven>,
    // edges: Option<Vec<&Edges>>,
    edges: Vec<Option<Edge>>,
    visit_count: f32,
}

impl Node {
    pub fn new(game: &Game, actions: Vec<MoveOnBoardEleven>, edges: Vec<Option<Edge>>) -> Result<Self> {

        if actions.len() != edges.len() {
            return Err(ErrReport::msg("a node must have the same number of actions and edges on creation"));
        }
        Ok(Self {
            game: game.clone(),
            actions,
            edges,
            visit_count: 0.0,
        })
    }

    fn generate(game: Game, actions: Vec<MoveOnBoardEleven>) -> Self {
        let num_actions = actions.len().to_owned();
        Self { 
            game,
            actions,
            edges: (0..num_actions).into_iter().map(|i| None).collect(),
            visit_count: 0.0,
        }
    }

    fn expand_selectively(&mut self, action_idx: usize, prior: f32) -> Result<()>{
        let new_edge = Edge::from_node_action_id(self, action_idx, prior)?;
        self.edges[action_idx] = Some(new_edge);
        Ok(())
    }

    fn expand(&mut self, priors: Vec<f32>) -> Result<()> {
        let edges: Result<Vec<_>> = self.actions.iter()
            .zip(priors)
            .map(|(a, prior)| Edge::from_node_action(self, a, prior).map(Some))
            .collect();
        self.edges = edges?;
        Ok(())
    }

    fn get_edge(&self, action_idx: usize) -> Option<&Edge> {
        self.edges[action_idx].as_ref()
    }

    fn get_edge_mut(&mut self, action_idx: usize) -> Option<&mut Edge> {
        self.edges.get_mut(action_idx)?.as_mut()
    }

    // if all of its edges are None, it is a leaf node
    fn is_leaf(&self) -> bool {
        self.edges.iter().all(|edge| edge.is_none())
    }

    fn select_action_critically(&self, c_puct: f32) -> Option<usize> {

        let criteria = self.edges
            .iter()
            .enumerate()
            .filter_map(|(idx, edge_opt)| {
                edge_opt.as_ref().map(|edge| {
                    let score = 
                    edge.q() + 
                    c_puct *
                    edge.prior * 
                    self.visit_count.powf(0.5.into()) / (1.0 + edge.n());
                    (idx, score)
                })
        });

        let best_action_idx = criteria.max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx);

        best_action_idx

    }

    // epsilon should be in range (0,1).
    // In AlphaGo Zero paper... alpha: 0.03, epsilon: 0.25
    fn select_action_critically_w_noise(&self, c_puct: f32, alpha: f64, epsilon: f32) -> Option<usize> {
        let k = self.actions.len();
        let dir = SymmetricDirichlet::new_unchecked(alpha, k);
        let mut rng = thread_rng();
        let noise: Vec<f64> = dir.draw(&mut rng);

        let criteria = self.edges
            .iter()
            .zip(noise)
            .enumerate()
            .filter_map(|(idx, (edge_opt, noise))| {
                edge_opt.as_ref().map(|edge| {
                    let score = 
                    edge.q() + 
                    c_puct *
                    ((1.0 - epsilon) * edge.prior + epsilon * noise as f32) * 
                    self.visit_count.powf(0.5.into()) / (1.0 + edge.n());
                    (idx, score)
                })
        });
        let best_action_idx = criteria.max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx);

        best_action_idx

    }

    fn get_edge_id_w_highest_visit_count(&self) -> Result<usize> {

        let selected_edge_id = self.edges
            .iter()
            .map(|edge_opt| {
                let edge = edge_opt.as_ref().unwrap();
                edge.visit_count
            })
            .enumerate()
            .max_by(|(_,a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i,_)| i)
            .ok_or_eyre("Could not find the argmax from the visit counts in the MCTS tree")?;
        Ok(selected_edge_id)
    }

    fn generate_dist_from_visit_counts(&self, temperature: Temperature) -> Result<Vec<(&MoveOnBoardEleven, f32)>> {

        let dist = match temperature {
            Temperature::Temp(temp) => {

                let tempered_visit_count = self.edges
                    .iter()
                    .map(|edge_opt| {
                        let edge = edge_opt.as_ref().unwrap();
                        edge.visit_count.powf(1.0 / temp)
                    });
                
                let sum: f32 = tempered_visit_count.clone().sum();

                let dist: Vec<(&MoveOnBoardEleven, f32)> = tempered_visit_count
                    .enumerate()
                    .map(|(i, f)| (self.actions.get(i).unwrap(), f / sum))
                    .collect();

                dist
            },
            Temperature::Zero => {

                let argmax = self.get_edge_id_w_highest_visit_count()?;

                let dist: Vec<(&MoveOnBoardEleven, f32)> = self.actions
                    .iter()
                    .enumerate()
                    .map(|(i, action)| {
                        if i == argmax { (action, 1.0) }
                        else { (action, 0.0) }
                    })
                    .collect();
                
                dist
            }
        };
        Ok(dist)
    }

    // This will drain self.actions, leaving it empty
    fn generate_dist_from_visit_count_destructively(&mut self, temperature: Temperature)
    -> Result<Vec<(MoveOnBoardEleven, f32)>> {

        let dist = match temperature {
            Temperature::Temp(temp) => {

                let tempered_visit_count = self.edges
                    .iter()
                    .map(|edge_opt| {
                        let edge = edge_opt.as_ref().unwrap();
                        edge.visit_count.powf(1.0 / temp)
                    });
                
                let sum: f32 = tempered_visit_count.clone().sum();
                let dist = self.actions
                    .drain(..)
                    .zip(tempered_visit_count)
                    .map(|(a,f)| (a, f / sum))
                    .collect::<Vec<_>>();

                dist
            },
            Temperature::Zero => {

                let argmax = self.get_edge_id_w_highest_visit_count()?;

                let dist: Vec<(MoveOnBoardEleven, f32)> = self.actions
                    .drain(..)
                    .enumerate()
                    .map(|(i, action)| {
                        if i == argmax { (action, 1.0) }
                        else { (action, 0.0) }
                    })
                    .collect();
                
                dist
            }
        };
        Ok(dist)
    }

}

#[derive(Debug)]
pub struct Edge {
    child: Rc<Node>,
    prior: f32,
    acc_value: f32,
    q_value: f32,
    visit_count: f32,
}

impl Edge {
    fn new(child: Rc<Node>, prior: f32, acc_value: f32, q_value: f32, visit_count: f32) -> Self {
        Self {
            child,
            prior,
            q_value,
            acc_value,
            visit_count,
        }
    }

    fn from_node_action_id(node: &Node, action_idx: usize, prior: f32) -> Result<Self> {
        let action = node.actions.get(action_idx)
            .ok_or(ErrReport::msg("action index out of bounds"))?;
        Edge::from_node_action(node, action, prior)
    }

    fn from_node_action(node: &Node, action: &MoveOnBoardEleven, prior: f32) -> Result<Self> {

        let (new_game,
            reason,
            next_actions
        ) = node.game.do_move_and_update_whole(action, None, true)?;
        
        let next_actions = next_actions
            .unwrap_or(vec![]);

        let new_node = Node::generate(new_game, next_actions);
        let child = Rc::new(new_node);
        Ok(Self {
            child,
            q_value: 0.0,
            visit_count: 0.0,
            acc_value: 0.0,
            prior,
        })
    }
    
    fn q(&self) -> f32 {
        self.q_value
    }

    fn n(&self) -> f32 {
        self.visit_count
    }

    // get the prior probability of that edge 
    // which should be given by the inference process
    fn prior(&self) -> f32 {
        self.prior
    }
}

#[derive(Debug)]
pub struct MCTSTree {
    root: Rc<Node>,
    c_puct: f32,
    alpha: f64,
    epsilon: f32,
    temperature: Temperature,
}

impl Default for MCTSTree {
    fn default() -> Self {
        MCTSTree::generate(1.0, 0.03, 0.25, Temperature::Temp(1.0))
    }
}

impl MCTSTree {

    pub fn generate(c_puct: f32, alpha: f64, epsilon: f32, temperature: Temperature) -> Self {
        let game = Game::default();
        let actions = game.get_possible_actions();
        let root = Node::generate(game, actions);
        MCTSTree::new(root, c_puct, alpha, epsilon, temperature)
    }

    pub fn new(root: Node, c_puct: f32, alpha: f64, epsilon: f32, temperature: Temperature) -> Self {
        Self {
            root: Rc::new(root),
            c_puct,
            alpha,
            epsilon,
            temperature
        }
    }

    pub fn root_is_terminal(&self) -> bool {
        !self.root.game.state.is_ongoing()
    }

    // traverse the tree from root and find the leaf node, and returns the edge that leads to the (potentially) new node
    fn traverse(&self) -> Option<(&Node, Vec<usize>)> {
        let mut cur_node = &*self.root;
        let mut path: Vec<usize> = vec![];
        let mut depth_count = 0;

        while !cur_node.is_leaf() {

            let best_action_idx = if depth_count == 0 {
                cur_node.select_action_critically_w_noise(self.c_puct, self.alpha, self.epsilon)
            } else {
                cur_node.select_action_critically(self.c_puct)
            };

            match best_action_idx {
                Some(idx) => {
                    if let Some(edge) = cur_node.get_edge(idx) {
                        cur_node = &*edge.child;
                        path.push(idx);
                    } else { return None; }
                }
                None => {return None;}
            }

            depth_count += 1;
        }
        Some((cur_node, path))
    }

    fn traverse_and_as_ref_mut(&mut self) -> Option<(Rc<Node>, Vec<usize>)> {

        let mut cur_node = self.root.clone();

        let mut path: Vec<usize> = vec![];
        let mut depth_count = 0;

        while !cur_node.is_leaf() {

            let best_action_idx = if depth_count == 0 {
                cur_node.select_action_critically_w_noise(self.c_puct, self.alpha, self.epsilon)
            } else {
                cur_node.select_action_critically(self.c_puct)
            };

            match best_action_idx {
                Some(idx) => {
                    if let Some(edge) = cur_node.get_edge(idx) {
                        cur_node = edge.child.clone();
                        path.push(idx);
                    } else { return None; }
                }
                None => {return None;}
            }

            depth_count += 1;
        }
        Some((cur_node, path))
    }

    fn expand_from_leaf(leaf: &mut Node, actor: &Actor) -> Result<f32> {

        // When the leaf node is a terminal state, we just return the reward
        if !leaf.game.state.is_ongoing() {
            let value = match leaf.game.state.get_victor() {
                Side::Att => 1.0,
                Side::Def => -1.0,
            };
            return Ok(value)
        }

        let actions: Vec<&MoveOnBoardEleven> = leaf.actions.iter().collect();
        let (logits, value) = actor.infer(&leaf.game, Some(actions))?;
        let priors = get_vec_priors(&logits, &leaf.actions)?;

        leaf.expand(priors)?;
        // for (id, (action, prior)) in leaf.actions.iter().zip(priors).enumerate() {
        //     let (new_game,
        //         _,
        //         opt_actions) 
        //         = leaf.game.do_move_and_update_whole(&action, None, true)?;

        //     let actions = opt_actions.unwrap();
        //     let uninitialized_edges = actions.iter().map(|_| None).collect();
        //     let node = Node::new(&new_game, actions, uninitialized_edges)?;
        //     let edge = Edge::new(
        //         Rc::new(node),
        //         prior,
        //         0.0,
        //         0.0,
        //         0.0);
            
        //     leaf.edges[id] = Some(edge);
        // }
        Ok(value)
    }

    fn backup(&mut self, value: f32, path: Vec<usize>) -> Result<()>{
        let edges_to_update = path.into_iter()
            .fold(self.root.clone(), |mut rc_node, action_idx| {
                let node = Rc::get_mut(&mut rc_node).unwrap();
                let player: Side = node.game.state.show_side();
                let edge = node.get_edge_mut(action_idx).unwrap();
                edge.visit_count += 1.0;
                edge.acc_value += match player{
                    Side::Att => value,
                    Side::Def => -1.0 * value,
                };
                edge.q_value = edge.acc_value / edge.visit_count;
                edge.child.clone()
            });
        Ok(())
    }

    fn get_posterior(&self) -> Result<Vec<(&MoveOnBoardEleven, f32)>> {

        self.root.generate_dist_from_visit_counts(self.temperature)

    }


    fn get_posterior_w_sampled_action_index(&self, greedy: bool) 
    -> Result<(Vec<(&MoveOnBoardEleven, f32)>, usize)>
    {

        let dist = self.root
            .generate_dist_from_visit_counts(self.temperature)?;

        match self.temperature {

            Temperature::Temp(temp) => {

                let sampled_action_idx = if greedy {

                    let argmax_action_idx = self.root.get_edge_id_w_highest_visit_count()?;

                    argmax_action_idx
                } else {

                    let mut rng = thread_rng();
                    let index_vec = sample_weighted(
                        &mut rng,
                        dist.len(),
                        |i| dist[i].1,
                        1
                    )?;
                    debug_assert_eq!(index_vec.len(), 1);
                    let index = index_vec.index(0);

                    index
                };
                
                Ok((dist, sampled_action_idx))
            }
            Temperature::Zero => {

                let argmax = self.root.get_edge_id_w_highest_visit_count()?;

                Ok((dist, argmax))
            }
        }
    }

    fn get_posterior_w_sampled_action_index_destructively(&mut self, greedy: bool) 
    -> Result<(Vec<(MoveOnBoardEleven, f32)>, usize)>
    {

        let dist = {
            let mut_ref_root = Rc::get_mut(&mut self.root)
                .ok_or_eyre("Tried to get mutable reference to the tree root when there are more than one reference to it")?;
            mut_ref_root
                .generate_dist_from_visit_count_destructively(self.temperature)?
        };
        // let mut_ref_root = Rc::get_mut(&mut self.root)
        //     .ok_or("Tried to get mutable reference to the tree root when there are more than one reference to it")?;
        // let dist = mut_ref_root
        //     .generate_dist_from_visit_count_destructively(self.temperature)?;

        match self.temperature {

            Temperature::Temp(temp) => {

                let sampled_action_idx = if greedy {

                    let argmax_action_idx = self.root.get_edge_id_w_highest_visit_count()?;

                    argmax_action_idx
                } else {

                    let mut rng = thread_rng();
                    let index_vec = sample_weighted(
                        &mut rng,
                        dist.len(),
                        |i| dist[i].1,
                        1
                    )?;
                    debug_assert_eq!(index_vec.len(), 1);
                    let index = index_vec.index(0);

                    index
                };
                
                Ok((dist, sampled_action_idx))
            }
            Temperature::Zero => {

                let argmax = self.root.get_edge_id_w_highest_visit_count()?;

                Ok((dist, argmax))
            }
        }
    }

    fn greedily_sample_next_action(&self) -> Result<usize> {
        self.root.get_edge_id_w_highest_visit_count()
    }

    fn trim_root(&mut self, action_id: usize) -> Result<(Game, MoveOnBoardEleven)> {

        dbg!(Rc::strong_count(&self.root));
        dbg!(Rc::weak_count(&self.root));
        let edge_opt = self.root.edges.get(action_id)
            .ok_or_eyre("Action index out of bounds")?;
        let edge = edge_opt
            .as_ref()
            .ok_or_eyre("The edge chosen from the root node in the MCTS tree was not initialized")?;
        let next_root = edge.child.clone();

        dbg!(Rc::strong_count(&self.root));
        dbg!(Rc::weak_count(&self.root));

        // now this should work, as self is the only one who holds reference to the root
        let cur_root= Rc::get_mut(&mut self.root)
            .ok_or_eyre("try_unwrap operation failed at trim_root")?;
        let chosen_action = cur_root.actions
            .drain(..)
            .nth(action_id)
            .ok_or_eyre("action index out of bounds")?;
        let cur_game = std::mem::replace::<Game>(&mut cur_root.game, Game::ghastly());

        dbg!(Rc::strong_count(&self.root));
        dbg!(Rc::weak_count(&self.root));

        // Now the only remaining reference to the old root (held by self) is dropped, so the old root will disappear
        self.root = next_root;

        dbg!(Rc::strong_count(&self.root));
        dbg!(Rc::weak_count(&self.root));

        // Do these two still remain valid (still own the values)? I'm not sure
        Ok((cur_game, chosen_action))
    }

    fn search_expand_backup(&mut self, actor: &Actor) -> Result<()> {

        let (mut rc_leaf, path) = self.traverse_and_as_ref_mut()
            .ok_or_eyre("Traversing the MCTS tree didn't work")?;

        let leaf = Rc::get_mut(&mut rc_leaf)
            .ok_or_eyre("Cannot get mutable reference to leaf node")?;

        let value = MCTSTree::expand_from_leaf(leaf, actor)?;

        self.backup(value, path)?;

        Ok(())
    } 

    pub fn get_policy_and_update(&mut self, actor: &Actor, n_simulation: usize, greedy: bool) 
    -> Result<Vec<(MoveOnBoardEleven, f32)>> 
    {

        for _ in 0..n_simulation {
            self.search_expand_backup(actor)?;
        }

        let (posterior, action_id) 
        = self.get_posterior_w_sampled_action_index_destructively(greedy)?;

        self.trim_root(action_id)?;

        Ok(posterior)
    }

}

#[derive(Clone, Copy, Debug)]
pub enum Temperature {
    Temp(f32),
    Zero
}

// pub fn mcts(game: &Game, vp_sender: &mut Arc<Sender<Tensor>>, vp_receiver: &mut Receiver<Tensor>) -> Tensor{
//     todo!()
// } 

#[cfg(test)]
mod tests {
    use super::*;
    use game::game::Game;
    use game::board::TaflBoardEleven;
    use std::rc::Rc;

    #[test]
    fn trim_root_works() {
        let mut tree = MCTSTree::default();

        {
            let root = Rc::get_mut(&mut tree.root).unwrap();
            root.expand_selectively(0, 0.0).unwrap();
        }

        let original_game = tree.root.game.clone();
        let original_action = tree.root.actions[0].clone();

        assert_eq!(Rc::strong_count(&tree.root), 1);

        println!("{:?}", tree);
        let (game, action) = tree.trim_root(0).unwrap();

        assert_eq!(game, original_game);
        assert_eq!(action, original_action);
        println!("{:?}", tree);
    }
}

