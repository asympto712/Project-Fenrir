#![cfg(feature = "torch")]
#![allow(dead_code)]
#![allow(unused)]

use color_eyre::eyre::OptionExt;
use rand::thread_rng;
use rand::Rng;
use rand::seq::index::sample_weighted;

use rv::traits::Mode;
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
use crate::model::PVModel;
use crate::utils::BoardTensor;
use crate::utils::ModelInput;
use crate::utils::{MoveRepresentation, TBoard, Rotation, TAction, VectorBasedMove, DirectionalMove, ActionTensor};
use crate::model::Evaluation;
use crate::model::PVNet;
use crate::self_play::{Batch, Directer, Request, Query};
use game::board::TaflBoard;
use game::game::{Game, SimpleGame, GameState, GameLogic, Side, ShortHistory, get_rep_counter_for_oldest};
use bitboard::Direction;
use bitboard::{BitBoard, PositionalEncoding, MoveOnBoard};
use bitboard::eleven::{BoardEleven, MoveOnBoardEleven};
use bitboard::seven::{BoardSeven, MoveOnBoardSeven};

// multi-threading related 
use crossbeam::channel::{unbounded, Sender, Receiver};
use std::borrow::Borrow;
use std::thread;
use std::sync::Arc;
use rayon::{iter::{IntoParallelIterator, ParallelIterator}};

// std
use std::rc::Rc;

pub type PosteriorDist<B: BitBoard> = Vec<(<B as BitBoard>::Movement, f32)>;

#[derive(Debug, Copy, Clone)]
pub struct MCTSConfig {
    c_puct: f32,
    n_sim: usize,
    dirichlet_alpha: f64,
    dirichlet_epsilon: f32,
    temp_schedule_target_policy: fn(usize) -> Temperature,
    temp_schedule_move_selection: fn(usize) -> Temperature
}

impl Default for MCTSConfig {
    fn default() -> Self {
        let c_puct: f32 = 0.3;
        let n_sim: usize = 1600;
        let dirichlet_alpha: f64 = 0.03;
        let dirichlet_epsilon: f32 = 0.25;
        fn temp_schedule_target_policy(i: usize) -> Temperature {
            Temperature::Temp(1.0)
        }
        fn temp_schedule_move_selection(i: usize) -> Temperature {
            if i < 30 {
                Temperature::Temp(1.0)
            } else {
                Temperature::Zero
            }
        }
        MCTSConfig { 
            c_puct,
            n_sim,
            dirichlet_alpha,
            dirichlet_epsilon,
            temp_schedule_target_policy,
            temp_schedule_move_selection
        }
    }
}

pub trait Oracle<G: GameLogic> 
where 
TBoard<G>: ModelInput<G>,
TAction<G::B>: ActionTensor,
{
    fn infer(&self, game: &G, actions: Option<Vec<&<G::B as BitBoard>::Movement>>) -> Result<(Tensor, f32)>;
}

pub struct NewActor {
    sender: Arc<Sender<Request>>,
}

impl<G: GameLogic> Oracle<G> for NewActor
where
TBoard<G>: ModelInput<G>,
TAction<G::B>: ActionTensor
{
    // returns the distribution of the valid actions and the value
    fn infer(&self, game: &G, actions: Option<Vec<&<G::B as BitBoard>::Movement>>) -> Result<(Tensor, f32)> {
        let (s, r) = unbounded::<Evaluation>();
        let mut rng = thread_rng();
        let k: u8 = rng.gen_range(0..4);
        let query: Query = <TBoard<G> as ModelInput<G>>::get_pnet_input(game, Rotation::Do(k), (Kind::Float, Device::Cpu)).get();
        let request: Request = Request::new(query, Arc::new(s));
        self.sender.send(request)
            .map_err(|_| ErrReport::msg("Could not send the inference request"))?;

        let (pre_dist, pre_value) = r.recv()?;
        // Not sure, but the received result should have the shape
        // [1,20,11,11] and [1,1]
        // Calling squeeze just in case
        let value: f32 = pre_value.squeeze().f_double_value(&[0])? as f32;
        
        // mask has shape [20,11,11]
        let mask = if let Some(mobs) = actions {
            let vbms: Vec<VectorBasedMove> = mobs.iter()
                .map(|&mob| <VectorBasedMove as MoveRepresentation<G::B>>::convert_from(mob))
                .collect::<Result<Vec<_>>>()?;
            TAction::<G::B>::vec_vbm_one_hot_encode(&vbms)
        } else {
            let vbms: Vec<VectorBasedMove> = game.get_possible_actions()
                .into_iter().map(|mbe| <VectorBasedMove as MoveRepresentation<G::B>>::convert_from(&mbe))
                .collect::<Result<Vec<_>>>()?;
            TAction::<G::B>::vec_vbm_one_hot_encode(&vbms)
        };

        let pre_dist = pre_dist.squeeze().rot90(-1 * k as i64, [-2,-1]);
        let dist = &pre_dist * mask.get().to_kind(Kind::Float);
        let sum: f64 = dist.sum(Kind::Float).double_value(&[0]);
        let dist = dist.divide_scalar(sum);
        Ok((dist, value))

    }
}

impl NewActor {
    pub fn new(sender: Arc<Sender<Request>>) -> Self {
        Self { sender }
    }
}

// Might be replace by NewActor...
pub struct Actor {
    sender: Arc<Sender<(Request, usize)>>,
    // model id will be passed to the director along with the query to determine which batch to push it into.
    model_id: usize
}

// Might be deprecated
impl<G: GameLogic> Oracle<G> for Actor 
where 
TBoard<G>: ModelInput<G>,
TAction<<G as GameLogic>::B>: ActionTensor 
{

    // returns the distribution of the valid actions and the value
    fn infer(&self, game: &G, actions: Option<Vec<&<G::B as BitBoard>::Movement>>) -> Result<(Tensor, f32)> {
        let (s, r) = unbounded::<Evaluation>();
        let mut rng = thread_rng();
        let k: u8 = rng.gen_range(0..4);
        let query: Query = <TBoard<G> as ModelInput<G>>::get_pnet_input(game, Rotation::Do(k), (Kind::Float, Device::Cpu)).get();
        let request: Request = Request::new(query, Arc::new(s));
        self.sender.send((request, self.model_id))
            .map_err(|_| ErrReport::msg("Could not send the inference request"))?;


        let (pre_dist, pre_value) = r.recv()?;
        // Not sure, but the received result should have the shape
        // [1,20,11,11] and [1,1]
        // Calling squeeze just in case
        let value: f32 = pre_value.squeeze().f_double_value(&[0])? as f32;
        
        // mask has shape [20,11,11]
        let mask = if let Some(mbes) = actions {
            let vbms: Vec<VectorBasedMove> = mbes.iter()
                .map(|&mbe| <VectorBasedMove as MoveRepresentation<G::B>>::convert_from(mbe))
                .collect::<Result<Vec<_>>>()?;
            TAction::<G::B>::vec_vbm_one_hot_encode(&vbms)
        } else {
            let vbms: Vec<VectorBasedMove> = game.get_possible_actions()
                .into_iter().map(|mbe| <VectorBasedMove as MoveRepresentation<G::B>>::convert_from(&mbe))
                .collect::<Result<Vec<_>>>()?;
            TAction::<G::B>::vec_vbm_one_hot_encode(&vbms)
        };

        let pre_dist = pre_dist.squeeze().rot90(-1 * k as i64, [-2,-1]);
        let dist = &pre_dist * mask.get().to_kind(Kind::Float);
        let sum: f64 = dist.sum(Kind::Float).double_value(&[0]);
        let dist = dist.divide_scalar(sum);
        Ok((dist, value))

    }
}

impl Actor {
    pub fn new_with_model_name<P: PVModel>(directer: &Directer<P>, sender: Arc<Sender<(Request, usize)>>, model_name: String) -> Result<Self> {
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

pub fn get_vec_priors<B: BitBoard>(dist: &Tensor, ordered_actions: &Vec<B::Movement>) -> Result<Vec<f32>> {
    let flattened_dist = dist.flatten(0, -1);
    let priors: Vec<f32> = ordered_actions.iter()
        .map(|mbe| {
            let vbm = <VectorBasedMove as MoveRepresentation<B>>::convert_from(mbe)?;
            let index = <VectorBasedMove as DirectionalMove<B>>::to_index(&vbm);
            let p: f32 = flattened_dist.double_value(&[index]) as f32;
            Ok(p)
        }).collect::<Result<Vec<_>>>()?;
    Ok(priors)
}

#[derive(Debug)]
pub struct Node<G: GameLogic> {
    game: G,
    actions: Vec<<G::B as BitBoard>::Movement>,
    // edges: Option<Vec<&Edges>>,
    edges: Vec<Option<Edge<G>>>,
    visit_count: f32,
}

impl<G: GameLogic> Node<G> {
    pub fn new(game: &G, actions: Vec<<G::B as BitBoard>::Movement>, edges: Vec<Option<Edge<G>>>) -> Result<Self> {

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

    fn generate(game: G, actions: Vec<<G::B as BitBoard>::Movement>) -> Self {
        let num_actions = actions.len().to_owned();
        Self { 
            game,
            actions,
            edges: (0..num_actions).into_iter().map(|i| None).collect(),
            visit_count: 0.0,
        }
    }

    fn expand_selectively(&mut self, action_idx: usize, prior: f32) -> Result<()>{
        let new_edge = Edge::<G>::from_node_action_id(self, action_idx, prior)?;
        self.edges[action_idx] = Some(new_edge);
        Ok(())
    }

    fn expand(&mut self, priors: Vec<f32>) -> Result<()> {
        let edges: Result<Vec<_>> = self.actions.iter()
            .zip(priors)
            .map(|(a, prior)| Edge::<G>::from_node_action(self, a, prior).map(Some))
            .collect();
        self.edges = edges?;
        Ok(())
    }

    fn get_edge(&self, action_idx: usize) -> Option<&Edge<G>> {
        self.edges[action_idx].as_ref()
    }

    fn get_edge_mut(&mut self, action_idx: usize) -> Option<&mut Edge<G>> {
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

    fn generate_dist_from_visit_counts(&self, temperature: Temperature) -> Result<PosteriorDist<G::B>> {

        let dist = match temperature {
            Temperature::Temp(temp) => {

                let tempered_visit_count = self.edges
                    .iter()
                    .map(|edge_opt| {
                        let edge = edge_opt.as_ref().unwrap();
                        edge.visit_count.powf(1.0 / temp)
                    });
                
                let sum: f32 = tempered_visit_count.clone().sum();

                let dist: Vec<(<G::B as BitBoard>::Movement, f32)> = tempered_visit_count
                    .enumerate()
                    .map(|(i, f)| (self.actions[i].clone(), f / sum))
                    .collect();

                dist
            },
            Temperature::Zero => {

                let argmax = self.get_edge_id_w_highest_visit_count()?;

                let dist: Vec<(<G::B as BitBoard>::Movement, f32)> = self.actions
                    .iter()
                    .enumerate()
                    .map(|(i, action)| {
                        if i == argmax { (action.clone(), 1.0) }
                        else { (action.clone(), 0.0) }
                    })
                    .collect();
                
                dist
            }
        };
        Ok(dist)
    }

    // This will drain self.actions, leaving it empty
    fn generate_dist_from_visit_count_destructively(&mut self, temperature: Temperature)
    -> Result<PosteriorDist<G::B>> {

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

                let dist: Vec<(<G::B as BitBoard>::Movement, f32)> = self.actions
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
pub struct Edge<G: GameLogic> {
    child: Rc<Node<G>>,
    prior: f32,
    acc_value: f32,
    q_value: f32,
    visit_count: f32,
}

impl<G: GameLogic> Edge<G> {
    fn new(child: Rc<Node<G>>, prior: f32, acc_value: f32, q_value: f32, visit_count: f32) -> Self {
        Self {
            child,
            prior,
            q_value,
            acc_value,
            visit_count,
        }
    }

    fn from_node_action_id(node: &Node<G>, action_idx: usize, prior: f32) -> Result<Self> {
        let action = node.actions.get(action_idx)
            .ok_or(ErrReport::msg("action index out of bounds"))?;
        Edge::<G>::from_node_action(node, action, prior)
    }

    fn from_node_action(node: &Node<G>, action: &<G::B as BitBoard>::Movement, prior: f32) -> Result<Self> {

        let (new_game,
            reason,
            next_actions
        ) = node.game.do_move_and_update_whole(action, None, true)?;
        
        let next_actions = next_actions
            .unwrap_or(vec![]);

        let new_node = Node::<G>::generate(new_game, next_actions);
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
pub struct MCTSTree<G: GameLogic> {
    root: Rc<Node<G>>,
    turn_count: usize,
    n_sim: usize,
    c_puct: f32,
    alpha: f64,
    epsilon: f32,
    temp_schedule_target_policy: fn(usize) -> Temperature,
    temp_schedule_move_selection: fn(usize) -> Temperature
}

impl<G: GameLogic> Default for MCTSTree<G>
where TBoard<G>: ModelInput<G>,
TAction<G::B>: ActionTensor {
    fn default() -> Self {
        let game: G = G::default();
        let config: MCTSConfig = MCTSConfig::default();
        Self::generate(game, config)
    }
}

impl<G: GameLogic> MCTSTree<G> where TBoard<G>: ModelInput<G>,
TAction<G::B>: ActionTensor {

    pub fn generate(game: G, config: MCTSConfig) -> Self {
        let actions = game.get_possible_actions();
        let turn_count = game.get_state().get_turn_count() as usize;
        let root = Node::<G>::generate(game, actions);
        let n_sim = config.n_sim;
        let c_puct = config.c_puct;
        let alpha = config.dirichlet_alpha;
        let epsilon = config.dirichlet_epsilon;
        let temp_schedule_target_policy = config.temp_schedule_target_policy;
        let temp_schedule_move_selection = config.temp_schedule_move_selection;
        MCTSTree::<G>{
            root: root.into(),
            turn_count,
            n_sim,
            c_puct,
            alpha,
            epsilon,
            temp_schedule_target_policy,
            temp_schedule_move_selection,
        }
    }

    pub fn root_is_terminal(&self) -> bool {
        !self.root.game.get_state().is_ongoing()
    }

    pub fn get_winner(&self) -> Side {
        self.root.game.get_state().get_victor()
    }

    // traverse the tree from root and find the leaf node, and returns the edge that leads to the (potentially) new node
    fn traverse(&self) -> Option<(&Node<G>, Vec<usize>)> {
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

    fn traverse_and_as_ref_mut(&mut self) -> Option<(Rc<Node<G>>, Vec<usize>)> {

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

    fn expand_from_leaf<O: Oracle<G>>(leaf: &mut Node<G>, actor: &O) -> Result<f32> {

        // When the leaf node is a terminal state, we just return the reward
        if !leaf.game.get_state().is_ongoing() {
            let value = match leaf.game.get_state().get_victor() {
                Side::Att => 1.0,
                Side::Def => -1.0,
            };
            return Ok(value)
        }

        let actions: Vec<&<G::B as BitBoard>::Movement> = leaf.actions.iter().collect();
        let (dist, value) = actor.infer(&leaf.game, Some(actions))?;
        let priors = get_vec_priors::<G::B>(&dist, &leaf.actions)?;

        leaf.expand(priors)?;
        Ok(value)
    }

    fn backup(&mut self, value: f32, path: Vec<usize>) -> Result<()>{
        let edges_to_update = path.into_iter()
            .fold(self.root.clone(), |mut rc_node, action_idx| {
                let node = Rc::get_mut(&mut rc_node).unwrap();
                let player: Side = node.game.get_state().show_side();
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

    fn get_posterior(&self) -> Result<PosteriorDist<G::B>> {

        let temperature = (self.temp_schedule_target_policy)(self.turn_count);
        self.root.generate_dist_from_visit_counts(temperature)

    }


    fn get_posterior_w_sampled_action_index(&self) 
    -> Result<(PosteriorDist<G::B>, usize)>
    {

        let temp_posterior = (self.temp_schedule_target_policy)(self.turn_count);
        let temp_move_selection = (self.temp_schedule_move_selection)(self.turn_count);
        let dist = self.root
            .generate_dist_from_visit_counts(temp_posterior)?;

        match temp_move_selection {

            Temperature::Temp(temp) => {

                let mut rng = thread_rng();
                let index_vec = sample_weighted(
                    &mut rng,
                    dist.len(),
                    |i| dist[i].1,
                    1
                )?;
                debug_assert_eq!(index_vec.len(), 1);
                let sampled_action_idx = index_vec.index(0);
                
                Ok((dist, sampled_action_idx))
            }
            Temperature::Zero => {

                let argmax = self.root.get_edge_id_w_highest_visit_count()?;

                Ok((dist, argmax))
            }
        }
    }

    fn get_posterior_w_sampled_action_index_destructively(&mut self) 
    -> Result<(PosteriorDist<G::B>, usize)>
    {

        let temp_posterior = (self.temp_schedule_target_policy)(self.turn_count);
        let temp_move_selection = (self.temp_schedule_move_selection)(self.turn_count);
        let dist = {
            let mut_ref_root = Rc::get_mut(&mut self.root)
                .ok_or_eyre("Tried to get mutable reference to the tree root when there are more than one reference to it")?;
            mut_ref_root
                .generate_dist_from_visit_count_destructively(temp_posterior)?
        };
        match temp_move_selection {

            Temperature::Temp(temp) => {

                let mut rng = thread_rng();
                let index_vec = sample_weighted(
                    &mut rng,
                    dist.len(),
                    |i| dist[i].1,
                    1
                )?;
                debug_assert_eq!(index_vec.len(), 1);
                let sampled_action_idx = index_vec.index(0);
                
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

    fn trim_root(&mut self, action_id: usize) -> Result<(G, <G::B as BitBoard>::Movement)> {

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
        let cur_game = std::mem::replace::<G>(&mut cur_root.game, <G as GameLogic>::ghastly());

        dbg!(Rc::strong_count(&self.root));
        dbg!(Rc::weak_count(&self.root));

        // Now the only remaining reference to the old root (held by self) is dropped, so the old root will disappear
        self.root = next_root;

        dbg!(Rc::strong_count(&self.root));
        dbg!(Rc::weak_count(&self.root));

        // Do these two still remain valid (still own the values)? I'm not sure
        Ok((cur_game, chosen_action))
    }

    fn search_expand_backup<O: Oracle<G>>(&mut self, actor: &O) -> Result<()> {

        let (mut rc_leaf, path) = self.traverse_and_as_ref_mut()
            .ok_or_eyre("Traversing the MCTS tree didn't work")?;

        let leaf = Rc::get_mut(&mut rc_leaf)
            .ok_or_eyre("Cannot get mutable reference to leaf node")?;

        let value = Self::expand_from_leaf::<O>(leaf, actor)?;

        self.backup(value, path)?;

        Ok(())
    } 

    pub fn get_policy_and_update<O: Oracle<G>>(&mut self, actor: &O) 
    -> Result<(G, <G::B as BitBoard>::Movement, PosteriorDist<G::B>)> 
    {

        for _ in 0..self.n_sim {
            self.search_expand_backup::<O>(actor)?;
        }

        let (posterior, action_id) 
        = self.get_posterior_w_sampled_action_index_destructively()?;

        let (game, action) = self.trim_root(action_id)?;
        self.turn_count += 1;

        Ok((game, action, posterior))
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
        let mut tree = MCTSTree::<Game>::default();

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

