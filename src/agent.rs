#![cfg(feature = "torch")]
#![allow(dead_code)]
#![allow(unused)]

use game::board::TaflBoardEleven;
// external
use tch::Tensor;

// internal
use game::game::Game;
use game::game::GameState;
use bitboard::eleven::{ElevenBoardPositionalEncoding, MoveOnBoardEleven, BoardEleven};

// multi-threading related 
use crossbeam::channel::{unbounded, Sender, Receiver};
use std::intrinsics::powf32;
use std::thread;
use std::sync::Arc;
use rayon::{iter::{IntoParallelIterator, ParallelIterator}};

// std
use std::rc::Rc;

// struct Game_ {
//     b: TaflBoardEleven,
//     state: GameState,
//     recent_history: 
// }

struct Node {
    game: Game,
    actions: Vec<MoveOnBoardEleven>,
    // edges: Option<Vec<&Edges>>,
    edges: Vec<Option<Edge>>,
    visit_count: f32,
}

impl Node {
    fn new(game: &Game, actions: Option<Vec<MoveOnBoardEleven>>) -> Self {

        let actions: Vec<MoveOnBoardEleven> = 
        if let Some(actions) = actions {
            actions 
        } else { 
            game.get_possible_actions() 
        };
        Self {
            game: game.clone(),
            actions: actions,
            edges: (0..actions.len()).into_iter().map(|i| None).collect(),
            visit_count: 0.0,
        }
    }

    fn generate(game: Game, actions: Vec<MoveOnBoardEleven>) -> Self {
        Self { 
            game,
            actions,
            edges: (0..actions.len()).into_iter().map(|i| None).collect(),
            visit_count: 0.0,
        }
    }

    fn expand_selectively(&mut self, action_idx: usize) {
        let new_edge = Edge::from_node(self, action_idx);
        self.edges[action_idx] = Some(new_edge);
    }

    fn expand(&mut self) {
        self.edges = self.actions.iter().map(|a| Some(Edge::from_node_action(self, a)));
    }

    fn get_edge(&self, action_idx: usize) -> Option<&Edge> {
        self.edges[action_idx].as_ref()
    }

    fn is_leaf(&self) -> bool {
        let t = self.edges.iter().fold(true, |acc, e| {
            let is = if let Some(_) = e { true } else { false };
            acc && is
        });
        !t
    }

}

struct Edge {
    child: Rc<Node>,
    prior: f32,
    q_value: f32,
    visit_count: f32,
}

impl Edge {
    fn new(child: Rc<Node>) -> Self {
        Self {
            child: Rc::clone(child),
            q_value: 0.0,
            visit_count: 0.0,
            prior: 0.0,
        }
    }

    fn from_node(node: &Node, action_idx: usize) -> Self {
        let action = node.actions[action_idx];
        Edge::from_node_action(node, &action)
    }

    fn from_node_action(node: &Node, action: &MoveOnBoardEleven) -> Self {

        let (new_game,
            reason,
            next_actions
        ) = node.game.do_move_and_update_whole(action, None, true).unwrap();
        
        let next_actions = next_actions.unwrap();
        let new_node = Node::generate(new_game, next_actions);
        let child = Rc::new(new_node);
        Self {
            child,
            q_value: 0.0,
            visit_count: 0.0,
            prior: 0.0,
        }
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

pub struct MCTSTree {
    root: Rc<Node>,
    c_puct: f32,
}

impl MCTSTree {

    // traverse the tree from root and find the leaf node, and returns the edge that leads to the (potentially) new node
    pub fn traverse(&self) -> Option<&Node> {
        let mut cur_node = &*self.root;

        while !cur_node.is_leaf() {

            let criteria = cur_node.edges
                .iter()
                .enumerate()
                .filter_map(|(idx, edge_opt)| {
                    edge_opt.as_ref().map(|edge| {
                        let score = 
                        edge.q() + 
                        self.c_puct *
                        edge.prior() * 
                        ref_node.visit_count.powf(0.5.into()) / (1.0 + edge.n());
                        (idx, score)
                    })
            });
            let best_action_idx = criteria.max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx);

            match best_action_idx {
                Some(idx) => {
                    if let Some(edge) = cur_node.get_edge(idx) {
                        cur_node = &*edge.child;
                    } else { return None; }
                }
                None => {return None;}
            }
        }
        Some(cur_node)
    }
}

pub fn mcts(game: &Game, vp_sender: &mut Arc<Sender<Tensor>>, vp_receiver: &mut Receiver<Tensor>) -> Tensor{



    
} 

