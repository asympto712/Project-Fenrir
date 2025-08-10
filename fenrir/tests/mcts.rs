use fenrir::agent::{MCTSConfig, MCTSTree, Oracle};
use fenrir::utils::{ActionTensor, ModelInput, TBoard, TAction, VectorBasedMove, MoveRepresentation};
use fenrir::{visualization::*};
use game::game::{Game, GameLogic};

use rand::prelude::*;
use std::cell::RefCell;
use std::collections::VecDeque;

struct RandomActor<R: Rng> {
    rng: RefCell<R>
}

impl<G: GameLogic, R: Rng> Oracle<G> for RandomActor<R>
where 
TBoard<G>: ModelInput<G>,
TAction<G::B>: ActionTensor {
    fn infer(&self, game: &G, actions: Option<Vec<&<<G as GameLogic>::B as bitboard::BitBoard>::Movement>>) -> color_eyre::eyre::Result<(tch::Tensor, f32)> {

        if let Some(actions) = actions {
        if !actions.is_empty() {
            
            let mut r = self.rng.borrow_mut();
            let idx = r.gen_range(0..actions.len());
            let a = actions[idx];
            let vbm = <VectorBasedMove as MoveRepresentation<G::B>>::convert_from(a)?;
            let ta = <TAction<G::B> as ActionTensor>::vbm_one_hot_encode(&vbm);
            Ok((ta.inner(), 0.0f32))
        } else {
            let ta = <TAction::<G::B> as ActionTensor>::vec_vbm_one_hot_encode(&vec![]);
            Ok((ta.inner(), 0.0f32))
        }
        } else {
            let actions = game.get_possible_actions();
            if !actions.is_empty() {

                let mut r = self.rng.borrow_mut();
                let idx = r.gen_range(0..actions.len());
                let a = &actions[idx];
                let vbm = <VectorBasedMove as MoveRepresentation<G::B>>::convert_from(a)?;
                let ta = <TAction<G::B> as ActionTensor>::vbm_one_hot_encode(&vbm);
                Ok((ta.inner(), 0.0f32))
            } else {
                let ta = <TAction::<G::B> as ActionTensor>::vec_vbm_one_hot_encode(&vec![]);
                Ok((ta.inner(), 0.0f32))
            }
        }
    }
}
#[test]
fn mcts_tree_draw() {

    let mcts_config = MCTSConfig::default();
    let mut mcts_tree = MCTSTree::generate(Game::init_std(), mcts_config);
    let actor = RandomActor{
        rng: RefCell::new(thread_rng())
    };

    for _ in 0..100 {
        mcts_tree.search_expand_backup(&actor).unwrap()
    }
    let tree = Tree::from_mcts_tree(&mcts_tree);
    let draw_tree = tree.pre_draw();
    
    // fidelity check
    let mut queue = VecDeque::new();
    queue.push_back(tree.root.clone());
    while !queue.is_empty() {
        let popped = queue.pop_front().unwrap();
        if !popped.children.is_empty() {
            assert_eq!(popped.visit_count as i64 - 1, popped.children.iter().fold(0.0, |acc, x| acc + x.visit_count) as i64);
        }
        for child in popped.children.iter(){
            queue.push_back(child.clone());
        }
    }

    draw_tree.draw("graphs/tmp/turn0.svg", XY::new(1000.0, 1000.0));

    let (_posterior, action_id) = mcts_tree.get_posterior_w_sampled_action_index().unwrap();
    let (trimmed_game, _action) = mcts_tree.trim_root(action_id).unwrap();
    mcts_tree.turn_count += 1;
    assert_eq!(trimmed_game, Game::default());
    mcts_tree.draw("graphs/tmp/turn1.svg", XY::new(200.0, 200.0));
}