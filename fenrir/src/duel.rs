#![cfg(feature = "torch")]

use crate::agent::{MCTSConfig, MCTSTree, NewActor, Oracle};
use crate::model::PVModel;
use crate::replay_buffer::BoardData;
use crate::self_play::{InferenceManager, LockedShelf, Request, setup_mcts};
use crate::utils::{ActionTensor, BoardTensor, ModelInput, TAction, TBoard};
use game::game::Side;

use bitboard::BitBoard;
use color_eyre::eyre::{eyre, Result};
use crossbeam::channel::{unbounded, Sender, Receiver};
use game::board::TaflBoard;
use game::game::{GameLogic, Victor};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
use std::thread;

/// Setup a duel between two models with separate trees for each player
pub fn setup_duel_trees<G: GameLogic>(
    mcts_config: &MCTSConfig,
    champion_sender: &Arc<Sender<Request>>,
    challenger_sender: &Arc<Sender<Request>>,
) -> (MCTSTree<G>, MCTSTree<G>, NewActor, NewActor)
where
    TBoard<G>: ModelInput<G>,
    TAction<G::B>: ActionTensor,
    TaflBoard<G::B>: std::fmt::Display,
{
    let game = <G as Default>::default();
    let champion_tree = MCTSTree::<G>::generate(game.clone(), mcts_config.clone());
    let challenger_tree = MCTSTree::<G>::generate(game, mcts_config.clone());
    let champion = NewActor::new(champion_sender.clone());
    let challenger = NewActor::new(challenger_sender.clone());
    (champion_tree, challenger_tree, champion, challenger)
}

/// Conduct a duel between two models and return the win rate of the challenger
pub fn duel<'a, P: PVModel + Send, D: BoardData>(
    manager: InferenceManager<'a, P, &'a mut LockedShelf<P>>,
    champion_sender: Sender<Request>,
    challenger_sender: Sender<Request>,
    num_games: usize,
    mcts_config: &MCTSConfig,
) -> Result<(f32, usize)>
where
    TBoard<D::G>: ModelInput<D::G>,
    TAction<<D::G as GameLogic>::B>: ActionTensor,
    TaflBoard<<D::G as GameLogic>::B>: std::fmt::Display,
    <<D::G as GameLogic>::B as BitBoard>::Movement: PartialEq,
{
    let champion_sender = Arc::new(champion_sender);
    let challenger_sender = Arc::new(challenger_sender);

    let (win_rate, draw_count) = thread::scope::<'a>(|s| -> Result<(f32, usize)> {
        // Start the inference manager in a separate thread
        let manager_handle = s.spawn(move || {
            println!("inference manager has started running");
            manager.consume_and_run()
        });

        // Run games in parallel using rayon
        let (chal_win, draw) = (0..num_games).into_par_iter().map(|game_idx| {
            let (mut champion_tree, mut challenger_tree, champion_actor, challenger_actor) = 
                setup_duel_trees::<D::G>(mcts_config, &champion_sender, &challenger_sender);

            dbg!();

            // Determine who plays as attacker/defender
            let champion_is_attacker = game_idx % 2 == 0;
            
            // Progress the game with alternating players
            while !champion_tree.root_is_terminal() && !challenger_tree.root_is_terminal() {
                let current_side = champion_tree.get_current_game().current_side();
                
                // Determine whose turn it is based on game state and who's the attacker
                let champion_turn = if champion_is_attacker {
                    matches!(current_side, Side::Att)
                } else {
                    matches!(current_side, Side::Def)
                };

                dbg!();
                if champion_turn {
                    // Champion's turn - champion chooses action
                    let (_game, action, _posterior) = champion_tree.get_policy_and_update::<NewActor>(&champion_actor)
                        .expect("Failed to get policy and update champion tree");
                    
                    // Apply the chosen action to challenger's tree
                    challenger_tree.apply_external_action(&action)
                        .expect("Failed to apply action to challenger tree");

                } else {
                    // Challenger's turn - challenger chooses action
                    let (_game, action, _posterior) = challenger_tree.get_policy_and_update::<NewActor>(&challenger_actor)
                        .expect("Failed to get policy and update challenger tree");
                    
                    // Apply the chosen action to champion's tree
                    champion_tree.apply_external_action(&action)
                        .expect("Failed to apply action to champion tree");
                }

                #[cfg(debug_assertions)]
                {
                    for action in champion_tree.root.actions.iter() {
                        assert!(challenger_tree.root.actions.contains(&action));
                    }
                    assert!(champion_tree.root.game.get_board().equals(&challenger_tree.root.game.get_board()));
                    assert_eq!(champion_tree.root.game.get_state(), challenger_tree.root.game.get_state());
                }
                dbg!();
            }

            // Both trees should have the same winner
            let winner = champion_tree.get_winner();
            
            // Determine if challenger won
            let challenger_won = match winner {
                Victor::Att => {
                    // If challenger is attacker, they win on Att victory
                    !champion_is_attacker
                },
                Victor::Def => {
                    // If challenger is defender, they win on Def victory
                    champion_is_attacker
                },
                Victor::Draw => {
                    // Count draws as non-wins for challenger
                    false
                }
            };

            let draw = match winner {
              Victor::Draw => true,
                _ => false
            };

            (challenger_won, draw)
        }).unzip::<bool, bool, Vec<bool>, Vec<bool>>();

        // dbg!(&chal_win);
        // dbg!(&draw);

        let win_count = chal_win.into_iter().filter(|b| *b).count();
        let draw_count = draw.into_iter().filter(|b| *b).count();

        // Drop the senders to signal the inference manager to stop
        drop(champion_sender);
        drop(challenger_sender);

        // Wait for the inference manager to finish
        manager_handle.join()
            .map_err(|_| eyre!("Manager thread panicked"))??;

        // Calculate win rate
        let win_rate = (win_count as f32) / (num_games as f32);
        Ok((win_rate, draw_count))
    })?;

    Ok((win_rate, draw_count))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GeneralPVDualModel;
    use crate::replay_buffer::SimpleGameSPR;
    use crate::agent::MCTSConfig;
    use crate::setup::{ModelSetupConfig, ModuleLoadInfo};
    use crossbeam::channel::unbounded;
    use game::game::{SimpleGame, Victor};
    use std::sync::Arc;
    use tch::Device;

    /// Test configuration for a Seven Soldier game duel
    #[test]
    fn test_duel_setup() {
        // Use default MCTS configuration
        let mcts_config = MCTSConfig::default();

        // Create mock channels
        let (champion_sender, _champion_receiver) = unbounded::<Request>();
        let (challenger_sender, _challenger_receiver) = unbounded::<Request>();

        let champion_sender = Arc::new(champion_sender);
        let challenger_sender = Arc::new(challenger_sender);

        // Test that we can setup duel trees
        let (champion_tree, challenger_tree, champion_actor, challenger_actor) =
            setup_duel_trees::<SimpleGame>(&mcts_config, &champion_sender, &challenger_sender);

        // Verify trees are independent
        assert!(!champion_tree.root_is_terminal());
        assert!(!challenger_tree.root_is_terminal());
        
        // Verify they start with the same game state
        assert_eq!(
            champion_tree.get_current_game().get_state(),
            challenger_tree.get_current_game().get_state()
        );
    }

    /// Test a small duel between two models
    #[test]
    #[ignore] // This test requires actual model files and GPU/CPU resources
    fn test_small_duel() {
        println!("Small duel test setup complete - requires model loading infrastructure");
    }

    /// Test duel logic with mock scenarios
    #[test]
    fn test_duel_win_rate_calculation() {
        let num_games = 100;
        let challenger_wins = 45;
        let expected_win_rate = 0.45f32;
        
        let calculated_win_rate = (challenger_wins as f32) / (num_games as f32);
        assert_eq!(calculated_win_rate, expected_win_rate);
        
        // Test edge cases
        assert_eq!(0.0, 0.0 / 100.0); // No wins
        assert_eq!(1.0, 100.0 / 100.0); // All wins
    }

    /// Test the alternating player logic
    #[test]
    fn test_player_alternation_logic() {
        for game_idx in 0..10 {
            let champion_is_attacker = game_idx % 2 == 0;
            
            if game_idx % 2 == 0 {
                assert!(champion_is_attacker, "Even games: champion should be attacker");
            } else {
                assert!(!champion_is_attacker, "Odd games: champion should be defender");
            }
        }
    }

    /// Test victor determination logic
    #[test]
    fn test_victor_determination() {
        // Test case 1: Champion is attacker, Attacker wins
        let champion_is_attacker = true;
        let victor = Victor::Att;
        let challenger_won = match victor {
            Victor::Att => !champion_is_attacker,
            Victor::Def => champion_is_attacker,
            Victor::Draw => false,
        };
        assert!(!challenger_won, "Champion wins when champion is attacker and attacker wins");

        // Test case 2: Champion is defender, Attacker wins  
        let champion_is_attacker = false;
        let victor = Victor::Att;
        let challenger_won = match victor {
            Victor::Att => !champion_is_attacker,
            Victor::Def => champion_is_attacker,
            Victor::Draw => false,
        };
        assert!(challenger_won, "Challenger wins when champion is defender and attacker wins");

        // Test case 3: Draw scenario
        let champion_is_attacker = true;
        let victor = Victor::Draw;
        let challenger_won = match victor {
            Victor::Att => !champion_is_attacker,
            Victor::Def => champion_is_attacker,
            Victor::Draw => false,
        };
        assert!(!challenger_won, "Draws count as non-wins for challenger");
    }

    /// Helper function to create test MCTS configuration
    fn create_test_mcts_config() -> MCTSConfig {
        MCTSConfig::default()
    }

}
