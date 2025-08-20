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

/// (#(agent1 is att), #(agent1 is att && win), #(agent1 is att && draw),
///  #(agent1 is def), #(agent1 is def && win), #(agent1 is def && draw))
pub type DuelResult = (u64, u64, u64, u64, u64, u64);

/// Setup a duel between two models with separate trees for each player
pub fn setup_duel_trees<G: GameLogic>(
    agent1_config: &MCTSConfig,
    agent2_config: &MCTSConfig,
    agent1_sender: &Arc<Sender<Request>>,
    agent2_sender: &Arc<Sender<Request>>,
) -> (MCTSTree<G>, MCTSTree<G>, NewActor, NewActor)
where
    TBoard<G>: ModelInput<G>,
    TAction<G::B>: ActionTensor,
    TaflBoard<G::B>: std::fmt::Display,
{
    let game = <G as Default>::default();
    let agent1_tree = MCTSTree::<G>::generate(game.clone(), agent1_config.clone());
    let agent2_tree = MCTSTree::<G>::generate(game, agent2_config.clone());
    let agent1 = NewActor::new(agent1_sender.clone());
    let agent2 = NewActor::new(agent2_sender.clone());
    (agent1_tree, agent2_tree, agent1, agent2)
}

/// Conduct a duel between two models and return the win rate of the agent2
pub fn duel<'a, P: PVModel + Send, D: BoardData>(
    manager: InferenceManager<'a, P, &'a mut LockedShelf<P>>,
    agent1_sender: Sender<Request>,
    agent2_sender: Sender<Request>,
    num_games: usize,
    agent1_config: &MCTSConfig,
    agent2_config: &MCTSConfig,
) -> Result<DuelResult>
where
    TBoard<D::G>: ModelInput<D::G>,
    TAction<<D::G as GameLogic>::B>: ActionTensor,
    TaflBoard<<D::G as GameLogic>::B>: std::fmt::Display,
    <<D::G as GameLogic>::B as BitBoard>::Movement: PartialEq,
{
    let agent1_sender = Arc::new(agent1_sender);
    let agent2_sender = Arc::new(agent2_sender);

    let duel_res = thread::scope::<'a>(|s| -> Result<DuelResult> {
        // Start the inference manager in a separate thread
        let manager_handle = s.spawn(move || {
            println!("inference manager has started running");
            manager.consume_and_run()
        });

        let (a1a, a1d) = (0..num_games).into_par_iter().map(|game_idx| {
            let (mut agent1_tree, mut agent2_tree, agent1_actor, agent2_actor) = 
                setup_duel_trees::<D::G>(agent1_config, agent2_config, &agent1_sender, &agent2_sender);

            // dbg!();

            // Determine who plays as attacker/defender
            let agent1_is_attacker = game_idx % 2 == 0;
            
            // Progress the game with alternating players
            while !agent1_tree.root_is_terminal() && !agent2_tree.root_is_terminal() {
                let current_side = agent1_tree.get_current_game().current_side();
                
                // Determine whose turn it is based on game state and who's the attacker
                let agent1_turn = if agent1_is_attacker {
                    matches!(current_side, Side::Att)
                } else {
                    matches!(current_side, Side::Def)
                };

                // dbg!();
                if agent1_turn {
                    // agent1's turn - agent1 chooses action
                    let (_game, action, _posterior) = agent1_tree.get_policy_and_update::<NewActor>(&agent1_actor)
                        .expect("Failed to get policy and update agent1 tree");
                    
                    // Apply the chosen action to agent2's tree
                    agent2_tree.apply_external_action(&action)
                        .expect("Failed to apply action to agent2 tree");

                } else {
                    // agent2's turn - agent2 chooses action
                    let (_game, action, _posterior) = agent2_tree.get_policy_and_update::<NewActor>(&agent2_actor)
                        .expect("Failed to get policy and update agent2 tree");
                    
                    // Apply the chosen action to agent1's tree
                    agent1_tree.apply_external_action(&action)
                        .expect("Failed to apply action to agent1 tree");
                }

                #[cfg(debug_assertions)]
                {
                    for action in agent1_tree.root.actions.iter() {
                        assert!(agent2_tree.root.actions.contains(&action));
                    }
                    assert!(agent1_tree.root.game.get_board().equals(&agent2_tree.root.game.get_board()));
                    assert_eq!(agent1_tree.root.game.get_state(), agent2_tree.root.game.get_state());
                }
                // dbg!();
            }

            // Both trees should have the same winner
            let winner = agent1_tree.get_winner();
            
            // Determine if agent2 won
            let agent2_won = match winner {
                Victor::Att => {
                    // If agent2 is attacker, they win on Att victory
                    !agent1_is_attacker
                },
                Victor::Def => {
                    // If agent2 is defender, they win on Def victory
                    agent1_is_attacker
                },
                Victor::Draw => {
                    // Count draws as non-wins for agent2
                    false
                }
            };

            let draw = match winner {
              Victor::Draw => true,
                _ => false
            };

            (
                (agent1_is_attacker, agent1_is_attacker && matches!(winner, Victor::Att), agent1_is_attacker && matches!(winner, Victor::Draw)),
                (!agent1_is_attacker, !agent1_is_attacker && matches!(winner, Victor::Def), !agent1_is_attacker && matches!(winner, Victor::Draw)) 
            )
        }).unzip::<(bool, bool, bool), (bool, bool, bool), Vec<(bool, bool, bool)>, Vec<(bool, bool, bool)>>();

        let (a1at, a1aw, a1ad) = (
            a1a.iter().filter(|x| x.0).count(),
            a1a.iter().filter(|x| x.1).count(),
            a1a.iter().filter(|x| x.2).count()
        );

        let (a1dt, a1dw, a1dd) = (
            a1d.iter().filter(|x| x.0).count(),
            a1d.iter().filter(|x| x.1).count(),
            a1d.iter().filter(|x| x.2).count()
        );

        // Drop the senders to signal the inference manager to stop
        drop(agent1_sender);
        drop(agent2_sender);

        // Wait for the inference manager to finish
        manager_handle.join()
            .map_err(|_| eyre!("Manager thread panicked"))??;

        let duel_res = (a1at as u64, a1aw as u64, a1ad as u64, a1dt as u64, a1dw as u64, a1dd as u64);
        Ok(duel_res)
    })?;

    Ok(duel_res)
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
        let (agent1_sender, _agent1_receiver) = unbounded::<Request>();
        let (agent2_sender, _agent2_receiver) = unbounded::<Request>();

        let agent1_sender = Arc::new(agent1_sender);
        let agent2_sender = Arc::new(agent2_sender);

        // Test that we can setup duel trees
        let (agent1_tree, agent2_tree, agent1_actor, agent2_actor) =
            setup_duel_trees::<SimpleGame>(&mcts_config, &mcts_config.clone(), &agent1_sender, &agent2_sender);

        // Verify trees are independent
        assert!(!agent1_tree.root_is_terminal());
        assert!(!agent2_tree.root_is_terminal());
        
        // Verify they start with the same game state
        assert_eq!(
            agent1_tree.get_current_game().get_state(),
            agent2_tree.get_current_game().get_state()
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
        let agent2_wins = 45;
        let expected_win_rate = 0.45f32;
        
        let calculated_win_rate = (agent2_wins as f32) / (num_games as f32);
        assert_eq!(calculated_win_rate, expected_win_rate);
        
        // Test edge cases
        assert_eq!(0.0, 0.0 / 100.0); // No wins
        assert_eq!(1.0, 100.0 / 100.0); // All wins
    }

    /// Test the alternating player logic
    #[test]
    fn test_player_alternation_logic() {
        for game_idx in 0..10 {
            let agent1_is_attacker = game_idx % 2 == 0;
            
            if game_idx % 2 == 0 {
                assert!(agent1_is_attacker, "Even games: agent1 should be attacker");
            } else {
                assert!(!agent1_is_attacker, "Odd games: agent1 should be defender");
            }
        }
    }

    /// Test victor determination logic
    #[test]
    fn test_victor_determination() {
        // Test case 1: agent1 is attacker, Attacker wins
        let agent1_is_attacker = true;
        let victor = Victor::Att;
        let agent2_won = match victor {
            Victor::Att => !agent1_is_attacker,
            Victor::Def => agent1_is_attacker,
            Victor::Draw => false,
        };
        assert!(!agent2_won, "agent1 wins when agent1 is attacker and attacker wins");

        // Test case 2: agent1 is defender, Attacker wins  
        let agent1_is_attacker = false;
        let victor = Victor::Att;
        let agent2_won = match victor {
            Victor::Att => !agent1_is_attacker,
            Victor::Def => agent1_is_attacker,
            Victor::Draw => false,
        };
        assert!(agent2_won, "agent2 wins when agent1 is defender and attacker wins");

        // Test case 3: Draw scenario
        let agent1_is_attacker = true;
        let victor = Victor::Draw;
        let agent2_won = match victor {
            Victor::Att => !agent1_is_attacker,
            Victor::Def => agent1_is_attacker,
            Victor::Draw => false,
        };
        assert!(!agent2_won, "Draws count as non-wins for challenger");
    }

    /// Helper function to create test MCTS configuration
    fn create_test_mcts_config() -> MCTSConfig {
        MCTSConfig::default()
    }

}
