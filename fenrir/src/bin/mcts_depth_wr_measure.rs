use std::sync::Arc;

#[allow(unused_imports)]
use fenrir::model::{GeneralPVDualModel, GeneralPVSepModel};
#[allow(unused_imports)]
use fenrir::replay_buffer::{SimpleGameSPR, GameSPR, BoardData};
use fenrir::setup::{ModelSetupConfig, setup_wo_mpi};
use fenrir::agent::{MCTSConfig, NewActor};
use fenrir::self_play::{InferenceManager, LockedShelf, ModuleShelf};
use fenrir::duel::{setup_duel_trees, DuelResult};
use game::game::{Side, Victor, GameLogic};

use itertools::{self, Itertools};
use color_eyre::eyre::{Result, eyre};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

const BS: usize = 8;
const NUM_GAMES: usize = 100;

fn main() -> Result<()>{

    type P = GeneralPVSepModel;
    type D = SimpleGameSPR;

    // might be needed to correctly link cuda-related shared objects.
    let _ = tch::Cuda::cudnn_is_available();

    let argv = std::env::args().collect::<Vec<_>>(); // [_, comp_config]
    if argv.len() != 2 {
        eprintln!("Usage: {} <setup config file>", argv[0]);
    }

    let setup_config: ModelSetupConfig = ModelSetupConfig::parse_toml(&argv[1]).unwrap();
    let loaded_modules = setup_wo_mpi::<P>(&setup_config).unwrap();

    println!("setup config: \n{:?}", setup_config);
    println!("Player1 Player2 Draw");

    let mut mcts_config_1 = MCTSConfig::default();
    let mut mcts_config_2 = MCTSConfig::default();
    let num_sims: [usize; 5] = [100, 200, 400, 800, 1600];

    let (table, name_lookup) = ModelSetupConfig::create_lookup_table(&setup_config, None, loaded_modules).unwrap();

    assert_eq!(table.len(), 1);

    let mut shelf = ModuleShelf::module_shelf(table, name_lookup);

    // total of 10 combinations
    for (n_sim1, n_sim2) in num_sims.into_iter().tuple_combinations() {

        println!("n_sim1:{} n_sim2:{}", n_sim1, n_sim2);
        mcts_config_1.n_sim = n_sim1;
        mcts_config_2.n_sim = n_sim2;
        let mut locked_shelf = LockedShelf::convert_from_shelf(shelf);

        let (manager, mut request_senders)
        = InferenceManager::<GeneralPVSepModel, _>::new(&mut locked_shelf, BS);
        assert_eq!(request_senders.len(), 1);

        let request_sender = request_senders.remove(0);
        let rs = Arc::new(request_sender);

        let duel_res = std::thread::scope::<'_>(|s| -> Result<DuelResult> {
            // Start the inference manager in a separate thread
            let manager_handle = s.spawn(move || {
                manager.consume_and_run()
            });

            let (a1a, a1d) = (0..NUM_GAMES).into_par_iter().map(|game_idx| {

                let rs_1 = Arc::clone(&rs);
                let rs_2 = Arc::clone(&rs);

                let (mut agent1_tree, mut agent2_tree, agent1, agent2)
                = setup_duel_trees::<<D as BoardData>::G>(&mcts_config_1, &mcts_config_2, &rs_1, &rs_2);

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
                        let (_game, action, _posterior) = agent1_tree.get_policy_and_update::<NewActor>(&agent1)
                            .expect("Failed to get policy and update agent1 tree");
                        
                        // Apply the chosen action to agent2's tree
                        agent2_tree.apply_external_action(&action)
                            .expect("Failed to apply action to agent2 tree");

                    } else {
                        // agent2's turn - agent2 chooses action
                        let (_game, action, _posterior) = agent2_tree.get_policy_and_update::<NewActor>(&agent2)
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
            drop(rs);

            // Wait for the inference manager to finish
            manager_handle.join()
                .map_err(|_| eyre!("Manager thread panicked"))??;

            let duel_res = (a1at as u64, a1aw as u64, a1ad as u64, a1dt as u64, a1dw as u64, a1dd as u64);
            Ok(duel_res)
        })?;

        
        println!("{} {} {}",
            duel_res.1 + duel_res.4, // player 1 win
            duel_res.0 - duel_res.1 - duel_res.2 + duel_res.3 - duel_res.4 - duel_res.5, // player 2 win
            duel_res.2 + duel_res.5 // draw
        );

        shelf = LockedShelf::convert_into_shelf(locked_shelf);
    }
    Ok(())
}