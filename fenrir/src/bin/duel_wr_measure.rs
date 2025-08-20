#![cfg(not(feature = "bench"))]
#[allow(unused_imports)]
use fenrir::model::{GeneralPVSepModel, GeneralPVDualModel};
use fenrir::agent::{load_mcts_config};
#[allow(unused_imports)]
use fenrir::replay_buffer::{GameSPR, SimpleGameSPR};
use fenrir::self_play::{ModuleShelf, LockedShelf, InferenceManager};
use fenrir::setup::{setup_wo_mpi, ModelSetupConfig};
use fenrir::duel::{duel, DuelResult};

use color_eyre::eyre::{Result};
use tch;

use std::path::{Path};

const BS: usize = 8;
const NUM_GAMES: usize = 1000;

fn main() -> Result<()>{

    type P = GeneralPVSepModel;
    type D = SimpleGameSPR;

    // might be needed to correctly link cuda-related shared objects.
    let _ = tch::Cuda::cudnn_is_available();

    let argv = std::env::args().collect::<Vec<_>>();

    if argv.len() < 4 {
        eprintln!(
            "Usage: {} <setup_config_file> <mcts_config_file> <mcts_config_file2> ",
        argv[0]);
        std::process::exit(1);
    }

    if !Path::new(&argv[1]).exists() {
    eprintln!("Setup config file not found: {}", argv[1]);
    std::process::exit(1);
    }

    if !Path::new(&argv[2]).exists() {
    eprintln!("mcts config file 1 not found: {}", argv[2]);
    std::process::exit(1);
    }

    if !Path::new(&argv[3]).exists() {
    eprintln!("mcts config file 2 not found: {}", argv[2]);
    std::process::exit(1);
    }

    let config: ModelSetupConfig = ModelSetupConfig::parse_toml(&argv[1])?;
    let mcts_config_1 = load_mcts_config(&argv[2]);
    let mcts_config_2 = load_mcts_config(&argv[3]);

    let loaded_modules = setup_wo_mpi::<P>(&config)?;
    let (table,name_lookup) = config.create_lookup_table(None, loaded_modules)?;

    assert_eq!(table.len(), 2);

    let shelf = ModuleShelf::module_shelf(table, name_lookup);
    let mut shelf = LockedShelf::convert_from_shelf(shelf);

    let (manager, mut rs) = InferenceManager::<P, _>::new(&mut shelf, BS);
    assert_eq!(rs.len(), 2);
    let player_1_sender = rs.remove(0);
    let player_2_sender = rs.remove(0);
    let duel_result: DuelResult = duel::<P, D>(manager, player_1_sender, player_2_sender, NUM_GAMES, &mcts_config_1, &mcts_config_2)?;

    println!("(1 attack total), (1 attack win), (1 attack draw), (1 defend total), (1 defend win), (1 defend draw)");
    println!("{}, {}, {}, {}, {}, {}", duel_result.0, duel_result.1, duel_result.2, duel_result.3, duel_result.4, duel_result.5);

    Ok(())

}


