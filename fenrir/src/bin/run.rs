#![cfg(not(feature = "bench"))]
#[cfg(feature = "mpi")]
use fenrir::agent::{MCTSConfig, load_mcts_config};
use fenrir::model::{GeneralPVSepModel, PVModel};
use fenrir::node::{Node, SelfPlayNode, TrainNode, TestNode};
use fenrir::replay_buffer::{BoardData, ReplayBuffer, Sampler, SimpleGameSPR};
use fenrir::utils::{ActionTensor, ModelInput, TAction, TBoard};
use fenrir::{load_mpi_config, load_comp_config, CompConfig, MpiConfig};

use bincode::Encode;
use game::board::TaflBoard;
use game::game::GameLogic;
use bitboard::BitBoard;
use mpi::traits::Communicator;
use color_eyre::eyre::Result;

use std::path::Path;

fn main() -> Result<()> {

    // might be needed to correctly link cuda-related shared objects.
    let _ = tch::Cuda::cudnn_is_available();

    let argv = std::env::args().collect::<Vec<_>>();

    if argv.len() < 6 {
        eprintln!(
            "Usage: {} <config_file> <mpi_config_file> <mcts_config_file> <model_dir> <data_dir> (<initial self play game count>, <initial training step count>)",
        argv[0]
        );
        std::process::exit(1);
    }

    if !Path::new(&argv[1]).exists() {
    eprintln!("Config file not found: {}", argv[1]);
    std::process::exit(1);
    }

    if !Path::new(&argv[2]).exists() {
    eprintln!("Config file not found: {}", argv[2]);
    std::process::exit(1);
    }

    if !Path::new(&argv[3]).exists() {
    eprintln!("Config file not found: {}", argv[3]);
    std::process::exit(1);
    }

    let config: CompConfig = load_comp_config(&argv[1]);
    let mpi_config = load_mpi_config(&argv[2]);
    let duel_mcts_config = load_mcts_config(&argv[3]);
    let model_dir = &argv[4];
    let data_dir = &argv[5];
    let _init_self_play_count: usize = if let Some(i) = argv.get(6) {
        i.parse().unwrap()
    } else {
        0
    };
    let init_training_step_count: usize = if let Some(i) = argv.get(7) {
        i.parse().unwrap()
    } else {
        0
    };

    run_cnt::<GeneralPVSepModel, SimpleGameSPR>(
        config,
        mpi_config,
        duel_mcts_config,
        &model_dir,
        &data_dir,
        init_training_step_count,
    );

    Ok(())

}

pub fn run_cnt<P: PVModel + Send, D: BoardData + Encode>(
    config: CompConfig,
    mpi_config: MpiConfig,
    evaluation_mcts_config: MCTSConfig,
    model_store_dir: &str,
    data_store_dir: &str,
    init_training_step_count: usize,
)
where 
TBoard<D::G>: ModelInput<D::G>,
TAction<<D::G as GameLogic>::B>: ActionTensor,
TaflBoard<<D::G as GameLogic>::B>: std::fmt::Display,
D: Send + Sync,
<<D::G as GameLogic>::B as BitBoard>::Movement: PartialEq,
ReplayBuffer<D>: Sampler
{
    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Multiple).unwrap();
    let world = universe.world();
    let rank = world.rank();
    
    if rank == mpi_config.test {
        println!("Hello from rank {}", rank);
        let mut node = TestNode::<P, D>::init(config);
        node.run(&mpi_config, &evaluation_mcts_config, model_store_dir);
    } else if rank == mpi_config.train {
        println!("Hello from rank {}", rank);
        let mut node = TrainNode::<P, D>::init(config);
        node.run(&mpi_config, data_store_dir, init_training_step_count);
    } else if mpi_config.self_play.contains(&rank) {
        println!("Hello from rank {}", rank);
        let mut node = SelfPlayNode::<P, D>::init(config);
        node.run(&mpi_config, data_store_dir);
    }

}
