#![allow(unused_imports)]
use std::path::{PathBuf, Path};

use fenrir::agent::MCTSConfig;
use fenrir::{CompConfig, MpiConfig};
use fenrir::run;
use fenrir::model::{GeneralPVDualModel, GeneralPVSepModel};
use fenrir::replay_buffer::{GameSPR, SimpleGameSPR};
use game::game::{SimpleGame, Game};

fn main() {
    let comp_config_dir: PathBuf = Path::new("/work/kurokawa/project_fenrir/config").join("mk1-1node-comp.toml");
    let mpi_config_dir: PathBuf = Path::new("/work/kurokawa/project_fenrir/config").join("mk1-1node-mpi.toml");
    let evaluation_mcts_config_dir: PathBuf = Path::new("/work/kurokawa/project_fenrir/config").join("mk1-1node-eval_mcts.toml");
    let data_store_dir = "/fast/kurokawa/fenrir";
    let model_store_dir = "/work/kurokawa/project_fenrir/models";
    let comp_config: CompConfig = fenrir::load_comp_config(&comp_config_dir);
    let mpi_config: MpiConfig = fenrir::load_mpi_config(&mpi_config_dir);
    let evaluation_mcts_config: MCTSConfig = fenrir::agent::load_mcts_config(&evaluation_mcts_config_dir);
    run::run_cnt::<GeneralPVSepModel, SimpleGameSPR>(
        comp_config,
        mpi_config,
        evaluation_mcts_config,
        model_store_dir,
        data_store_dir,
    );
}