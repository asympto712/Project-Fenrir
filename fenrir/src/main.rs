#![allow(unused_imports)]
use std::path::{PathBuf, Path};

use fenrir::{CompConfig, MpiConfig};
use fenrir::run;
use fenrir::model::{GeneralPVDualModel, GeneralPVSepModel};
use fenrir::replay_buffer::{GameSPR, SimpleGameSPR};
use game::game::{SimpleGame, Game};

fn main() {
    let comp_config_dir: PathBuf = Path::new("./config").join("comp_cnt_test.toml");
    let mpi_config_dir: PathBuf = Path::new("./config").join("mpi_test.toml");
    let comp_config: CompConfig = fenrir::load_comp_config(&comp_config_dir);
    let mpi_config: MpiConfig = fenrir::load_mpi_config(&mpi_config_dir);
    run::run_cnt::<GeneralPVDualModel, SimpleGameSPR>(comp_config, mpi_config);
}