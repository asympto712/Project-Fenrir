#![allow(unused)]
#![allow(dead_code)]

use std::{fs::File, io::Read};

use crate::{
    agent::MCTSConfig,
    model::ModelConfig,
    setup::{FenrirConfig, FenrirConfigWrapper, ModelSetupConfig, ModelSetupConfigWrapper, ModuleLoadInfo, ModuleLoadInfoWrapper},
    train::ModelSyncConfig
};
use serde::Deserialize;

#[cfg(feature = "torch")]
pub mod model;
#[cfg(feature = "torch")]
pub mod replay_buffer;
#[cfg(feature = "torch")]
pub mod agent;
#[cfg(feature = "torch")]
pub mod self_play;
#[cfg(feature = "torch")]
pub mod train;

pub mod setup;

pub mod run;

pub mod visualization;

pub mod statistics;

pub mod utils;
pub mod schedule;

use std::fs;
use toml;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct CompConfigWrapper {
    name: String,
    fenrir_config: FenrirConfigWrapper,
    mcts_config: MCTSConfig,
    model_sync_config: ModelSyncConfig,
    setup_config: ModelSetupConfigWrapper,
}

pub struct CompConfig {
    pub name: String,
    pub fenrir_config: FenrirConfig,
    pub mcts_config: MCTSConfig,
    pub model_sync_config: ModelSyncConfig,
    pub setup_config: ModelSetupConfig,
}

impl From<CompConfigWrapper> for CompConfig {
    fn from(value: CompConfigWrapper) -> Self {
        Self {
            name: value.name,
            fenrir_config: value.fenrir_config.into(),
            mcts_config: value.mcts_config,
            model_sync_config: value.model_sync_config,
            setup_config: value.setup_config.into()
        }
    }
}

pub fn load_comp_config<P: AsRef<Path>>(filename: P) -> CompConfig {
    let data = fs::read_to_string(&filename).unwrap();
    let wrapper = toml::from_str::<CompConfigWrapper>(&data).unwrap();
    Into::<CompConfig>::into(wrapper)
}

#[test]
fn toml_parse_works() {
    use std::fs::File;
    use toml;
    let path = "./config/comp_config_test.toml";
    let file = File::open(path).unwrap();
    let data = std::fs::read_to_string(path).unwrap();
    let result = toml::from_str::<CompConfigWrapper>(&data);
    assert!(result.is_ok());
    let wrapper = result.unwrap();
    let comp_config: CompConfig = wrapper.into();
}

