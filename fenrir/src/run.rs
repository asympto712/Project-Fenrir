#![cfg(feature = "mpi")]

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::sync::{Mutex, RwLock};
use std::time::Duration;
use std::path::Path;
use std::io::{Read, Seek, Cursor};
use std::convert::AsRef;

use game::game::GameLogic;
use game::board::TaflBoard;
use serde::Deserialize;
use crate::agent::MCTSConfig;
use crate::model::{self, ModelConfig};
use crate::model::PVModel;
use crate::replay_buffer::{BoardData, GameSPR, ReplayBuffer, SimpleGameSPR, Sampler};
use crate::run;
use crate::schedule::lr_sch_initialize;
use crate::self_play::{self, self_play_new, InferenceManager, LockedShelf, ModuleShelf};
use crate::self_play::self_play;
use crate::train::{self, ModelSyncConfig, Trainer};
use crate::self_play::Shelf;
use crate::utils::{ActionTensor, BoardTensor, ModelInput, TAction, TBoard};

use color_eyre::eyre::{eyre, OptionExt};
use mpi::point_to_point::*;
use mpi::topology::SimpleCommunicator;
use mpi::traits::Communicator;
use mpi::MpiError;
use mpi::datatype::Equivalence;

use color_eyre::eyre::Result;
use rand::prelude::*;
use rv::traits::Mode;
use tch::nn::ModuleT;
use tch::nn::VarStore;
use tch::vision::image::load;
use tch::Device;
use chrono::{Utc, Local};

const BUFFER_LEN: usize = 1024;

#[derive(Debug, Clone)]
pub struct FenrirConfig{
    pub n_self_play_nodes: usize,
    pub n_train_nodes: usize,
    pub n_model_replica_self_play: usize,
    pub n_model_replica_train: usize,
    pub use_mpi: bool,
    pub mini_batch_size: usize,
    pub n_training_step_per_cycle: usize,
    pub n_self_play_games: usize,
    pub n_games_per_tournament: usize,
    pub model_update_threshold: f32,
    pub concurrent_training: bool,
    pub replay_buffer_capacity: usize,
    pub run_time_hr: u64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub learning_rate_schedule: fn(usize) -> f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FenrirConfigWrapper {
    n_self_play_nodes: usize,
    n_train_nodes: usize,
    n_model_replica_self_play: usize,
    n_model_replica_train: usize,
    use_mpi: bool,
    mini_batch_size: usize,
    n_training_step_per_cycle: usize,
    n_self_play_games: usize,
    n_games_per_tournament: usize,
    model_update_threshold: f32,
    concurrent_training: bool,
    replay_buffer_capacity: usize,
    run_time_hr: u64,
    momentum: f64,
    weight_decay: f64,
    learning_rate_schedule: String,
}

impl From<FenrirConfigWrapper> for FenrirConfig {
    fn from(value: FenrirConfigWrapper) -> Self {
        let mut table = lr_sch_initialize();
        let learning_rate_schedule = table.remove(&value.learning_rate_schedule).unwrap();
        Self {
            n_self_play_nodes: value.n_self_play_nodes,
            n_train_nodes: value.n_train_nodes,
            n_model_replica_self_play: value.n_model_replica_self_play,
            n_model_replica_train: value.n_model_replica_train,
            use_mpi: value.use_mpi,
            mini_batch_size: value.mini_batch_size,
            n_training_step_per_cycle: value.n_training_step_per_cycle,
            n_self_play_games: value.n_self_play_games,
            n_games_per_tournament: value.n_games_per_tournament,
            model_update_threshold: value.model_update_threshold,
            concurrent_training: value.concurrent_training,
            replay_buffer_capacity: value.replay_buffer_capacity,
            run_time_hr: value.run_time_hr,
            momentum: value.momentum,
            weight_decay: value.weight_decay,
            learning_rate_schedule,
        }
    }
}

const fn agz_lr_schedule(n_steps: usize) -> f64 {
    match n_steps {
        0..400000 => 0.01f64,
        400000..600000 => 0.001f64,
        600000.. => 0.0001f64,
    }
}

// Set up related stuff
#[derive(Debug)]
pub struct ModelSetupConfig{
    use_mpi: bool,
    module_load_infos: Vec<ModuleLoadInfo>,
}

#[derive(Debug, Deserialize)]
pub struct ModelSetupConfigWrapper {
    use_mpi: bool,
    module_load_infos: Vec<ModuleLoadInfoWrapper>,
}

impl From<ModelSetupConfigWrapper> for ModelSetupConfig {
    fn from(value: ModelSetupConfigWrapper) -> Self {
        let use_mpi = value.use_mpi;
        let module_load_infos: Vec<ModuleLoadInfo> = value.module_load_infos.into_iter().map(|x| x.into()).collect::<Vec<_>>();
        Self {
            use_mpi,
            module_load_infos
        }
    }
}

impl ModelSetupConfig {
    pub fn model_setup_config(use_mpi: bool, module_load_infos: Vec<ModuleLoadInfo>) -> Self {
        Self {
            use_mpi,
            module_load_infos
        }
    }
    // This takes the output of the setup function (loaded models along with their var stores and their path names), and create correspondence between their names and their group id.
    // This function operates locally (inside each mpi node)
    pub fn create_lookup_table<P: PVModel>(&self, rank: Option<i32>, loaded_model: Vec<(OsString, P, VarStore)>) -> 
    Result<(Vec<Vec<(P, VarStore)>>, Vec<OsString>)> {
        
        let mut table: Vec<Vec<(P, VarStore)>> = vec![];
        let mut name_lookup: Vec<OsString> = vec![];

        // taking XOR
        if self.use_mpi ^ rank.is_some() {
            return Err(eyre!(
                "inconsistency in mpi usage-related input: rank should be None if and only if use_mpi is false"
            ));
        }
        
        for ((name, model, vs), info) in loaded_model.into_iter().zip(self.module_load_infos.iter()) {
            
            if let Some(i) = rank && info.rank != Some(i) {
                continue;
            }

            if let Some(id) = name_lookup.iter().position(|s| *s == name){
                table.get_mut(id).unwrap().push((model, vs));
            } else {
                // If this is the first time that the 'name' comes up, 
                table.push(vec![]);
                name_lookup.push(name.clone());
                table.last_mut().unwrap().push((model, vs));
            }
        }

        Ok((table, name_lookup))
    }
}

#[derive(Debug)]
pub struct ModuleLoadInfo {
    // path to the file that stores the model weight
    path: OsString,
    // only specify when using mpi
    rank: Option<i32>,
    // device to load the model onto.
    device: Device,
    // model config
    config: ModelConfig,
}

#[derive(Debug, Deserialize)]
pub struct ModuleLoadInfoWrapper {
    path: String,
    rank: Option<i32>,
    device: DeviceWrapper,
    config: ModelConfig,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", content = "details")]
enum DeviceWrapper {
    Cpu,
    Cuda(usize),
}

impl From<ModuleLoadInfoWrapper> for ModuleLoadInfo {
    fn from(value: ModuleLoadInfoWrapper) -> Self {
        let path: OsString = value.path.into();
        let rank = value.rank;
        let device: Device = match value.device {
            DeviceWrapper::Cpu => Device::Cpu,
            DeviceWrapper::Cuda(i) => Device::Cuda(i)
        };
        let config = value.config;
        Self {
            path,
            rank,
            device,
            config
        }
    }
}

impl ModuleLoadInfo {
    pub fn module_load_info(path: OsString, rank: Option<i32>, device: Device, config: ModelConfig) -> Self {
        Self{
            path,
            rank,
            device,
            config
        }
    }
}

pub fn load_module<P: PVModel>(info: &ModuleLoadInfo) -> Result<(OsString, P, VarStore)> {

    let path = Path::new(&info.path);
    let mut vs = VarStore::new(info.device);
    let module: P = P::new(&vs.root(), &info.config);
    vs.load(path)?;
    Ok((info.path.to_owned(), module, vs))

}

pub fn load_module_from_stream<P: PVModel, S: Read + Seek>(info: &ModuleLoadInfo, stream: S) -> Result<(OsString, P, VarStore)> {

    let mut vs = VarStore::new(info.device);
    let module: P = P::new(&vs.root(), &info.config);
    vs.load_from_stream(stream)?;
    Ok((info.path.to_owned(), module, vs))

}

pub fn setup<P: PVModel>(config: &ModelSetupConfig) -> Result<Vec<(OsString, P, VarStore)>>{

    #[cfg(not(feature = "mpi"))]
    {
        if config.use_mpi {
            use color_eyre::eyre::eyre;

            return Err(eyre!("MPI functionality requested but 'mpi' feature is not enabled. 
            recompile with --features mpi"));
        }
    }

    #[cfg(feature = "mpi")]
    {
        if config.use_mpi {
            setup_w_mpi::<P>(config)
        } else {
            setup_wo_mpi::<P>(config)
        }
    }

    #[cfg(not(feature = "mpi"))]
    {
        setup_wo_mpi::<P>(config)
    }

}

fn add_module_to_list<P: PVModel>(info: &ModuleLoadInfo, list: &mut Vec<(OsString, P, VarStore)>, available_cuda: &mut Vec<Device>) -> Result<()> {

    match info.device {

        Device::Cpu => {
            let (name, module, var_store) = load_module::<P>(&info)?;
            list.push((name, module, var_store));
            Ok(())
        },
        Device::Cuda(i) => {

            if !available_cuda.contains(&info.device) {
                return Err(eyre!("cuda allocation failed: Check the ModelSetupConfig again"));
            }
            let device_index = available_cuda.iter().position(|&d| d == info.device).ok_or_eyre("Could not find the device")?;
            available_cuda.remove(device_index);
            let (name, module, var_store) = load_module::<P>(&info)?;
            list.push((name, module, var_store));
            Ok(())
            
        },
        _ => {
            Err(eyre!("unexpected device encountered"))
        }
    }
}

fn add_module_from_stream_to_list<P: PVModel, S: Seek + Read>(
    info: &ModuleLoadInfo,
    list: &mut Vec<(OsString, P, VarStore)>,
    available_cuda: &mut Vec<Device>,
    stream: S) -> Result<()> {

    match info.device {

        Device::Cpu => {
            let (name, module, var_store) = load_module_from_stream::<P, S>(&info, stream)?;
            list.push((name, module, var_store));
            Ok(())
        },
        Device::Cuda(i) => {

            if !available_cuda.contains(&info.device) {
                return Err(eyre!("cuda allocation failed: Check the ModelSetupConfig again"));
            }
            let device_index = available_cuda.iter().position(|&d| d == info.device).ok_or_eyre("Could not find the device")?;
            available_cuda.remove(device_index);
            let (name, module, var_store) = load_module_from_stream::<P, S>(&info, stream)?;
            list.push((name, module, var_store));
            Ok(())
            
        },
        _ => {
            Err(eyre!("unexpected device encountered"))
        }
    }
}

// This function loads the models, ignoring the mpi options. It is intended to be used inside the 'setup' function
fn setup_wo_mpi<P: PVModel>(config: &ModelSetupConfig) -> Result<Vec<(OsString, P, VarStore)>> {

    let mut loaded_models: Vec<(OsString, P, VarStore)> = vec![];

    let n_cuda = tch::Cuda::device_count();
    let mut available_cuda: Vec<Device>  = (0..n_cuda).map(|i| Device::Cuda(i as usize)).collect();

    for module_info in config.module_load_infos.iter() {
        add_module_to_list(module_info, &mut loaded_models, &mut available_cuda)?;
    }
    Ok(loaded_models)
}

#[cfg(feature = "mpi")]
fn setup_w_mpi<P: PVModel>(config: &ModelSetupConfig) -> Result<Vec<(OsString, P, VarStore)>> {

    use mpi::traits::Communicator;
    // mpi related setup
    let universe = mpi::initialize().ok_or_eyre("mpi initialization failed")?;
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let n_cuda = tch::Cuda::device_count();
    let mut available_cuda: Vec<Device> = (0..n_cuda).map(|i| Device::Cuda(i as usize)).collect();
    let mut loaded_models: Vec<(OsString, P, VarStore)> = vec![];

    for module_info in config.module_load_infos.iter() {

        if module_info.rank != Some(rank) {
            continue;
        }

        let path = Path::new(&module_info.path);

        if path.exists() {
            add_module_to_list(module_info, &mut loaded_models, &mut available_cuda)?;
        } else if rank != 0 {
            let cursor = request_model_from_root(&world, 0, path.as_os_str().to_str().unwrap());
            add_module_from_stream_to_list(module_info, &mut loaded_models, &mut available_cuda, cursor)?;
        } else {
            // If you are a root node and you can't find the file, then something has gone wrong.
            return Err(eyre!("Could not find the file {:?} on the root node.", path.as_os_str()));
        }
    }

    if rank == 0 {
        process_model_request(&world, size)?;
    }

    Ok(loaded_models)

}

fn request_model_from_root(world: &SimpleCommunicator, root: i32, path: &str) -> Cursor<Vec<u8>> {

    // Send request
    world.process_at_rank(root).send(path.as_bytes());

    let mut data: Vec<u8> = Vec::new();
    let mut buffer = vec![0u8; BUFFER_LEN];
    loop {

        let status = world.any_process().receive_into(&mut buffer);
        let just_a_byte = u8::equivalent_datatype();
        let count = status.count(just_a_byte) as usize;
        if count == 0 { break; }
        else {
            data.extend_from_slice(&buffer[..count]);
        }

    }
    // Signal that there is no more requests coming from this rank
    world.process_at_rank(root).send("NoMoreRequest".as_bytes());

    let cursor = Cursor::new(data);
    cursor
}

fn process_model_request(world: &SimpleCommunicator, size: i32) -> Result<()> {

    for rank in 1..size {

        let (v, status) = world.process_at_rank(rank).receive_vec::<u8>();
        let filename = std::str::from_utf8(&v)?;

        if filename == "NoMoreRequest" {
            continue;
        }

        let path = Path::new(&filename);
        if !path.exists() {
            return Err(eyre!("The requested file {} does not exist in the root rank", filename));
        }
        let mut file = File::open(path)?;
        let target = world.process_at_rank(status.source_rank());

        let mut buffer = [0u8; BUFFER_LEN];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                // Send the EOF signal
                target.send(&[] as &[u8]);
                break;
            } 
            else {
                target.send(&buffer[..count]);
            }
        }

    }
    Ok(())
}

pub fn now_into_filename() -> OsString {
    let now = Local::now();
    let filename = format!("{}.pv", now.format("%Y%m%d_%H%M"));
    filename.into()
}

fn run_wo_mpi_sequential<P: PVModel + Send + 'static, D: BoardData>(
    config: &FenrirConfig,
    model_config: ModelConfig,
    mcts_config: MCTSConfig
) -> Result<()> 
where
ReplayBuffer<D>: Sampler,
TBoard<<D as BoardData>::G>: ModelInput<D::G>,
TAction<<D::G as GameLogic>::B>: ActionTensor,
TaflBoard<<D::G as GameLogic>::B>: std::fmt::Display {

    assert!(!config.use_mpi);

    // let start = std::time::Instant::now();
    // let run_time = Duration::from_hours(config.run_time_hr);

    // Create a new model
    let vs = VarStore::new(Device::Cpu);
    let new_model = <P as PVModel>::new(&vs.root(), &model_config);
    let mut path: OsString = {
        let mut p = OsString::new();
        p.push("./test_models/");
        p.push(now_into_filename().as_os_str());
        p
    };

    let modules_load_infos = vec![ModuleLoadInfo::module_load_info(path.clone(), None, Device::Cpu, model_config.clone())];
    let model_setup_config = ModelSetupConfig::model_setup_config(false, modules_load_infos);
    let loaded_modules = setup::<P>(&model_setup_config)?;
    let (table, name_lookup) = model_setup_config.create_lookup_table(None, loaded_modules)?;

    let mut shelf = ModuleShelf::module_shelf(table, name_lookup);
    let replay_buffer = ReplayBuffer::<D>::new(config.replay_buffer_capacity);
    
    let mut storage = Cursor::new(Vec::<u8>::new());
 
    for i in 0..1 {

        let mut locked_shelf = LockedShelf::<P>::convert_from_shelf(shelf);

        if i != 0 {
            locked_shelf.update_modules_from_stream(0, &mut storage)?;
        }

        let (manager, request_senders) = InferenceManager::<'_, P, &'_ mut LockedShelf<P>>::new(&mut locked_shelf, config.mini_batch_size);

        self_play_new::<P,D>(
            manager,
            request_senders[0].clone(),
            config.n_self_play_games,
            &mcts_config,
            &replay_buffer
        )?;

        shelf = LockedShelf::<P>::convert_into_shelf(locked_shelf);
        let mut trainer = Trainer::<P>::new::<tch::nn::Sgd>(
            shelf.get_group_mut(0).expect("shelf is empty"),
            tch::nn::sgd(config.momentum, 0.0f64, config.weight_decay, false),
            (config.learning_rate_schedule)(0),
            config.weight_decay,
            config.mini_batch_size,
            ModelSyncConfig::new(train::SumOrAve::Ave)
        )?;

        trainer.train::<D, _>(config.n_training_step_per_cycle, config.n_training_step_per_cycle, &replay_buffer)?;
        trainer.save_to_stream(&mut storage);

        // path = OsString::new().push("./test_models/").push(now_into_filename().as_os_str());
        // loaded_modules[0].2.save(path)?;
        
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::fs;
    use tempfile::TempDir;
    use super::model::MOCK_MODEL_CONFIG;
    
    // Mock PVModel for testing
    #[derive(Clone, Debug)]
    struct MockPVModel {
        device: Device,
    }

    impl Default for MockPVModel {
        fn default() -> Self {
            Self { device: Device::Cpu }
        }
    }
    
    impl PVModel for MockPVModel {
        fn new(vs: &tch::nn::Path, config: &ModelConfig) -> Self {
            Self { device: vs.device() }
        }
        
        fn evaluate_t(&self, xs: &tch::Tensor, train: bool) -> (tch::Tensor, tch::Tensor) {
            let batch_size = xs.size()[0];
            (
                tch::Tensor::zeros(&[batch_size, 64], tch::kind::FLOAT_CPU),
                tch::Tensor::zeros(&[batch_size, 1], tch::kind::FLOAT_CPU)
            )
        }
        
        fn device(&self) -> Device {
            self.device
        }
    }


    #[test]
    fn test_agz_lr_schedule() {
        assert_eq!(agz_lr_schedule(0), 0.01);
        assert_eq!(agz_lr_schedule(200000), 0.01);
        assert_eq!(agz_lr_schedule(399999), 0.01);
        assert_eq!(agz_lr_schedule(400000), 0.001);
        assert_eq!(agz_lr_schedule(500000), 0.001);
        assert_eq!(agz_lr_schedule(599999), 0.001);
        assert_eq!(agz_lr_schedule(600000), 0.0001);
        assert_eq!(agz_lr_schedule(1000000), 0.0001);
    }

    #[test]
    fn test_model_setup_config_creation() {
        let module_info = ModuleLoadInfo::module_load_info(
            "test.pv".into(),
            None,
            Device::Cpu,
            MOCK_MODEL_CONFIG,
        );
        let config = ModelSetupConfig::model_setup_config(false, vec![module_info]);
        
        assert!(!config.use_mpi);
        assert_eq!(config.module_load_infos.len(), 1);
        assert_eq!(config.module_load_infos[0].path, OsString::from("test.pv"));
    }

    #[test]
    fn test_module_load_info_creation() {
        let info = ModuleLoadInfo::module_load_info(
            "model.pv".into(),
            Some(0),
            Device::Cuda(0),
            MOCK_MODEL_CONFIG,
        );
        
        assert_eq!(info.path, OsString::from("model.pv"));
        assert_eq!(info.rank, Some(0));
        assert_eq!(info.device, Device::Cuda(0));
    }

    #[test]
    fn test_create_lookup_table_without_mpi() {
        let loaded_models = vec![
            ("model1.pv".into(), MockPVModel::default(), VarStore::new(Device::Cpu)),
            ("model2.pv".into(), MockPVModel::default(), VarStore::new(Device::Cpu)),
            ("model1.pv".into(), MockPVModel::default(), VarStore::new(Device::Cpu)), // Duplicate name
        ];
        
        let module_infos = vec![
            ModuleLoadInfo::module_load_info("model1.pv".into(), None, Device::Cpu, MOCK_MODEL_CONFIG),
            ModuleLoadInfo::module_load_info("model2.pv".into(), None, Device::Cpu, MOCK_MODEL_CONFIG),
            ModuleLoadInfo::module_load_info("model1.pv".into(), None, Device::Cpu, MOCK_MODEL_CONFIG),
        ];
        
        let config = ModelSetupConfig::model_setup_config(false, module_infos);
        let result = config.create_lookup_table(None, loaded_models);
        
        assert!(result.is_ok());
        let (table, name_lookup) = result.unwrap();
        
        // Should have 2 unique names
        assert_eq!(name_lookup.len(), 2);
        assert!(name_lookup.contains(&OsString::from("model1.pv")));
        assert!(name_lookup.contains(&OsString::from("model2.pv")));
        
        // model1.pv should have 2 instances, model2.pv should have 1
        assert_eq!(table.len(), 2);
        let model1_idx = name_lookup.iter().position(|x| x == "model1.pv").unwrap();
        let model2_idx = name_lookup.iter().position(|x| x == "model2.pv").unwrap();
        assert_eq!(table[model1_idx].len(), 2);
        assert_eq!(table[model2_idx].len(), 1);
    }

    #[test]
    fn test_create_lookup_table_with_mpi_rank_filtering() {
        let loaded_models = vec![
            ("model1.pv".into(), MockPVModel::default(), VarStore::new(Device::Cpu)),
            ("model2.pv".into(), MockPVModel::default(), VarStore::new(Device::Cpu)),
            ("model3.pv".into(), MockPVModel::default(), VarStore::new(Device::Cpu)),
        ];
        
        let module_infos = vec![
            ModuleLoadInfo::module_load_info("model1.pv".into(), Some(0), Device::Cpu, MOCK_MODEL_CONFIG),
            ModuleLoadInfo::module_load_info("model2.pv".into(), Some(1), Device::Cpu, MOCK_MODEL_CONFIG),
            ModuleLoadInfo::module_load_info("model3.pv".into(), Some(0), Device::Cpu, MOCK_MODEL_CONFIG),
        ];
        
        let config = ModelSetupConfig::model_setup_config(true, module_infos);
        let result = config.create_lookup_table(Some(0), loaded_models);
        
        assert!(result.is_ok());
        let (table, name_lookup) = result.unwrap();
        
        // Should only have models for rank 0 (model1 and model3)
        assert_eq!(name_lookup.len(), 2);
        assert!(name_lookup.contains(&OsString::from("model1.pv")));
        assert!(name_lookup.contains(&OsString::from("model3.pv")));
        assert!(!name_lookup.contains(&OsString::from("model2.pv")));
    }

    #[test]
    fn test_create_lookup_table_mpi_consistency_check() {

        // Test inconsistency: use_mpi=true but rank=None
        let loaded_models = vec![];
        let module_infos = vec![];
        let config = ModelSetupConfig::model_setup_config(true, module_infos);
        let result = config.create_lookup_table::<MockPVModel>(None, loaded_models);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("inconsistency in mpi usage"));
        
        // Test inconsistency: use_mpi=false but rank=Some(0)
        let loaded_models = vec![];
        let module_infos = vec![];
        let config = ModelSetupConfig::model_setup_config(false, module_infos);
        let result = config.create_lookup_table::<MockPVModel>(Some(0), loaded_models);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("inconsistency in mpi usage"));
    }

    #[test] 
    fn test_now_into_filename() {
        let filename = now_into_filename();
        let filename_str = filename.to_string_lossy();
        
        // Should end with .pv
        assert!(filename_str.ends_with(".pv"));
        
        // Should have the format YYYYMMDD_HHMM.pv (at least)
        assert!(filename_str.len() >= 14); // "20250126_1234.pv" = 16 chars
        
        // Should contain underscore
        assert!(filename_str.contains('_'));
        
        // Should start with a digit (year)
        assert!(filename_str.chars().next().unwrap().is_ascii_digit());
    }

    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_setup_without_mpi_feature_enabled() {
        let module_infos = vec![
            ModuleLoadInfo::module_load_info("test.pv".into(), None, Device::Cpu, MOCK_MODEL_CONFIG)
        ];
        let config = ModelSetupConfig::model_setup_config(true, module_infos);
        
        let result = setup::<MockPVModel>(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("MPI functionality requested"));
    }

    #[test]
    fn test_setup_wo_mpi_with_nonexistent_file() {
        let module_infos = vec![
            ModuleLoadInfo::module_load_info("nonexistent.pv".into(), None, Device::Cpu, MOCK_MODEL_CONFIG)
        ];
        let config = ModelSetupConfig::model_setup_config(false, module_infos);
        
        let result = setup::<MockPVModel>(&config);
        // This should fail because the file doesn't exist
        assert!(result.is_err());
    }

    #[test]
    fn test_load_module_info_with_different_devices() {
        // Test CPU device
        let cpu_info = ModuleLoadInfo::module_load_info(
            "cpu_model.pv".into(),
            None,
            Device::Cpu,
            MOCK_MODEL_CONFIG
        );
        assert_eq!(cpu_info.device, Device::Cpu);
        
        // Test CUDA device
        let cuda_info = ModuleLoadInfo::module_load_info(
            "cuda_model.pv".into(),
            Some(0),
            Device::Cuda(0),
            MOCK_MODEL_CONFIG
        );
        assert_eq!(cuda_info.device, Device::Cuda(0));
    }

    #[test]
    fn test_fenrir_config_with_agz_lr_schedule() {
        let config = FenrirConfig {
            n_self_play_nodes: 2,
            n_train_nodes: 1,
            n_model_replica_self_play: 4,
            n_model_replica_train: 2,
            use_mpi: false,
            mini_batch_size: 64,
            n_training_step_per_cycle: 1000,
            n_self_play_games: 100,
            n_games_per_tournament: 10,
            model_update_threshold: 0.55,
            concurrent_training: true,
            replay_buffer_capacity: 50000,
            run_time_hr: 2,
            momentum: 0.9,
            learning_rate_schedule: agz_lr_schedule,
            weight_decay: 0.0001,
        };
        
        // Test that the learning rate schedule function works
        assert_eq!((config.learning_rate_schedule)(100000), 0.01);
        assert_eq!((config.learning_rate_schedule)(500000), 0.001);
        assert_eq!((config.learning_rate_schedule)(700000), 0.0001);
        
        assert!(!config.use_mpi);
        assert_eq!(config.mini_batch_size, 64);
        assert_eq!(config.momentum, 0.9);
    }
}
