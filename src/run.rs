#![cfg(feature = "mpi")]

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::sync::RwLock;
use std::time::Duration;
use std::path::Path;
use std::io::{Read, Seek, Cursor};
use std::convert::AsRef;

use crate::agent::MCTSConfig;
use crate::model::{self, ModelConfig};
use crate::model::PVModel;
use crate::replay_buffer::ReplayBuffer;
use crate::run;
use crate::self_play::{self, self_play_new, InferenceManager, ModuleShelf};
use crate::self_play::self_play;
use crate::train::{self, ModelSyncConfig, Trainer};
use crate::replay_buffer;

use color_eyre::eyre::eyre;
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
use chrono::{Uts, Local};

#[derive(Debug, Clone)]
struct FenrirConfig {
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
    learning_rate_schedule: fn(usize) -> f64,
    weight_decay: f64,
}

const fn agz_lr_schedule(n_steps: usize) -> f64 {
    match n_steps {
        0..400000 => 0.01f64,
        400000..600000 => 0.001f64,
        600000.. => 0.0001f64,
    }
}

// Set up related stuff
struct ModelSetupConfig{
    use_mpi: bool,
    module_load_infos: Vec<ModuleLoadInfo>,
}

impl ModelSetupConfig {
    fn model_setup_config(use_mpi: bool, module_load_infos: Vec<ModuleLoadInfo>) -> Self {
        Self {
            use_mpi,
            module_load_infos
        }
    }
    // This takes the output of the setup function (loaded models along with their var stores and their path names), and create correspondence between their names and their group id.
    // This function operates locally (inside each mpi node)
    fn create_lookup_table<P: PVModel>(&self, rank: Option<i32>, loaded_model: Vec<(OsString, P, VarStore)>) -> 
    Result<(Vec<Vec<(P, VarStore)>>, Vec<OsString>)> {
        
        let mut table: Vec<Vec<(P, VarStore)>> = vec![];
        let mut name_lookup: Vec<OsString> = vec![];

        // taking XOR
        if self.use_mpi ^ rank.is_some() {
            return Err(eyre!(
                "inconsistency in mpi usage-related input: rank should be None if and only if use_mpi is false"
            ));
        }
        
        for ((name, model, vs), info) in loaded_model.into_iter().zip(self.modules.iter()) {
            
            if let Some(i) = rank && info.rank != i {
                continue;
            }

            if let Some(id) = name_lookup.iter().position(|&&s| s == name){
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

struct ModuleLoadInfo {
    // path to the file that stores the model weight
    path: OsString,
    // only specify when using mpi
    rank: Option<i32>,
    // device to load the model onto.
    device: Device,
    // model config
    config: ModelConfig,
}

impl ModuleLoadInfo {
    fn module_load_info(path: OsString, rank: Option<i32>, device: Device, config: ModelConfig) -> Self {
        Self{
            path,
            rank,
            device,
            config
        }
    }
}

fn load_module<P: PVModel>(info: &ModuleLoadInfo) -> (OsString, P, VarStore) {

    let path = Path::from(info.path);
    let mut vs = VarStore::new(info.device);
    let module: P = P::new(&vs.root(), info.config);
    vs.load(path);
    (info.path.to_owned(), module, vs)

}

fn load_module_from_stream<P: PVModel, S: Read + Seek>(info: &ModuleLoadInfo, stream: S) -> (OsString, P, VarStore) {

    let mut vs = VarStore::new(info.device);
    let module: P = P::new(&vs.root(), info.config);
    vs.load_from_stream(stream);
    (info.path.to_owned(), module, vs)

}

fn setup<P: PVModel>(config: &ModelSetupConfig) -> Result<Vec<(OsString, P, VarStore)>>{

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
            let (name, module, var_store) = load_module::<P>(&info);
            list.push((name, module, var_store));
        },
        Device::Cuda(i) => {

            if !available_cuda.contains(&info.device) {
                return Err(eyre!("cuda allocation failed: Check the ModelSetupConfig again"));
            }
            available_cuda.remove(i);
            let (name, module, var_store) = load_module::<P>(&info);
            list.push((name, module, var_store));
            
        },
        _ => {
            return Err(eyre!("unexpected device encountered"));
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
            let (name, module, var_store) = load_module_from_stream::<P, S>(&info, stream);
            list.push((name, module, var_store));
        },
        Device::Cuda(i) => {

            if !available_cuda.contains(&info.device) {
                return Err(eyre!("cuda allocation failed: Check the ModelSetupConfig again"));
            }
            available_cuda.remove(i);
            let (name, module, var_store) = load_module_from_stream::<P, S>(&info, stream);
            list.push((name, module, var_store));
            
        },
        _ => {
            return Err(eyre!("unexpected device encountered"));
        }
    }
}

// This function loads the models, ignoring the mpi options. It is intended to be used inside the 'setup' function
fn setup_wo_mpi<P: PVModel>(config: &ModelSetupConfig) -> Result<Vec<(OsString, P, VarStore)>> {

    let mut loaded_models: Vec<(OsString, P, VarStore)> = vec![];

    let n_cuda = tch::Cuda::device_count();
    let mut available_cuda: Vec<Device>  = (0..n_cuda).map(|i| Device::Cuda(i)).collect();

    for module_info in config.modules.iter() {
        add_module_to_list(module_info, &mut loaded_models, &mut available_cuda)?;
    }
    Ok(loaded_models)
}

#[cfg(feature = "mpi")]
fn setup_w_mpi<P: PVModel>(config: &ModelSetupConfig) -> Result<Vec<(OsString, P, VarStore)>> {

    use mpi::traits::Communicator;
    // mpi related setup
    let universe = mpi::initialize()?;
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let n_cuda = tch::Cuda::device_count();
    let mut available_cuda: Vec<Device> = (0..n_cuda).map(|i| Device::Cuda(i)).collect();
    let mut loaded_models: Vec<(OsString, P, VarStore)> = vec![];

    for module_info in config.modules.iter() {

        if module_info.rank != rank {
            continue;
        }

        let path = Path::from(module_info.path);

        if path.exists() {
            add_module_to_list(module_info, &mut loaded_models, &mut available_cuda)?;
        } else if rank != 0 {
            let cursor = request_model_from_root(&world, 0, path.as_os_str().to_str().unwrap());
            add_module_from_stream_to_list(module_info, loaded_models, available_cuda, cursor)?;
        } else {
            // If you are a root node and you can't find the file, then something has gone wrong.
            return Err(eyre!("Could not find the file {} on the root node.", path.as_os_str()));
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
    let mut buffer = vec![0u8; 1024];
    loop {

        let status = world.any_process().receive_into(&mut buffer);
        let just_a_byte = u8::equivalent_datatype();
        let count = status.count(just_a_byte);
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
        let filename = str::from_utf8(v)?;

        if filename == "NoMoreRequest" {
            continue;
        }

        let path = Path::new(&filename);
        if !path.exists() {
            return Err(eyre!("The requested file {} does not exist in the root rank", filename));
        }
        let file = File::open(path)?;
        let target = world.process_at_rank(status.source_rank());

        let mut buffer = [0u8; 1024];
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

fn now_into_filename() -> OsString {
    let now = Local::now();
    let filename = format!("{}.pv", now.format("%Y%m%d_%H%M"));
    filename.into()
}

fn run_wo_mpi_sequential<P: PVModel>(config: &FenrirConfig, model_config: &ModelConfig, mcts_config: &MCTSConfig) -> Result<()> {

    assert!(!config.use_mpi);

    // let start = std::time::Instant::now();
    // let run_time = Duration::from_hours(config.run_time_hr);

    // Create a new model
    let vs = VarStore::new(Device::Cpu);
    let new_model = P::new(vs.root(), model_config.clone());
    let mut path: OsString = OsString::new().push("./test_models/").push(now_into_filename().as_os_str());

    let modules_load_infos = vec![ModuleLoadInfo::module_load_info(path.clone(), None, Device::Cpu, model_config.clone())];
    let model_setup_config = ModelSetupConfig::model_setup_config(false, modules_load_infos);
    let loaded_modules = setup::<P>(model_setup_config)?;
    let (table, name_lookup) = model_setup_config.create_lookup_table(None, loaded_modules)?;
    let mut shelf = ModuleShelf::module_shelf(table, name_lookup);
    let replay_buffer = ReplayBuffer::new(config.replay_buffer_capacity);
    
    let mut storage = Vec::<u8>::new();
 
    for i in 0..2 {

        if i != 0 {
            shelf.update_modules_from_stream(0, Cursor::new(storage))?;
        }

        let (manager, request_senders) = InferenceManager::<P>::new(&mut shelf, config.mini_batch_size);

        self_play_new(
            manager,
            request_senders[0],
            config.n_self_play_games,
            mcts_config,
            replay_buffer
        )?;

        let trainer = Trainer::<P>::new::<tch::nn::Sgd>(
            shelf.get_group_mut(0),
            tch::nn::sgd(config.momentum, 0.0f64, config.weight_decay, false),
            (config.learning_rate_schedule)(0),
            config.weight_decay,
            config.mini_batch_size,
            ModelSyncConfig::new(train::SumOrAve::Ave)
        )?;

        trainer.train(config.n_training_step_per_cycle, config.n_training_step_per_cycle, replay_buffer)?;
        trainer.save_to_stream(Cursor::new(&mut storage));

        // path = OsString::new().push("./test_models/").push(now_into_filename().as_os_str());
        // loaded_modules[0].2.save(path)?;
        
    }
}
