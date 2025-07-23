#![cfg(feature = "mpi")]

use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::time::Duration;
use std::path::Path;
use std::io::{Read, Seek, Cursor};

use crate::agent::MCTSConfig;
use crate::model::ModelConfig;
use crate::model::PVModel;
use crate::replay_buffer::ReplayBuffer;
use crate::run;
use crate::self_play;
use crate::self_play::self_play;
use crate::train;
use crate::replay_buffer;

use color_eyre::eyre::eyre;
use mpi::point_to_point::*;
use mpi::topology::SimpleCommunicator;
use mpi::traits::Communicator;
use mpi::MpiError;
use mpi::datatype::Equivalence;

use color_eyre::eyre::Result;
use rand::prelude::*;
use tch::nn::ModuleT;
use tch::nn::VarStore;
use tch::vision::image::load;
use tch::Device;

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
    weight_decay_factor: f64,
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
    modules: Vec<ModuleLoadInfo>,
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

fn load_module<P: PVModel>(info: &ModuleLoadInfo) -> (P, VarStore) {

    let path = Path::from(info.path);
    let mut vs = VarStore::new(info.device);
    let module: P = P::new(&vs.root(), info.config);
    vs.load(path);
    (module, vs)

}

fn load_module_from_stream<P: PVModel, S: Read + Seek>(info: &ModuleLoadInfo, stream: S) -> (P, VarStore) {

    let mut vs = VarStore::new(info.device);
    let module: P = P::new(&vs.root(), info.config);
    vs.load_from_stream(stream);
    (module, vs)

}

fn setup<P: PVModel>(config: ModelSetupConfig) -> Result<Vec<(P, VarStore)>>{

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

// This function loads the models, ignoring the mpi options. It is intended to be used inside the 'setup' function
fn setup_wo_mpi<P: PVModel>(config: ModelSetupConfig) -> Result<Vec<(P, VarStore)>> {

    let mut loaded_models: Vec<(P, VarStore)> = vec![];

    let n_cuda = tch::Cuda::device_count();
    let available_cuda: Vec<Device>  = (0..n_cuda).map(|i| Device::Cuda(i)).collect();

    for module_info in config.modules {
        if !available_cuda.contains(&module_info.device) {
            return Err(eyre!("cuda allocation failed: Check the ModelSetupConfig again"));
        }
        if let Device::Cuda(i) = module_info.device {
            available_cuda.remove(i);
        }
        let (module, var_store) = load_module::<P>(&module_info);
        loaded_models.push((module, var_store));
    }
    Ok(loaded_models)
}

#[cfg(feature = "mpi")]
fn setup_w_mpi<P: PVModel>(config: ModelSetupConfig) -> Result<Vec<(P, VarStore)>> {

    use mpi::traits::Communicator;

    let universe = mpi::initialize()?;
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let mut loaded_models: Vec<(P, VarStore)> = vec![];

    for module_info in config.modules {

        if module_info.rank != rank {
            continue;
        }

        let path = Path::from(module_info.path);
        if path.exists() {
            let (module, var_store) = load_module::<P>(&module_info);
            loaded_models.push((module, var_store));
        } else if rank != 0 {
            let cursor = request_model_from_root(&world, 0, path.as_os_str().to_str().unwrap());
            let (module, var_store) = load_module_from_stream::<P>(&module_info, cursor);
            loaded_models.push((module, var_store));
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

fn run_wo_mpi_sequential(config: &FenrirConfig, model_config: &ModelConfig, mcts_config: &MCTSConfig) -> Result<()> {
    assert!(!config.use_mpi);
    let start = std::time::Instant::now();
    let run_time = Duration::from_hours(config.run_time_hr);
    let mut model_opt = None;
    let mut rng = thread_rng();
    while start.elapsed() < run_time {

        let replay_buffer = ReplayBuffer::new(config.replay_buffer_capacity);
        self_play(
            model_opt,
            config.n_model_replica_self_play,
            config.n_self_play_games,
            config.replay_buffer_capacity,
            model_config,
            mcts_config,
            &replay_buffer
        )?;

        
    }
}
