#![allow(unused)]
#![allow(dead_code)]
#![cfg(feature = "torch")]

//internal
use game::game::{Game, GameState, Side};
use game::board::{TaflBoardEleven};
use bitboard::eleven::{ElevenBoardPositionalEncoding, MoveOnBoardEleven};
use mpi::datatype::MutView;
use crate::{model, utils};
use crate::replay_buffer::{self, Episode, EpisodeUnit, ReplayBuffer};
use crate::model::{ModelConfig, PVModel};
use crate::model::PVNet;
use crate::model::Evaluation;
use crate::agent::{self, Actor, MCTSConfig, MCTSTree, Temperature, PosteriorDist};
use crate::run::ModelSetupConfig;

//external
use tch::Tensor;
use tch::nn;
use tch::Device;
use tch::Kind;
use tch::nn::VarStore;
use color_eyre::eyre::{eyre, Context, ErrReport, OptionExt, Result};

// multithreading-related
use crossbeam::channel::{unbounded, Sender, Receiver};
use rayon::{iter::{IntoParallelIterator, ParallelIterator}};

use std::mem::MaybeUninit;
// std
use std::path::Path;
use std::thread::{self, current, JoinHandle};
use std::sync::{Arc, RwLock};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use std::ffi::OsString;
use std::path;
use std::convert::AsRef;
use std::borrow::Borrow;

// needs experimenting
const BATCHSIZE: i64 = 64;
// maximum duration for the director to wait for request in millis
const DIRECTOR_RCV_TIMEOUT: Duration = Duration::from_millis(1);

// Batching works as follows. Each worker sends a request (a Tensor and a Sender of the inference result) to the Directer instance.
// Directer then fills the vector of requests, which is then sent to the designated Device when the send threshold is reached. 
// Batch has two states: Free and Computing. Free means it is currently not waiting for an inference process to finish, and can send 
// the batch anytime. Computing means it is currently waiting for the inference results to come back, and it cannot send another batch 
// until the results get sent back to requesting workers, although you can still fill the next batch.
// Dispatcher keeps track of the length of each input tensors and holds the sender until the inference results comes back.
#[derive(Debug)]
pub struct Batch {
    requests: Vec<Request>,
    dispatcher: Vec<(i64, Arc<Sender<Evaluation>>)>,
    capacity: usize,
    send_threshold: i64,
    state: BatchState,
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum BatchState {
    Free,
    Computing,
}

impl Default for BatchState {
    fn default() -> Self {
        BatchState::Free
    }
}

impl Batch {
    pub fn new(capacity: usize, send_threshold: i64) -> Self {
        Batch {
            requests: Vec::with_capacity(capacity),
            dispatcher: vec![],
            capacity,
            send_threshold,
            state: BatchState::default()
        }
    }

    pub fn prepare_to_send(&mut self) -> Tensor {

        self.dispatcher.clear();
        // This will leave the self.requests empty
        let requests = std::mem::take(&mut self.requests);
        let (queries, dispatcher_info):(Vec<Query>, Vec<(i64, Arc<Sender<Evaluation>>)>) =
            requests.into_iter()
            .map(|r| {
                let size = r.query.size()[0];
                let sender = r.sender.clone();
                (r.query, (size, sender))
            })
            .unzip();
        let ts = Tensor::cat(&queries, 0);
        self.dispatcher = dispatcher_info;

        ts
    }

    pub fn status_free(&mut self) {
        self.state = BatchState::Free;
    }

    pub fn status_computing(&mut self) {
        self.state = BatchState::Computing;
    }

    pub fn push(&mut self, r: Request) {
        self.requests.push(r);
    }

    pub fn extend<T: AsRef<[Request]>>(&mut self, slice: T) {
        self.requests.extend_from_slice(slice.as_ref());
    }

    // This will make the vr empty
    pub fn append(&mut self, vr: &mut Vec<Request>) {
        self.requests.append(vr);
    }

    pub fn send_criteria_check(&self) -> bool {
        self.internal_length > self.send_threshold
    }

    pub fn clear_everything_except_dispatcher(&mut self) {
        self.requests.clear();
        self.internal_length = 0;
    }

    pub fn clear_dispatcher(&mut self) {
        self.dispatcher.clear();
    }

    pub fn clear(&mut self) {
        self.clear_everything_except_dispatcher();
        self.clear_dispatcher();
    }
}


#[derive(Debug)]
pub struct Request {
    query: Query,
    sender: Arc<Sender<Evaluation>>,
}

impl Request {
    pub fn new(query: Query, sender: Arc<Sender<Evaluation>>) -> Self {
        Self { query, sender }
    }
}

// This is to denote the mini-batched inference request from a single worker at a time.
// So it should have the size of (N, in_features, 11, 11)
// TODO: Figure out a way of indexing the returned tensor into each individual request (game state encoding)
pub type Query = Tensor;


// A struct that manages model allocation, receiving inference requests, collection and redistribution of results.
#[derive(Debug)]
pub struct Directer<P: PVModel> {
    device_allocation: Vec<Device>,
    modules: Vec<P>,
    receiver: Receiver<(Request, usize)>,
    // each element holds one batch for one model.
    pub batches: Vec<Batch>,
    // Stores the indices of the model in "modules" field
    module_ind: Vec<Vec<usize>>,
    // table of model name ("new" for newly created model, file_name for pre-trained model) against indexes in the "batches" vec. 
    model_lookup: HashMap::<String, usize>
}

impl<P: PVModel> Directer<P: PVModel> {
    // Creates a new instance of Directer along with the Sender corresponding to its receiver.
    pub fn new<P: PVModel, S: AsRef<OsString>>(config: &ModelConfig, new_model_count: i64, trained_models: Vec<(i64, S)>, batch_capacity: usize) -> Result<(Self, Sender<(Request, usize)>)> {

        let mut device_allocation: Vec<Device> = vec![];
        let mut modules: Vec<P> = vec![];
        let mut model_lookup = HashMap::<String, usize>::new();
        let mut module_ind: Vec<Vec<usize>> = vec![];

        let mut cuda_left = tch::Cuda::device_count();
        if cuda_left < new_model_count + trained_models.iter().fold(0, |acc, (n,_)| acc + n)
        {
            println!("The number of required models exceeds the number of CUDA devices available. Some models will be loaded onto Cpu");
        }

        let mut model_id: usize = 0;
        let mut cuda_index: usize = 0;
        for (id_minus_one, (n, ostr)) in trained_models.iter().enumerate()
        {
            let path = path::Path::new(ostr.as_ref());
            debug_assert!(path.is_file());
            let mut module_indices = vec![];
            for _ in 0..*n
            {
                let net = if cuda_left == 0 
                {
                    let mut vs = nn::VarStore::new(Device::Cpu);
                    device_allocation.push(Device::Cpu);
                    vs.load(path)?;
                    P::new(&vs.root(), config)
                } 
                else 
                {
                    let mut vs = nn::VarStore::new(Device::Cuda(cuda_index));
                    device_allocation.push(Device::Cuda(cuda_index));
                    vs.load(path)?;
                    cuda_index += 1;
                    cuda_left -= 1;
                    P::new(&vs.root(), config)
                };
                module_indices.push(modules.len());
                modules.push(net);
            }
            model_lookup.insert(path.clone(), model_id);
            module_ind.push(module_indices);
            model_id += 1;
        }

        let mut new_module_indices = vec![];
        if new_model_count > 0 {

            let original_vs = nn::VarStore::new(Device::Cpu);
            // define a dummy model just so that original_vs will be properly initialized
            let dummy_model = P::new(&original_vs.root(), config);

            for _ in 0..new_model_count {
                let net = if cuda_left == 0 
                {
                    let mut vs = nn::VarStore::new(Device::Cpu);
                    let replica = P::new(&vs.root(), config);
                    vs.copy(&original_vs);
                    device_allocation.push(Device::Cpu);
                    replica
                } else 
                {
                    let mut vs = nn::VarStore::new(Device::Cpu);
                    let replica = P::new(&vs.root(), config);
                    vs.copy(&original_vs);
                    vs.set_device(Device::Cuda(cuda_index));
                    device_allocation.push(Device::Cuda(cuda_index));
                    cuda_index += 1;
                    cuda_left -= 1;
                    replica
                };
                new_module_indices.push(modules.len());
                modules.push(net);
            }
            

        }
        model_lookup.insert("new".to_owned(), model_id);
        module_ind.push(new_module_indices);
        model_id += 1;

        let (sender, receiver) = unbounded::<(Request, usize)>();

        let batches: Vec<Batch> = (0..module_ind.len()).map(|i| {
            let send_threshold = BATCHSIZE * module_ind[i].len() as i64;
            Batch::new(batch_capacity, send_threshold)
        }).collect();

        Ok((
            Self {
                device_allocation,
                modules,
                receiver,
                batches,
                model_lookup,
                module_ind
            },
            sender
        ))
    }

    pub fn lookup_batch_from_model(&self, model: String) -> Result<&Batch> {
        let id = self.model_lookup.get(&model)
            .ok_or_eyre(format!("Couldn't find a model named {}", model))?
            .to_owned();

        let batch_ref = self.batches.get(id)
            .ok_or_eyre("could not find the target batch from the director")?;
        Ok(batch_ref)
    }

    pub fn lookup_id_from_model(&self, model: String) -> Result<usize> {
        let id = self.model_lookup.get(&model)
            .ok_or_eyre(format!("Couldn't find a model named {}", model))?
            .to_owned();

        Ok(id)
    }

    pub fn get_batch_ref(&self, model_id: usize) -> Result<&Batch> {
        let batch_ref = self.batches.get(model_id)
            .ok_or_eyre("could not find the target batch from the director")?;
        Ok(batch_ref)
    }

    pub fn accept_request(&mut self, request: Request, model_id: usize) -> Result<()> {
        let batch_ref = self.batches.get_mut(model_id)
            .ok_or_eyre("Could not find the specified model id")?;
        batch_ref.push(request);
        Ok(())
    }

    pub fn process_batch(&mut self, batch_id: usize) -> Result<()> {
        let batch = &mut self.batches[batch_id];

        let ts = batch.prepare_to_send();
        let num_devices = self.module_ind[batch_id].len();

        if num_devices == 0 {
            return Err(ErrReport::msg("No device was allocated for this batch"));
        }
        
        let mut split_points: Vec<usize> = Vec::with_capacity(num_devices);
        let mut split_sizes: Vec<i64> = Vec::with_capacity(num_devices);
        let mut current_size = 0i64;

        let target_size_per_device = batch.internal_length / num_devices as i64;

        // book-keeping the slice locations so that later we can create mini-dispatchers
        // let mut dispatcher_slicer: Vec<usize> = vec![];

        for (n, (len, _)) in batch.dispatcher.iter().enumerate() {

            current_size += len;

            if current_size > target_size_per_device && split_sizes.len() < num_devices - 1 {
                split_sizes.push(current_size);
                // dispatcher_slicer.push(n);
                split_points.push(n + 1);
                current_size = 0;
            }
        }

        if current_size > 0 {
            split_sizes.push(current_size);
        }

        // dispatcher_slicer.push(batch.dispatcher.len());
        
        // make the batch state "Computing"
        batch.clear_everything_except_dispatcher();
        batch.status_computing();

        let mut dispatcher_ranges: Vec<(usize, usize)> = Vec::new();
        let mut start = 0;
        for end in split_points {
            dispatcher_ranges.push((start, end));
            start = end;
        }
        dispatcher_ranges.push((start, batch.dispatcher.len()));

        // creating mini-batchers, responsible of broadcasting the results from the sliced inputs back to the correct workers
        // TODO! right now it doesn't work, have to find a better way
        // let mut mini_dispatchers: Vec<&[(i64, Arc<Sender<Evaluation>>)]> = Vec::with_capacity(num_devices);
        // mini_dispatchers.push(&batch.dispatcher[0..=dispatcher_slicer[0]]);
        // for i in 0..num_devices - 1{
        //     mini_dispatchers.push(&batch.dispatcher[dispatcher_slicer[i]+1..=dispatcher_slicer[i+1]]);
        // }

        let tss = ts.split_with_sizes(split_sizes.as_slice(), 0);

        // Now, feed the tensors into each device and get the result
        let infer_results: Result<Vec<Evaluation>> = tss
            .into_iter()
            .enumerate()
            .map(|(i,xs)| -> Result<Evaluation> {

                if i >= self.module_ind[batch_id].len() {
                    return Err(ErrReport::msg("Device index out of bounds"));
                }
                let module_id = self.module_ind[batch_id][i];
                let xs_allocated = xs.to(self.device_allocation[module_id]);
                let mut evaluation = self.modules[module_id].evaluate_t(&xs_allocated, false);
                evaluation.0 = evaluation.0.to(Device::Cpu);
                evaluation.1 = evaluation.1.to(Device::Cpu);
                Ok(evaluation)

            }).collect();

        let infer_results = infer_results?;

        for (device_idx, (start_idx, end_idx)) in dispatcher_ranges.into_iter().enumerate() {
            if device_idx >= infer_results.len() {
                continue;
            }
            let dispatcher_slice = &batch.dispatcher[start_idx..end_idx];
            let device_split_sizes: Vec<i64> = dispatcher_slice.iter()
                .map(|(len, _)| *len).collect();

            if device_split_sizes.is_empty() {
                continue;
            }

            let policy_splits = infer_results[device_idx].0
                .split_with_sizes(&device_split_sizes, 0);
            let values_splits = infer_results[device_idx].1
                .split_with_sizes(&device_split_sizes, 0);

            for ((policy, values), (_, sender)) in policy_splits
                .into_iter()
                .zip(values_splits)
                .zip(dispatcher_slice.iter())
            {
                sender.send((policy, values))
                    .map_err(|_| ErrReport::msg("Error happened while sending the inference results"))?;
            }
        }

        // Now, broadcast the evaluations back to the requesting workers
        // for (n, &mini_dispatcher) in mini_dispatchers.into_iter().enumerate() {
        //     let split_sizes: Vec<i64> = mini_dispatcher.iter().map(|(i, _)| *i).collect();
        //     infer_results[n].0
        //     .split_with_sizes(split_sizes, 0)
        //     .into_iter()
        //     .zip(infer_results[n].1.split_with_sizes(split_sizes, 0))
        //     .zip(mini_dispatcher.into_iter())
        //     .map(|(eval, (_, arc_sender))| {
        //         arc_sender.send(eval);
        //     })?;
        // }

        batch.clear_dispatcher();
        batch.status_free();

        Ok(())

    }

    pub fn run(&mut self) -> Result<()> {
        loop {
            if let Ok((request, model_id)) = self.receiver.recv_timeout(DIRECTOR_RCV_TIMEOUT)
            {
                self.accept_request(request, model_id)?;
                let batch_ref = self.get_batch_ref(model_id)?;

                if batch_ref.state == BatchState::Free && batch_ref.send_criteria_check() {
                    self.process_batch(model_id)?;
                } else { continue; }

            }
            else {
                for model_id in 0..self.batches.len() {
                    let batch_state = self.batches[model_id].state;
                    let is_ok_to_send = self.batches[model_id].send_criteria_check();
                    match batch_state {
                        BatchState::Free if is_ok_to_send => {
                            self.process_batch(model_id)?;
                        },
                        _ => continue
                    };
                }
            }
        }
        Ok(())
    }
}



// play the game for the specified amount using the specified model, create the replay buffer
pub fn self_play(
    model_opt: Option<String>,
    n_replica: usize,
    num_games: usize,
    batch_capacity: usize,
    model_config: &ModelConfig,
    mcts_config: &MCTSConfig,
    replay_buffer: &ReplayBuffer,
) -> Result<()>{

    let (mut directer, sender) = match model_opt {

        Some(ref model_name) => Directer::new(
            model_config, 
            0,
            vec![(n_replica, model_name.clone())],
            batch_capacity),
        None => Directer::new(
            model_config,
            n_replica,
            vec![],
            batch_capacity)

    }?;

    let request_sender = Arc::new(sender);
    let model_name = model_opt.unwrap_or("new".to_owned());
    let model_id = directer.lookup_id_from_model(model_name.clone())?;

    let directer_channel = thread::spawn( move || {
        directer.run()
    });


    (0..num_games).into_par_iter().for_each_init(
        || {
            let (sender, receiver) = unbounded::<Evaluation>();
            let sender = Arc::new(sender);
            let game = Game::default();
            let mut mcts_tree = MCTSTree::generate(game, *mcts_config);
            let actor = Actor::new_with_model_id(request_sender.clone(), model_id);
            (sender, receiver, mcts_tree, actor)
        }, 
    |(
            vp_sender,
            vp_receiver,
            mcts_tree,
            actor
        ), worker_idx| {

            let mut episode: Episode = Episode::new();

            while !mcts_tree.root_is_terminal() {
                let (game,_, posterior) = mcts_tree.get_policy_and_update(actor).unwrap();
                episode.append_wo_reward(&game, posterior);
            }
            let winner = mcts_tree.get_winner();
            match winner {
                Side::Att => episode.give_reward(1i64),
                Side::Def => episode.give_reward(-1i64),
            };
            replay_buffer.append_from_episode(&mut episode);
    });

    directer_channel.join().unwrap();

    Ok(())
}

// N denotes the number of groups (different models)
// ModuleShelf stores the model replicas and its variable stores, grouped by the model they are derived from.
pub struct ModuleShelf<P: PVModel, A: AsRef<P>> {
    // This separates the model by group id
    table: Vec<Vec<(A, VarStore)>>,
    // This is a look up function for model_name (file_name) <-> group id
    name_lookup: Vec<OsString>,
}

impl<P: PVModel, A: AsRef<P>> ModuleShelf<P, A> {

    pub fn module_shelf<P: PVModel, A: AsRef<P>>(table: Vec<Vec<(A, VarStore)>>, name_lookup: Vec<OsString>) -> Self{
        Self { table, name_lookup }
    }

    pub fn get_group_id<T: Borrow<OsStr>>(&self, name: T) -> Option<usize>
    {
        self.name_lookup
            .iter()
            .position(|os_str| os_str.borrow() == name.borrow())
    }
}

struct NewBatch {
    requests: VecDeque<Request>,
}

impl NewBatch {
    fn new() -> Self {
        let requests = VecDeque::<Request>::new();
        Self { requests }
    }

    fn push_request(&mut self, request: Request) {
        self.requests.push_back(request);
    }
}

// This struct collects the inference request during self-play and tournament, and batch them appropriately.
// It is intended to be set up per machine
pub struct InferenceManager<'a, P: PVModel, const N: usize> {

    shelf: &'a mut ModuleShelf<P, P>,
    batch: Vec<NewBatch>,
    request_receivers: Vec<Receiver<Request>>,
    status: Vec<Vec<ModuleStatus>>,
    completion_receivers: Vec<Receiver<usize>>,
    completion_senders: Vec<Sender<usize>>,
    batch_size: usize,
    threads: [JoinHandle<()>; N],
    job_senders: [Sender<(Tensor, P, usize, Vec<Arc<Sender<Evaluation>>>, Sender<usize>)>; N],
    compass: usize,
}

#[derive(Debug, PartialEq)]
enum ModuleStatus {
    Free,
    Occupied,
}

impl Default for ModuleStatus {
    fn default() -> Self {
        ModuleStatus::Free
    }
}

impl ModuleStatus {
    fn flip(&mut self) {
        match *self {
            Self::Free => self = Self::Occupied,
            Self::Occupied => self = Self::Free,
        };
    }
}

type Job<P> = (Tensor, P, usize, Vec<Arc<Sender<Evaluation>>>, Sender<usize>);

impl<'a, P: PVModel, const N: usize> InferenceManager<'a, P, N> {

    pub fn new(ref_shelf: &'a mut ModuleShelf<P, P>, capacity: usize, send_threshold: usize, batch_size: usize) -> (Self, Vec<Sender<Request>>) {

        let len = ref_shelf.table.len();
        let batch = vec![NewBatch::new(); len];
        let status = ref_shelf.table.iter().map(|x| {
            vec![ModuleStatus::default(); x.len()]
        }).collect::<Vec<Vec<ModuleStatus>>>();


        let (completion_senders, completion_receivers) : (Vec<_>, Vec<_>) = (0..len)
            .map(|_| unbounded::<usize>())
            .unzip();

        let (request_senders, request_receivers) : (Vec<_>, Vec<_>) = (0..len)
            .map(|_| unbounded::<Request>())
            .unzip();

        let mut uninit_job_senders: [MaybeUninit::<Sender<Job<P>>>; N] = unsafe {
            MaybeUninit::uninit().assume_init()
        };
        let mut uninit_job_receivers: [MaybeUninit::<Receiver<Job<P>>>; N] = unsafe {
            MaybeUninit::uninit().assume_init()
        };
        for i in 0..N {
            let (s,r) 
            = unbounded::<Job<P>>();
            uninit_job_senders[i].write(s);
            uninit_job_receivers[i].write(r);
        }

        let job_senders: [Sender<Job<P>>; N] = unsafe {
            std::mem::transmute(uninit_job_senders)
        };
        let job_receivers: [Receiver<Job<P>>; N] = unsafe {
            std::mem::transmute(uninit_job_receivers)
        };

        let mut uninit_threads: [MaybeUninit<JoinHandle<()>>; N] = unsafe {
            MaybeUninit::uninit().assume_init()
        };

        for i in 0..N {
            uninit_threads[i].write(Self::spawn_job_thread(job_receivers[i]));
        }

        let threads: [JoinHandle<()>; N] = unsafe {
            std::mem::transmute(uninit_threads)
        };

        let compass = 0usize;

        (Self { 
            shelf: ref_shelf,
            batch,
            request_receivers,
            status,
            batch_size,
            completion_receivers,
            completion_senders,
            threads,
            job_senders,
            compass,
        }, request_senders)
    }

    fn spawn_job_thread(job_receiver: Receiver<Job<P>>) -> JoinHandle<()> {
        thread::spawn(move || {
            while let Ok((
                mini_batch,
                module_ref,
                module_id,
                senders,
                completion_sender
            )) = job_receiver.recv() {

                if let Err(e) = do_inference_job(mini_batch, module_ref, senders, module_id, completion_sender) {
                    eprintln!("Inference job failed: {:?}", e);
                }

            }
        })
    }

    pub fn n_group(&self) -> usize {
        self.batch.len()
    }

    // Look for the available module in the given group solely by looking at the status
    fn get_available_module(&self, group_id: usize) -> Option<usize> {
        let group_status = &self.status[group_id];
        group_status.iter().position(|x| *x == ModuleStatus::Free)
    }

    fn increment_compass(&mut self) {
        self.compass = (self.compass + 1) % N;
    }

    // look for the available module in the given group. When none is available, wait for job completion.
    // side note: Ideally we should spend as little time as possible here (get_available_module should immediately return Some())
    // maybe todo: add some dbg logic that counts how much time we spend inside the loop
    fn wait_for_available_module(&mut self, group_id: usize) -> Result<usize> {

        if group_id >= self.status.len() {
            return Err(eyre!("group id out of bounds"));
        }

        if let Some(module_id) = self.get_available_module(group_id) {
            return Ok(module_id);
        }

        loop {
            match self.completion_receivers[group_id].recv() {
                Ok(completed_module_id) => {

                    self.status[group_id][completed_module_id] = ModuleStatus::Free;

                    return Ok(completed_module_id)
                }
                Err(e) => {
                    return Err(eyre!("Failed to receive inference job completion message for group {}: {:?}", group_id, e));
                }
            }
        }
    }

    fn launch_inference_job(
        &mut self,
        group_id: usize,
        module_id: usize,
        mini_batch: Tensor,
        senders: Vec<Arc<Sender<Evaluation>>>) -> Result<()> {

            let module_ref = self.shelf.table[group_id][module_id].0.clone();
            let completion_sender = self.completion_senders[group_id].clone();

            self.job_senders[self.compass].send((mini_batch, module_ref, module_id, senders, completion_sender)).map_err(|e| {
                eyre!("inference job was not sent: {:?}", e)
            })?;

            self.status[group_id][module_id] = ModuleStatus::Occupied;

            Ok(())
    }


    fn process_batch(&mut self, group_id: usize) -> Result<()>{

        if group_id >= self.shelf.table.len() {
            return Err(eyre!("group id out of bounds"));
        }

        let mini_batch_size = std::cmp::min(self.batch_size, self.batch[group_id].requests.len());
        if mini_batch_size == 0{
            return Ok(());
        }

        let drain = self.batch[group_id].requests.drain(0..mini_batch_size);
        let (queries, senders): (Vec<_>, Vec<_>)= drain.map(|request| {
            (request.query, request.sender)
        }).unzip();
        let mini_batch = Tensor::stack(queries.as_slice(), 0);

        let module_id = self.wait_for_available_module(group_id)?;

        self.launch_inference_job(group_id, module_id, mini_batch, senders)?;

        // change the destination for the next job
        self.increment_compass();

        Ok(())
    }

    pub fn run(&mut self) -> Result<()> {


        loop {
            for (group_id, request_receiver) in self.request_receivers.iter().enumerate() {

                if let Ok(request) = request_receiver.recv_timeout(DIRECTOR_RCV_TIMEOUT)
                {
                    self.batch[group_id].push_request(request);

                    if self.batch[group_id].requests.len() >= self.batch_size {

                        self.process_batch(group_id)?;
                    }

                }
                else {
                    self.process_batch(group_id)?;
                }
            }
        }
        Ok(())
    }
}

fn do_inference_job<P: PVModel>(
    mini_batch: Tensor,
    module_ref: &P,
    senders: Vec<Arc<Sender<Evaluation>>>,
    module_id: usize,
    completion_sender: Sender<usize>)
    -> Result<()> {

    let module = module_ref;
    let device = module.device();
    let input = mini_batch.to_device(device);
    let evaluation = module.evaluate_t(&input, false);
    let cpu_evaluation = evaluation.0.to(Device::Cpu).split(1, 0)
        .into_iter()
        .zip(evaluation.1.to(Device::Cpu).split(1, 0)
    );

    for (sender, evaluation) in senders.iter().zip(cpu_evaluation) {
        sender.send(evaluation)
        .map_err(|e| eyre!("Failed to send back inference result: {:?}", e))?;
    }

    completion_sender.send(module_id)
    .map_err(|e| eyre!("Failed to send inference job completion message: {:?}", e))?;

    Ok(())
}


// play the game for the specified amount using the specified model, create the replay buffer
pub fn self_play_new<P: PVModel, const N: usize>(
    manager: InferenceManager<'_, P, N>,
    request_sender: Sender<Request>,
    num_games: usize,
    mcts_config: &MCTSConfig,
    replay_buffer: &ReplayBuffer,
) -> Result<()>{

    let request_sender = Arc::new(request_sender);
    let model_id = 0;

    let directer_channel = thread::spawn( move || {
        directer.run()
    });


    (0..num_games).into_par_iter().for_each_init(
        || {
            let (sender, receiver) = unbounded::<Evaluation>();
            let sender = Arc::new(sender);
            let game = Game::default();
            let mut mcts_tree = MCTSTree::generate(game, *mcts_config);
            let actor = Actor::new_with_model_id(request_sender.clone(), model_id);
            (sender, receiver, mcts_tree, actor)
        }, 
    |(
            vp_sender,
            vp_receiver,
            mcts_tree,
            actor
        ), worker_idx| {

            let mut episode: Episode = Episode::new();

            while !mcts_tree.root_is_terminal() {
                let (game,_, posterior) = mcts_tree.get_policy_and_update(actor).unwrap();
                episode.append_wo_reward(&game, posterior);
            }
            let winner = mcts_tree.get_winner();
            match winner {
                Side::Att => episode.give_reward(1i64),
                Side::Def => episode.give_reward(-1i64),
            };
            replay_buffer.append_from_episode(&mut episode);
    });

    directer_channel.join().unwrap();

    Ok(())
}

// This struct collects the inference request during self-play and tournament, and batch them appropriately.
// It is intended to be set up per machine
pub struct InferenceManager<'a, P: PVModel> {

    shelf: &'a mut ModuleShelf<P, P>,
    batch: Vec<NewBatch>,
    request_receivers: Vec<Receiver<Request>>,
    status: Vec<Vec<ModuleStatus>>,
    batch_size: usize,
}

impl<'a, P: PVModel> InferenceManager<'a, P> {

    pub fn new(ref_shelf: &'a mut ModuleShelf<P, P>, batch_size: usize) -> (Self, Vec<Sender<Request>>) {

        let len = ref_shelf.table.len();
        let batch = (0..len).map(|_| NewBatch::new()).collect::<Vec<_>>();
        let status = ref_shelf.table.iter().map(|x| {
            vec![ModuleStatus::default(); x.len()]
        }).collect::<Vec<Vec<ModuleStatus>>>();

        let (request_senders, request_receivers): (Vec<_>, Vec<_>) = (0..len)
            .map(|_| { unbounded::<Request>() })
            .unzip();

        (Self { 
            shelf: ref_shelf,
            batch,
            request_receivers,
            status,
            batch_size,
        }, request_senders)
    }

    pub fn n_group(&self) -> usize {
        self.batch.len()
    }

    // Since tch Tensor does not implement Send + Sync, we utilize scoped threads to access module references.
    pub fn run(&mut self) -> Result<()> {

        let job_queue = Arc::new(StdMutex::new(VecDeque::<Job>::new()));
        let module_status = Arc::new(StdMutex::new(self.status.clone()));

        let wrapped_table: Vec<Vec<Arc<StdMutex<&P>>>> = self.shelf.table
            .iter()
            .map(|group| {
                group.iter()
                    .map(|(module, _)| Arc::new(StdMutex::new(module)))
                    .collect()
            })
            .collect();

        thread::scope(|s| {
            
            let job_queue_clone = job_queue.clone();
            
            s.spawn(move || {
                loop{

                    for (group_id, request_receiver) in self.request_receivers.iter().enumerate() {

                        if let Ok(request) = request_receiver.try_recv() {
                            &mut self.batch[group_id].push_request(request);

                            if self.batch[group_id].requests.len() > self.batch_size {

                                let mini_batch_size = std::cmp::min(self.batch_size, self.batch[group_id].requests.len());
                                if mini_batch_size == 0{
                                    break;
                                }

                                let drain = self.batch[group_id].requests.drain(0..mini_batch_size);
                                let (queries, senders): (Vec<_>, Vec<_>)= drain.map(|request| {
                                    (request.query, request.sender)
                                }).unzip();

                                let mini_batch = Tensor::stack(queries.as_slice(), 0);

                                job_queue_clone.lock().unwrap().push_back((mini_batch, group_id, senders));
                            }
                        } else {
                            continue;
                        }

                    }
                }
            });

            let job_queue2 = job_queue.clone();
            let module_status2 = module_status.clone();
            let wrapped_table_clone = wrapped_table.clone();

            s.spawn(move || {
                loop {
                    
                    if let Some(job) = job_queue2.lock().unwrap().pop_front() {

                        let (mini_batch, group_id, senders) = job;

                        if let Some(module_id) = {
                            let mut status = module_status2.lock().unwrap();
                            let opt = status[group_id].iter().position(|x| *x == ModuleStatus::Free);
                            if let Some(module_id) = opt {
                                status[group_id][module_id].occupy();
                            }
                            opt
                        } 
                        {

                            let module_ref = wrapped_table_clone[group_id][module_id].clone();
                        
                            let result = do_inference_job_sync(&mini_batch, &module_ref, &senders);
                            module_status2.lock().unwrap()[group_id][module_id].free();

                            if let Err(e) = result {
                                eprintln!("Inference job failed: {:?}", e);
                            }

                        } else {
                            // No free modules, put job back
                            job_queue2.lock().unwrap().push_front(job);
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    } else {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
            });
        });
        Ok(())
    }
}

fn do_inference_job<P: PVModel>(
    mini_batch: &Tensor,
    module_ref: &P,
    senders: &Vec<Arc<Sender<Evaluation>>>,
)   -> Result<()> {

    let module = module_ref;
    let device = module.device();
    let input = mini_batch.to(device);
    let evaluation = module.evaluate_t(&input, false);
    let cpu_evaluation = evaluation.0.to(Device::Cpu).split(1, 0)
        .into_iter()
        .zip(evaluation.1.to(Device::Cpu).split(1, 0).into_iter());

    for (sender, evaluation) in senders.iter().zip(cpu_evaluation) {
        sender.send(evaluation)
        .map_err(|e| eyre!("Failed to send back inference result: {:?}", e))?;
    }

    Ok(())
}

fn do_inference_job_sync<P: PVModel>(
    mini_batch: &Tensor,
    module_ref: &Arc<StdMutex<&P>>,
    senders: &Vec<Arc<Sender<Evaluation>>>,
) -> Result<()> {

    let evaluation = {
        let module = module_ref.lock().unwrap();
        let device = module.device();
        let input = mini_batch.to(device);
        let mut evaluation = module.evaluate_t(&input, false);
        
        // Move to CPU before releasing the lock
        evaluation.0 = evaluation.0.to(Device::Cpu);
        evaluation.1 = evaluation.1.to(Device::Cpu);
        evaluation
    }; // Lock is released here

    let cpu_evaluation = evaluation.0.split(1, 0)
        .into_iter()
        .zip(evaluation.1.split(1, 0).into_iter());

    for (sender, evaluation) in senders.iter().zip(cpu_evaluation) {
        sender.send(evaluation)
        .map_err(|e| eyre!("Failed to send back inference result: {:?}", e))?;
    }

    Ok(())
}
