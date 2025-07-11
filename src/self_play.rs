#![allow(unused)]
#![allow(dead_code)]
#![cfg(feature = "torch")]

//internal
use game::game::{Game, GameState};
use game::board::{TaflBoardEleven};
use bitboard::eleven::{ElevenBoardPositionalEncoding, MoveOnBoardEleven};
use crate::{model, utils};
use crate::replay_buffer;
use crate::model::ModelConfig;
use crate::model::PVNet;
use crate::model::Evaluation;
use crate::agent::{self, Actor, MCTSTree, Temperature};

//external
use tch::Tensor;
use tch::nn;
use tch::Device;
use tch::Kind;
use color_eyre::eyre::{Context, ErrReport, OptionExt, Result};

// multithreading-related
use crossbeam::channel::{unbounded, Sender, Receiver};
use rayon::{iter::{IntoParallelIterator, ParallelIterator}};

// std
use std::path::Path;
use std::thread::{self, current};
use std::sync::Arc;
use std::collections::HashMap;
use std::time::Duration;

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
    internal_length: i64,
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
            internal_length: 0,
            state: BatchState::default()
        }
    }

    pub fn prepare_to_send(&mut self) -> Tensor {

        self.dispatcher.clear();
        // This will leave the self.requests empty
        let requests = std::mem::take(&mut self.requests);
        // let vec_tensor: Vec<Query> = requests.iter().map(|r| *r.query).collect();
        // let ts = Tensor::cat(vec_tensor.as_slice(), 0);
        // for request in requests.into_iter() {
        //     self.dispatcher.push((request.query.size()[0], request.sender));
        // }
        // ts
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

    pub fn send_requests(&self) {
        todo!()
    }

    pub fn push(&mut self, r: Request) {
        let length: i64 = r.query.size()[0].clone();
        self.requests.push(r);
        self.internal_length += length;
    }

    pub fn extend(&mut self, vr: Vec<Request>) {
        let length: i64 = vr.iter().fold(0, |acc, r| 
        acc + r.query.size()[0]);
        self.internal_length += length;
        self.requests.extend(vr);
    }

    // This will make the vr empty
    pub fn append(&mut self, vr: &mut Vec<Request>) {
        let length: i64 = vr.iter().fold(0, |acc, r| 
        acc + r.query.size()[0]);
        self.internal_length += length;
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
pub struct Directer {
    device_allocation: Vec<Device>,
    modules: Vec<PVNet>,
    receiver: Receiver<(Request, usize)>,
    // each element holds one batch for one model.
    pub batches: Vec<Batch>,
    // Stores the indices of the model in "modules" field
    module_ind: Vec<Vec<usize>>,
    // table of model name ("new" for newly created model, file_name for pre-trained model) against indexes in the "batches" vec. 
    model_lookup: HashMap::<String, usize>
}

impl Directer {
    // Creates a new instance of Directer along with the Sender corresponding to its receiver.
    pub fn new(config: &ModelConfig, new_model_count: i64, trained_models: Vec<(i64, String)>, batch_capacity: usize) -> Result<(Self, Sender<(Request, usize)>)> {

        let mut device_allocation: Vec<Device> = vec![];
        let mut modules: Vec<PVNet> = vec![];
        let mut model_lookup = HashMap::<String, usize>::new();
        let mut module_ind: Vec<Vec<usize>> = vec![];

        let mut cuda_left = tch::Cuda::device_count();
        if cuda_left < new_model_count + trained_models.iter().fold(0, |acc, (n,_)| acc + n)
        {
            println!("The number of required models exceeds the number of CUDA devices available. Some models will be loaded onto Cpu");
        }

        let mut model_id: usize = 0;
        let mut cuda_index: usize = 0;
        for (id_minus_one, (n, path)) in trained_models.iter().enumerate()
        {
            let mut module_indices = vec![];
            for _ in 0..*n
            {
                let net = if cuda_left == 0 
                {
                    let mut vs = nn::VarStore::new(Device::Cpu);
                    device_allocation.push(Device::Cpu);
                    vs.load(path)?;
                    PVNet::model(&vs.root(), config)
                } 
                else 
                {
                    let mut vs = nn::VarStore::new(Device::Cuda(cuda_index));
                    device_allocation.push(Device::Cuda(cuda_index));
                    vs.load(path)?;
                    cuda_index += 1;
                    cuda_left -= 1;
                    PVNet::model(&vs.root(), config)
                };
                module_indices.push(modules.len());
                modules.push(net);
            }
            model_lookup.insert(path.clone(), model_id);
            module_ind.push(module_indices);
            model_id += 1;
        }

        let mut new_module_indices = vec![];
        for _ in 0..new_model_count {
            let net = if cuda_left == 0 
            {
                let vs = nn::VarStore::new(Device::Cpu);
                device_allocation.push(Device::Cpu);
                PVNet::model(&vs.root(), config)
            } else 
            {
                let vs = nn::VarStore::new(Device::Cuda(cuda_index));
                device_allocation.push(Device::Cuda(cuda_index));
                cuda_index += 1;
                cuda_left -= 1;
                PVNet::model(&vs.root(), config)
            };
            new_module_indices.push(modules.len());
            modules.push(net);
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
                let mut evaluation = self.modules[module_id].infer(&xs_allocated);
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

            let logits_splits = infer_results[device_idx].0
                .split_with_sizes(&device_split_sizes, 0);
            let values_splits = infer_results[device_idx].1
                .split_with_sizes(&device_split_sizes, 0);

            for ((logits, values), (_, sender)) in logits_splits
                .into_iter()
                .zip(values_splits)
                .zip(dispatcher_slice.iter())
            {
                sender.send((logits, values))
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
pub fn self_play(model_opt: Option<String>, num_games: usize, batch_capacity: usize, config: &ModelConfig) -> Result<()>{

    let (mut directer, sender) = match model_opt {
        Some(ref model_name) => Directer::new(config, 0, vec![(1, model_name.clone())], batch_capacity),
        None => Directer::new(config, 1, vec![], batch_capacity)
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
            let mut mcts_tree = MCTSTree::generate(0.3, 0.03, 0.25, Temperature::Temp(1.0));
            let actor = Actor::new_with_model_id(request_sender.clone(), model_id);
            (sender, receiver, mcts_tree, actor)
        }, 
    |(
            vp_sender,
            vp_receiver,
            mcts_tree,
            actor
        ), worker_idx| {
        while !mcts_tree.root_is_terminal() {
            let posterior = mcts_tree.get_policy_and_update(actor, 100, true).unwrap();
        }
    });

    directer_channel.join().unwrap();

    Ok(())
}

pub fn make_episode() {
    todo!()
}