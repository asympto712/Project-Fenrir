use crate::agent::{MCTSConfig, MCTSTree, Oracle, NewActor};
use crate::model::{self, ModelConfig};
use crate::model::PVModel;
use crate::replay_buffer::{Episode, BoardData, GameSPR, ReplayBuffer, SimpleGameSPR, Sampler};
use crate::setup::{self, FenrirConfig};
use crate::schedule::lr_sch_initialize;
use crate::self_play::{self, self_play_new, setup_mcts, InferenceManager, LockedShelf, ModuleShelf, Shelf, Request};
#[cfg(feature = "mpi")]
use crate::setup::ModelSetupConfig;
use crate::train::{self, ModelSyncConfig, NewTrainer, Trainer};
use crate::utils::{ActionTensor, BoardTensor, ModelInput, TAction, TBoard};
use crate::CompConfig;

use bincode::Encode;
use crossbeam::epoch::default_collector;
use game::game::{Game, GameLogic, SimpleGame, Side};
use game::board::TaflBoard;
use bitboard::{BitBoard, MoveOnBoard};

#[cfg(feature = "mpi")]
use mpi::environment::Universe;
#[cfg(feature = "mpi")]
use mpi::Rank;
use mpi::{point_to_point::*, Tag};
use mpi::topology::SimpleCommunicator;
use mpi::traits::Communicator;
use mpi::MpiError;
use mpi::datatype::{DynBufferMut, Equivalence};

use color_eyre::eyre::{eyre, OptionExt, Result};
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rv::traits::Mode;
use tch::nn::ModuleT;
use tch::nn::VarStore;
use tch::vision::image::load;
use tch::{Device, Kind};
use chrono::{Utc, Local};
use rand::*;

use std::ffi::OsString;
use std::iter::FlatMap;
use std::marker::PhantomData;
use std::os::raw::c_int;
use std::io::Cursor;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use crossbeam::channel::{bounded, unbounded};
use crossbeam::channel::{Sender, Receiver};

pub const TAG_HALT: Tag = 100;
pub const TAG_MODEL_REQUEST: Tag = 101;
pub const TAG_MODEL_UPDATE: Tag = 102;
pub const TAG_REPLAY_SEND: Tag = 103;

// TainNode-related constants
pub const SUB_RB_CAP: usize = 10000;
pub const RB_MIN_LEN: usize = 500; // minimum length of the replay buffer to start the training
pub const N_STEP_PER_CYCLE: usize = 100; // number of training steps per cycle (until the next model update)
pub const SYNC_BN_STATS_EVERY: usize = 10; // How often to synchronize the stats of Batch Norm layers during training
pub const DATA_BUF_CAP: usize = 1000;
pub const EP_BUF_CAP: usize = 1000;
pub const HE_BUF_CAP: usize = 1000; /* capacity of buffer to keep the one episode of data inside TrainNode. 
It should be roughly (Maximum length of the game) * (Number of bytes it takes to encode one play data) */

pub const TRAIN_RANK: Rank = 1;
pub const TESTER_RANK: Rank = 0;

pub const N_DUEL: usize = 50; // How many times do the champion and the challenger play to calculate the win% ?

// communication flags to use internally inside each Node
#[derive(Debug, Clone, Copy, PartialEq)]
enum CommFlag { 
    Terminate,
    RbUpdate,
    ModelSend(Rank),
    Play,
 }

// returns (number of elements, how many bytes it takes in total) for a model with the given config
fn numel_for_model(vs: &VarStore) -> (usize, usize) {
    let mut numel: usize = 0;
    for (_name, variable) in vs.variables().iter() {
        numel += variable.numel();
    } 
    let kind = vs.kind();
    let n_bytes = match kind {
        Kind::Float => numel * 4,
        Kind::Double => numel * 8,
        Kind::Half => numel * 2,
        Kind::BFloat16 => numel * 2,
        _ => panic!("Encountered an unexpected Kind")
    };
    (numel, n_bytes)
}

#[cfg(feature = "mpi")]
pub trait Node<S: Shelf> {
    // Call it after mpi is initialized
    fn world() -> SimpleCommunicator {
        SimpleCommunicator::world()
    }
    fn shelf(&self) -> &S;
    fn shelf_mut(&mut self) -> &mut S;
    // Since we are expecting frequent exchange of model variables, and the model can be modestly large,
    // Each Node should own a preallocated Vector and reuse it as the buffer
    fn buffer(&mut self) -> &mut Vec<u8>;
    fn mut_slice(&mut self) -> &mut[u8]{
        &mut self.buffer()[..]
    }
    // Since we are reusing the same buffer repeatedly
    // it is extremely important to make sure they only contain valid data
    fn clear_buffer(&mut self) {
        self.buffer().clear();
    }
    fn request_model(target_rank: Rank) {
        Self::world().process_at_rank(target_rank).send_with_tag(&1u64, TAG_MODEL_REQUEST);
    }
    fn send_model(&mut self, target_rank: Rank) {
        self.clear_buffer();
        let mut buf = self.buffer().drain(..).collect::<Vec<_>>();
        self.shelf_mut().write_to_stream(/*group_id:*/0, &mut buf);
        Self::world().process_at_rank(target_rank).send_with_tag(&buf, TAG_MODEL_UPDATE);
    }
    fn receive_model(&mut self, source_rank: Rank) {
        Self::world().process_at_rank(source_rank).receive_into_with_tag(self.buffer(), TAG_MODEL_UPDATE);
    }
    fn init(config: CompConfig) -> Self;
    fn update_model_from_buffer(&mut self) -> Result<()>;
}

pub struct SelfPlayNode<P: PVModel + Send, D: BoardData> {
    shelf: LockedShelf<P>,
    buffer: Vec<u8>,
    config: CompConfig,
    _marker: PhantomData<D>
}

impl<P: PVModel + Send, D: BoardData> Node<LockedShelf<P>> for SelfPlayNode<P, D> {
    fn buffer(&mut self) -> &mut Vec<u8> {
        &mut self.buffer
    }

    fn shelf(&self) -> &LockedShelf<P> {
        &self.shelf
    }
    fn shelf_mut(&mut self) -> &mut LockedShelf<P> {
        &mut self.shelf
    }
    fn update_model_from_buffer(&mut self) -> Result<()>{
        let buf = std::mem::take(&mut self.buffer);
        let mut cursor = Cursor::new(buf);
        let group = self.shelf.get_group_mut(0).ok_or_eyre("Could not obtain group")?;
        for (_, mutex_vs) in group.iter_mut() {
            cursor.set_position(0);
            let mut unlocked_vs = mutex_vs.lock().map_err(|_| eyre!("Could not obtain mutex lock"))?;
            unlocked_vs.load_from_stream(&mut cursor)?;
        }
        Ok(())
    }
    fn init(config: CompConfig) -> Self {
        assert!(config.setup_config.use_mpi);
        let rank = Some(Self::world().this_process().rank());
        let loaded_modules = setup::setup_w_mpi::<P>(&config.setup_config).unwrap();
        let (table, name_lookup) = config.setup_config.create_lookup_table::<P>(rank, loaded_modules).unwrap();
        assert_eq!(table.len(), 1);
        assert!(table[0].len() > 0);
        let (numel, capacity) = numel_for_model(&table[0][0].1);
        let shelf = ModuleShelf::module_shelf(table, name_lookup);
        let locked_shelf = LockedShelf::convert_from_shelf(shelf);

        // preallocate buffer
        let buffer: Vec<u8> = Vec::with_capacity(capacity);
        Self{
            shelf: locked_shelf,
            buffer,
            config,
            _marker: PhantomData
        }
    }
}

impl<P: PVModel + Send, D: BoardData> SelfPlayNode<P, D>
where
TBoard<D::G>: ModelInput<D::G>,
TAction<<D::G as GameLogic>::B>: ActionTensor,
TaflBoard<<D::G as GameLogic>::B>: std::fmt::Display,
{
    fn update_model_safely(&mut self, buf: Arc<Mutex<Vec<u8>>>) -> Result<()> {
        let mut unlocked = buf.lock().map_err(|_| eyre!("Could not acquire lock"))?;
        let mut cursor = Cursor::new(unlocked.as_mut_slice());
        let group = self.shelf.get_group_mut(1).ok_or_eyre("Could not get the group")?;
        for (_, mutex_vs) in group.iter_mut() {
            cursor.set_position(0);
            let mut unlocked_vs = mutex_vs.lock().map_err(|_| eyre!("Could not acquire lock"))?;
            unlocked_vs.load_from_stream(&mut cursor)?;
        }
        Ok(())
    }

    fn run(&mut self, batch_size: usize, mcts_config: &MCTSConfig, rank_trainer: i32) {

        let mut run: bool = true;
        let mut update_flag = false;

        let buffer = std::mem::take(&mut self.buffer);
        let buffer = Arc::new(Mutex::new(buffer));

        let mut msg_opt: Option<Message> = None;

        while run {

            std::thread::scope(|s| {
                let (manager, request_senders) = InferenceManager::<'_, P, &'_ mut LockedShelf<P>>::new(&mut self.shelf, batch_size);
                assert_eq!(request_senders.len(), 1);
                let mut iter = request_senders.into_iter().take(1);
                // It is imperative that this gets dropped in order for the manager thread to stop
                let request_sender = Arc::new(iter.next().unwrap());

                // Get the (recommended) number of threads
                let thread_count = std::thread::available_parallelism().unwrap().get();
                let pool = rayon::ThreadPoolBuilder::new().num_threads(thread_count).build().unwrap();

                // Channel for halting messages for threads in the pool
                let (tx, rx) = bounded::<bool>(1);
                let rx = Arc::new(rx);

                // manager is responsible for accepting and managing inference requests from any worker thread.
                let manager_handler = s.spawn(move || {
                    // This halts only when all the request sender clones get  dropped.
                    manager.consume_and_run().unwrap();
                });

                // These jobs run implicitly, so we must make sure it actually halts.
                pool.spawn_broadcast({
                    // This block before the move closure only gets called once, not per thread.
                    let rx_cloned = Arc::clone(&rx);
                    let request_sender_cloned = Arc::clone(&request_sender);
                    let rank_trainer = rank_trainer.clone();
                    let mcts_config = mcts_config.clone();

                    move |_| {

                        let rx_cloned = rx_cloned.clone();
                        let request_sender_cloned = request_sender_cloned.clone();

                        // keep using the same buffer for send operation
                        let mut buf: Vec<u8> = Vec::new();

                        loop {

                            if let Ok(_) = rx_cloned.try_recv() {
                                drop(request_sender_cloned); // (strong count of request_sender) -= 1
                                break;
                            }

                            let (
                                mut tree,
                                mut actor
                            ) = setup_mcts::<D::G>(&mcts_config, &request_sender_cloned);

                            let mut episode: Episode<D>= Episode::<D>::new();

                            while !tree.root_is_terminal() {

                                let (game,_action, posterior) = tree.get_policy_and_update::<NewActor>(&actor).unwrap();
                                episode.append_wo_reward(&game, posterior);

                            }
                            let winner = tree.get_winner();

                            match winner {
                                Side::Att => episode.give_reward(1i64),
                                Side::Def => episode.give_reward(-1i64),
                            };

                            episode.save(&mut buf);
                            // This requires MultiThread support from the MPI, so make sure to initialize MPI with Threading::Multiple
                            SimpleCommunicator::world().process_at_rank(rank_trainer).send_with_tag(&buf, TAG_REPLAY_SEND);
                            buf.clear();

                        }
                    }   
                });

                drop(rx);
                
                loop {
                    let opt = Self::world().any_process().immediate_matched_probe();
                    match opt {
                        None => {
                            std::thread::sleep(Duration::from_secs(10));
                        },
                        Some((msg ,status)) => {
                            match status.tag() {
                                TAG_MODEL_UPDATE => {
                                    update_flag = true;
                                    msg_opt = Some(msg);
                                    break;
                                },
                                TAG_HALT => {
                                    run = false;
                                    let mut unlocked_buffer = buffer.lock().unwrap();
                                    let mut buf = DynBufferMut::new(&mut unlocked_buffer);
                                    msg.matched_receive_into(&mut buf);
                                    break;
                                }
                                _ => { panic!("unexpected tag")}
                            }
                        }
                    };
                }

                drop(request_sender);
                // If the main thread reaches here, that means it received a communication,
                // so we must finish the current self-play to prepare for the model update
                loop {
                    if let Err(_) = tx.send(true) {
                        break;
                    }
                }
                // After the loop completes, all the request sender clones should be dropped..
                manager_handler.join().unwrap();
            });

            if update_flag {
                let tmp = Arc::clone(&buffer);
                let mut unlocked_buffer = tmp.lock().unwrap();
                unlocked_buffer.clear();
                let mut buf = DynBufferMut::new(&mut unlocked_buffer);
                let msg = std::mem::take(&mut msg_opt);
                let _ = msg.unwrap().matched_receive_into(&mut buf);
                drop(unlocked_buffer);
                self.update_model_safely(tmp);
            }
            update_flag = false;
        }
    }
}

// MEMO: Call initialize_with_threading(Threading::Multiple) at the start of the main()

struct TrainNode<P: PVModel + Send, D: BoardData + Encode> {
    trainer: NewTrainer<P>,
    replay_buffer: ReplayBuffer<D>,
    buffer: Vec<u8>,
}

impl<P: PVModel + Send, D: BoardData + Encode> Node<ModuleShelf<P, P>> for TrainNode<P, D> {
    fn buffer(&mut self) -> &mut Vec<u8> {
        &mut self.buffer
    }
    fn shelf(&self) -> &ModuleShelf<P, P> {
        &self.trainer.shelf
    }
    fn shelf_mut(&mut self) -> &mut ModuleShelf<P, P> {
        &mut self.trainer.shelf
    }
    fn init(config: CompConfig) -> Self {
        assert!(config.setup_config.use_mpi);
        let rank = Some(Self::world().this_process().rank());
        let loaded_modules = setup::setup_w_mpi::<P>(&config.setup_config).unwrap();
        let (table, name_lookup) = config.setup_config.create_lookup_table::<P>(rank, loaded_modules).unwrap();
        assert_eq!(table.len(), 1);
        assert!(table[0].len() > 0);
        let (numel, capacity) = numel_for_model(&table[0][0].1);
        let shelf = ModuleShelf::module_shelf(table, name_lookup);

        // preallocate buffer
        let buffer: Vec<u8> = Vec::with_capacity(capacity);

        let trainer: NewTrainer<P> = NewTrainer::new(
            shelf,
            tch::nn::sgd(config.fenrir_config.momentum, 0.0f64, config.fenrir_config.weight_decay, false),
            (config.fenrir_config.learning_rate_schedule)(0),
            config.fenrir_config.weight_decay,
            config.fenrir_config.mini_batch_size,
            ModelSyncConfig::new(train::SumOrAve::Ave)
        ).unwrap();

        let replay_buffer = ReplayBuffer::<D>::new(config.fenrir_config.replay_buffer_capacity);
        Self{
            trainer,
            buffer,
            replay_buffer,
        }
    }

    fn update_model_from_buffer(&mut self) -> Result<()> {
        let buf = std::mem::take(&mut self.buffer);
        let mut cursor = Cursor::new(buf);
        let group = self.trainer.shelf.get_group_mut(0).ok_or_eyre("Could not get the group")?;
        for (_, vs) in group.iter_mut() {
            cursor.set_position(0);
            vs.load_from_stream(&mut cursor);
        }
        Ok(())
    }
}

impl<P: PVModel + Send, D: BoardData + Encode> TrainNode<P, D>
where 
ReplayBuffer<D>: Sampler{
    fn run(&mut self) {

        use CommFlag::*;

        // buffer to receive the incoming (encoded) replay data
        let mut data_buffer: Vec<u8> = Vec::with_capacity(DATA_BUF_CAP);

        // secondary replay buffer that keeps the incoming replay data while the main buffer is busy
        let episode_buffer = Episode::<D>{
            episode: Vec::with_capacity(EP_BUF_CAP)
        };
        let mut episode_buffer: Mutex<Episode<D>> = Mutex::new(episode_buffer);
        // helper buffer to decode byte data into episode_buffer
        let mut helper_buffer: Vec<u8> = Vec::with_capacity(HE_BUF_CAP);

        let (tx, rx) = unbounded::<CommFlag>();

        std::thread::scope(|s| {

            // thread to listens for communication
            s.spawn(|| {
                loop {
                    let opt = Self::world().any_process().immediate_matched_probe();
                    match opt {
                        None => {
                            std::thread::sleep(Duration::from_secs(1));
                        },
                        Some((msg, status)) => {
                            match status.tag() {
                                TAG_REPLAY_SEND => {

                                    let mut unlocked_ep_buf = episode_buffer.lock().unwrap();
                                    msg.matched_receive_into(&mut data_buffer);
                                    let cursor = Cursor::new(&data_buffer[..]);
                                    unlocked_ep_buf.extend_from_reader(cursor, &mut helper_buffer);

                                    // If the episode buffer has ample amount of samples to give...
                                    if unlocked_ep_buf.len() > EP_BUF_CAP / 2{
                                        tx.send(CommFlag::RbUpdate).unwrap();
                                    }
                                    drop(unlocked_ep_buf);

                                    data_buffer.clear();
                                    helper_buffer.clear();
                                },
                                TAG_MODEL_REQUEST => {
                                    tx.send(CommFlag::ModelSend(status.source_rank())).unwrap();
                                },
                                TAG_HALT => {
                                    tx.send(CommFlag::Terminate).unwrap();
                                },
                                _ => { panic!("Unexpected tag");}
                            }
                        }
                    }
                }
            });

            s.spawn( || {
                loop {

                    let mut rng = rand::thread_rng();

                    let mut flag_opt: Option<CommFlag> = None;
                    let mut step_count: usize = 0;
                    if self.replay_buffer.len() < RB_MIN_LEN {

                    } else {
                        for _ in 0..N_STEP_PER_CYCLE {
                            if step_count >= N_STEP_PER_CYCLE{
                                flag_opt = Some(ModelSend(TESTER_RANK));
                                break;
                            }
                            if let Ok(flag) = rx.try_recv() {
                                flag_opt = Some(flag);
                                break;
                            }
                            let sync_bn_stats =
                                if step_count % SYNC_BN_STATS_EVERY == 0 { true} else {false};
                            self.trainer.step(&self.replay_buffer, sync_bn_stats, &mut rng).unwrap();
                            step_count += 1;
                        }
                    }

                    if let Some(flag) = flag_opt {
                        match flag {
                            Terminate => {break;},
                            RbUpdate => {
                                let mut unlocked_ep_buf = episode_buffer.lock().unwrap();
                                self.replay_buffer.append_from_episode(&mut unlocked_ep_buf);
                                drop(unlocked_ep_buf);
                            },
                            ModelSend(target) =>  {
                                self.send_model(target);
                            }
                            _ => {
                                panic!("unexpected flag");
                            }
                        }
                    }
                }
            });
        })

    }
}

// This node tests the strength of the new model against the current best one, broadcast it if it 'beats' the current best
struct TestNode<P: PVModel + Send, D: BoardData> {
    shelf: LockedShelf<P>,
    buffer: Vec<u8>,
    config: CompConfig,
    _marker: PhantomData<D>,
}

impl<P: PVModel + Send, D: BoardData> Node<LockedShelf<P>> for TestNode<P, D> {
    fn buffer(&mut self) -> &mut Vec<u8> {
        &mut self.buffer
    }

    fn shelf_mut(&mut self) -> &mut LockedShelf<P> {
        &mut self.shelf
    }

    fn init(config: CompConfig) -> Self {
        assert!(config.setup_config.use_mpi);
        let rank = Some(Self::world().this_process().rank());
        let loaded_modules = setup::setup_w_mpi::<P>(&config.setup_config).unwrap();
        let (mut table, mut name_lookup) = config.setup_config.create_lookup_table::<P>(rank, loaded_modules).unwrap();
        assert!(table.len() > 0); // At the beginning, there might only be one model loaded
        assert!(table[0].len() > 0);

        // create new models in the place of challengers temporarily
        if table.len() == 1 {
            //get the number of cuda available
            let n_cuda = tch::Cuda::device_count();
            let mut available_cuda: Vec<Device>  = (0..n_cuda).map(|i| Device::Cuda(i as usize)).collect();
            // remove the already-occupied device
            for (_module, vs) in table[0].iter() {
                if let Some(idx) = available_cuda.iter().position(|d| *d == vs.device()) {
                    let _ = available_cuda.remove(idx);
                }
            }
            // ideally at this point available_cuda has the equal number of elements as table[0]
            dbg!(available_cuda.len());
            let mut tmp: Vec<(P, VarStore)> = Vec::new();
            for d in available_cuda {
                let vs = VarStore::new(d);
                let module: P  = P::new(&vs.root(), &config.setup_config.module_load_infos[0].config);
                tmp.push((module, vs));
            }
            table.push(tmp);
            name_lookup.push(OsString::from("challenger"));
        }

        let (numel, capacity) = numel_for_model(&table[0][0].1);
        let shelf = ModuleShelf::module_shelf(table, name_lookup);
        let locked_shelf = LockedShelf::convert_from_shelf(shelf);

        // preallocate buffer
        let buffer: Vec<u8> = Vec::with_capacity(capacity);
        Self{
            shelf: locked_shelf,
            buffer,
            config,
            _marker: PhantomData
        }
    }
    fn shelf(&self) -> &LockedShelf<P> {
        &self.shelf
    }
    // This node should have two groups of modules, one that it keeps and updates internally as its current champion,
    // and the other that it updates from the TrainNode to challenge the champion
    fn update_model_from_buffer(&mut self) -> Result<()> {

        let buf = std::mem::take(&mut self.buffer);
        let mut cursor = Cursor::new(buf);

        let group = self.shelf.get_group(1).ok_or_eyre("Tester Node should have 2 groups, but it didn't")?;
        for (_, mutex_vs) in group.iter() {
            cursor.set_position(0);
            let mut unlocked_vs = mutex_vs.lock().map_err(|_| eyre!("Could not acquire lock"))?;
            unlocked_vs.load_from_stream(&mut cursor)?;
        }
        Ok(())
    }
}

impl<P: PVModel + Send, D> TestNode<P, D>
where
D: BoardData,
TBoard<D::G>: ModelInput<D::G>,
TAction<<D::G as GameLogic>::B>: ActionTensor,
TaflBoard<<D::G as GameLogic>::B>: std::fmt::Display,
D: Send + Sync
{

    fn update_model_safely(&mut self, buf: Arc<Mutex<Vec<u8>>>) -> Result<()> {
        let mut unlocked = buf.lock().map_err(|_| eyre!("Could not acquire lock"))?;
        let mut cursor = Cursor::new(unlocked.as_mut_slice());
        let group = self.shelf.get_group_mut(1).ok_or_eyre("Could not get the group")?;
        for (_, mutex_vs) in group.iter_mut() {
            cursor.set_position(0);
            let mut unlocked_vs = mutex_vs.lock().map_err(|_| eyre!("Could not acquire lock"))?;
            unlocked_vs.load_from_stream(&mut cursor)?;
        }
        Ok(())
    }

    fn run(&'static mut self, mcts_config: &MCTSConfig, self_play_ranks: Vec<Rank>) {

        let mut duel: bool = false;
        let mut update: bool = false;
        let mut run: bool = true;

        // Use the same buffer for sending the champion model data
        let send_buf: Vec<u8> = Vec::with_capacity(self.buffer.capacity());
        let buffer = std::mem::replace(&mut self.buffer, vec![]);
        let rec_buf = Arc::new(Mutex::new(buffer));

        let (tx, rx) = unbounded::<CommFlag>(); 

        let rec_buf_clone = rec_buf.clone();
        let handler = std::thread::spawn(move || {
            loop {
                let opt = Self::world().any_process().immediate_matched_probe();
                match opt {
                    None => {
                        std::thread::sleep(Duration::from_secs(10));
                    },
                    Some((msg, status)) => {
                        match status.tag() {
                            TAG_HALT => {
                                tx.send(CommFlag::Terminate);
                                let mut buf = rec_buf_clone.lock().unwrap();
                                let mut buffer_mut = DynBufferMut::new::<u8>(&mut buf);
                                msg.matched_receive_into(&mut buffer_mut);
                                break;
                            },
                            TAG_MODEL_UPDATE => {
                                tx.send(CommFlag::Play);
                                let mut buf = rec_buf_clone.lock().unwrap();
                                buf.clear(); /* It is possible that
                                TrainNode sends new challengers too often for the TestNode to keep up.
                                In that case, it only takes the most recent challenger. 
                                */
                                let mut buffer_mut = DynBufferMut::new::<u8>(&mut buf);
                                msg.matched_receive_into(&mut buffer_mut);
                                drop(buf);
                            },
                            _ => {panic!("unexpected tag")}
                        }
                    }
                }
            }
        });

        loop {

            let mut pending_messages =  rx.try_iter().collect::<Vec<_>>();
            if pending_messages.is_empty() {
                let msg_opt = rx.recv();
                if let Ok(msg) = msg_opt {
                    pending_messages.push(msg);
                } else {
                    // This means the channel is disconnected
                    break;
                }
            }
            if let Some(CommFlag::Terminate) = pending_messages.last() {
                break;
            }

            if pending_messages.contains(&CommFlag::Play) { // This means a new challenger has arrived

                std::thread::scope(|s| {

                    self.update_model_safely(rec_buf.clone()); // Update the challenger from the buffer first

                    let (manager, mut request_senders)
                        = InferenceManager::<'_, P, &'_ mut LockedShelf<P>>::new(&mut self.shelf, self.config.fenrir_config.mini_batch_size);

                    assert_eq!(request_senders.len(), 2);

                    let mut iter = request_senders.into_iter().take(2);
                    let (champion_rs, challenger_rs) = (Arc::new(iter.next().unwrap()), Arc::new(iter.next().unwrap()));

                    let thread_count = std::thread::available_parallelism().unwrap().get();
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(thread_count).build().unwrap();

                    let (tx, rx) = bounded::<bool>(1);
                    let rx = Arc::new(rx);

                    let manager_handler = s.spawn(move || {
                        manager.consume_and_run().unwrap();
                    });

                    let win_rate = pool.install({
                        let rx_cloned = Arc::clone(&rx);
                        let champ_rs_cloned = Arc::clone(&champion_rs);
                        let chal_rs_cloned = Arc::clone(&challenger_rs);
                        
                        move || -> f32 {

                            let champ_rs_cloned = champ_rs_cloned.clone();
                            let chal_rs_cloned = chal_rs_cloned.clone();

                            let win_count = (0..N_DUEL).into_par_iter().filter(|i| {

                                let (mut tree, champion, challenger) = setup_duel::<D::G>(mcts_config, &champ_rs_cloned, &chal_rs_cloned);

                                let players = if i % 2 == 0 {
                                    [champion, challenger]
                                } else {
                                    [challenger, champion]
                                };
                                let mut cycle = players.iter().cycle();

                                while !tree.root_is_terminal() {
                                    let (game, _action, posterior) = tree.get_policy_and_update::<NewActor>(cycle.next().unwrap()).unwrap();
                                }

                                let winner = tree.get_winner();
                                let cond = match winner {
                                    Side::Att => {
                                        if (i % 2 == 0) {false} else {true}
                                    },
                                    Side::Def => {
                                        if (i % 2 == 0) {true} else {false}
                                    }
                                };
                                return cond
                            }).collect::<Vec<_>>().len();

                            (win_count / N_DUEL) as f32
                        }
                    });

                    if win_rate > self.config.fenrir_config.model_update_threshold {
                        update = true;
                    }

                    drop(champion_rs);
                    drop(challenger_rs);

                });

                if update {
                    for rank in self_play_ranks.iter(){
                        self.send_model(*rank);
                    }
                }   
            } 
        }
        handler.join().unwrap();
    }

}

pub fn setup_duel<G: GameLogic>(mcts_config: &MCTSConfig, champion_rs: &Arc<Sender<Request>>, challenger_rs: &Arc<Sender<Request>>)
-> (MCTSTree<G>, NewActor, NewActor)
where
TBoard<G>: ModelInput<G>,
TAction<G::B>: ActionTensor,
TaflBoard<G::B>: std::fmt::Display
{
    let game = <G as Default>::default();
    let mut mcts_tree = MCTSTree::<G>::generate(game, mcts_config.clone());
    let champion = NewActor::new(champion_rs.clone());
    let challenger = NewActor::new(challenger_rs.clone());
    (mcts_tree, champion, challenger)

}
