use crate::agent::{MCTSConfig, MCTSTree, Oracle, NewActor};
use crate::duel::duel;
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
use crate::{CompConfig, MpiConfig};

use bincode::Encode;
use chrono::format::{DelayedFormat, StrftimeItems};
use crossbeam::atomic::AtomicConsume;
use crossbeam::epoch::default_collector;
use game::game::{Game, GameLogic, Side, SimpleGame, Victor};
use game::board::TaflBoard;
use bitboard::{BitBoard, MoveOnBoard};

use mpi::collective::SystemOperation;
#[cfg(feature = "mpi")]
use mpi::environment::Universe;
#[cfg(feature = "mpi")]
use mpi::Rank;
use mpi::{point_to_point::*, Tag};
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives, Group, Root};
use mpi::MpiError;
use mpi::datatype::{DynBufferMut, Equivalence};

use color_eyre::eyre::{eyre, OptionExt, Result};
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::BroadcastContext;
use rv::traits::Mode;
use tch::nn::ModuleT;
use tch::nn::VarStore;
use tch::vision::image::load;
use tch::{data, Device, Kind};
use chrono::{Utc, Local};

use std::default;
use std::ffi::OsString;
use std::fs::File;
use std::iter::FlatMap;
use std::marker::PhantomData;
use std::ops::AddAssign;
use std::os::raw::c_int;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crossbeam::channel::{bounded, unbounded, RecvTimeoutError, SendError, TryRecvError, TrySendError};
use crossbeam::channel::{Sender, Receiver};

pub const TAG_HALT: Tag = 100;
pub const TAG_MODEL_REQUEST: Tag = 101;
pub const TAG_MODEL_UPDATE: Tag = 102;
pub const TAG_REPLAY_SEND: Tag = 103;

// TainNode-related constants
pub const RB_MIN_LEN: usize = 2000; // minimum length of the replay buffer to start the training
/* when you change the replay buffer length in the config, make sure to reconsider this value too! 
For example, if the replay buffer length is 1000 and this value is 2000, the training never starts!!  */

pub const SYNC_BN_STATS_EVERY: usize = 10; // How often to synchronize the stats of Batch Norm layers during training
pub const DATA_BUF_CAP: usize = 60000;
pub const EP_BUF_CAP: usize = 2000;
pub const HE_BUF_CAP: usize = 60000; /* capacity of buffer to keep the one episode of data inside TrainNode. 
It should be roughly (Maximum length of the game) * (Number of bytes it takes to encode one play data) */


/* memo: About the termination process. When the test node detects timeout and send Halt Tag to other nodes, 
nodes might still be in the middle of a computing process or, even worse, sending operations. If one node terminates while other nodes are trying or 
is about to try to send some data to it, that sending operation will not finish. In order to prevent that, we need some kind of synchronization between nodes
to ensure that all processes can finish. */

// communication flags to use internally inside each Node
#[derive(Debug, Clone, Copy, PartialEq)]
enum CommFlag { 
    Terminate,
    RbUpdate,
    ModelSend(Rank),
    Play,
 }

// returns (number of elements, how many bytes it takes in total) for a model with the given config
pub fn numel_for_model(vs: &VarStore) -> (usize, usize) {
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
        Self::world().process_at_rank(target_rank).send_with_tag::<[u8; 0]>(&[], TAG_MODEL_REQUEST);
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
        let buffer: Vec<u8> = vec![0; 2 * capacity];
        print_now(&format!("self node at rank {} has been initialized", Self::world().rank()));
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
    fn update_model_safely(&mut self, count: usize) -> Result<()> {

        // let mut unlocked = buf.lock().map_err(|_| eyre!("Could not acquire lock"))?;
        let mut cursor = Cursor::new(&mut self.buffer[..count]);
        let group = self.shelf.get_group_mut(1).ok_or_eyre("Could not get the group")?;
        for (_, mutex_vs) in group.iter_mut() {
            cursor.set_position(0);
            let mut unlocked_vs = mutex_vs.lock().map_err(|_| eyre!("Could not acquire lock"))?;
            unlocked_vs.load_from_stream(&mut cursor)?;
        }
        Ok(())
    }

    pub fn run<A: AsRef<Path> + Send + Sync>(&mut self, mpi_config: &MpiConfig, win_rate_dir: A) {

        let mut run: bool = true;

        let att_win_count: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        let def_win_count: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        let draw_count: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        let game_count: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        
        let world = Self::world();
        let group = world.group();
        let self_play_group = group.include(&mpi_config.self_play);
        let sub_comm = world.split_by_subgroup(&self_play_group).unwrap();

        print_now(&format!("Hello from rank {} in self-play communicator", sub_comm.rank()));

        let mut f = if sub_comm.rank() == 0 {

            std::fs::create_dir_all(&win_rate_dir);
            let mut path = win_rate_dir.as_ref().to_path_buf().join(format!("{}-win-rates", self.config.name));
            let _ = path.set_extension("dat");
            let mut file = File::create(path).unwrap();
            Some(file)
        } else {
            None
        };

        if let Some(ref mut f) = f {
            writeln!(f, "att def draw total");
        }

        // let buffer = std::mem::take(&mut self.buffer);
        // let buffer = Arc::new(Mutex::new(buffer));

        let mut msg_opt: Option<(Message, usize)> = None; // msg and its size (in bytes) to pass to the model update process

        let (mtx, mrx) = bounded::<u8>(1);


        print_now(&format!("self play node at rank {} has started running!", Self::world().rank()));



        let measure = std::thread::spawn({
            let att_win_count_clone = Arc::clone(&att_win_count);
            let def_win_count_clone = Arc::clone(&def_win_count);
            let draw_count_clone = Arc::clone(&draw_count);
            let mut tmp_att_win_count: u64 = 0;
            let mut tmp_def_win_count: u64 = 0;
            let mut tmp_draw_count: u64 = 0;
            let self_play_ranks = mpi_config.self_play.clone();


            let mut all_ready = true; // boolean to determine if the reduce operation can take place.
            // The transition from true to false is irreversible - once even one node bails out, and this becomes false,
            // no more reduce operation will happen.

            move || {

                let world = Self::world();
                let group = world.group();
                let self_play_group = group.include(&self_play_ranks);
                let sub_comm = world.split_by_subgroup(&self_play_group).unwrap();

                loop {
                    match mrx.recv_timeout(Duration::from_secs(20)) {
                        Ok(_) => {

                            // A bail-out mechanism so that the nodes can signal their participation in the reduce operation.
                            // Note that once it signals false, the reduce operation will no longer take place and only the root node will track the statistics
                            if all_ready {
                                let ready: bool = false;
                                sub_comm.all_reduce_into(&ready, &mut all_ready, SystemOperation::logical_and());
                            }
                            break;
                        }
                        Err(e) => {
                            match e {
                                RecvTimeoutError::Disconnected => {
                                    if all_ready {
                                        let ready: bool = false;
                                        sub_comm.all_reduce_into(&ready, &mut all_ready, SystemOperation::logical_and());
                                    }
                                    break;
                                }
                                RecvTimeoutError::Timeout => {
                                    tmp_att_win_count += att_win_count_clone.swap(0, Ordering::Relaxed);
                                    tmp_def_win_count += def_win_count_clone.swap(0, Ordering::Relaxed);
                                    tmp_draw_count += draw_count_clone.swap(0, Ordering::Relaxed);
                                    let sum = tmp_att_win_count + tmp_def_win_count + tmp_draw_count;

                                    // Final check before proceeding to reduce operation
                                    if all_ready {
                                        let ready: bool = false;
                                        sub_comm.all_reduce_into(&ready, &mut all_ready, SystemOperation::logical_and());
                                    }

                                    if all_ready {

                                        if sub_comm.rank() == 0 {
                                            sub_comm.process_at_rank(0)
                                                .reduce_into_root(&tmp_att_win_count.clone(), &mut tmp_att_win_count, SystemOperation::sum());
                                            sub_comm.process_at_rank(0)
                                                .reduce_into_root(&tmp_def_win_count.clone(), &mut tmp_def_win_count, SystemOperation::sum());
                                            sub_comm.process_at_rank(0)
                                                .reduce_into_root(&tmp_draw_count.clone(), &mut tmp_draw_count, SystemOperation::sum());

                                            if let Some(ref mut file) = f && (tmp_att_win_count + tmp_def_win_count + tmp_draw_count) > 100 {
                                                writeln!(file, "{} {} {}", tmp_att_win_count, tmp_def_win_count, tmp_draw_count);
                                            }

                                        } else {
                                            sub_comm.process_at_rank(0)
                                                .reduce_into(&tmp_att_win_count.clone(), SystemOperation::sum());
                                            sub_comm.process_at_rank(0)
                                                .reduce_into(&tmp_def_win_count.clone(), SystemOperation::sum());
                                            sub_comm.process_at_rank(0)
                                                .reduce_into(&tmp_draw_count.clone(), SystemOperation::sum());
                                        }

                                    } else if sub_comm.rank() == 0 && (tmp_att_win_count + tmp_def_win_count + tmp_draw_count) > 100 {

                                        if let Some(ref mut file) = f {
                                            writeln!(file, "{} {} {}", tmp_att_win_count, tmp_def_win_count, tmp_draw_count);
                                        }

                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        while run {

            std::thread::scope(|s| {

                // Channel for halting messages for threads in the pool
                let (tx, rx) = bounded::<u8>(1);
                let rx = Arc::new(rx);

                let (manager, mut request_senders) = InferenceManager::<'_, P, &'_ mut LockedShelf<P>>::new(&mut self.shelf, self.config.fenrir_config.inference_bs);
                assert_eq!(request_senders.len(), 1);
                let request_sender = request_senders.remove(0);
                // It is imperative that this gets dropped in order for the manager thread to stop
                let request_sender = Arc::new(request_sender);

                // Get the (recommended) number of threads
                let thread_count = std::thread::available_parallelism().unwrap().get();
                let pool = rayon::ThreadPoolBuilder::new().num_threads(thread_count).build().unwrap();


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
                    let rank_trainer = mpi_config.train.clone();
                    let mcts_config = self.config.mcts_config.clone();
                    let att_win_count_clone = Arc::clone(&att_win_count);
                    let def_win_count_clone = Arc::clone(&def_win_count);
                    let draw_count_clone = Arc::clone(&draw_count);
                    let game_count_clone = Arc::clone(&game_count);

                    move |_| {

                        let rx_cloned = Arc::clone(&rx_cloned);
                        let request_sender_cloned = Arc::clone(&request_sender_cloned);

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
                            ) = setup_mcts::<D::G>(&mcts_config, request_sender_cloned.clone());

                            let mut episode: Episode<D>= Episode::<D>::new();

                            while !tree.root_is_terminal() {

                                let (game,_action, posterior) = tree.get_policy_and_update::<NewActor>(&actor).unwrap();
                                episode.append_wo_reward(&game, posterior);

                            }
                            let winner = tree.get_winner();

                            game_count_clone.fetch_add(1, Ordering::Relaxed);

                            match winner {
                                Victor::Att => {
                                    episode.give_reward(1i64);
                                    att_win_count_clone.fetch_add(1, Ordering::Relaxed);
                                },
                                Victor::Def => {
                                    episode.give_reward(-1i64);
                                    def_win_count_clone.fetch_add(1, Ordering::Relaxed);
                                },
                                Victor::Draw => {
                                    episode.give_reward(0i64);
                                    draw_count_clone.fetch_add(1, Ordering::Relaxed);
                                }
                            }

                            episode.save(&mut buf);

                            // try catching termination signal here again because send operation is blocking,
                            // and I suspect it blocks even if the target is already terminated.
                            if let Ok(_) = rx_cloned.try_recv() {
                                drop(request_sender_cloned); // (strong count of request_sender) -= 1
                                break;
                            }

                            mpi::request::scope(|scope| {
                                let req = Self::world().process_at_rank(rank_trainer).immediate_send_with_tag(scope, &buf, TAG_REPLAY_SEND);
                                match req.test() {
                                    Ok(_) => (),
                                    Err(rreq) => {
                                        std::thread::sleep(Duration::from_secs(10));
                                        if let Err(rrreq) = rreq.test() {
                                            print_now("Could not send episode from self play node, cancelling send operation..");
                                            rrreq.cancel();
                                            rrreq.wait();
                                            print_now("Cancellation complete");
                                        }
                                    }
                                }
                            });
                            // // This requires MultiThread support from the MPI, so make sure to initialize MPI with Threading::Multiple
                            // SimpleCommunicator::world().process_at_rank(rank_trainer).send_with_tag(&buf, TAG_REPLAY_SEND);

                            #[cfg(feature = "verbose_lvl2")]
                            print_now("an episode was sent from self play node");

                            buf.clear();

                        }


                        #[cfg(feature = "verbose_lvl1")]
                        println!("self play thread finished");

                    }   

                });

                loop {
                    let opt = Self::world().any_process().immediate_matched_probe();
                    match opt {
                        None => {
                            std::thread::sleep(Duration::from_secs(10));
                        },
                        Some((msg ,status)) => {
                            match status.tag() {
                                TAG_MODEL_UPDATE => {
                                    print_now(&format!("rank {} received model update message from rank {}", Self::world().rank(), status.source_rank()));
                                    let size = status.count(<u8 as Equivalence>::equivalent_datatype()) as usize;
                                    msg_opt = Some((msg, size));
                                    break;
                                },
                                TAG_HALT => { 

                                /*Received halt tag from test node. that means there is no more incoming comm to this rank,
                                only there might be (upcoming) sending operation FROM this node (to the train node)*/

                                    print_now(&format!("self play node at rank {} received halt message", Self::world().rank()));
                                    run = false;
                                    // HALT messages send empty array [u8;0], receive into empty array
                                    let mut empty_data: [u8; 4] = [0; 4];
                                    msg.matched_receive_into(&mut empty_data);
                                    break;
                                }
                                _ => { panic!("unexpected tag")}
                            }
                        }
                    };
                }

                // dbg!(Arc::strong_count(&request_sender));
                drop(request_sender);
                // If the main thread reaches here, that means it received a communication,
                // so we must finish the current self-play to prepare for the model update
                let count = Arc::strong_count(&rx);
                drop(rx);
                // dbg!(count);
                loop {
                    if let Err(e) = tx.send(1) {
                        // dbg!(e);
                        break;
                    }
                }

                // After the loop completes, all the request sender clones should be dropped..
                manager_handler.join().unwrap();
                #[cfg(feature = "verbose_lvl1")]
                print_now("inference manager was shut down correctly");

                

                print_now(&format!("At rank {}, total of {} games were played so far", Self::world().rank(), game_count.load(Ordering::Relaxed)));
            });


            if let Some((msg, size)) = msg_opt {
                // let tmp = Arc::clone(&buffer);
                // let mut unlocked_buffer = tmp.lock().unwrap();
                // unlocked_buffer.clear();
                // let mut buf = DynBufferMut::new(&mut unlocked_buffer);
                // let msg = std::mem::take(&mut msg_opt);

                if size > self.buffer.len() {
                    print_now(
                        &format!("msg size{} exceeding that of buffer{} detected at rank {}, adjusting the buffer length..",
                        size,
                        self.buffer().len(),
                        Self::world().rank()
                    ));
                    self.buffer.resize(size, 0);
                }

                let status = msg.matched_receive_into(&mut self.buffer);
                let count = status.count(<u8 as Equivalence>::equivalent_datatype()) as usize;
                // drop(unlocked_buffer);
                self.update_model_safely(count);

                print_now("model update completed");
                msg_opt = None;
            }

        }

        // let world = Self::world();
        // let group = world.group();
        // let self_play_group = group.include(&mpi_config.self_play);
        // let sub_comm = world.split_by_subgroup(&self_play_group).unwrap();
        mtx.send(1).unwrap();
        measure.join().unwrap();

        /* synchronize barrier only amongst self-play nodes */
        sub_comm.barrier();
        
        /* self node has finished its run, so now it can safely send halt tag to train node. 
        But only one of the self play nodes should send this message */
        if sub_comm.rank() == 0 {
            Self::world().process_at_rank(mpi_config.train).send_with_tag::<[u8;0]>(&[], TAG_HALT);
        }

        print_now(&format!("At rank {}, total of {} games were played", Self::world().rank(), game_count.load(Ordering::Relaxed)));
        print_now(&format!("Self play node at rank {} has finished all of its run process", Self::world().rank()));
    }
}

// MEMO: Call initialize_with_threading(Threading::Multiple) at the start of the main()

pub struct TrainNode<P: PVModel + Send, D: BoardData + Encode> {
    trainer: NewTrainer<P, ModuleShelf<P, P>>,
    replay_buffer: ReplayBuffer<D>,
    buffer: Vec<u8>,
    config: CompConfig,
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
        let buffer: Vec<u8> = vec![0; 2 * capacity];

        let trainer: NewTrainer<P, ModuleShelf<P, P>> = NewTrainer::new(
            shelf,
            tch::nn::sgd(config.fenrir_config.momentum, 0.0f64, config.fenrir_config.weight_decay, false),
            (config.fenrir_config.learning_rate_schedule)(0),
            config.fenrir_config.weight_decay,
            config.fenrir_config.train_bs,
            ModelSyncConfig::new(train::SumOrAve::Ave)
        ).unwrap();

        let replay_buffer = ReplayBuffer::<D>::new(config.fenrir_config.replay_buffer_capacity);

        print_now("train node has been initialized");

        Self{
            trainer,
            buffer,
            replay_buffer,
            config,
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
    pub fn run<A: AsRef<Path> + Send + Sync>(&mut self, mpi_config: &MpiConfig, data_store_dir: A) {

        use CommFlag::*;

        // buffer to receive the incoming (encoded) replay data
        let mut data_buffer: Vec<u8> = vec![0; DATA_BUF_CAP];

        // secondary replay buffer that keeps the incoming replay data while the main buffer is busy
        let episode_buffer = Episode::<D>{
            episode: Vec::with_capacity(EP_BUF_CAP)
        };
        let mut episode_buffer: Mutex<Episode<D>> = Mutex::new(episode_buffer);
        // helper buffer to decode byte data into episode_buffer
        let mut helper_buffer: Vec<u8> = vec![0; HE_BUF_CAP];

        print_now(&format!("train node at rank {} has started running!", Self::world().rank()));

        let (tx, rx) = unbounded::<CommFlag>();

        std::thread::scope(|s| {

            // thread to listens for communication
            let comm_handle = s.spawn(|| {

                loop {

                    let opt = Self::world().any_process().immediate_matched_probe();
                    match opt {
                        None => {
                            std::thread::sleep(Duration::from_secs(1));
                        },
                        Some((msg, status)) => {
                            match status.tag() {
                                TAG_REPLAY_SEND => {

                                    #[cfg(feature = "verbose_lvl2")]
                                    {
                                        let byte = <u8 as Equivalence>::equivalent_datatype();
                                        println!("byte count in one episode: {}", status.count(byte));
                                    }

                                    let mut unlocked_ep_buf = episode_buffer.lock().unwrap();
                                    let size = status.count(<u8 as Equivalence>::equivalent_datatype()) as usize;
                                    if size > data_buffer.len() {

                                        print_now(
                                            &format!("episode size{} exceeding that of buffer{} detected at rank {}, adjusting the buffer length..",
                                            size,
                                            data_buffer.len(),
                                            Self::world().rank()
                                        ));
                                        data_buffer.resize(size, 0);

                                    }
                                    msg.matched_receive_into(&mut data_buffer);
                                    let cursor = Cursor::new(&data_buffer[..size]);
                                    unlocked_ep_buf.extend_from_reader(cursor, &mut helper_buffer);

                                    #[cfg(feature = "verbose_lvl2")]
                                    print_now(&format!("episode buffer length {}/{}", unlocked_ep_buf.len(), EP_BUF_CAP));

                                    // If the episode buffer has ample amount of samples to give...
                                    if unlocked_ep_buf.len() > EP_BUF_CAP / 2{
                                        tx.send(CommFlag::RbUpdate).unwrap();
                                    }
                                    drop(unlocked_ep_buf);

                                    helper_buffer.clear();
                                },
                                TAG_MODEL_REQUEST => {  // In practice this should not happen

                                    let mut dyn_buffer = DynBufferMut::new(&mut data_buffer);
                                    msg.matched_receive_into(&mut dyn_buffer);  //the message should be empty

                                    print_now("train node has received the model request tag");

                                    tx.send(CommFlag::ModelSend(status.source_rank())).unwrap();
                                },
                                TAG_HALT => { /* Received halt tag from self nodes, that means there is no more comm coming from self play nodes.
                                    Meaning it is safe to terminate once it finishes whatever it is doing right now. */

                                    // HALT messages send empty array [u8;0], receive into empty array
                                    let mut empty_data: [u8; 4] = [0; 4];
                                    msg.matched_receive_into(&mut empty_data);

                                    print_now("train node has received the halt tag");

                                    tx.send(CommFlag::Terminate).unwrap();
                                    break;
                                },
                                _ => { panic!("Unexpected tag");}
                            }
                        }
                    }
                }
                // dbg!();
            });


            let train_handle = s.spawn( || {

                std::fs::create_dir_all(&data_store_dir);
                let mut path: PathBuf = data_store_dir.as_ref().to_path_buf();
                path = path.join(format!("{}-loss-data", self.config.name));
                let _ = path.set_extension("dat");
                let mut file = File::create(path).unwrap();
                let mut step_count: usize = 0;

                let n_step_per_cycle = self.config.fenrir_config.n_training_step_per_cycle;
                let test_rank = mpi_config.test;

                let mut run = true;
                let mut model_send: Option<Rank> = None;

                while run {

                    let mut rng = rand::thread_rng();

                    if self.replay_buffer.len() < RB_MIN_LEN {  // cannot start training yet
                        #[cfg(feature = "verbose_lvl1")]
                        print_now(&format!("cannot start training yet.. Replay Buffer needs to reach {} ({})", RB_MIN_LEN, self.replay_buffer.len()));

                        std::thread::sleep(Duration::from_secs(10));

                    } else {

                        #[cfg(feature = "verbose_lvl1")]
                        print_now(&format!("training: current step {}", self.trainer.step_count));

                        let new_lr = (&self.config.fenrir_config.learning_rate_schedule)(self.trainer.step_count);
                        self.trainer.update_lr(new_lr);

                        for _ in 0..n_step_per_cycle {
                            if step_count >= n_step_per_cycle {

                                model_send = Some(test_rank);
                                step_count = 0;  // reset step count
                                break;
                            }

                            let sync_bn_stats =
                                if step_count % SYNC_BN_STATS_EVERY == 0 { true} else {false};
                            self.trainer.step(&self.replay_buffer, sync_bn_stats, &mut rng).unwrap();
                            step_count += 1;
                        }
                        self.trainer.flush_loss_record(&mut file);
                    }

                    if let Some(target) = model_send {
                        // dbg!();
                        self.send_model(target);
                        print_now(&format!("train node sent model to rank {}", target));
                        model_send = None;
                    }

                    loop {
                        match rx.try_recv() {
                            Ok(flag) => {
                                match flag {
                                    Terminate => {
                                        // dbg!();
                                        self.trainer.write_loss_record(&mut file);
                                        run = false;
                                    },
                                    RbUpdate => {

                                        // dbg!();
                                        let mut unlocked_ep_buf = episode_buffer.lock().unwrap();
                                        self.replay_buffer.append_from_episode(&mut unlocked_ep_buf);
                                        drop(unlocked_ep_buf);

                                        #[cfg(feature = "verbose_lvl1")]
                                        print_now(&format!("updated replay_buffer. Current size: {}/{}", self.replay_buffer.len(), self.replay_buffer.capacity));

                                    },
                                    ModelSend(target) =>  {
                                        // dbg!();
                                        self.send_model(target);
                                        print_now(&format!("train node sent model to rank {}", target));
                                    }
                                    _ => {
                                        panic!("unexpected flag");
                                    }
                                }
                            }
                            Err(e) => match e {
                                TryRecvError::Disconnected => {
                                    // dbg!();
                                    run = false;
                                    break;
                                }
                                TryRecvError::Empty => {
                                    // dbg!();
                                    break;
                                }
                            }
                        }
                    }
                }
            });

            comm_handle.join().unwrap();
            // dbg!();
            train_handle.join().unwrap();
            // dbg!();
            print_now("train node has been terminated correctly");
            Self::world().process_at_rank(mpi_config.test).send_with_tag::<[u8;0]>(&[], TAG_HALT);
        });

    }
}

// This node tests the strength of the new model against the current best one, broadcast it if it 'beats' the current best
pub struct TestNode<P: PVModel + Send, D: BoardData> {
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
            // dbg!(available_cuda.len());
            let mut tmp: Vec<(P, VarStore)> = Vec::new();

            // In this case store the new model on Cpu
            if available_cuda.is_empty() {
                let vs = VarStore::new(Device::Cpu);
                let module: P  = P::new(&vs.root(), &config.setup_config.module_load_infos[0].config);
                tmp.push((module, vs));
            } else {

                for d in available_cuda {
                    let vs = VarStore::new(d);
                    let module: P  = P::new(&vs.root(), &config.setup_config.module_load_infos[0].config);
                    tmp.push((module, vs));
                }
            }
            table.push(tmp);
            name_lookup.push(OsString::from("challenger"));

            // dbg!(&name_lookup);
        }

        let (numel, capacity) = numel_for_model(&table[0][0].1);
        let shelf = ModuleShelf::module_shelf(table, name_lookup);
        let locked_shelf = LockedShelf::convert_from_shelf(shelf);

        // preallocate buffer
        let buffer: Vec<u8> = vec![0; 2 * capacity];

        println!("test node has been initialized!");

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
D: Send + Sync,
<<D::G as GameLogic>::B as BitBoard>::Movement: PartialEq,
{

    fn update_model_safely(&mut self, buf: Arc<Mutex<(Vec<u8>, usize)>>) -> Result<()> {
        let mut unlocked = buf.lock().map_err(|_| eyre!("Could not acquire lock"))?;
        let size = unlocked.1;
        let mut cursor = Cursor::new(&mut unlocked.0[..size]);
        let group = self.shelf.get_group_mut(1).ok_or_eyre("Could not get the group")?;
        for (_, mutex_vs) in group.iter_mut() {
            cursor.set_position(0);
            let mut unlocked_vs = mutex_vs.lock().map_err(|_| eyre!("Could not acquire lock"))?;
            unlocked_vs.load_from_stream(&mut cursor)?;
        }
        Ok(())
    }

    pub fn run<A: AsRef<Path> + Send>(&mut self, mpi_config: &MpiConfig, evaluation_mcts_config: &MCTSConfig, model_store_dir: A) {

        let mut update: bool = false;
        let start = Instant::now();
        let time_limit = Duration::from_secs_f32(3600.0 * self.config.fenrir_config.run_time_hr);

        // Use the same buffer for sending the champion model data
        let send_buf: Vec<u8> = Vec::with_capacity(self.buffer.capacity());
        let buffer = std::mem::replace(&mut self.buffer, vec![]);
        let module_size: usize = 0;
        let cha_buf = Arc::new(Mutex::new((buffer, module_size)));

        let (tx, rx) = unbounded::<CommFlag>(); 
        let (finish_tx, finish_rx) = bounded(0);
        let mpi_config_clone = mpi_config.clone();

        print_now(&format!("test node at rank {} has started running!", Self::world().rank()));

        let cha_buf_clone = cha_buf.clone();
        let run_time = self.config.fenrir_config.run_time_hr;
        let handler = std::thread::spawn(move || {
            loop {

                let duration = start.elapsed();
                if duration > time_limit {

                    // From the second time onward, rx will have been dropped, so this send will block unless if we handle disconnected error explicitly.
                    if let Ok(_) = tx.send(CommFlag::Terminate){

                        finish_rx.recv().unwrap(); /* This blocks until the duel ended, and there is no longer a possibility of communication happening 
                        from test node to self-play node */

                        for i in mpi_config_clone.self_play.iter() {
                            Self::world().process_at_rank(*i).send_with_tag::<[u8;0]>(&[], TAG_HALT);
                        }
                        // dbg!();

                    }

                } 
                let opt = Self::world().any_process().immediate_matched_probe();
                match opt {
                    None => {
                        std::thread::sleep(Duration::from_secs(10));
                    },
                    Some((msg, status)) => {
                        match status.tag() {
                            
                            // message with the halt tag is the very last message this node will ever receive.
                            // So it is safe to break out of loop here, as there will be no more messages coming
                            TAG_HALT => {

                                print_now("test node has received the halt tag");

                                // HALT messages send empty array [u8;0], receive into empty array
                                let mut empty_data: [u8; 4] = [0;4];
                                msg.matched_receive_into(&mut empty_data);
                                break;
                            },
                            TAG_MODEL_UPDATE => {

                                print_now("test node has received the model update tag");

                                match tx.try_send(CommFlag::Play){
                                    Ok(_) => (),
                                    Err(e) => match e {
                                        // In this case the termination message was received by the main thread so 
                                        // this new model will not be tested. We still have to receive the message though.
                                        TrySendError::Disconnected(_) => {
                                            print_now("Halting process has started. This new model will not be tested");
                                        }
                                        // this is an unbounded channel so this should not happen
                                        TrySendError::Full(_) => {
                                            // dbg!();
                                        }
                                    }
                                }
                                let mut buf = cha_buf_clone.lock().unwrap();
                                /* It is possible that
                                TrainNode sends new challengers too often for the TestNode to keep up.
                                In that case, it only takes the most recent challenger. 
                                */
                                let size = status.count(<u8 as Equivalence>::equivalent_datatype()) as usize;
                                if size > buf.0.len() {

                                    print_now(
                                        &format!("message size{} exceeding that of buffer{} detected at rank {}, adjusting the buffer length..",
                                        size,
                                        buf.0.len(),
                                        Self::world().rank()
                                    ));
                                    buf.0.resize(size, 0);

                                }
                                let mut buffer_mut = DynBufferMut::new::<u8>(&mut buf.0);
                                msg.matched_receive_into(&mut buffer_mut);
                                buf.1 = size;
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
                let mut path: PathBuf = model_store_dir.as_ref().to_path_buf();
                std::fs::create_dir_all(&path);
                path = path.join(format!("{}", self.config.name));
                let _ = path.set_extension("pv");
                let mut file = File::create(&path).unwrap();
                self.shelf.write_to_stream(0, &mut file);

                print_now(&format!("current champion model was saved at {:?}", path));

                break;
            }

            if pending_messages.contains(&CommFlag::Play) { // This means a new challenger has arrived

                std::thread::scope(|s| {

                    self.update_model_safely(cha_buf.clone()); // Update the challenger from the buffer first

                    print_now("test node has accepted a new challenger!");

                    let (manager, mut request_senders)
                        = InferenceManager::<'_, P, &'_ mut LockedShelf<P>>::new(&mut self.shelf, self.config.fenrir_config.inference_bs);

                    assert_eq!(request_senders.len(), 2);
                    
                    let challenger_rs = request_senders.remove(1);
                    let champion_rs = request_senders.remove(0);


                    let (win_rate, draw_count) = duel::<P, D>(
                        manager,
                        champion_rs,
                        challenger_rs,
                        self.config.fenrir_config.n_games_per_tournament,
                        evaluation_mcts_config
                    ).unwrap();


                    // dbg!();

                    print_now(&format!("new model test has been completed. win rate against the current best was {}", win_rate));

                    if win_rate > self.config.fenrir_config.model_update_threshold {
                        update = true;
                    }

                });

                if update {
                    for rank in mpi_config.self_play.iter(){
                        self.send_model(*rank);
                    }

                    print_now(&format!("test node has sent model to self play nodes"));
                    update = false;
                }   
            } 
        }

        drop(rx);
        finish_tx.send(()).unwrap();

        handler.join().unwrap();
        print_now("test node has been terminated correctly");
    }

}

pub fn setup_duel<G: GameLogic>(mcts_config: &MCTSConfig, champion_rs: Arc<Sender<Request>>, challenger_rs: Arc<Sender<Request>>)
-> (MCTSTree<G>, NewActor, NewActor)
where
TBoard<G>: ModelInput<G>,
TAction<G::B>: ActionTensor,
TaflBoard<G::B>: std::fmt::Display
{
    let game = <G as Default>::default();
    let mut mcts_tree = MCTSTree::<G>::generate(game, mcts_config.clone());
    let champion = NewActor::new(champion_rs);
    let challenger = NewActor::new(challenger_rs);
    (mcts_tree, champion, challenger)

}

fn write_now(msg: &str, w: &mut impl Write) {
    let datetime = chrono::Local::now();
    let format = datetime.format("%m/%d %H:%M:%S");
    write!(w, "{}   {}", format, msg).unwrap()
}

pub fn print_now(msg: &str) {
    let datetime = chrono::Local::now();
    let format = datetime.format("%m/%d %H:%M:%S");
    println!("{}   {}", format, msg)
}

#[cfg(test)]
mod tests {
    use crate::node::write_now;
    use super::*;

    #[test]
    fn write_now_works() {
        let msg = "This is a test\n";
        write_now(msg, &mut std::io::stdout());
    }
}