#![allow(unused)]
#![allow(dead_code)]
#![cfg(feature = "torch")]

//internal
use game::game::{Game, GameState};
use game::board::{TaflBoardEleven};
use bitboard::eleven::{ElevenBoardPositionalEncoding, MoveOnBoardEleven};
use crate::utils;
use crate::replay_buffer;
use crate::model;
use crate::agent;

//external
use tch::Tensor;
use tch::nn;

// multithreading-related
use crossbeam::channel::{unbounded, Sender, Receiver};
use std::thread;
use std::sync::Arc;
use rayon::{iter::{IntoParallelIterator, ParallelIterator}};

const MAXBATCHSIZE: usize = 1000;

#[derive(Debug)]
struct Batch {
    batch: Vec<Request>,
    capacity: usize,
    send_criteria: usize,
}

impl Batch {
    pub fn new(capacity: usize, send_criteria: usize) -> Self {
        Batch {
            batch: Vec::with_capacity(capacity),
            capacity,
            send_criteria
        }
    }

    pub fn send_requests(&self) {
        todo!()
    }

    pub fn push(&mut self, r: Request) {
        self.batch.push(r);
    }

    pub fn extend(&mut self, vr: Vec<Request>) {
        self.batch.extend(vr);
    }

    // This will make the vr empty
    pub fn append(&mut self, vr: &mut Vec<Request>) {
        self.batch.append(vr);
    }
}


#[derive(Debug)]
struct Request {
    query: Query,
    sender: Arc<Sender<Tensor>>,
}

// This is to denote the mini-batched inference request from a single worker at a time.
// So it should have the size of (N, in_features, 11, 11)
// TODO: Figure out a way of indexing the returned tensor into each individual request (game state encoding)
type Query = Tensor;


// play the game for the specified amount using the specified model, create the replay buffer
pub fn self_play(model: nn::Sequential, num_games: usize, batch_capacity: usize) {

    let (s, r) = unbounded::<Request>();

    // a dedicated thread that oversees the collecting & sending of batch and redistribution of the inference results
    let communicator = thread::spawn( move || {
        // send criteria might need some tinkering
        let mut batch = Batch::new(batch_capacity, (3 * batch_capacity) / 4 );
        loop
        {
            // TODO: Try timeout logic and compare 
            match r.try_recv()  // a non-blocking check of input
            {
                Ok(received) => {
                    batch.push(received);
                    if batch.len() >= batch.send_criteria {
                        process_batch(&batch);
                        batch.clear();
                    }
                },
                Err(_) => {
                    if !batch.is_empty() {
                        process_batch(&batch);
                        batch.clear();
                    }

                    // Try a blocking check of input
                    if let Ok(received) = r.recv() {
                        batch.push(received);
                        continue;
                    } else {
                        // This means the channel is closed for some reason...
                        break;
                    }
                }
            }
        }
    });

    (0..num_games).into_par_iter().for_each_init(
        || {
            let (sender, receiver) = unbounded::<Tensor>();
            let sender = Arc::new(sender);
            (sender, receiver)
        }, 
    |(vp_sender, vp_receiver), worker_idx| {
        play_game(tensor_sender, tensor_receiver, )
    });
}

pub fn make_episode(
    vp_sender: &mut Arc<Sender<Tensor>>,
    vp_receiver: &mut Receiver<Tensor>) -> replay_buffer::Episode {

    let mut game = Game::default();
    while game.state.is_ongoing() {
        todo!()
    }
}