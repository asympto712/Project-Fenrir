// TODO!
#![allow(dead_code)]
#![allow(unused)]


use bincode::config::{Configuration, LittleEndian, Fixint};
use color_eyre::owo_colors::OwoColorize;
use game::game::{Game, GameState};
use game::board::TaflBoardEleven;
use game::game::get_rep_counter_for_oldest;
use game::game::Side;
use bitboard::Direction;
use bitboard::eleven::{ElevenBoardPositionalEncoding, BoardEleven, MoveOnBoardEleven};
use libc::free;
use rand::distributions::Uniform;

use crate::agent::PosteriorDist;
use crate::replay_buffer;
use crate::utils::IndexPolicy;
#[cfg(feature = "torch")]
use crate::utils::TBoardEleven;
#[cfg(feature = "torch")]
use tch::Tensor;
#[cfg(feature = "torch")]
use tch::{Kind, Device};

use bincode;
use color_eyre::eyre::{eyre, Result};
use rand::prelude::*;

use std::collections::VecDeque;
use std::ops::RangeBounds;
use std::path::Path;
#[cfg(feature = "torch")]
use std::sync::MutexGuard;
use std::sync::{Arc, Mutex};
use std::{fs, usize};
use std::io::{prelude::*, BufReader};

// Temporary definition
// reward is only given at the end of the game and is ALWAYS from the perspective of Attacker.
// That means, reward = +1 if (attacker won) and -1 if (defender won), 0 if (draw)
type Reward = i64;

const REPLAY_BUFFER_ENCODE_CONFIG: bincode::config::Configuration<LittleEndian, Fixint> 
    = bincode::config::standard()
    .with_little_endian()
    .with_fixed_int_encoding()
    .with_no_limit();


#[derive(Debug, Default, Copy, Clone, bincode::Encode, bincode::Decode)]
pub struct ElevenBoardForRB((BoardEleven, BoardEleven, BoardEleven));

impl ElevenBoardForRB {
    pub fn from_tbe(tbe: &TaflBoardEleven) -> Self {
        Self((tbe.bit_att, tbe.bit_def, tbe.bit_king))
    }

    #[cfg(feature = "torch")]
    pub fn to_tboard(&self, options: (Kind, Device)) -> TBoardEleven {
        let att = TBoardEleven::from_bitboard(&self.0.0, options);
        let def = TBoardEleven::from_bitboard(&self.0.1, options);
        let king = TBoardEleven::from_bitboard(&self.0.2, options);
        let ts = Tensor::stack(&[att.get(), def.get(), king.get()], 0);
        TBoardEleven::new(ts)
    }

}

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub enum EpisodeUnit {
    SPR(SPR),
    Sep
}

impl From<SPR> for EpisodeUnit {
    fn from(value: SPR) -> Self {
        EpisodeUnit::SPR(value)
    }
}

impl EpisodeUnit {
    pub fn give_reward(&mut self, reward: Reward) {
        if let Self::SPR(spr) = self {
            spr.give_reward(reward);
        }
    }

    #[cfg(feature = "torch")]
    pub fn to_tboard(&self, options: (Kind, Device)) -> TBoardEleven {
        match self {
            EpisodeUnit::SPR(spr) => spr.board.to_tboard(options),
            EpisodeUnit::Sep => {
                TBoardEleven::new(Tensor::zeros([3, 11, 11], options))
            }
        }
    }
}

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct SPR {
    board: ElevenBoardForRB,
    state: GameState,
    policy: IndexPolicy,
    repetition_counter: u8,
    reward: Reward,
}

impl SPR {
    pub fn reward_uninitialized(game: &Game, posterior: PosteriorDist) -> Self {
        let board = ElevenBoardForRB::from_tbe(&game.board);
        let state = game.state.clone();
        let repetition_counter = get_rep_counter_for_oldest(game.repetition_counter);
        let policy: IndexPolicy = IndexPolicy::from(posterior);
        Self { 
            board,
            state,
            policy,
            repetition_counter,
            reward: 0 
        }
    }

    pub fn reward_initialized(game: &Game, posterior: PosteriorDist, reward: Reward) -> Self {
        let board = ElevenBoardForRB::from_tbe(&game.board);
        let state = game.state.clone();
        let repetition_counter = get_rep_counter_for_oldest(game.repetition_counter);
        let policy: IndexPolicy = IndexPolicy::from(posterior);
        Self { 
            board,
            state,
            policy,
            repetition_counter,
            reward
        }
    }

    pub fn give_reward(&mut self, reward: Reward) {
        self.reward = reward;
    }
}


#[derive(Debug, Default)]
pub struct Episode {
    pub episode: Vec<EpisodeUnit>
}

impl Episode {
    pub fn give_reward(&mut self, reward: Reward) {
        for u in self.episode.iter_mut() {
            u.give_reward(reward);
        }
    }
    pub fn new() -> Self {
        let episode = vec![EpisodeUnit::Sep];
        Self { episode }
    }
    pub fn append_wo_reward(&mut self, game: &Game, posterior: PosteriorDist) {
        let spr = SPR::reward_uninitialized(game, posterior);
        let u = EpisodeUnit::from(spr);
        self.episode.push(u);
    }
    pub fn len(&self) -> usize {
        self.episode.len()
    }
}

#[derive(Debug, Default, bincode::Encode, bincode::Decode)]
pub struct ReplayBuffer {
    pub replay_buffer: Mutex<Vec<EpisodeUnit>>,
    pub capacity: usize,
}
// TODO: define custom default so that memory efficient

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        let replay_buffer = Vec::<EpisodeUnit>::with_capacity(capacity);
        let replay_buffer = Mutex::new(replay_buffer);
        Self { replay_buffer, capacity }
    }

    // append to the replay_buffer from the episode.
    // This enforces the capacity, meaning the oldest episodes would be removed if necessary
    pub fn extend_from_episode(&self, episode: Episode) {

        let mut q = self.replay_buffer.lock().unwrap();
        let dif: i64 = self.capacity as i64 - q.len() as i64 - episode.len() as i64;
        if dif >= 0 {
            q.extend(episode.episode);
        } else {
            let offset_opt 
                = ReplayBuffer::get_next_sep_pos_exceeding_offset(&q, (-1 * dif) as usize);
            if let Some(offset) = offset_opt {
                q.drain(0..offset);
            } else {
                q.drain(..);
            }
            q.extend(episode.episode);
        }
    }

    // append to the replay_buffer, draining the episode.
    // This enforces the capacity, meaning the oldest episodes would be removed if necessary
    pub fn append_from_episode(&self, episode: &mut Episode) {
        let mut q = self.replay_buffer.lock().unwrap();
        let dif: i64 = self.capacity as i64 - q.len() as i64 - episode.len() as i64;
        if dif >= 0 {
            q.append(&mut episode.episode);
        } else {
            let offset_opt 
                = ReplayBuffer::get_next_sep_pos_exceeding_offset(&q, (-1 * dif) as usize);
            if let Some(offset) = offset_opt {
                q.drain(0..offset);
            } else {
                q.drain(..);
            }
            q.append(&mut episode.episode);
        }
    }

    // empty the buffer
    pub fn empty(&self) {
        let mut q = self.replay_buffer.lock().unwrap();
        let _ = q.drain(..);
    }

    pub fn change_capacity(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;
    }

    // enforce the capacity, deleting the oldest episodes from the front if necessary.
    pub fn enforce_capacity(&self) {
        let mut q = self.replay_buffer.lock().unwrap();
        if q.len() > self.capacity {
            let offset_opt = ReplayBuffer::get_next_sep_pos_exceeding_offset(
                &q, 
                q.len() - self.capacity
            );
            if let Some(offset) = offset_opt {
                q.drain(0..offset);
            } else {
                q.drain(..);
            }
        }
    }

    #[cfg(feature = "torch")]
    pub fn get_inference_input(q: &Vec<EpisodeUnit>, idx: usize, options: (Kind, Device)) -> Option<(Tensor, Tensor, Tensor)> {

        if let Some(eu) = q.get(idx) {
            match eu {
                EpisodeUnit::Sep => {
                    None
                }
                EpisodeUnit::SPR(spr) => {
                    use crate::utils::TAction;


                    let policy = TAction::from_index_policy(&spr.policy)
                        .inner()
                        .to_kind(options.0)
                        .to_device(options.1);

                    let reward = Tensor::from_slice(&[spr.reward as f32])
                        .to_kind(options.0)
                        .to_device(options.1);

                    let cur_board = spr.board.to_tboard(options).get();
                    let mut past = true;
                    let mut short_history: [Option<Tensor>; 4] = std::array::from_fn(|i| {

                        let ts = if past {

                            if let Some(eu) = q.get(idx - i - 1) {
                                if let EpisodeUnit::SPR(spr) = eu {
                                    spr.board.to_tboard(options).get()
                                } else {
                                    past = false;
                                    Tensor::zeros([3,11,11], options)
                                }
                            } else {
                                past = false;
                                Tensor::zeros([3,11,11], options)
                            }

                        } else {
                            Tensor::zeros([3,11,11], options)
                        };
                        Some(ts)

                    });

                    let tot_move_count: i64 = spr.state.get_turn_count().into();
                    let tmc: Tensor = Tensor::scalar_tensor(tot_move_count, options);
                    let tmc: Tensor = tmc.broadcast_to([1, 11, 11]);

                    let rc: Tensor = Tensor::scalar_tensor(spr.repetition_counter as i64, options);
                    let rc: Tensor = rc.broadcast_to([1, 11, 11]);

                    let side: i64 = match spr.state.show_side() {
                        Side::Att => 1,
                        Side::Def => 0,
                    };
                    let side: Tensor = Tensor::scalar_tensor(side, options);
                    let side: Tensor = side.broadcast_to([1, 11, 11]);

                    let position = Tensor::cat(&[
                        cur_board,
                        short_history[0].take().unwrap(),
                        short_history[1].take().unwrap(),
                        short_history[2].take().unwrap(),
                        short_history[3].take().unwrap(),
                        tmc,
                        rc,
                        side
                        ], 0);
                    
                    Some((position, policy, reward))
                }
            }
        } else {
            None
        }
    }

    pub fn sample_batch<R: Rng + ?Sized>(&self, batch_size: usize, options: (Kind, Device), rng: &mut R) -> Result<(Tensor, Tensor, Tensor)> {

        let q = self.replay_buffer.lock().unwrap();
        let len = q.len();
        if len < batch_size {
            return Err(eyre!("replay buffer didn't have enough size"));
        }
        let uniform = Uniform::new(0, len);
        let mut start = 0;
        let mut position_v: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut policy_v: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut reward_v: Vec<Tensor> = Vec::with_capacity(batch_size);
        while start < batch_size {
            let idx = uniform.sample(rng);
            if let Some((position, policy, reward)) = Self::get_inference_input(&q, idx, options) {
                position_v.push(position);
                policy_v.push(policy);
                reward_v.push(reward);
                start += 1;
            }
        }

        let positions = Tensor::stack(&position_v, 0);
        let policies = Tensor::stack(&policy_v, 0);
        let rewards = Tensor::stack(&reward_v, 0);

        Ok((positions, policies, rewards))
    } 

    // write a range of episodes into a file
    // Episodes are serialized and has to be decoded. 
    // Each episode is encoded as Vec<SPR>, followed by how many bytes it took to encode it.
    // This will be essential when retrieving data from a file
    pub fn save<R: RangeBounds<usize>, W: Write>(self, writer: &mut W, game_range: R) -> Result<()> {
        let mut v: Vec<SPR> = vec![];
        let mut freed_history = self.replay_buffer.into_inner().map_err(|_| eyre!("Could not acquire lock for replay buffer"))?;
        let mut game_idx = 0;
        let mut pos: usize = 0;
        let tot_len = freed_history.len();

        for eu in freed_history {
            match eu {
                EpisodeUnit::SPR(spr) => {
                    v.push(spr);
                },
                EpisodeUnit::Sep => {
                    if v.len() == 0 {
                        continue;
                    }
                    let v_spr = std::mem::replace(&mut v, vec![]);
                    if game_range.contains(&game_idx) {
                        write_one_episode(v_spr, writer)?;
                    }
                    game_idx += 1;
                }
            }
        }
        if game_range.contains(&game_idx) {
            write_one_episode(v, writer)?;
        }
        Ok(())
    }


    // load the latest n episodes from the specified file.
    // The capacity of the output is set to be usize::MAX
    pub fn load_from_reader<R: Read + Seek>(mut reader: R, n_episodes: usize) -> Result<Self> {

        use std::io::SeekFrom;
        let mut output: Vec<EpisodeUnit> = vec![];
        let mut pos = reader.seek(SeekFrom::End(0))?;
        let mut episode_n_byte_buf: [u8;8] = [0;8];
        let mut episode_count = 0;

        while episode_count < n_episodes && reader.stream_position().is_ok() {
            if pos < 8 {
                break;
            }
            pos -= 8;
            reader.seek_relative(-8)?;
            reader.read_exact(&mut episode_n_byte_buf)?;
            let episode_n_byte: u64 = u64::from_le_bytes(episode_n_byte_buf);

            pos -= episode_n_byte;
            reader.seek_relative(-8 - episode_n_byte as i64)?;
            let mut data_buf = vec![0u8; episode_n_byte as usize];
            reader.read_exact(&mut data_buf)?;

            let (mut v_spr, u) = bincode::decode_from_slice::<Vec<SPR>, Configuration<LittleEndian, Fixint>>(
                data_buf.as_slice(), 
                REPLAY_BUFFER_ENCODE_CONFIG
            )?;
            reader.seek_relative(-1 * u as i64)?;
            output.extend(v_spr.into_iter().map(|spr| EpisodeUnit::SPR(spr)).rev());
            output.push(EpisodeUnit::Sep);

            episode_count += 1;
        }

        output.reverse();

        let replay_buffer = Vec::from(output);
        let replay_buffer = Mutex::new(replay_buffer);
        let decoded = ReplayBuffer { replay_buffer, capacity: usize::MAX};
        Ok(decoded)
    }

    // Get the minimum index of EpisodeUnit::Sep that would exceed the specified offset
    pub fn get_next_sep_pos_exceeding_offset(v: &Vec<EpisodeUnit>, offset: usize) -> Option<usize> {
        let mut count: usize = 0;
        for eu in v.iter(){
            if let EpisodeUnit::Sep = *eu && count > offset{
                return Some(count)
            }
            count += 1;
        }
        None
    }

}

fn write_one_episode<W: Write> (v: Vec<SPR>, w: &mut W) -> Result<()> {
    let n_bytes = bincode::encode_into_std_write(
        v,
        w,
        REPLAY_BUFFER_ENCODE_CONFIG
    )? as u64;
    // At the end of each episode, store how many bytes it took to encode the episode
    w.write_all(&n_bytes.to_le_bytes())?;
    Ok(())
}