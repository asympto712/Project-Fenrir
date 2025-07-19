// TODO!
#![allow(dead_code)]
#![allow(unused)]


use color_eyre::owo_colors::OwoColorize;
use game::game::{Game, GameState};
use game::board::TaflBoardEleven;
use game::game::get_rep_counter_for_oldest;
use game::game::Side;
use bitboard::Direction;
use bitboard::eleven::{ElevenBoardPositionalEncoding, BoardEleven, MoveOnBoardEleven};

use crate::replay_buffer;
#[cfg(feature = "torch")]
use crate::utils::TBoardEleven;
#[cfg(feature = "torch")]
use tch::Tensor;
#[cfg(feature = "torch")]
use tch::{Kind, Device};

use bincode;
use color_eyre::eyre::Result;

use std::collections::VecDeque;
use std::ops::RangeBounds;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{fs, usize};
use std::io::{prelude::*, BufReader};

// Temporary definition
// reward is only given at the end of the game and is ALWAYS from the perspective of Attacker.
// That means, reward = +1 if (attacker won) and -1 if (defender won), 0 if (draw)
type Reward = i64;

const REPLAY_BUFFER_ENCODE_CONFIG: bincode::config::Configuration 
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

#[derive(Debug, Clone, Copy, bincode::Encode, bincode::Decode)]
pub enum EpisodeUnit {
    SAR(SAR),
    Sep
}

impl From<SAR> for EpisodeUnit {
    fn from(value: SAR) -> Self {
        EpisodeUnit::SAR(value)
    }
}

impl EpisodeUnit {
    pub fn give_reward(&mut self, reward: Reward) {
        if let Self::SAR(sar) = self {
            sar.give_reward(reward);
        }
    }

    #[cfg(feature = "torch")]
    pub fn to_tboard(&self, options: (Kind, Device)) -> TBoardEleven {
        match self {
            EpisodeUnit::SAR(sar) => sar.board.to_tboard(options),
            EpisodeUnit::Sep => {
                TBoardEleven::new(Tensor::zeros([3, 11, 11], options))
            }
        }
    }
}

#[derive(Debug, Copy, Clone, bincode::Encode, bincode::Decode)]
pub struct SAR {
    board: ElevenBoardForRB,
    state: GameState,
    action: MoveOnBoardEleven,
    repetition_counter: u8,
    reward: Reward,
}

impl SAR {
    pub fn reward_uninitialized(game: &Game, action: &MoveOnBoardEleven) -> Self {
        let board = ElevenBoardForRB::from_tbe(&game.board);
        let state = game.state.clone();
        let action = action.clone();
        let repetition_counter = get_rep_counter_for_oldest(game.repetition_counter);
        Self { 
            board,
            state,
            action,
            repetition_counter,
            reward: 0 
        }
    }

    pub fn reward_initialized(game: &Game, action: &MoveOnBoardEleven, reward: Reward) -> Self {
        let board = ElevenBoardForRB::from_tbe(&game.board);
        let state = game.state.clone();
        let action = action.clone();
        let repetition_counter = get_rep_counter_for_oldest(game.repetition_counter);
        Self { 
            board,
            state,
            action,
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
    pub fn append_wo_reward(&mut self, game: &Game, action: &MoveOnBoardEleven) {
        let sar = SAR::reward_uninitialized(game, action);
        let u = EpisodeUnit::from(sar);
        self.episode.push(u);
    }
    pub fn len(&self) -> usize {
        self.episode.len()
    }
}

#[derive(Debug, Default, bincode::Encode, bincode::Decode)]
pub struct ReplayBuffer {
    pub replay_buffer: Mutex<VecDeque<EpisodeUnit>>,
    pub capacity: usize,
}
// TODO: define custom default so that memory efficient

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        let replay_buffer = VecDeque::<EpisodeUnit>::with_capacity(capacity);
        let replay_buffer = Mutex::new(replay_buffer);
        Self { replay_buffer, capacity }
    }

    // append to the replay_buffer from the episode.
    // This enforces the capacity, meaning the oldest episodes would be removed if necessary
    pub fn extend_from_episode(&mut self, episode: Episode) {
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
            q.extend(episode.episode);
        }
    }

    // append to the replay_buffer, draining the episode.
    // This enforces the capacity, meaning the oldest episodes would be removed if necessary
    pub fn append_from_episode(&mut self, episode: &mut Episode) {
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
    pub fn empty(&mut self) {
        let mut q = self.replay_buffer.lock().unwrap();
        let _ = q.drain(..);
    }

    pub fn change_capacity(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;
    }

    // enforce the capacity, deleting the oldest episodes from the front if necessary.
    pub fn enforce_capacity(&mut self) {
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
    pub fn get_inference_input(&self, idx: usize, options: (Kind, Device)) -> Option<Tensor> {
        debug_assert!(1 <= idx && idx < self.replay_buffer.len());
        match self.replay_buffer[idx] {
            EpisodeUnit::Sep => None,
            EpisodeUnit::SAR(sar) => {
                let cur_board = sar.board.to_tboard(options).get();
                let mut past = true;
                let mut short_history: [Option<Tensor>; 4] = std::array::from_fn(|i| {

                    let ts = if past {

                        if let Some(eu) = self.replay_buffer.get(idx - i - 1) {
                            if let EpisodeUnit::SAR(sar) = eu {
                                sar.board.to_tboard(options).get()
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

                let tot_move_count: i64 = sar.state.get_turn_count().into();
                let tmc: Tensor = Tensor::scalar_tensor(tot_move_count, options);
                let tmc: Tensor = tmc.broadcast_to([1, 11, 11]);

                let rc: Tensor = Tensor::scalar_tensor(sar.repetition_counter as i64, options);
                let rc: Tensor = rc.broadcast_to([1, 11, 11]);

                let side: i64 = match sar.state.show_side() {
                    Side::Att => 1,
                    Side::Def => 0,
                };
                let side: Tensor = Tensor::scalar_tensor(side, options);
                let side: Tensor = side.broadcast_to([1, 11, 11]);

                let ts = Tensor::cat(&[
                    cur_board,
                    short_history[0].take().unwrap(),
                    short_history[1].take().unwrap(),
                    short_history[2].take().unwrap(),
                    short_history[3].take().unwrap(),
                    tmc,
                    rc,
                    side
                    ], 0);
                Some(ts)

            }
        }
    }

    // write a range of episodes into a file
    // Episodes are serialized and has to be decoded. 
    // Each episode is encoded as Vec<SAR>, followed by how many bytes it took to encode it.
    // This will be essential when retrieving data from a file
    pub fn save<R: RangeBounds<usize>>(self, path: &Path, game_range: R) -> Result<()> {
        let mut file = fs::File::create(path)?;
        let mut v: Vec<SAR> = vec![];
        let mut q = self.replay_buffer.lock()?;
        let mut game_idx = 0;
        let mut pos: usize = 0;
        let tot_len = q.len();
        let mut eu_iter = q.into_iter();

        for eu in eu_iter {
            match eu {
                EpisodeUnit::SAR(sar) => {
                    v.push(sar);
                },
                EpisodeUnit::Sep => {
                    if v.len() == 0 {
                        continue;
                    }
                    let v_sar = std::mem::replace(&mut v, vec![]);
                    if game_range.contains(game_idx) {
                        write_one_episode(v_sar, &mut file)?;
                    }
                    game_idx += 1;
                }
            }
        }
        if game_range.contains(game_idx) {
            write_one_episode(v, &mut file)?;
        }
        Ok(())
    }


    // load the latest n episodes from the specified file.
    // The capacity of the output is set to be usize::MAX
    pub fn load_from_file(path: &Path, n_episodes: usize) -> Result<Self> {

        use std::io::SeekFrom;
        let mut output: Vec<EpisodeUnit> = vec![];
        let mut file = fs::File::open(path)?;
        let mut buf_reader = BufReader::new(file);
        let mut pos = buf_reader.seek(SeekFrom::End(0))?;
        let mut episode_n_byte_buf: [u8;8] = [0;8];
        let mut episode_count = 0;

        while episode_count < n_episodes && buf_reader.stream_position().is_ok() {
            if pos < 8 {
                break;
            }
            pos = -8;
            buf_reader.seek_relative(-8)?;
            buf_reader.read(&mut episode_n_byte_buf)?;
            let episode_n_byte: u64 = u64::from_le_bytes(episode_n_byte_buf);
            pos -= episode_n_byte;
            buf_reader.seek_relative(-1 * 8 - episode_n_byte as i64)?;
            let mut data_buf = vec![0u8; episode_n_byte as usize];
            buf_reader.read(&mut data_buf)?;
            let (mut v_sar, u) = bincode::decode_from_slice::<Vec<SAR>>(
                data_buf.as_slice(), 
                REPLAY_BUFFER_ENCODE_CONFIG
            )?;
            buf_reader.seek_relative(-1 * u as i64)?;
            output.extend(v_sar.into_iter().map(|sar| EpisodeUnit::SAR(sar)).rev());
            output.push(EpisodeUnit::Sep);

            episode_count += 1;
        }
        output.reverse();
        let replay_buffer = VecDeque::from(output);
        let replay_buffer = Mutex::new(replay_buffer);
        let decoded = ReplayBuffer { replay_buffer, capacity: usize::MAX};
        Ok(decoded)
    }

    // Get the minimum index of EpisodeUnit::Sep that would exceed the specified offset
    pub fn get_next_sep_pos_exceeding_offset(v: &VecDeque<EpisodeUnit>, offset: usize) -> Option<usize> {
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

fn write_one_episode<W: Write> (v: Vec<SAR>, w: &mut W) -> Result<()> {
    let n_bytes = bincode::encode_into_std_write(
        v,
        w,
        REPLAY_BUFFER_ENCODE_CONFIG
    )? as u64;
    // At the end of each episode, store how many bytes it took to encode the episode
    w.write_all(&n_bytes.to_le_bytes())?;
    Ok(())
}