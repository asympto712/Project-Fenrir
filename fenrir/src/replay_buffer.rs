// TODO!
#![allow(dead_code)]
#![allow(unused)]


use bitboard::seven::BoardSeven;
use game::game::{Game, GameLogic, GameState, SimpleGame};
use game::board::{BoardErrorFlag, TaflBoard};
use game::game::get_rep_counter_for_oldest;
use game::game::Side;
use bitboard::{Direction, BitBoard, MoveOnBoard};
use bitboard::eleven::{ElevenBoardPositionalEncoding, BoardEleven, MoveOnBoardEleven};
use libc::free;
use rand::distributions::Uniform;

use crate::agent::PosteriorDist;
use crate::replay_buffer;
use crate::utils::IndexPolicy;
use crate::utils::{ActionTensor, BoardTensor, TAction};
#[cfg(feature = "torch")]
use crate::utils::TBoard;
#[cfg(feature = "torch")]
use tch::Tensor;
#[cfg(feature = "torch")]
use tch::{Kind, Device};

use bincode;
use bincode::config::{Configuration, LittleEndian, Fixint};
use color_eyre::owo_colors::OwoColorize;
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
pub struct BoardPosForRB<B: BitBoard>((B, B, B));

impl<B: BitBoard> BoardPosForRB<B> {
    pub fn from_taflboard(tb: &TaflBoard<B>) -> Self {
        Self((tb.bit_att, tb.bit_def, tb.bit_king))
    }
    pub fn get(&self) -> &(B, B, B) {
        &self.0
    }
}

impl BoardPosForRB<BoardEleven> {
    #[cfg(feature = "torch")]
    pub fn to_tboard(&self, options: (Kind, Device)) -> TBoard<Game> {
        use crate::utils::BoardTensor;

        type B = BoardEleven;
        let att = <TBoard::<Game> as BoardTensor>::from_bitboard(&self.0.0, options);
        let def = <TBoard::<Game> as BoardTensor>::from_bitboard(&self.0.1, options);
        let king = <TBoard::<Game> as BoardTensor>::from_bitboard(&self.0.2, options);
        let ts = Tensor::stack(&[att.get(), def.get(), king.get()], 0);
        <TBoard::<Game> as BoardTensor>::new(ts)
    }
}

impl BoardPosForRB<BoardSeven> {
    #[cfg(feature = "torch")]
    pub fn to_tboard(&self, options: (Kind, Device)) -> TBoard<SimpleGame> {
        use crate::utils::BoardTensor;

        let att = <TBoard::<SimpleGame> as BoardTensor>::from_bitboard(&self.0.0, options);
        let def = <TBoard::<SimpleGame> as BoardTensor>::from_bitboard(&self.0.1, options);
        let king = <TBoard::<SimpleGame> as BoardTensor>::from_bitboard(&self.0.2, options);
        let ts = Tensor::stack(&[att.get(), def.get(), king.get()], 0);
        <TBoard::<SimpleGame> as BoardTensor>::new(ts)
    }
}

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub enum EpisodeUnit<D: BoardData> {
    SPR(D),
    Sep
}

impl<D: BoardData> From<D> for EpisodeUnit<D> {
    fn from(value: D) -> Self {
        EpisodeUnit::SPR(value)
    }
}

impl<D: BoardData> EpisodeUnit<D> {
    pub fn give_reward(&mut self, reward: Reward) {
        if let Self::SPR(spr) = self {
            spr.give_reward(reward);
        }
    }
}

#[cfg(feature = "torch")]
impl EpisodeUnit<GameSPR> {
    pub fn to_tboard(&self, options: (Kind, Device)) -> TBoard<Game> {
        match self {
            EpisodeUnit::SPR(spr) => spr.board.to_tboard(options),
            EpisodeUnit::Sep => {
                <TBoard::<Game> as BoardTensor>::new(Tensor::zeros([3, 11, 11], options))
            }
        }
    }
}

#[cfg(feature = "torch")]
impl EpisodeUnit<SimpleGameSPR> {
    pub fn to_tboard(&self, options: (Kind, Device)) -> TBoard<SimpleGame> {
        match self {
            EpisodeUnit::SPR(spr) => spr.board.to_tboard(options),
            EpisodeUnit::Sep => {
                <TBoard::<SimpleGame> as BoardTensor>::new(Tensor::zeros([3, 7, 7], options))
            }
        }
    }
}

pub trait BoardData:
    bincode::Encode +
    bincode::Decode<()> +
    Sync +
    Send
{
    type G: GameLogic;
    fn reward_uninitialized(game: &Self::G, posterior: PosteriorDist<<Self::G as GameLogic>::B>) -> Self;
    fn reward_initialized(game: &Self::G, posterior: PosteriorDist<<Self::G as GameLogic>::B>, reward: Reward) -> Self;
    fn give_reward(&mut self, reward: Reward);
}

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct GameSPR {
    board: BoardPosForRB<BoardEleven>,
    state: GameState,
    policy: IndexPolicy,
    repetition_counter: u8,
    reward: Reward,
}

impl BoardData for GameSPR{
    type G = Game;
    fn reward_uninitialized(game: &Self::G, posterior: PosteriorDist<<Self::G as GameLogic>::B>) -> Self {
        let board = BoardPosForRB::<BoardEleven>::from_taflboard(&game.board);
        let state = game.state.clone();
        let repetition_counter = get_rep_counter_for_oldest(game.repetition_counter);
        let policy: IndexPolicy = IndexPolicy::from_posterior::<BoardEleven>(posterior);
        Self { 
            board,
            state,
            policy,
            repetition_counter,
            reward: 0 
        }
    }

    fn reward_initialized(game: &Self::G, posterior: PosteriorDist<<Self::G as GameLogic>::B>, reward: Reward) -> Self {
        let board = BoardPosForRB::<BoardEleven>::from_taflboard(&game.board);
        let state = game.state.clone();
        let repetition_counter = get_rep_counter_for_oldest(game.repetition_counter);
        let policy: IndexPolicy = IndexPolicy::from_posterior::<BoardEleven>(posterior);
        Self { 
            board,
            state,
            policy,
            repetition_counter,
            reward
        }
    }

    fn give_reward(&mut self, reward: Reward) {
        self.reward = reward;
    }
}

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct SimpleGameSPR{
    board: BoardPosForRB<BoardSeven>,
    state: GameState,
    policy: IndexPolicy,
    reward: Reward,
}

impl BoardData for SimpleGameSPR {

    type G = SimpleGame;

    fn reward_uninitialized(game: &Self::G, posterior: PosteriorDist<<Self::G as GameLogic>::B>) -> Self {
        let board = BoardPosForRB::<BoardSeven>::from_taflboard(&game.board);
        let state = game.state.clone();
        let policy: IndexPolicy = IndexPolicy::from_posterior::<BoardSeven>(posterior);
        Self { 
            board,
            state,
            policy,
            reward: 0 
        }
    }

    fn reward_initialized(game: &Self::G, posterior: PosteriorDist<<Self::G as GameLogic>::B>, reward: Reward) -> Self {
        let board = BoardPosForRB::<BoardSeven>::from_taflboard(&game.board);
        let state = game.state.clone();
        let policy: IndexPolicy = IndexPolicy::from_posterior::<BoardSeven>(posterior);
        Self { 
            board,
            state,
            policy,
            reward
        }
    }
    fn give_reward(&mut self, reward: Reward) {
        self.reward = reward;
    }
}

#[derive(Debug, Default)]
pub struct Episode<D: BoardData> {
    pub episode: Vec<EpisodeUnit<D>>
}

impl<D: BoardData> Episode<D> {
    pub fn give_reward(&mut self, reward: Reward) {
        for u in self.episode.iter_mut() {
            u.give_reward(reward);
        }
    }
    pub fn new() -> Self {
        let episode = vec![EpisodeUnit::<D>::Sep];
        Self { episode }
    }
    pub fn append_wo_reward(&mut self, game: &D::G, posterior: PosteriorDist<<D::G as GameLogic>::B>) {
        let spr = D::reward_uninitialized(game, posterior);
        let u = EpisodeUnit::<D>::from(spr);
        self.episode.push(u);
    }
    pub fn len(&self) -> usize {
        self.episode.len()
    }

    // This will make the episode vector empty
    pub fn save<W: Write>(&mut self, writer: &mut W) -> Result<()> {
        if self.episode.len() == 0 {
            return Err(eyre!("episode must have at least one length"));
        }
        if let EpisodeUnit::SPR(_) = self.episode[0] {
            return Err(eyre!("episode must start with sep"));
        }
        let mut v: Vec<D> = vec![];

        for eu in self.episode.drain(..) {
            match eu {
                EpisodeUnit::<D>::SPR(spr) => {
                    v.push(spr);
                },
                EpisodeUnit::<D>::Sep => {}
            }
        }
        write_one_episode::<W,D>(v, writer)?;
        Ok(())
    }

    pub fn extend_from_reader<R: Read + Seek>(&mut self, mut reader: R, buf: &mut Vec<u8>) -> Result<()> {

        use std::io::SeekFrom;
        let mut pos = reader.seek(SeekFrom::End(0))?;
        let mut episode_n_byte_buf: [u8;8] = [0;8];
        let mut episode_count = 0;
        buf.clear();

        while reader.stream_position().is_ok() {
            if pos < 8 {
                break;
            }
            pos -= 8;
            reader.seek_relative(-8)?;
            reader.read_exact(&mut episode_n_byte_buf)?;
            let episode_n_byte: u64 = u64::from_le_bytes(episode_n_byte_buf);

            pos -= episode_n_byte;
            reader.seek_relative(-8 - episode_n_byte as i64)?;

            reader.read_exact(buf)?;

            let (mut v_spr, u) = bincode::decode_from_slice::<Vec<D>, Configuration<LittleEndian, Fixint>>(
                &buf[..], 
                REPLAY_BUFFER_ENCODE_CONFIG
            )?;
            buf.clear();
            reader.seek_relative(-1 * u as i64)?;
            self.episode.push(EpisodeUnit::Sep);
            self.episode.extend(v_spr.into_iter().map(|spr| EpisodeUnit::SPR(spr)));

            episode_count += 1;
        }
        Ok(())
    }

    pub fn load_from_reader<R: Read + Seek>(mut reader: R) -> Result<Self> {

        use std::io::SeekFrom;
        let mut output: Vec<EpisodeUnit<D>> = vec![];
        let mut pos = reader.seek(SeekFrom::End(0))?;
        let mut episode_n_byte_buf: [u8;8] = [0;8];
        let mut episode_count = 0;

        while reader.stream_position().is_ok() {
            if pos < 8 {
                break;
            }
            pos -= 8;
            reader.seek_relative(-8)?;
            reader.read_exact(&mut episode_n_byte_buf)?;
            let episode_n_byte: u64 = u64::from_le_bytes(episode_n_byte_buf);

            pos -= episode_n_byte;
            reader.seek_relative(-8 - episode_n_byte as i64)?;
            // Maybe if we cap the total moves per game, we can allocate this on the stack
            let mut data_buf = vec![0u8; episode_n_byte as usize];
            reader.read_exact(&mut data_buf)?;

            let (mut v_spr, u) = bincode::decode_from_slice::<Vec<D>, Configuration<LittleEndian, Fixint>>(
                data_buf.as_slice(), 
                REPLAY_BUFFER_ENCODE_CONFIG
            )?;
            reader.seek_relative(-1 * u as i64)?;
            output.extend(v_spr.into_iter().map(|spr| EpisodeUnit::SPR(spr)).rev());
            output.push(EpisodeUnit::Sep);

            episode_count += 1;
        }

        output.reverse();

        let decoded = Episode::<D>{ episode: output};
        Ok(decoded)
    }
}

#[derive(Debug, Default, bincode::Encode, bincode::Decode)]
pub struct ReplayBuffer<D: BoardData + bincode::Encode> {
    pub replay_buffer: Mutex<Vec<EpisodeUnit<D>>>,
    pub capacity: usize,
}
// TODO: define custom default so that memory efficient

impl<D: BoardData> ReplayBuffer<D> {

    pub fn len(&self) -> usize {
        self.replay_buffer.lock().unwrap().len()
    }

    pub fn new(capacity: usize) -> Self {
        let replay_buffer = Vec::<EpisodeUnit<D>>::with_capacity(capacity);
        let replay_buffer = Mutex::new(replay_buffer);
        Self { replay_buffer, capacity }
    }

    // append to the replay_buffer from the episode.
    // This enforces the capacity, meaning the oldest episodes would be removed if necessary
    pub fn extend_from_episode(&self, episode: Episode<D>) {

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
    pub fn append_from_episode(&self, episode: &mut Episode<D>) {
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


    // write a range of episodes into a file
    // Episodes are serialized and has to be decoded. 
    // Each episode is encoded as Vec<SPR>, followed by how many bytes it took to encode it.
    // This will be essential when retrieving data from a file
    pub fn save<R: RangeBounds<usize>, W: Write>(self, writer: &mut W, game_range: R) -> Result<()> {
        let mut v: Vec<D> = vec![];
        let mut freed_history = self.replay_buffer.into_inner().map_err(|_| eyre!("Could not acquire lock for replay buffer"))?;
        let mut game_idx = 0;
        let mut pos: usize = 0;
        let tot_len = freed_history.len();

        for eu in freed_history {
            match eu {
                EpisodeUnit::<D>::SPR(spr) => {
                    v.push(spr);
                },
                EpisodeUnit::<D>::Sep => {
                    if v.len() == 0 {
                        continue;
                    }
                    let v_spr = std::mem::replace(&mut v, vec![]);
                    if game_range.contains(&game_idx) {
                        write_one_episode::<W,D>(v_spr, writer)?;
                    }
                    game_idx += 1;
                }
            }
        }
        if game_range.contains(&game_idx) {
            write_one_episode::<W,D>(v, writer)?;
        }
        Ok(())
    }

    // load the latest n episodes from the specified file.
    // The capacity of the output is set to be usize::MAX
    pub fn load_from_reader<R: Read + Seek>(mut reader: R, n_episodes: usize) -> Result<Self> {

        use std::io::SeekFrom;
        let mut output: Vec<EpisodeUnit<D>> = vec![];
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

            let (mut v_spr, u) = bincode::decode_from_slice::<Vec<D>, Configuration<LittleEndian, Fixint>>(
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
    pub fn get_next_sep_pos_exceeding_offset(v: &Vec<EpisodeUnit<D>>, offset: usize) -> Option<usize> {
        let mut count: usize = 0;
        for eu in v.iter(){
            if let EpisodeUnit::<D>::Sep = *eu && count > offset{
                return Some(count)
            }
            count += 1;
        }
        None
    }

}

#[cfg(feature = "torch")]
pub trait Sampler {
    fn sample_batch<R: Rng + ?Sized>(&self, batch_size: usize, options: (Kind, Device), rng: &mut R) -> Result<(Tensor, Tensor, Tensor)>; 
}

#[cfg(feature = "torch")]
impl ReplayBuffer<GameSPR> {

    pub fn get_inference_input(q: &Vec<EpisodeUnit<GameSPR>>, idx: usize, options: (Kind, Device)) -> Option<(Tensor, Tensor, Tensor)> {

        if let Some(eu) = q.get(idx) {
            match eu {
                EpisodeUnit::<GameSPR>::Sep => {
                    None
                }
                EpisodeUnit::<GameSPR>::SPR(spr) => {

                    let policy = <TAction::<BoardEleven> as ActionTensor>::from_index_policy(&spr.policy)
                        .inner()
                        .to_kind(options.0)
                        .view([-1])
                        .to_device(options.1);

                    let reward = Tensor::from_slice(&[spr.reward as f32])
                        .to_kind(options.0)
                        .to_device(options.1);

                    let cur_board = spr.board.to_tboard(options).get();
                    let mut past = if idx > 0 {true} else {false}; // fail-safe logic to ensure that subtract with overflow won't happen
                    let mut short_history: [Option<Tensor>; 4] = std::array::from_fn(|i| {

                        let ts = if past {

                            if (idx - 1 - i) > 0 && let Some(eu) = q.get(idx - i - 1) {
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

}

impl Sampler for ReplayBuffer<GameSPR> {

    fn sample_batch<R: Rng + ?Sized>(&self, batch_size: usize, options: (Kind, Device), rng: &mut R) -> Result<(Tensor, Tensor, Tensor)> {

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
}


#[cfg(feature = "torch")]
impl ReplayBuffer<SimpleGameSPR> {

    pub fn get_inference_input(q: &Vec<EpisodeUnit<SimpleGameSPR>>, idx: usize, options: (Kind, Device)) -> Option<(Tensor, Tensor, Tensor)> {

        if let Some(eu) = q.get(idx) {
            match eu {
                EpisodeUnit::<SimpleGameSPR>::Sep => {
                    None
                }
                EpisodeUnit::<SimpleGameSPR>::SPR(spr) => {

                    let policy = <TAction::<BoardSeven> as ActionTensor>::from_index_policy(&spr.policy)
                        .inner()
                        .to_kind(options.0)
                        .view([-1])
                        .to_device(options.1);

                    let reward = Tensor::from_slice(&[spr.reward as f32])
                        .to_kind(options.0)
                        .to_device(options.1);

                    let tboard = spr.board.to_tboard(options);
                    let cur_board = <TBoard<SimpleGame> as BoardTensor>::get(tboard);

                    let tot_move_count: i64 = spr.state.get_turn_count().into();
                    let tmc: Tensor = Tensor::scalar_tensor(tot_move_count, options);
                    let tmc: Tensor = tmc.broadcast_to([1, 7, 7]);

                    let side: i64 = match spr.state.show_side() {
                        Side::Att => 1,
                        Side::Def => 0,
                    };
                    let side: Tensor = Tensor::scalar_tensor(side, options);
                    let side: Tensor = side.broadcast_to([1, 7, 7]);

                    let position = Tensor::cat(&[
                        cur_board,
                        tmc,
                        side
                        ], 0);
                    
                    Some((position, policy, reward))
                }
            }
        } else {
            None
        }
    }

}

impl Sampler for ReplayBuffer<SimpleGameSPR> {
    fn sample_batch<R: Rng + ?Sized>(&self, batch_size: usize, options: (Kind, Device), rng: &mut R) -> Result<(Tensor, Tensor, Tensor)> {

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
    
} 

fn write_one_episode<W: Write, D: BoardData + bincode::Encode> (v: Vec<D>, w: &mut W) -> Result<()> {
    let n_bytes = bincode::encode_into_std_write(
        v,
        w,
        REPLAY_BUFFER_ENCODE_CONFIG
    )? as u64;
    // At the end of each episode, store how many bytes it took to encode the episode
    w.write_all(&n_bytes.to_le_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use game::board::TaflBoard;
    use game::game::{Game, SimpleGame};
    use bitboard::eleven::{BoardEleven as BE, ElevenBoardPositionalEncoding, MoveOnBoardEleven};
    use bitboard::seven::{BoardSeven as BS, SevenBoardPositionalEncoding, MoveOnBoardSeven};
    use bitboard::{BitBoard, MoveOnBoard, PositionalEncoding};
    use crate::agent::PosteriorDist;
    use std::collections::HashMap;
    use std::io::Cursor;
    use rand::thread_rng;


    // Helper function to create a mock posterior distribution
    fn create_mock_posterior<B: BitBoard>() -> PosteriorDist<B> {
        let mut posterior = Vec::new();
        let start = <B::Position as PositionalEncoding>::new(5, 5);
        let dst = <B::Position as PositionalEncoding>::new(5, 6);
        let movement = <B::Movement as MoveOnBoard>::new(start, dst);
        posterior.push((movement, 0.8));
        
        let start2 = <B::Position as PositionalEncoding>::new(4, 4);
        let dst2 = <B::Position as PositionalEncoding>::new(4, 5);
        let movement2 = <B::Movement as MoveOnBoard>::new(start2, dst2);
        posterior.push((movement2, 0.2));
        
        posterior
    }

    // Helper function to create a mock game
    fn create_mock_game() -> Game {
        Game::default()
    }

    // Helper function to create a mock simple game
    fn create_mock_simple_game() -> SimpleGame {
        SimpleGame::default()
    }

    // Tests without torch feature
    mod non_torch_tests {
        use super::*;

        #[test]
        fn test_board_pos_for_rb_from_taflboard() {
            let mut rng = thread_rng();
            let tafl_board = TaflBoard::<BE>::generate_random_board(&mut rng);
            let board_pos = BoardPosForRB::<BE>::from_taflboard(&tafl_board);
            
            // Verify that the conversion preserves the bitboards
            assert_eq!(board_pos.0.0, tafl_board.bit_att);
            assert_eq!(board_pos.0.1, tafl_board.bit_def);
            assert_eq!(board_pos.0.2, tafl_board.bit_king);
        }

        #[test]
        fn test_episode_unit_from_trait() {
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            let spr = GameSPR::reward_uninitialized(&game, posterior);
            let episode_unit: EpisodeUnit<GameSPR> = EpisodeUnit::from(spr);
            
            match episode_unit {
                EpisodeUnit::SPR(_) => {}, // Expected
                EpisodeUnit::Sep => panic!("Expected SPR, got Sep"),
            }
        }

        #[test]
        fn test_episode_unit_give_reward() {
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            let spr = GameSPR::reward_uninitialized(&game, posterior);
            let mut episode_unit = EpisodeUnit::<GameSPR>::from(spr);
            
            episode_unit.give_reward(100);
            
            if let EpisodeUnit::SPR(spr) = episode_unit {
                assert_eq!(spr.reward, 100);
            } else {
                panic!("Expected SPR");
            }
        }

        #[test]
        fn test_game_spr_reward_uninitialized() {
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            let spr = GameSPR::reward_uninitialized(&game, posterior);
            
            assert_eq!(spr.reward, 0);
            assert_eq!(spr.state.get_turn_count(), game.state.get_turn_count());
        }

        #[test]
        fn test_game_spr_reward_initialized() {
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            let reward = 150;
            let spr = GameSPR::reward_initialized(&game, posterior, reward);
            
            assert_eq!(spr.reward, reward);
            assert_eq!(spr.state.get_turn_count(), game.state.get_turn_count());
        }

        #[test]
        fn test_game_spr_give_reward() {
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            let mut spr = GameSPR::reward_uninitialized(&game, posterior);
            
            assert_eq!(spr.reward, 0);
            spr.give_reward(200);
            assert_eq!(spr.reward, 200);
        }

        #[test]
        fn test_episode_new() {
            let episode: Episode<GameSPR> = Episode::new();
            assert_eq!(episode.len(), 1); // Should contain one Sep unit
            
            match &episode.episode[0] {
                EpisodeUnit::Sep => {}, // Expected
                EpisodeUnit::SPR(_) => panic!("Expected Sep, got SPR"),
            }
        }

        #[test]
        fn test_episode_append_wo_reward() {
            let mut episode: Episode<GameSPR> = Episode::new();
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            
            episode.append_wo_reward(&game, posterior);
            assert_eq!(episode.len(), 2); // Sep + SPR
            
            match &episode.episode[1] {
                EpisodeUnit::SPR(spr) => assert_eq!(spr.reward, 0),
                EpisodeUnit::Sep => panic!("Expected SPR, got Sep"),
            }
        }

        #[test]
        fn test_episode_give_reward() {
            let mut episode: Episode<GameSPR> = Episode::new();
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            
            episode.append_wo_reward(&game, posterior.clone());
            episode.append_wo_reward(&game, posterior);
            
            episode.give_reward(50);
            
            // Check that all SPR units received the reward
            for unit in &episode.episode {
                if let EpisodeUnit::SPR(spr) = unit {
                    assert_eq!(spr.reward, 50);
                }
            }
        }

        #[test]
        fn test_replay_buffer_new() {
            let capacity = 1000;
            let buffer: ReplayBuffer<GameSPR> = ReplayBuffer::new(capacity);
            
            assert_eq!(buffer.capacity, capacity);
            let guard = buffer.replay_buffer.lock().unwrap();
            assert_eq!(guard.len(), 0);
            assert_eq!(guard.capacity(), capacity);
        }

        #[test]
        fn test_replay_buffer_extend_from_episode() {
            let buffer: ReplayBuffer<GameSPR> = ReplayBuffer::new(100);
            let mut episode: Episode<GameSPR> = Episode::new();
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            
            episode.append_wo_reward(&game, posterior);
            let initial_len = episode.len();
            
            buffer.extend_from_episode(episode);
            
            let guard = buffer.replay_buffer.lock().unwrap();
            assert_eq!(guard.len(), initial_len);
        }

        #[test]
        fn test_replay_buffer_append_from_episode() {
            let buffer: ReplayBuffer<GameSPR> = ReplayBuffer::new(100);
            let mut episode: Episode<GameSPR> = Episode::new();
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            
            episode.append_wo_reward(&game, posterior);
            let initial_len = episode.len();
            
            buffer.append_from_episode(&mut episode);
            
            let guard = buffer.replay_buffer.lock().unwrap();
            assert_eq!(guard.len(), initial_len);
            assert_eq!(episode.len(), 0); // Episode should be drained
        }

        #[test]
        fn test_replay_buffer_empty() {
            let buffer: ReplayBuffer<GameSPR> = ReplayBuffer::new(100);
            let mut episode: Episode<GameSPR> = Episode::new();
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            
            episode.append_wo_reward(&game, posterior);
            buffer.extend_from_episode(episode);
            
            // Verify buffer has content
            {
                let guard = buffer.replay_buffer.lock().unwrap();
                assert!(guard.len() > 0);
            }
            
            buffer.empty();
            
            let guard = buffer.replay_buffer.lock().unwrap();
            assert_eq!(guard.len(), 0);
        }

        #[test]
        fn test_replay_buffer_change_capacity() {
            let mut buffer: ReplayBuffer<GameSPR> = ReplayBuffer::new(100);
            assert_eq!(buffer.capacity, 100);
            
            buffer.change_capacity(200);
            assert_eq!(buffer.capacity, 200);
        }

        #[test]
        fn test_get_next_sep_pos_exceeding_offset() {
            let mut v: Vec<EpisodeUnit<GameSPR>> = vec![];
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            let spr = GameSPR::reward_uninitialized(&game, posterior);
            
            // Create a pattern: SPR, SPR, Sep, SPR, Sep, SPR, SPR, Sep
            v.push(EpisodeUnit::SPR(spr.clone()));
            v.push(EpisodeUnit::SPR(spr.clone()));
            v.push(EpisodeUnit::Sep);
            v.push(EpisodeUnit::SPR(spr.clone()));
            v.push(EpisodeUnit::Sep);
            v.push(EpisodeUnit::SPR(spr.clone()));
            v.push(EpisodeUnit::SPR(spr.clone()));
            v.push(EpisodeUnit::Sep);
            
            // Test with offset 1 (should find first Sep at index 2)
            let result = ReplayBuffer::<GameSPR>::get_next_sep_pos_exceeding_offset(&v, 1);
            assert_eq!(result, Some(2));
            
            // Test with offset 3 (should find Sep at index 4)
            let result = ReplayBuffer::<GameSPR>::get_next_sep_pos_exceeding_offset(&v, 3);
            assert_eq!(result, Some(4));
            
            // Test with offset exceeding all elements
            let result = ReplayBuffer::<GameSPR>::get_next_sep_pos_exceeding_offset(&v, 10);
            assert_eq!(result, None);
        }

        #[test]
        fn test_replay_buffer_enforce_capacity() {
            let capacity = 5;
            let buffer: ReplayBuffer<GameSPR> = ReplayBuffer::new(capacity);
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            
            // Add more episodes than capacity
            for _ in 0..3 {
                let mut episode: Episode<GameSPR> = Episode::new();
                episode.append_wo_reward(&game, posterior.clone());
                episode.append_wo_reward(&game, posterior.clone());
                buffer.extend_from_episode(episode);
            }
            
            // Force capacity enforcement
            buffer.enforce_capacity();
            
            let guard = buffer.replay_buffer.lock().unwrap();
            assert!(guard.len() <= capacity);
        }

        #[test]
        fn test_replay_buffer_save_and_load() {
            let buffer: ReplayBuffer<GameSPR> = ReplayBuffer::new(100);
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            
            // Add some episodes
            for i in 0..3 {
                let mut episode: Episode<GameSPR> = Episode::new();
                episode.append_wo_reward(&game, posterior.clone());
                episode.give_reward(i as i64 * 10);
                buffer.extend_from_episode(episode);
            }
            
            // Save to buffer
            let mut write_buffer = Vec::new();
            {
                let original_len = buffer.replay_buffer.lock().unwrap().len();
                assert!(original_len > 0);
            }
            
            // Test saving (note: this consumes the buffer)
            let result = buffer.save(&mut write_buffer, ..);
            assert!(result.is_ok());
            assert!(!write_buffer.is_empty());
            
            // Test loading
            let mut cursor = Cursor::new(write_buffer);
            let loaded_buffer = ReplayBuffer::<GameSPR>::load_from_reader(cursor, 3);
            assert!(loaded_buffer.is_ok());
            
            let loaded = loaded_buffer.unwrap();
            let guard = loaded.replay_buffer.lock().unwrap();
            assert!(guard.len() > 0);
        }

        #[test]
        fn gamespr_encode_size() {
            let game = Game::init_std();
            let posterior = game.get_possible_actions().into_iter().map(|x| (x, 1.0f32)).collect::<Vec<_>>();
            let gamespr = GameSPR::reward_initialized(&game, posterior, 1);
            let v = bincode::encode_to_vec(gamespr, REPLAY_BUFFER_ENCODE_CONFIG).unwrap();
            println!("{}", v.len());
        }

        #[test]
        fn simplegamespr_encode_size() {
            let game = SimpleGame::init_std();
            let posterior = game.get_possible_actions().into_iter().map(|x| (x, 1.0f32)).collect::<Vec<_>>();
            let gamespr = SimpleGameSPR::reward_initialized(&game, posterior, 1);
            let v = bincode::encode_to_vec(gamespr, REPLAY_BUFFER_ENCODE_CONFIG).unwrap();
            println!("{}", v.len());
        }

        #[test]
        fn bincode_test() {
            let policy: Vec<(i64, f32)> = vec![(1, 1.0); 10];
            let v = bincode::encode_to_vec(policy, REPLAY_BUFFER_ENCODE_CONFIG).unwrap();
            println!("{}", v.len());
            assert_eq!(v.len(), 10 * 12 + 8);
        }
    }

    // Tests that require torch feature
    #[cfg(feature = "torch")]
    mod torch_tests {
        use super::*;
        use tch::{Device, Kind, Tensor};
        use crate::utils::TBoard;

        fn get_torch_options() -> (Kind, Device) {
            (Kind::Float, Device::Cpu)
        }

        #[test]
        fn test_board_pos_for_rb_to_tboard_eleven() {
            let mut rng = thread_rng();
            let tafl_board = TaflBoard::<BoardEleven>::generate_random_board(&mut rng);
            let board_pos = BoardPosForRB::from_taflboard(&tafl_board);
            let options = get_torch_options();
            
            let tboard = board_pos.to_tboard(options);
            let tensor = tboard.get_ref();
            
            assert_eq!(tensor.size(), [3, 11, 11]);
            assert_eq!(tensor.kind(), Kind::Float);
            assert_eq!(tensor.device(), Device::Cpu);
        }

        #[test]
        fn test_episode_unit_to_tboard() {
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            let spr = GameSPR::reward_uninitialized(&game, posterior);
            let episode_unit = EpisodeUnit::SPR(spr);
            let options = get_torch_options();
            
            let tboard = episode_unit.to_tboard(options);
            let tensor = tboard.get_ref();
            
            assert_eq!(tensor.size(), [3, 11, 11]);
            assert_eq!(tensor.kind(), Kind::Float);
        }

        #[test]
        fn test_episode_unit_sep_to_tboard() {
            let episode_unit: EpisodeUnit<GameSPR> = EpisodeUnit::Sep;
            let options = get_torch_options();
            
            let tboard = episode_unit.to_tboard(options);
            let tensor = tboard.get_ref();
            
            assert_eq!(tensor.size(), [3, 11, 11]);
            assert_eq!(tensor.kind(), Kind::Float);
            // Should be all zeros for Sep
            // sum will reduce it to 0-dim tensor, so we need to unsqueeze it?
            let sum = tensor.sum(Kind::Float).unsqueeze(0).double_value(&[0]);
            assert_eq!(sum, 0.0);
        }

        #[test]
        fn test_replay_buffer_get_inference_input() {
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            let spr = GameSPR::reward_initialized(&game, posterior, 100);
            let mut buffer_data = vec![EpisodeUnit::SPR(spr)];
            let options = get_torch_options();
            
            let result = ReplayBuffer::<GameSPR>::get_inference_input(&buffer_data, 0, options);
            assert!(result.is_some());
            
            let (position, policy, reward) = result.unwrap();
            
            // Check dimensions
            assert_eq!(position.size().len(), 3); // Should be 3D tensor
            assert_eq!(policy.size().len(), 3);   // Should be 3D tensor (20, 11, 11)
            assert_eq!(reward.size(), [1]);       // Should be scalar tensor
            
            // Check reward value
            let reward_value = reward.unsqueeze(0).double_value(&[0]);
            assert_eq!(reward_value, 100.0);
        }

        #[test]
        fn test_replay_buffer_get_inference_input_with_sep() {
            let sep_unit: EpisodeUnit<GameSPR> = EpisodeUnit::Sep;
            let buffer_data = vec![sep_unit];
            let options = get_torch_options();
            
            let result = ReplayBuffer::<GameSPR>::get_inference_input(&buffer_data, 0, options);
            assert!(result.is_none());
        }

        #[test]
        fn test_replay_buffer_get_inference_input_with_history() {
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            
            // Create buffer with history: older SPR, then current SPR
            let older_spr = GameSPR::reward_initialized(&game, posterior.clone(), 50);
            let current_spr = GameSPR::reward_initialized(&game, posterior, 100);
            let buffer_data = vec![
                EpisodeUnit::SPR(older_spr),
                EpisodeUnit::SPR(current_spr)
            ];
            let options = get_torch_options();
            
            let result = ReplayBuffer::<GameSPR>::get_inference_input(&buffer_data, 1, options);
            assert!(result.is_some());
            
            let (position, policy, reward) = result.unwrap();
            
            // Position should include history from previous states
            assert_eq!(position.size().len(), 3);
            let expected_channels = 3 + 4*3 + 1 + 1 + 1; // current + history + tmc + rc + side
            assert_eq!(position.size()[0], expected_channels as i64);
        }

        #[test]
        fn test_replay_buffer_sample_batch() {
            let buffer: ReplayBuffer<GameSPR> = ReplayBuffer::new(100);
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            
            // Add multiple episodes to have enough samples
            for i in 0..10 {
                let spr = GameSPR::reward_initialized(&game, posterior.clone(), i * 10);
                let episode_unit = EpisodeUnit::SPR(spr);
                buffer.replay_buffer.lock().unwrap().push(episode_unit);
            }
            
            let mut rng = thread_rng();
            let batch_size = 5;
            let options = get_torch_options();
            
            let result = buffer.sample_batch(batch_size, options, &mut rng);
            assert!(result.is_ok());
            
            let (positions, policies, rewards) = result.unwrap();
            
            // Check batch dimensions
            assert_eq!(positions.size()[0], batch_size as i64);
            assert_eq!(policies.size()[0], batch_size as i64);
            assert_eq!(rewards.size()[0], batch_size as i64);
            
            // Check individual sample dimensions
            assert_eq!(positions.size().len(), 4); // (batch, channels, height, width)
            assert_eq!(policies.size().len(), 4);   // (batch, actions, height, width)
            assert_eq!(rewards.size().len(), 2);    // (batch, 1)
        }

        #[test]
        fn test_replay_buffer_sample_batch_insufficient_data() {
            let buffer: ReplayBuffer<GameSPR> = ReplayBuffer::new(100);
            let mut rng = thread_rng();
            let batch_size = 5;
            let options = get_torch_options();
            
            // Try to sample from empty buffer
            let result = buffer.sample_batch(batch_size, options, &mut rng);
            assert!(result.is_err());
        }

        #[test]
        fn test_simple_game_spr_inference_input() {
            let game = create_mock_simple_game();
            let posterior = create_mock_posterior::<BoardSeven>();
            let spr = SimpleGameSPR::reward_initialized(&game, posterior, 75);
            let buffer_data = vec![EpisodeUnit::SPR(spr)];
            let options = get_torch_options();
            
            let result = ReplayBuffer::<SimpleGameSPR>::get_inference_input(&buffer_data, 0, options);
            assert!(result.is_some());
            
            let (position, policy, reward) = result.unwrap();
            
            // SimpleGame should have fewer channels than full Game
            assert_eq!(position.size().len(), 3);
            let expected_channels = 3 + 1 + 1; // board + tmc + side (no history or repetition counter)
            assert_eq!(position.size()[0], expected_channels as i64);
            
            let reward_value = reward.unsqueeze(0).double_value(&[0]);
            assert_eq!(reward_value, 75.0);
        }

        #[test]
        fn test_tensor_device_and_kind_consistency() {
            let game = create_mock_game();
            let posterior = create_mock_posterior::<BoardEleven>();
            let spr = GameSPR::reward_initialized(&game, posterior, 123);
            let buffer_data = vec![EpisodeUnit::SPR(spr)];
            
            let options = (Kind::Double, Device::Cpu);
            let result = ReplayBuffer::<GameSPR>::get_inference_input(&buffer_data, 0, options);
            assert!(result.is_some());
            
            let (position, policy, reward) = result.unwrap();
            
            // All tensors should have the requested kind and device
            assert_eq!(position.kind(), Kind::Double);
            assert_eq!(policy.kind(), Kind::Double);
            assert_eq!(reward.kind(), Kind::Double);
            
            assert_eq!(position.device(), Device::Cpu);
            assert_eq!(policy.device(), Device::Cpu);
            assert_eq!(reward.device(), Device::Cpu);
        }
    }
}