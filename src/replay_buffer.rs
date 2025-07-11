// TODO!
#![allow(dead_code)]
#![allow(unused)]


use game::game::{Game, GameState};
use game::board::TaflBoardEleven;
use game::game::get_rep_counter_for_oldest;
use game::game::Side;
use bitboard::Direction;
use bitboard::eleven::{ElevenBoardPositionalEncoding, BoardEleven, MoveOnBoardEleven};

#[cfg(feature = "torch")]
use crate::utils::TBoardEleven;
#[cfg(feature = "torch")]
use tch::Tensor;
#[cfg(feature = "torch")]
use tch::{Kind, Device};

// Temporary definition
// reward is only given at the end of the game and is ALWAYS from the perspective of Attacker.
// That means, reward = +1 if (attacker won) and -1 if (defender won), 0 if (draw)
type Reward = i64;


#[derive(Debug, Default, Copy, Clone)]
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

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Copy, Clone)]
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
struct Episode {
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
}

#[derive(Debug, Default)]
struct ReplayBuffer {
    pub replay_buffer: Vec<EpisodeUnit>
}
// TODO: define custom default so that memory efficient

impl ReplayBuffer {
    pub fn extend(&mut self, episode: Episode) {
        self.replay_buffer.extend(episode.episode);
    }

    pub fn append(&mut self, episode: &mut Episode) {
        self.replay_buffer.append(&mut episode.episode);
    }

    pub fn flush(&mut self) {
        self.replay_buffer = vec![];
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
}