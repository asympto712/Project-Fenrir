// TODO!
#![allow(dead_code)]
#![allow(unused)]

use game::game::{Game, GameState};
use game::board::TaflBoardEleven;
use bitboard::Direction;
use bitboard::eleven::{ElevenBoardPositionalEncoding, BoardEleven, MoveOnBoardEleven};

// Temporary definition
// reward is only given at the end of the game and is ALWAYS from the perspective of Attacker.
// That means, reward = +1 if (attacker won) and -1 if (defender won), 0 if (draw)
type Reward = i64;

#[derive(Debug, Default)]
struct ReplayBuffer {
    pub replay_buffer: Vec<(TaflBoardEleven, MoveOnBoardEleven, Reward)>
}
// TODO: define custom default so that memory efficient

#[derive(Debug, Default)]
struct Episode {
    pub episode: Vec<(TaflBoardEleven, MoveOnBoardEleven, Reward)>
}
// TODO: define custom default so that memory efficient


impl ReplayBuffer {
    pub fn extend(&mut self, episode: Episode) {
        self.replay_buffer.extend(episode.episode);
    }

    pub fn flush(&mut self) {
        self.replay_buffer = vec![];
    }
}