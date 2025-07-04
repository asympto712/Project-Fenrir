// TODO!
#![allow(dead_code)]
#![allow(unused)]

use game::game::{Game, GameState};
use game::board::TaflBoardEleven;
use bitboard::Direction;
use bitboard::eleven::{ElevenBoardPositionalEncoding, BoardEleven, MoveOnBoardEleven};

#[derive(Debug, Default)]
struct ReplayBuffer {
    pub replay_buffer: Vec<(TaflBoardEleven, MoveOnBoardEleven)>
}
// TODO: define custom default so that memory efficient

#[derive(Debug, Default)]
struct Episode {
    pub episode: Vec<(TaflBoardEleven, MoveOnBoardEleven)>
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