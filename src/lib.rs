#![allow(unused)]
#![allow(dead_code)]
#[cfg(feature = "torch")]
pub mod model;
#[cfg(feature = "torch")]
pub mod replay_buffer;
#[cfg(feature = "torch")]
pub mod agent;
#[cfg(feature = "torch")]
pub mod self_play;
#[cfg(feature = "torch")]
pub mod train;
#[cfg(feature = "mpi")]
pub mod run;

pub mod utils;


// internal
use game::board::TaflBoardEleven;
use game::game::Game;
use bitboard::Direction;
use bitboard::eleven::{MoveOnBoardEleven, ElevenBoardPositionalEncoding};

// external
#[cfg(feature = "torch")]
use tch::Tensor;

use color_eyre::eyre::{ErrReport, Result};


