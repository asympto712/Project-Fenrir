#![allow(unused)]
use std::borrow::Borrow;
use std::process::id;

// workspace
use game::board::{self, TaflBoard};
use game::game::{Game, SimpleGame, GameLogic, Side, ShortHistory, get_rep_counter_for_oldest};
use bitboard::Direction;
use bitboard::{BitBoard, PositionalEncoding, MoveOnBoard};
use bitboard::eleven::{BoardEleven, MoveOnBoardEleven};
use bitboard::seven::{BoardSeven, MoveOnBoardSeven, SevenBoardPositionalEncoding};

// external
use color_eyre::eyre::{ErrReport, Result};
use rand::prelude;
use bincode;
use rand::thread_rng;
use rand::Rng;

#[cfg(feature = "torch")]
use tch::Tensor;
#[cfg(feature = "torch")]
use tch::{Kind, Device};

#[cfg(feature = "torch")]
use libc::c_void;

use crate::agent::PosteriorDist;

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct IndexPolicy(Vec<(i64, f32)>);

impl<B: BitBoard> From<PosteriorDist<B>> for IndexPolicy {
    fn from(value: PosteriorDist<B>) -> Self {
        let v = value.into_iter()
            .map(|(movement, p)| {
                let vbm = VectorBasedMove::convert_from(&movement).unwrap();
                (vbm.to_index(), p)
            })
            .collect::<Vec<(i64, f32)>>();
        Self(v)
    }
}

impl IndexPolicy {
    fn get(&self) -> &Vec<(i64, f32)> {
        &self.0
    }
}



// Directional move. Assumes Periodical Boundary
#[derive(Copy, Clone, Debug, PartialEq)]
enum MoveVector<B: BitBoard> {
    S(u8),
    E(u8),
} 

impl<B: BitBoard> MoveVector<B> {

    pub fn movelen_lim() -> u8{
        B::BOARD_SIZE - 1
    }

    pub fn from_index(i: i64) -> Self {
        assert!(0 <= i && i < Self::movelen_lim() * 2);
        if i < 10 { Self::S((i + 1).try_into().unwrap()) }
        else { Self::E( (i + 1 - Self::movelen_lim()).try_into().unwrap())}
    }

    pub fn to_index(self) -> i64 {
        match self {
            MoveVector::S(i) => (i - 1).into(),
            MoveVector::E(i) => (i + Self::movelen_lim() - 1).into()
        }
    }

    pub fn rotate90(&self) -> Self {
        match self {
            MoveVector::S(n) => MoveVector::E(*n),
            MoveVector::E(n) => MoveVector::S(Self::movelen_lim() + 1 - n),
        }
    }

    pub fn rotate180(&self) -> Self {
        match self {
            MoveVector::S(n) => MoveVector::S(Self::movelen_lim() + 1 - n),
            MoveVector::E(n) => MoveVector::E(Self::movelen_lim() + 1 - n),
        }
    }

    pub fn rotate270(&self) -> Self {
        match self {
            MoveVector::S(n) => MoveVector::E(Self::movelen_lim() + 1 - n),
            MoveVector::E(n) => MoveVector::S(*n),
        }
    }

    pub fn rotate_from_i64(&self, k: i64) -> Self {
        let k = k % 4;
        let k = if k < 0 { (4 + k) % 4 }
            else { k };
        match k {
            0 => self.clone(),
            1 => self.rotate90(),
            2 => self.rotate180(),
            3 => self.rotate270(),
            _ => panic!(),
        }
    }

    pub fn rotate_from_usize(&self, k: usize) -> Self {
        let k = k % 4;
        match k {
            0 => self.clone(),
            1 => self.rotate90(),
            2 => self.rotate180(),
            3 => self.rotate270(),
            _ => panic!()
        }
    }

    pub fn rotate_randomly(&self) -> Self {
        let mut rng = thread_rng();
        let k: usize = rng.gen_range(0..4);
        self.rotate_from_usize(k)
    }

    pub fn rotate(&self, rot: Rotation) -> Self {
        match rot {
            Rotation::No => self.clone(),
            Rotation::Do(k) => self.rotate_from_usize(k as usize),
            Rotation::Random => self.rotate_randomly(),
        }
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VectorBasedMove<B: BitBoard> {
    x: u8,
    y: u8,
    v: MoveVector<B>
}

pub fn board_coordinate_rotate<B: BitBoard>(x: u8, y: u8, k: u8) -> (u8, u8) {
    let movelen_lim = B::BOARD_SIZE - 1;
    let k = k % 4;
    match k {
        0 => (x, y),
        1 => (y, movelen_lim - x),
        2 => (movelen_lim - x, movelen_lim - y),
        3 => (movelen_lim - y, x),
        _ => panic!()
    }
}

pub trait MoveRepresentation<B: BitBoard> {
    fn convert_from(movement: &B::Movement) -> Result<Self> where Self: Sized;
    fn convert_into(&self) -> B::Movement;
}

impl<B: BitBoard> VectorBasedMove<B>{
    fn new(x: u8, y: u8, v: MoveVector<B>) -> Self {
        Self {x, y, v}
    }
}
impl<B: BitBoard> MoveRepresentation<B> for VectorBasedMove<B> {
    fn convert_from(movement: &B::Movement) -> Result<Self>{
        let start =  movement.start;
        let dst = movement.dst;
        let (x,y) = start.get_coordinate();
        let (xx, yy) = dst.get_coordinate();
        let (dx, dy) : (i64, i64) = (xx as i64 - x as i64, yy as i64 - y as i64);
        if dx != 0 && dy != 0 {
            return Err(ErrReport::msg(format!("couldn't convert MoveOnBoard {} to VectorBasedMove", movement)));
        }
        if dx == 0 && dy == 0 {
            return Err(ErrReport::msg("couldn't convert MoveOnBoard to VectorBasedMove because there was no movement"));
        }
        if dx == 0 {
            // That means dy is non-zero...
            let true_dy = if dy > 0 { dy } else { B::BOARD_SIZE as i64 + dy };
            Ok(Self { x, y, v: MoveVector<B>::S(true_dy.try_into().unwrap())})
        } else {
            // That means dy is zero..
            let true_dx = if dx > 0 { dx } else { B::BOARD_SIZE as i64 + dx };
            Ok(Self { x, y, v: MoveVector<B>::E(true_dx.try_into().unwrap())})
        }
    }

    fn convert_into(&self) -> B::Movement {
        let start = <B::Position as PositionalEncoding>::new(self.x, self.y);
        let board_len = B::BOARD_SIZE;
        let (x, y): (u8, u8) = match self.v {
            MoveVector::S(i) => {
                let offset = self.y;
                if offset + i < board_len {
                    (0, offset + i)
                } else {
                    (0, board_len - i)
                }
            }
            MoveVector::E(i) => {
                let offset = self.x;
                if offset + i < board_len {
                    (offset + i, 0)
                } else {
                    (board_len - i, 0)
                }
            }
        };
        let dst = <B::Position as PositionalEncoding>::new(x, y);
        <B::Movement as MoveOnBoard>::new(start, dst)

    }
}

impl<B: BitBoard> VectorBasedMove<B> {
    pub fn from_index(idx: i64) -> Self {
        assert!(idx >= 0);
        let n = B::BOARD_SIZE as i64;
        let i = idx / (n * n);
        let y = (idx % (n * n)) / n;
        let x = (idx % (n * n)) % n;
        Self { x: x.try_into().unwrap(), y: y.try_into().unwrap(), v: MoveVector::<B>::from_index(i)}
    }
    
    pub fn to_index(&self) -> i64 {
        let n = B::BOARD_SIZE;
        ((self.x) as i64 + (self.y * n) as i64 + MoveVector::to_index(self.v) * (n * n) as i64).into()
    }

    pub fn rotate(&self, rot: Rotation) -> Self {
        match rot {
            Rotation::No => self.clone(),
            Rotation::Do(k) => self.rotate_from_usize(k as usize),
            Rotation::Random => self.rotate_randomly()
        }
    }

    pub fn rotate90(&self) -> Self {
        let (new_x, new_y) = board_coordinate_rotate::<B>(self.x, self.y, 1);
        let new_v = self.v.rotate90();
        Self { x: new_x, y: new_y, v: new_v }
    }
    pub fn rotate180(&self) -> Self {
        let (new_x, new_y) = board_coordinate_rotate::<B>(self.x, self.y, 2);
        let new_v = self.v.rotate180();
        Self { x: new_x, y: new_y, v: new_v }
    }
    pub fn rotate270(&self) -> Self {
        let (new_x, new_y) = board_coordinate_rotate::<B>(self.x, self.y, 3);
        let new_v = self.v.rotate270();
        Self { x: new_x, y: new_y, v: new_v }
    }
    pub fn rotate_from_i64(&self, k: i64) -> Self {
        let k = k % 4;
        let k = if k < 0 { k + 4 } else {k};
        match k {
            0 => self.clone(),
            1 => self.rotate90(),
            2 => self.rotate180(),
            3 => self.rotate270(),
            _ => panic!(),
        }
    }
    pub fn rotate_from_usize(&self, k: usize) -> Self {
        let k = k % 4;
        match k {
            0 => self.clone(),
            1 => self.rotate90(),
            2 => self.rotate180(),
            3 => self.rotate270(),
            _ => panic!(),
        }
    }
    pub fn rotate_randomly(&self) -> Self {
        let mut rng = thread_rng();
        let k: usize = rng.gen_range(0..4);
        self.rotate_from_usize(k)
    }
}

pub enum Rotation {
    Do(u8),
    Random,
    No
}

impl Default for Rotation {
    fn default() -> Self {
        Rotation::No
    }
}

impl Rotation {
    fn get_k(&self) -> u8 {
        match self {
            Rotation::Do(k) => k % 4,
            Rotation::No => 0,
            Rotation::Random => {
                let mut rng = thread_rng();
                rng.gen_range(0..4)
            }
        }
    }
}

#[cfg(feature = "torch")]
#[derive(Debug)]
pub struct TBoard<B: BitBoard>(tch::Tensor);

#[cfg(feature = "torch")]
impl TBoard<BoardEleven> {
    pub fn new(ts: tch::Tensor) -> Self {
        Self(ts)
    }

    pub fn get_ref(&self) -> &Tensor {
        &self.0
    }

    pub fn get(self) -> Tensor {
        self.0
    }

    pub fn get_mut(&mut self) -> &mut Tensor {
        &mut self.0
    }

    pub fn from_bitboard(b: &BoardEleven, options: (tch::Kind, tch::Device)) -> Self {
        use std::array;

        use bitboard::eleven::BLOCKLEN;

        let ts = Tensor::zeros([11 * 11], options);
        let mut array: [f32; 11 * 11] = [0.0; 121];
        for i in 0..11 {
            for j in 0..11 {

                let block_id = match j {
                    0..4 => 0,
                    4..8 => 1,
                    8..11 => 2,
                    _ => panic!()
                };
                let offset = match block_id {
                    0 => 0,
                    1 => BLOCKLEN[0],
                    2 => BLOCKLEN[0] + BLOCKLEN[1],
                    _ => panic!()
                };
                let address = 12 * j + i - offset;
                let par = match block_id {
                    0 => b.par1,
                    1 => b.par2,
                    2 => b.par3,
                    _ => panic!()
                };
                // arr[(11 * j + i) as usize] = ((par >> address) & 1).into();
                array[(11 * j + i) as usize] = (((par >> address) & 1) as u8).into();

            }
        }

        // }
        let ts = Tensor::from_slice::<f32>(&array);
        let tts = ts.view([11, 11]);
        TBoardEleven(tts)
    }

    pub fn from_game_board(b: &TaflBoard<BoardEleven>, options: (tch::Kind, tch::Device)) -> Self {
        let ts1 = TBoard::<BoardEleven>::from_bitboard(&b.bit_att, options);
        let ts2 = TBoard::<BoardEleven>::from_bitboard(&b.bit_def, options);
        let ts3 = TBoard::<BoardEleven>::from_bitboard(&b.bit_king, options);
        let ts = Tensor::stack(&[ts1.0, ts2.0, ts3.0], 0);
        TBoard::<BoardEleven>(ts)
    }

    pub fn from_short_history(sh: &ShortHistory<BoardEleven>, options: (tch::Kind, tch::Device)) -> Self {
        let ts: [Tensor; 4] = std::array::from_fn(|i| {

            if let Some(b) = sh.get(i) {
                TBoard::<BoardEleven>::from_game_board(b, options).0
            } else {
                Tensor::zeros([3,11,11], options)
            }

        });
        let ts = Tensor::cat(&ts[..], 0);
        TBoard::<BoardEleven>(ts)
    }

    pub fn from_tot_move_count(g: &Game, options: (tch::Kind, tch::Device)) -> Self {
        let tot_move_count: i64 = g.state.get_turn_count().into();
        let tmc: Tensor = Tensor::scalar_tensor(tot_move_count, options);
        let tmc: Tensor = tmc.broadcast_to([11, 11]);
        TBoard::<BoardEleven>(tmc)
    }

    pub fn from_play_side(g: &Game, options: (tch::Kind, tch::Device)) -> Self {
        let side: i64 = match g.state.show_side() {
            Side::Att => 1,
            Side::Def => 0,
        };
        let side: Tensor = Tensor::scalar_tensor(side, options);
        let side: Tensor = side.broadcast_to([11, 11]);
        TBoard::<BoardEleven>(side)
    }

    pub fn from_repetition_count(g: &Game, options: (tch::Kind, tch::Device)) -> Self {
        let repetition_count: i64 = get_rep_counter_for_oldest(g.repetition_counter).into();
        let rc: Tensor = Tensor::scalar_tensor(repetition_count, options);
        let rc: Tensor = rc.broadcast_to([11, 11]);
        TBoard::<BoardEleven>(rc)
    }

    pub fn get_pnet_input(g: &Game, rot: Rotation, options: (tch::Kind, tch::Device)) -> Self {

        let short_history = TBoard::<BoardEleven>::from_short_history(&g.short_history, options);
        let rotated_short_history = short_history.0.rot90(rot.get_k().into(), [1,2]);
        // (3 * 4, 11, 11)

        let board = Self::from_game_board(&g.board, options);
        let rotated_board: Tensor = board.0.rot90(rot.get_k().into(), [1,2]);
        // (3, 11, 11)

        let tmc = TBoard::<BoardEleven>::from_tot_move_count(g, options).0.unsqueeze(0);
        // (1, 11, 11)
        let rc = TBoard::<BoardEleven>::from_repetition_count(g, options).0.unsqueeze(0);
        // (1, 11, 11)
        let side = TBoard::<BoardEleven>::from_play_side(g, options).0.unsqueeze(0);
        // (1, 11, 11)
        let ts = Tensor::cat(&[rotated_board, rotated_short_history, tmc, rc, side], 0);
        TBoard::<BoardEleven>(ts)
        
    }

    pub fn get_vnet_input(g: &Game, rot: Rotation, options: (tch::Kind, tch::Device)) -> Self {

        let short_history = TBoard::<BoardEleven>::from_short_history(&g.short_history, options);
        let rotated_short_history = short_history.0.rot90(rot.get_k().into(), [1,2]);
        // (3 * 4, 11, 11)

        let board = Self::from_game_board(&g.board, options);
        let rotated_board: Tensor = board.0.rot90(rot.get_k().into(), [1,2]);
        // (3, 11, 11)

        let tmc = TBoard::<BoardEleven>::from_tot_move_count(g, options).0.unsqueeze(0);
        // (1, 11, 11)
        let rc = TBoard::<BoardEleven>::from_repetition_count(g, options).0.unsqueeze(0);
        // (1, 11, 11)
        let ts = Tensor::cat(&[rotated_board, rotated_short_history, tmc, rc], 0);
        // (1, 11, 11)
        TBoard::<BoardEleven>(ts)
    }
}

impl TBoard<BoardSeven> {
    pub fn new(ts: tch::Tensor) -> Self {
        Self(ts)
    }

    pub fn get_ref(&self) -> &Tensor {
        &self.0
    }

    pub fn get(self) -> Tensor {
        self.0
    }

    pub fn get_mut(&mut self) -> &mut Tensor {
        &mut self.0
    }

    pub fn from_bitboard(b: &BoardSeven, options: (tch::Kind, tch::Device)) -> Self {
        use std::array;

        let ts = Tensor::zeros([7 * 7], options);
        let mut array: [f32; 7 * 7] = [0.0; 49];
        for i in 0..7 {
            for j in 0..7 {

                let address = SevenBoardPositionalEncoding::new(i, j).0;
                array[(7 * j + i) as usize] = (((b.0 >> address) & 1) as u8).into();
            }
        }

        let ts = Tensor::from_slice::<f32>(&array);
        let tts = ts.view([11, 11]);
        Self(tts)
    }

    pub fn from_game_board(b: &TaflBoard<BoardSeven>, options: (tch::Kind, tch::Device)) -> Self {
        let ts1 = TBoard::<BoardSeven>::from_bitboard(&b.bit_att, options);
        let ts2 = TBoard::<BoardSeven>::from_bitboard(&b.bit_def, options);
        let ts3 = TBoard::<BoardSeven>::from_bitboard(&b.bit_king, options);
        let ts = Tensor::stack(&[ts1.0, ts2.0, ts3.0], 0);
        Self(ts)
    }

    pub fn from_tot_move_count(g: &SimpleGame, options: (tch::Kind, tch::Device)) -> Self {
        let tot_move_count: i64 = g.state.get_turn_count().into();
        let tmc: Tensor = Tensor::scalar_tensor(tot_move_count, options);
        let tmc: Tensor = tmc.broadcast_to([11, 11]);
        Self(tmc)
    }

    pub fn from_play_side(g: &SimpleGame, options: (tch::Kind, tch::Device)) -> Self {
        let side: i64 = match g.state.show_side() {
            Side::Att => 1,
            Side::Def => 0,
        };
        let side: Tensor = Tensor::scalar_tensor(side, options);
        let side: Tensor = side.broadcast_to([11, 11]);
        Self(side)
    }

    // output shape is (5, 7, 7)
    pub fn get_pnet_input(g: &SimpleGame, rot: Rotation, options: (tch::Kind, tch::Device)) -> Self {

        let board = Self::from_game_board(&g.board, options);
        let rotated_board: Tensor = board.0.rot90(rot.get_k().into(), [1,2]);
        // (3, 7, 7)

        let tmc = TBoard::<BoardEleven>::from_tot_move_count(g, options).0.unsqueeze(0);
        // (1, 7, 7)
        let side = TBoard::<BoardEleven>::from_play_side(g, options).0.unsqueeze(0);
        // (1, 7, 7)
        let ts = Tensor::cat(&[rotated_board, tmc, side], 0);
        Self(ts)
    }

    pub fn get_vnet_input(g: &SimpleGame, rot: Rotation, options: (tch::Kind, tch::Device)) -> Self {

        let board = Self::from_game_board(&g.board, options);
        let rotated_board: Tensor = board.0.rot90(rot.get_k().into(), [1,2]);
        // (3, 7, 7)

        let tmc = TBoard::<BoardEleven>::from_tot_move_count(g, options).0.unsqueeze(0);
        // (1, 7, 7)
        let ts = Tensor::cat(&[rotated_board, tmc], 0);
        Self(ts)
    }

}

pub struct TAction<B: BitBoard>(Tensor);

impl<B: BitBoard> TAction<B> {
    pub fn get(&self) -> &Tensor {
        &self.0
    }

    pub fn inner(self) -> Tensor {
        self.0
    }

    // returns the actions indicated by the non-zero indices in self. 
    // will this work for Boolean Tensor ??
    pub fn to_vbm(&self) -> Vec<VectorBasedMove<B>> {
        let mut vbms: Vec<VectorBasedMove<B>> = vec![];
        let ts = self.0.flatten(0, -1);
        let nonzero_locations: Tensor = ts.nonzero().flatten(0, -1);
        // should be a Tensor of size [N]. e.g. [0, 1, 3, 6, ...]

        let vec = Vec::<i64>::try_from(nonzero_locations).unwrap();
        for idx in vec {
            vbms.push(VectorBasedMove::from_index(idx));
        }
        vbms
    }

    // take the actions with highest probabilities
    // Tensor is assumed to be of shape (bs, action_size, board_size, board_size)
    pub fn get_move_from(&self) -> Result<Vec<VectorBasedMove<B>>>{
        let ts = &self.0;
        let batch_size = ts.size()[0];
        let tts = ts.view([batch_size, -1]);
        let idxs = Tensor::zeros([batch_size], (tch::Kind::Int64, tts.device()));
        tts.argmax_out(&idxs, 1, false);
        println!("{:?}",idxs.kind());
        // size = (bs)
        debug_assert_eq!(idxs.size(), [batch_size]);
        let v: Vec<i64> = Vec::try_from(idxs)?;
        let v = v.into_iter().map(|idx| VectorBasedMove::<B>::from_index(idx)).collect();
        Ok(v)
    }

}

impl TAction<BoardEleven> {
    pub fn vbm_one_hot_encode(vbm: &VectorBasedMove<BoardEleven>) -> Self {
        let index = vbm.to_index();
        let mut arr: [i64; 11 * 11 * 20] = [0; 11 * 11 * 20];
        arr[index as usize] = 1;
        let ts = Tensor::from_slice(&arr);
        let ts = ts.view([20,11,11]);
        Self(ts)
    }
    pub fn vec_vbm_one_hot_encode(vbms: &Vec<VectorBasedMove<BoardEleven>>) -> Self {
        let mut arr: [i64; 11 * 11 * 20] = [0; 11 * 11 * 20];
        let indices = vbms.iter()
            .map(|a| a.to_index());
        for index in indices {
            arr[index as usize] = 1;
        }
        let ts = Tensor::from_slice(&arr[..]);
        let ts = ts.view([20,11,11]);
        debug_assert!(ts.kind() == Kind::Int64);
        Self(ts)
    }
    pub fn iter_vbm_one_hot_encode<T>(vbms: &T) -> Self
    where for<'a> &'a T: IntoIterator<Item = VectorBasedMove<BoardEleven>>,
    {
        let mut arr: [i64; 11 * 11 * 20] = [0; 11 * 11 * 20];
        let indices = vbms.into_iter()
            .map(|a| a.to_index());
        for index in indices {
            arr[index as usize] = 1;
        }
        let ts = Tensor::from_slice(&arr[..]);
        let ts = ts.view([20,11,11]);
        debug_assert!(ts.kind() == Kind::Int64);
        Self(ts)
    }
    pub fn vec_vbm_one_hot_encode_boolean(vbms: &Vec<VectorBasedMove<BoardEleven>>) -> Self {
        let mut arr: [bool; 11 * 11 * 20] = [false; 11 * 11 * 20];
        let indices = vbms.iter()
            .map(|a| a.to_index());
        for index in indices {
            arr[index as usize] = true;
        }
        let ts = Tensor::from_slice(&arr[..]);
        let ts = ts.view([20,11,11]);
        debug_assert!(ts.kind() == Kind::Bool);
        Self(ts)
    }
    pub fn from_index_policy(idx_policy: &IndexPolicy) -> Self {
        let mut slice = &mut [0.0f32; 20 * 11 * 11];
        for (idx, value) in idx_policy.get() {
            slice[*idx as usize] = *value;
        }
        let mut ts = Tensor::from_slice(slice);
        ts.view([20, 11, 11]);
        Self(ts)
    }
}

impl TAction<BoardSeven> {
    pub fn vbm_one_hot_encode(vbm: &VectorBasedMove<BoardSeven>) -> Self {
        let index = vbm.to_index();
        let mut arr: [i64; 7 * 7 * 12] = [0; 7 * 7 * 12];
        arr[index as usize] = 1;
        let ts = Tensor::from_slice(&arr);
        let ts = ts.view([12,7,7]);
        Self(ts)
    }
    pub fn vec_vbm_one_hot_encode(vbms: &Vec<VectorBasedMove<BoardEleven>>) -> Self {
        let mut arr: [i64; 7 * 7 * 12] = [0; 7 * 7 * 12];
        let indices = vbms.iter()
            .map(|a| a.to_index());
        for index in indices {
            arr[index as usize] = 1;
        }
        let ts = Tensor::from_slice(&arr[..]);
        let ts = ts.view([12,7,7]);
        debug_assert!(ts.kind() == Kind::Int64);
        Self(ts)
    }
    pub fn iter_vbm_one_hot_encode<T>(vbms: &T) -> Self
    where for<'a> &'a T: IntoIterator<Item = VectorBasedMove<BoardEleven>>,
    {
        let mut arr: [i64; 7 * 7 * 12] = [0; 7 * 7 * 12];
        let indices = vbms.into_iter()
            .map(|a| a.to_index());
        for index in indices {
            arr[index as usize] = 1;
        }
        let ts = Tensor::from_slice(&arr[..]);
        let ts = ts.view([12,7,7]);
        debug_assert!(ts.kind() == Kind::Int64);
        Self(ts)
    }
    pub fn vec_vbm_one_hot_encode_boolean(vbms: &Vec<VectorBasedMove<BoardEleven>>) -> Self {
        let mut arr: [bool; 7 * 7 * 12] = [false; 7 * 7 * 12];
        let indices = vbms.iter()
            .map(|a| a.to_index());
        for index in indices {
            arr[index as usize] = true;
        }
        let ts = Tensor::from_slice(&arr[..]);
        let ts = ts.view([12,7,7]);
        debug_assert!(ts.kind() == Kind::Bool);
        Self(ts)
    }
    pub fn from_index_policy(idx_policy: &IndexPolicy) -> Self {
        let mut slice = &mut [0.0f32; 12 * 7 * 7];
        for (idx, value) in idx_policy.get() {
            slice[*idx as usize] = *value;
        }
        let mut ts = Tensor::from_slice(slice);
        ts.view([12, 7, 7]);
        Self(ts)
    }
}


#[cfg(feature = "torch")]
// take the actions with highest probabilities
// Tensor is assumed to be of shape (bs, action_size, board_size, board_size)
pub fn get_move_from_tensor(ts: &Tensor) -> Result<Vec<VectorBasedMove>>{
    let batch_size = ts.size()[0];
    let tts = ts.view([batch_size, -1]);
    let idxs = Tensor::zeros([batch_size], (tch::Kind::Int64, tts.device()));
    tts.argmax_out(&idxs, 1, false);
    println!("{:?}",idxs.kind());
    // size = (bs)
    debug_assert_eq!(idxs.size(), [batch_size]);
    let v: Vec<i64> = Vec::try_from(idxs)?;
    let v = v.into_iter().map(|idx| VectorBasedMove::from_index(idx)).collect();
    Ok(v)
}

#[cfg(test)]
mod tests {

    // internal
    use game::board::TaflBoardEleven;
    use game::game::Game;
    use game::game::ShortHistory;
    use bitboard::Direction;
    use bitboard::eleven::{BoardEleven, ElevenBoardPositionalEncoding, MoveOnBoardEleven};
    use crate::utils::*;

    // external
    use color_eyre::eyre::{ErrReport, Result};
    use rand::prelude;

    #[cfg(feature = "torch")]
    use tch::Tensor;
    use tch::nn;
    use tch::Device;
    use tch::Kind;

    #[test]
    fn vbm_mbe_conversion_works() {
        let start = ElevenBoardPositionalEncoding::new(2,3);
        let dst = ElevenBoardPositionalEncoding::new(2, 6);
        let mbe: MoveOnBoardEleven = MoveOnBoardEleven { start, dst};
        let s = mbe.start.get_coordinate();
        let d = mbe.dst.get_coordinate();
        println!("{:?}", s);
        println!("{:?}", d);
        let vbm: VectorBasedMove = VectorBasedMove::convert_from(&mbe).unwrap();
        let intended_vbm = VectorBasedMove::new(2, 3, MoveVector::S(3));
        assert_eq!(vbm, intended_vbm);
    }

    #[cfg(feature = "torch")]
    #[test]
    fn get_move_from_tensor_works() {
        let ts = Tensor::rand([10, 20, 11, 11], (tch::Kind::Float, tch::Device::Cpu));
        let moves = get_move_from_tensor(&ts).unwrap();
        let moves: Vec<MoveOnBoardEleven> = moves.into_iter().map(|a| a.convert_into()).collect();
        println!("{:?}", moves);
    }

    #[cfg(feature = "torch")]
    #[test]
    fn tboard_eleven_from_bitboard_works() {
        let mut rng = thread_rng();
        let b = BoardEleven::random(&mut rng);
        println!("{}", b);
        let tb = TBoardEleven::from_bitboard(&b, (tch::Kind::Float, tch::Device::Cpu));
        println!("{:?}", tb);
        assert!(tb.0.is_contiguous());
    }

    #[cfg(feature = "torch")]
    #[test]
    fn tboard_eleven_from_taflboard_works() {
        let mut rng = thread_rng();
        let b = TaflBoardEleven::generate_random_board(&mut rng);
        println!("{}", b);
        let tb = TBoardEleven::from_game_board(&b, (tch::Kind::Float, tch::Device::Cpu));
        assert_eq!(tb.0.size(), [3,11,11]);
        assert!(tb.0.is_contiguous());
    }

    #[cfg(feature = "torch")]
    #[test]
    fn get_vnet_input_works() {
        use game::game::Game;
        let mut rng = thread_rng();
        let mut game = Game::default();
        let device = if tch::Cuda::is_available() { tch::Device::Cuda(0)} else {tch::Device::Cpu};
        let kind = tch::Kind::Float;
        let vnet_input = TBoardEleven::get_vnet_input(&game, Rotation::No, (kind, device));
        assert_eq!(vnet_input.0.size(), [17, 11, 11]);
        assert!(vnet_input.0.is_contiguous());
        println!("{:?}", vnet_input.0);
    }

    #[cfg(feature = "torch")]
    #[test]
    fn get_pnet_input_works() {
        use game::game::Game;
        let mut rng = thread_rng();
        let mut game = Game::default();
        let device = if tch::Cuda::is_available() { tch::Device::Cuda(0)} else {tch::Device::Cpu};
        let kind = tch::Kind::Float;
        let pnet_input = TBoardEleven::get_pnet_input(&game, Rotation::No, (kind, device));
        assert_eq!(pnet_input.0.size(), [18, 11, 11]);
        assert!(pnet_input.0.is_contiguous());
        println!("{:?}", pnet_input.0);
    }

    #[cfg(feature = "torch")]
    #[test]
    fn move_vector_rotation_fidelity_check() {
        let mv: MoveVector = MoveVector::E(3);
        let rotated90 = mv.rotate90();
        let rotated180 = mv.rotate180();
        let rotated270 = mv.rotate270();
        assert_eq!(rotated90.rotate90(), rotated180);
        assert_eq!(rotated90.rotate180(), rotated270);
        assert_eq!(rotated90.rotate270(), mv);
        assert_eq!(rotated180.rotate90(), rotated270);
        assert_eq!(rotated180.rotate180(), mv);
        assert_eq!(rotated270.rotate90(), mv);
        assert_eq!(rotated270.rotate180(), rotated90);
        assert_eq!(rotated270.rotate270(), rotated180);

    }

    #[cfg(feature = "torch")]
    #[test]
    fn tensor_action_translation_works() {
        let mut vec_vbm: Vec<VectorBasedMove> = vec![];
        vec_vbm.push(VectorBasedMove::new(2, 3, MoveVector::E(3)));
        vec_vbm.push(VectorBasedMove::new(10, 2, MoveVector::S(3)));
        vec_vbm.push(VectorBasedMove::new(0, 6, MoveVector::E(10)));
        let ta: TAction = TAction::vec_vbm_one_hot_encode(&vec_vbm);
        let translated_back: Vec<VectorBasedMove> = ta.to_vbm();
        println!("{:?}", translated_back);
        for vbm in &vec_vbm {
            assert!(translated_back.contains(vbm));
        }
        let bool_ta: TAction = TAction::vec_vbm_one_hot_encode_boolean(&vec_vbm);
        let translated_back_from_bool_ta: Vec<VectorBasedMove> = bool_ta.to_vbm();
        println!("{:?}", translated_back_from_bool_ta);
        for vbm in &vec_vbm {
            assert!(translated_back_from_bool_ta.contains(vbm));
        }

    }

    #[cfg(feature = "torch")]
    #[test]
    fn vbm_to_and_from_index_works() {
        let vbm: VectorBasedMove = VectorBasedMove::new(2, 3, MoveVector::E(3));
        let index = vbm.to_index();
        let vbm_back = VectorBasedMove::from_index(index);
        assert_eq!(vbm, vbm_back);
    }

    #[cfg(feature = "torch")]
    #[test]
    fn tensor_get_nonzero_works() {
        let vbm: VectorBasedMove = VectorBasedMove::new(2, 3, MoveVector::E(3));
        let index = vbm.to_index();
        let mut arr: [i64; 20*11*11] = [0i64; 20*11*11];
        arr[index as usize] = 1;
        let ts = Tensor::from_slice(&arr[..]);
        println!("{:?}", ts.size());
        let nonzero = ts.nonzero();
        // I don't know why, but size is [1,1]
        println!("{:?}", nonzero.size());
        let nonzero: Vec<i64> = Vec::<i64>::try_from(nonzero.flatten(0, -1)).unwrap();
        println!("{:?}", nonzero);
        assert_eq!(index, nonzero[0]);
    }

}

