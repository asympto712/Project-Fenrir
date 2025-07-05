#![allow(unused)]
// internal
use game::board::TaflBoardEleven;
use game::game::Game;
use game::game::ShortHistory;
use game::game::get_rep_counter_for_oldest;
use game::game::Side;
use bitboard::Direction;
use bitboard::eleven::{BoardEleven, ElevenBoardPositionalEncoding, MoveOnBoardEleven};

// external
use color_eyre::eyre::{ErrReport, Result};
use rand::rng;

#[cfg(feature = "torch")]
use tch::Tensor;
#[cfg(feature = "torch")]
use libc::c_void;
// #[cfg(feature = "torch")]
// use tch::Tensor::C_Tensor;

// Directional move. Assumes Periodical Boundary
#[derive(Copy, Clone, Debug, PartialEq)]
enum MoveVector {
    S(u8),
    W(u8),
} 

impl MoveVector {
    pub fn from_index(i: i64) -> Self {
        assert!(0 <= i && i < 20);
        if i < 10 { Self::S((i + 1).try_into().unwrap()) }
        else { Self::W( (i + 1 - 10).try_into().unwrap())}
    }

    pub fn to_index(self) -> i64 {
        match self {
            MoveVector::S(i) => (i - 1).into(),
            MoveVector::W(i) => (i + 10 - 1).into()
        }
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
struct VectorBasedMove {
    x: u8,
    y: u8,
    v: MoveVector
}

pub trait MoveRepresentation {
    fn convert_from(mbe: &MoveOnBoardEleven) -> Result<Self> where Self: Sized;
    fn convert_into(&self) -> MoveOnBoardEleven;
}

impl VectorBasedMove {
    fn new(x: u8, y: u8, v: MoveVector) -> Self {
        Self {x, y, v}
    }
}
impl MoveRepresentation for VectorBasedMove {
    fn convert_from(mbe: &MoveOnBoardEleven) -> Result<Self>{
        let start =  mbe.start;
        let dst = mbe.dst;
        let (x,y) = start.get_coordinate();
        let (xx, yy) = dst.get_coordinate();
        let (dx, dy) : (i64, i64) = (xx as i64 - x as i64, yy as i64 - y as i64);
        if dx != 0 && dy != 0 {
            return Err(ErrReport::msg(format!("couldn't convert MoveOnBoardEleven {} to VectorBasedMove", mbe)));
        }
        if dx == 0 && dy == 0 {
            return Err(ErrReport::msg("couldn't convert MoveOnBoardEleven to VectorBasedMove because there was no movement"));
        }
        if dx == 0 {
            // That means dy is non-zero...
            let true_dy = if dy > 0 { dy } else { 11 + dy };
            Ok(Self { x, y, v: MoveVector::S(true_dy.try_into().unwrap())})
        } else {
            // That means dy is zero..
            let true_dx = if dx > 0 { dx } else { 11 + dx };
            Ok(Self { x, y, v: MoveVector::W(true_dx.try_into().unwrap())})
        }
    }

    fn convert_into(&self) -> MoveOnBoardEleven {
        let start = ElevenBoardPositionalEncoding::new(self.x, self.y);
        let board_len = 11;
        let (x, y): (u8, u8) = match self.v {
            MoveVector::S(i) => {
                let offset = self.y;
                if offset + i < board_len {
                    (0, offset + i)
                } else {
                    (0, board_len - i)
                }
            }
            MoveVector::W(i) => {
                let offset = self.x;
                if offset + i < board_len {
                    (offset + i, 0)
                } else {
                    (board_len - i, 0)
                }
            }
        };
        let dst = ElevenBoardPositionalEncoding::new(x, y);
        MoveOnBoardEleven{ start, dst }

    }
}

impl VectorBasedMove {
    pub fn from_index(idx: i64) -> Self {
        assert!(idx >= 0);
        let i = idx / 121;
        let y = (idx % 121) / 11;
        let x = (idx % 121) % 11;
        Self { x: x.try_into().unwrap(), y: y.try_into().unwrap(), v: MoveVector::from_index(i)}
    }
    
    pub fn to_index(&self) -> i64 {
        ((self.x) as i64 + (self.y * 11) as i64 + MoveVector::to_index(self.v) * 121).into()
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
                rand::random_range(0..4)
            }
        }
    }
}

#[cfg(feature = "torch")]
#[derive(Debug)]
pub struct TBoardEleven(tch::Tensor);

#[cfg(feature = "torch")]
impl TBoardEleven {
    pub fn new(ts: tch::Tensor) -> Self {
        Self(ts)
    }

    pub fn from_bitboard(b: &BoardEleven, options: (tch::Kind, tch::Device)) -> Self {
        use std::array;

        use bitboard::eleven::BLOCKLEN;

        let ts = Tensor::zeros([11 * 11], options);
        let mut array: [f32; 11 * 11] = [0.0; 121];
        // unsafe {

            // let ptr = ts.data_ptr();
            // let arr: &mut [c_void] = std::slice::from_raw_parts_mut::<c_void>(ptr, ts.size()[0] as usize);
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

    pub fn from_game_board(b: &TaflBoardEleven, options: (tch::Kind, tch::Device)) -> Self {
        let ts1 = TBoardEleven::from_bitboard(&b.bit_att, options);
        let ts2 = TBoardEleven::from_bitboard(&b.bit_def, options);
        let ts3 = TBoardEleven::from_bitboard(&b.bit_king, options);
        let ts = Tensor::stack(&[ts1.0, ts2.0, ts3.0], 0);
        TBoardEleven(ts)
    }

    pub fn from_short_history(sh: &ShortHistory, options: (tch::Kind, tch::Device)) -> Self {
        let ts: [Tensor; 4] = std::array::from_fn(|i| {

            if let Some(b) = sh.get(i) {
                TBoardEleven::from_game_board(b, options).0
            } else {
                Tensor::zeros([3,11,11], options)
            }

        });
        let ts = Tensor::cat(&ts[..], 0);
        TBoardEleven(ts)
    }

    pub fn from_tot_move_count(g: &Game, options: (tch::Kind, tch::Device)) -> Self {
        let tot_move_count: i64 = g.state.get_turn_count().into();
        let tmc: Tensor = Tensor::scalar_tensor(tot_move_count, options);
        let tmc: Tensor = tmc.broadcast_to([11, 11]);
        TBoardEleven(tmc)
    }

    pub fn from_play_side(g: &Game, options: (tch::Kind, tch::Device)) -> Self {
        let side: i64 = match g.state.show_side() {
            Side::Att => 1,
            Side::Def => 0,
        };
        let side: Tensor = Tensor::scalar_tensor(side, options);
        let side: Tensor = side.broadcast_to([11, 11]);
        TBoardEleven(side)
    }

    pub fn from_repetition_count(g: &Game, options: (tch::Kind, tch::Device)) -> Self {
        let repetition_count: i64 = get_rep_counter_for_oldest(g.repetition_counter).into();
        let rc: Tensor = Tensor::scalar_tensor(repetition_count, options);
        let rc: Tensor = rc.broadcast_to([11, 11]);
        TBoardEleven(rc)
    }

    pub fn get_pnet_input(g: &Game, rot: Rotation, options: (tch::Kind, tch::Device)) -> Self {

        let short_history = TBoardEleven::from_short_history(&g.short_history, options);
        let rotated_short_history = short_history.0.rot90(rot.get_k().into(), [1,2]);
        // (3 * 4, 11, 11)

        let board = Self::from_game_board(&g.board, options);
        let rotated_board: Tensor = board.0.rot90(rot.get_k().into(), [1,2]);
        // (3, 11, 11)

        let tmc = TBoardEleven::from_tot_move_count(g, options).0.unsqueeze(0);
        // (1, 11, 11)
        let rc = TBoardEleven::from_repetition_count(g, options).0.unsqueeze(0);
        // (1, 11, 11)
        let side = TBoardEleven::from_play_side(g, options).0.unsqueeze(0);
        // (1, 11, 11)
        let ts = Tensor::cat(&[rotated_board, rotated_short_history, tmc, rc, side], 0);
        TBoardEleven(ts)
        
    }

    pub fn get_vnet_input(g: &Game, rot: Rotation, options: (tch::Kind, tch::Device)) -> Self {

        let short_history = TBoardEleven::from_short_history(&g.short_history, options);
        let rotated_short_history = short_history.0.rot90(rot.get_k().into(), [1,2]);
        // (3 * 4, 11, 11)

        let board = Self::from_game_board(&g.board, options);
        let rotated_board: Tensor = board.0.rot90(rot.get_k().into(), [1,2]);
        // (3, 11, 11)

        let tmc = TBoardEleven::from_tot_move_count(g, options).0.unsqueeze(0);
        // (1, 11, 11)
        let rc = TBoardEleven::from_repetition_count(g, options).0.unsqueeze(0);
        // (1, 11, 11)
        let ts = Tensor::cat(&[rotated_board, rotated_short_history, tmc, rc], 0);
        // (1, 11, 11)
        TBoardEleven(ts)
    }
}


#[cfg(feature = "torch")]
// take the actions with highest probabilities
// Tensor is assumed to be of shape (bs, action_size, board_size, board_size)
pub fn get_move_from_tensor(ts: &Tensor) -> Result<Vec<VectorBasedMove>>{
    let batch_size = ts.size()[0];
    let tts = ts.view([batch_size, -1]);
    let idxs = Tensor::zeros(tts.size(), (tch::Kind::Int64, tts.device()));
    tts.argmax_out(&idxs, 1, false);
    println!("{:?}",idxs.kind());
    // size = (bs)
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
    use rand::rng;

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
        let mut rng = rng();
        let b = BoardEleven::random(&mut rng);
        println!("{}", b);
        let tb = TBoardEleven::from_bitboard(&b, (tch::Kind::Float, tch::Device::Cpu));
        println!("{:?}", tb);
        assert!(tb.0.is_contiguous());
    }

    #[cfg(feature = "torch")]
    #[test]
    fn tboard_eleven_from_taflboard_works() {
        let mut rng = rng();
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
        let mut rng = rng();
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
        let mut rng = rng();
        let mut game = Game::default();
        let device = if tch::Cuda::is_available() { tch::Device::Cuda(0)} else {tch::Device::Cpu};
        let kind = tch::Kind::Float;
        let pnet_input = TBoardEleven::get_pnet_input(&game, Rotation::No, (kind, device));
        assert_eq!(pnet_input.0.size(), [18, 11, 11]);
        assert!(pnet_input.0.is_contiguous());
        println!("{:?}", pnet_input.0);
    }
}

