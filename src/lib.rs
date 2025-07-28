use rand::prelude::*;
use bitflags::bitflags;
use std::{fmt::Display, ops::*};

pub mod eleven;
pub mod seven;

pub enum Direction{
    E(usize),
    W(usize),
    S(usize),
    N(usize),
    All(usize),
}

pub trait PositionalEncoding:
    TryFrom<u8> +
    TryFrom<String> +
    Display
{
    const BOARD_SIZE: u8;
    fn new(x: u8, y: u8) -> Self;
    fn get_coordinate(&self) -> (u8, u8);
}

pub trait MoveOnBoard:
    Display +
    TryFrom<String>
{
    type Position: PositionalEncoding;
    fn new(start: Self::Position, dst: Self::Position) -> Self;
    fn find_path(&self) -> Option<Direction>;
    fn start(&self) -> Self::Position;
    fn dst(&self) -> Self::Position;
}
pub trait Shift{
    fn shift_w(&self) -> Self;
    fn shift_e(&self) -> Self;
    fn shift_n(&self) -> Self;
    fn shift_s(&self) -> Self;
}

pub trait BitBoard:
    PartialEq +
    Copy +
    Default + 
    TryFrom<String> +
    BitAnd<Output = Self> +
    BitAndAssign +
    BitOr<Output = Self> +
    BitOrAssign +
    BitXor<Output = Self> + 
    BitXorAssign + 
    Not<Output = Self> +
    Display +
    Shift
{
    type Position: PositionalEncoding;
    type Movement: MoveOnBoard::<Position = Self::Position>;
    const BLACKMASK: Self;
    const ONLY_PAD: Self;
    const ONLY_EDGES: Self;
    const CORNERS: Self;

    // required method
    fn new() -> Self;
    fn empty() -> Self;
    fn random<T: Rng>(rng: &mut T) -> Self;
    fn ghastly() -> Self;
    fn is_nonzero(&self) -> bool;
    fn move_piece(&mut self, movement: Self::Movement);
    fn shift_e_without_padding_reset(&self) -> Self;
    fn shift_w_without_padding_reset(&self) -> Self;
    fn locate_ones(&self) -> Vec<Self::Position>;
    fn neighbor_of(location: &Self::Position) -> Self;
    fn generate_action(&self, mask: &Self) -> Vec<Self::Movement>;
    fn flip_target_bit(&self, position: &Self::Position) -> Self;
    fn force_move(&self, movement: &Self::Movement) -> Self;
    fn flip_target_bit_mut(&mut self, position: &Self::Position);
    fn tile_is_empty_at(&self, position: &Self::Position) -> bool;

    // provided method
    fn mask_board() -> Self{
        Self::BLACKMASK
    }
    fn only_pad() -> Self{
        Self::ONLY_PAD
    }
    fn reset_padding(&self) -> Self{
        *self & Self::BLACKMASK
    }
    fn dilation(&self, d: Direction) -> Self{
        let mut tmp = *self;
        match d{
            Direction::E(u) => {
                for _ in 0..u{
                    tmp = tmp | tmp.shift_e();
                }
                tmp
            },
            Direction::W(u) => {
                for _ in 0..u{
                    tmp = tmp | tmp.shift_w();
                }
                tmp
            },
            Direction::N(u) => {
                for _ in 0..u{
                    tmp = tmp | tmp.shift_n();
                }
                tmp
            },
            Direction::S(u) => {
                for _ in 0..u{
                    tmp = tmp | tmp.shift_s();
                }
                tmp
            },
            Direction::All(u) => {
                for _ in 0..u{
                    tmp = tmp | tmp.shift_e() | tmp.shift_n() | tmp.shift_s() | tmp.shift_w();
                }
                tmp
            }
        }
    }
    fn erosion(&self, d: Direction) -> Self{
        let mut tmp = *self;
        match d{
            Direction::E(u) => {
                for _ in 0..u{
                    tmp = tmp & tmp.shift_e();
                }
                tmp
            },
            Direction::W(u) => {
                for _ in 0..u{
                    tmp = tmp & tmp.shift_w();
                }
                tmp
            },
            Direction::N(u) => {
                for _ in 0..u{
                    tmp = tmp & tmp.shift_n();
                }
                tmp
            },
            Direction::S(u) => {
                for _ in 0..u{
                    tmp = tmp & tmp.shift_s();
                }
                tmp
            },
            Direction::All(u) => {
                for _ in 0..u{
                    tmp = tmp & tmp.shift_e() & tmp.shift_n() & tmp.shift_s() & tmp.shift_w();
                }
                tmp
            }
        }
    }
    fn equals(&self, rhs: &Self) -> bool{
        let tmp = *self ^ *rhs;
        !tmp.is_nonzero()
    }
    // flip the bits on the board, but keep the paddings to zero. Make sure to only use this to boards whose padding are reset.
    // Otherwise it might cause unexpected behavior. When in doubt, .reset_padding()
    fn complement(&self) -> Self{
        *self ^ Self::mask_board()
    }
    fn shift_e_with_step(&self, step: u8) -> Self{
        let mut tmp = *self;
        for _ in 0..step{
            tmp = tmp.shift_e();
        }
        tmp
    }
    fn shift_w_with_step(&self, step: u8) -> Self{
        let mut tmp = *self;
        for _ in 0..step{
            tmp = tmp.shift_w();
        }
        tmp
    }
    fn shift_n_with_step(&self, step: u8) -> Self{
        let mut tmp = *self;
        for _ in 0..step{
            tmp = tmp.shift_n();
        }
        tmp
    }
    fn shift_s_with_step(&self, step: u8) -> Self{ 
        let mut tmp = *self;
        for _ in 0..step{
            tmp = tmp.shift_s();
        }
        tmp
    }
    fn list_moves_east(&self, mask: &Self, barricade: &Self, movelen_lim: u8) -> Vec<Self::Movement>{

        let mut move_list: Vec<Self::Movement> = Vec::new();

        let mut tmp = *self;
        for step in 1..=movelen_lim{
            // If a path starting from a certain position is blocked at some point, there is no going further.
            // Therefore after the book-keeping procedure we remove those blocked starting points from tmp.
            let blocked = tmp.shift_e() & *barricade;
            let masked = blocked & *mask;
            if masked.is_nonzero(){

                let dst_positions = masked.locate_ones();
                let start_positions = masked.shift_w_with_step(step).locate_ones();
                assert_eq!(dst_positions.len(), start_positions.len());
                for (start, dst) in start_positions.into_iter().zip(dst_positions){
                    move_list.push(Self::Movement::new(start, dst));
                }
            }
            tmp = blocked;
        }
        move_list
    }
    fn list_moves_west(&self, mask: &Self, barricade: &Self, movelen_lim: u8) -> Vec<Self::Movement>{

        let mut move_list: Vec<Self::Movement> = Vec::new();

        let mut tmp = *self;
        for step in 1..=movelen_lim{
            // If a path starting from a certain position is blocked at some point, there is no going further.
            // Therefore after the book-keeping procedure we remove those blocked starting points from tmp.
            let blocked = tmp.shift_w() & *barricade;
            let masked = blocked & *mask;
            if masked.is_nonzero(){

                let dst_positions = masked.locate_ones();
                let start_positions = masked.shift_e_with_step(step).locate_ones();
                assert_eq!(dst_positions.len(), start_positions.len());
                for (start, dst) in start_positions.into_iter().zip(dst_positions){
                    move_list.push(Self::Movement::new(start, dst));
                }
            }
            tmp = blocked;
        }
        move_list
    }
    fn list_moves_south(&self, mask: &Self, barricade: &Self, movelen_lim: u8) -> Vec<Self::Movement>{

        let mut move_list: Vec<Self::Movement> = Vec::new();

        let mut tmp = *self;
        for step in 1..=movelen_lim{
            // If a path starting from a certain position is blocked at some point, there is no going further.
            // Therefore after the book-keeping procedure we remove those blocked starting points from tmp.
            let blocked = tmp.shift_s() & *barricade;
            let masked = blocked & *mask;
            if masked.is_nonzero(){

                let dst_positions = masked.locate_ones();
                let start_positions = masked.shift_n_with_step(step).locate_ones();
                assert_eq!(dst_positions.len(), start_positions.len());
                for (start, dst) in start_positions.into_iter().zip(dst_positions){
                    move_list.push(Self::Movement::new(start, dst));
                }
            }
            tmp = blocked;
        }
        move_list

    }
    fn list_moves_north(&self, mask: &Self, barricade: &Self, movelen_lim: u8) -> Vec<Self::Movement>{

        let mut move_list: Vec<Self::Movement> = Vec::new();

        let mut tmp = *self;
        for step in 1..=movelen_lim{
            // If a path starting from a certain position is blocked at some point, there is no going further.
            // Therefore after the book-keeping procedure we remove those blocked starting points from tmp.
            let blocked = tmp.shift_n() & *barricade;
            let masked = blocked & *mask;
            if masked.is_nonzero(){

                let dst_positions = masked.locate_ones();
                let start_positions = masked.shift_s_with_step(step).locate_ones();
                assert_eq!(dst_positions.len(), start_positions.len());
                for (start, dst) in start_positions.into_iter().zip(dst_positions){
                    move_list.push(Self::Movement::new(start, dst));
                }
            }
            tmp = blocked;
        }
        move_list

    }

    fn force_move_mut(&mut self, movement: &Self::Movement){
        self.flip_target_bit_mut(&movement.start());
        self.flip_target_bit_mut(&movement.dst());
    }

    fn list_horizontal_pincer(&self, barricade: &Self) -> Vec<Self::Position> {
        // Make sure the barricade is padding-reset
        let shifted_e = self.shift_e() & *barricade;
        let survivor_e = shifted_e.shift_w();
        let shifted_w = self.shift_w() & *barricade;
        let survivor_w = shifted_w.shift_e();
        (survivor_e & survivor_w).locate_ones()

    }

    fn list_vertical_pincer(&self, barricade: &Self) -> Vec<Self::Position> {

        let shifted_s = self.shift_s() & *barricade;
        let survivor_s = shifted_s.shift_n();
        let shifted_n = self.shift_n() & *barricade;
        let survivor_n = shifted_n.shift_s();
        (survivor_n & survivor_s).locate_ones()
    }

    fn list_besieged(&self, barricade: &Self) -> Vec<Self::Position> {

        let shifted_s = self.shift_s() & *barricade;
        let survivor_s = shifted_s.shift_n();
        let shifted_n = self.shift_n() & *barricade;
        let survivor_n = shifted_n.shift_s();
        let shifted_e = self.shift_e() & *barricade;
        let survivor_e = shifted_e.shift_w();
        let shifted_w = self.shift_w() & *barricade;
        let survivor_w = shifted_w.shift_e();
        (survivor_s & survivor_n & survivor_w & survivor_e).locate_ones()
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PositionalInputErrorFlag: u8{
        const ZERO = 0;
        const ILLEGALFORMAT = 1;
        const OUTOFBOUNDS = 1 << 1;
    }
}