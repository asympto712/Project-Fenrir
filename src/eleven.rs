#![allow(dead_code)]
#![allow(clippy::unusual_byte_groupings)]
use rand::prelude::*;
use std::{fmt::Display, ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not}};
use crate::{Shift, Direction, PositionalEncoding, MoveOnBoard, BitBoard};

use rawbytes::RawBytes;
use bincode;

pub const BOARDELEVENPART1: usize = 48;
pub const BOARDELEVENPART2: usize = 96;
pub const BOARDELEVENPART3: usize = 132;

// These denote the number of bits that blocks occupy INCLUDING the padding
const BLOCKLEN: [u8;3] = [BLOCK1LEN, BLOCK2LEN, BLOCK3LEN];
const BLOCK1LEN: u8 = 48;
const BLOCK2LEN: u8 = 48;
const BLOCK3LEN: u8 = 36;

#[allow(dead_code)]
const PRESET1: &str = "10001010110\n00100111101\n00100111111\n00001111100\n11110011010\n00110101101\n11000011011\n11000000111\n00000000000\n00011111111\n00011001111\n";
const PRESET_MASK: &str = "11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n";

pub const PRESETX1: [u64;3] = [0x4aa5552aa555, 0x7ff7ff7ff000, 0x7ff7ff7ff];
pub const PRESETX_EMPTY: [u64;3] = [0x000000000000, 0x000000000000, 0x000000000];
pub const PRESETX_BACKGROUND: [u64;3] = [0x7ff7ff7ff7ff, 0x7ff7ff7ff7ff, 0x7ff7ff7ff];


const fn num_to_alphabet(n: u8) -> char{
    match n{
        0 => 'A',
        1 => 'B',
        2 => 'C',
        3 => 'D',
        4 => 'E',
        5 => 'F',
        6 => 'G',
        7 => 'H',
        8 => 'I',
        9 => 'J',
        10 => 'K',
        11 => 'L',
        _ => 'Z',
    }
}

// The least significant 2 digits encodes the index of board partition. The remaining 6 digits (2^6 = 64) encodes the position 
// on the specified partition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElevenBoardPositionalEncoding(pub u8);

impl PositionalEncoding for ElevenBoardPositionalEncoding{

    const BOARD_SIZE: u8 = 11;

    fn new(x: u8,y: u8) -> Self{
        let block: u8 = match y{
            0..4 => 0,
            4..8 => 1,
            8..11 => 2,
            _ => {panic!("y-coordinate out of bound")},
        };
        let new_y = y - block * 4;
        assert!(x < 11, "x-coordinate out of bound");
        Self(((new_y * 12 + x) << 2) + block)
    }

    // obtain the coordinate in the screen-like coordinate system (north left: origin) 
    // returning (x,y), where x ranges 0..11 and y ranges 0..11
    fn get_coordinate(&self) -> (u8,u8) {
        let block: u8 = match self.0 & 3 {
            0 => 0,
            1 => 4,
            2 => 8,
            _ => panic!("unexpected board partition")
        };
        let address: u8 = self.0 >> 2;
        let y: u8 = address / 12 + block;
        let x: u8 = address % 12;
        (x,y)
    }
}

impl TryFrom<u8> for ElevenBoardPositionalEncoding{
    type Error = String;
    fn try_from(n: u8) -> Result<Self, Self::Error> {
        if n >= 121 { return Err("input address out of bounds".to_string())}
        Ok(Self::new(n % 11, n / 11))
    }
}

impl TryFrom<&[char]> for ElevenBoardPositionalEncoding{
    type Error = String;
    fn try_from(value: &[char]) -> Result<Self, Self::Error> {
        
        fn int_from_char(value: char) -> Option<u8>{
            if value.is_ascii_digit(){
                Some( (value as u32 - 0x30) as u8 )
            } else { None }
        }
        fn int_from_char_slice(value: &[char]) -> Option<u8>{
            match value.len(){
                1 => {
                    let n = int_from_char(value[0])?;
                    if n == 0 { None }
                    else { Some(n - 1) }
                },
                2 => {
                    let n1 = int_from_char(value[0])?;
                    let n2 = int_from_char(value[1])?;
                    let n = n1 * 10 + n2;
                    if n == 0 { None }
                    else {Some(n - 1)}
                },
                _ => None
            }
        }
        fn int_from_ascii_lowercase(value: char) -> Option<u8>{
            if value.is_ascii_lowercase(){
                Some( (value as u32 - 0x61) as u8 )
            } else { None }
        }
        fn int_check_if_in_bound(value: u8) -> Result<u8, String>{
            if value < 11 { Ok(value)}
            else { Err("Input position out of bound".to_string())}
        }

        let msg1: &str = "Invalid input format";

        if value.len() > 3 {return Err(msg1.to_string())}

        let x: u8 = match int_from_ascii_lowercase(value[0]){
            None => { return Err(msg1.to_string()) },
            Some(n) => { int_check_if_in_bound(n)? }
        };

        let y: u8 = match int_from_char_slice(&value[1..]){
            None => { return Err(msg1.to_string())},
            Some(n) => { int_check_if_in_bound(n)? }
        };

        Ok(Self::new(x,y))
    }
}

impl TryFrom<String> for ElevenBoardPositionalEncoding{
    type Error = String;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        let chars: Vec<char> = value.to_lowercase().chars().collect();
        Self::try_from(&chars[..])
    }
}

impl std::fmt::Display for ElevenBoardPositionalEncoding{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let block = self.0 & 0b11 ;
        let position = self.0 >> 2;
        let (x,y): (u8,u8) = if block < 3{
            (position % 12, position / 12 + 4 * block)
        } else { panic!("invalid block index encountered")};
        write!(f, "{}{}", num_to_alphabet(x), y+1)?;
        Ok(())
    }
}
// represent a piece move by its starting point and the end point. First two bits specifies which partition the position is in, and the latter
// 6 bits (2^6 = 64) specifies the location of the target bit
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveOnBoardEleven {
    pub start: ElevenBoardPositionalEncoding, 
    pub dst: ElevenBoardPositionalEncoding,
}

impl Display for MoveOnBoardEleven{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.start, self.dst)?;
        Ok(())
    }
}

impl TryFrom<String> for MoveOnBoardEleven{
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let alphabet_positions: Vec<usize> = value.to_lowercase().chars().enumerate().filter_map(
            |(i,c)| c.is_ascii_lowercase().then_some(i)).collect();
        if alphabet_positions.len() != 2{
            Err("Wrong input pattern for movement: try e.g. A2D2".to_string())
        } else if alphabet_positions[1] - alphabet_positions[0] <= 1{
            return Err("Wrong input pattern for movement: try e.g. A2D2".to_string())
        } else {
            let chars: Vec<char> = value.to_lowercase().chars().collect();
            let start: ElevenBoardPositionalEncoding =
            ElevenBoardPositionalEncoding::try_from(&chars[alphabet_positions[0]..alphabet_positions[1]])?;

            let dst: ElevenBoardPositionalEncoding =
            ElevenBoardPositionalEncoding::try_from(&chars[alphabet_positions[1]..])?;
            return Ok(Self{start, dst})
        }
    }
}

impl MoveOnBoard for MoveOnBoardEleven{

    type Position = ElevenBoardPositionalEncoding;

    fn new(start: Self::Position, dst: Self::Position) -> Self {
        Self { start, dst }
    }

    fn start(&self) -> Self::Position {
        self.start
    }

    fn dst(&self) -> Self::Position {
        self.dst
    }

    // Find out how many steps to go in which direction to go from start to dst. This is not so trivial as locations are encoded. 
    // Consider using this as little as possible to reduce performance overhead.
    fn find_path(&self) -> Option<Direction>{
        let init = BoardEleven::new().flip_target_bit(&self.start);
        let target = BoardEleven::new().flip_target_bit(&self.dst);

        // First, try moving east
        let mut tmp = init;
        for i in 1..=10{
            tmp = tmp.shift_e();
            if (tmp & target).is_nonzero() { return Some(Direction::E(i))}
        } 

        // Next, try moving west
        let mut tmp = init;
        for i in 1..=10{
            tmp = tmp.shift_w();
            if (tmp & target).is_nonzero() { return Some(Direction::W(i))}
        } 
        
        // Next, try moving south
        let mut tmp = init;
        for i in 1..=10{
            tmp = tmp.shift_s();
            if (tmp & target).is_nonzero() { return Some(Direction::S(i))}
        } 


        // Next, try moving north
        let mut tmp = init;
        for i in 1..=10{
            tmp = tmp.shift_n();
            if (tmp & target).is_nonzero() { return Some(Direction::N(i))}
        } 

        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, bincode::Encode, bincode::Decode)]
#[repr(C)]
pub struct BoardEleven {
    pub par1: u64,
    pub par2: u64,
    pub par3: u64,
}


impl Default for BoardEleven {
    fn default() -> Self {
        BoardEleven::from(PRESETX1)
    }
}

impl From<[u64;3]> for BoardEleven{
    fn from(value: [u64;3]) -> Self {
        Self {par1: value[0], par2: value[1], par3: value[2]}
    }
}

impl BitAnd for BoardEleven{
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self{
            par1: self.par1 & rhs.par1,
            par2: self.par2 & rhs.par2,
            par3: self.par3 & rhs.par3,
        }
    }
}
impl BitAndAssign for BoardEleven{
    fn bitand_assign(&mut self, rhs: Self) {
        self.par1 &= rhs.par1;
        self.par2 &= rhs.par2;
        self.par3 &= rhs.par3;
    }
}

impl BitOr for BoardEleven{
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self{
            par1: self.par1 | rhs.par1,
            par2: self.par2 | rhs.par2,
            par3: self.par3 | rhs.par3,
        }
    }
}
impl BitOrAssign for BoardEleven{
    fn bitor_assign(&mut self, rhs: Self) {
        self.par1 |= rhs.par1;
        self.par2 |= rhs.par2;
        self.par3 |= rhs.par3;
    }
}
impl BitXor for BoardEleven{
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self{
            par1: self.par1 ^ rhs.par1,
            par2: self.par2 ^ rhs.par2,
            par3: self.par3 ^ rhs.par3,
        }
    }
}
impl BitXorAssign for BoardEleven{
    fn bitxor_assign(&mut self, rhs: Self) {
        self.par1 ^= rhs.par1;
        self.par2 ^= rhs.par2;
        self.par3 ^= rhs.par3;
    }
}


impl Display for BoardEleven {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut count: u8 = 0;
        fn extract_bits_and_draw(
            byte: u8,
            count: &mut u8,
            f: &mut std::fmt::Formatter<'_>,
            bound: usize,
        ) -> Result<bool, std::fmt::Error> {
            for i in 0..8 {

                if *count as usize == bound {
                    // *count += 1;
                    return Ok(true);
                }

                // Ignore the padding
                if *count % 12 == 11 { 
                    *count += 1; 
                    writeln!(f)?;
                    continue;
                }

                let shift: u8 = 1 << i;
                let r = (shift & byte) > 0;

                write!(f, "{:b} ", r as u8)?;
                
                // if *count % 11 == 10 {
                //     writeln!(f)?;
                // }
                
                *count += 1;
            }
            Ok(false)
        }
        fn draw(
            b: u64,
            count: &mut u8,
            f: &mut std::fmt::Formatter<'_>,
            bound: usize,
        ) -> Result<(), std::fmt::Error> {
            let bytes: &[u8] = RawBytes::bytes_view(&b);
            for byte in bytes {
                // println!("{:b}", byte);
                match extract_bits_and_draw(*byte, count, f, bound) {
                    Ok(false) => continue,
                    Ok(true) => return Ok(()),
                    Err(e) => return Err(e),
                };
            }
            Ok(())
        }
        draw(self.par1, &mut count, f, BOARDELEVENPART1)?;
        // println!("par1 printed");
        draw(self.par2, &mut count, f, BOARDELEVENPART2)?;
        // println!("par3 printed");
        draw(self.par3, &mut count, f, BOARDELEVENPART3)?;
        // println!("par3 printed");
        Ok(())
    }
}

impl TryFrom<String> for BoardEleven{
    type Error = String;
    fn try_from(value: String) -> Result<Self, Self::Error> {

        enum PieceOrNewline {
            Piece(u64),
            Newline,
        }

        fn inspect(c: char) -> Result<PieceOrNewline, String>{
            // print!("{c}");
            if c == '1' { Ok(PieceOrNewline::Piece(1)) }
            else if c == '0' { Ok(PieceOrNewline::Piece(0)) }
            else if c == '\n' { Ok(PieceOrNewline::Newline)}
            else { Err("invalid character encountered".to_owned())}
        }

        fn store(parboard: &mut u64, shift: usize, c: char, count: &mut usize) -> Result<(),String>{
            if let PieceOrNewline::Piece(piece) = inspect(c)?{
                *parboard |= piece << shift;
                *count += 1;
            }
            Ok(())
        }
        let chars = value.chars();
        let mut par1: u64 = 0;
        let mut par2: u64 = 0;
        let mut par3: u64 = 0;

        let mut count: usize = 0;

        for c in chars{
            
            if count < BOARDELEVENPART1{
                if count % 12 == 11 { 
                    count += 1;
                }
                store(&mut par1, count, c, &mut count)?;
            
            }
            
            else if count < BOARDELEVENPART2{
                if count % 12 == 11 {
                    count += 1;
                }
                let i = count - BOARDELEVENPART1;
                store(&mut par2, i, c, &mut count)?;
        
            }
            
            else if count < BOARDELEVENPART3{
                if count % 12 == 11 {
                    count += 1;
                }
                let i = count - BOARDELEVENPART2;
                store(&mut par3, i, c, &mut count)?;
                
            }
        }
        Ok(BoardEleven { par1, par2, par3 })
    }
}

// This operation will flip all the bits, including the paddings. Only use this when you are sure that you don't want padding-reset.
// For flipping bits on the board only, use self.complement() instead. 
impl Not for BoardEleven{
    type Output = Self;
    fn not(self) -> Self::Output {
        let c: BoardEleven = Self{
            par1: !self.par1,
            par2: !self.par2,
            par3: !self.par3,
        };
        // let mask: BoardEleven = BoardEleven::mask_board();
        // c & mask
        c
    }
}
impl BitBoard for BoardEleven{
    
    const BLACKMASK: BoardEleven = BoardEleven{par1: 0x7ff7ff7ff7ff, par2: 0x7ff7ff7ff7ff, par3: 0x7ff7ff7ff};
    const ONLY_PAD: BoardEleven = BoardEleven{par1: 0xffff_800_800_800_800, par2: 0xffff_800_800_800_800, par3: 0xfffffff_800_800_800};
    const ONLY_EDGES: BoardEleven = BoardEleven{par1: 0x401_401_401_7ff, par2: 0x401_401_401_401, par3: 0x7ff_401_401};
    const CORNERS: BoardEleven = BoardEleven{par1: 0x000_000_000_401, par2: 0x000_000_000_000, par3: 0x401_000_000};

    type Movement = MoveOnBoardEleven;
    type Position = ElevenBoardPositionalEncoding;

    fn new() -> Self {
        Self { par1: 0, par2: 0, par3: 0 }
    }

    fn random<T>(rng: &mut T) -> Self 
    where T: Rng
    {
        let tmp = Self{
            par1: rng.next_u64(),
            par2: rng.next_u64(),
            par3: rng.next_u64(),
        };
        tmp.reset_padding()
    }
    fn empty() -> Self {
        Self { par1: 0, par2: 0, par3: 0 }
    }

    // only for the creation of dummy data
    #[inline]
    fn ghastly() -> Self {
        Self { par1: 0, par2: 0, par3: 0 }
    }
    
    #[inline]
    fn is_nonzero(&self) -> bool{
        let b = self.reset_padding();
        b.par1 > 0 || b.par2 > 0 || b.par3 > 0
    }

    #[inline]
    fn move_piece(&mut self, movement: MoveOnBoardEleven){

        #[inline]
        fn flip_bit(n: &mut u64, loc: u8){
            let mask = 1u64 << loc;
            *n ^= mask;
        }
        let st_loc: u8 = movement.start.0 >> 2;
        match movement.start.0 & 3u8{
            0 => flip_bit(&mut self.par1, st_loc),
            1 => flip_bit(&mut self.par2, st_loc),
            2 => flip_bit(&mut self.par3, st_loc),
            _ => {},
        }

        let dst_loc: u8 = movement.dst.0 >> 2;
        match movement.dst.0 & 3u8{
            0 => flip_bit(&mut self.par1, dst_loc),
            1 => flip_bit(&mut self.par2, dst_loc),
            2 => flip_bit(&mut self.par3, dst_loc),
            _ => {},
        }
    }

    #[inline]
    fn shift_e_without_padding_reset(&self) -> Self {
        Self { par1: self.par1 << 1, par2: self.par2 << 1, par3: self.par3 << 1}
    }

    #[inline]
    fn shift_w_without_padding_reset(&self) -> Self {
        Self { par1: self.par1 >> 1, par2: self.par2 >> 1, par3: self.par3 >> 1 }
    }

    fn locate_ones(&self) -> Vec<ElevenBoardPositionalEncoding> {
        let mut value: Vec<ElevenBoardPositionalEncoding> = Vec::new();
        fn traverse_and_keep_ones(
        value: &mut Vec<ElevenBoardPositionalEncoding>,
        target: &u64,
        len: u8,
        skip_every: u8,
        block: u8){
            
            for i in 0..len{
                let tmp = target >> i;
                if (i + 1) % skip_every == 0{ continue }
                if tmp & 1 == 1 {
                    let position: u8 = block + (i << 2);
                    value.push(ElevenBoardPositionalEncoding(position));
                }
            }
        }
        traverse_and_keep_ones(&mut value, &self.par1, BLOCK1LEN, 12, 0);
        traverse_and_keep_ones(&mut value, &self.par2, BLOCK2LEN, 12, 1);
        traverse_and_keep_ones(&mut value, &self.par3, BLOCK3LEN, 12, 2);
        value
    }

    fn neighbor_of(location: &ElevenBoardPositionalEncoding) -> Self{
        let address = location.0 >> 2;
        let tmp = match location.0 & 3{
            0 => {
                Self {par1: 1 << address, par2: 0, par3: 0}
            },
            1 => { 
                Self {par1: 0, par2: 1 << address, par3: 0} 
            },
            2 => {
                Self {par1: 0, par2: 0, par3: 1 << address}
            },
            _ => panic!("Invalid positional encoding")
        };
        tmp | tmp.shift_e() | tmp.shift_w() | tmp.shift_s() | tmp.shift_n()
    }

    // Older implementation of a function that lists legal moves. "list_moves_{direction}" is more recommended.
    fn generate_action(&self, mask: &Self) -> Vec<MoveOnBoardEleven>{

        #[inline]
        fn calculate_start_position_for_move_south(u: u8, block: &mut u8, step: u8) -> u8{
            let mut v = u;
            let n: u8 = step * 12;
            while v < n{
                *block -= 1;
                v += BLOCKLEN[*block as usize];
            }
            (v - n) << 2 | *block
        }

        #[inline]
        fn calculate_start_position_for_move_north(u: u8, block: &mut u8, step: u8) -> u8{
            let n: u8 = step * 12;
            let mut v = u + n;
            while v > BLOCKLEN[*block as usize]{
                v -= BLOCKLEN[*block as usize];
                *block += 1;
            }
            v << 2 | *block
        }

        #[inline]
        fn calculate_start_position_for_move_east(u: u8, block: u8, step: u8) -> u8{
            (u - step) << 2 | block
        }

        #[inline]
        fn calculate_start_position_for_move_west(u: u8, block: u8, step: u8) -> u8{
            (u + step) << 2 | block
        }

        let mut moves: Vec<MoveOnBoardEleven> = Vec::new();

        // Moves to East are dealt with here
        let mut tmp = *self;
        for i in 1..=10{
            tmp = tmp.shift_e() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions[0].len() + positions[1].len() + positions[2].len() == 0 {
                break;
            }
            for (block, item) in positions.iter().enumerate(){
                for u in item{
                    let dst: u8 = u << 2 | block as u8;
                    let start: u8 = calculate_start_position_for_move_east(*u, block as u8, i);
                    moves.push(MoveOnBoardEleven{
                        start: ElevenBoardPositionalEncoding( start ), 
                        dst: ElevenBoardPositionalEncoding( dst )
                    });
                }
            }
        }

        // Moves to West are dealt with here
        let mut tmp = *self;
        for i in 1..=10{
            tmp = tmp.shift_w() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions[0].len() + positions[1].len() + positions[2].len() == 0 {
                break;
            }
            for (block, item) in positions.iter().enumerate(){
                for u in item{
                    let dst: u8 = u << 2 | block as u8;
                    let start: u8 = calculate_start_position_for_move_west(*u, block as u8, i);
                    moves.push(MoveOnBoardEleven{
                        start: ElevenBoardPositionalEncoding( start ), 
                        dst: ElevenBoardPositionalEncoding( dst )
                    });
                }
            }
        }

        // Moves to South are dealt with here
        let mut tmp = *self;
        for i in 1..=10{
            tmp = tmp.shift_s() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions[0].len() + positions[1].len() + positions[2].len() == 0 {
                break;
            }
            for (block, item) in positions.iter().enumerate(){
                for u in item{
                    let dst: u8 = u << 2 | block as u8;
                    let start: u8 = calculate_start_position_for_move_south(*u, &mut (block as u8), i);
                    moves.push(MoveOnBoardEleven{
                        start: ElevenBoardPositionalEncoding( start ), 
                        dst: ElevenBoardPositionalEncoding( dst )
                    });
                }
            }
        }

        // Moves to North are dealt with here
        let mut tmp = *self;
        for i in 1..=10{
            tmp = tmp.shift_s() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions[0].len() + positions[1].len() + positions[2].len() == 0 {
                break;
            }
            for (block, item) in positions.iter().enumerate(){
                for u in item{
                    let dst: u8 = u << 2 | block as u8;
                    let start: u8 = calculate_start_position_for_move_north(*u, &mut (block as u8), i);
                    moves.push(MoveOnBoardEleven{
                        start: ElevenBoardPositionalEncoding( start ), 
                        dst: ElevenBoardPositionalEncoding( dst )
                    });
                }
            }
        }

        moves
    }

    #[inline]
    fn flip_target_bit(&self, position: &ElevenBoardPositionalEncoding) -> Self{
        let block = position.0 & 3;
        let address = position.0 >> 2;
        match block {
            0 => {
                Self {par1: self.par1 ^ (1 << address), par2: self.par2, par3: self.par3 }
            },
            1 => {
                Self {par1: self.par1, par2: self.par2 ^ (1 << address), par3: self.par3 }
            },
            2 => {
                Self {par1: self.par1, par2: self.par2, par3: self.par3 ^ (1 << address) }
            },
            _ => panic!("specified position encode is invalid. Non-existent block was specified!")
        }
    }

    #[inline]
    fn force_move(&self, movement: &MoveOnBoardEleven) -> Self {
        let blocks = (movement.start.0 & 3, movement.dst.0 & 3);
        let addresses = (movement.start.0 >> 2, movement.dst.0 >> 2);
        match blocks{
            (0,0) => {
                Self {par1: self.par1 ^ (1 << addresses.0 ^ 1 << addresses.1), par2: self.par2, par3: self.par3}
            },
            (0,1) => {
                Self {par1: self.par1 ^ (1 << addresses.0), par2: self.par2 ^ (1 << addresses.1), par3: self.par3}
            },
            (0,2) => {
                Self {par1: self.par1 ^ (1 << addresses.0), par2: self.par2, par3: self.par3 ^ (1 << addresses.1)}
            },
            (1,0) => {
                Self {par1: self.par1 ^ (1 << addresses.1), par2: self.par2 ^ (1 << addresses.0), par3: self.par3}
            },
            (1,1) => {
                Self {par1: self.par1, par2: self.par2 ^ (1 << addresses.0 ^ 1 << addresses.1), par3: self.par3}
            },
            (1,2) => {
                Self {par1: self.par1, par2: self.par2 ^ (1 << addresses.0), par3: self.par3 ^ (1 << addresses.1)}
            },
            (2,0) => {
                Self {par1: self.par1 ^ (1 << addresses.1), par2: self.par2, par3: self.par3 ^ (1 << addresses.0)}
            },
            (2,1) => {
                Self {par1: self.par1, par2: self.par2 ^ (1 << addresses.1) , par3: self.par3 ^ (1 << addresses.0)}
            },
            (2,2) => {
                Self {par1: self.par1, par2: self.par2, par3: self.par3 ^ (1 << addresses.0 ^ 1 << addresses.1)}
            },
            _ => { panic!("specified position encode is invalid. Non-existent block was specified!")}
        }

    }

    fn flip_target_bit_mut(&mut self, position: &ElevenBoardPositionalEncoding){
        match position.0 & 3{
            0 => self.par1 ^= 1 << (position.0 >> 2),
            1 => self.par2 ^= 1 << (position.0 >> 2),
            2 => self.par3 ^= 1 << (position.0 >> 2),
            _ => {} 
        }
    }

    fn tile_is_empty_at(&self, position: &ElevenBoardPositionalEncoding) -> bool{
        let (block, address) = (position.0 & 3, position.0 >> 2);
        let target_block = match block{
            0 => self.par1,
            1 => self.par2,
            2 => self.par3,
            _ => panic!("Invalid board block was specified"),
        };
        (target_block >> address) & 1 == 0
    }

}

impl BoardEleven {
    fn get_ones(&self) -> [Vec<u8>;3] {
        let mut result: [Vec<u8>;3] = [Vec::with_capacity(64), Vec::with_capacity(64), Vec::with_capacity(64)];
        
        fn fill(position: &mut Vec<u8>, u: u64){
            for i in 0..64{
                let tmp = (u >> i) & 1;
                if tmp == 1 { position.push(i) }
            }
        }
        
        fill(&mut result[0], self.par1);
        fill(&mut result[1], self.par2);
        fill(&mut result[2], self.par3);
        result
    }
}

impl Shift for BoardEleven{
    #[inline]
    fn shift_e(&self) -> Self {
        let pre_masked: BoardEleven = Self{
            par1: self.par1 << 1,
            par2: self.par2 << 1,
            par3: self.par3 << 1,
        };
        pre_masked.reset_padding()
    }
    #[inline]
    fn shift_w(&self) -> Self {
        let pre_masked: BoardEleven = Self {
            par1: self.par1 >> 1,
            par2: self.par2 >> 1, 
            par3: self.par3 >> 1, 
        };
        pre_masked.reset_padding()
    }
    #[inline]
    fn shift_s(&self) -> Self {
        const SHIFT: usize = 4;
        let borrowed_1: &[u8] = &RawBytes::bytes_view(&self.par1)[4..=5];
        let borrowed_2: &[u8] = &RawBytes::bytes_view(&self.par2)[4..=5];
        let mut patch_1: [u8;8] = [0;8];
        let mut patch_2: [u8;8] = [0;8];
        patch_1[0..=1].copy_from_slice(borrowed_1);
        patch_2[0..=1].copy_from_slice(borrowed_2);
        let u1: u64 = u64::from_le_bytes(patch_1) >> SHIFT;
        let u2: u64 = u64::from_le_bytes(patch_2) >> SHIFT;
        let mut step1: BoardEleven = Self { par1: self.par1 << 12, par2: self.par2 << 12, par3: self.par3 << 12 };
        step1.par2 |= u1;
        step1.par3 |= u2;
        step1.reset_padding()
    }
    #[inline]
    fn shift_n(&self) -> Self {
        const SHIFT: usize = 4;
        let borrowed_1: &[u8] = &RawBytes::bytes_view(&self.par2)[0..=1];
        let borrowed_2: &[u8] = &RawBytes::bytes_view(&self.par3)[0..=1];
        let mut patch_1: [u8;8] = [0;8];
        let mut patch_2: [u8;8] = [0;8];
        patch_1[4..=5].copy_from_slice(borrowed_1);
        patch_2[4..=5].copy_from_slice(borrowed_2);
        let u1: u64 = u64::from_le_bytes(patch_1) << SHIFT;
        let u2: u64 = u64::from_le_bytes(patch_2) << SHIFT;
        let mut step1: BoardEleven = Self { par1: self.par1 >> 12, par2: self.par2 >> 12, par3: self.par3 >> 12 };
        step1.par1 |= u1;
        step1.par2 |= u2;
        step1.reset_padding()
    }
}

#[cfg(test)]
mod tests {


    use super::*;

    #[test] 
    fn io_works() {
        let preset: String = PRESET1.to_string();
        let board: BoardEleven = BoardEleven::try_from(preset).unwrap();
        // println!("{:b}", board.par1);
        // println!("{:b}", board.par2);
        // println!("{:b}", board.par3);
        print!("{board}");
    }

    #[test] 
    fn shift_works() {
        let preset: String = PRESET1.to_string();
        let board: BoardEleven = BoardEleven::try_from(preset).unwrap();
        println!("{board}");
        let shifted = board.shift_e();
        println!("{shifted}");
        let shifted = board.shift_w();
        println!("{shifted}");
        let shifted = board.shift_s();
        println!("{shifted}");
        let shifted = board.shift_n();
        println!("{shifted}");
    }

    #[test] 
    fn move_gen_works() {
        let b: BoardEleven = BoardEleven::from(PRESETX1);
        print!("{b}");
        let background: BoardEleven = BoardEleven::from(PRESETX_BACKGROUND);
        let moves = b.generate_action(&background);
        for (i,m) in moves.into_iter().enumerate(){
            print!("{m} ");
            if i % 10 == 9 { println!(""); }
        }
    }

    #[test] 
    fn get_ones_works() {
        let b: BoardEleven = BoardEleven::from(PRESETX1);
        print!("{b}");
        let bb = b.get_ones();
        for block in 0..3{
            for pos in &bb[block]{
                print!("{pos} ");
            }
            println!("");
        }
    }

    #[test] 
    fn blackmask() {
        let pre_bm: String = PRESET_MASK.to_string();
        let bm = BoardEleven::try_from(pre_bm).unwrap();
        assert_eq!(bm.par1, BoardEleven::BLACKMASK.par1);
        assert_eq!(bm.par2, BoardEleven::BLACKMASK.par2);
        assert_eq!(bm.par3, BoardEleven::BLACKMASK.par3);
    }

    #[test] 
    fn is_nonzero_works() {
        let b: BoardEleven = BoardEleven::from(PRESETX1);
        assert!(b.is_nonzero()); 
        let empty_b: BoardEleven = BoardEleven::from([0,0,0]);
        assert!(!empty_b.is_nonzero());
    }

    #[test] 
    fn eleven_board_positional_encoding_works(){
        let ee = ElevenBoardPositionalEncoding(14 << 2);
        assert_eq!(ElevenBoardPositionalEncoding::new(2,1), ee);
        let chars = vec!['c', '2'];
        assert_eq!(ElevenBoardPositionalEncoding::try_from(&chars[..]).unwrap(), ee); 
        assert_eq!(ElevenBoardPositionalEncoding::try_from("c2".to_string()).unwrap(), ee);
    }

    #[test] 
    fn eleven_board_positional_encoding_rejects_correctly() {
        assert!(ElevenBoardPositionalEncoding::try_from("aac2".to_string()).is_err());
        assert!(ElevenBoardPositionalEncoding::try_from("a21".to_string()).is_err());
        assert!(ElevenBoardPositionalEncoding::try_from("a_3".to_string()).is_err());
        
    }

    #[test] 
    #[should_panic]
    fn eleven_board_positional_encoding_new_panics_correctly(){
        let _: ElevenBoardPositionalEncoding = ElevenBoardPositionalEncoding::new(11,9);
    }

    #[test] 
    fn eleven_board_movement_input_works(){
        let movement: MoveOnBoardEleven = MoveOnBoardEleven { 
            start: ElevenBoardPositionalEncoding((31<<2)+2),
            dst: ElevenBoardPositionalEncoding((7<<2)+0),
        };
        assert_eq!(ElevenBoardPositionalEncoding::try_from("H11".to_string()).unwrap(), movement.start);
        assert_eq!(MoveOnBoardEleven::try_from("H11h1".to_string()).unwrap(), movement);
    }
    #[test] 
    fn move_piece_works() {
        let mut b: BoardEleven = BoardEleven::try_from(PRESET1.to_string()).unwrap();
        println!("{}", b);
        let movement = MoveOnBoardEleven::try_from("F6F9".to_string()).unwrap();
        b.force_move_mut(&movement);
        println!("{}", b);
    }

    #[test] 
    fn locate_ones_works(){
        let b = BoardEleven::try_from(PRESETX1).unwrap();
        let ones = b.locate_ones();
        println!("{}", b);
        print!("raw positional encoding: ");
        for position in ones.iter(){
            print!("{} ", position.0);
        }
        println!();
        print!("In a1 coordinate system: ");
        for position in ones.iter(){
            print!("{} ", position);
        }

        let mask = BoardEleven::mask_board();
        assert_eq!(mask.locate_ones().len(), 121);

        let empty = BoardEleven::try_from(PRESETX_EMPTY).unwrap();
        assert_eq!(empty.locate_ones().len(), 0);

    }

    #[test] 
    fn list_moves_works(){
        let b = BoardEleven::try_from(PRESETX1).unwrap();
        println!("{}", b);
        let moves = b.list_moves_east(&BoardEleven::BLACKMASK, &b.complement(), 10);
        assert_eq!(moves.len(), 20);
        print!("Moves to the east: ");
        for m in moves{
            print!("{} ", m);
        }
        println!();
        let moves = b.list_moves_west(&BoardEleven::BLACKMASK, &b.complement(), 10);
        assert_eq!(moves.len(), 21);
        print!("Moves to the west: ");
        for m in moves{
            print!("{} ", m);
        }
        println!();
        let moves = b.list_moves_south(&BoardEleven::BLACKMASK, &b.complement(), 10);
        assert_eq!(moves.len(), 28);
        print!("Moves to the south: ");
        for m in moves{
            print!("{} ", m);
        }
        println!();
        let moves = b.list_moves_north(&BoardEleven::BLACKMASK, &b.complement(), 10);
        assert_eq!(moves.len(), 33);
        print!("Moves to the north: ");
        for m in moves{
            print!("{} ", m);
        }
    }

    #[test]
    fn force_move_works() {
        let b = BoardEleven::try_from(PRESETX1).unwrap();
        let nb = b.force_move(&MoveOnBoardEleven::try_from("F6F9".to_owned()).unwrap());
        println!("{}", b);
        println!("{}", nb);
    }

}
