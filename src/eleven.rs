use core::num;
use std::{fmt::{format, Display}, ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not}};
use crate::Direction;

use rawbytes::RawBytes;
use bitflags::{bitflags, bitflags_match};

pub const BOARDELEVENPART1: usize = 48;
pub const BOARDELEVENPART2: usize = 96;
pub const BOARDELEVENPART3: usize = 132;
pub const BLOCKLEN: [u8;3] = [BLOCK1LEN, BLOCK2LEN, BLOCK3LEN];
pub const BLOCK1LEN: u8 = 48;
pub const BLOCK2LEN: u8 = 48;
pub const BLOCK3LEN: u8 = 36;

#[allow(dead_code)]
pub const PRESET1: &'static str = "10001010110\n00100111101\n00100111111\n00001111100\n11110011010\n00110101101\n11000011011\n11000000111\n00000000000\n00011111111\n00011001111\n";
pub const PRESET_MASK: &'static str = "11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n";

pub const PRESETX1: [u64;3] = [0x4aa5552aa555, 0x7ff7ff7ff000, 0x7ff7ff7ff];
pub const PRESETX_EMPTY: [u64;3] = [0x000000000000, 0x000000000000, 0x000000000];
pub const PRESETX_BACKGROUND: [u64;3] = [0x7ff7ff7ff7ff, 0x7ff7ff7ff7ff, 0x7ff7ff7ff];

pub const BLACKMASK: BoardEleven = BoardEleven{par1: 0x7ff7ff7ff7ff, par2: 0x7ff7ff7ff7ff, par3: 0x7ff7ff7ff};

pub const fn num_to_alphabet(n: u8) -> char{
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

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BoardEleven {
    pub par1: u64,
    pub par2: u64,
    pub par3: u64,
}

// enum Direction{
//     E(usize),
//     W(usize),
//     S(usize),
//     N(usize),
//     All(usize),
// }

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElevenBoardPositionalEncoding(u8);

impl ElevenBoardPositionalEncoding{
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
                return Some( (value as u32 - 0x30) as u8 )
            } else { return None }
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
                return Some( (value as u32 - 0x61) as u8 )
            } else { return None }
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
// represent a piece move by its starting point and the end point. First two bits specifies which partition the position is in, and the latter
// 6 bits (2^6 = 64) specifies the location of the target bit
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveOnBoardEleven {
    start: ElevenBoardPositionalEncoding, 
    dst: ElevenBoardPositionalEncoding,
}

impl Display for MoveOnBoardEleven{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn position_to_str(u:u8) -> Result<String, String>{
            let block: u8 = u & 3;
            let offset: u8 = match block{
                0 => 0,
                1 => BOARDELEVENPART1 as u8,
                2 => BOARDELEVENPART2 as u8,
                _ => { return Err("error occured.".to_owned())}
            };
            let address: u8 = (u >> 2) + offset; //Don't forget the parenthesis!!
            let coordinate: (u8,u8) = (address % 12, address / 12);
            let s = format(format_args!("{}{}", num_to_alphabet(coordinate.0), 11 - coordinate.1));
            Ok(s)
        }
        let s1 = position_to_str(self.start.0).map_err(|_| std::fmt::Error)?;
        let s2 = position_to_str(self.dst.0).map_err(|_| std::fmt::Error)?;
        write!(f, "{}{}", s1,s2)?;
        Ok(())
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

impl TryFrom<String> for MoveOnBoardEleven{
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let alphabet_positions: Vec<usize> = value.to_lowercase().chars().enumerate().filter_map(
            |(i,c)| c.is_ascii_lowercase().then_some(i)).collect();
        if alphabet_positions.len() != 2{
            return Err("Wrong input pattern for movement: try e.g. A2D2".to_string())
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

impl BoardEleven {
    pub fn new() -> Self {
        Self { par1: 0, par2: 0, par3: 0 }
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
        return Self{
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
        return Self{
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
        return Self{
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
            if c == '1' { return Ok(PieceOrNewline::Piece(1)) }
            else if c == '0' { return Ok(PieceOrNewline::Piece(0)) }
            else if c == '\n' { return Ok(PieceOrNewline::Newline)}
            else { return Err("invalid character encountered".to_owned())}
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

// This operation will flip all the bits ON THE BOARD. extra paddings are all reset to 0 (as they should be at all times)
impl Not for BoardEleven{
    type Output = Self;
    fn not(self) -> Self::Output {
        let c: BoardEleven = Self{
            par1: !self.par1,
            par2: !self.par2,
            par3: !self.par3,
        };
        let mask: BoardEleven = BoardEleven::mask_board();
        c & mask
    }
}
impl BoardEleven{
    // When you want to reset the padding bits to zero, you can simply generate mask with this and take the Bitwise AND
    pub const fn mask_board() -> Self{
        BLACKMASK  
    }
    #[inline]
    pub fn reset_padding(&self) -> Self{
        let mask = BoardEleven::mask_board();
        *self & mask
    }
    
    #[inline]
    pub fn is_nonzero(&self) -> bool{
        let b = self.reset_padding();
        if b.par1 > 0 || b.par2 > 0 || b.par3 > 0 {true}
        else {false}
    }

    #[inline]
    pub fn complement(&self) -> Self{
        *self ^ BoardEleven::mask_board()
    }

    #[inline]
    pub fn move_piece(&mut self, movement: MoveOnBoardEleven){

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

    pub fn dilation(&self, d: Direction) -> Self{
        let mut tmp = self.clone();
        match d{
            Direction::E(u) => {
                for _ in 0..u{
                    tmp = tmp | tmp.shift_e();
                }
                return tmp
            },
            Direction::W(u) => {
                for _ in 0..u{
                    tmp = tmp | tmp.shift_w();
                }
                return tmp
            },
            Direction::N(u) => {
                for _ in 0..u{
                    tmp = tmp | tmp.shift_n();
                }
                return tmp
            },
            Direction::S(u) => {
                for _ in 0..u{
                    tmp = tmp | tmp.shift_s();
                }
                return tmp
            },
            Direction::All(u) => {
                for _ in 0..u{
                    tmp = tmp | tmp.shift_e() | tmp.shift_n() | tmp.shift_s() | tmp.shift_w();
                }
                return tmp
            }
        }
    }
    pub fn erosion(&self, d: Direction) -> Self{
        let mut tmp = self.clone();
        match d{
            Direction::E(u) => {
                for _ in 0..u{
                    tmp = tmp & tmp.shift_e();
                }
                return tmp
            },
            Direction::W(u) => {
                for _ in 0..u{
                    tmp = tmp & tmp.shift_w();
                }
                return tmp
            },
            Direction::N(u) => {
                for _ in 0..u{
                    tmp = tmp & tmp.shift_n();
                }
                return tmp
            },
            Direction::S(u) => {
                for _ in 0..u{
                    tmp = tmp & tmp.shift_s();
                }
                return tmp
            },
            Direction::All(u) => {
                for _ in 0..u{
                    tmp = tmp & tmp.shift_e() & tmp.shift_n() & tmp.shift_s() & tmp.shift_w();
                }
                return tmp
            }
        }
    }
}

pub trait Shift{
    fn shift_w(&self) -> Self;
    fn shift_e(&self) -> Self;
    fn shift_n(&self) -> Self;
    fn shift_s(&self) -> Self;
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
        let step2 = step1.reset_padding();
        step2
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
        let step2 = step1.reset_padding();
        step2
    }
}

impl BoardEleven{

    pub fn get_ones(&self) -> [Vec<u8>;3] {
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

    pub fn generate_action(&self, mask: &Self) -> Vec<MoveOnBoardEleven>{

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
        let mut tmp = self.clone();
        for i in 1..=10{
            tmp = tmp.shift_e() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions[0].len() + positions[1].len() + positions[2].len() == 0 {
                break;
            }
            for block in 0..=2{
                for u in &positions[block]{
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
        let mut tmp = self.clone();
        for i in 1..=10{
            tmp = tmp.shift_w() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions[0].len() + positions[1].len() + positions[2].len() == 0 {
                break;
            }
            for block in 0..=2{
                for u in &positions[block]{
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
        let mut tmp = self.clone();
        for i in 1..=10{
            tmp = tmp.shift_s() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions[0].len() + positions[1].len() + positions[2].len() == 0 {
                break;
            }
            for block in 0..=2{
                for u in &positions[block]{
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
        let mut tmp = self.clone();
        for i in 1..=10{
            tmp = tmp.shift_s() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions[0].len() + positions[1].len() + positions[2].len() == 0 {
                break;
            }
            for block in 0..=2{
                for u in &positions[block]{
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

    fn flip_target_bit(&mut self, position: ElevenBoardPositionalEncoding){
        match position.0 & 3{
            0 => self.par1 ^= 1 << (position.0 >> 2),
            1 => self.par2 ^= 1 << (position.0 >> 2),
            2 => self.par3 ^= 1 << (position.0 >> 2),
            _ => {} 
        }
    }
    fn force_move(&mut self, movement: MoveOnBoardEleven){
        self.flip_target_bit(movement.start);
        self.flip_target_bit(movement.dst);
    }
}


// fn main() {
//     let b = BoardEleven::from(PRESETX1);
//     println!("{b}");
//     let mut tmp = b.clone();
//     for i in 1..=10{
//         tmp = tmp.shift_s();
//         println!("{tmp}");
//     }
// }

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
    assert_eq!(bm.par1, BLACKMASK.par1);
    assert_eq!(bm.par2, BLACKMASK.par2);
    assert_eq!(bm.par3, BLACKMASK.par3);
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
    b.force_move(movement);
    println!("{}", b);
}