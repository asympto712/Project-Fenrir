#![allow(dead_code)]
#![allow(clippy::unusual_byte_groupings)]
use rand::prelude::*;
use std::{fmt::Display, ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not}};
use crate::{Shift, Direction, PositionalEncoding, MoveOnBoard, BitBoard};

use rawbytes::RawBytes;
use bincode;

const BOARDSEVENPART1: usize = 56;
const BLOCKLEN: u8 = 56;

#[allow(dead_code)]
const PRESET1: &str = "1000101\n0010011\n0010011\n0000111\n1111001\n0011010\n1100001\n";
const PRESET_MASK: &str = "1111111\n1111111\n1111111\n1111111\n1111111\n1111111\n1111111";

const PRESETX1: u64 = 0x43_2c_4f_70_64_64_51; // Same board as PRESET1
const PRESETX_EMPTY: u64 = 0x00_00_00_00_00_00_00;
const PRESETX_BACKGROUND: u64 = 0x7f_7f_7f_7f_7f_7f_7f;


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
        _ => 'Z',
    }
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SevenBoardPositionalEncoding(pub u8);

impl PositionalEncoding for SevenBoardPositionalEncoding{

    const BOARD_SIZE: u8 = 7;

    // specify the coordinate to get the positional encoding. The coordinate starts from 0. Will panic on out-of-bounds coordinate.
    fn new(x: u8,y: u8) -> Self{
        assert!(x < 7, "x-coordinate out of bound");
        assert!(y < 7, "y-coordinate out of bound");
        Self(y * 8 + x)
    }

    // obtain the coordinate in the screen-like coordinate system (north left: origin) 
    // returning (x,y), where x ranges 0..7 and y ranges 0..7
    fn get_coordinate(&self) -> (u8,u8) {
        let y: u8 = self.0 / 8;
        let x: u8 = self.0 % 8;
        (x,y)
    }
}

impl TryFrom<u8> for SevenBoardPositionalEncoding{
    type Error = String;
    fn try_from(n: u8) -> Result<Self, Self::Error> {
        if n >= 49 { return Err("input address out of bounds".to_string())}
        Ok(Self::new(n % 7, n / 7))
    }
}

impl TryFrom<&[char]> for SevenBoardPositionalEncoding{
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
            if value < 8 { Ok(value)}
            else { Err("Input position out of bound".to_string())}
        }

        let msg1: &str = "Invalid input format";

        if value.len() != 2 {return Err(msg1.to_string())}

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

impl TryFrom<String> for SevenBoardPositionalEncoding{
    type Error = String;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        let chars: Vec<char> = value.to_lowercase().chars().collect();
        Self::try_from(&chars[..])
    }
}

impl std::fmt::Display for SevenBoardPositionalEncoding{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (x,y): (u8, u8) = (self.0 % 8, self.0 / 8);
        write!(f, "{}{}", num_to_alphabet(x), y+1)?;
        Ok(())
    }
}
// represent a piece move by its starting point and the end point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveOnBoardSeven {
    pub start: SevenBoardPositionalEncoding, 
    pub dst: SevenBoardPositionalEncoding,
}

impl Display for MoveOnBoardSeven{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.start, self.dst)?;
        Ok(())
    }
}

impl TryFrom<String> for MoveOnBoardSeven{
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
            let start: SevenBoardPositionalEncoding =
            SevenBoardPositionalEncoding::try_from(&chars[alphabet_positions[0]..alphabet_positions[1]])?;

            let dst: SevenBoardPositionalEncoding =
            SevenBoardPositionalEncoding::try_from(&chars[alphabet_positions[1]..])?;
            return Ok(Self{start, dst})
        }
    }
}

impl MoveOnBoard for MoveOnBoardSeven{

    type Position = SevenBoardPositionalEncoding;

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
        let init = BoardSeven::new().flip_target_bit(&self.start);
        let target = BoardSeven::new().flip_target_bit(&self.dst);

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
pub struct BoardSeven(pub u64);

impl Default for BoardSeven {
    fn default() -> Self {
        BoardSeven::from(PRESETX1)
    }
}

impl From<u64> for BoardSeven{
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl BitAnd for BoardSeven{
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}
impl BitAndAssign for BoardSeven{
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for BoardSeven{
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}
impl BitOrAssign for BoardSeven{
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl BitXor for BoardSeven{
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}
impl BitXorAssign for BoardSeven{
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}


impl Display for BoardSeven {
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
                if *count % 8 == 7 { 
                    *count += 1; 
                    writeln!(f)?;
                    continue;
                }

                let shift: u8 = 1 << i;
                let r = (shift & byte) > 0;

                write!(f, "{:b} ", r as u8)?;
                
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
        draw(self.0, &mut count, f, BOARDSEVENPART1)?;
        Ok(())
    }
}

impl TryFrom<String> for BoardSeven{
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

        fn store(board: &mut u64, shift: usize, c: char, count: &mut usize) -> Result<(),String>{
            if let PieceOrNewline::Piece(piece) = inspect(c)?{
                *board |= piece << shift;
                *count += 1;
            }
            Ok(())
        }
        let chars = value.chars();
        let mut board: u64 = 0;

        let mut count: usize = 0;

        for c in chars{
            if count < BOARDSEVENPART1{
                if count % 8 == 7 { 
                    count += 1;
                }
                store(&mut board, count, c, &mut count)?;
            }
        }
        Ok(BoardSeven(board))
    }
}

// This operation will flip all the bits, including the paddings. Only use this when you are sure that you don't want padding-reset.
// For flipping bits on the board only, use self.complement() instead. 
impl Not for BoardSeven{
    type Output = Self;
    fn not(self) -> Self::Output {
        let c: BoardSeven = Self(!self.0);
        // let mask: BoardSeven = BoardSeven::mask_board();
        // c & mask
        c
    }
}

impl BitBoard for BoardSeven{

    const BLACKMASK: BoardSeven = BoardSeven(0x7f_7f_7f_7f_7f_7f_7f);
    const ONLY_PAD: BoardSeven = BoardSeven(0x80_80_80_80_80_80_80);
    const ONLY_EDGES: BoardSeven = BoardSeven(0x7f_41_41_41_41_41_7f);
    const CORNERS: BoardSeven = BoardSeven(0x41_00_00_00_00_00_41);

    type Movement = MoveOnBoardSeven;
    type Position = SevenBoardPositionalEncoding;

    fn new() -> Self {
        Self(0)
    }

    fn random<T>(rng: &mut T) -> Self 
    where T: Rng
    {
        let tmp = Self(rng.next_u64());
        tmp.reset_padding()
    }
    fn empty() -> Self {
        Self(0)
    }
    // only for the creation of dummy data
    #[inline]
    fn ghastly() -> Self {
        Self(0)
    }

    #[inline]
    fn is_nonzero(&self) -> bool{
        let b = self.reset_padding();
        b.0 > 0
    }

    #[inline]
    fn move_piece(&mut self, movement: MoveOnBoardSeven){

        #[inline]
        fn flip_bit(n: &mut u64, loc: u8){
            let mask = 1u64 << loc;
            *n ^= mask;
        }
        
        flip_bit(&mut self.0, movement.start.0);
        flip_bit(&mut self.0, movement.dst.0);
    }

    #[inline]
    fn shift_e_without_padding_reset(&self) -> Self {
        Self(self.0 << 1)
    }

    #[inline]
    fn shift_w_without_padding_reset(&self) -> Self {
        Self(self.0 >> 1)
    }

    fn locate_ones(&self) -> Vec<SevenBoardPositionalEncoding> {
        let mut value: Vec<SevenBoardPositionalEncoding> = Vec::new();
        
        for i in 0..BLOCKLEN{
            let tmp = self.0 >> i;
            if (i + 1) % 8 == 0{ continue }  // Skip padding bits
            if tmp & 1 == 1 {
                value.push(SevenBoardPositionalEncoding(i));
            }
        }
        value
    }

    fn neighbor_of(location: &SevenBoardPositionalEncoding) -> Self{
        let address = location.0;
        let tmp = Self(1u64 << address);
        tmp | tmp.shift_e() | tmp.shift_w() | tmp.shift_s() | tmp.shift_n()
    }

    // Older implementation of a function that lists legal moves. "list_moves_{direction}" is more recommended.
    fn generate_action(&self, mask: &Self) -> Vec<MoveOnBoardSeven>{

        #[inline]
        fn calculate_start_position_for_move_south(u: u8, _block: &mut u8, step: u8) -> u8{
            u - step * 8  // In 7x7 board, moving south by 1 step = subtract 8 positions
        }

        #[inline]
        fn calculate_start_position_for_move_north(u: u8, _block: &mut u8, step: u8) -> u8{
            u + step * 8  // In 7x7 board, moving north by 1 step = add 8 positions
        }

        #[inline]
        fn calculate_start_position_for_move_east(u: u8, _block: u8, step: u8) -> u8{
            u - step  // In 7x7 board, moving east by 1 step = subtract 1 position
        }

        #[inline]
        fn calculate_start_position_for_move_west(u: u8, _block: u8, step: u8) -> u8{
            u + step  // In 7x7 board, moving west by 1 step = add 1 position
        }

        let mut moves: Vec<MoveOnBoardSeven> = Vec::new();

        // Moves to East are dealt with here
        let mut tmp = *self;
        for i in 1..=6{  // Changed from 10 to 6 for 7x7 board
            tmp = tmp.shift_e() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions.is_empty() {
                break;
            }
            for u in &positions{
                let dst: u8 = *u;
                let start: u8 = calculate_start_position_for_move_east(*u, 0, i);
                moves.push(MoveOnBoardSeven{
                    start: SevenBoardPositionalEncoding( start ), 
                    dst: SevenBoardPositionalEncoding( dst )
                });
            }
        }

        // Moves to West are dealt with here
        let mut tmp = *self;
        for i in 1..=6{  // Changed from 10 to 6 for 7x7 board
            tmp = tmp.shift_w() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions.is_empty() {
                break;
            }
            for u in &positions{
                let dst: u8 = *u;
                let start: u8 = calculate_start_position_for_move_west(*u, 0, i);
                moves.push(MoveOnBoardSeven{
                    start: SevenBoardPositionalEncoding( start ), 
                    dst: SevenBoardPositionalEncoding( dst )
                });
            }
        }

        // Moves to South are dealt with here
        let mut tmp = *self;
        for i in 1..=6{  // Changed from 10 to 6 for 7x7 board
            tmp = tmp.shift_s() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions.is_empty() {
                break;
            }
            for u in &positions{
                let dst: u8 = *u;
                let start: u8 = calculate_start_position_for_move_south(*u, &mut 0, i);
                moves.push(MoveOnBoardSeven{
                    start: SevenBoardPositionalEncoding( start ), 
                    dst: SevenBoardPositionalEncoding( dst )
                });
            }
        }

        // Moves to North are dealt with here
        let mut tmp = *self;
        for i in 1..=6{  // Changed from 10 to 6 for 7x7 board
            tmp = tmp.shift_n() & self.complement() & *mask;
            let positions = tmp.get_ones();
            if positions.is_empty() {
                break;
            }
            for u in &positions{
                let dst: u8 = *u;
                let start: u8 = calculate_start_position_for_move_north(*u, &mut 0, i);
                moves.push(MoveOnBoardSeven{
                    start: SevenBoardPositionalEncoding( start ), 
                    dst: SevenBoardPositionalEncoding( dst )
                });
            }
        }

        moves
    }

    #[inline]
    fn flip_target_bit(&self, position: &SevenBoardPositionalEncoding) -> Self{
        let address = position.0;
        Self(self.0 ^ (1 << address))
    }

    #[inline]
    fn flip_target_bit_mut(&mut self, position: &SevenBoardPositionalEncoding){
        let address = position.0;
        self.0 ^= 1 << address;
    }

    #[inline]
    fn force_move(&self, movement: &MoveOnBoardSeven) -> Self {
        let addresses = (movement.start.0, movement.dst.0);
        Self(self.0 ^ (1 << addresses.0) ^ (1 << addresses.1))
    }

    fn tile_is_empty_at(&self, position: &SevenBoardPositionalEncoding) -> bool{
        let address = position.0;
        (self.0 >> address) & 1 == 0
    }
}

impl Shift for BoardSeven{
    #[inline]
    fn shift_e(&self) -> Self {
        let pre_masked: BoardSeven = Self(self.0 << 1);
        pre_masked.reset_padding()
    }
    #[inline]
    fn shift_w(&self) -> Self {
        let pre_masked: BoardSeven = Self(self.0 >> 1);
        pre_masked.reset_padding()
    }
    #[inline]
    fn shift_s(&self) -> Self {
        let pre_masked: BoardSeven = Self(self.0 << 8);
        pre_masked.reset_padding()
    }
    #[inline]
    fn shift_n(&self) -> Self {
        let pre_masked: BoardSeven = Self(self.0 >> 8);
        pre_masked.reset_padding()
    }
}

impl BoardSeven{
    fn get_ones(&self) -> Vec<u8> {
        let mut result: Vec<u8> = Vec::with_capacity(64);
        
        for i in 0..64{
            let tmp = (self.0 >> i) & 1;
            if tmp == 1 { result.push(i) }
        }
        
        result
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test] 
    fn io_works() {
        let preset: String = PRESET1.to_string();
        let board: BoardSeven = BoardSeven::try_from(preset).unwrap();
        assert_eq!(board, BoardSeven::from(PRESETX1));
        assert!(board.equals(&BoardSeven::from(PRESETX1)));
        print!("{board}");
    }

    #[test] 
    fn shift_works() {
        let preset: String = PRESET1.to_string();
        let board: BoardSeven = BoardSeven::try_from(preset).unwrap();
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
        let b: BoardSeven = BoardSeven::from(PRESETX1);
        print!("{b}");
        let background: BoardSeven = BoardSeven::from(PRESETX_BACKGROUND);
        let moves = b.generate_action(&background);
        for (i,m) in moves.iter().enumerate(){
            print!("{m} ");
            if i % 10 == 9 { println!(""); }
        }
        assert_eq!(moves.len(),
            b.list_moves_east(&BoardSeven::BLACKMASK, &b.complement(), 6).len()
            +b.list_moves_west(&BoardSeven::BLACKMASK, &b.complement(), 6).len()
            +b.list_moves_north(&BoardSeven::BLACKMASK, &b.complement(), 6).len()
            +b.list_moves_south(&BoardSeven::BLACKMASK, &b.complement(), 6).len()
        );
        
    }

    #[test] 
    fn get_ones_works() {
        let b: BoardSeven = BoardSeven::from(PRESETX1);
        print!("{b}");
        let bb = b.get_ones();
        assert_eq!(bb.len(), 23);
        for pos in &bb{
            print!("{pos} ");
        }
        println!("");
        assert_eq!(BoardSeven::BLACKMASK.get_ones().len(), 49);
        assert_eq!(BoardSeven::ONLY_PAD.get_ones().len(), 7);
        assert_eq!(BoardSeven::ONLY_EDGES.get_ones().len(), 24);
        
    }

    #[test] 
    fn blackmask() {
        let pre_bm: String = PRESET_MASK.to_string();
        let bm = BoardSeven::try_from(pre_bm).unwrap();
        assert_eq!(bm.0, BoardSeven::BLACKMASK.0);
    }

    #[test] 
    fn is_nonzero_works() {
        let b: BoardSeven = BoardSeven::from(PRESETX1);
        assert!(b.is_nonzero()); 
        let empty_b: BoardSeven = BoardSeven::from(0);
        assert!(!empty_b.is_nonzero());
    }

    #[test] 
    fn seven_board_positional_encoding_works(){
        let ee = SevenBoardPositionalEncoding(10);  // Position (2,1) -> 1*8 + 2 = 10
        assert_eq!(SevenBoardPositionalEncoding::new(2,1), ee);
        let chars = vec!['c', '2'];
        assert_eq!(SevenBoardPositionalEncoding::try_from(&chars[..]).unwrap(), ee); 
        assert_eq!(SevenBoardPositionalEncoding::try_from("c2".to_string()).unwrap(), ee);
    }

    #[test] 
    fn seven_board_positional_encoding_rejects_correctly() {
        assert!(SevenBoardPositionalEncoding::try_from("aac2".to_string()).is_err());
        assert!(SevenBoardPositionalEncoding::try_from("a21".to_string()).is_err());
        assert!(SevenBoardPositionalEncoding::try_from("a_3".to_string()).is_err());
        
    }

    #[test] 
    #[should_panic]
    fn seven_board_positional_encoding_new_panics_correctly(){
        let _: SevenBoardPositionalEncoding = SevenBoardPositionalEncoding::new(11,9);
    }

    #[test] 
    fn seven_board_movement_input_works(){
        let movement: MoveOnBoardSeven = MoveOnBoardSeven { 
            start: SevenBoardPositionalEncoding(6),
            dst: SevenBoardPositionalEncoding(54),
        };
        assert_eq!(SevenBoardPositionalEncoding::try_from("G1".to_string()).unwrap(), movement.start);
        assert_eq!(MoveOnBoardSeven::try_from("G1G7".to_string()).unwrap(), movement);
    }
    #[test] 
    fn move_piece_works() {
        let mut b: BoardSeven = BoardSeven::try_from(PRESET1.to_string()).unwrap();
        println!("{}", b);
        let movement = MoveOnBoardSeven::try_from("F6F7".to_string()).unwrap();
        b.force_move_mut(&movement);
        println!("{}", b);
    }

    #[test] 
    fn locate_ones_works(){
        let b = BoardSeven::try_from(PRESETX1).unwrap();
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

        let mask = BoardSeven::mask_board();
        assert_eq!(mask.locate_ones().len(), 49);

        let empty = BoardSeven::try_from(PRESETX_EMPTY).unwrap();
        assert_eq!(empty.locate_ones().len(), 0);

    }

    #[test] 
    fn list_moves_works(){
        let b = BoardSeven::try_from(PRESETX1).unwrap();
        println!("{}", b);
        let moves = b.list_moves_east(&BoardSeven::BLACKMASK, &b.complement(), 6);
        assert_eq!(moves.len(), 16);
        print!("Moves to the east: ");
        for m in moves{
            print!("{} ", m);
        }
        println!();
        let moves = b.list_moves_west(&BoardSeven::BLACKMASK, &b.complement(), 6);
        assert_eq!(moves.len(), 25);
        print!("Moves to the west: ");
        for m in moves{
            print!("{} ", m);
        }
        println!();
        let moves = b.list_moves_south(&BoardSeven::BLACKMASK, &b.complement(),6);
        assert_eq!(moves.len(), 16);
        print!("Moves to the south: ");
        for m in moves{
            print!("{} ", m);
        }
        println!();
        let moves = b.list_moves_north(&BoardSeven::BLACKMASK, &b.complement(), 6);
        assert_eq!(moves.len(), 20);
        print!("Moves to the north: ");
        for m in moves{
            print!("{} ", m);
        }
    }

    #[test]
    fn force_move_works() {
        let b = BoardSeven::try_from(PRESETX1).unwrap();
        let nb = b.force_move(&MoveOnBoardSeven::try_from("F6F7".to_owned()).unwrap());
        println!("{}", b);
        println!("{}", nb);
    }

}