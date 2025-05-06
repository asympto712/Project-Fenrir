use std::{fmt::{format, Display}, ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not}};

use rawbytes::RawBytes;

const BOARDELEVENPART1: usize = 48;
const BOARDELEVENPART2: usize = 96;
const BOARDELEVENPART3: usize = 132;
const BLOCKLEN: [u8;3] = [BLOCK1LEN, BLOCK2LEN, BLOCK3LEN];
const BLOCK1LEN: u8 = 48;
const BLOCK2LEN: u8 = 48;
const BLOCK3LEN: u8 = 36;

#[allow(dead_code)]
const PRESET1: &'static str = "10001010110\n00100111101\n00100111111\n00001111100\n11110011010\n00110101101\n11000011011\n11000000111\n00000000000\n00011111111\n00011001111\n";
const PRESET_MASK: &'static str = "11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n11111111111\n";

const PRESETX1: [u64;3] = [0x4aa5552aa555, 0x7ff7ff7ff000, 0x7ff7ff7ff];
const PRESETX_EMPTY: [u64;3] = [0x000000000000, 0x000000000000, 0x000000000];
const PRESETX_BACKGROUND: [u64;3] = [0x7ff7ff7ff7ff, 0x7ff7ff7ff7ff, 0x7ff7ff7ff];

const BLACKMASK: BoardEleven = BoardEleven{par1: 0x7ff7ff7ff7ff, par2: 0x7ff7ff7ff7ff, par3: 0x7ff7ff7ff};

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

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct BoardEleven {
    par1: u64,
    par2: u64,
    par3: u64,
}

enum Direction{
    E(usize),
    W(usize),
    S(usize),
    N(usize),
    All(usize),
}

// represent a piece move by its starting point and the end point. First two bits specifies which partition the position is in, and the latter
// 6 bits (2^6 = 64) specifies the location of the target bit
#[derive(Debug, Clone, Copy, PartialEq)]
struct Move {
    start: u8, 
    dst: u8,
}

impl Display for Move{
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
        let s1 = position_to_str(self.start).map_err(|_| std::fmt::Error)?;
        let s2 = position_to_str(self.dst).map_err(|_| std::fmt::Error)?;
        write!(f, "{}{}", s1,s2)?;
        Ok(())
    }
}

impl BoardEleven {
    fn new() -> Self {
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
    const fn mask_board() -> Self{
        BLACKMASK  
    }
    fn reset_padding(&self) -> Self{
        let mask = BoardEleven::mask_board();
        *self & mask
    }

    fn complement(&self) -> Self{
        *self ^ BoardEleven::mask_board()
    }

    fn dilation(&self, d: Direction) -> Self{
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
    fn erosion(&self, d: Direction) -> Self{
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

trait Shift{
    fn shift_w(&self) -> Self;
    fn shift_e(&self) -> Self;
    fn shift_n(&self) -> Self;
    fn shift_s(&self) -> Self;
}

impl Shift for BoardEleven{
    fn shift_e(&self) -> Self {
        let pre_masked: BoardEleven = Self{
            par1: self.par1 << 1,
            par2: self.par2 << 1,
            par3: self.par3 << 1,
        };
        pre_masked.reset_padding()
    }
    fn shift_w(&self) -> Self {
        let pre_masked: BoardEleven = Self {
            par1: self.par1 >> 1,
            par2: self.par2 >> 1, 
            par3: self.par3 >> 1, 
        };
        pre_masked.reset_padding()
    }
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

    fn generate_action(&self, mask: &Self) -> Vec<Move>{

        fn calculate_start_position_for_move_south(u: u8, block: &mut u8, step: u8) -> u8{
            let mut v = u;
            let n: u8 = step * 12;
            while v < n{
                *block -= 1;
                v += BLOCKLEN[*block as usize];
            }
            (v - n) << 2 | *block
        }

        fn calculate_start_position_for_move_north(u: u8, block: &mut u8, step: u8) -> u8{
            let n: u8 = step * 12;
            let mut v = u + n;
            while v > BLOCKLEN[*block as usize]{
                v -= BLOCKLEN[*block as usize];
                *block += 1;
            }
            v << 2 | *block
        }

        fn calculate_start_position_for_move_east(u: u8, block: u8, step: u8) -> u8{
            (u - step) << 2 | block
        }

        fn calculate_start_position_for_move_west(u: u8, block: u8, step: u8) -> u8{
            (u + step) << 2 | block
        }

        let mut moves: Vec<Move> = Vec::new();

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
                    moves.push(Move{start, dst});
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
                    moves.push(Move{start, dst});
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
                    moves.push(Move{start, dst});
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
                    moves.push(Move{start, dst});
                }
            }
        }

        moves
    }
}


fn main() {
    let b = BoardEleven::from(PRESETX1);
    println!("{b}");
    let mut tmp = b.clone();
    for i in 1..=10{
        tmp = tmp.shift_s();
        println!("{tmp}");
    }
}

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
