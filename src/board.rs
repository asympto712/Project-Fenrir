use std::{fmt::Display};

use bitboard::eleven::{self, BoardEleven, PRESETX1};
use bitflags::{bitflags, bitflags_match};

pub const ELEVENBOARDPRESET_STD_ATT: [u64;3] = [0x201_000_020_0F8, 0x401_401_603_401, 0x0F8_020_000];
pub const ELEVENBOARDPRESET_STD_DEF: [u64;3] = [0x020_000_000_000, 0x020_070_0D8_070, 0x000_000_000];
pub const ELEVENBOARDPRESET_STD_KING: [u64;3] = [0x000_000_000_000, 0x000_000_020_000, 0x000_000_000];
pub const ELEVENBOARDPRESET_STD_HOS: [u64;3] = [0x000_000_000_401, 0x000_000_020_000, 0x401_000_000];

pub struct TaflBoardEleven{
    bit_att: BoardEleven,
    bit_def: BoardEleven,
    bit_king: BoardEleven,
    hostile: BoardEleven
}

impl TaflBoardEleven{
    pub fn init_std() -> Self{
        let bit_att = BoardEleven::from(ELEVENBOARDPRESET_STD_ATT);
        let bit_def = BoardEleven::from(ELEVENBOARDPRESET_STD_DEF);
        let bit_king = BoardEleven::from(ELEVENBOARDPRESET_STD_KING);
        let hostile = BoardEleven::from(ELEVENBOARDPRESET_STD_HOS);
        Self{bit_att, bit_def, bit_king, hostile}
    } 
}
bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct BoardErrorFlag: u8{
        const ZERO = 0;
        const AD = 1;
        const AK = 1 << 1;
        const DK = 1 << 2;
        const AR = 1 << 3;
        const DR = 1 << 4;
    }
}
#[derive(Debug, Clone)]
struct BoardError{
    flag: BoardErrorFlag,
}

impl BoardError{
    fn generate(board: &TaflBoardEleven) -> Self{
        let mut flag: BoardErrorFlag = BoardErrorFlag::ZERO;
        if (board.bit_att & board.bit_def).is_nonzero() {flag = flag | BoardErrorFlag::AD}
        if (board.bit_att & board.bit_king).is_nonzero() {flag = flag | BoardErrorFlag::AK}
        if (board.bit_def & board.bit_king).is_nonzero() {flag = flag | BoardErrorFlag::DK}
        if (board.bit_att & board.hostile).is_nonzero() {flag = flag | BoardErrorFlag::AR}
        if (board.bit_def & board.hostile).is_nonzero() {flag = flag | BoardErrorFlag::DR}
        Self{flag}
    }
}

impl Display for BoardError{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        bitflags_match!(self.flag, {
            BoardErrorFlag::AD => write!(f, "Attackers and Defenders have an overlap"),
            BoardErrorFlag::AK => write!(f, "Attackers and King have an overlap"),
            BoardErrorFlag::AR => write!(f, "Attackers and Hostile tiles have an overlap"),
            BoardErrorFlag::DK => write!(f, "Defenders and King have an overlap"),
            BoardErrorFlag::DR => write!(f, "Defenders and Hostile tiles have an overlap"),
            _ => Ok(())
        })
}
}

impl TaflBoardEleven{
    pub fn new(bit_att: BoardEleven, bit_def: BoardEleven, bit_king: BoardEleven, hostile: BoardEleven) -> Self{
        Self{bit_att, bit_def, bit_king, hostile}
    }
}

impl Display for TaflBoardEleven{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let be = BoardError::generate(self);
        if be.flag != BoardErrorFlag::ZERO{
            writeln!(f, "This board contains an abnormality, abandoning printing..")?;
            println!("{}", be);
            return Ok(())
        } 
        fn draw(
            att_b: u64,
            def_b: u64,
            king_b: u64,
            hos_b: u64,
            f: &mut std::fmt::Formatter<'_>,
            bound: usize,
            pad: usize,
        ) -> Result<(), std::fmt::Error> {
            for i in 1..=bound{
                if i > bound {
                    break;
                }

                if i % pad == 0{
                    writeln!(f)?;
                    continue;
                }

                if (att_b >> (i-1)) & 1 == 1{
                    write!(f, "A")?;
                    continue;
                }
                if (def_b >> (i-1)) & 1 == 1{
                    write!(f, "D")?;
                    continue;
                }
                if (king_b >> (i-1)) & 1 == 1{
                    write!(f, "K")?;
                    continue;
                }
                if (hos_b >> (i-1)) & 1 == 1{
                    write!(f, "\u{2612}" )?; 
                    // cross inside square
                    continue;
                }
                write!(f, "\u{25A1}")?;
                // Empty square

            }
            Ok(())
        }
        draw(self.bit_att.par1, self.bit_def.par1,
        self.bit_king.par1, self.hostile.par1 , f, 48, 12)?;

        draw(self.bit_att.par2, self.bit_def.par2,
        self.bit_king.par2, self.hostile.par2, f, 48, 12)?;
        
        draw(self.bit_att.par3, self.bit_def.par3,
        self.bit_king.par3, self.hostile.par3, f, 36, 12)?;
        Ok(())
    }
}


#[test] 
fn board_print_works1(){
    let b = TaflBoardEleven::init_std();
    print!("{}", b);
}

#[test] 
fn board_print_works2(){
    let mut b = TaflBoardEleven::init_std();
    let def_board = BoardEleven::from(ELEVENBOARDPRESET_STD_DEF);
    b.bit_att = def_board;
    // The Attacker's board is the same as Defender's board, so
    // this should NOT print the board.
    // Check with cargo run -- board_print_works2 --show-output 
    println!("{}", b);
    let e = BoardError::generate(&b);
    print!("{:?}", e);
    assert!((b.bit_att & b.bit_def).is_nonzero());

}