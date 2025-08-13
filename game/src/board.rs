use rand::prelude::*;
use std::{fmt::Display};

use bitboard::{BitBoard, PositionalEncoding, MoveOnBoard, Shift};
use bitboard::eleven::{BoardEleven, MoveOnBoardEleven};
use bitboard::seven::BoardSeven;
use bitflags::{bitflags, bitflags_match};

use crate::game::{PieceType, Side, InvalidActionError};

pub trait StdPreset {
    type Int;
    const PRESET_STD_ATT: Self::Int;
    const PRESET_STD_DEF: Self::Int;
    const PRESET_STD_KING: Self::Int;
    const PRESET_STD_HOS: Self::Int;
}

// The most standard Hnefatafl position
impl StdPreset for BoardEleven{
    type Int = [u64; 3];
    const PRESET_STD_ATT: Self::Int = [0x401_000_020_0F8, 0x401_401_603_401, 0x0F8_020_000];
    const PRESET_STD_DEF: Self::Int = [0x020_000_000_000, 0x020_070_0D8_070, 0x000_000_000];
    const PRESET_STD_KING: Self::Int =[0x000_000_000_000, 0x000_000_020_000, 0x000_000_000] ;
    const PRESET_STD_HOS: Self::Int = [0x000_000_000_401, 0x000_000_020_000, 0x401_000_000];
} 
// This position is based on Brandubh rule. It is probably the simplest position.
impl StdPreset for BoardSeven{
    type Int = u64;
    const PRESET_STD_ATT: Self::Int = 0x08_08_00_63_00_08_08;
    const PRESET_STD_DEF: Self::Int = 0x00_00_08_14_08_00_00;
    const PRESET_STD_KING: Self::Int = 0x00_00_00_08_00_00_00;
    const PRESET_STD_HOS: Self::Int = 0x41_00_00_00_00_00_41;
}

#[derive(Debug, Copy, Clone)]
pub struct TaflBoard<B: BitBoard>{
    pub bit_att: B,
    pub bit_def: B,
    pub bit_king: B,
    pub hostile: B,
}

impl<B: BitBoard> PartialEq for TaflBoard<B> {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl<B: BitBoard + StdPreset> TaflBoard<B>
where B: From<<B as StdPreset>::Int>
{

    pub fn init_std() -> Self{
        let bit_att = B::from(<B as StdPreset>::PRESET_STD_ATT);
        let bit_def = B::from(<B as StdPreset>::PRESET_STD_DEF);
        let bit_king = B::from(<B as StdPreset>::PRESET_STD_KING);
        let hostile = B::from(<B as StdPreset>::PRESET_STD_HOS);
        Self{bit_att, bit_def, bit_king, hostile}
    } 

    pub fn generate_random_board<T>(rng: &mut T) -> Self
    where T: Rng{
        let king_loc = (rng.gen_range(0..Self::BOARD_SIZE), rng.gen_range(0..Self::BOARD_SIZE));
        let mut king_board: B = B::ghastly();
        king_board.flip_target_bit_mut(&B::Position::new(king_loc.0, king_loc.1));
        let hostile = B::from(<B as StdPreset>::PRESET_STD_HOS);
        let mut mask = hostile.complement() & king_board.complement();
        if rng.gen_bool(0.5){
            let att_board = B::random(rng) & mask;
            mask &= att_board.complement();
            let def_board = B::random(rng) & mask;
            Self::new(att_board, def_board, king_board, hostile)
        } else {
            let def_board = B::random(rng) & mask;
            mask &= def_board.complement();
            let att_board = B::random(rng) & mask;
            Self::new(att_board, def_board, king_board, hostile)
        }
    }
}

impl<B: BitBoard> TaflBoard<B> {

    #[allow(dead_code)]
    const N: usize = B::Position::BOARD_SIZE as usize;
    const BOARD_SIZE: u8 = B::Position::BOARD_SIZE;
    const MOVELEN_LIM: u8 = B::Position::BOARD_SIZE - 1;

    pub fn equals(&self, rhs: &Self) -> bool{
        self.bit_att.equals(&rhs.bit_att) && self.bit_def.equals(&rhs.bit_def) && self.bit_king.equals(&rhs.bit_king)
    }
    pub fn new(bit_att: B, bit_def: B, bit_king: B, hostile: B) -> Self{
        Self{bit_att, bit_def, bit_king, hostile}
    }
    pub fn generate_actions_for_attsoldiers(&self) -> Vec<B::Movement>{
        let mut move_list: Vec<B::Movement> = Vec::new();
        let start = self.bit_att;
        let mask = self.hostile.complement();
        let barricade = (self.bit_att | self.bit_def | self.bit_king).complement();

        move_list.extend(start.list_moves_east(&mask, &barricade, Self::MOVELEN_LIM));
        move_list.extend(start.list_moves_west(&mask, &barricade, Self::MOVELEN_LIM));
        move_list.extend(start.list_moves_south(&mask, &barricade, Self::MOVELEN_LIM));
        move_list.extend(start.list_moves_north(&mask, &barricade, Self::MOVELEN_LIM));

        move_list
    }
    pub fn generate_actions_for_defsoldiers(&self) -> Vec<B::Movement>{
        let mut move_list: Vec<B::Movement> = Vec::new();
        let start = self.bit_def;
        let mask = self.hostile.complement();
        let barricade = (self.bit_att | self.bit_def | self.bit_king).complement();

        move_list.extend(start.list_moves_east(&mask, &barricade, Self::MOVELEN_LIM));
        move_list.extend(start.list_moves_west(&mask, &barricade, Self::MOVELEN_LIM));
        move_list.extend(start.list_moves_south(&mask, &barricade, Self::MOVELEN_LIM));
        move_list.extend(start.list_moves_north(&mask, &barricade, Self::MOVELEN_LIM));

        move_list
    }

    pub fn generate_actions_for_king(&self) -> Vec<B::Movement>{
        let mut move_list: Vec<B::Movement> = Vec::new();
        let start = self.bit_king;
        let mask = B::BLACKMASK;
        let barricade = (self.bit_att | self.bit_def).complement();

        move_list.extend(start.list_moves_east(&mask, &barricade, Self::MOVELEN_LIM));
        move_list.extend(start.list_moves_west(&mask, &barricade, Self::MOVELEN_LIM));
        move_list.extend(start.list_moves_south(&mask, &barricade, Self::MOVELEN_LIM));
        move_list.extend(start.list_moves_north(&mask, &barricade, Self::MOVELEN_LIM));

        move_list

    }

    pub fn list_def_captures(&self) -> Vec<B::Position>{
        let mut list = self.bit_def.list_horizontal_pincer(&(self.bit_att | self.hostile));
        let list2 = self.bit_def.list_vertical_pincer(&(self.bit_att | self.hostile));
        list.extend(list2);
        list 
    }

    pub fn list_att_captures(&self) -> Vec<B::Position>{
        let mut list = self.bit_att.list_horizontal_pincer(&(self.bit_def | self.hostile));
        let list2 = self.bit_att.list_vertical_pincer(&(self.bit_def | self.hostile));
        list.extend(list2);
        list 
    }

    pub fn list_king_captures(&self) -> Vec<B::Position>{
        self.bit_king.list_besieged(&(self.bit_att | self.hostile))
    }

    pub fn att_force_move(&self, action: &B::Movement) -> Self {
        let new_bit_add = self.bit_att.force_move(action);
        Self {
            bit_att: new_bit_add,
            bit_def: self.bit_def,
            bit_king: self.bit_king,
            hostile: self.hostile,
        }
    }

    pub fn def_force_move(&self, action: &B::Movement) -> Self {
        let new_bit_def = self.bit_def.force_move(action);
        Self {
            bit_att: self.bit_att,
            bit_def: new_bit_def,
            bit_king: self.bit_king,
            hostile: self.hostile,
        }
    }

    pub fn king_force_move(&self, action: &B::Movement) -> Self {
        let new_bit_king = self.bit_king.force_move(action);
        Self {
            bit_att: self.bit_att,
            bit_def: self.bit_def,
            bit_king: new_bit_king,
            hostile: self.hostile,
        }
    }

    pub fn att_force_move_mut(&mut self, action: &B::Movement){
        self.bit_att.force_move_mut(action);
    }

    pub fn def_force_move_mut(&mut self, action: &B::Movement){
        self.bit_def.force_move_mut(action);
    }

    pub fn king_force_move_mut(&mut self, action: &B::Movement){
        self.bit_king.force_move_mut(action);
    }

    pub fn def_is_encircled(&self) -> bool{
        let mut tmp = self.bit_def | self.bit_king;
        let mask = self.bit_att.complement();
        for _i in 0..11{
            if (tmp & B::ONLY_EDGES).is_nonzero() { return false }
            else {
                // tmp = tmp.dilation(bitboard::Direction::All(i)) & mask;
                tmp = (tmp.shift_e() | tmp.shift_n() | tmp.shift_s() | tmp.shift_w()) & mask;
            }
        }
        true
    }

    pub fn determine_action_piecetype(&self, side: Side, action: &B::Movement) -> Result<PieceType, InvalidActionError> {
        match side{
            Side::Att => {
                if !self.bit_att.tile_is_empty_at(&action.start()) {
                    Ok(PieceType::AttSoldier)
                } else { Err(InvalidActionError::NO_PIECE_AT_STARTINGPOS) }
            },
            Side::Def => {
                let def_has_piece_there: bool = !self.bit_def.tile_is_empty_at(&action.start());
                let king_has_piece_there: bool = !self.bit_king.tile_is_empty_at(&action.start());
                if def_has_piece_there && !king_has_piece_there { Ok(PieceType::DefSoldier) }
                else if !def_has_piece_there && king_has_piece_there { Ok(PieceType::King) }
                else if !def_has_piece_there && !king_has_piece_there { Err(InvalidActionError::NO_PIECE_AT_STARTINGPOS) }
                else { panic!("Defender soldiers and king has an overlap!")}
            }
        }
    }

    // returns the defender pieces that are captured by the movement from the attacker pieces.
    // To actually update the board, simply take the XOR of the output with self.bit_def 
    // NOTE!: Call this function on the board BEFORE the action is applied to
    pub fn def_capture(&self, action: &B::Movement) -> B{
        let dst_neighbor = B::neighbor_of(&action.dst());
        let restricted_def = self.bit_def & dst_neighbor;
        let mut mask = self.bit_att | self.hostile;
        mask.flip_target_bit_mut(&action.dst());
        let cpt_candid_e = (restricted_def.shift_e() & mask).shift_w();
        let cpt_candid_w = (restricted_def.shift_w() & mask).shift_e();
        let cpt_candid_s = (restricted_def.shift_s() & mask).shift_n();
        let cpt_candid_n = (restricted_def.shift_n() & mask).shift_s();
        (cpt_candid_e & cpt_candid_w) | (cpt_candid_n & cpt_candid_s)
    }
    // returns the attacker pieces that are captured by the movement from the defender pieces.
    // To actually update the board, simply take the XOR of the output with self.bit_att 
    // NOTE!: Call this function on the board BEFORE the action is applied to
    pub fn att_capture(&self, action: &B::Movement) -> B{
        let dst_neighbor = B::neighbor_of(&action.dst());
        let restricted_att = self.bit_att & dst_neighbor;
        let mut mask = self.bit_def | self.bit_king | self.hostile;
        mask.flip_target_bit_mut(&action.dst());
        let cpt_candid_e = (restricted_att.shift_e() & mask).shift_w();
        let cpt_candid_w = (restricted_att.shift_w() & mask).shift_e();
        let cpt_candid_s = (restricted_att.shift_s() & mask).shift_n();
        let cpt_candid_n = (restricted_att.shift_n() & mask).shift_s();
        (cpt_candid_e & cpt_candid_w) | (cpt_candid_n & cpt_candid_s)
        
    }
    // returns the king pieces that are captured by the movement from the attacker pieces.
    // To actually update the board, simply take the XOR of the output with self.bit_king 
    // NOTE!: Call this function on the board BEFORE the action is applied to
    pub fn king_capture(&self, action: &B::Movement) -> B{
        let dst_neighbor = B::neighbor_of(&action.dst());
        let restricted_king = self.bit_king & dst_neighbor;
        if !restricted_king.is_nonzero(){ return B::empty() }

        let mut mask = self.bit_att | self.hostile;
        mask.flip_target_bit_mut(&action.dst());
        let cpt_candid_e = (restricted_king.shift_e() & mask).shift_w();
        let cpt_candid_w = (restricted_king.shift_w() & mask).shift_e();
        let cpt_candid_s = (restricted_king.shift_s() & mask).shift_n();
        let cpt_candid_n = (restricted_king.shift_n() & mask).shift_s();
        cpt_candid_e & cpt_candid_w & cpt_candid_n & cpt_candid_s
        
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
    fn generate<B: BitBoard>(board: &TaflBoard<B>) -> Self{
        let mut flag: BoardErrorFlag = BoardErrorFlag::ZERO;
        if (board.bit_att & board.bit_def).is_nonzero() {flag |= BoardErrorFlag::AD}
        if (board.bit_att & board.bit_king).is_nonzero() {flag |= BoardErrorFlag::AK}
        if (board.bit_def & board.bit_king).is_nonzero() {flag |= BoardErrorFlag::DK}
        if (board.bit_att & board.hostile).is_nonzero() {flag |= BoardErrorFlag::AR}
        if (board.bit_def & board.hostile).is_nonzero() {flag |= BoardErrorFlag::DR}
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

// helper function for the board display
#[allow(clippy::too_many_arguments)]
fn draw(
    att_b: u64,
    def_b: u64,
    king_b: u64,
    hos_b: u64,
    f: &mut std::fmt::Formatter<'_>,
    bound: usize,
    pad: usize,
    internal_counter: &mut u8,
) -> Result<(), std::fmt::Error> {
    for i in 1..=bound{
        if i > bound {
            break;
        }

        if i % pad == 0{
            *internal_counter += 1;
            if *internal_counter > pad as u8 - 1{
                writeln!(f)?;
            } else if *internal_counter >= 10 {
                write!(f, "\n{internal_counter} ")?;
            } else {
                write!(f, "\n{internal_counter}  ")?;
            }
            continue;
        }

        if (att_b >> (i-1)) & 1 == 1{
            write!(f, "A ")?;
            continue;
        }
        if (def_b >> (i-1)) & 1 == 1{
            write!(f, "D ")?;
            continue;
        }
        if (king_b >> (i-1)) & 1 == 1{
            write!(f, "K ")?;
            continue;
        }
        if (hos_b >> (i-1)) & 1 == 1{
            write!(f, "\u{2612} " )?; 
            // cross inside square
            continue;
        }
        write!(f, "\u{25A1} ")?;
        // Empty square

    }
    Ok(())
}

impl Display for TaflBoard<BoardEleven>{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let be = BoardError::generate(self);
        if be.flag != BoardErrorFlag::ZERO{
            writeln!(f, "This board contains an abnormality, abandoning printing..")?;
            writeln!(f, "{be}")?;
            return Ok(())
        } 
        write!(f, "   A B C D E F G H I J K\n1  ")?;
        let mut internal_counter: u8 = 1;
        draw(self.bit_att.par1, self.bit_def.par1,
        self.bit_king.par1, self.hostile.par1 , f, 48, 12, &mut internal_counter)?;

        draw(self.bit_att.par2, self.bit_def.par2,
        self.bit_king.par2, self.hostile.par2, f, 48, 12, &mut internal_counter)?;
        
        draw(self.bit_att.par3, self.bit_def.par3,
        self.bit_king.par3, self.hostile.par3, f, 36, 12, &mut internal_counter)?;
        Ok(())
    }
}

impl Display for TaflBoard<BoardSeven> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let be = BoardError::generate(self);
        if be.flag != BoardErrorFlag::ZERO{
            writeln!(f, "This board contains an abnormality, abandoning printing..")?;
            writeln!(f, "{be}")?;
            return Ok(())
        } 
        write!(f, "   A B C D E F G\n1  ")?;
        let mut internal_counter: u8 = 1;
        draw(self.bit_att.0, self.bit_def.0,
        self.bit_king.0, self.hostile.0 , f, 56, 8, &mut internal_counter)?;

        Ok(())
    }
}

// We implement shield wall rule only for 11 * 11 board.
impl TaflBoard<BoardEleven>{

    pub fn shield_wall_capture(&self, action: &MoveOnBoardEleven) -> BoardEleven{
        enum Wall{ East, West, South, North}
        fn detect_wall(block: u8, address: u8) -> Option<Wall>{
            if address % 12 == 0 {return Some(Wall::West)}
            else if address % 12 == 10 {return Some(Wall::East)}
            if block == 0 && address < 11 {return Some(Wall::North)}
            else if block == 2 && address >= 24 {return Some(Wall::South)}
            None
        }

        fn bridges_shield_wall_north(
        singleton: &BoardEleven,
        bit_att: &BoardEleven,
        hostile: &BoardEleven,
        path: &BoardEleven
        ) -> BoardEleven{
            let mut tmp = *singleton;
            let mut count: Option<u8> = None;
            for i in 0..10{
                tmp = tmp.shift_n();
                if (tmp & *bit_att).is_nonzero(){ 
                    if i >= 2 {
                        count = Some(i);
                        break; 
                    } else {
                        count = None;
                        break;
                    }
                }
                if (tmp & *hostile).is_nonzero(){
                    count = None;
                    break;
                }
            }
            if let Some(c ) = count{
                let mut bridge = *singleton;
                let mut rev_bridge = tmp;
                for _ in 0..c{
                    bridge = (bridge | bridge.shift_n()) & *path;
                    rev_bridge = (rev_bridge | rev_bridge.shift_s()) & *path;
                }
                bridge & rev_bridge} 
            else { BoardEleven{par1: 0, par2: 0, par3: 0}}
        }

        fn bridges_shield_wall_south(
        singleton: &BoardEleven,
        bit_att: &BoardEleven,
        hostile: &BoardEleven,
        path: &BoardEleven
        ) -> BoardEleven{
            let mut tmp = *singleton;
            let mut count: Option<u8> = None;
            for i in 0..10{
                tmp = tmp.shift_s();
                if (tmp & *bit_att).is_nonzero(){ 
                    if i >= 2 {
                        count = Some(i);
                        break; 
                    } else {
                        count = None;
                        break;
                    }
                }
                if (tmp & *hostile).is_nonzero(){
                    count = None;
                    break;
                }
            }
            if let Some(c ) = count{
                let mut bridge = *singleton;
                let mut rev_bridge = tmp;
                for _ in 0..c{
                    bridge = (bridge | bridge.shift_s()) & *path;
                    rev_bridge = (rev_bridge | rev_bridge.shift_n()) & *path;
                }
                bridge & rev_bridge} 
            else { BoardEleven{par1: 0, par2: 0, par3: 0}}
        }

        fn bridges_shield_wall_east(
        singleton: &BoardEleven,
        bit_att: &BoardEleven,
        hostile: &BoardEleven,
        path: &BoardEleven
        ) -> BoardEleven{
            let mut tmp = *singleton;
            let mut count: Option<u8> = None;
            for i in 0..10{
                tmp = tmp.shift_e();
                if (tmp & *bit_att).is_nonzero(){ 
                    if i >= 2 {
                        count = Some(i);
                        break; 
                    } else {
                        count = None;
                        break;
                    }
                }
                if (tmp & *hostile).is_nonzero(){
                    count = None;
                    break;
                }
            }
            if let Some(c ) = count{
                let mut bridge = *singleton;
                let mut rev_bridge = tmp;
                for _ in 0..c{
                    bridge = (bridge | bridge.shift_e()) & *path;
                    rev_bridge = (rev_bridge | rev_bridge.shift_w()) & *path;
                }
                bridge & rev_bridge} 
            else { BoardEleven{par1: 0, par2: 0, par3: 0}}
        }

        fn bridges_shield_wall_west(
        singleton: &BoardEleven,
        bit_att: &BoardEleven,
        hostile: &BoardEleven,
        path: &BoardEleven
        ) -> BoardEleven{
            let mut tmp = *singleton;
            let mut count: Option<u8> = None;
            for i in 0..10{
                tmp = tmp.shift_w();
                if (tmp & *bit_att).is_nonzero(){ 
                    if i >= 2 {
                        count = Some(i);
                        break; 
                    } else {
                        count = None;
                        break;
                    }
                }
                if (tmp & *hostile).is_nonzero(){
                    count = None;
                    break;
                }
            }
            if let Some(c ) = count{
                let mut bridge = *singleton;
                let mut rev_bridge = tmp;
                for _ in 0..c{
                    bridge = (bridge | bridge.shift_w()) & *path;
                    rev_bridge = (rev_bridge | rev_bridge.shift_e()) & *path;
                }
                bridge & rev_bridge} 
            else { BoardEleven{par1: 0, par2: 0, par3: 0}}
        }

        let block = action.dst.0 & 3;
        let address = action.dst.0 >> 2;
        let dst = match block{
            0 => BoardEleven{par1: 1 << address, par2: 0, par3: 0},
            1 => BoardEleven { par1: 0, par2: 1 << address, par3: 0 },
            2 => BoardEleven { par1: 0, par2: 0, par3: 1 << address },
            _ => panic!("during shield_wall_capture, invalid action was detected")
        };

        if let Some(wall) = detect_wall(block, address){
            match wall{
                Wall::East => {
                    let path = (self.bit_def | self.bit_king) & self.bit_att.shift_e();
                    let bridge_north = bridges_shield_wall_north(
                        &dst, &self.bit_att, &self.hostile, &path);
                    let bridge_south = bridges_shield_wall_south(
                        &dst, &self.bit_att, &self.hostile, &path);
                    
                    (bridge_south | bridge_north) & (self.bit_def | self.bit_king)
                },

                Wall::West => {
                    let path = (self.bit_def | self.bit_king) & self.bit_att.shift_w();
                    let bridge_north = bridges_shield_wall_north(
                        &dst, &self.bit_att, &self.hostile, &path);
                    let bridge_south = bridges_shield_wall_south(
                        &dst, &self.bit_att, &self.hostile, &path);
                    
                    (bridge_south | bridge_north) & (self.bit_def | self.bit_king)
                },

                Wall::North => {
                    let path = (self.bit_def | self.bit_king) & self.bit_att.shift_n();
                    let bridge_east = bridges_shield_wall_east(
                        &dst, &self.bit_att, &self.hostile, &path);
                    let bridge_west = bridges_shield_wall_west(
                        &dst, &self.bit_att, &self.hostile, &path);
                    
                    (bridge_east | bridge_west) & (self.bit_def | self.bit_king)
                },
                Wall::South => {
                    let path = (self.bit_def | self.bit_king) & self.bit_att.shift_s();
                    let bridge_east = bridges_shield_wall_east(
                        &dst, &self.bit_att, &self.hostile, &path);
                    let bridge_west = bridges_shield_wall_west(
                        &dst, &self.bit_att, &self.hostile, &path);
                    
                    (bridge_east | bridge_west) & (self.bit_def | self.bit_king)
                },
            }
        } else {BoardEleven{par1: 0, par2: 0, par3: 0}}
    }
    // Shield Wall capture refers to the relatively new capture rule in Copenhagen ruleset, where a row of defenders + king on 
    // one of the edges on the board can be captured if they are tightly surrounded by attackers. However, for that to happen the attacker must make 
    // a flanking move.
    // This function detects the tight entrapment of the defenders's shield walls by simulating a dilation starting from the proper position. 
    // Returns: union of defender_board and king_board with the captured pieces removed.
    pub fn shield_wall_capture_tmp(&self) -> BoardEleven{
        let shield_walls = (self.bit_def | self.bit_king) & BoardEleven::ONLY_EDGES;
        if !shield_walls.is_nonzero() { return self.bit_def | self.bit_king }
        let mask = self.bit_att.complement();
        let mut well = ( self.bit_att | shield_walls ).complement();
        for _ in 0..10{
            well = (well | well.shift_e() | well.shift_w() | well.shift_s() | well.shift_n()) & mask;
            if !(well.complement() & shield_walls).is_nonzero(){
                return self.bit_def | self.bit_king
            }
        }
        well & shield_walls | ( (self.bit_def | self.bit_king) & !BoardEleven::ONLY_EDGES )

    }

}

#[cfg(test)]
mod tests{

    use super::*;
    use bitboard::eleven::{BoardEleven as BE};
    use bitboard::seven::{BoardSeven as BS};

    #[test] 
    fn board_print_works1(){
        let b = TaflBoard::<BE>::init_std();
        println!("{}", b);
        let b = TaflBoard::<BS>::init_std();
        println!("{}", b);
    }

    #[test] 
    fn board_print_works2(){
        let mut b = TaflBoard::<BE>::init_std();
        let def_board = BoardEleven::from(BE::PRESET_STD_DEF);
        b.bit_att = def_board;
        // The Attacker's board is the same as Defender's board, so
        // this should NOT print the board.
        // Check with cargo run -- board_print_works2 --show-output 
        println!("{}", b);
        let e = BoardError::generate(&b);
        print!("{:?}", e);
        assert!((b.bit_att & b.bit_def).is_nonzero());

    }

    #[test] 
    fn generate_actions_for_attsoldiers_works() {
        let b = TaflBoard::<BE>::init_std();
        let attsoldiers_moves = b.generate_actions_for_attsoldiers();
        println!("{}", b);
        println!("Attsoldiers' moves");
        for m in attsoldiers_moves{
            print!(" {} ", m);
        }
    }

    #[test]
    fn list_captures_works() {
        let mut rng = thread_rng();
        let b = TaflBoard::<BE>::generate_random_board(&mut rng);
        println!("{}", b);
        let list = b.list_att_captures();
        println!("The captured attacker soldiers are: ");
        for pos in list{
            print!("{} ", pos);
        }
        let list = b.list_def_captures();
        println!("The captured defender soldiers are: ");
        for pos in list{
            print!("{} ", pos);
        }
        let list = b.list_king_captures();
        println!("The captured king is: ");
        if list.len() == 0{
            println!("None");
        }
        for pos in list{
            print!("{} ", pos);
        }
    }

    #[test]
    fn shield_wall_capture_works(){
        let b = TaflBoard{
            bit_att: BoardEleven::from([0x000_020_01c_002, 0x000_000_000_000, 0x000_000_000]),
            bit_def: BoardEleven::from([0x000_000_000_01c, 0x000_000_000_000, 0x000_000_000]),
            bit_king: BoardEleven::from(BE::PRESET_STD_KING),
            hostile: BoardEleven::from(BE::PRESET_STD_HOS),
        };
        let action: MoveOnBoardEleven = MoveOnBoardEleven::try_from("F3F1".to_owned()).unwrap();
        println!("{}", b);
        let val = b.shield_wall_capture(&action);
        println!("{}", val);
    }

    #[test]
    fn action_on_board_works() {
        let b = TaflBoard::<BE>::init_std();
        let action = <BE as BitBoard>::Movement::try_from("D1D2".to_owned()).unwrap();
        let nb = b.att_force_move(&action);
        println!("{}", b);
        println!("{}", nb);
    }

}