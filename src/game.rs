use crate::board::*;
use bitboard::eleven::*;
use bitboard::Direction;

use bitflags::{bitflags, bitflags_match, Flag, Flags};

pub enum Side { Att, Def}
pub enum PieceType { AttSoldier, DefSoldier, King}

impl PieceType{
    pub fn aligns_with_side(&self, side: Side) -> bool{
        match (self, side){
            (PieceType::AttSoldier, Side::Att) => true,
            (PieceType::DefSoldier, Side::Def) | (PieceType::King , Side::Def) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GameState(u8);

bitflags! {
    impl GameState: u8{
        const TURN_ATT =  1 << 2;
        const TERMINATED = 1 << 3;
    }
}

impl GameState {
    pub fn new() -> Self {
        Self::from_bits_retain(0b0000_0100)
    }
    pub fn is_ongoing(&self) -> bool{
        (Self::TERMINATED.bits() & self.bits()) == 0
    }
    pub fn is_turn_att(&self) -> bool{
        self.bits() & Self::TURN_ATT.bits() > 0
    }
    pub fn show_side(&self) -> Side{
        if self.is_turn_att() { Side::Att}
        else { Side::Def}
    }
    pub fn show_repetition_count(&self) -> u8{
        self.bits() & 3
    }
    pub fn repetition_count_up(&mut self){
        self.0 += 1;
    }
    pub fn reset_repetition_count(&mut self){
        self.0 &= 0b1111_1100;
    }
    pub fn change_turn(&mut self) {
        self.0 ^= 0b0000_0100;
    }
    pub fn game_over(&mut self) {
        self.0 ^= 0b0000_1000;
    }
}

impl std::fmt::Display for GameState{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s1: &str = if self.is_ongoing() { "ongoing" } else {"over"};
        let s2: &str = if self.is_turn_att() { "Attacker"} else {"Defender"};
        let s3: &str = &format!("{}", self.show_repetition_count());
        let msg: String = format!("The game is {}. {}'s turn. The repetition count is {} ", s1, s2, s3);
        write!(f, "{}", msg)?;
        Ok(())
    }
}


bitflags!{
    #[derive(Debug)]
    pub struct InvalidActionError: u8{
        const NO_PIECE_AT_STARTINGPOS = 1;
        const OBSTACLE_IN_PATH = 1 << 1;
        const THIRD_REPETITION = 1 << 2;
        const DST_IS_RESTRICTED = 1 << 3;
        const PIECETYPE_CONTRADICTS_STATE = 1 << 4;
        const MOVEMENT_NOT_ORTHOGONAL = 1 << 5;
    }
}

pub struct Game{
    pub board: TaflBoardEleven,
    pub state: GameState,
}

impl Game{
    pub fn from_board(board: TaflBoardEleven, side: Side) -> Self{
        let state = match side{
            Side::Att => GameState(0b0000_0100),
            Side::Def => GameState(0b0000_0000),
        };
        Self { board, state }
    }
    pub fn from_board_and_state(board: TaflBoardEleven, state: GameState) -> Self{
        Self{ board, state}
    }
    pub fn init_std() -> Self{
        let board = TaflBoardEleven::init_std();
        let state = GameState::new();
        Self { board, state}
    }
    pub fn check_action_validity(&self, action: &MoveOnBoardEleven, piece_type: &Option<PieceType>) -> Result<(), InvalidActionError>{

        let mut e = InvalidActionError::from_bits_retain(0);
        let playing_side = self.state.show_side();

        let opt = action.find_path();
        if let None = opt{
            e.insert(InvalidActionError::MOVEMENT_NOT_ORTHOGONAL);
        }

        // If the piece_type is specified, then we check the validity assuming that the movement happens for the specified piece.
        if let Some(p) = piece_type {
            if !p.aligns_with_side(playing_side) { e.insert(InvalidActionError::PIECETYPE_CONTRADICTS_STATE);}
            
            // check the starting position
            let target_board = match p{
                PieceType::AttSoldier => self.board.bit_att,
                PieceType::DefSoldier => self.board.bit_def,
                PieceType::King => self.board.bit_king,
            };
            if target_board.tile_is_empty_at(&action.start) { e.insert(InvalidActionError::NO_PIECE_AT_STARTINGPOS); }

            // Next, check the movement path and see if there is any obstacle
            if let Some(dir) = opt{

                let barricade = self.board.bit_att | self.board.bit_def | self.board.bit_king;
                let restricted = match p{
                    PieceType::AttSoldier | PieceType::DefSoldier => self.board.hostile,
                    PieceType::King => BoardEleven::new(),
                };
                check_path_obstacle(&dir, &action, &barricade, &restricted, &mut e);
            }

        } else {
            // If no piece type is specified, we postulate from the game state and board information which board to consider the movement for.
            match playing_side{
                Side::Att => {
                    // First, we check the starting position
                    if self.board.bit_att.tile_is_empty_at(&action.start) { e.insert(InvalidActionError::NO_PIECE_AT_STARTINGPOS);}

                    //Next, we check the movement path and see if there is any obstacle
                    if let Some(dir) = opt{
                        let barricade = self.board.bit_att | self.board.bit_def | self.board.bit_king;
                        let restricted = self.board.hostile;
                        check_path_obstacle(&dir, &action, &barricade, &restricted, &mut e);
                    }
                },
                Side::Def => {
                    // First, we check the starting position
                    if self.board.bit_def.tile_is_empty_at(&action.start) 
                        && self.board.bit_king.tile_is_empty_at(&action.start) {
                             e.insert(InvalidActionError::NO_PIECE_AT_STARTINGPOS)
                        }
                    
                    if let Some(dir) = opt {
                        if !self.board.bit_king.tile_is_empty_at(&action.start) {
                            let barricade = self.board.bit_att | self.board.bit_def | self.board.bit_king;
                            let restricted = BoardEleven::new();
                            check_path_obstacle(&dir, &action, &barricade, &restricted, &mut e);
                        } else {
                            let barricade = self.board.bit_att | self.board.bit_def | self.board.bit_king;
                            let restricted = self.board.hostile;
                            check_path_obstacle(&dir, &action, &barricade, &restricted, &mut e);
                        }
                    }
                }
            }
        }


        fn check_path_obstacle(dir: &Direction, action: &MoveOnBoardEleven, barricade: &BoardEleven, restricted: &BoardEleven, e: &mut InvalidActionError){
                let mut tmp = BoardEleven::new().flip_target_bit(&action.start);
                // let barricade = self.board.bit_att | self.board.bit_def | self.board.bit_king;
                // let restricted = match p{
                //     PieceType::AttSoldier | PieceType::DefSoldier => self.board.hostile,
                //     PieceType::King => BoardEleven::new(),
                // };

                match *dir{
                    Direction::E(step) => {
                        for i in 0..step{
                            tmp = tmp.shift_e();
                            if i == step - 1 {
                                if (tmp & *restricted).is_nonzero() { e.insert(InvalidActionError::OBSTACLE_IN_PATH);}
                            }
                            if (tmp & *barricade).is_nonzero() { e.insert(InvalidActionError::DST_IS_RESTRICTED);}
                        }
                    },
                    Direction::W(step) => {
                        for i in 0..step{
                            tmp = tmp.shift_w();
                            if i == step - 1 {
                                if (tmp & *restricted).is_nonzero() { e.insert(InvalidActionError::OBSTACLE_IN_PATH);}
                            }
                            if (tmp & *barricade).is_nonzero() { e.insert(InvalidActionError::DST_IS_RESTRICTED);}
                        }
                    },
                    Direction::S(step) => {
                        for i in 0..step{
                            tmp = tmp.shift_s();
                            if i == step - 1 {
                                if (tmp & *restricted).is_nonzero() { e.insert(InvalidActionError::OBSTACLE_IN_PATH);}
                            }
                            if (tmp & *barricade).is_nonzero() { e.insert(InvalidActionError::DST_IS_RESTRICTED);}
                        }
                    },
                    Direction::N(step) => {
                        for i in 0..step{
                            tmp = tmp.shift_n();
                            if i == step - 1 {
                                if (tmp & *restricted).is_nonzero() { e.insert(InvalidActionError::OBSTACLE_IN_PATH);}
                            }
                            if (tmp & *barricade).is_nonzero() { e.insert(InvalidActionError::DST_IS_RESTRICTED);}
                        }
                    },
                    _ => {}
                }
            }

        if e.bits() == 0 { Ok(()) } else { Err(e) }
    }

    // recommended to check the validity of the action before. It ALLOWS repetitive action
    pub fn do_action_wo_validity_check(&self, action: &MoveOnBoardEleven, piece_type: &Option<PieceType>) -> Self{
        // let e = self.check_action_validity(&action, piece_type)?;
        let side = self.state.show_side();
        match side{
            Side::Att => {
                let mut new_board = self.board.att_force_move(action);
                let def_capt = self.board.def_capture(action);
                new_board.bit_def ^= def_capt;
                let king_capt = self.board.king_capture(action);
                new_board.bit_king ^= king_capt;
                let shield_wall_capt = self.board.shield_wall_capture(action);
                new_board.bit_def ^= shield_wall_capt & self.board.bit_def;
                new_board.bit_king ^= shield_wall_capt & self.board.bit_king;
                let mut new_state = self.state.clone();
                new_state.change_turn();
                if king_capt.is_nonzero(){
                    new_state.game_over();
                }
                if new_board.def_is_encircled() {
                    new_state.game_over();
                }
                Self { board: new_board, state: new_state }
                
            },
            Side::Def => {
                // If the user specified which piece to move (defender soldier or king), we trust that information
                let mut new_board = if let Some(p) = piece_type {
                    match p{
                        PieceType::DefSoldier => self.board.def_force_move(action),
                        PieceType::King => self.board.king_force_move(action),
                        PieceType::AttSoldier => panic!("???"),
                    }
                } else {
                    if !self.board.bit_def.tile_is_empty_at(&action.start) {
                        self.board.def_force_move(action)
                    } else if !self.board.bit_king.tile_is_empty_at(&action.start) {
                        self.board.king_force_move(action)
                    } else { panic!("???") }
                };
                let att_capt = self.board.att_capture(action);
                new_board.bit_att ^= att_capt;
                let mut new_state = self.state;
                new_state.change_turn();
                if !new_board.bit_att.is_nonzero() {
                    new_state.game_over();
                }
                if (new_board.bit_king & GOALS).is_nonzero() {
                    new_state.game_over();
                } 
                Self {board: new_board, state: new_state }
            },
        }
        
    }
}



#[test]
fn gamestate_display_works() {
    let game = Game{
        board: TaflBoardEleven::init_std(),
        state: GameState::new(),
    };
    print!("{}", game.state);
}

