use std::fmt::Display;

use crate::board::*;
use bitboard::eleven::*;
use bitboard::Direction;

use bitflags::{bitflags, bitflags_match, Flag, Flags};
use arraydeque::{ArrayDeque, Wrapping};

pub enum Side { Att, Def}

impl Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self{
            Side::Att => write!(f, "Attacker"),
            Side::Def => write!(f, "Defender"),
        }
    }
}
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
// 1st bit is 1 if (current player = attacker), 0 else
// 2nd bit is 1 if (game is over on THIS turn) 
// <- the last player made a winning move 
// - this includes such a move that the opponent would have no move, or only the 3rd repetition)
// 3rd ~ 8th bits: so far no use
// 9~16th bits: turn counter (maybe useful at some point)
#[derive(Clone, Copy, Debug)]
pub struct GameState(u16);

bitflags! {
    impl GameState: u16{
        const TURN_ATT =  1 << 0;
        const TERMINATED = 1 << 1;
        const ATTACKER_VICTORY = 1 << 2;
    }
}

impl GameState {
    pub fn new() -> Self {
        Self::from_bits_retain(0x00_01)
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
    // pub fn show_repetition_count(&self) -> u8{
    //     self.bits() & 3
    // }
    // pub fn repetition_count_up(&mut self){
    //     self.0 += 1;
    // }
    // pub fn reset_repetition_count(&mut self){
    //     self.0 &= 0b1111_1100;
    // }
    pub fn change_side_mut(&mut self) {
        self.0 ^= Self::TURN_ATT.bits();
    }
    pub fn change_side(&self) -> Self{
        Self(self.0 ^ Self::TURN_ATT.bits())
    }
    pub fn game_over_mut(&mut self) {
        self.0 ^= Self::TERMINATED.bits();
    }
    pub fn game_over(&self) -> Self {
        Self(self.0 ^ Self::TERMINATED.bits())
    }
    pub fn get_turn_count(&self) -> u8 {
        let count = self.bits() >> 8;
        count as u8
    }
    pub fn incre_turn_count_mut(&mut self) {
        if self.0 >> 8 < std::u8::MAX as u16 {
            self.0 += 1 << 8;
        }
    }
    pub fn incre_turn_count(&self) -> Self {
        Self(self.0 + 1 << 8)
    }
}

impl std::fmt::Display for GameState{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s1: &str = if self.is_ongoing() { "ongoing" } else {"over"};
        let s2: &str = if self.is_turn_att() { "Attacker"} else {"Defender"};
        // let s3: &str = &format!("{}", self.show_repetition_count());
        // let msg: String = format!("The game is {}. {}'s turn. The repetition count is {} ", s1, s2, s3);
        let msg: String = format!("The game is {}. {}'s turn. Currently {}th turn.", s1, s2, self.get_turn_count());
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

impl Display for InvalidActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

        if self.contains(InvalidActionError::NO_PIECE_AT_STARTINGPOS) {
            writeln!(f, "There is no piece at the starting position")?;
        }
        if self.contains(InvalidActionError::OBSTACLE_IN_PATH) {
            writeln!(f, "The movement path is obstructed")?;
        }
        if self.contains(InvalidActionError::THIRD_REPETITION) {
            writeln!(f, "The move would result in a third repetition")?;
        }
        if self.contains(InvalidActionError::DST_IS_RESTRICTED) {
            writeln!(f, "The destination is restricted")?;
        }
        if self.contains(InvalidActionError::PIECETYPE_CONTRADICTS_STATE) {
            writeln!(f, "The piece type contradicts the current game state")?;
        }
        if self.contains(InvalidActionError::MOVEMENT_NOT_ORTHOGONAL) {
            writeln!(f, "The movement is not orthogonal")?;
        }
        Ok(())
    }
}

impl std::error::Error for InvalidActionError {}

// [About the Repetition Rule]
// We only consider successive repetition of board positions to be the violating repetitions. That is, in order for a 
// position to be counted as repetition, it has to be the same position from exactly 4 steps ago 
// (Example) 
// (att, position1) -> (def, position2) -> (att, position3) -> (def, position4) -> (att, position1) 
//           +                                                                              + 
//           |                                                                              | 
//           +------------------------------------------------------------------------------+
// The player who made the move that resulted in repetition should bear the penalty
// In the above case, the defender is the offending side
// There are many opinions as to how to determine the offending side (Some claim that defender should always bear the blame), and how to punish the repetition,
// But I decided that the move that would result in a 3rd repetition should simply  be rejected.
// So, a player loses if she has only one move and that move would result in a 3rd repetition.

pub struct Game{
    pub board: TaflBoardEleven,
    pub state: GameState,

    pub short_history: ShortHistory, 
    // This serves as the small ring buffer that stores the most recent boards 
    // (including the current board, since Game.board is designed to be mutated on every move) for the repetition check
    // turn 0 -> turn 1 -> turn 2 -> turn3 -> ...
    //   |          |           |        |
    //   +          +           +        |
    //  [0]        [1]         [2]      [3]       <= short_history
    //                   ::
    //                   ::     
    //             [0]         [1]      [2]      [3]
    pub repetition_counter: RepetitionCounter 
    // There can be at most 4 repetitive states at the same time. We keep track of these counts by
    // The *repetition_counter*, which is a 1byte integer in which each 2 consecutive bits stores the count of the position 
    // (up to 4, but in this rule 2 is the maximum).
    // Each time a move is made, this count is shifted to left by 2, with the new added 2 bits 0 if (the 4th to last position was not repeated),
    // and [previous count at 7~8 bits + 1] if (it was repeated)
    // [Example]
    // repetition_counter : [01][00][10][01]  => [00][10][01][10] (if repetition occurred)
}

type RepetitionCounter = u8;

#[derive(Debug)]
pub struct ShortHistory (ArrayDeque<TaflBoardEleven, 4, Wrapping> );

impl ShortHistory{
    pub fn push_back(&mut self, board: TaflBoardEleven) 
    {
        self.0.push_back(board);
    }
    pub fn check_repetition(&self, new_board: &TaflBoardEleven) -> bool
    {
        if let Some(b) = self.0.get(0)
        {
            b.equals(new_board)
        }
        else { false }
    }
    pub fn new() -> Self
    {
        ShortHistory(ArrayDeque::<TaflBoardEleven, 4, Wrapping>::new())
    }
    pub fn update_repetition_counter(&self, new_board: &TaflBoardEleven, repetition_counter: RepetitionCounter) -> RepetitionCounter
    {
        let tmp_count = repetition_counter >> 6;
        let tmp = repetition_counter << 2;
        if self.check_repetition(new_board)
        {
            tmp + tmp_count + 1
        }
        else 
        {
            tmp
        }
    }
    pub fn iter(&self) -> arraydeque::Iter<TaflBoardEleven> {
        self.0.iter()
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::init_std()
    }
}

impl Game{

    //NOTE:  Call this function BEFORE the board is changed
    pub fn update_short_history_mut(&mut self) {
        self.short_history.push_back(self.board);
    }

    // returns true if the new_board (board after action) is repetitive
    pub fn check_repetition(&self, new_board: &TaflBoardEleven) -> bool {
        self.short_history.check_repetition(new_board)
    }

    // NOTE: Call this function AFTER the move is made
    pub fn update_repetition_counter_mut(&mut self) {
        let tmp_count = self.repetition_counter >> 6;
        let tmp = self.repetition_counter << 2;
        if self.check_repetition(&self.board) {
            self.repetition_counter = tmp + tmp_count + 1;
        } else {
            self.repetition_counter = tmp;
        }
    }
    pub fn from_board(board: TaflBoardEleven, side: Side) -> Self{
        let state = match side{
            Side::Att => GameState::TURN_ATT,
            Side::Def => GameState(0x0000),
        };
        let repetition_counter: u8 = 0x00;
        let short_history = ShortHistory::new();
        Self { board, state, short_history, repetition_counter}
    }

    // generate standard position
    pub fn init_std() -> Self{
        let board = TaflBoardEleven::init_std();
        Self::from_board(board, Side::Att)
    }

    pub fn check_action_validity(&self, action: &MoveOnBoardEleven, piece_type: Option<PieceType>) -> Result<(), InvalidActionError>{

        let mut e = InvalidActionError::from_bits_retain(0);
        let playing_side = self.state.show_side();

        let opt = action.find_path();
        if let None = opt{
            e.insert(InvalidActionError::MOVEMENT_NOT_ORTHOGONAL);
        }

        // If the piece_type is specified, then we check the validity assuming that the movement happens for the specified piece.
        if let Some(p) = piece_type
        {
            if !p.aligns_with_side(playing_side) { e.insert(InvalidActionError::PIECETYPE_CONTRADICTS_STATE);}
            invalid_action_check_given_piece(&mut e, &self, &p, action, &opt);
        }
        else
        {
            let p: PieceType = match playing_side{
                Side::Att => PieceType::AttSoldier,
                Side::Def => {
                    let def_has_piece_there = !self.board.bit_def.tile_is_empty_at(&action.start);
                    let king_has_piece_there = !self.board.bit_king.tile_is_empty_at(&action.start);
                    if def_has_piece_there && !king_has_piece_there {PieceType::DefSoldier}
                    else if king_has_piece_there && !def_has_piece_there {PieceType::King}
                    else if !king_has_piece_there && !def_has_piece_there { 

                        e.insert(InvalidActionError::PIECETYPE_CONTRADICTS_STATE);
                        PieceType::DefSoldier // as a dummy value
                    }
                    else { 
                        // This is an irrecoverable error, shutting the game down..
                        panic!("Defender and King pieces have an overlap!")
                    }
                },
            };

            invalid_action_check_given_piece(&mut e, &self, &p, action, &opt);
        }

        // Helper func.
        fn invalid_action_check_given_piece(e: &mut InvalidActionError, game: &Game, p: &PieceType, action: &MoveOnBoardEleven, opt: &Option<Direction>) {
            // check the starting position
            let target_board = match p{
                PieceType::AttSoldier => game.board.bit_att,
                PieceType::DefSoldier => game.board.bit_def,
                PieceType::King => game.board.bit_king,
            };
            if target_board.tile_is_empty_at(&action.start) { e.insert(InvalidActionError::NO_PIECE_AT_STARTINGPOS); }

            // Next, check the movement path and see if there is any obstacle
            if let Some(dir) = opt{

                let barricade = game.board.bit_att | game.board.bit_def | game.board.bit_king;
                let restricted = match p{
                    PieceType::AttSoldier | PieceType::DefSoldier => game.board.hostile,
                    PieceType::King => BoardEleven::new(),
                };
                check_path_obstacle(&dir, &action, &barricade, &restricted, e);
            }

            // Finally, check for the repetition
            let resulting_board: TaflBoardEleven = match p {
                PieceType::AttSoldier => game.board.att_force_move(action),
                PieceType::DefSoldier => game.board.def_force_move(action),
                PieceType::King       => game.board.king_force_move(action),
            };
            if game.check_repetition(&resulting_board) {
                let rep_count = game.repetition_counter >> 6;
                if rep_count >= 2 {
                    e.insert(InvalidActionError::THIRD_REPETITION);
                }
            }

        }

        // Helper func. 
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
    pub fn do_action_wo_validity_check(&self, action: &MoveOnBoardEleven, piece_type: Option<PieceType>) -> Self{

        let side = self.state.show_side();
        let mut new_short_history = ShortHistory::new();
        for b in self.short_history.0.iter()
        {
            new_short_history.push_back(*b);
        }


        // Be careful not to call this elsewhere!!
        new_short_history.push_back(self.board);

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
                new_state.change_side_mut();
                if king_capt.is_nonzero(){
                    new_state.game_over_mut();
                }
                if new_board.def_is_encircled() {
                    new_state.game_over_mut();
                }
                new_state.incre_turn_count_mut();

                let new_repetition_counter = new_short_history.update_repetition_counter(&new_board, self.repetition_counter);                

                Self { board: new_board, state: new_state, short_history: new_short_history, repetition_counter: new_repetition_counter}
                
            },
            Side::Def => {
                // If the user specified which piece to move (defender soldier or king), we trust that information
                let mut new_board = if let Some(p) = piece_type
                {
                    match p
                    {
                        PieceType::DefSoldier => self.board.def_force_move(action),
                        PieceType::King => self.board.king_force_move(action),
                        PieceType::AttSoldier => panic!("???"),
                    }
                } else 
                {
                    let def_has_piece_there = !self.board.bit_def.tile_is_empty_at(&action.start);
                    let king_has_piece_there = !self.board.bit_king.tile_is_empty_at(&action.start);

                    if def_has_piece_there && !king_has_piece_there 
                    {
                        self.board.def_force_move(action)
                    }
                    else if king_has_piece_there && !def_has_piece_there
                    {
                        self.board.king_force_move(action)
                    }
                    else if !def_has_piece_there && !king_has_piece_there
                    {
                        self.board
                    }
                    else
                    { 
                        panic!("Defender soldiers and King had an overlap!") 
                    }
                };
                let att_capt = self.board.att_capture(action);
                new_board.bit_att ^= att_capt;
                let mut new_state = self.state;
                new_state.change_side_mut();
                if !new_board.bit_att.is_nonzero()
                {
                    new_state.game_over();
                }
                if (new_board.bit_king & GOALS).is_nonzero()
                {
                    new_state.game_over();
                } 

                new_state.incre_turn_count_mut();

                let new_repetition_counter = new_short_history.update_repetition_counter(&new_board, self.repetition_counter);                

                Self {board: new_board, state: new_state, short_history: new_short_history, repetition_counter: new_repetition_counter}
            },
        }
        
    }
}

pub fn parse_repetition_counter(rep_c: RepetitionCounter) -> [u8; 4]{
    let mut a: [u8;4] = [0;4];
    for i in 0..4{
        let c = (rep_c >> 2 * i) & 0b11;
        a[i] = c;
    }
    a
}

impl std::fmt::Display for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.board)?;
        writeln!(f, "{}", self.state)?;
        let repetitions = parse_repetition_counter(self.repetition_counter);
        let show_rep = format!("Repetition count: {},{},{},{}", repetitions[0],repetitions[1],repetitions[2],repetitions[3]);
        writeln!(f, "{}", show_rep)?;
        Ok(())
    }
}


#[test]
fn game_display_works() {
    let game = Game::init_std();
    print!("{}", game);
}

#[test]
fn repetition_count_works() {
    let mut game = Game::init_std();
    let move1: MoveOnBoardEleven = MoveOnBoardEleven::try_from("F2G2".to_owned()).unwrap();
    let move2: MoveOnBoardEleven = MoveOnBoardEleven::try_from("F4G4".to_owned()).unwrap();
    let re_move1: MoveOnBoardEleven = MoveOnBoardEleven::try_from("G2F2".to_owned()).unwrap();
    let re_move2: MoveOnBoardEleven = MoveOnBoardEleven::try_from("G4F4".to_owned()).unwrap();

    let actions: [MoveOnBoardEleven;4] = [move1.clone(), move2.clone(), re_move1.clone(), re_move2.clone()];

    for action in actions{
        println!("{}", game);
        println!("short history");
        for b in game.short_history.0.iter() {
            println!("{}", b);
        }
        println!("=======================");
        let new_game = game.do_action_wo_validity_check(&action, None);
        game = new_game;
    }
    println!("{}", game);
    println!("short history");
    for b in game.short_history.0.iter() {
        println!("{}", b);
    }
    println!("=======================");
}


