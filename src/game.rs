use std::fmt::Display;

use crate::board::TaflBoardEleven;
use bitboard::eleven::*;
use bitboard::Direction;

use bitflags::{bitflags};
use arraydeque::{ArrayDeque, Wrapping};

#[derive(Debug, Copy, Clone)]
pub enum Side { Att, Def }

impl Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self{
            Side::Att => write!(f, "Attacker"),
            Side::Def => write!(f, "Defender"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug)]
pub enum ReasonForTermination {
    KingCaptured,
    KingEscaped,
    NoAttSoldier,
    DefEncircled,
    ViolatedRepetition(Side),
    NoMoveLeft(Side),
    NoLegalMoveLeft(Side),
}

impl Display for ReasonForTermination {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DefEncircled => write!(f, "Defenders are encircled!"),
            Self::KingCaptured => write!(f, "King was captured"),
            Self::KingEscaped => write!(f, "King has escaped"),
            Self::NoAttSoldier => write!(f, "Attackers are annihilated"),
            Self::ViolatedRepetition(side) => write!(f, "{} violated the repetition rule", side),
            Self::NoMoveLeft(side) => write!(f, "{} has no move left", side),
            Self::NoLegalMoveLeft(side) => write!(f, "{} has no legal move left", side),
        }
    }
}
// 1st bit is 1 if (current player = attacker), 0 else
// 2nd bit is 1 if (game is over on THIS turn) 
// <- the last player made a winning move 
// - this includes such a move that the opponent would have no move, or only the 3rd repetition)
// UPDATE: 2nd bit is 1 if the game is over, regardless of when that happened.
// 3rd ~ 8th bits: so far no use
// 9~16th bits: turn counter (maybe useful at some point)
#[derive(Clone, Copy, Debug, PartialEq)]
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
        self.toggle(Self::TURN_ATT);
    }
    pub fn change_side(&self) -> Self{
        Self(self.0 ^ Self::TURN_ATT.bits())
    }
    pub fn game_over_mut(&mut self) {
        self.insert(Self::TERMINATED);
    }
    pub fn game_over(&self) -> Self {
        Self(self.0 | Self::TERMINATED.bits())
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
    pub fn get_victor(&self) -> Side {
        if self.contains(Self::ATTACKER_VICTORY) {
            Side::Att
        } else {
            Side::Def
        }
    }
    pub fn set_victor(&mut self, side: Side) {
        match side {
            Side::Att => self.insert(Self::ATTACKER_VICTORY),
            Side::Def => self.remove(Self::ATTACKER_VICTORY),
        }
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone)]
pub struct ShortHistory (ArrayDeque<TaflBoardEleven, 4, Wrapping> );

impl PartialEq for ShortHistory {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter()
            .zip(other.0.iter())
            .all(|(a,b)| a.equals(b))
    }
}

impl ShortHistory{
    pub fn push_front(&mut self, board: TaflBoardEleven) 
    {
        self.0.push_front(board);
    }
    pub fn push_back(&mut self, board: TaflBoardEleven)
    {
        self.0.push_back(board);
    }
    pub fn get_most_recent(&self) -> Option<&TaflBoardEleven> {
        self.0.get(0)
    }
    pub fn get_oldest(&self) -> Option<&TaflBoardEleven> {
        self.0.get(3)
    }
    pub fn get_second_oldest(&self) -> Option<&TaflBoardEleven> {
        self.0.get(2)
    }
    pub fn get(&self, idx: usize) -> Option<&TaflBoardEleven> {
        self.0.get(idx)
    }
    pub fn check_repetition(&self, new_board: &TaflBoardEleven) -> bool
    {
        if let Some(b) = self.get_oldest()
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
    pub fn iter(&'_ self) -> arraydeque::Iter<'_, TaflBoardEleven> {
        self.0.iter()
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::init_std()
    }
}

impl Game{

    pub fn get_possible_actions(&self) -> Vec<MoveOnBoardEleven> {
        match self.state.show_side() {
            Side::Att => {
                self.board.generate_actions_for_attsoldiers()
            },
            Side::Def => {
                let mut vec = self.board.generate_actions_for_defsoldiers();
                vec.extend(self.board.generate_actions_for_king());
                vec
            }
        }
    }

    //NOTE:  Call this function BEFORE the board is changed
    pub fn update_short_history_mut(&mut self) {
        self.short_history.push_front(self.board);
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

    // only for cheap creation of dummy variable
    #[inline]
    pub fn ghastly() -> Self {
        let board = TaflBoardEleven::new(
            BoardEleven::ghastly(),
            BoardEleven::ghastly(),
            BoardEleven::ghastly(),
            BoardEleven::ghastly()
        );
        Self { 
            board,
            state: GameState(0),
            short_history: ShortHistory::new(),
            repetition_counter: 0
        }
    }

    // When the action would result in the repetition, returns true. If not, (even when the action is invalid), returns false.
    pub fn move_would_result_in_repetition(&self, side: Side, piece_type: Option<PieceType>, action: &MoveOnBoardEleven) -> bool {

        let piece_type = 
        if let Some(p) = piece_type { p }
        else { match self.board.determine_action_piecetype(side, &action) {
            Ok(pt) => pt,
            Err(_) => { return false; }
        }
        };

        let resulting_board_prior = match piece_type {
            PieceType::AttSoldier => self.board.att_force_move(action),
            PieceType::DefSoldier => self.board.def_force_move(action),
            PieceType::King => self.board.king_force_move(action),
        };
        // let resulting_board_prior = match side {
        //     Side::Att => self.board.att_force_move(action),
        //     Side::Def => {
        //         match self.board.determine_action_piecetype(Side::Def, &action){
        //             Ok(PieceType::DefSoldier) => self.board.def_force_move(action),
        //             Ok(PieceType::King) => self.board.king_force_move(action),
        //             _ => { return false; }
        //         }
        //     }
        // };
        
        // First test is if the movement itself without subsequent captures would result in repetition.
        // This is a very order-specific way of checking, if one wants more safety, consider instead calling short_history.contains(resulting_board_prior)
        if let Some(reference_board) = self.short_history.get_second_oldest()
        {
            if resulting_board_prior.equals(reference_board) {
                // Second test is if any capture would occur (if it does, then it cannot be a repetition)
                match side {
                    Side::Att => {
                        if self.board.def_capture(action).is_nonzero() || self.board.king_capture(action).is_nonzero()
                        || self.board.shield_wall_capture(action).is_nonzero()  
                        { false } else { true }
                    },
                    Side::Def => {
                        if self.board.att_capture(action).is_nonzero()  
                        { false } else { true }
                    }
                }
            } else { false }
        } else {
            false
        }
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
            let p: PieceType = self.board.determine_action_piecetype(playing_side, &action)?;

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
            let side = match p {
                PieceType::AttSoldier => Side::Att,
                PieceType::DefSoldier | PieceType::King => Side::Def,
            };
            if get_rep_counter_for_oldest(game.repetition_counter) == 2 && game.move_would_result_in_repetition(side, Some(*p), action) {
                e.insert(InvalidActionError::THIRD_REPETITION);
            }
            // let resulting_board: TaflBoardEleven = match p {
            //     PieceType::AttSoldier => game.board.att_force_move(action),
            //     PieceType::DefSoldier => game.board.def_force_move(action),
            //     PieceType::King       => game.board.king_force_move(action),
            // };
            // if game.check_repetition(&resulting_board) {
            //     let rep_count = game.repetition_counter >> 6;
            //     if rep_count >= 2 {
            //         e.insert(InvalidActionError::THIRD_REPETITION);
            //     }
            // }

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

    pub fn do_action_wo_validity_check_mut(&mut self, action: &MoveOnBoardEleven, piece_type: Option<PieceType>) {

        let side = self.state.show_side();

        // Be careful not to call this elsewhere!!
        self.short_history.push_front(self.board);

        match side{
            Side::Att => {

                let def_capt = self.board.def_capture(action);
                let king_capt = self.board.king_capture(action);
                let shield_wall_capt = self.board.shield_wall_capture(action);

                self.board.att_force_move_mut(action);
                self.board.bit_def ^= def_capt | (shield_wall_capt & self.board.bit_def);
                self.board.bit_king ^= king_capt | (shield_wall_capt & self.board.bit_king);

                self.state.change_side_mut();
                if king_capt.is_nonzero(){
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Att);
                }
                if self.board.def_is_encircled() {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Att);
                }
                self.state.incre_turn_count_mut();

                let new_repetition_counter = self.short_history.update_repetition_counter(&self.board, self.repetition_counter);                
                self.repetition_counter = new_repetition_counter;

            },
            Side::Def => {

                let att_capt = self.board.att_capture(action);

                // If the user specified which piece to move (defender soldier or king), we trust that information
                if let Some(p) = piece_type 
                {
                    match p 
                    {
                        PieceType::DefSoldier => self.board.def_force_move_mut(action),
                        PieceType::King => self.board.king_force_move_mut(action),
                        PieceType::AttSoldier => {}
                    };
                } else 
                {
                    let def_has_piece_there = !self.board.bit_def.tile_is_empty_at(&action.start);
                    let king_has_piece_there = !self.board.bit_king.tile_is_empty_at(&action.start);

                    if def_has_piece_there && !king_has_piece_there 
                    {
                        self.board.def_force_move_mut(action);
                    }
                    else if king_has_piece_there && !def_has_piece_there
                    {
                        self.board.king_force_move_mut(action);
                    }
                    else if !def_has_piece_there && !king_has_piece_there
                    {
                        {}
                    }
                    else
                    { 
                        panic!("Defender soldiers and King had an overlap!"); 
                    }

                }
                self.board.bit_att ^= att_capt;

                // change state
                self.state.change_side_mut();
                if !self.board.bit_att.is_nonzero()
                {
                    self.state.game_over();
                    self.state.set_victor(Side::Def);
                }
                if (self.board.bit_king & GOALS).is_nonzero()
                {
                    self.state.game_over();
                    self.state.set_victor(Side::Def);
                } 

                self.state.incre_turn_count_mut();

                let new_repetition_counter = self.short_history.update_repetition_counter(&self.board, self.repetition_counter);                
                self.repetition_counter = new_repetition_counter;

            },
        }
        
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
        new_short_history.push_front(self.board);

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

    pub fn forward_turn(&mut self) {
        self.state.incre_turn_count_mut();
        self.state.change_side_mut();
    } 

    // This checks the victory condition for the specified side and update if necessary based on the current board. 
    // It assumes that the specified side is the one who made the last move
    pub fn update_victory(&mut self, side: Side, pass_next_moves: bool) -> (Option<ReasonForTermination>, Option<Vec<MoveOnBoardEleven>>) {
        use ReasonForTermination::*;

        let mut opt: Option<ReasonForTermination> = None;
        match side {
            Side::Att => {
                if !self.board.bit_king.is_nonzero() {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Att);
                    opt = Some(KingCaptured);
                }
                if self.board.def_is_encircled() {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Att);
                    opt = Some(DefEncircled);
                }
                let mut def_possible_moves = self.board.generate_actions_for_defsoldiers();
                let mut king_possible_moves = self.board.generate_actions_for_king();

                if def_possible_moves.len() == 0 && king_possible_moves.len() == 0 {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Att);
                    opt = Some(NoMoveLeft(Side::Def));

                } else if get_rep_counter_for_oldest(self.repetition_counter) == 2 {

                    let no_legal_move_for_defsol = def_possible_moves.iter().fold(true, |s, action| 
                        s && self.move_would_result_in_repetition(Side::Def, None, action)
                    );
                    let no_legal_move_for_king = king_possible_moves.iter().fold(true, |s,action| 
                        s && self.move_would_result_in_repetition(Side::Def, None, action)
                    );
                    if no_legal_move_for_defsol && no_legal_move_for_king {
                        self.state.game_over_mut();
                        self.state.set_victor(Side::Att);
                        opt = Some(NoLegalMoveLeft(Side::Def));
                    }
                }
                if pass_next_moves {
                    def_possible_moves.append(&mut king_possible_moves);
                    (opt, Some(def_possible_moves))
                } else {
                    (opt, None)
                }
            }
            Side::Def => {
                if !self.board.bit_att.is_nonzero() {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Def);
                    opt = Some(NoAttSoldier);
                }
                let att_possible_moves = self.board.generate_actions_for_attsoldiers();
                if att_possible_moves.len() == 0 {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Def);
                    opt = Some(NoMoveLeft(Side::Att));
                } else if get_rep_counter_for_oldest(self.repetition_counter) == 2 {
                    let no_legal_move_for_att = att_possible_moves.iter().fold(true, |s, action|
                        s && self.move_would_result_in_repetition(Side::Att, Some(PieceType::AttSoldier), action)
                    );
                    if no_legal_move_for_att {
                        self.state.game_over_mut();
                        self.state.set_victor(Side::Def);
                        opt = Some(NoLegalMoveLeft(Side::Att));
                    }
                }
                if (self.board.bit_king & GOALS).is_nonzero() {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Def);
                    opt = Some(KingEscaped)
                }
                if pass_next_moves {
                    (opt, Some(att_possible_moves))
                } else {
                    (opt, None)
                }
            }
        }
    }

    // Check if the last move was illegal (currently only checks the repetition) and update the state
    // Assumes 'side' is the one who made the last move
    pub fn update_if_lost(&mut self, side: Side) -> Option<ReasonForTermination> {

        use ReasonForTermination::*;
        if get_rep_counter_for_cur(self.repetition_counter) == 3 {
            match side {
                Side::Att => {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Def);
                    Some(ViolatedRepetition(Side::Att))
                },
                Side::Def => {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Att);
                    Some(ViolatedRepetition(Side::Def))
                }
            }
        } else { None }
    }

    pub fn do_move_and_update_whole_mut(&mut self, action: &MoveOnBoardEleven, piecetype: Option<PieceType>, pass_next_moves: bool) 
    -> Result<(Option<ReasonForTermination>, Option<Vec<MoveOnBoardEleven>>), InvalidActionError> {

        let side = self.state.show_side();
            
        self.short_history.push_front(self.board);
        let piecetype 
            = if let Some(p) = piecetype { p } 
            else { self.board.determine_action_piecetype(side, action)? };

        match piecetype {
            PieceType::AttSoldier => {

                let def_capt = self.board.def_capture(action);
                let king_capt = self.board.king_capture(action);
                let shield_wall_capt = self.board.shield_wall_capture(action);

                self.board.att_force_move_mut(action);
                self.board.bit_def ^= def_capt | (shield_wall_capt & self.board.bit_def);
                self.board.bit_king ^= king_capt | (shield_wall_capt & self.board.bit_king);

            }
            PieceType::DefSoldier => {

                let att_capt = self.board.att_capture(action);

                self.board.def_force_move_mut(action);
                self.board.bit_att ^= att_capt;
            }
            PieceType::King => {

                let att_capt = self.board.att_capture(action);

                self.board.king_force_move_mut(action);
                self.board.bit_att ^= att_capt;
            }
        }

        let new_repetition_counter = self.short_history.update_repetition_counter(&self.board, self.repetition_counter);                
        self.repetition_counter = new_repetition_counter;
        
        // Check for game termination
        let value = if let Some(r) = self.update_if_lost(side) {
            Ok((Some(r), None))
        } else if let (Some(rr), v) = self.update_victory(side, pass_next_moves) {
            Ok((Some(rr), v))
        } else { Ok((None, None))
        };

        self.forward_turn();

        value
    }

    // TODO! unit-test
    pub fn do_move_and_update_whole(&self, action: &MoveOnBoardEleven, piecetype: Option<PieceType>, pass_next_moves: bool)
     -> Result<(Self, Option<ReasonForTermination>, Option<Vec<MoveOnBoardEleven>>), InvalidActionError>
    {

        let side = self.state.show_side();
            
        let mut new_short_history = self.short_history.clone();
        new_short_history.push_front(self.board);

        let piecetype 
            = if let Some(p) = piecetype { p } 
            else { self.board.determine_action_piecetype(side, action)? };

        let new_board = match piecetype {
            PieceType::AttSoldier => {

                let def_capt = self.board.def_capture(action);
                let king_capt = self.board.king_capture(action);
                let shield_wall_capt = self.board.shield_wall_capture(action);

                let mut new_board = self.board.att_force_move(action);
                new_board.bit_def ^= def_capt | (shield_wall_capt & self.board.bit_def);
                new_board.bit_king ^= king_capt | (shield_wall_capt & self.board.bit_king);
                new_board
            }
            PieceType::DefSoldier => {

                let att_capt = self.board.att_capture(action);

                let mut new_board = self.board.def_force_move(action);
                new_board.bit_att ^= att_capt;
                new_board
            }
            PieceType::King => {

                let att_capt = self.board.att_capture(action);

                let mut new_board = self.board.king_force_move(action);
                new_board.bit_att ^= att_capt;
                new_board
            }
        };

        let new_repetition_counter = self.short_history.update_repetition_counter(&self.board, self.repetition_counter);

        let mut new_game = Game {
            board: new_board,
            state: self.state.clone(),
            short_history: new_short_history,
            repetition_counter: new_repetition_counter,
        };
        
        // Check for game termination
        let postfix = if let Some(r) = new_game.update_if_lost(side) {
            (Some(r), None)
        } else if let (Some(rr), v) = new_game.update_victory(side, pass_next_moves) {
            (Some(rr), v)
        } else { (None, None)
        };

        new_game.forward_turn();

        Ok((new_game, postfix.0, postfix.1))

    }

    pub fn check_and_do_and_update_whole_mut(&mut self, action: &MoveOnBoardEleven, piece_type: Option<PieceType>, pass_next_moves: bool)
        -> Result<(Option<ReasonForTermination>, Option<Vec<MoveOnBoardEleven>>), InvalidActionError> {

            let piece_type = 
            if let Some(p) = piece_type { p } 
            else { self.board.determine_action_piecetype(self.state.show_side(), action)? }; 

            self.check_action_validity(action, Some(piece_type))?;

            self.do_move_and_update_whole_mut(action, Some(piece_type), pass_next_moves)
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

pub fn get_rep_counter_for_oldest(rep_c: RepetitionCounter) -> u8 {
    rep_c >> 6 & 0b11
}

pub fn get_rep_counter_for_cur(rep_c: RepetitionCounter) -> u8 {
    rep_c & 0b11
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


