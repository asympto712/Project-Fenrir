use std::fmt::{Debug, Display};

use crate::board::TaflBoard;
use bitboard::Direction;
use bitboard::{BitBoard, MoveOnBoard};
use bitboard::eleven::BoardEleven;
use bitboard::seven::BoardSeven;

use bitflags::{bitflags};
use arraydeque::{ArrayDeque, Wrapping};
use bincode;

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
        matches!((self, side), 
            (PieceType::AttSoldier, Side::Att) |
            (PieceType::DefSoldier, Side::Def) |
            (PieceType::King , Side::Def)
        )
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
            Self::ViolatedRepetition(side) => write!(f, "{side} violated the repetition rule"),
            Self::NoMoveLeft(side) => write!(f, "{side} has no move left"),
            Self::NoLegalMoveLeft(side) => write!(f, "{side} has no legal move left"),
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
#[derive(Clone, Copy, Debug, PartialEq, bincode::Encode, bincode::Decode)]
pub struct GameState(u16);

bitflags! {
    impl GameState: u16{
        const TURN_ATT =  1 << 0;
        const TERMINATED = 1 << 1;
        const ATTACKER_VICTORY = 1 << 2;
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
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
        if self.0 >> 8 < u8::MAX as u16 {
            self.0 += 1 << 8;
        }
    }
    pub fn incre_turn_count(&self) -> Self {
        Self((self.0 + 1) << 8)
    }
    pub fn get_victor(&self) -> Side {
        debug_assert!(!self.is_ongoing());
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
        write!(f, "{msg}")?;
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

pub type MoveResult<B> = (Option<ReasonForTermination>, Option<Vec<<B as BitBoard>::Movement>>);
pub type MoveResultWithSelf<T, B> = Result<(T, Option<ReasonForTermination>, Option<Vec<<B as BitBoard>::Movement>>), InvalidActionError>;
pub type SimpleMoveResult<B> = Result<MoveResult<B>, InvalidActionError>;

pub trait GameLogic: Sized + Default + Clone + std::fmt::Debug{
    type B: BitBoard + std::fmt::Debug;

    fn _assert_display() where TaflBoard<Self::B>: std::fmt::Display {}

    fn get_board(&self) -> &TaflBoard<Self::B>;
    fn get_board_mut(&mut self) -> &mut TaflBoard<Self::B>;
    fn get_state(&self) -> &GameState;
    fn get_state_mut(&mut self) -> &mut GameState;
    fn ghastly() -> Self;
    fn current_side(&self) -> Side{
        self.get_state().show_side()
    }
    fn determine_action_piecetype(&self, side: Side, action: &<Self::B as BitBoard>::Movement) -> Result<PieceType, InvalidActionError>{
        self.get_board().determine_action_piecetype(side, action)
    }
    fn get_possible_actions(&self) -> Vec<<Self::B as BitBoard>::Movement>{
        match self.get_state().show_side() {
            Side::Att => {
                self.get_board().generate_actions_for_attsoldiers()
            },
            Side::Def => {
                let mut vec = self.get_board().generate_actions_for_defsoldiers();
                vec.extend(self.get_board().generate_actions_for_king());
                vec
            }
        }
    }
    fn invalid_action_check_basics(&self, e: &mut InvalidActionError, p: &PieceType, action: &<Self::B as BitBoard>::Movement, opt: &Option<Direction>) {
        // check the starting position
        let target_board = match p{
            PieceType::AttSoldier => self.get_board().bit_att,
            PieceType::DefSoldier => self.get_board().bit_def,
            PieceType::King => self.get_board().bit_king,
        };
        if target_board.tile_is_empty_at(&action.start()) { e.insert(InvalidActionError::NO_PIECE_AT_STARTINGPOS); }

        // Next, check the movement path and see if there is any obstacle
        if let Some(dir) = opt{

            let barricade = self.get_board().bit_att | self.get_board().bit_def | self.get_board().bit_king;
            let restricted = match p{
                PieceType::AttSoldier | PieceType::DefSoldier => self.get_board().hostile,
                PieceType::King => Self::B::new(),
            };
            check_path_obstacle(dir, action, &barricade, &restricted, e);
        }
    }

    fn update_repetition_counter(&mut self){}
    fn update_short_history(&mut self){}
    fn invalid_action_check_advanced(&self, _e: &mut InvalidActionError, _p: &PieceType, _action: &<Self::B as BitBoard>::Movement, _opt: &Option<Direction>) {}
    fn invalid_action_check(&self, e: &mut InvalidActionError, p: &PieceType, action: &<Self::B as BitBoard>::Movement, opt: &Option<Direction>){
        self.invalid_action_check_basics(e, p, action, opt);
        self.invalid_action_check_advanced(e, p, action, opt);
    }
    fn yield_captured_def(&self, action: &<Self::B as BitBoard>::Movement) -> (Self::B, Self::B){
        // Does not consider shield wall by default
        (self.get_board().def_capture(action), self.get_board().king_capture(action))
    }
    fn yield_captured_att(&self, action: &<Self::B as BitBoard>::Movement) -> Self::B {
        self.get_board().att_capture(action)
    }

    fn do_action_wo_validity_check_mut(&mut self, action: &<Self::B as BitBoard>::Movement, piece_type: Option<PieceType>){

        self.update_short_history();
        let side = self.get_state().show_side();
        match side{
            Side::Att => {

                let (def_capt, king_capt) = self.yield_captured_def(action);

                self.get_board_mut().att_force_move_mut(action);
                self.get_board_mut().bit_def ^= def_capt;
                self.get_board_mut().bit_king ^= king_capt;

                self.get_state_mut().change_side_mut();
                if king_capt.is_nonzero(){
                    self.get_state_mut().game_over_mut();
                    self.get_state_mut().set_victor(Side::Att);
                }
                if self.get_board().def_is_encircled() {
                    self.get_state_mut().game_over_mut();
                    self.get_state_mut().set_victor(Side::Att);
                }
                self.get_state_mut().incre_turn_count_mut();

                self.update_repetition_counter();

            },
            Side::Def => {

                let att_capt = self.yield_captured_att(action);

                // If the user specified which piece to move (defender soldier or king), we trust that information
                let p = piece_type.unwrap_or_else(|| self.determine_action_piecetype(Side::Def, action).unwrap());
                match p{
                    PieceType::DefSoldier => self.get_board_mut().def_force_move_mut(action),
                    PieceType::King => self.get_board_mut().king_force_move_mut(action),
                    PieceType::AttSoldier => {}
                }
                self.get_board_mut().bit_att ^= att_capt;

                // change state
                self.get_state_mut().change_side_mut();
                if !self.get_board().bit_att.is_nonzero()
                {
                    self.get_state_mut().game_over();
                    self.get_state_mut().set_victor(Side::Def);
                }
                if (self.get_board().bit_king & <Self::B as BitBoard>::CORNERS).is_nonzero()
                {
                    self.get_state_mut().game_over();
                    self.get_state_mut().set_victor(Side::Def);
                } 

                self.get_state_mut().incre_turn_count_mut();

                self.update_repetition_counter();

            },
        }

    }
    fn do_action_unchecked(&self, action: &<Self::B as BitBoard>::Movement, piece_type: Option<PieceType>) -> Self;
    fn update_victory(&mut self, side: Side, pass_next_moves: bool) -> MoveResult<Self::B>;
    fn update_if_lost(&mut self, side: Side) -> Option<ReasonForTermination>;
    fn do_move_and_update_whole_mut(&mut self, action: &<Self::B as BitBoard>::Movement, piece_type: Option<PieceType>, pass_next_moves: bool)
    -> SimpleMoveResult<Self::B>{

        let side = self.get_state().show_side();
            
        self.update_short_history();
        let piecetype 
            = if let Some(p) = piece_type { p } 
            else { self.determine_action_piecetype(side, action)? };

        match piecetype {
            PieceType::AttSoldier => {

                let (def_capt, king_capt) = self.yield_captured_def(action);

                self.get_board_mut().att_force_move_mut(action);
                self.get_board_mut().bit_def ^= def_capt;
                self.get_board_mut().bit_king ^= king_capt;

            }
            PieceType::DefSoldier => {

                let att_capt = self.yield_captured_att(action);

                self.get_board_mut().def_force_move_mut(action);
                self.get_board_mut().bit_att ^= att_capt;
            }
            PieceType::King => {

                let att_capt = self.yield_captured_att(action);

                self.get_board_mut().king_force_move_mut(action);
                self.get_board_mut().bit_att ^= att_capt;
            }
        }

        self.update_repetition_counter();
        
        // // Check for game termination
        // let value = if let Some(r) = self.update_if_lost(side) {
        //     Ok((Some(r), None))
        // } else if let (Some(rr), v) = self.update_victory(side, pass_next_moves) {
        //     Ok((Some(rr), v))
        // } else { Ok((None, None))
        // };new_game

        self.forward_turn();

        if let Some(reason) = self.update_if_lost(side) {
            return Ok((Some(reason), None));
        } else {
            let (r, v) = self.update_victory(side, pass_next_moves);
            if r.is_none() {
                return Ok((None, v));
            } else {
                return Ok((Some(r.unwrap()), v));
            }
        }

        // value

    }
    fn do_move_and_update_whole(&self, action: &<Self::B as BitBoard>::Movement, piece_type: Option<PieceType>, pass_next_moves: bool) 
    -> MoveResultWithSelf<Self, Self::B>;

    fn forward_turn(&mut self){
        self.get_state_mut().incre_turn_count_mut();
        self.get_state_mut().change_side_mut();
    }
    fn check_action_validity(&self, action: &<Self::B as BitBoard>::Movement, piece_type: Option<PieceType>) -> Result<(), InvalidActionError>{
        let mut e = InvalidActionError::from_bits_retain(0);
        let playing_side = self.current_side();

        let opt = action.find_path();
        if opt.is_none(){
            e.insert(InvalidActionError::MOVEMENT_NOT_ORTHOGONAL);
        }

        // If the piece_type is specified, then we check the validity assuming that the movement happens for the specified piece.
        if let Some(p) = piece_type
        {
            if !p.aligns_with_side(playing_side) { e.insert(InvalidActionError::PIECETYPE_CONTRADICTS_STATE);}
            self.invalid_action_check(&mut e, &p, action, &opt);
        }
        else
        {
            let p: PieceType = self.determine_action_piecetype(playing_side, action)?;

            self.invalid_action_check(&mut e, &p, action, &opt);
        }

        if e.bits() == 0 {Ok(())} else { Err(e)}
    }

    fn check_and_do_and_update_whole_mut(&mut self, action: &<Self::B as BitBoard>::Movement, piece_type: Option<PieceType>, pass_next_moves: bool)
    -> SimpleMoveResult<Self::B>{

        let piece_type = 
        if let Some(p) = piece_type { p } 
        else { self.determine_action_piecetype(self.current_side(), action)? }; 

        self.check_action_validity(action, Some(piece_type))?;

        self.do_move_and_update_whole_mut(action, Some(piece_type), pass_next_moves)
    }
}

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
    pub board: TaflBoard<BoardEleven>,
    pub state: GameState,

    pub short_history: ShortHistory<BoardEleven>, 
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
pub struct ShortHistory<B: BitBoard> (ArrayDeque<TaflBoard<B>, 4, Wrapping> );

impl<B: BitBoard> PartialEq for ShortHistory<B> {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter()
            .zip(other.0.iter())
            .all(|(a,b)| a.equals(b))
    }
}

impl<B: BitBoard> ShortHistory<B>{
    pub fn push_front(&mut self, board: TaflBoard<B>) 
    {
        self.0.push_front(board);
    }
    pub fn push_back(&mut self, board: TaflBoard<B>)
    {
        self.0.push_back(board);
    }
    pub fn get_most_recent(&self) -> Option<&TaflBoard<B>> {
        self.0.get(0)
    }
    pub fn get_oldest(&self) -> Option<&TaflBoard<B>> {
        self.0.get(3)
    }
    pub fn get_second_oldest(&self) -> Option<&TaflBoard<B>> {
        self.0.get(2)
    }
    pub fn get(&self, idx: usize) -> Option<&TaflBoard<B>> {
        self.0.get(idx)
    }
    pub fn check_repetition(&self, new_board: &TaflBoard<B>) -> bool
    {
        if let Some(b) = self.get_oldest()
        {
            b.equals(new_board)
        }
        else { false }
    }
    pub fn new() -> Self
    {
        ShortHistory(ArrayDeque::<TaflBoard<B>, 4, Wrapping>::new())
    }
    pub fn update_repetition_counter(&self, new_board: &TaflBoard<B>, repetition_counter: RepetitionCounter) -> RepetitionCounter
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
    pub fn iter(&'_ self) -> arraydeque::Iter<'_, TaflBoard<B>> {
        self.0.iter()
    }
}

impl<B: BitBoard> Default for ShortHistory<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::init_std()
    }
}

// Game-specific methods go here
impl Game{

    //NOTE:  Call this function BEFORE the board is changed
    pub fn update_short_history_mut(&mut self) {
        self.short_history.push_front(self.board);
    }

    // returns true if the new_board (board after action) is repetitive
    pub fn check_repetition(&self, new_board: &TaflBoard<BoardEleven>) -> bool {
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

    pub fn from_board(board: TaflBoard<BoardEleven>, side: Side) -> Self{
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
        let board = TaflBoard::<BoardEleven>::init_std();
        Self::from_board(board, Side::Att)
    }

    // When the action would result in the repetition, returns true. If not, (even when the action is invalid), returns false.
    pub fn move_would_result_in_repetition(&self, side: Side, piece_type: Option<PieceType>, action: &<BoardEleven as BitBoard>::Movement) -> bool {

        let piece_type = 
        if let Some(p) = piece_type { p }
        else { match self.board.determine_action_piecetype(side, action) {
            Ok(pt) => pt,
            Err(_) => { return false; }
        }
        };

        let resulting_board_prior = match piece_type {
            PieceType::AttSoldier => self.board.att_force_move(action),
            PieceType::DefSoldier => self.board.def_force_move(action),
            PieceType::King => self.board.king_force_move(action),
        };
        
        // First test is if the movement itself without subsequent captures would result in repetition.
        // This is a very order-specific way of checking, if one wants more safety, consider instead calling short_history.contains(resulting_board_prior)
        if let Some(reference_board) = self.short_history.get_second_oldest()
        {
            if resulting_board_prior.equals(reference_board) {
                // Second test is if any capture would occur (if it does, then it cannot be a repetition)
                match side {
                    Side::Att => {
                        !(self.board.def_capture(action).is_nonzero() || self.board.king_capture(action).is_nonzero()
                        || self.board.shield_wall_capture(action).is_nonzero())
                    },
                    Side::Def => {
                        !self.board.att_capture(action).is_nonzero()  
                    }
                }
            } else { false }
        } else {
            false
        }
    }

}

impl GameLogic for Game{

    type B = BoardEleven;

    fn get_board(&self) -> &TaflBoard<Self::B> {
        &self.board
    }
    fn get_board_mut(&mut self) -> &mut TaflBoard<Self::B> {
        &mut self.board
    }
    fn get_state(&self) -> &GameState {
        &self.state
    }

    fn get_state_mut(&mut self) -> &mut GameState {
        &mut self.state
    }

    // only for cheap creation of dummy variable
    #[inline]
    fn ghastly() -> Self {
        let board = TaflBoard::<BoardEleven>::new(
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

    fn invalid_action_check_advanced(&self, e: &mut InvalidActionError, p: &PieceType, action: &<Self::B as BitBoard>::Movement, _opt: &Option<Direction>) {
        let side = match p {
            PieceType::AttSoldier => Side::Att,
            PieceType::DefSoldier | PieceType::King => Side::Def,
        };
        if get_rep_counter_for_oldest(self.repetition_counter) == 2 && self.move_would_result_in_repetition(side, Some(*p), action) {
            e.insert(InvalidActionError::THIRD_REPETITION);
        }
        
    }

    fn update_short_history(&mut self) {
        self.short_history.push_back(self.board);
    }

    fn update_repetition_counter(&mut self) {
        self.repetition_counter = self.short_history.update_repetition_counter(&self.board, self.repetition_counter)
    }

    fn yield_captured_def(&self, action: &<Self::B as BitBoard>::Movement) -> (Self::B, Self::B) {
        let def_capt = self.get_board().def_capture(action);
        let king_capt = self.get_board().king_capture(action);
        let shield_wall_capt = self.get_board().shield_wall_capture(action);
        (def_capt | (shield_wall_capt & self.get_board().bit_def), king_capt | (shield_wall_capt & self.get_board().bit_king))
    }

    fn do_action_wo_validity_check_mut(&mut self, action: &<Self::B as BitBoard>::Movement, piece_type: Option<PieceType>) {

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
                } else {
                    let def_has_piece_there = !self.board.bit_def.tile_is_empty_at(&action.start());
                    let king_has_piece_there = !self.board.bit_king.tile_is_empty_at(&action.start());

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
                if (self.board.bit_king & BoardEleven::CORNERS).is_nonzero()
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
    fn do_action_unchecked(&self, action: &<Self::B as BitBoard>::Movement, piece_type: Option<PieceType>) -> Self{

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
                let mut new_state = self.state;
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
                let p = piece_type.unwrap_or_else(|| self.determine_action_piecetype(Side::Def, action).unwrap());
                let mut new_board = match p
                {
                    PieceType::DefSoldier => self.board.def_force_move(action),
                    PieceType::King => self.board.king_force_move(action),
                    PieceType::AttSoldier => panic!("???"),
                };
                let att_capt = self.board.att_capture(action);
                new_board.bit_att ^= att_capt;
                let mut new_state = self.state;
                new_state.change_side_mut();
                if !new_board.bit_att.is_nonzero()
                {
                    new_state.game_over();
                }
                if (new_board.bit_king & BoardEleven::CORNERS).is_nonzero()
                {
                    new_state.game_over();
                } 

                new_state.incre_turn_count_mut();

                let new_repetition_counter = new_short_history.update_repetition_counter(&new_board, self.repetition_counter);                

                Self {board: new_board, state: new_state, short_history: new_short_history, repetition_counter: new_repetition_counter}
            },
        }
        
    }

    // This checks the victory condition for the specified side and update if necessary based on the current board. 
    // It assumes that the specified side is the one who made the last move
    fn update_victory(&mut self, side: Side, pass_next_moves: bool) -> (Option<ReasonForTermination>, Option<Vec<<Self::B as BitBoard>::Movement>>) {
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

                if def_possible_moves.is_empty() && king_possible_moves.is_empty(){
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Att);
                    opt = Some(NoMoveLeft(Side::Def));

                } else if get_rep_counter_for_oldest(self.repetition_counter) == 2 {

                    let no_legal_move_for_defsol = def_possible_moves.iter()
                        .all(|action| self.move_would_result_in_repetition(Side::Def, None, action)
                    );
                    let no_legal_move_for_king = king_possible_moves.iter()
                        .all(|action| self.move_would_result_in_repetition(Side::Def, None, action)
                    );
                    if no_legal_move_for_defsol && no_legal_move_for_king {
                        self.state.game_over_mut();
                        self.state.set_victor(Side::Att);
                        opt = Some(NoLegalMoveLeft(Side::Def));
                    }
                }
                if pass_next_moves {
                    // dbg!();
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
                if att_possible_moves.is_empty() {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Def);
                    opt = Some(NoMoveLeft(Side::Att));
                } else if get_rep_counter_for_oldest(self.repetition_counter) == 2 {
                    let no_legal_move_for_att = att_possible_moves.iter().all(|action|
                        self.move_would_result_in_repetition(Side::Att,Some(PieceType::AttSoldier), action)
                    );
                    if no_legal_move_for_att {
                        self.state.game_over_mut();
                        self.state.set_victor(Side::Def);
                        opt = Some(NoLegalMoveLeft(Side::Att));
                    }
                }
                if (self.board.bit_king & BoardEleven::CORNERS).is_nonzero() {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Def);
                    opt = Some(KingEscaped)
                }
                if pass_next_moves {
                    // dbg!();
                    (opt, Some(att_possible_moves))
                } else {
                    (opt, None)
                }
            }
        }
    }

    // Check if the last move was illegal (currently only checks the repetition) and update the state
    // Assumes 'side' is the one who made the last move
    fn update_if_lost(&mut self, side: Side) -> Option<ReasonForTermination> {

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

    fn do_move_and_update_whole_mut(&mut self, action: &<Self::B as BitBoard>::Movement, piecetype: Option<PieceType>, pass_next_moves: bool) 
    -> Result<(Option<ReasonForTermination>, Option<Vec<<Self::B as BitBoard>::Movement>>), InvalidActionError> {

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
        // let value = if let Some(r) = self.update_if_lost(side) {
        //     Ok((Some(r), None))
        // } else if let (Some(rr), v) = self.update_victory(side, pass_next_moves) {
        //     Ok((Some(rr), v))
        // } else { Ok((None, None))
        // };

        self.forward_turn();

        if let Some(reason) = self.update_if_lost(side) {
            return Ok((Some(reason), None));
        } else {
            let (r, v) = self.update_victory(side, pass_next_moves);
            if r.is_none() {
                return Ok((None, v));
            } else {
                return Ok((Some(r.unwrap()), v));
            }
        }

        // value

    }

    // TODO! unit-test
    fn do_move_and_update_whole(&self, action: &<Self::B as BitBoard>::Movement, piecetype: Option<PieceType>, pass_next_moves: bool)
     -> Result<(Self, Option<ReasonForTermination>, Option<Vec<<Self::B as BitBoard>::Movement>>), InvalidActionError>
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
            state: self.state,
            short_history: new_short_history,
            repetition_counter: new_repetition_counter,
        };
        
        // Check for game termination
        // let postfix = if let Some(r) = new_game.update_if_lost(side) {
        //     dbg!();
        //     (Some(r), None)
        // } else if let (Some(rr), v) = new_game.update_victory(side, pass_next_moves) {
        //     dbg!();
        //     (Some(rr), v)
        // } else { (None, None)
        // };

        new_game.forward_turn();

        if let Some(reason) = new_game.update_if_lost(side) {
            return Ok((new_game, Some(reason), None));
        } else {
            let (r, v) = new_game.update_victory(side, pass_next_moves);
            if r.is_none() {
                return Ok((new_game, None, v));
            } else {
                return Ok((new_game, Some(r.unwrap()), v));
            }
        }

        // Ok((new_game, postfix.0, postfix.1))

    }

}

pub fn parse_repetition_counter(rep_c: RepetitionCounter) -> [u8; 4]{
    let mut a: [u8;4] = [0;4];
    for (i, item) in a.iter_mut().enumerate() {
        let c = (rep_c >> (2 * i)) & 0b11;
        *item = c;
    }
    a
}

pub fn get_rep_counter_for_oldest(rep_c: RepetitionCounter) -> u8 {
    rep_c >> 6 & 0b11
}

pub fn get_rep_counter_for_cur(rep_c: RepetitionCounter) -> u8 {
    rep_c & 0b11
}

fn check_path_obstacle<B: BitBoard>(dir: &Direction, action: &B::Movement, barricade: &B, restricted: &B, e: &mut InvalidActionError){
    let mut tmp = B::new().flip_target_bit(&action.start());
    // let barricade = self.board.bit_att | self.board.bit_def | self.board.bit_king;
    // let restricted = match p{
    //     PieceType::AttSoldier | PieceType::DefSoldier => self.board.hostile,
    //     PieceType::King => BoardEleven::new(),
    // };

    match *dir{
        Direction::E(step) => {
            for i in 0..step{
                tmp = tmp.shift_e();
                if i == step - 1 
                    && (tmp & *restricted).is_nonzero() {
                        e.insert(InvalidActionError::OBSTACLE_IN_PATH);
                    }
                if (tmp & *barricade).is_nonzero() { e.insert(InvalidActionError::DST_IS_RESTRICTED);}
            }
        },
        Direction::W(step) => {
            for i in 0..step{
                tmp = tmp.shift_w();
                if i == step - 1 
                    && (tmp & *restricted).is_nonzero() {
                        e.insert(InvalidActionError::OBSTACLE_IN_PATH);
                    }
                if (tmp & *barricade).is_nonzero() { e.insert(InvalidActionError::DST_IS_RESTRICTED);}
            }
        },
        Direction::S(step) => {
            for i in 0..step{
                tmp = tmp.shift_s();
                if i == step - 1 
                    && (tmp & *restricted).is_nonzero() {
                        e.insert(InvalidActionError::OBSTACLE_IN_PATH);
                    }
                if (tmp & *barricade).is_nonzero() { e.insert(InvalidActionError::DST_IS_RESTRICTED);}
            }
        },
        Direction::N(step) => {
            for i in 0..step{
                tmp = tmp.shift_n();
                if i == step - 1 
                    && (tmp & *restricted).is_nonzero() { 
                        e.insert(InvalidActionError::OBSTACLE_IN_PATH);
                    }
                
                if (tmp & *barricade).is_nonzero() { e.insert(InvalidActionError::DST_IS_RESTRICTED);}
            }
        },
        _ => {}
    }
}

impl std::fmt::Display for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.board)?;
        writeln!(f, "{}", self.state)?;
        let repetitions = parse_repetition_counter(self.repetition_counter);
        let show_rep = format!("Repetition count: {},{},{},{}", repetitions[0],repetitions[1],repetitions[2],repetitions[3]);
        writeln!(f, "{show_rep}")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimpleGame {
    pub board: TaflBoard<BoardSeven>,
    pub state: GameState,
}

impl Default for SimpleGame {
    fn default() -> Self {
        Self::init_std()
    }
}

impl SimpleGame {
    pub fn from_board(board: TaflBoard<BoardSeven>, side: Side) -> Self{
        let state = match side{
            Side::Att => GameState::TURN_ATT,
            Side::Def => GameState(0x0000),
        };
        Self { board, state }
    }

    // generate standard position
    pub fn init_std() -> Self{
        let board = TaflBoard::<BoardSeven>::init_std();
        Self::from_board(board, Side::Att)
    }

}

impl GameLogic for SimpleGame {
    type B = BoardSeven;
    fn get_board(&self) -> &TaflBoard<Self::B> {
        &self.board
    }
    fn get_board_mut(&mut self) -> &mut TaflBoard<Self::B> {
        &mut self.board
    }
    fn get_state(&self) -> &GameState {
        &self.state
    }
    fn get_state_mut(&mut self) -> &mut GameState {
        &mut self.state
    }
    // only for cheap creation of dummy variable
    #[inline]
    fn ghastly() -> Self {
        let board = TaflBoard::<BoardSeven>::new(
            BoardSeven::ghastly(),
            BoardSeven::ghastly(),
            BoardSeven::ghastly(),
            BoardSeven::ghastly()
        );
        Self { 
            board,
            state: GameState(0),
        }
    }

    fn do_action_unchecked(&self, action: &<Self::B as BitBoard>::Movement, piece_type: Option<PieceType>) -> Self {
        let side = self.state.show_side();

        match side{
            Side::Att => {
                let mut new_board = self.board.att_force_move(action);
                let (def_capt, king_capt) = self.yield_captured_def(action);
                new_board.bit_def ^= def_capt;
                new_board.bit_king ^= king_capt;
                let mut new_state = self.state;
                new_state.change_side_mut();
                if king_capt.is_nonzero(){
                    new_state.game_over_mut();
                }
                if new_board.def_is_encircled() {
                    new_state.game_over_mut();
                }
                new_state.incre_turn_count_mut();

                Self { board: new_board, state: new_state }
                
            },
            Side::Def => {
                // If the user specified which piece to move (defender soldier or king), we trust that information
                let p = piece_type.unwrap_or_else(|| self.determine_action_piecetype(Side::Def, action).unwrap());
                let mut new_board = match p
                {
                    PieceType::DefSoldier => self.board.def_force_move(action),
                    PieceType::King => self.board.king_force_move(action),
                    PieceType::AttSoldier => panic!("???"),
                };
                let att_capt = self.board.att_capture(action);
                new_board.bit_att ^= att_capt;
                let mut new_state = self.state;
                new_state.change_side_mut();
                if !new_board.bit_att.is_nonzero()
                {
                    new_state.game_over();
                }
                if (new_board.bit_king & BoardSeven::CORNERS).is_nonzero()
                {
                    new_state.game_over();
                } 

                new_state.incre_turn_count_mut();

                Self {board: new_board, state: new_state }
            },
        }
    }
    fn update_victory(&mut self, side: Side, pass_next_moves: bool) -> (Option<ReasonForTermination>, Option<Vec<<Self::B as BitBoard>::Movement>>) {
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

                if def_possible_moves.is_empty() && king_possible_moves.is_empty() {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Att);
                    opt = Some(NoMoveLeft(Side::Def));
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
                if att_possible_moves.is_empty() {
                    self.state.game_over_mut();
                    self.state.set_victor(Side::Def);
                    opt = Some(NoMoveLeft(Side::Att));
                }
                if (self.board.bit_king & BoardSeven::CORNERS).is_nonzero() {
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
    fn update_if_lost(&mut self, _side: Side) -> Option<ReasonForTermination> {
        None
    }
    fn do_move_and_update_whole(&self, action: &<Self::B as BitBoard>::Movement, piece_type: Option<PieceType>, pass_next_moves: bool) 
        -> Result<(Self, Option<ReasonForTermination>, Option<Vec<<Self::B as BitBoard>::Movement>>), InvalidActionError> {
        
        let side = self.state.show_side();

        let piecetype 
            = if let Some(p) = piece_type { p } 
            else { self.board.determine_action_piecetype(side, action)? };

        let new_board = match piecetype {
            PieceType::AttSoldier => {

                let (def_capt, king_capt) = self.yield_captured_def(action);

                let mut new_board = self.board.att_force_move(action);
                new_board.bit_def ^= def_capt;
                new_board.bit_king ^= king_capt;
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

        let mut new_game = SimpleGame {
            board: new_board,
            state: self.state,
        };
        
        // Check for game termination
        let postfix 
        = if let Some(r) = new_game.update_if_lost(side) 
        { 
            (Some(r), None) 
        } else if let (Some(rr), v) = new_game.update_victory(side, pass_next_moves) 
        {
            (Some(rr), v)
        } else {
            (None, None)
        };

        new_game.forward_turn();

        Ok((new_game, postfix.0, postfix.1))
    }
}

#[cfg(test)]
mod tests{

    use super::*;

    #[test]
    fn game_display_works() {
        let game = Game::init_std();
        print!("{}", game);
    }

    #[test]
    fn repetition_count_works() {
        type ME = <BoardEleven as BitBoard>::Movement;
        let mut game = Game::init_std();
        let move1: ME = ME::try_from("F2G2".to_owned()).unwrap();
        let move2: ME = ME::try_from("F4G4".to_owned()).unwrap();
        let re_move1: ME = ME::try_from("G2F2".to_owned()).unwrap();
        let re_move2: ME = ME::try_from("G4F4".to_owned()).unwrap();

        let actions: [ME;4] = [move1.clone(), move2.clone(), re_move1.clone(), re_move2.clone()];

        for action in actions{
            println!("{}", game);
            println!("short history");
            for b in game.short_history.0.iter() {
                println!("{}", b);
            }
            println!("=======================");
            let new_game = game.do_action_unchecked(&action, None);
            game = new_game;
        }
        println!("{}", game);
        println!("short history");
        for b in game.short_history.0.iter() {
            println!("{}", b);
        }
        println!("=======================");
    }

}

