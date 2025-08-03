#![allow(unused)]
// workspace modules
use bitboard::{BitBoard, PositionalEncoding};
use bitboard::eleven::{BoardEleven, ElevenBoardPositionalEncoding, MoveOnBoardEleven};
use bitboard::seven::{MoveOnBoardSeven, SevenBoardPositionalEncoding};
use color_eyre::eyre::eyre;
use color_eyre::owo_colors::OwoColorize;
use game::game::{parse_repetition_counter, Game, GameState, ShortHistory, Side, SimpleGame, GameLogic};
use game::board::{self, TaflBoard};

// external modules
use color_eyre::{eyre::{self, bail, WrapErr, Report, ErrReport}, Result};
use crossterm::{event::{self, Event, KeyCode, KeyEvent, KeyEventKind}};
use rand::Fill;
use ratatui::symbols;
use ratatui::text::{ToLine, ToText};
use ratatui::widgets::LegendPosition;
use ratatui::{
    layout::{Constraint, Layout, Rect, Position},
    style::{self, Style, Color, Modifier, Stylize},
    text::{Line, Text, Span},
    widgets::{Block, BorderType, List, Padding, Paragraph, Widget, Clear, Borders, Wrap},
    DefaultTerminal,
    Frame,
    buffer::Buffer,
};
use bitflags::{bitflags, bitflags_match};
use arraydeque::ArrayDeque;

use std::error::Error;
use std::fmt::{Debug, Display};
// std
use std::rc::Rc;
use std::sync::LazyLock;

const DARK_GREEN: Color = Color::Rgb(47,115,46);
const DIRT: Color = Color::Rgb(122,54,20);

const SMALL_ATTACKER: &str = "▲";
const SMALL_DEFENDER: &str = "▼";
const SMALL_KING: &str = "♚";
const SMALL_HOSTILE: &str = "▧";
const SMALL_EMPTY: &str = "◻";

const TILE_WIDTH: u16 = 5;           
const TILE_HEIGHT: u16 = 3;

static BOARD_BORDER_SET: symbols::border::Set = symbols::border::Set{
    vertical_left: "▒",
    vertical_right: "▒",
    horizontal_bottom: "▒",
    horizontal_top: "▒",
    top_left: "▒",
    top_right: "▒",
    bottom_left: "▒",
    bottom_right: "▒",
};

//                                  ________
//▲ ▼ ⛊ ♚ ■ ▣ □ ⬜ ◻ ▦ ▩ ▧ ▒▒▒▒▒ ░░░░░  b|b|b
// ☬☬☨☧☦⚚  
// [] A
//⌐\/=≡
// /\ │
//===============================
//   -) A    o] D   +] K  £££££
//  ⌐\/=≡   #}{=#  £≡≡≈ε  £   £
//   /\ │    ╜╙ │   ║║ │  £££££

static BIG_ATTACKER_TEXT: LazyLock<Text<'static>> = LazyLock::new(|| {
    Text::from(vec![
        Line::raw(" -) A"),
        Line::raw(r"⌐\/=≡"),
        Line::raw(r" /\ │")
    ])
});

fn big_attacker() -> Text<'static> {
    BIG_ATTACKER_TEXT.clone()
}

static BIG_DEFENDER: LazyLock<Text<'static>> = LazyLock::new(|| {
    Text::from(vec![
        Line::raw(" o] D"),
        Line::raw(r"#}{=#"),
        Line::raw(r" ╜╙ │")
    ])
});

fn big_defender() -> Text<'static> {
    BIG_DEFENDER.clone()
}

static BIG_KING: LazyLock<Text<'static>> = LazyLock::new(|| {
    Text::from(vec![
        Line::raw(" +] K"),
        Line::raw(r"£≡≡≈ε"),
        Line::raw(r" ║║ │")
    ])
});

fn big_king() -> Text<'static> {
    BIG_KING.clone()
}

static BIG_HOSTILE: LazyLock<Text<'static>> = LazyLock::new(|| {
    Text::from(vec![
        Line::raw("£££££"),
        Line::raw("£   £"),
        Line::raw("£££££")
    ])
});

fn big_hostile() -> Text<'static> {
    BIG_HOSTILE.clone()
}

fn get_big_tile_text<B: BitBoard>(b: &TaflBoard<B>, pos: &<B as BitBoard>::Position) -> Option<Text<'static>>{
    if !b.bit_att.tile_is_empty_at(pos)
    {
        Some(big_attacker().light_red())
    }
    else if !b.bit_king.tile_is_empty_at(pos)
    {
        Some(big_king().yellow())
    }
    else if !b.bit_def.tile_is_empty_at(pos)
    {
        Some(big_defender().white())
    }
    else if !b.hostile.tile_is_empty_at(pos)
    {
        Some(big_hostile())
    }
    else
    {
        None
    }
}


fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let mut terminal = ratatui::init();
    let mut app = App::<Game>::default();
    let app_result = app.run(&mut terminal);
    ratatui::restore();
    app_result
}

#[derive(Default)]
struct App<G: GameLogic>{
    game: G,
    flags: AppFlags,
    input: String,
    character_index: usize,
    error_report: Option<ErrReport>,
    game_end_message: String,
}

bitflags! {
    pub struct AppFlags: u8{
        const ALLOWS_INPUT = 1 << 0;
        const QUIT = 1 << 1;
        const ACCEPT_INPUT = 1 << 2;
        const POPUP = 1 << 3;
        const TERMINATED = 1 << 4;
    }
}

impl Default for AppFlags{
    fn default() -> Self {
        Self::ALLOWS_INPUT | Self::ACCEPT_INPUT
    }
}

trait Draw {
    fn draw(&mut self, frame: &mut Frame);
}

impl Draw for App<Game> {
    
    fn draw(&mut self, frame: &mut Frame){

        use Constraint::{Length,Min,Percentage,Fill,Max};

        let area = frame.area();
        let mut buf = frame.buffer_mut();

        let vertical = Layout::vertical(
            [Length(50), Min(0)]
        ); 
        let [board_display_area, misc_area] = vertical.areas(area);
        let main_board_height = board_display_area.height;
        let board_horizontal = Layout::horizontal(
            [Length(60), Fill(1), Length(54)]
        );

        let [cur_board_area, input_area_prior, recent_history_area_prior] = board_horizontal.areas(board_display_area);
        let [game_state_area, recent_history_area, _] = Layout::vertical([Length(4),Max(40),Fill(1)]).areas(recent_history_area_prior);

        // Render the current board
        let cur_board = self.game.get_board();
        draw_main_board(cur_board_area, buf, cur_board).map_err(|e| {
            let msg = Line::from(format!("{e}"));
            msg.left_aligned().render(cur_board_area, buf);
        });

        // Render the recent histories
        let outer_block = Block::bordered().border_type(BorderType::Rounded).title("recent history (repetition count)");
        let inner_block = outer_block.inner(recent_history_area);
        outer_block.render(recent_history_area, buf);

        if let Ok(rects) = prepare_recent_history_rects(inner_block).map_err(|e| {
            let msg = Line::from(format!("{e}"));
            msg.left_aligned().render(inner_block,buf);
        }) {

            let rc = parse_repetition_counter(self.game.repetition_counter);
            // let recent_history = self.game.short_history
            //     .iter().map(|board| TaflBoardElevenWidget{ board });
            for (i,(rect, b)) in rects.into_iter().zip(self.game.short_history.iter()).enumerate(){
                let true_idx = (i + 1) % 4;
                draw_small_board(b, format!("{} ({})",i+1, rc[true_idx]).to_string(), rect, buf);
                // b.render(rect, buf);
            }

        }

        // Render game state
        self.draw_game_state_block(game_state_area, buf);

        // Render input
        let input_area = Layout::vertical([Length(1),Length(3),Min(0)]).areas::<3>(input_area_prior)[1];
        self.draw_input_block(input_area, buf);
        if self.flags.contains(AppFlags::ALLOWS_INPUT) {
            frame.set_cursor_position(Position::new(
                input_area.x + self.character_index as u16 + 1,
                input_area.y + 1,
            ));
        }

        // Render popup (Error message)
        if self.flags.contains(AppFlags::POPUP)
        {
            let x = frame.area().x;
            let y = frame.area().y;
            let width = frame.area().width;
            let height = frame.area().height;
            let popup_rect = Rect{
                x: x + width / 3,
                y: y + height / 3,
                width: width / 2,
                height: height / 2
            };

            frame.render_widget(Clear, popup_rect);
            let instruction = Line::from(
                "Press <p> to close this message"
            );
            let msg = if let Some(er) = &self.error_report
            {
                Text::from(format!("{er}"))
            } else {
                Text::default()
            };
            Paragraph::new(msg)
                .left_aligned()
                .block(Block::bordered().border_type(BorderType::Double)
                .title_bottom(instruction))
                .render(popup_rect, frame.buffer_mut());
        }

        // Render game end message
        if self.flags.contains(AppFlags::TERMINATED)
        {
            let popup_area = get_popup_area(frame.area());
            frame.render_widget(Clear, popup_area);
            let instruction = vec![
                Span::raw("Press <g> to start new game. "),
                Span::raw("Press <q> to quit")
            ];
            let msg = Text::from(self.game_end_message.to_string());
            Paragraph::new(msg)
                .left_aligned()
                .block(Block::bordered().border_type(BorderType::Double).border_style(Style::default().blue())
                .title_bottom(Line::from(instruction)))
                .render(popup_area, frame.buffer_mut());
        }
    } 


}

impl<G: GameLogic + Default> App<G>
where App<G>: Draw
{

    fn start_afresh(&mut self) {
        // Reset the game to initial state
        self.game = <G as Default>::default();

        // Reset all flags to default state
        self.flags = AppFlags::default();

        // Clear input and reset cursor
        self.input = String::new();
        self.character_index = 0;

        // Clear any error reports
        self.error_report = None;

        // Clear game end message
        self.game_end_message = String::new();

    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()>{
        while !self.flags.contains(AppFlags::QUIT) {
            terminal.draw(|frame| self.draw(frame))?;
            let _ = self.handle_events();
        }
        Ok(())
    }

    fn push_char(&mut self, c: char) {
        if valid_input_char(c) {
            self.input.push(c);
            self.character_index += 1;
        }
    }

    fn pop_char(&mut self) {
        let new_ind = self.character_index.saturating_sub(1);
        if !self.input.is_empty() {
            self.input.pop();
        }
        self.character_index = new_ind;
    }

    fn flush_input(&mut self) {
        self.input = String::new();
        self.character_index = 0;
    }

    fn submit_input(&mut self) -> Result<<G::B as BitBoard>::Movement>{
        let movement = <G::B as BitBoard>::Movement::try_from(self.input.clone()).map_err(|e| ErrReport::msg("e"))?;
        Ok(movement)
    }

    fn handle_events(&mut self) -> Result<()> {
        match event::read()? {
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                // self.handle_key_event(key_event).wrap_err("handling key event failed")?
                let _ = self.handle_key_event(key_event).map_err(|report| {
                    let mut er = if let Some(er) = &self.error_report {
                        Report::msg(format!("{er}"))
                    } else { Report::msg("")};

                    let er = er.wrap_err(report);
                    self.error_report = Some(er);
                    self.flags.insert(AppFlags::POPUP);

                });
            }
            _ => {}
        };
        Ok(())
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) -> Result<()> {
        match key_event.code {
            KeyCode::Char('q') => self.flags.insert(AppFlags::QUIT),
            KeyCode::Char('p') => {
                self.flags.toggle(AppFlags::POPUP);
                self.error_report = None;
            },
            KeyCode::Char('g') if self.flags.contains(AppFlags::TERMINATED) => { self.start_afresh() },
            KeyCode::Char(c) => if self.flags.contains(AppFlags::ALLOWS_INPUT) {self.push_char(c)},
            KeyCode::Backspace => if self.flags.contains(AppFlags::ALLOWS_INPUT) {self.pop_char()},
            KeyCode::Enter => if self.flags.contains(AppFlags::ACCEPT_INPUT) {
                let action = self.submit_input()?;

                let maybe_finished = self.game.check_and_do_and_update_whole_mut(&action, None, false)
                    .map_err(|e| ErrReport::msg(format!("Invalid Action Error!\n{e}")))?;
                self.flush_input();
                if let (Some(r), _) = maybe_finished 
                {
                    self.flags.insert(AppFlags::TERMINATED);
                    let victor = self.game.get_state().get_victor();
                    self.game_end_message = format!("Game finished!\n Winner: {victor}\n {r}");
                    self.flags.remove(AppFlags::ALLOWS_INPUT);
                    self.flags.remove(AppFlags::ACCEPT_INPUT);

                }
            }
            _ => {}
        }
        Ok(())
    }

    fn draw_input_block(&self, area: Rect, buf: &mut Buffer) {
        let input = Line::raw(self.input.as_str())
            .style( if self.flags.contains(AppFlags::ALLOWS_INPUT) {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            });
        Paragraph::new(input).left_aligned().block(Block::bordered().title("Input")).render(area, buf);
    }

    fn draw_game_state_block(&self, area: Rect, buf: &mut Buffer) {
        let msg = Text::raw(format!("{}", self.game.get_state()));
        let mut lines: Vec<Line> = vec![];
        lines.push(Line::raw(format!("Turn: {}", self.game.get_state().get_turn_count())));
        let player: Side = self.game.get_state().show_side();
        let mut player_span = vec![];
        player_span.push(Span::raw("Player: "));
        match player {
            Side::Att => player_span.push(Span::raw("Attacker").red()),
            Side::Def => player_span.push(Span::raw("Defender"))
        };
        lines.push(Line::from(player_span));

        Paragraph::new(lines).block(Block::bordered().title("Game State").border_style(Style::default().blue())).render(area, buf);

    }
}

pub struct GameWidget<'a> {
    pub game: &'a Game,
}

impl<'a> GameWidget<'a> {
    pub fn new(game: &'a Game) -> Self {
        Self { game }
    }
}

fn get_popup_area(area: Rect) -> Rect {

    let x = area.x;
    let y = area.y;
    let width = area.width;
    let height = area.height;
    Rect{
        x: x + width / 3,
        y: y + height / 3,
        width: width / 2,
        height: height / 2
    }
}

fn draw_main_board<B: BitBoard>(area: Rect, buf: &mut Buffer, board: &TaflBoard<B>) -> Result<(), DisplayError>{

// board_border --------+
//-------rank_label-+---+
    let board_size = B::BOARD_SIZE as u16;
    if area.width < 2 + 2 + TILE_WIDTH * board_size || area.height < 1 + 2 + TILE_HEIGHT * board_size
    {
        return Err(DisplayError::TooSmall)
    }

    use Constraint::{Length, Min, Fill, Max};
    let vertical_constraint = [Min(0),Length(1),Length(TILE_HEIGHT * board_size + 2),Min(0)];
    let vlayout = Layout::vertical(vertical_constraint);
    let [_, pre_file_label_area, pre_board_area, _] = vlayout.areas::<4>(area);
    let horizontal_constraint = [Min(0),Length(2),Length(TILE_WIDTH * board_size + 2),Min(0)];
    let hlayout = Layout::horizontal(horizontal_constraint);
    let [_, _, file_label_area_prior, _] = hlayout.clone().areas::<4>(pre_file_label_area);
    let [_,rank_label_prior,board_area,_] = hlayout.areas::<4>(pre_board_area);

    //First, draw the column labels
    let file_label_area = Layout::horizontal([Length(1),Fill(1),Length(1)]).areas::<3>(file_label_area_prior)[1];
    let file_label_cells = Layout::horizontal((0..board_size).map(|_| Max(TILE_WIDTH))).split(file_label_area );
    for i in 0..board_size as usize
    {
        let cell = file_label_cells[i];
        Text::raw(format!("{}", get_file_char(i as u16)))
            .left_aligned()
            .bold()
            .render(cell, buf);
    }

    //Second, draw the row labels
    let rank_label_area = Layout::vertical([Length(1),Fill(1),Length(1)]).areas::<3>(rank_label_prior)[1];
    let rank_label_cells = Layout::vertical((0..board_size).map(|_| Max(TILE_HEIGHT))).split(rank_label_area);
    for i in 0..board_size as usize 
    {
        let cell = rank_label_cells[i];
        Text::raw(format!("{}",i+1))
            .bold()
            .left_aligned()
            .render(cell, buf);
    }

    //Lastly, draw the board with the shady border
    let block = Block::bordered().border_set(BOARD_BORDER_SET).border_style(Style::new().blue());
    let inner_block = block.inner(board_area);
    block.render(board_area, buf);
    draw_big_board(board, inner_block, buf)?;
    Ok(())

}

fn prepare_recent_history_rects(area: Rect) -> Result<[Rect;4] , DisplayError> {

    use Constraint::{Length,Min};
    if area.width < 11 * 2 + 4 || area.height < 12 * 2 + 2{
        return Err(DisplayError::TooSmall)
    }

    let v_constraints = [Min(0),Length(16),Min(2),Length(16),Min(0)];
    let v_layout = Layout::vertical(v_constraints);
    let [_, row1, _, row2,_] = v_layout.areas::<5>(area);
    let h_constraints = [Min(0),Length(2 * 11),Min(4),Length(2 * 11),Min(0)];
    let h_layout = Layout::horizontal(h_constraints);
    let [_, board1, _, board2, _] = h_layout.areas::<5>(row1);
    let [_, board3, _, board4, _] = h_layout.areas::<5>(row2);

    Ok([board1, board2, board3, board4])
}

fn take_square(area: Rect, n: u32) -> Vec<Rect>
{
    use Constraint::{Length, Min, Fill, Ratio, Percentage};
    let main_square 
    = if  2 * area.height > area.width 
    { 
        let pre = Layout::vertical([Min(1), Length(area.width / 2), Min(1)])
        .areas::<3>(area)[1];
        Layout::horizontal([Min(2), Percentage(100), Min(2)])
        .areas::<3>(pre)[1]
    } else {
        let pre = Layout::horizontal([Min(1), Length(2 * area.height), Min(1)])
        .areas::<3>(area)[1];
        Layout::vertical([Min(1), Percentage(100), Min(1)])
        .areas::<3>(pre)[1]
    };

    let vertical_constraints = (0..n).map(|_| Ratio(1,n));
    let vertical = Layout::vertical(vertical_constraints); 
    let rows = vertical.split(main_square).to_vec();
    let horizontal = Layout::horizontal((0..n).map(|_| Ratio(1,n)));
    rows.into_iter().flat_map(|row| horizontal.split(row).to_vec()).collect()
}


#[derive(Debug)]
enum DisplayError{
    TooSmall,
}

impl Display for DisplayError{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self
        {
            DisplayError::TooSmall => write!(f, "Terminal is too small. Try expanding it")
        }
    }
}

impl Error for DisplayError {}

fn draw_small_board<B: BitBoard>(board: &TaflBoard<B>, title: String, area: Rect, buf: &mut Buffer) -> Result<(), DisplayError> {

    use Constraint::{Length, Min, Max, Fill};
    
    let board_size = B::BOARD_SIZE as u16;
    if area.width < board_size || area.height < board_size
    {
        return Err(DisplayError::TooSmall)
    }

    // extract the board area from the input area
    let v_constraint = [Min(0),Length(15),Min(0)];
    let v_layout = Layout::vertical(v_constraint);
    let [_,board_area_prior,_] = v_layout.areas::<3>(area);
    let h_constraint = [Min(0),Length(2 * board_size),Min(0)];
    let h_layout = Layout::horizontal(h_constraint);
    let [_,board_area,_] = h_layout.areas::<3>(board_area_prior);

    // Draw outer block with title
    let outer_block = Block::new()
        .padding(Padding::top(1))
        .title(Line::raw(title).centered());
    let inner_block = outer_block.inner(board_area);
    outer_block.render(board_area, buf);

    // Draw the board
    let mut rows: Vec<Line> = Vec::with_capacity(board_size as usize + 1);
    for j in 0..board_size as usize {
        let row = extract_row_from_small_board(board, j as u8);
        rows.push(row);
    }
    let text = Text::from(rows);
    text.render(inner_block, buf);

    Ok(())
}

fn extract_row_from_small_board<B: BitBoard>(board: &TaflBoard<B>, row_ind: u8) -> Line<'_> {
    let mut row = Line::default();
    for col_ind in 0..B::BOARD_SIZE as usize{
        let pos = <B::Position as PositionalEncoding>::new(col_ind as u8,row_ind);
        if !board.bit_att.tile_is_empty_at(&pos)
        {
            row.spans.push(Span::raw(SMALL_ATTACKER).red());
        }
        else if !board.bit_king.tile_is_empty_at(&pos)
        {
            row.spans.push(Span::raw(SMALL_KING).yellow());
        }
        else if !board.bit_def.tile_is_empty_at(&pos)
        {
            row.spans.push(Span::raw(SMALL_DEFENDER).white());
        }
        else if !board.hostile.tile_is_empty_at(&pos)
        {
            row.spans.push(Span::raw(SMALL_HOSTILE));
        }
        else 
        {
            row.spans.push(Span::raw(SMALL_EMPTY));
        }
        row.spans.push(Span::raw(" "));
        
    }
    row
}

fn draw_big_board<B: BitBoard>(board: &TaflBoard<B>, area: Rect, buf: &mut Buffer) -> Result<(), DisplayError> {

    use Constraint::Length;

    if area.width < TILE_WIDTH * B::BOARD_SIZE as u16 || area.height < TILE_HEIGHT * B::BOARD_SIZE as u16
    {
        return Err(DisplayError::TooSmall)
    }

    // split the area to cells
    let vertical_layout = Layout::vertical(vec![Length(TILE_HEIGHT); B::BOARD_SIZE as usize]).split(area);
    let row_iter = vertical_layout.iter().map(|rect| Layout::horizontal( vec![Length(TILE_WIDTH); B::BOARD_SIZE as usize] ).split(*rect));
    for (j,row) in row_iter.enumerate(){
        for (i,cell) in row.iter().enumerate(){

            if (i+j) % 2 == 0 {
                Block::new().bg(DARK_GREEN).render(*cell, buf);
            } else {
                Block::new().bg(DIRT).render(*cell, buf);
            }
            let pos = <B::Position as PositionalEncoding>::new(i as u8, j as u8);
            if let Some(txt) = get_big_tile_text::<B>(board, &pos){
                txt.render(*cell, buf);
            }
        }
    }
    Ok(())
}


const fn get_file_char(n: u16) -> char {
    match n {
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
        _ => '?',
    }
}

const fn valid_input_char(c: char) -> bool{
    let i = c.len_utf8();
    if i > 1 {
        false
    } else {
        let mut buffer: [u8; 1] = [0; 1];
        c.encode_utf8(&mut buffer);
        let n = buffer[0];
        match n {
            97..108 => true,
            65..76 => true,
            48..=57 => true, // '0' ~ '9'
            _ => false
        }
    }
}

// Wrapper type for TaflBoardEleven to implement Widget
// Deprecated
struct TaflBoardWidget<'a, B: BitBoard> {
    board: &'a TaflBoard<B>,
}

impl<'a, B: BitBoard> TaflBoardWidget<'a, B> {
    pub fn new(board: &'a TaflBoard<B>) -> Self {
        Self { board }
    }
}

// Deprecated
impl<'a, B: BitBoard> Widget for TaflBoardWidget<'a, B> {
    // renders one board using the full area. Note: area is assumed to be square.
    fn render(self, area: Rect, buf: &mut ratatui::prelude::Buffer)
        where
            Self: Sized 
    {
            
        let board_size = B::BOARD_SIZE as u16;
        let cell_width = area.width / board_size;
        let cell_height = area.height / board_size;
        let x = area.x;
        let y = area.y;

        for i in 0..board_size
        {
            for j in 0..board_size
            {
                let cell: Rect = Rect {
                    x: x + i * cell_width,
                    y: y + j * cell_height,
                    width: cell_width,
                    height: cell_height 
                };

                let position= <B::Position as PositionalEncoding>::new(i as u8,j as u8);

                let _cross: char = char::from_u32(0x274C).unwrap();
                let _crown: char = char::from_u32(0x2654).unwrap();
                let text: Text = 
                if !self.board.bit_att.tile_is_empty_at(&position)
                {
                    'A'.to_text().black()
                }
                else if !self.board.bit_def.tile_is_empty_at(&position)
                {
                    'D'.to_text().white()
                }
                else if !self.board.bit_king.tile_is_empty_at(&position)
                {
                    'K'.to_text().white()
                }
                else if !self.board.hostile.tile_is_empty_at(&position)
                {
                    'X'.to_text().red()
                }
                else
                {
                    Text::default()
                };

                let mut borders = Borders::empty();
                if i != board_size - 1 {borders |= Borders::RIGHT;}
                if j != board_size - 1 {borders |= Borders::BOTTOM;}
                Paragraph::new(text.bold())
                    .wrap(Wrap {trim: true})
                    .centered()
                    .block(Block::new().borders(borders).padding(Padding::ZERO).border_style(Style::new().black()))
                    .style(Style::default().bg(Color::Indexed(34)).bold())
                    .render(cell, buf);
            }
        }
    }
}


// Deprecated. 
impl<'a> Widget for GameWidget<'a> {
    fn render(self, area: Rect, buf: &mut ratatui::prelude::Buffer)
        where
            Self: Sized 
    {
        use Constraint::{Length,Min,Percentage,Fill,Max};

        let vertical = Layout::vertical(
            [Length(50), Min(0)]
        ); 
        let [board_display_area, misc_area] = vertical.areas(area);
        let main_board_height = board_display_area.height;
        let board_horizontal = Layout::horizontal(
            [Length(60), Fill(1), Length(60)]
        );

        let [cur_board_area, _, recent_history_area_prior] = board_horizontal.areas(board_display_area);
        let [recent_history_area, _] = Layout::vertical([Length(40),Fill(1)]).areas(recent_history_area_prior);

        // Render the current board
        // let cur_board = TaflBoardElevenWidget{board: &self.game.board };
        draw_main_board(cur_board_area, buf, &self.game.board).map_err(|e| {
            let msg = Line::from(format!("{e}"));
            msg.left_aligned().render(cur_board_area, buf);
        });

        // Render the recent histories
        let outer_block = Block::bordered().border_type(BorderType::Rounded).title("recent history");
        let inner_block = outer_block.inner(recent_history_area);
        outer_block.render(recent_history_area, buf);

        if let Ok(rects) = prepare_recent_history_rects(inner_block).map_err(|e| {
            let msg = Line::from(format!("{e}"));
            msg.left_aligned().render(inner_block,buf);
        }) {

            // let recent_history = self.game.short_history
            //     .iter().map(|board| TaflBoardElevenWidget{ board });
            for (i,(rect, b)) in rects.into_iter().zip(self.game.short_history.iter()).enumerate(){
                draw_small_board(b, format!("{}",i+1).to_string(), rect, buf);
            }

        }
        // for (t, rect) in recent_history.zip(take_square(recent_history_area, 2))
        // {
        //     t.render(rect, buf);
        // }
    }
}
