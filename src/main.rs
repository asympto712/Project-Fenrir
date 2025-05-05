use std::fmt::Display;

use rawbytes::RawBytes;

const BOARDELEVENPART1: usize = 48;
const BOARDELEVENPART2: usize = 96;
const BOARDELEVENPART3: usize = 132;

const PRESET1: &'static str = "10001010110\n00100111101\n00100111111\n00001111100\n11110011010\n00110101101\n11000011011\n11000000111\n00000000000\n00011111111\n00011001111\n";

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct BoardEleven {
    par1: u64,
    par2: u64,
    par3: u64,
}
impl BoardEleven {
    fn new() -> Self {
        Self { par1: 0, par2: 0, par3: 0 }
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

fn main() {
    let b = BoardEleven::new();
    // b.par1 += 0b10000;
    print!("{b}");
}

#[test] 
fn works() {
    let preset: String = PRESET1.to_string();
    let board: BoardEleven = BoardEleven::try_from(preset).unwrap();
    // println!("{:b}", board.par1);
    // println!("{:b}", board.par2);
    // println!("{:b}", board.par3);
    print!("{board}");
}
