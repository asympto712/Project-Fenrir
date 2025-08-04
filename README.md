# Project Fenrir &#x130E7;
This project aims to train super-human level AI for Tafl, an ancient Nordic strategy boardgame. It uses 100% pure Rust, because Rust is cool.
# Project overview
The Cargo workspace consists of 3 crates: **bitboard**, **game**, and most importantly, **Fenrir**.
The **bitboard** crate provides the foundation of everything: It defines the bitfield representation of the game board for efficient game implementation.
The **game** crate implements the game logic, and with 
```
cd game
Cargo run
```
, you can play the game on your terminal! 
![This application was made using [_Rataui_](https://ratatui.rs/)](figures/game_tui.png)


