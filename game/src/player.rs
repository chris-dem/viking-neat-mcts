use crate::{board::Board, game::{
    Pos2D, PlayerEnum, PlayOutcome,
}};

use std::io::stdin;

pub trait PlayerAgent {
    fn receive_outcome(&self, outcome: &PlayOutcome);
    fn play(&self, board : &Board, turn: PlayerEnum) -> (Pos2D,Pos2D);
}

pub struct Person;

fn read_pos(text : &str) -> (Pos2D, Pos2D) {
    let mut buffer = String::new();
    loop {
        buffer.clear();
        println!("{text}");
        if stdin().read_line(&mut buffer).is_err() {
            continue;
        }
        let v = buffer.split_ascii_whitespace().map(|s| s.parse::<usize>()).collect::<Result<Vec<_>,_>>();
        if let Err(_) = v {
            continue;
        }
        let v = v.unwrap();
        if v.len() != 4 {
            continue;
        }
        return ((v[0],v[1]), (v[2],v[3]))
    }
}

impl PlayerAgent for Person {
    fn play(&self, board : &Board, turn : PlayerEnum) ->  (Pos2D,Pos2D) {
        println!("{board}");
        read_pos(format!("Please select your piece and destination, {turn:?}, in the form : 'start_row start_col end_row end_col'").as_str())
    }

    fn receive_outcome(&self, outcome: &PlayOutcome) {
        match outcome {
            PlayOutcome::GameFinished(out) => println!("Player {out:?} won the game!"),
            PlayOutcome::IllegalMove => println!("You played an illegal move"),
            PlayOutcome::ValidMove => (),
        }
    }
}