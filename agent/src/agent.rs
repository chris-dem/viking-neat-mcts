use itertools::Itertools;
use rand::prelude::*;
use std::{
    borrow::BorrowMut,
    cell::RefCell,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};

use game::{
    game::{GameMove, PlayerEnum},
    piece::{Piece, WhitePiece},
    player::PlayerAgent,
};
use mcts::{
    transposition_table::{ApproxTable, TranspositionHash},
    tree_policy::AlphaGoPolicy,
    CycleBehaviour, Evaluator, GameState, MCTSManager, SearchHandle, MCTS,
};
use rustneat::*;

pub struct Simulation {
    enemy_agent: AgentTrain,
}

impl Environment for Simulation {
    fn test(&self, organism: &mut Organism) -> f64 {
        let current_agent = AgentTrain {
            brain: organism.clone(),
        };
        let sim = SimulationEnvironent {
            game: game::game::Game::new(),
            white_agent: Arc::new(Mutex::new(current_agent)),
            black_agent: Arc::new(Mutex::new(self.enemy_agent.clone())),
        };
        let mut mcts = MCTSManager::new(
            sim,
            MyMCTS,
            MyEvaluator,
            AlphaGoPolicy::new(0.5),
            ApproxTable::new(1024),
        );
        mcts.playout_n_parallel(1_000, 4);
        0.
    }
}

#[derive(Debug, Clone)]
pub struct AgentTrain {
    pub brain: Organism,
}

fn to_input_sensors(board: &game::board::Board) -> Vec<f64> {
    board
        .board
        .into_iter()
        .flat_map(|s| match s {
            Some(Piece::Black) => [0., 1., 0., 0.],
            Some(Piece::White(WhitePiece::King)) => [0., 0., 1., 0.],
            Some(Piece::White(WhitePiece::Solider)) => [0., 0., 0., 1.],
            None => [1., 0., 0., 0.],
        })
        .collect_vec()
}

impl AgentTrain {
    pub fn eval_position(
        &mut self,
        board: &game::board::Board,
        moves: &[GameMove],
    ) -> (Vec<f64>, f32) {
        let mut output = vec![0.];
        let mut initial_read = to_input_sensors(board);
        initial_read.push(0.);
        initial_read.push(0.);
        initial_read.push(0.);
        initial_read.push(0.);
        let n = initial_read.len();

        self.brain.activate(&initial_read, &mut output);
        let state = output[0] as f32;
        let v = moves
            .into_iter()
            .map(|GameMove { start_pos, end_pos }| {
                initial_read[n - 4] = start_pos.0 as f64;
                initial_read[n - 3] = start_pos.1 as f64;
                initial_read[n - 2] = end_pos.0 as f64;
                initial_read[n - 1] = end_pos.1 as f64;
                self.brain.activate(&initial_read, &mut output);
                return output[0];
            })
            .collect_vec();
        (v, state)
    }
}

impl PlayerAgent for AgentTrain {
    fn receive_outcome(&self, outcome: &game::game::PlayOutcome) {
        ()
    }

    fn play(
        &self,
        board: &game::board::Board,
        turn: game::game::PlayerEnum,
    ) -> (game::game::Pos2D, game::game::Pos2D) {
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CountingGame(i64);

#[derive(Debug, Clone, Copy)]
enum Direction {
    North,
    West,
    South,
    East,
}

#[derive(Debug, Clone)]
struct SimulationEnvironent {
    game: game::game::Game,
    white_agent: Arc<Mutex<AgentTrain>>,
    black_agent: Arc<Mutex<AgentTrain>>,
}

impl GameState for SimulationEnvironent {
    type Move = GameMove;
    type Player = PlayerEnum;
    type MoveList = Vec<GameMove>;

    fn current_player(&self) -> Self::Player {
        match self.game.state {
            game::game::GameState::Turn(p) => p,
            game::game::GameState::Finished(p) => p,
        }
    }
    fn available_moves(&self) -> Vec<GameMove> {
        match self.game.state {
            game::game::GameState::Turn(p) => self.game.get_available_positions(p),
            game::game::GameState::Finished(_) => vec![],
        }
    }
    fn make_move(&mut self, mov: &Self::Move) {
        self.game.play_round(*mov);
    }
}

impl TranspositionHash for SimulationEnvironent {
    fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        let (a, b) = self.game.board.encode();
        a.hash(&mut hasher);
        b.hash(&mut hasher);
        hasher.finish()
    }
}

struct MyEvaluator;

impl Evaluator<MyMCTS> for MyEvaluator {
    type StateEvaluation = f32;

    fn evaluate_new_state(
        &self,
        state: &SimulationEnvironent,
        moves: &Vec<GameMove>,
        _: Option<SearchHandle<MyMCTS>>,
    ) -> (Vec<f64>, f32) {
        match state.game.state {
            game::game::GameState::Turn(p) => {
                let agent = match p {
                    PlayerEnum::Black => state.black_agent.clone(),
                    PlayerEnum::White => state.white_agent.clone(),
                };
                let mut agent = agent.lock().expect("");
                agent.eval_position(&state.game.board, moves)
            }
            game::game::GameState::Finished(PlayerEnum::Black) => (vec![], -15.),
            game::game::GameState::Finished(PlayerEnum::White) => (vec![], 15.),
        }
    }

    /// Check how every player gets affected
    fn interpret_evaluation_for_player(&self, evaln: &f32, player: &PlayerEnum) -> i64 {
        let res = (evaln * 10.).round() as i64;
        match player {
            PlayerEnum::Black => res * -1,
            PlayerEnum::White => res,
        }
    }
    fn evaluate_existing_state(
        &self,
        _: &SimulationEnvironent,
        evaln: &f32,
        _: SearchHandle<MyMCTS>,
    ) -> f32 {
        *evaln
    }
}

#[derive(Default)]
struct MyMCTS;

impl MCTS for MyMCTS {
    type State = SimulationEnvironent;
    type Eval = MyEvaluator;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = AlphaGoPolicy;
    type TranspositionTable = ApproxTable<Self>;

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
}
