use mcts::transposition_table::*;
use mcts::tree_policy::*;
use mcts::*;

#[derive(Clone, Debug, PartialEq)]
struct CountingGame(i64);

#[derive(Clone, Debug, PartialEq)]
enum Move {
    Add,
    Sub,
}

impl GameState for CountingGame {
    type Move = Move;
    type Player = ();
    type MoveList = Vec<Move>;

    fn current_player(&self) -> Self::Player {
        ()
    }
    fn available_moves(&self) -> Vec<Move> {
        let x = self.0;
        if x == 100 {
            vec![]
        } else {
            vec![Move::Add, Move::Sub]
        }
    }
    fn make_move(&mut self, mov: &Self::Move) {
        match *mov {
            Move::Add => self.0 += 1,
            Move::Sub => self.0 -= 1,
        }
    }
}

impl TranspositionHash for CountingGame {
    fn hash(&self) -> u64 {
        self.0 as u64
    }
}

struct MyEvaluator;

impl Evaluator<MyMCTS> for MyEvaluator {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &CountingGame,
        moves: &Vec<Move>,
        _: Option<SearchHandle<MyMCTS>>,
    ) -> (Vec<()>, i64) {
        (vec![(); moves.len()], state.0)
    }
    fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
        *evaln
    }
    fn evaluate_existing_state(
        &self,
        _: &CountingGame,
        evaln: &i64,
        _: SearchHandle<MyMCTS>,
    ) -> i64 {
        *evaln
    }
}

#[derive(Default)]
struct MyMCTS;

impl MCTS for MyMCTS {
    type State = CountingGame;
    type Eval = MyEvaluator;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ApproxTable<Self>;

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
}
