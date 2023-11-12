use std::collections::HashMap;

use itertools::Itertools;

use crate::{
    board::{double_to_single, Board, Square, HEIGHT, WIDTH},
    piece::{Piece, WhitePiece},
};

#[derive(Debug, Clone)]
pub struct Game {
    pub board: Board,
    pub state: GameState,
}

#[derive(Debug, Clone, Copy)]
pub enum GameState {
    Turn(PlayerEnum),
    Finished(PlayerEnum),
}

pub type Pos2D = (usize, usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerEnum {
    Black,
    White,
}

impl PlayerEnum {
    pub fn flip(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }

    /// Check that the player picked the right piece
    #[inline]
    pub fn check_piece(&self, other: Option<Piece>) -> bool {
        matches!(
            (self, other),
            (PlayerEnum::Black, Some(Piece::Black)) | (PlayerEnum::White, Some(Piece::White(_)))
        )
    }

    /// Check that the player can capture solider
    #[inline]
    pub fn check_capture_small(&self, other: Option<Piece>) -> bool {
        matches!(
            (self, other),
            (PlayerEnum::White, Some(Piece::Black))
                | (PlayerEnum::Black, Some(Piece::White(WhitePiece::Solider)))
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayOutcome {
    GameFinished(PlayerEnum),
    IllegalMove,
    ValidMove,
}

type PosInt = (i32, i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GameMove {
    pub start_pos: Pos2D,
    pub end_pos: Pos2D,
}

impl Game {
    pub fn new() -> Self {
        Self {
            board: Board::default(),
            state: GameState::Turn(PlayerEnum::Black),
        }
    }

    pub(crate) fn test_new() -> Self {
        Self {
            board: Board::test_new(),
            state: GameState::Turn(PlayerEnum::Black),
        }
    }

    pub(crate) fn test_fixed(pos: &[Square]) -> Self {
        Self {
            board: Board::test_fixed(pos),
            state: GameState::Turn(PlayerEnum::Black),
        }
    }

    /// Check that all positions in between start and and end are empty. Does not validated input
    fn check_positions(&self, start @ (s_x, s_y): Pos2D, (e_x, e_y): Pos2D) -> bool {
        if s_x == e_x {
            let (s, e) = ((e_y).min(s_y), (e_y).max(s_y));
            (s..=e)
                .filter(|p| (s_x, *p) != start)
                .map(|el| self.board.get((s_x, el)))
                .all(|x| x.is_none())
        } else {
            let (s, e) = ((e_x).min(s_x), (e_x).max(s_x));
            (s..=e)
                .filter(|p| (*p, s_y) != start)
                .map(|el| self.board.get((el, s_y)))
                .all(|x| x.is_none())
        }
    }

    /// Check that soliders don't finish on king squares
    #[inline]
    fn check_move(&self, piece: Option<Piece>, pos_end: Pos2D) -> bool {
        match piece {
            Some(Piece::White(WhitePiece::Solider)) | Some(Piece::Black) => {
                !Board::is_in_king_squares(pos_end)
            }
            _ => true,
        }
    }

    /// Funciton to check if the play of the player agent is valid
    #[inline]
    fn check_valid(&self, pos_start: Pos2D, pos_end: Pos2D, turn: PlayerEnum) -> bool {
        if !self.board.check_boundary(pos_start) || !self.board.check_boundary(pos_end) {
            return false;
        }
        let piece = self.board.get(pos_start);
        turn.check_piece(piece)
        // Check that the player made a move
        && pos_start != pos_end
        // Ensure either vertical or horizontal move

        && (pos_start.0 == pos_end.0 || pos_start.1 == pos_end.1)
        && self.check_positions(pos_start, pos_end)
        && self.check_move(piece, pos_end)
    }

    /// Apply the move and give out come for any possibilities
    fn apply_move(&mut self, pos_start: Pos2D, pos_end: Pos2D, turn: PlayerEnum) -> PlayOutcome {
        // Check valid move
        if !self.check_valid(pos_start, pos_end, turn) {
            return PlayOutcome::IllegalMove;
        }
        // Move piece
        let p = self.board.get(pos_start);
        *self.board.get_mut(pos_end) = Some(p.unwrap());
        *self.board.get_mut(pos_start) = None;
        // Check captures
        let capts = self.check_captures(pos_end, turn);
        for c in capts {
            match c {
                CaptureEnum::King(_) => return PlayOutcome::GameFinished(PlayerEnum::Black),
                CaptureEnum::Solider(p) => *self.board.get_mut(p) = None,
                CaptureEnum::Nothing => (),
            }
        }
        // Check King squares
        if self.board.is_winning() {
            return PlayOutcome::GameFinished(PlayerEnum::White);
        }
        PlayOutcome::ValidMove
    }

    fn check_capture(
        &self,
        (f_x, f_y): PosInt,
        (s_x, s_y): PosInt,
        turn: PlayerEnum,
    ) -> CaptureEnum {
        if f_x < 0 || f_y < 0 || s_x < 0 || s_y < 0 {
            return CaptureEnum::Nothing;
        }
        let p1 @ (r, c) = (f_x as usize, f_y as usize);
        let p2 = (s_x as usize, s_y as usize);
        let small = self
            .board
            .safe_get(p1)
            .filter(|e| turn.check_capture_small(*e))
            .is_some()
            && self
                .board
                .safe_get(p2)
                .filter(|e| turn.check_piece(*e) || (e.is_none() && Board::is_in_king_squares(p2)))
                .is_some();
        let big = if self
            .board
            .safe_get(p1)
            .filter(|p| matches!(p, Some(Piece::White(WhitePiece::King))))
            .is_some()
        {
            [(r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c)]
                .into_iter()
                .map(|e| self.board.safe_get(e))
                .collect::<Option<Vec<_>>>()
                .filter(|v| {
                    if v.len() != 4 {
                        false
                    } else {
                        v.into_iter().all(|p| matches!(p, Some(Piece::Black)))
                    }
                })
                .is_some()
        } else {
            false
        };
        if big {
            CaptureEnum::King(p1)
        } else if small {
            CaptureEnum::Solider(p1)
        } else {
            CaptureEnum::Nothing
        }
    }

    fn check_captures(&self, (r, c): Pos2D, turn: PlayerEnum) -> [CaptureEnum; 4] {
        let (r, c) = (r as i32, c as i32);
        [
            self.check_capture((r, c + 1), (r, c + 2), turn),
            self.check_capture((r, c - 1), (r, c - 2), turn),
            self.check_capture((r + 1, c), (r + 2, c), turn),
            self.check_capture((r - 1, c), (r - 2, c), turn),
        ]
    }

    pub fn play_round(&mut self, game_move: GameMove) -> Option<GameState> {
        let turn = match &self.state {
            GameState::Turn(p) => *p,
            r => return Some(*r),
        };

        let GameMove { start_pos, end_pos } = game_move;
        let outcome = self.apply_move(start_pos, end_pos, turn);

        match outcome {
            PlayOutcome::GameFinished(p) => return Some(GameState::Finished(p)),
            PlayOutcome::IllegalMove => None, // If invalid, don't do anything
            PlayOutcome::ValidMove => Some(GameState::Turn(turn.flip())),
        }
    }

    fn available_dir(
        &self,
        outer_range: std::ops::Range<usize>,
        inner_range: std::ops::Range<usize>,
        should_flip: bool,
        map: &mut HashMap<Pos2D, u32>,
        directions: (Direction, Direction),
        player: PlayerEnum,
    ) {
        let mut counter = 0;
        let mut prev_piece: Option<Pos2D> = None;
        for i in outer_range {
            for j in inner_range.clone() {
                let (i, j) = if should_flip { (j, i) } else { (i, j) };
                let curr_cell = self.board.board[double_to_single(i, j)];
                if let Some(curr_piece) = curr_cell {
                    update_previous(prev_piece, counter, map, directions.0);
                    if player.check_piece(curr_cell) {
                        prev_piece = Some((i, j));
                        *map.entry((i, j)).or_insert((curr_piece).to_num() as u32) ^=
                            counter << (4 * directions.1 as u32) + 2;
                    } else {
                        prev_piece = None;
                    }
                    counter = 0;
                } else {
                    counter += 1
                }
            }
            update_previous(prev_piece, counter, map, directions.0);
            counter = 0;
            prev_piece = None;
        }
    }
    pub fn get_available_positions(&self, player: PlayerEnum) -> Vec<GameMove> {
        let mut map: HashMap<Pos2D, u32> = HashMap::with_capacity(24);
        self.available_dir(
            0..HEIGHT,
            0..WIDTH,
            false,
            &mut map,
            (Direction::East, Direction::West),
            player,
        );
        self.available_dir(
            0..WIDTH,
            0..HEIGHT,
            true,
            &mut map,
            (Direction::South, Direction::North),
            player,
        );
        map.into_iter()
            .flat_map(|(start_pos @ (x, y), v)| {
                let kind = Piece::from_num(v & 3)
                    .expect(format!("Index should be 1 2 or 3, not {}", v & 3).as_str());

                let [north, east, south, west] = [
                    (v >> (2 + 4 * 0)) & 15,
                    (v >> (2 + 4 * 1)) & 15,
                    (v >> (2 + 4 * 2)) & 15,
                    (v >> (2 + 4 * 3)) & 15,
                ];
                let it = [
                    Box::new((1..=north).map(move |n| (x - n as usize, y)))
                        as Box<dyn Iterator<Item = Pos2D>>,
                    Box::new((1..=east).map(move |n| (x, y + n as usize)))
                        as Box<dyn Iterator<Item = Pos2D>>,
                    Box::new((1..=south).map(move |n| (x + n as usize, y)))
                        as Box<dyn Iterator<Item = Pos2D>>,
                    Box::new((1..=west).map(move |n| (x, y - n as usize)))
                        as Box<dyn Iterator<Item = Pos2D>>,
                ]
                .into_iter()
                .flatten();
                let v = if kind != Piece::White(WhitePiece::King) {
                    Box::new(it.filter(|pos| !Board::is_in_king_squares(*pos)))
                        as Box<dyn Iterator<Item = Pos2D>>
                } else {
                    Box::new(it)
                };
                v.map(move |p| GameMove {
                    start_pos,
                    end_pos: p,
                })
            })
            .collect_vec()
    }
}

#[inline]
fn update_previous(
    prev: Option<Pos2D>,
    counter: u32,
    map: &mut HashMap<Pos2D, u32>,
    direction: Direction,
) {
    if counter != 0 {
        if let Some(p) = prev {
            map.entry(p)
                .and_modify(|bitwise| *bitwise ^= counter << ((4 * direction.as_index()) + 2));
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    fn as_index(&self) -> usize {
        (*self).into()
    }
}

impl Into<usize> for Direction {
    fn into(self) -> usize {
        match self {
            Direction::North => 0,
            Direction::East => 1,
            Direction::South => 2,
            Direction::West => 3,
        }
    }
}

#[derive(Debug)]
enum CaptureEnum {
    Nothing,
    Solider(Pos2D),
    King(Pos2D),
}

#[cfg(test)]
mod tests {
    use crate::board::BOARD_SIZE;

    use super::*;
    use itertools::Itertools;
    use rand::prelude::*;
    use std::collections::BTreeSet;

    mod seed_23 {
        use super::*;
        #[test]
        fn test_1() {
            let mut rng = StdRng::seed_from_u64(23);
            let mut board = [None; BOARD_SIZE];
            let pos = board
                .into_iter()
                .enumerate()
                .choose_multiple(&mut rng, 2)
                .into_iter()
                .map(|el| el.0)
                .collect_vec();
            for el in pos {
                board[el] = Some(Piece::Black);
            }

            let game = Game::test_fixed(&board);
            let avail_pos_vec = game.get_available_positions(PlayerEnum::Black);
            let avail_pos = avail_pos_vec.clone().into_iter().collect::<BTreeSet<_>>();
            assert_eq!(avail_pos.len(), avail_pos_vec.len());
            let test_pos = (1..=6)
                .map(|el| GameMove {
                    start_pos: (7, 0),
                    end_pos: (el, 0),
                })
                .chain((8..=9).map(|el| GameMove {
                    start_pos: (7, 0),
                    end_pos: (el, 0),
                }))
                .chain((1..=10).map(|el| GameMove {
                    start_pos: (7, 0),
                    end_pos: (7, el),
                }))
                .chain((0..=2).map(|el| GameMove {
                    start_pos: (3, 3),
                    end_pos: (el, 3),
                }))
                .chain((4..=10).map(|el| GameMove {
                    start_pos: (3, 3),
                    end_pos: (el, 3),
                }))
                .chain((0..=2).map(|el| GameMove {
                    start_pos: (3, 3),
                    end_pos: (3, el),
                }))
                .chain((4..=10).map(|el| GameMove {
                    start_pos: (3, 3),
                    end_pos: (3, el),
                }))
                .collect::<BTreeSet<_>>();
            let n = test_pos.len();
            assert_eq!(avail_pos.intersection(&test_pos).count(), n);
        }

        #[test]
        fn test_2() {
            let mut rng = StdRng::seed_from_u64(23);
            let mut board = [None; BOARD_SIZE];
            let pos = board
                .into_iter()
                .enumerate()
                .choose_multiple(&mut rng, 2)
                .into_iter()
                .map(|el| el.0)
                .collect_vec();
            board[pos[0]] = Some(Piece::Black);
            board[pos[1]] = Some(Piece::White(WhitePiece::King));

            let game = Game::test_fixed(&board);
            // Black check
            let avail_pos_vec = game.get_available_positions(PlayerEnum::Black);
            let avail_pos = avail_pos_vec.clone().into_iter().collect::<BTreeSet<_>>();
            assert_eq!(avail_pos.len(), avail_pos_vec.len());
            let test_pos_black = (0..=2)
                .map(|el| GameMove {
                    start_pos: (3, 3),
                    end_pos: (el, 3),
                })
                .chain((4..=10).map(|el| GameMove {
                    start_pos: (3, 3),
                    end_pos: (el, 3),
                }))
                .chain((0..=2).map(|el| GameMove {
                    start_pos: (3, 3),
                    end_pos: (3, el),
                }))
                .chain((4..=10).map(|el| GameMove {
                    start_pos: (3, 3),
                    end_pos: (3, el),
                }))
                .collect::<BTreeSet<_>>();
            let n = test_pos_black.len();
            assert_eq!(avail_pos.intersection(&test_pos_black).count(), n);

            // White check
            let avail_pos_vec = game.get_available_positions(PlayerEnum::White);
            let avail_pos = avail_pos_vec.clone().into_iter().collect::<BTreeSet<_>>();
            assert_eq!(avail_pos.len(), avail_pos_vec.len());
            let test_pos_white = (0..=6)
                .map(|el| GameMove {
                    start_pos: (7, 0),
                    end_pos: (el, 0),
                })
                .chain((8..=10).map(|el| GameMove {
                    start_pos: (7, 0),
                    end_pos: (el, 0),
                }))
                .chain((1..=10).map(|el| GameMove {
                    start_pos: (7, 0),
                    end_pos: (7, el),
                }))
                .collect::<BTreeSet<_>>();
            let n = test_pos_white.len();
            assert_eq!(avail_pos.intersection(&test_pos_white).count(), n);
        }
    }

    mod seed_43 {
        use super::*;

        #[ignore = "testing"]
        #[test]
        fn display_board() {
            let mut rng = StdRng::seed_from_u64(43);
            let mut board = [None; BOARD_SIZE];
            let pos = board
                .into_iter()
                .enumerate()
                .choose_multiple(&mut rng, 5)
                .into_iter()
                .map(|el| el.0)
                .collect_vec();

            for el in pos {
                board[el] = Some(Piece::Black);
            }

            let game = Game::test_fixed(&board);
            println!("{}", game.board);
        }

        #[test]
        fn test_3() {
            let mut rng = StdRng::seed_from_u64(43);
            let mut board = [None; BOARD_SIZE];
            let pos = board
                .into_iter()
                .enumerate()
                .choose_multiple(&mut rng, 5)
                .into_iter()
                .map(|el| el.0)
                .collect_vec();

            for el in pos {
                board[el] = Some(Piece::Black);
            }

            let game = Game::test_fixed(&board);
            let avail_pos_vec = game.get_available_positions(PlayerEnum::Black);
            let avail_pos = avail_pos_vec.clone().into_iter().collect::<BTreeSet<_>>();
            assert_eq!(avail_pos.len(), avail_pos_vec.len());

            let test_pos = [
                Box::new(
                    [
                        Box::new((0..=1).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((3..=9).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=0).map(|el: usize| (el, 2usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..=10).map(|el: usize| (el, 2usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (1, 2),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((3..=9).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..10).map(|el: usize| (el, 10usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (1, 10),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=3).map(|el: usize| (4usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((5..=10).map(|el: usize| (4usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=3).map(|el: usize| (el, 4usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((5..=10).map(|el: usize| (el, 4usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (4, 4),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=0).map(|el: usize| (6usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..=10).map(|el: usize| (6usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=5).map(|el: usize| (el, 1usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((7..=10).map(|el: usize| (el, 1usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (6, 1),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=7).map(|el: usize| (8usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((9..=10).map(|el: usize| (8usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=7).map(|el: usize| (el, 8usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((9..=10).map(|el: usize| (el, 8usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (8, 8),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
            ]
            .into_iter()
            .flatten()
            .collect::<BTreeSet<_>>();

            let n = test_pos.len();
            assert_eq!(avail_pos.intersection(&test_pos).count(), n);
        }

        #[test]
        fn test_4() {
            let mut rng = StdRng::seed_from_u64(43);
            let mut board = [None; BOARD_SIZE];
            let pos = board
                .into_iter()
                .enumerate()
                .choose_multiple(&mut rng, 5)
                .into_iter()
                .map(|el| el.0)
                .collect_vec();

            for el in pos {
                board[el] = Some(Piece::White(WhitePiece::Solider));
            }

            board[double_to_single(1, 10)] = Some(Piece::White(WhitePiece::King));

            let game = Game::test_fixed(&board);
            let avail_pos_vec = game.get_available_positions(PlayerEnum::White);
            let avail_pos = avail_pos_vec.clone().into_iter().collect::<BTreeSet<_>>();
            assert_eq!(avail_pos.len(), avail_pos_vec.len());

            let test_pos = [
                Box::new(
                    [
                        Box::new((0..=1).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((3..=9).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=0).map(|el: usize| (el, 2usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..=10).map(|el: usize| (el, 2usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (1, 2),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((3..=9).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=0).map(|el: usize| (el, 10usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..=10).map(|el: usize| (el, 10usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (1, 10),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=3).map(|el: usize| (4usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((5..=10).map(|el: usize| (4usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=3).map(|el: usize| (el, 4usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((5..=10).map(|el: usize| (el, 4usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (4, 4),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=0).map(|el: usize| (6usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..=10).map(|el: usize| (6usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=5).map(|el: usize| (el, 1usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((7..=10).map(|el: usize| (el, 1usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (6, 1),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=7).map(|el: usize| (8usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((9..=10).map(|el: usize| (8usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=7).map(|el: usize| (el, 8usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((9..=10).map(|el: usize| (el, 8usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (8, 8),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
            ]
            .into_iter()
            .flatten()
            .collect::<BTreeSet<_>>();
            let n = test_pos.len();
            assert_eq!(avail_pos.intersection(&test_pos).count(), n);
        }

        #[test]
        fn test_5() {
            let mut rng = StdRng::seed_from_u64(43);
            let mut board = [None; BOARD_SIZE];
            let pos = board
                .into_iter()
                .enumerate()
                .choose_multiple(&mut rng, 5)
                .into_iter()
                .map(|el| el.0)
                .collect_vec();

            for el in pos {
                board[el] = Some(Piece::White(WhitePiece::Solider));
            }

            board[double_to_single(6, 5)] = Some(Piece::White(WhitePiece::Solider));

            let game = Game::test_fixed(&board);
            let avail_pos_vec = game.get_available_positions(PlayerEnum::White);
            let avail_pos = avail_pos_vec.clone().into_iter().collect::<BTreeSet<_>>();
            assert_eq!(avail_pos.len(), avail_pos_vec.len());

            let test_pos = [
                Box::new(
                    [
                        Box::new((0..=1).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((3..=9).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=0).map(|el: usize| (el, 2usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..=10).map(|el: usize| (el, 2usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (1, 2),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((3..=9).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..10).map(|el: usize| (el, 10usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (1, 10),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=3).map(|el: usize| (4usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((5..=10).map(|el: usize| (4usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=3).map(|el: usize| (el, 4usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((5..=10).map(|el: usize| (el, 4usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (4, 4),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=0).map(|el: usize| (6usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..=4).map(|el: usize| (6usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=5).map(|el: usize| (el, 1usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((7..=10).map(|el: usize| (el, 1usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (6, 1),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=7).map(|el: usize| (8usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((9..=10).map(|el: usize| (8usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=7).map(|el: usize| (el, 8usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((9..=10).map(|el: usize| (el, 8usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (8, 8),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((2..=4).map(|el: usize| (6usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((6..=10).map(|el: usize| (6usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((7..=10).map(|el: usize| (el, 5usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (6, 5),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
            ]
            .into_iter()
            .flatten()
            .collect::<BTreeSet<_>>();

            let n = test_pos.len();
            assert_eq!(avail_pos.intersection(&test_pos).count(), n);
        }

        #[test]
        fn test_6() {
            let mut rng = StdRng::seed_from_u64(43);
            let mut board = [None; BOARD_SIZE];
            let pos = board
                .into_iter()
                .enumerate()
                .choose_multiple(&mut rng, 5)
                .into_iter()
                .map(|el| el.0)
                .collect_vec();
            for el in pos.iter().copied() {
                board[el] = Some(Piece::White(WhitePiece::Solider));
            }
            board[double_to_single(6, 5)] = Some(Piece::Black);

            let game = Game::test_fixed(&board);
            let avail_pos_vec = game.get_available_positions(PlayerEnum::White);
            let avail_pos = avail_pos_vec.clone().into_iter().collect::<BTreeSet<_>>();
            assert_eq!(avail_pos.len(), avail_pos_vec.len());

            let test_pos = [
                Box::new(
                    [
                        Box::new((0..=1).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((3..=9).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=0).map(|el: usize| (el, 2usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..=10).map(|el: usize| (el, 2usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (1, 2),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((3..=9).map(|el: usize| (1usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..10).map(|el: usize| (el, 10usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (1, 10),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=3).map(|el: usize| (4usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((5..=10).map(|el: usize| (4usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=3).map(|el: usize| (el, 4usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((5..=10).map(|el: usize| (el, 4usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (4, 4),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=0).map(|el: usize| (6usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((2..=4).map(|el: usize| (6usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=5).map(|el: usize| (el, 1usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((7..=10).map(|el: usize| (el, 1usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (6, 1),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
                Box::new(
                    [
                        Box::new((0..=7).map(|el: usize| (8usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((9..=10).map(|el: usize| (8usize, el)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((0..=7).map(|el: usize| (el, 8usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                        Box::new((9..=10).map(|el: usize| (el, 8usize)))
                            as Box<dyn Iterator<Item = Pos2D>>,
                    ]
                    .into_iter()
                    .flatten()
                    .map(|end_pos| GameMove {
                        start_pos: (8, 8),
                        end_pos,
                    }),
                ) as Box<dyn Iterator<Item = GameMove>>,
            ]
            .into_iter()
            .flatten()
            .collect::<BTreeSet<_>>();

            let n = test_pos.len();
            assert_eq!(avail_pos.intersection(&test_pos).count(), n);
        }
    }

    mod proptests {
        use proptest::prelude::*;

        use crate::{
            board::{double_to_single, Board, BOARD_SIZE},
            game::{Game, GameMove, PlayerEnum},
            piece::Piece,
        };
        use itertools::Itertools;

        proptest! {
                #[test]
                fn should_be_valid_moves(a in proptest::array::uniform16((0..11usize,0..11usize,1..=3u32))
                .prop_map(|arr| arr.map(|(x,y,s)| (x,y,Piece::from_num(s).unwrap())))
                .prop_filter("Should contain unique values", |arr| arr.len() == arr.iter().map(|(x,y,_)| (x,y)).unique().count())
                        .prop_filter("Should contain legal starting positions", |arr| arr.iter().all(|(x,y,s)| {
                            *s == Piece::White(crate::piece::WhitePiece::King) || !Board::is_in_king_squares((*x,*y))
                        })), p in  any::<bool>()
                    )

                        {
                            let player = if p { PlayerEnum::White } else {PlayerEnum::White};
                            let mut board = [None ; BOARD_SIZE];
                            for (row,col,p) in a {
                                board[double_to_single(row, col)] = Some(p);
                            }

                            let game = Game::test_fixed(&board);
                            let cond = game.get_available_positions(player).into_iter().all(|GameMove {start_pos, end_pos}| game.check_valid(start_pos, end_pos, player));
                            prop_assert!(cond);
                        }
        }
    }
}
