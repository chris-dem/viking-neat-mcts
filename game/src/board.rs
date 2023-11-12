use crate::{game::Pos2D, piece::*};
use bitvec::prelude::*;
use itertools::Itertools;
use std::{fmt::Display, hash::Hash};

/// COL
pub const WIDTH: usize = 11;
/// ROW
pub const HEIGHT: usize = 11;
pub const BOARD_SIZE: usize = HEIGHT * WIDTH;

pub(crate) fn single_to_double(indx: usize) -> (usize, usize) {
    (indx / HEIGHT as usize, indx % HEIGHT as usize)
}

pub(crate) fn double_to_single(row: usize, col: usize) -> usize {
    row * HEIGHT + col
}

pub type Square = Option<Piece>;

#[derive(Debug, Clone)]
pub struct Board {
    pub board: [Option<Piece>; BOARD_SIZE],
}

impl Board {
    pub fn get(&self, (r, c): Pos2D) -> Option<Piece> {
        self.board[double_to_single(r, c)]
    }

    pub fn get_mut(&mut self, (r, c): Pos2D) -> &mut Option<Piece> {
        &mut self.board[double_to_single(r, c)]
    }

    pub fn safe_get(&self, p: Pos2D) -> Option<Option<Piece>> {
        if !self.check_boundary(p) {
            None
        } else {
            Some(self.get(p))
        }
    }

    #[inline]
    pub fn check_boundary(&self, (r, c): Pos2D) -> bool {
        r < HEIGHT && c < WIDTH
    }

    pub fn is_in_king_squares(pos: Pos2D) -> bool {
        pos == START_POSITION || WINNING_POSITIONS.iter().any(|el| *el == pos)
    }

    pub fn is_winning(&self) -> bool {
        WINNING_POSITIONS
            .iter()
            .any(|el| matches!(self.get(*el), Some(Piece::White(WhitePiece::King))))
    }

    pub fn encode(&self) -> (u128, u128) {
        self.board
            .map(|x| {
                (
                    matches!(x, Some(Piece::Black)) as u8,
                    matches!(x, Some(Piece::White(_))) as u8,
                )
            })
            .chunks(11)
            .map(|c| {
                c.into_iter()
                    .fold((0, 0), |(b, w), (bx, wx)| (b + bx, w + wx))
            })
            .fold((0u128, 0u128), |(ab, aw), (xb, xw)| {
                (ab << 4 + xb, aw << 4 + xw)
            })
    }
}

// First triangle
const FIRST_TRIANGLE: [(usize, usize); 6] = [(0, 3), (0, 4), (0, 5), (1, 5), (0, 6), (0, 7)];

// Second triangle
const SECOND_TRIANGLE: [(usize, usize); 6] = [(3, 0), (4, 0), (5, 0), (5, 1), (6, 0), (7, 0)];

// Third triangle
const THIRD_TRIANGLE: [(usize, usize); 6] = [
    (3, WIDTH - 1),
    (4, WIDTH - 1),
    (5, WIDTH - 1),
    (5, WIDTH - 2),
    (6, WIDTH - 1),
    (7, WIDTH - 1),
];

// Forth triangle
const FORTH_TRIANGLE: [(usize, usize); 6] = [
    (HEIGHT - 1, 3),
    (HEIGHT - 1, 4),
    (HEIGHT - 1, 5),
    (HEIGHT - 1 - 1, 5),
    (HEIGHT - 1, 6),
    (HEIGHT - 1, 7),
];

const WHITE_START: [(usize, usize); 12] = [
    (3, 5),
    (4, 4),
    (4, 5),
    (4, 6),
    (5, 3),
    (5, 4),
    (5, 6),
    (5, 7),
    (6, 4),
    (6, 5),
    (6, 6),
    (7, 5),
];

const WINNING_POSITIONS: [(usize, usize); 4] = [
    (0, 0),
    (0, WIDTH - 1),
    (HEIGHT - 1, 0),
    (HEIGHT - 1, WIDTH - 1),
];

const START_POSITION: (usize, usize) = (WIDTH / 2, HEIGHT / 2);

impl Default for Board {
    fn default() -> Self {
        let mut board = [None; BOARD_SIZE];
        for idx in [
            FIRST_TRIANGLE,
            SECOND_TRIANGLE,
            THIRD_TRIANGLE,
            FORTH_TRIANGLE,
        ]
        .iter()
        .flatten()
        .map(|(a, b)| double_to_single(*a, *b))
        {
            board[idx] = Some(Piece::Black);
        }

        for idx in WHITE_START.iter().map(|(a, b)| double_to_single(*a, *b)) {
            board[idx] = Some(Piece::White(WhitePiece::Solider));
        }

        board[double_to_single(START_POSITION.0, START_POSITION.1)] =
            Some(Piece::White(WhitePiece::King));

        Self { board }
    }
}

impl Board {
    pub fn test_new() -> Self {
        let mut board = [None; BOARD_SIZE];
        for idx in [FIRST_TRIANGLE]
            .iter()
            .flatten()
            .map(|(a, b)| double_to_single(*a, *b))
        {
            board[idx] = Some(Piece::Black);
        }

        for idx in WHITE_START
            .iter()
            .take(5)
            .map(|(a, b)| double_to_single(*a, *b))
        {
            board[idx] = Some(Piece::White(WhitePiece::Solider));
        }

        board[double_to_single(START_POSITION.0, START_POSITION.1)] =
            Some(Piece::White(WhitePiece::King));

        Self { board }
    }

    pub fn test_fixed(pos: &[Square]) -> Self {
        assert!(pos.len() == BOARD_SIZE);
        Self {
            board: pos.into_iter().copied().collect_vec().try_into().unwrap(),
        }
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "- |{}", (0..WIDTH).map(|el| format!("{el}")).join("|"))?;
        let output = (0..HEIGHT)
            .map(|r| {
                format!(
                    "{r:2}|{}|",
                    (0..HEIGHT)
                        .map(|c| match self.board[double_to_single(r, c)] {
                            None => " ".to_owned(),
                            Some(p) => format!("{p}"),
                        })
                        .join("|")
                )
            })
            .join("\n");
        write!(f, "{output}")
    }
}
