use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Piece {
    White(WhitePiece),
    Black,
}

impl Piece {
    pub(crate) fn to_num(&self) -> u8 {
        match self {
            Piece::Black => 1,
            Piece::White(WhitePiece::Solider) => 2,
            Piece::White(WhitePiece::King) => 3,
        }
    }

    pub(crate) fn from_num(index: u32) -> Option<Self> {
        match index {
            1 => Some(Piece::Black),
            2 => Some(Piece::White(WhitePiece::Solider)),
            3 => Some(Piece::White(WhitePiece::King)),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum WhitePiece {
    Solider,
    King,
}

impl Display for WhitePiece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::King => write!(f, "*"),
            Self::Solider => write!(f, "^"),
        }
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::White(p) => write!(f, "{p}"),
            Self::Black => write!(f, "#"),
        }
    }
}
