pub(crate) mod agent;
pub(crate) mod simulation;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use ndarray::{
        iter::{Axes, AxisIter},
        Array2, Array3, ArrayViewMut2, Axis,
    };

    #[test]
    fn test_windows() {
        let mut arr = Array3::from_shape_vec((5, 5, 4), (0..5 * 5 * 4).collect_vec()).unwrap();
        let min_element = arr.fold(*arr.first().unwrap(), |acc, el| acc.min(*el));
        arr.push(
            Axis(1),
            (Array2::ones((arr.dim().0, arr.dim().2)) * min_element).view(),
        )
        .unwrap();
        arr.push(
            Axis(0),
            (Array2::ones((arr.dim().1, arr.dim().2)) * min_element).view(),
        )
        .unwrap();
        println!("{arr} \n {:?}", arr.dim());

        let win = arr.exact_chunks((2, 2, 1));
        let arr = Array3::from_shape_vec(
            (3, 3, 4),
            win.into_iter()
                .map(|el| el.fold(el[[0, 0, 0]], |acc, el| acc.max(*el)))
                .collect_vec(),
        )
        .unwrap();
        println!("{arr}")
    }
}
