use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

struct ThresholdAndGini {
    threshold: f64,
    gini: f64,
}

#[pyfunction]
fn get_best_split(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<u8>,
    nb_class: usize,
    thresholds: PyReadonlyArray1<f64>,
) -> Option<(f64, f64)> {
    if thresholds.len() == 0 {
        return None;
    }

    let thresholds = thresholds.as_array();

    let m = y.len() as f64;

    let mut threshold_and_gini: Option<ThresholdAndGini> = None;

    for &threshold in thresholds {
        let gini_sum = get_gini_sum(&x, &y, m, nb_class, threshold);

        match &mut threshold_and_gini {
            Some(ThresholdAndGini {
                threshold: best_threshold,
                gini,
            }) => {
                if *gini > gini_sum {
                    *gini = gini_sum;
                    *best_threshold = threshold;
                }
            }
            None => {
                threshold_and_gini = Some(ThresholdAndGini {
                    threshold,
                    gini: gini_sum,
                })
            }
        }
    }

    threshold_and_gini.map(|ThresholdAndGini { threshold, gini }| (gini, threshold))
}

#[inline(always)]
fn get_gini_sum(
    x: &PyReadonlyArray1<f64>,
    y: &PyReadonlyArray1<u8>,
    m: f64,
    nb_class: usize,
    threshold: f64,
) -> f64 {
    let x = x.as_array();
    let y = y.as_array();

    let mut left_uniq: Vec<f64> = vec![0.; nb_class as usize];
    let mut right_uniq: Vec<f64> = vec![0.; nb_class as usize];

    let mut left_count: f64 = 0.;
    let mut right_count: f64 = 0.;

    for (x_value, &y_value) in x.iter().zip(y) {
        if threshold.gt(x_value) {
            left_count += 1.;
            unsafe {
                *left_uniq.get_unchecked_mut(y_value as usize) += 1.;
            }
        } else {
            right_count += 1.;
            unsafe {
                *right_uniq.get_unchecked_mut(y_value as usize) += 1.;
            }
        }
    }

    let left_gini = 1.
        - left_uniq
            .into_iter()
            .filter(|&count| count != 0.)
            .map(|count| (count / left_count).powi(2))
            .reduce(|a, b| a + b)
            .unwrap_or(0.);

    let right_gini = 1.
        - right_uniq
            .into_iter()
            .filter(|&count| count != 0.)
            .map(|count| (count / right_count).powi(2))
            .reduce(|a, b| a + b)
            .unwrap_or(0.);

    let sum = left_count / m * left_gini + right_count / m * right_gini;

    sum
}

/// A Python module implemented in Rust.
#[pymodule]
fn faimg_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_best_split, m)?)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use numpy::IntoPyArray;
    use pyo3::Python;

    use crate::{get_best_split, get_gini_sum};

    #[test]
    fn test_gini_sum_null() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let x = vec![1., 23., 34., 5., 345., 356., 567.]
                .into_pyarray(py)
                .readonly();
            let y = vec![0, 0, 0, 0, 0, 0, 1].into_pyarray(py).readonly();
            let m = y.len() as f64;
            let nb_class = 2;

            let threshold = 566.;

            // Gini sum supposed to be 0 when homogenous
            assert_eq!(0., get_gini_sum(&x, &y, m, nb_class, threshold));
        });
    }

    #[test]
    fn test_gini_sum_ordering() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let x = vec![1., 2., 3., 4., 5., 6., 7.].into_pyarray(py).readonly();
            let y = vec![1, 1, 1, 0, 0, 0, 0].into_pyarray(py).readonly();
            let m = y.len() as f64;
            let nb_class = 2;

            // Best split is 4.5
            let threshold: f64 = 3.5;

            let gini_best = get_gini_sum(&x, &y, m, nb_class, threshold);

            // Not good split is 5.5
            let threshold: f64 = 5.5;

            let gini_bad = get_gini_sum(&x, &y, m, nb_class, threshold);

            // very bad split is 0
            let threshold: f64 = 0.;

            let gini_very_bad = get_gini_sum(&x, &y, m, nb_class, threshold);

            assert!(gini_best < gini_bad);

            assert!(gini_bad < gini_very_bad);
        });
    }

    #[test]
    fn test_get_best_split() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let x = vec![1., 2., 3., 4., 5., 6., 7.].into_pyarray(py).readonly();
            let y = vec![1, 1, 1, 0, 0, 0, 0].into_pyarray(py).readonly();
            let thresholds = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
                .into_pyarray(py)
                .readonly();

            let m = y.len() as f64;
            let nb_class = 2;

            let actual_best_split = 3.5;

            let actual_best_gini = get_gini_sum(&x, &y, m, nb_class, actual_best_split);
            let (best_gini, best_split) =
                get_best_split(x, y, nb_class, thresholds).expect("to found a result");

            assert_eq!(best_split, actual_best_split);
            assert_eq!(best_gini, actual_best_gini);
        });
    }
}
