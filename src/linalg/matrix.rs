use std::{
    array,
    fmt::{self, Debug},
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
    usize,
};

use crate::linalg::{traits::Field, vector::Vector};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Matrix<T: Field, const N: usize, const M: usize>([[T; N]; M]);
impl<T: Field, const N: usize, const M: usize> Matrix<T, N, M> {
    pub fn convert<E: Field + From<T>>(self) -> Matrix<E, N, M> {
        Matrix(self.0.map(|row| row.map(<E as From<T>>::from)))
    }
    // Turns Matrix<T, N, M> into Matrix<T, M, N>
    pub fn transpose(self) -> Matrix<T, M, N> {
        Matrix(array::from_fn(|i| array::from_fn(|j| self[(j, i)])))
    }
    pub fn as_column_arrays(self) -> [[T; M]; N] {
        self.transpose().as_row_arrays()
    }
    pub fn from_outer_product(row_vec: Vector<T, M>, col_vec: Vector<T, M>) -> Self {
        let mut data = Self::zero();
        for i in 0..N {
            for j in 0..M {
                data[(i, j)] = row_vec[i] * col_vec[j]
            }
        }
        data
    }
    pub fn as_column_vectors(self) -> [Vector<T, M>; N] {
        self.as_column_arrays().map(Vector::from)
    }
    pub fn as_row_arrays(self) -> [[T; N]; M] {
        self.0
    }
    pub fn as_row_vectors(self) -> [Vector<T, N>; M] {
        self.0.map(Vector::from)
    }
    pub fn rank(&self) -> usize {
        self.column_space().len()
    }
    pub fn column_space(&self) -> Vec<Vector<T, M>> {
        Vector::span(&self.as_column_vectors())
    }
    pub fn zero() -> Self {
        Self([[<T as From<u8>>::from(0); N]; M])
    }
}

// Square matrices
impl<T: Field, const N: usize> Matrix<T, N, N> {
    pub fn from_diagonal(vals: [T; N]) -> Self {
        Self(array::from_fn(|i| {
            array::from_fn(|j| match i == j {
                true => vals[i],
                false => T::zero(),
            })
        }))
    }
    pub fn identity() -> Self {
        Self(array::from_fn(|i| {
            array::from_fn(|j| match i == j {
                true => <T as From<u8>>::from(1),
                false => <T as From<u8>>::from(0),
            })
        }))
    }
    pub fn from_upper_triangular(vals: &[T]) -> Self {
        assert_eq!(vals.len(), N * (N + 1) / 2, "Invalid values size");
        let mut data = [[T::zero(); N]; N];
        let mut vals_idx = 0;
        for i in 0..N {
            let len = N - i;
            data[i][i..N].copy_from_slice(&vals[vals_idx..vals_idx + len]);
            vals_idx += len;
        }
        Self(data)
    }
    pub fn from_lower_triangular(vals: &[T]) -> Self {
        assert_eq!(vals.len(), N * (N + 1) / 2, "Invalid vals size");
        let mut data = [[T::zero(); N]; N];
        let mut vals_idx = 0;
        for i in 0..N {
            let len = i + 1;
            data[i][..len].copy_from_slice(&vals[vals_idx..vals_idx + len]);
            vals_idx += len;
        }
        Self(data)
    }
    pub fn from_symmetric(vals: &[T]) -> Self {
        assert_eq!(vals.len(), N * (N + 1) / 2, "Invalid vals size");
        let mut data = [[T::zero(); N]; N];
        let mut vals_idx = 0;
        for i in 0..N {
            for j in i..N {
                let val = vals[vals_idx];
                data[i][j] = val; // Add upper
                data[j][i] = val; // Mirror to lower
                vals_idx += 1;
            }
        }
        Self(data)
    }
}

// [[T; N]; M] <-> Matrix<T, N, M>
impl<T: Field, const N: usize, const M: usize> From<[[T; N]; M]> for Matrix<T, N, M> {
    #[inline(always)]
    fn from(value: [[T; N]; M]) -> Self {
        Self(value)
    }
}
impl<T: Field, const N: usize, const M: usize> From<Matrix<T, N, M>> for [[T; N]; M] {
    #[inline(always)]
    fn from(value: Matrix<T, N, M>) -> Self {
        value.0
    }
}
// Indexing
impl<T: Field, const N: usize, const M: usize> Index<(usize, usize)> for Matrix<T, N, M> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1]
    }
}
impl<T: Field, const N: usize, const M: usize> IndexMut<(usize, usize)> for Matrix<T, N, M> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0][index.1]
    }
}
// Matrix x Matrix operations
impl<T: Field, const N: usize, const M: usize> Add for Matrix<T, N, M> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|i| {
            array::from_fn(|j| self[(i, j)] + rhs[(i, j)])
        }))
    }
}
impl<T: Field, const N: usize, const M: usize> Sub for Matrix<T, N, M> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|i| {
            array::from_fn(|j| self[(i, j)] - rhs[(i, j)])
        }))
    }
}
// Vector x Matrix operations
impl<T: Field, const N: usize, const M: usize> Mul<Vector<T, N>> for Matrix<T, N, M> {
    type Output = Vector<T, M>;
    fn mul(self, rhs: Vector<T, N>) -> Self::Output {
        let scalars = rhs.as_array();
        let cols = self.as_column_arrays();
        let combination = cols
            .map(|e| Vector::from(e))
            .into_iter()
            .zip(scalars.into_iter());
        Vector::from_lc(combination)
    }
}
// Scalar x Matrix operations
impl<T: Field, const N: usize, const M: usize> Mul<T> for Matrix<T, N, M> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0.map(|e| (Vector::from(e) * rhs).into()))
    }
}
impl<T: Field, const N: usize, const M: usize> Div<T> for Matrix<T, N, M> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        if rhs == T::zero() {
            panic!("division by zero")
        }
        Self(self.0.map(|e| (Vector::from(e) / rhs).into()))
    }
}
impl<T: Field + Debug, const N: usize, const M: usize> fmt::Display for Matrix<T, N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix([").unwrap();
        self.0
            .iter()
            .for_each(|e| writeln!(f, "  {:?}", e).unwrap());
        write!(f, "])")
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[test]
    fn test_matrix_constructors() {}
    #[test]
    fn test_matrix_linear_sys() {}
}
