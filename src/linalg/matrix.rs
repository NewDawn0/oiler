use std::{
    array,
    fmt::{self, Debug},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use crate::linalg::prelude::*;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Matrix<T: Numeric, const N: usize, const M: usize>(pub(crate) [[T; N]; M]);

impl<T: Numeric, const N: usize, const M: usize> Matrix<T, N, M> {
    /// Create matrix from array.
    #[inline(always)]
    pub const fn new(vals: [[T; N]; M]) -> Self {
        Self(vals)
    }
    /// Zero matrix.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self([[T::ZERO; N]; M])
    }
    /// Matrix from outer product of two vectors.
    #[inline(always)]
    pub fn from_outer_product(col_v: Vector<T, M>, row_v: Vector<T, N>) -> Self {
        Self(array::from_fn(|i| {
            array::from_fn(|j| col_v.0[i] * row_v.0[j])
        }))
    }
    /// Convert underlying numeric type.
    #[inline(always)]
    pub fn convert<E: Numeric + From<T>>(self) -> Matrix<E, N, M> {
        Matrix(self.0.map(|row| row.map(<E as From<T>>::from)))
    }
    /// Matrix as array of column vectors.
    #[inline(always)]
    pub fn as_column_vectors(self) -> [Vector<T, M>; N] {
        array::from_fn(|j| Vector(array::from_fn(|i| self.0[i][j])))
    }
    /// Matrix as array of row vectors.
    #[inline(always)]
    pub fn as_row_vectors(self) -> [Vector<T, N>; M] {
        self.0.map(Vector::from)
    }
    /// Transpose matrix.
    #[inline(always)]
    pub fn transpose(self) -> Matrix<T, M, N> {
        Matrix(array::from_fn(|i| array::from_fn(|j| self.0[j][i])))
    }

    /// Create matrix from column vectors.
    pub fn from_column_vectors<const S: usize>(cols: [Vector<T, M>; S]) -> Matrix<T, S, M> {
        Matrix(array::from_fn(|i| array::from_fn(|j| cols[j].0[i])))
    }
    /// Create matrix from row vectors.
    pub fn from_row_vectors<const S: usize>(rows: [Vector<T, N>; S]) -> Matrix<T, N, S> {
        Matrix(array::from_fn(|i| rows[i].0))
    }

    /// Row space basis.
    pub fn row_space(&self) -> Box<[Vector<T, N>]> {
        self.cr_factorize().1
    }

    /// Column space basis.
    pub fn column_space(&self) -> Box<[Vector<T, M>]> {
        self.cr_factorize().0
    }

    /// Matrix rank.

    pub fn rank(&self) -> usize {
        let mut rows = self.as_row_vectors();
        let mut r = 0;
        for j in 0..N {
            if r >= M {
                break;
            }
            let (best_row, max) = Self::find_pivot(&rows, j, r);
            if max == T::ZERO {
                continue;
            }
            rows.swap(r, best_row);
            let p_val = rows[r].0[j];
            let row_p = rows[r];
            for i in 0..M {
                if i != r {
                    let factor = rows[i].0[j] / p_val;
                    if factor != T::ZERO {
                        rows[i].sub_assign_scaled_from(&row_p, factor, j);
                    }
                }
            }
            r += 1;
        }
        r
    }

    #[inline(always)]
    fn find_pivot(rows: &[Vector<T, N>], col: usize, start_row: usize) -> (usize, T) {
        let mut best = start_row;
        let mut max = rows[start_row].0[col].abs();
        for i in (start_row + 1)..rows.len() {
            let val = rows[i].0[col].abs();
            if val > max {
                max = val;
                best = i;
            }
        }
        (best, max)
    }

    /// CR Factorization: decomposition into basis columns (C) and RREF non-zero rows (R).
    pub fn cr_factorize(&self) -> (Box<[Vector<T, M>]>, Box<[Vector<T, N>]>) {
        let mut rows = self.as_row_vectors();
        let mut pivot_indices = Vec::new();
        let mut pivot_row = 0;
        for j in 0..N {
            if pivot_row >= M {
                break;
            }
            let (best_row, max) = Self::find_pivot(&rows, j, pivot_row);
            if max == T::ZERO {
                continue;
            }
            rows.swap(pivot_row, best_row);
            let p_val = rows[pivot_row].0[j];
            rows[pivot_row] /= p_val;
            let row_p = rows[pivot_row];
            for i in 0..M {
                if i != pivot_row {
                    let factor = rows[i].0[j];
                    if factor != T::ZERO {
                        rows[i].sub_assign_scaled_from(&row_p, factor, j);
                    }
                }
            }
            pivot_indices.push(j);
            pivot_row += 1;
        }
        let cols = self.as_column_vectors();
        let c = (0..pivot_row)
            .map(|i| cols[pivot_indices[i]])
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let r = (0..pivot_row)
            .map(|i| rows[i])
            .collect::<Vec<_>>()
            .into_boxed_slice();
        (c, r)
    }
    /// Solve linear system Ax = b using Gaussian elimination
    pub fn gaussian_eliminate(&self, vec: &Vector<T, M>) -> Vector<T, N> {
        let mut b = vec.as_array();
        let mut rows = self.as_row_vectors();
        let pivot_lim = M.min(N);
        for i in 0..pivot_lim {
            let (row, _) = Self::find_pivot(&rows, i, i);
            if row != i {
                rows.swap(row, i);
                b.swap(row, i);
            }
            let val = rows[i].0[i];
            if val != T::ZERO {
                let (done, remaining) = rows.split_at_mut(i + 1);
                let row_i = &done[i];
                let (b_done, b_remaining) = b.split_at_mut(i + 1);
                let b_i = b_done[i];
                for (j, row_j) in remaining.iter_mut().enumerate() {
                    let factor_val = row_j.0[i];
                    if factor_val != T::ZERO {
                        let factor = factor_val / val;
                        row_j.sub_assign_scaled_from(row_i, factor, i);
                        b_remaining[j] -= b_i * factor;
                    }
                }
            }
        }
        // Back substitution
        let mut out: Vector<T, N> = Vector::zero();
        for i in (0..pivot_lim).rev() {
            let mut sum = T::ZERO;
            for j in (i + 1)..N {
                sum += rows[i].0[j] * out.0[j];
            }
            let diag = rows[i].0[i];
            if diag != T::ZERO {
                out.0[i] = (b[i] - sum) / diag;
            }
        }
        out
    }
}

impl<T: Numeric, const N: usize> Matrix<T, N, N> {
    /// Identity matrix.
    pub fn identity() -> Self {
        let mut out = Self::zero();
        for i in 0..N {
            out.0[i][i] = T::ONE;
        }
        out
    }
    /// Matrix with given values on the diagonal.
    pub fn from_diagonal(vals: [T; N]) -> Self {
        let mut out = Self::zero();
        for i in 0..N {
            out.0[i][i] = vals[i];
        }
        out
    }
    /// Create upper triangular matrix from slice.
    pub fn from_upper_triangular(vals: &[T]) -> Self {
        let mut data = [[T::ZERO; N]; N];
        let mut idx = 0;
        for i in 0..N {
            let len = N - i;
            data[i][i..N].copy_from_slice(&vals[idx..idx + len]);
            idx += len;
        }
        Self(data)
    }
    /// Create lower triangular matrix from slice.
    pub fn from_lower_triangular(vals: &[T]) -> Self {
        let mut data = [[T::ZERO; N]; N];
        let mut idx = 0;
        for i in 0..N {
            let len = i + 1;
            data[i][..len].copy_from_slice(&vals[idx..idx + len]);
            idx += len;
        }
        Self(data)
    }
    /// Create symmetric matrix from upper triangular slice.
    pub fn from_symmetric(vals: &[T]) -> Self {
        let mut data = [[T::ZERO; N]; N];
        let mut idx = 0;
        for i in 0..N {
            for j in i..N {
                data[i][j] = vals[idx];
                data[j][i] = vals[idx];
                idx += 1;
            }
        }
        Self(data)
    }

    /// Calculate determinant of the matrix.
    pub fn determinant(&self) -> T {
        let mut rows = self.as_row_vectors();
        let mut det = T::ONE;
        let mut swaps = 0;
        for i in 0..N {
            let (row, max) = Self::find_pivot(&rows, i, i);
            if max == T::ZERO {
                return T::ZERO;
            }
            if row != i {
                rows.swap(row, i);
                swaps += 1;
            }
            let val = rows[i].0[i];
            det *= val;
            let row_i = rows[i];
            for j in (i + 1)..N {
                let factor = rows[j].0[i] / val;
                if factor != T::ZERO {
                    rows[j].sub_assign_scaled_from(&row_i, factor, i);
                }
            }
        }
        if swaps % 2 == 1 {
            T::ZERO - det
        } else {
            det
        }
    }

    /// Invert the matrix. Returns None if singular.
    pub fn invert(&self) -> Option<Self> {
        let mut rows = self.as_row_vectors();
        let mut inv_rows = Self::identity().as_row_vectors();
        for i in 0..N {
            let (pivot_row, max) = Self::find_pivot(&rows, i, i);
            if max == T::ZERO {
                return None;
            }
            if pivot_row != i {
                rows.swap(pivot_row, i);
                inv_rows.swap(pivot_row, i);
            }
            let pivot_val = rows[i].0[i];
            rows[i] /= pivot_val;
            inv_rows[i] /= pivot_val;
            let row_i = rows[i];
            let inv_row_i = inv_rows[i];
            for j in 0..N {
                if i != j {
                    let factor = rows[j].0[i];
                    if factor != T::ZERO {
                        rows[j].sub_assign_scaled_from(&row_i, factor, i);
                        inv_rows[j].sub_assign_scaled(&inv_row_i, factor);
                    }
                }
            }
        }
        Some(Self(array::from_fn(|i| inv_rows[i].0)))
    }
}

impl<T: Numeric, const N: usize, const M: usize> Index<(usize, usize)> for Matrix<T, N, M> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1]
    }
}
impl<T: Numeric, const N: usize, const M: usize> IndexMut<(usize, usize)> for Matrix<T, N, M> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0][index.1]
    }
}
impl<T: Numeric, const N: usize, const M: usize> From<[[T; N]; M]> for Matrix<T, N, M> {
    #[inline(always)]
    fn from(value: [[T; N]; M]) -> Self {
        Self(value)
    }
}
impl<T: Numeric, const N: usize, const M: usize> From<Matrix<T, N, M>> for [[T; N]; M] {
    #[inline(always)]
    fn from(value: Matrix<T, N, M>) -> Self {
        value.0
    }
}
impl<T: Numeric + Debug, const N: usize, const M: usize> fmt::Display for Matrix<T, N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix([")?;
        for row in self.0.iter() {
            writeln!(f, "  {:?}", row)?;
        }
        write!(f, "])")
    }
}
impl<T: Numeric, const N: usize, const M: usize> Add for Matrix<T, N, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|i| {
            array::from_fn(|j| self.0[i][j] + rhs.0[i][j])
        }))
    }
}
impl<T: Numeric, const N: usize, const M: usize> AddAssign for Matrix<T, N, M> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self.0[i][j] += rhs.0[i][j];
            }
        }
    }
}
impl<T: Numeric, const N: usize, const M: usize> Mul<T> for Matrix<T, N, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0.map(|v| v.map(|e| e * rhs)))
    }
}
impl<T: Numeric, const N: usize, const M: usize> MulAssign<T> for Matrix<T, N, M> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        for row in self.0.iter_mut() {
            for e in row.iter_mut() {
                *e *= rhs;
            }
        }
    }
}
impl<T: Numeric, const N: usize, const M: usize> Sub for Matrix<T, N, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|i| {
            array::from_fn(|j| self.0[i][j] - rhs.0[i][j])
        }))
    }
}
impl<T: Numeric, const N: usize, const M: usize> SubAssign for Matrix<T, N, M> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self.0[i][j] -= rhs.0[i][j];
            }
        }
    }
}
impl<T: Numeric, const N: usize, const M: usize> Div<T> for Matrix<T, N, M> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: T) -> Self::Output {
        Self(self.0.map(|v| v.map(|e| e / rhs)))
    }
}
impl<T: Numeric, const N: usize, const M: usize> DivAssign<T> for Matrix<T, N, M> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        for row in self.0.iter_mut() {
            for e in row.iter_mut() {
                *e /= rhs;
            }
        }
    }
}
// Vector Matrix product
impl<T: Numeric, const N: usize, const M: usize> Mul<Vector<T, N>> for Matrix<T, N, M> {
    type Output = Vector<T, M>;
    #[inline(always)]
    fn mul(self, rhs: Vector<T, N>) -> Self::Output {
        let mut out = [T::ZERO; M];
        for i in 0..M {
            let mut sum = T::ZERO;
            for j in 0..N {
                sum += self.0[i][j] * rhs.0[j];
            }
            out[i] = sum;
        }
        Vector::from(out)
    }
}
impl<T: Numeric, const N: usize, const M: usize> Mul<Matrix<T, N, M>> for Vector<T, N> {
    type Output = Vector<T, M>;
    #[inline(always)]
    fn mul(self, rhs: Matrix<T, N, M>) -> Self::Output {
        rhs.mul(self)
    }
}

// Matrix multiplication: Matrix<T, P, M> * Matrix<T, N, P> -> Matrix<T, N, M>
impl<T: Numeric, const N: usize, const M: usize, const P: usize> Mul<Matrix<T, N, P>>
    for Matrix<T, P, M>
{
    type Output = Matrix<T, N, M>;
    #[inline(always)]
    fn mul(self, rhs: Matrix<T, N, P>) -> Self::Output {
        let rhs_cols = rhs.as_column_vectors();
        let mut out = [[T::ZERO; N]; M];
        for j in 0..N {
            let col_res = self * rhs_cols[j];
            for i in 0..M {
                out[i][j] = col_res.0[i];
            }
        }
        Matrix(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_matrix_constructors() {
        assert_eq!(
            Matrix::<i32, 2, 2>::zero(),
            Matrix([[0, 0], [0, 0]]),
            "zero mat creation"
        );
        assert_eq!(
            Matrix::<i32, 2, 2>::identity(),
            Matrix([[1, 0], [0, 1]]),
            "identity mat creation"
        );
        assert_eq!(
            Matrix::from_diagonal([1, 2]),
            Matrix([[1, 0], [0, 2]]),
            "diagonal mat creation"
        );
        assert_eq!(
            Matrix::from_upper_triangular(&[1, 2, 3]),
            Matrix([[1, 2], [0, 3]]),
            "upper tri creation"
        );
        assert_eq!(
            Matrix::from_lower_triangular(&[1, 2, 3]),
            Matrix([[1, 0], [2, 3]]),
            "lower tri creation"
        );
        assert_eq!(
            Matrix::from_symmetric(&[1, 2, 3]),
            Matrix([[1, 2], [2, 3]]),
            "symmetric creation"
        );
        assert_eq!(
            Matrix::from_outer_product(Vector([1, 2]), Vector([3, 4])),
            Matrix([[3, 4], [6, 8]]),
            "outer product creation"
        );
    }
    #[test]
    fn test_matrix_arithmetic() {
        let mut m1 = Matrix([[1, 2], [3, 4]]);
        let m2 = Matrix([[5, 6], [7, 8]]);
        assert_eq!(m1 + m2, Matrix([[6, 8], [10, 12]]), "mat addition");
        assert_eq!(m1 - m2, Matrix([[-4, -4], [-4, -4]]), "mat subtraction");
        assert_eq!(m1 * 2, Matrix([[2, 4], [6, 8]]), "scalar multiplication");
        m1 += m2;
        assert_eq!(m1, Matrix([[6, 8], [10, 12]]), "mat add-assign");
        m1 *= 2;
        assert_eq!(m1, Matrix([[12, 16], [20, 24]]), "scalar mul-assign");
        let res: Vector<i32, 2> = Matrix([[1, 2], [3, 4]]) * Vector([1, 2]);
        assert_eq!(res, Vector([5, 11]), "mat-vec multiplication");
    }
    #[test]
    fn test_matrix_advanced() {
        let m = Matrix([[1, 2], [3, 4]]);
        assert_eq!(m.transpose(), Matrix([[1, 3], [2, 4]]), "transpose");
        assert_eq!(
            m.as_row_vectors(),
            [Vector([1, 2]), Vector([3, 4])],
            "as row vectors"
        );
        assert_eq!(
            m.as_column_vectors(),
            [Vector([1, 3]), Vector([2, 4])],
            "as col vectors"
        );
        let a = Matrix::from([
            [1.0, 2.0, 0.0, 3.0],
            [2.0, 4.0, 1.0, 4.0],
            [3.0, 6.0, 2.0, 5.0],
        ]);
        let (c, r) = a.cr_factorize();
        let columns = [Vector::from([1.0, 2.0, 3.0]), Vector::from([0.0, 1.0, 2.0])];
        let rows = [
            Vector::from([1.0, 2.0, 0.0, 3.0]),
            Vector::from([0.0, 0.0, 1.0, -2.0]),
        ];
        assert_eq!(c.len(), 2, "CR column space dim");
        assert_eq!(r.len(), 2, "CR row space dim");
        assert_eq!(&*c, &columns, "Columns failed");
        assert_eq!(&*r, &rows, "Rows failed");

        let m2 = Matrix([[2.0, 1.0], [1.0, 1.0]]);
        let m2inv = Matrix([[1.0, -1.0], [-1.0, 2.0]]);
        assert_eq!(m2inv, m2.invert().unwrap(), "Inversion failed");

        let singular = Matrix([[1.0, 2.0], [2.0, 4.0]]);
        assert_eq!(singular.determinant(), 0.0, "singular det");
        assert_eq!(singular.invert(), None, "singular inv");
    }
    #[test]
    fn test_matrix_linear_sys() {
        let m = Matrix([[1.0, 2.0], [3.0, 4.0]]);
        let b = Vector([5.0, 11.0]);
        assert_eq!(
            m.gaussian_eliminate(&b),
            Vector([1.0, 2.0]),
            "gauss elimination"
        );
    }
}
