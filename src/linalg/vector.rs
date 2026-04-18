use crate::linalg::prelude::*;
use std::{
    array,
    fmt::{self, Debug},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub struct Vector<T: Field, const N: usize>([T; N]);
impl<T: Field, const N: usize> Vector<T, N> {
    /// Create vector from array
    #[inline]
    pub const fn new(vals: [T; N]) -> Self {
        Self(vals)
    }
    /// Zero vector
    #[inline]
    pub const fn zero() -> Self {
        Self([T::ZERO; N])
    }
    /// Unit vector (all ones)
    #[inline]
    pub const fn unit() -> Self {
        Self([T::ONE; N])
    }
    /// Linear combination of basis and coefficients.
    pub fn from_basis<const S: usize>(basis: [Self; S], coeffs: [T; S]) -> Self {
        let mut out = Self::zero();
        for i in 0..S {
            out += basis[i] * coeffs[i]
        }
        out
    }
    /// Create vector from linear combination iterator.
    pub fn from_lc<I: IntoIterator<Item = (Self, T)>>(combination: I) -> Self {
        combination
            .into_iter()
            .map(|(v, s)| v * s)
            .fold(Self::zero(), |acc, next| acc + next)
    }
    /// Convert underlying Field type.
    #[inline]
    pub fn convert<E: Field + From<T>>(self) -> Vector<E, N> {
        Vector(self.0.map(<E as From<T>>::from))
    }
    /// Return raw array.
    #[inline]
    pub const fn as_array(self) -> [T; N] {
        self.0
    }
    /// Dot product.
    #[inline]
    pub fn dotp(&self, other: &Self) -> T {
        let mut sum = T::ZERO;
        for i in 0..N {
            sum += self.0[i] * other.0[i];
        }
        sum
    }
    /// Squared magnitude.
    #[inline]
    pub fn magnitude_sq(&self) -> T {
        self.dotp(self)
    }
    /// Magnitude.
    pub fn magnitude(&self) -> T {
        self.magnitude_sq().sqrt()
    }
    /// Angle between vectors in degrees.
    pub fn angle(&self, other: &Self) -> T {
        let dot = self.dotp(other);
        let denom = self.magnitude() * other.magnitude();
        let y = (denom.powi(2) - dot.powi(2)).max(T::ZERO).sqrt();
        y.atan2(dot).to_degrees()
    }
    /// Normalize vector.
    #[inline]
    pub fn normalize(&self) -> Self {
        self.div(self.magnitude())
    }

    /// Scaled subtraction: self -= other * factor (starting from index).
    #[inline]
    pub fn sub_assign_scaled_from(&mut self, other: &Self, factor: T, start: usize) {
        for i in start..N {
            self.0[i] -= other.0[i] * factor;
        }
    }
    /// Scaled subtraction: self -= other * factor.
    #[inline(always)]
    pub fn sub_assign_scaled(&mut self, other: &Self, factor: T) {
        self.sub_assign_scaled_from(other, factor, 0)
    }
    /// Check if two vectors are colinear.
    pub fn is_colinear_with(&self, other: &Self) -> bool {
        if self == other
            || self.0[0] == T::ZERO && self.is_zero()
            || other.0[0] == T::ZERO && other.is_zero()
        {
            return true;
        }
        let pivot_idx = (0..N).find(|&i| self.0[i] != T::ZERO || other.0[i] != T::ZERO);
        match pivot_idx {
            None => true,
            Some(idx) => (0..N).all(|n| self.0[n] * other.0[idx] == other.0[n] * self.0[idx]),
        }
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x == &T::ZERO)
    }
    /// Check if vector is linearly dependent on a set of vectors.
    pub fn is_linearly_dependent(&self, vecs: &[Self]) -> bool {
        if self.is_zero() {
            return true;
        }
        // Note: Currently only checks pairwise colinearity.
        // For general linear dependence, a rank-based check is required.
        vecs.iter().any(|e| self.is_colinear_with(e))
    }
}

impl<T: Field, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl<T: Field, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl<T: Field, const N: usize> From<[T; N]> for Vector<T, N> {
    #[inline(always)]
    fn from(value: [T; N]) -> Self {
        Self(value)
    }
}
impl<T: Field, const N: usize> From<Vector<T, N>> for [T; N] {
    #[inline(always)]
    fn from(value: Vector<T, N>) -> Self {
        value.0
    }
}
impl<T: Field + Debug, const N: usize> fmt::Display for Vector<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec({:?})", self.0)
    }
}
impl<T: Field, const N: usize> Add for Vector<T, N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}
impl<T: Field, const N: usize> AddAssign for Vector<T, N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] += rhs.0[i];
        }
    }
}
impl<T: Field, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0.map(|e| e * rhs))
    }
}
impl<T: Field, const N: usize> MulAssign<T> for Vector<T, N> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        for e in self.0.iter_mut() {
            *e *= rhs;
        }
    }
}
impl<T: Field, const N: usize> Sub for Vector<T, N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|n| self.0[n] - rhs.0[n]))
    }
}
impl<T: Field, const N: usize> SubAssign for Vector<T, N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] -= rhs.0[i];
        }
    }
}
impl<T: Field, const N: usize> Div<T> for Vector<T, N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: T) -> Self::Output {
        Self(self.0.map(|e| e / rhs))
    }
}
impl<T: Field, const N: usize> DivAssign<T> for Vector<T, N> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        for e in self.0.iter_mut() {
            *e /= rhs;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_vector_constructors() {
        assert_eq!(
            Vector::<f32, 3>::zero(),
            Vector([0.0, 0.0, 0.0]),
            "zero vec creation"
        );
        assert_eq!(
            Vector::<f32, 3>::unit(),
            Vector([1.0, 1.0, 1.0]),
            "unit vec creation"
        );
        assert_eq!(
            Vector::new([1.0, 2.0]),
            Vector([1.0, 2.0]),
            "new vec creation"
        );
        assert_eq!(
            Vector::<f32, 2>::new([1., 2.]).convert::<f64>(),
            Vector([1., 2.]),
            "vec conversion"
        );
        let basis = [Vector([1.0, 0.0]), Vector([0.0, 1.0])];
        assert_eq!(
            Vector::from_basis(basis, [2.0, 3.0]),
            Vector([2.0, 3.0]),
            "basis combination"
        );
    }
    #[test]
    fn test_vector_arithmetic() {
        let v1 = Vector([1.0, 2.0]);
        let v2 = Vector([3.0, 4.0]);
        assert_eq!(v1 + v2, Vector([4.0, 6.0]), "vec addition");
        assert_eq!(v1 - v2, Vector([-2.0, -2.0]), "vec subtraction");
        assert_eq!(v1 * 2.0, Vector([2.0, 4.0]), "scalar multiplication");
        assert_eq!(v2 / 2.0, Vector([1.0, 2.0]), "scalar division");
        let mut v1 = Vector([1.0, 2.0]);
        v1 += v2;
        assert_eq!(v1, Vector([4.0, 6.0]), "vec add-assign");
        let mut v1 = Vector([1.0, 2.0]);
        v1 -= v2;
        assert_eq!(v1, Vector([1.0, 2.0]), "vec sub-assign");
        let mut v1 = Vector([1.0, 2.0]);
        v1 *= 2.0;
        assert_eq!(v1, Vector([2.0, 4.0]), "scalar mul-assign");
        let mut v1 = Vector([1.0, 2.0]);
        v1 /= 2.0;
        assert_eq!(v1, Vector([1.0, 2.0]), "scalar div-assign");
        assert_eq!(
            Vector([1.0, 2.0]).dotp(&Vector([3.0, 4.0])),
            11.0,
            "dot product"
        );
    }
    #[test]
    fn test_vector_advanced() {
        let v = Vector::<f32, 3>([3.0, 4.0, 0.0]);
        assert_eq!(v.magnitude_sq(), 25.0, "magnitude sq");
        assert_eq!(v.magnitude(), 5.0, "magnitude");
        assert_eq!(v.normalize(), Vector([0.6, 0.8, 0.0]), "normalize");
        assert!(
            Vector([1.0, 0.0]).is_colinear_with(&Vector([2.0, 0.0])),
            "colinearity pos"
        );
        assert!(
            !Vector([1.0, 0.0]).is_colinear_with(&Vector([0.0, 1.0])),
            "colinearity neg"
        );
        assert!(
            Vector([1.0, 2.0]).is_linearly_dependent(&vec![Vector([2.0, 4.0])]),
            "linear dependency"
        );
    }
}
