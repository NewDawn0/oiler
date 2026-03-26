use std::{
    array,
    fmt::{self, Debug},
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
    usize,
};

use crate::linalg::traits::Field;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vector<T: Field, const N: usize>([T; N]);
impl<T: Field + From<T>, const N: usize> Vector<T, N> {
    #[inline(always)]
    pub fn convert<E: Field + From<T>>(self) -> Vector<E, N> {
        Vector(self.0.map(<E as From<T>>::from))
    }
    pub fn zero() -> Self {
        Vector([<T as From<u8>>::from(0); N])
    }
    pub fn unit() -> Self {
        Vector([<T as From<u8>>::from(1); N])
    }
    // From basis and coefficients
    pub fn from_basis<const S: usize>(basis: [Self; S], coeffs: [T; S]) -> Self {
        basis
            .iter()
            .zip(coeffs.iter())
            .map(|(&v, &c)| v * c)
            .fold(Self::from([T::zero(); N]), |acc, next| acc + next)
    }
    // From linear combination pairs
    pub fn from_lc<I: IntoIterator<Item = (Self, T)>>(combination: I) -> Self {
        combination
            .into_iter()
            .map(|(vector, scalar)| vector * scalar)
            .fold(Self::zero(), |acc, next| acc + next)
    }
    #[inline(always)]
    pub fn as_array(self) -> [T; N] {
        self.0
    }
    pub fn is_linearly_dependent(&self, vecs: &[Self]) -> bool {
        if vecs.len() == 0 {
            return false;
        }
        vecs.iter().any(|vec| {
            let angle = self.angle(vec);
            angle == <T as From<u8>>::from(0) || angle == <T as From<u8>>::from(180)
        })
    }
    pub fn angle(&self, other: &Self) -> T {
        let dot = self.dotp(other);
        let denom = self.magnitude() * other.magnitude();
        let y = (denom.powi(2) - dot.powi(2))
            .max(<T as From<u8>>::from(0))
            .sqrt();
        y.atan2(dot).to_degrees()
    }
    pub fn dotp(&self, other: &Self) -> T {
        self.0
            .iter()
            .zip(other.0.iter())
            .fold(T::zero(), |acc, (&e, &o)| acc + e * o)
    }
    pub fn span(vecs: &[Self]) -> Vec<Self> {
        let mut basis: Vec<Self> = vec![];
        vecs.iter().for_each(|v| {
            if !v.is_linearly_dependent(&basis) {
                basis.push(*v);
            }
        });
        basis
    }
    pub fn magnitude(&self) -> T {
        self.0
            .iter()
            .fold(<T as From<u8>>::from(0), |acc, &next| acc + next * next)
            .sqrt()
    }
    pub fn normalize(&self) -> Self {
        self.div(self.magnitude())
    }
}

// [T; N] <-> Vector<T, N>
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

impl<T: Field, const N: usize> Add for Vector<T, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|i| self[i] + rhs[i]))
    }
}
impl<T: Field, const N: usize> Sub for Vector<T, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|i| self[i] - rhs[i]))
    }
}
impl<T: Field, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0.map(|e| e * rhs))
    }
}
impl<T: Field, const N: usize> Div<T> for Vector<T, N> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        if rhs == <T as From<u8>>::from(0) {
            panic!("division by zero")
        }
        Self(self.0.map(|e| e / rhs))
    }
}
impl<T: Field + Debug, const N: usize> fmt::Display for Vector<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector({:?})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < f32::EPSILON
    }

    #[test]
    fn test_vector_constructors() {
        // From array
        let v = Vector::from([1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);

        // Zero and one
        let zero = Vector::<f32, 3>::zero();
        let unit = Vector::<f32, 3>::unit();
        assert_eq!(zero[0], 0.0);
        assert_eq!(zero[1], 0.0);
        assert_eq!(unit[0], 1.0);
        assert_eq!(unit[2], 1.0);

        // from_basis - standard basis
        let e1 = Vector::from([1.0, 0.0]);
        let e2 = Vector::from([0.0, 1.0]);
        let v = Vector::from_basis([e1, e2], [3.0, 4.0]);
        assert_eq!(v, Vector::from([3.0, 4.0]));
        // from_basis - empty basis (zero vector)
        let v_empty: Vector<f32, 2> = Vector::from_basis([], []);
        assert_eq!(v_empty, Vector::zero());
        // from_lc - iterator-based linear combination
        let v = Vector::from_lc([(e1, 2.0), (e2, 5.0)]);
        assert_eq!(v, Vector::from([2.0, 5.0]));
        // as_array - conversion back to array
        let arr = v.as_array();
        assert_eq!(arr, [2.0, 5.0]);
    }

    #[test]
    fn test_vector_dotp() {
        // Orthogonal vectors (dot product = 0)
        let v1 = Vector::from([1.0, 0.0]);
        let v2 = Vector::from([0.0, 1.0]);
        assert_eq!(v1.dotp(&v2), 0.0);

        // Self dot product = magnitude²
        let v3 = Vector::from([3.0, 4.0]);
        assert!(near(v3.dotp(&v3), 25.0));

        // Standard dot product calculation
        let v4 = Vector::from([1.0, 2.0, 3.0]);
        let v5 = Vector::from([4.0, 5.0, 6.0]);
        assert_eq!(v4.dotp(&v5), 32.0); // 1*4 + 2*5 + 3*6 = 4+10+18

        // Commutativity: v·w = w·v
        assert_eq!(v4.dotp(&v5), v5.dotp(&v4));

        // Zero vector
        let zero = Vector::<f32, 3>::zero();
        assert_eq!(v4.dotp(&zero), 0.0);
    }

    #[test]
    fn test_vector_magnitude() {
        // Unit vector
        let unit = Vector::from([1.0, 0.0, 0.0]);
        assert!(near(unit.magnitude(), 1.0));

        // Zero vector
        let zero = Vector::<f32, 3>::zero();
        assert_eq!(zero.magnitude(), 0.0);

        // Pythagorean theorem (3-4-5 triangle)
        let pythagorean = Vector::from([3.0, 4.0]);
        assert!(near(pythagorean.magnitude(), 5.0));

        // Negative components (should square to positive)
        let neg = Vector::from([-3.0, -4.0]);
        assert!(near(neg.magnitude(), 5.0));

        // Higher dimensions
        let v5d = Vector::from([1.0, 2.0, 2.0, 0.0, 0.0]);
        assert!(near(v5d.magnitude(), 3.0)); // sqrt(1+4+4) = sqrt(9) = 3
    }

    #[test]
    fn test_vector_angle() {
        // Orthogonal vectors (90°)
        let v1 = Vector::from([1.0, 0.0]);
        let v2 = Vector::from([0.0, 1.0]);
        assert!(near(v1.angle(&v2), 90.0));

        // Parallel vectors (0°)
        let v3 = Vector::from([1.0, 0.0]);
        let v4 = Vector::from([2.0, 0.0]);
        assert!(near(v3.angle(&v4), 0.0));

        // Opposite vectors (180°)
        let v5 = Vector::from([1.0, 0.0]);
        let v6 = Vector::from([-1.0, 0.0]);
        assert!(near(v5.angle(&v6), 180.0));

        // Self angle (0°)
        let v7 = Vector::from([1.0, 1.0]);
        assert!(near(v7.angle(&v7), 0.0));

        // 45 degree angle
        let v8 = Vector::from([1.0, 0.0]);
        let v9 = Vector::from([1.0, 1.0]);
        assert!(near(v8.angle(&v9), 45.0));
    }

    #[test]
    fn test_vector_normalize() {
        // Normalization produces unit vector
        let v = Vector::from([3.0, 4.0]);
        let n = v.normalize();
        assert!(near(n.magnitude(), 1.0));
        assert!(near(n[0], 0.6)); // 3/5
        assert!(near(n[1], 0.8)); // 4/5

        // Verify parallel to original (angle = 0)
        assert!(near(v.angle(&n), 0.0));

        // Normalizing already normalized vector
        let n2 = n.normalize();
        assert!(near(n2.magnitude(), 1.0));
        assert_eq!(n, n2);

        // Different dimension
        let v3d = Vector::from([1.0, 0.0, 0.0]);
        assert!(near(v3d.normalize().magnitude(), 1.0));
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_vector_normalize_panic_on_zero_vector() {
        Vector([0.0; 6]).normalize();
    }

    #[test]
    fn test_vector_bijective_into_array() {
        // Round trip: array -> vector -> array
        let arr = [1.0, 2.0, 3.0];
        let v = Vector::from(arr);
        let arr2: [f32; 3] = v.into();
        assert_eq!(arr, arr2);

        // as_array method
        let arr3 = v.as_array();
        assert_eq!(arr, arr3);

        // Different dimensions
        let v2d = Vector::from([1.0, 2.0]);
        assert_eq!(v2d.as_array(), [1.0, 2.0]);

        let v5d = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(v5d.as_array(), [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_vector_scalar_operators() {
        let v = Vector::from([2.0, 4.0, 6.0]);

        // Scalar multiplication
        let scaled = v * 2.0;
        assert_eq!(scaled, Vector::from([4.0, 8.0, 12.0]));

        // Scalar multiplication by zero
        let zero_scaled = v * 0.0;
        assert_eq!(zero_scaled, Vector::zero());

        // Scalar multiplication by one (identity)
        let identity = v * 1.0;
        assert_eq!(identity, v);

        // Scalar division
        let divided = v / 2.0;
        assert_eq!(divided, Vector::from([1.0, 2.0, 3.0]));

        // Scalar division by one (identity)
        let div_identity = v / 1.0;
        assert_eq!(div_identity, v);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_vector_panic_on_division_by_zero() {
        let _ = Vector([0.1; 6]) / 0.0;
    }

    #[test]
    fn test_vector_vector_operators() {
        let v1 = Vector::from([1.0, 2.0, 3.0]);
        let v2 = Vector::from([4.0, 5.0, 6.0]);

        // Addition
        let sum = v1 + v2;
        assert_eq!(sum, Vector::from([5.0, 7.0, 9.0]));

        // Subtraction
        let diff = v2 - v1;
        assert_eq!(diff, Vector::from([3.0, 3.0, 3.0]));

        // Adding zero (identity)
        let zero = Vector::<f32, 3>::zero();
        assert_eq!(v1 + zero, v1);

        // Subtracting self (yields zero)
        assert_eq!(v1 - v1, zero);
    }

    #[test]
    fn test_vector_indexing() {
        let mut v = Vector::from([1.0, 2.0, 3.0]);

        // Read access
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);

        // Write access
        v[1] = 5.0;
        assert_eq!(v[1], 5.0);
        assert_eq!(v, Vector::from([1.0, 5.0, 3.0]));

        // Multiple modifications
        v[0] = 10.0;
        v[2] = 20.0;
        assert_eq!(v, Vector::from([10.0, 5.0, 20.0]));
    }

    #[test]
    fn test_vector_display() {
        let v = Vector::from([1.0, 2.0, 3.0]);
        let s = format!("{}", v);
        assert!(s.contains("Vector"));
        assert!(s.contains("1.0"));
        assert!(s.contains("2.0"));
        assert!(s.contains("3.0"));
    }

    #[test]
    fn test_vector_addition_commutativity() {
        // v + w = w + v
        let v = Vector::from([1.0, 2.0, 3.0]);
        let w = Vector::from([4.0, 5.0, 6.0]);
        assert_eq!(v + w, w + v);
    }

    #[test]
    fn test_vector_addition_associativity() {
        // (u + v) + w = u + (v + w)
        let u = Vector::from([1.0, 2.0]);
        let v = Vector::from([3.0, 4.0]);
        let w = Vector::from([5.0, 6.0]);
        assert_eq!((u + v) + w, u + (v + w));
    }

    #[test]
    fn test_vector_scalar_distributivity() {
        // a(v + w) = av + aw
        let v = Vector::from([1.0, 2.0, 3.0]);
        let w = Vector::from([4.0, 5.0, 6.0]);
        let a = 2.0;

        let lhs = (v + w) * a;
        let rhs = v * a + w * a;
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_vector_scalar_addition_distributivity() {
        // (a + b)v = av + bv
        let v = Vector::from([1.0, 2.0, 3.0]);
        let a = 2.0;
        let b = 3.0;

        let lhs = v * (a + b);
        let rhs = v * a + v * b;
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_vector_scalar_multiplication_associativity() {
        // a(bv) = (ab)v
        let v = Vector::from([1.0, 2.0, 3.0]);
        let a = 2.0;
        let b = 3.0;

        let lhs = (v * b) * a;
        let rhs = v * (a * b);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_vector_additive_identity() {
        // v + 0 = v
        let v = Vector::from([1.0, 2.0, 3.0]);
        let zero = Vector::<f32, 3>::zero();
        assert_eq!(v + zero, v);
        assert_eq!(zero + v, v);
    }

    #[test]
    fn test_vector_additive_inverse() {
        // v + (-v) = 0
        let v = Vector::from([1.0, 2.0, 3.0]);
        let zero = Vector::<f32, 3>::zero();
        assert_eq!(v + (v * -1.0), zero);
        assert_eq!((v * -1.0) + v, zero);
    }

    #[test]
    fn test_vector_multiplicative_identity() {
        // 1·v = v
        let v = Vector::from([1.0, 2.0, 3.0]);
        assert_eq!(v * 1.0, v);
    }

    #[test]
    fn test_vector_multiplicative_zero() {
        // 0·v = 0
        let v = Vector::from([1.0, 2.0, 3.0]);
        let zero = Vector::<f32, 3>::zero();
        assert_eq!(v * 0.0, zero);
    }

    #[test]
    fn test_vector_dotp_linearity() {
        // (av)·w = a(v·w)
        let v = Vector::from([1.0, 2.0, 3.0]);
        let w = Vector::from([4.0, 5.0, 6.0]);
        let a = 2.5;

        let lhs = (v * a).dotp(&w);
        let rhs = a * v.dotp(&w);
        assert!(near(lhs, rhs));

        // v·(aw) = a(v·w)
        let lhs2 = v.dotp(&(w * a));
        let rhs2 = a * v.dotp(&w);
        assert!(near(lhs2, rhs2));
    }

    #[test]
    fn test_vector_dotp_distributivity() {
        // u·(v + w) = u·v + u·w
        let u = Vector::from([1.0, 2.0, 3.0]);
        let v = Vector::from([4.0, 5.0, 6.0]);
        let w = Vector::from([7.0, 8.0, 9.0]);

        let lhs = u.dotp(&(v + w));
        let rhs = u.dotp(&v) + u.dotp(&w);
        assert!(near(lhs, rhs));
    }

    #[test]
    fn test_vector_subtraction_as_addition() {
        // v - w = v + (-w)
        let v = Vector::from([1.0, 2.0, 3.0]);
        let w = Vector::from([4.0, 5.0, 6.0]);
        assert_eq!(v - w, v + (w * -1.0));
    }

    #[test]
    fn test_vector_magnitude_scaling() {
        // |av| = |a| * |v|
        let v = Vector::from([3.0, 4.0]);
        let a = 2.0;
        let scaled = v * a;
        assert!(near(scaled.magnitude(), a * v.magnitude()));

        // Negative scalar
        let b = -3.0;
        let scaled_neg = v * b;
        assert!(near(scaled_neg.magnitude(), b.abs() * v.magnitude()));
    }

    #[test]
    fn test_vector_normalize_idempotent() {
        // Normalizing twice gives same result
        let v = Vector::from([3.0, 4.0, 5.0]);
        let n1 = v.normalize();
        let n2 = n1.normalize();
        assert_eq!(n1, n2);
        assert!(near(n2.magnitude(), 1.0));
    }
    #[test]
    fn test_vector_span() {
        let v1 = Vector::<f32, 2>::from([1.0, 2.0]);
        let v2 = Vector::<f32, 2>::from([1.0, 3.0]);
        let v3 = Vector::<f32, 2>::from([-2.0, -4.0]);
        let span1 = Vector::span(&[v1, v2, v3]);
        let span2 = Vector::span(&[v1 * 3.0, v2 * -1.0, v3 * 2.0]);
        assert!(
            span2
                .iter()
                .all(|v| Vector::span(&[v1, v2, v3, *v]) == span1),
            "Vector span equal failed"
        )
    }
}
