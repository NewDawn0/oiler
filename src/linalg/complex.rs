use crate::linalg::traits::Numeric;
use num_traits::Float;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Complex number in Cartesian form (re + im * i).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub struct Complex<T: Numeric> {
    pub re: T,
    pub im: T,
}

impl<T: Numeric> Complex<T> {
    /// Create new complex number.
    #[inline(always)]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
    /// Conjugate: a + bi -> a - bi.
    #[inline(always)]
    pub fn conj(self) -> Self
    where
        T: Neg<Output = T>,
    {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
    /// Squared magnitude: a^2 + b^2.
    #[inline(always)]
    pub fn magnitude_sq(self) -> T {
        self.re * self.re + self.im * self.im
    }
    /// Check if number is real (imaginary part is zero).
    #[inline(always)]
    pub fn is_real(&self) -> bool {
        self.im == T::ZERO
    }
    /// Check if number is imaginary (real part is zero).
    #[inline(always)]
    pub fn is_imaginary(&self) -> bool {
        self.re == T::ZERO
    }
}

impl<T: Numeric> From<T> for Complex<T> {
    #[inline(always)]
    fn from(re: T) -> Self {
        Self::new(re, T::ZERO)
    }
}

impl<T: Float + Numeric> Complex<T> {
    /// Magnitude (norm).
    #[inline(always)]
    pub fn magnitude(self) -> T {
        self.magnitude_sq().sqrt()
    }
    /// Convert to polar coordinates (radius, theta).
    #[inline(always)]
    pub fn to_polar(self) -> (T, T) {
        (self.magnitude(), self.im.atan2(self.re))
    }
    /// Create from polar coordinates.
    #[inline(always)]
    pub fn from_polar(r: T, theta: T) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }
}

// Addition
impl<T: Numeric> Add<Complex<T>> for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Complex<T>) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}
impl<T: Numeric> AddAssign<Complex<T>> for Complex<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Complex<T>) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}
impl<T: Numeric> Add<T> for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: T) -> Self {
        Self::new(self.re + rhs, self.im)
    }
}
impl<T: Numeric> AddAssign<T> for Complex<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: T) {
        self.re += rhs;
    }
}

// Subtraction
impl<T: Numeric> Sub<Complex<T>> for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Complex<T>) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}
impl<T: Numeric> SubAssign<Complex<T>> for Complex<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Complex<T>) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}
impl<T: Numeric> Sub<T> for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: T) -> Self {
        Self::new(self.re - rhs, self.im)
    }
}
impl<T: Numeric> SubAssign<T> for Complex<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: T) {
        self.re -= rhs;
    }
}

// Multiplication
impl<T: Numeric> Mul<Complex<T>> for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Complex<T>) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}
impl<T: Numeric> MulAssign<Complex<T>> for Complex<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Complex<T>) {
        *self = *self * rhs;
    }
}
impl<T: Numeric> Mul<T> for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: T) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}
impl<T: Numeric> MulAssign<T> for Complex<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        self.re *= rhs;
        self.im *= rhs;
    }
}

// Division
impl<T: Numeric> Div<Complex<T>> for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Complex<T>) -> Self {
        let d = rhs.magnitude_sq();
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / d,
            (self.im * rhs.re - self.re * rhs.im) / d,
        )
    }
}
impl<T: Numeric> DivAssign<Complex<T>> for Complex<T> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Complex<T>) {
        *self = *self / rhs;
    }
}
impl<T: Numeric> Div<T> for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: T) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}
impl<T: Numeric> DivAssign<T> for Complex<T> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        self.re /= rhs;
        self.im /= rhs;
    }
}

impl<T: Numeric + Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl<T: Numeric> Numeric for Complex<T>
where
    T: Neg<Output = T>,
{
    const ZERO: Self = Self {
        re: T::ZERO,
        im: T::ZERO,
    };
    const ONE: Self = Self {
        re: T::ONE,
        im: T::ZERO,
    };
    #[inline(always)]
    fn abs(self) -> Self {
        // Manhattan norm for general Numeric to avoid Float dependency in trait
        Self::new(self.re.abs() + self.im.abs(), T::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1, 2);
        let b = Complex::new(3, 4);
        assert_eq!(a + b, Complex::new(4, 6), "add");
        assert_eq!(a - b, Complex::new(-2, -2), "sub");
        assert_eq!(a * b, Complex::new(-5, 10), "mul"); // (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        assert_eq!(a * 2, Complex::new(2, 4), "scalar mul");
        let mut c = a;
        c *= b;
        assert_eq!(c, a * b, "mul assign");
    }

    #[test]
    fn test_complex_polar() {
        let c: Complex<f32> = Complex::new(1.0, 1.0);
        let (r, theta) = c.to_polar();
        assert!((r - 2.0f32.sqrt()).abs() < f32::EPSILON, "radius");
        assert!((theta - PI / 4.0).abs() < f32::EPSILON, "theta");
        let c2 = Complex::from_polar(r, theta);
        assert!((c2.re - 1.0).abs() < f32::EPSILON, "polar re");
        assert!((c2.im - 1.0).abs() < f32::EPSILON, "polar im");
    }

    #[test]
    fn test_complex_matrix() {
        use crate::linalg::prelude::*;
        let m = Matrix([
            [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
            [Complex::new(0.0, -1.0), Complex::new(1.0, 0.0)],
        ]);
        let v = Vector([Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)]);
        let res = m * v;
        assert_eq!(
            res,
            Vector([Complex::new(1.0, 1.0), Complex::new(1.0, -1.0)]),
            "mat-vec complex"
        );

        assert_eq!(m.determinant(), Complex::new(0.0, 0.0), "complex det");
        assert_eq!(m.invert(), None, "complex singular inv");

        let m2 = Matrix([
            [Complex::new(2.0, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(0.5, 0.0)],
        ]);
        assert_eq!(m2.determinant(), Complex::new(1.0, 0.0), "complex det 2");
        let inv2 = m2.invert().expect("invertible");
        assert_eq!(inv2.0[0][0], Complex::new(0.5, 0.0), "complex inv 00");
        assert_eq!(inv2.0[1][1], Complex::new(2.0, 0.0), "complex inv 11");
    }
}
