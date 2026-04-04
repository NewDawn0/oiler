use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Trait defining basic numeric operations for library types.
pub trait Numeric:
    Sized
    + Clone
    + Copy
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + AddAssign<Self>
    + Sub<Output = Self>
    + SubAssign<Self>
    + Mul<Output = Self>
    + MulAssign<Self>
    + Div<Output = Self>
    + DivAssign<Self>
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// Absolute value.
    fn abs(self) -> Self;
}

macro_rules! impl_numeric_unsigned {
    ($($target:ty),*) => {
        $(
            impl Numeric for $target {
                const ZERO: Self = 0 as Self;
                const ONE: Self = 1 as Self;
                #[inline(always)]
                fn abs(self) -> Self { self }
            }
        )*
    };
}
macro_rules! impl_numeric_signed {
    ($($target:ty),*) => {
        $(
            impl Numeric for $target {
                const ZERO: Self = 0 as Self;
                const ONE: Self = 1 as Self;
                #[inline(always)]
                fn abs(self) -> Self {
                    if self >= Self::ZERO { self } else { -self }
                }
            }
        )*
    };
}

macro_rules! impl_numeric_float {
    ($($target:ty),*) => {
        $(
            impl Numeric for $target {
                const ZERO: Self = 0.0 as Self;
                const ONE: Self = 1.0 as Self;
                #[inline(always)]
                fn abs(self) -> Self { self.abs() }
            }
        )*
    };
}

impl_numeric_unsigned!(u8, u16, u32, u64, u128, usize);
impl_numeric_signed!(i8, i16, i32, i64, i128, isize);
impl_numeric_float!(f32, f64);
