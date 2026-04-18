use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::Float;

mod sealed {
    pub trait Sealed {}
}

/// Trait defining basic numeric operations for library types.
pub trait Field:
    Sized
    + Float
    + sealed::Sealed
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
}

macro_rules! impl_field_for {
    ($($target:ty),*) => {
        $(
            impl sealed::Sealed for $target {}
            impl Field for $target {
                const ZERO: Self = 0.0 as Self;
                const ONE: Self = 1.0 as Self;
            }
        )*
    };
}

impl_field_for!(f32, f64);
