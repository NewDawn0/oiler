use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::Float;

pub trait Field:
    Sized
    + Copy
    + Clone
    + PartialOrd
    + PartialEq
    + Add<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + DivAssign
    + MulAssign
    + SubAssign
    + Float // Temporary (used by vec magnitude)
    + From<u8> // Used for comparing
{
}

macro_rules! gen_impls {
    ($target:ty, ($($type:ty),*)) => {
        $(impl $target for $type {})*
    };
}
// gen_impls!(Field, (i8, i16, i32, i64, i128));
// gen_impls!(Field, (u8, u16, u32, u64, u128));
gen_impls!(Field, (f32, f64));
