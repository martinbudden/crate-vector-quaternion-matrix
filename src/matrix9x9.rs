use core::fmt;
use core::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, RangeFull,
    RangeInclusive, Sub, SubAssign,
};
use core::slice::{ChunksExact, ChunksExactMut, Iter, IterMut};
use num_traits::{ConstOne, ConstZero, MulAdd, MulAddAssign, One, Zero, float::FloatCore};

use crate::{MathConstants, Matrix2x2, Matrix3x3, Matrix4x4, Matrix9x9Math, Vector3};

/// 9x9 matrix of `f32` values<br>
pub type Matrix9x9f32 = Matrix9x9<f32>;
/// 9x9 matrix of `f64` values<br><br>
pub type Matrix9x9f64 = Matrix9x9<f64>;

// **** Define ****

/// `Matrix9x9<T>`: 9x9 Matrix of type `T`.<br>
/// Provided to support Kalman filter matrix math and so not all functions are provided.<br>
/// In particular matrix by matrix multiply, determinant, adjugate, and inverse are not provided.<br>
/// Functions to extract and utilize 3x3 sub-matrices are provided.<br>
/// Aliases `Matrix9x9f32` and `Matrix9x9f64` are provided.<br>
/// Internal implementation is a flattened 9x9 matrix: an array of 81 elements stored in column-major order.
/// That is the element `m[row][col]` is at array position `[col * 9 + row]`, , so element `m01` is at `a[1]` and element `m12` is at `a[11]`.<br><br>
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Matrix9x9<T> {
    // Flattened 9x9 matrix: 81 elements in column-major order
    pub(crate) a: [T; 81],
}

impl<T> Default for Matrix9x9<T>
where
    T: Copy + Zero,
{
    fn default() -> Self {
        Self { a: [T::zero(); 81] }
    }
}

impl<T> fmt::Debug for Matrix9x9<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Start the struct block wrapper
        writeln!(f, "Matrix9x9 [")?;

        // Loop over rows using the Deref slice chunking behavior we added earlier
        for row in self.chunks_exact(9) {
            // Print 4 spaces of indentation for clean alignment
            write!(f, "    ")?;

            // Format the row elements neatly as a standard array slice
            fmt::Debug::fmt(row, f)?;
            writeln!(f, ",")?;
        }

        // Close the struct block wrapper
        write!(f, "]")
    }
}

/// Constants to index matrix elements.
#[allow(missing_docs)]
impl<T> Matrix9x9<T> {
    pub const SIZE: usize = 81;
    pub const ROW_COUNT: usize = 9;
    pub const COL_COUNT: usize = 9;
    // Column 1
    pub const M11: usize = 0;
    pub const M21: usize = 1;
    pub const M31: usize = 2;
    pub const M41: usize = 3;
    pub const M51: usize = 4;
    pub const M61: usize = 5;
    pub const M71: usize = 6;
    pub const M81: usize = 7;
    pub const M91: usize = 8;
    // Column 2
    pub const M12: usize = 9;
    pub const M22: usize = 10;
    pub const M32: usize = 11;
    pub const M42: usize = 12;
    pub const M52: usize = 13;
    pub const M62: usize = 14;
    pub const M72: usize = 15;
    pub const M82: usize = 16;
    pub const M92: usize = 17;
    // Column 3
    pub const M13: usize = 18;
    pub const M23: usize = 19;
    pub const M33: usize = 20;
    pub const M43: usize = 21;
    pub const M53: usize = 22;
    pub const M63: usize = 23;
    pub const M73: usize = 24;
    pub const M83: usize = 25;
    pub const M93: usize = 26;
    // Column 4
    pub const M14: usize = 27;
    pub const M24: usize = 28;
    pub const M34: usize = 29;
    pub const M44: usize = 30;
    pub const M54: usize = 31;
    pub const M64: usize = 32;
    pub const M74: usize = 33;
    pub const M84: usize = 34;
    pub const M94: usize = 35;
    // Column 5
    pub const M15: usize = 36;
    pub const M25: usize = 37;
    pub const M35: usize = 38;
    pub const M45: usize = 39;
    pub const M55: usize = 40;
    pub const M65: usize = 41;
    pub const M75: usize = 42;
    pub const M85: usize = 43;
    pub const M95: usize = 44;
    // Column 6
    pub const M16: usize = 45;
    pub const M26: usize = 46;
    pub const M36: usize = 47;
    pub const M46: usize = 48;
    pub const M56: usize = 49;
    pub const M66: usize = 50;
    pub const M76: usize = 51;
    pub const M86: usize = 52;
    pub const M96: usize = 53;
    // Column 7
    pub const M17: usize = 54;
    pub const M27: usize = 55;
    pub const M37: usize = 56;
    pub const M47: usize = 57;
    pub const M57: usize = 58;
    pub const M67: usize = 59;
    pub const M77: usize = 60;
    pub const M87: usize = 61;
    pub const M97: usize = 62;
    // Column 8
    pub const M18: usize = 63;
    pub const M28: usize = 64;
    pub const M38: usize = 65;
    pub const M48: usize = 66;
    pub const M58: usize = 67;
    pub const M68: usize = 68;
    pub const M78: usize = 69;
    pub const M88: usize = 70;
    pub const M98: usize = 71;
    // Column 9
    pub const M19: usize = 72;
    pub const M29: usize = 73;
    pub const M39: usize = 74;
    pub const M49: usize = 75;
    pub const M59: usize = 76;
    pub const M69: usize = 77;
    pub const M79: usize = 78;
    pub const M89: usize = 79;
    pub const M99: usize = 80;
}

// **** New ****

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Create a matrix.
    #[rustfmt::skip]
    #[inline]
    pub const fn new(a: [T; 81]) -> Self {
        Self {a: [
            a[0], a[9],  a[18], a[27], a[36], a[45], a[54], a[63], a[72],
            a[1], a[10], a[19], a[28], a[37], a[46], a[55], a[64], a[73],
            a[2], a[11], a[20], a[29], a[38], a[47], a[56], a[65], a[74],
            a[3], a[12], a[21], a[30], a[39], a[48], a[57], a[66], a[75],
            a[4], a[13], a[22], a[31], a[40], a[49], a[58], a[67], a[76],
            a[5], a[14], a[23], a[32], a[41], a[50], a[59], a[68], a[77],
            a[6], a[15], a[24], a[33], a[42], a[51], a[60], a[69], a[78],
            a[7], a[16], a[25], a[34], a[43], a[52], a[61], a[70], a[79],
            a[8], a[17], a[26], a[35], a[44], a[53], a[62], a[71], a[80],
        ] }
    }
}

// **** Other constructors ****

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Create a matrix with all its elements set to a single value.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_element(2.0);
    /// assert_eq!(2.0, m[9]);
    /// ```
    pub fn from_element(value: T) -> Self {
        Self { a: [value; 81] }
    }

    /// Matrix from 1D row array.
    ///```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_row_array([
    ///    1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
    ///   10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    ///   19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
    ///   28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
    ///   37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0,
    ///   46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0,
    ///   55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
    ///   64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0,
    ///   73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0,
    /// ]);
    /// let mut n = Matrix9x9f32::from_column_array([
    ///    1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
    ///   10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    ///   19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
    ///   28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
    ///   37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0,
    ///   46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0,
    ///   55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
    ///   64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0,
    ///   73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0,
    /// ]);
    /// assert_eq!(m, n.transpose());
    ///```
    pub fn from_row_array(a: [T; 81]) -> Self {
        // Initialize the output array with the first element of the input
        let mut column_major = [a[0]; 81];
        for r in 0..9 {
            for c in 0..9 {
                let row_idx = r * 9 + c;
                let col_idx = c * 9 + r;
                column_major[col_idx] = a[row_idx];
            }
        }
        Self { a: column_major }
    }

    /// Matrix from 1D column array.
    #[inline]
    pub const fn from_column_array(a: [T; 81]) -> Self {
        Self { a }
    }

    /// Try to create a matrix from a slice.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let valid_data = [2.0; 81];
    /// let invalid_data = [2.0; 3];
    /// let Some(m) = Matrix9x9f32::try_from_slice(&valid_data) else {
    ///     panic!("Expected Some(Matrix9x9), but got None");
    /// };
    /// assert_eq!(2.0, m[0]);
    /// let None = Matrix9x9f32::try_from_slice(&invalid_data) else {
    ///     panic!("Expected None for invalid data, but got Some");
    /// };
    /// ```
    pub fn try_from_slice(slice: &[T]) -> Option<Self> {
        if slice.len() != 81 {
            return None;
        }
        let mut a = [slice[0]; 81];
        a.copy_from_slice(&slice[0..81]);
        Some(Self { a })
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy + ConstZero,
{
    /// Create a matrix with the diagonal set to a single value.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_diagonal_element(2.0);
    /// assert_eq!(m, Matrix9x9f32::new([ 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_element(value: T) -> Self {
        Self { a: [
            value,   T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, value,   T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, value,   T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, value,   T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, value,   T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, value,   T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, value,   T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, value,   T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, value,
        ] }
    }
    /// Create a matrix with the diagonal set to an array.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_diagonal_array([ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]);
    /// assert_eq!(m, Matrix9x9f32::new([ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_array(a: [T;9]) -> Self {
        Self { a: [
            a[0],    T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, a[1],    T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, a[2],    T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, a[3],    T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, a[4],    T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, a[5],    T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, a[6],    T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, a[7],    T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, a[8],
        ] }
    }
}

// **** Zero ****

impl<T> Zero for Matrix9x9<T>
where
    T: Copy + Zero + PartialEq + Matrix9x9Math,
{
    /// Zero matrix.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// # use num_traits::Zero;
    /// let z = Matrix9x9f32::zero();
    /// assert!(z.is_zero());
    /// ```
    #[inline]
    fn zero() -> Self {
        Self { a: [T::zero(); 81] }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<T> ConstZero for Matrix9x9<T>
where
    T: Copy + ConstZero + PartialEq + Matrix9x9Math,
{
    /// Const zero matrix.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// # use num_traits::{zero,Zero,ConstZero};
    /// let m = Matrix9x9f32::ZERO;
    /// assert!(m.is_zero());
    /// ```
    const ZERO: Self = Self { a: [T::ZERO; 81] };
}

impl<T> Matrix9x9<T>
where
    T: Copy + FloatCore,
{
    /// Return true if matrix is near zero.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// # use num_traits::Zero;
    /// let z = Matrix9x9f32::zero();
    /// assert!(z.is_near_zero(1e-5));
    /// ```
    pub fn is_near_zero(self, epsilon: T) -> bool {
        for a in &self.a {
            if a.abs() > epsilon {
                return false;
            }
        }
        true
    }
}

// **** One ****

impl<T> One for Matrix9x9<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix9x9Math,
{
    /// Identity matrix.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// # use num_traits::One;
    /// let i = Matrix9x9f32::one();
    ///
    /// assert!(i.is_one());
    /// ```
    #[inline]
    fn one() -> Self {
        Self::ONE
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

impl<T> ConstOne for Matrix9x9<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix9x9Math,
{
    /// Const identity matrix.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// # use num_traits::{ConstOne, One};
    /// let i = Matrix9x9f32::ONE;
    ///
    /// assert!(i.is_one());
    /// ```
    #[rustfmt::skip]
    const ONE: Self = Self {
        a: [
            T::ONE,  T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ONE,  T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ONE,  T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ONE,  T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ONE,  T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ONE,  T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ONE,  T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ONE,  T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ONE,
        ],
    };
}

impl<T> Matrix9x9<T>
where
    T: Copy + Zero + One,
{
    /// Identity matrix.
    /// Alias for `one()` that does not require `num_traits::One`.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let i = Matrix9x9f32::identity();
    /// ```
    #[rustfmt::skip]
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        let mut m = Self { a: [T::zero(); 81] };
        for ii in 0..=8 {
            m.a[ii * 9 + ii] = T::one();
        }
        m
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy + FloatCore,
    Matrix9x9<T>: One + Sub<Output = Matrix9x9<T>>,
{
    /// Return true if matrix is near identity.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// # use num_traits::One;
    /// let i = Matrix9x9f32::one();
    /// assert!(i.is_near_identity(1e-5));
    /// ```
    pub fn is_near_identity(self, epsilon: T) -> bool {
        (self - Matrix9x9::<T>::one()).is_near_zero(epsilon)
    }
}

// **** Neg ****

impl<T> Neg for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Negate matrix.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let mut m = Matrix9x9f32::from_element(2.0);
    /// m = - m;
    /// ```
    #[inline]
    fn neg(self) -> Self {
        T::m9x9_neg(self)
    }
}

// **** Add ****

impl<T> Add for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Add two matrices.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_element(2.0);
    /// let n = Matrix9x9f32::from_element(3.0);
    /// let r = m + n;
    ///
    ///
    /// # use num_traits::Zero;
    ///
    /// let z = Matrix9x9f32::zero();
    /// let r2 = m + z;
    ///
    /// assert_eq!(r2, m);
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        T::m9x9_add(self, other)
    }
}

// **** AddAssign ****

impl<T> AddAssign for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// Add one matrix to another.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let mut m = Matrix9x9f32::from_element(2.0);
    /// let n = Matrix9x9f32::from_element(3.0);
    /// m += n;
    ///
    /// assert_eq!(m, Matrix9x9f32::from_element(5.0));
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

impl<T> MulAdd<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Multiply matrix by constant and add another matrix.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// # use num_traits::MulAdd;
    /// let m = Matrix9x9f32::from_element(2.0);
    /// let n = Matrix9x9f32::from_element(3.0);
    /// let k = 5.0;
    /// let r = m.mul_add(k, n);
    ///
    /// assert_eq!(r, Matrix9x9f32::from_element(13.0));
    /// ```
    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::m9x9_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

impl<T> MulAddAssign<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// Multiply matrix by constant and add another matrix in place.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// # use num_traits::MulAddAssign;
    /// let mut m = Matrix9x9f32::from_element(2.0);
    /// let n = Matrix9x9f32::from_element(3.0);
    /// let k = 5.0;
    /// m.mul_add_assign(k, n);
    ///
    /// assert_eq!(m, Matrix9x9f32::from_element(13.0));
    /// ```
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

impl<T> Sub for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Subtract two matrices.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_element(2.0);
    /// let n = Matrix9x9f32::from_element(3.0);
    /// let r = m - n;
    ///
    /// assert_eq!(r, Matrix9x9f32::from_element(-1.0));
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

impl<T> SubAssign for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// Subtract one matrix from another.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let mut m = Matrix9x9f32::from_element(2.0);
    /// let n = Matrix9x9f32::from_element(3.0);
    /// m -= n;
    ///
    /// assert_eq!(m, Matrix9x9f32::from_element(-1.0));
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Pre-multiply ****

impl Mul<Matrix9x9<f32>> for f32 {
    type Output = Matrix9x9<f32>;

    /// Pre-multiply a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_element(2.0);
    /// let r = 2.0 * m;
    ///
    /// assert_eq!(r, Matrix9x9f32::from_element(4.0));
    /// ```
    #[inline]
    fn mul(self, other: Matrix9x9<f32>) -> Matrix9x9<f32> {
        f32::m9x9_mul_scalar(other, self)
    }
}

impl Mul<Matrix9x9<f64>> for f64 {
    type Output = Matrix9x9<f64>;

    /// Pre-multiply a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_element(2.0);
    /// let r = 2.0 * m;
    ///
    /// assert_eq!(r, Matrix9x9f32::from_element(4.0));
    /// ```
    #[inline]
    fn mul(self, other: Matrix9x9<f64>) -> Matrix9x9<f64> {
        f64::m9x9_mul_scalar(other, self)
    }
}

// **** Mul Scalar ****

impl<T> Mul<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Multiply a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_element(2.0);
    /// let r = m * 2.0;
    ///
    /// assert_eq!(r, Matrix9x9f32::from_element(4.0));
    /// ```
    #[inline]
    fn mul(self, other: T) -> Self {
        T::m9x9_mul_scalar(self, other)
    }
}

// **** MulAssign ****

impl<T> MulAssign<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// In-place multiply a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let mut m = Matrix9x9f32::from_element(2.0);
    /// m *= 2.0;
    ///
    /// assert_eq!(m, Matrix9x9f32::from_element(4.0));
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

// **** Mul ****

impl<T> Mul<Matrix9x9<T>> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Multiply two matrices.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_element(2.0);
    /// let n = Matrix9x9f32::from_element(3.0);
    /// let r = m * n;
    ///
    /// assert_eq!(r, Matrix9x9f32::from_element(54.0));
    ///
    /// # use num_traits::{One,one};
    ///
    /// let i = Matrix9x9f32::one();
    /// let r2 = m * i;
    ///
    /// assert_eq!(r2, m);
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        T::m9x9_mul(self, other)
    }
}

// **** Div ****

impl<T> Div<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Divide a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_element(2.0);
    /// let r = m / 2.0;
    ///
    /// assert_eq!(r, Matrix9x9f32::from_element(1.0));
    /// ```
    #[inline]
    fn div(self, other: T) -> Self {
        T::m9x9_div_scalar(self, other)
    }
}

// **** DivAssign ****

impl<T> DivAssign<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// In-place divide a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let mut m = Matrix9x9f32::from_element(2.0);
    /// m /= 2.0;
    ///
    /// assert_eq!(m, Matrix9x9f32::from_element(1.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, other: T) {
        *self = *self / other;
    }
}

// **** AsRef ****

impl<T> AsRef<[T; 81]> for Matrix9x9<T> {
    /// Immutable reference to the raw array.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let mut m = Matrix9x9f32::from_element(2.0);
    /// let a: &[f32; 81] = m.as_ref();
    /// assert_eq!(2.0, a[Matrix9x9f32::M21]);
    /// ```
    #[inline]
    fn as_ref(&self) -> &[T; 81] {
        &self.a
    }
}

impl<T> AsMut<[T; 81]> for Matrix9x9<T> {
    /// Mutable reference to the raw array.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let mut m = Matrix9x9f32::from_element(2.0);
    /// let a: &mut [f32; 81] = m.as_mut();
    /// a[4] = 7.0;
    /// assert_eq!(7.0, m[4]);
    /// ```
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 81] {
        &mut self.a
    }
}

// **** Deref ****

impl<T> Deref for Matrix9x9<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        &self.a
    }
}

impl<T> DerefMut for Matrix9x9<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.a
    }
}

// **** Index ****

impl<T> Index<usize> for Matrix9x9<T> {
    type Output = T;

    /// Access matrix element by index.
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

impl<T> Index<Range<usize>> for Matrix9x9<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: Range<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<RangeFull> for Matrix9x9<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _index: RangeFull) -> &[T] {
        &self.a
    }
}

impl<T> Index<RangeInclusive<usize>> for Matrix9x9<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeInclusive<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<(usize, usize)> for Matrix9x9<T> {
    type Output = T;

    /// Access matrix element by ordered pair (row, column).
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < 9 && col < 9, "Matrix index out of bounds: row={row}, col={col}");
        &self.a[col * 9 + row]
    }
}

// **** IndexMut ****

impl<T> IndexMut<usize> for Matrix9x9<T> {
    /// Set matrix element by index.
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.a[index]
    }
}

impl<T> IndexMut<Range<usize>> for Matrix9x9<T> {
    #[inline]
    fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<RangeFull> for Matrix9x9<T> {
    #[inline]
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        &mut self.a
    }
}

impl<T> IndexMut<RangeInclusive<usize>> for Matrix9x9<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeInclusive<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix9x9<T> {
    /// Set matrix element by ordered pair (row, column).
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(row < 9 && col < 9, "Matrix index out of bounds: row={row}, col={col}");
        &mut self.a[col * 9 + row]
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Returns a row as a Vector3 3-tuple.
    #[inline]
    pub fn row_tuple_vector(&self, row_index: usize) -> (Vector3<T>, Vector3<T>, Vector3<T>) {
        let r = row_index;
        (
            Vector3 { x: self.a[r], y: self.a[r + 9], z: self.a[r + 18] },
            Vector3 { x: self.a[r + 27], y: self.a[r + 36], z: self.a[r + 45] },
            Vector3 { x: self.a[r + 54], y: self.a[r + 63], z: self.a[r + 72] },
        )
    }

    /// Returns a column as a Vector3 3-tuple.
    #[inline]
    pub fn column_tuple_vector(&self, col_index: usize) -> (Vector3<T>, Vector3<T>, Vector3<T>) {
        let offset = col_index * 9;
        (
            Vector3 { x: self.a[offset], y: self.a[offset + 1], z: self.a[offset + 2] },
            Vector3 { x: self.a[offset + 3], y: self.a[offset + 4], z: self.a[offset + 5] },
            Vector3 { x: self.a[offset + 6], y: self.a[offset + 7], z: self.a[offset + 8] },
        )
    }

    /// Return matrix diagonal as an array.
    /// ```
    /// # use vqm::Matrix9x9f32;
    ///
    /// let m = Matrix9x9f32::from_element(2.0);
    /// let a = m.diagonal_as_array();
    ///
    /// assert_eq!(a, [ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ]);
    /// ```
    pub fn diagonal_as_array(self) -> [T; 9] {
        [
            self.a[Self::M11],
            self.a[Self::M22],
            self.a[Self::M33],
            self.a[Self::M44],
            self.a[Self::M55],
            self.a[Self::M66],
            self.a[Self::M77],
            self.a[Self::M88],
            self.a[Self::M99],
        ]
    }
}

// **** abs ****

impl<T> Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// Return a copy of the matrix with all elements set to their absolute values.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_element(-2.0);
    /// let n = m.abs();
    ///
    /// assert_eq!(n, Matrix9x9f32::from_element(2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        T::m9x9_abs(self)
    }

    /// Set all elements of the matrix to their absolute values.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let mut m = Matrix9x9f32::from_element(-2.0);
    /// m.abs_in_place();
    ///
    /// assert_eq!(m, Matrix9x9f32::from_element(2.0));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = T::m9x9_abs(*self);
        self
    }
}

// **** clamp ****

impl<T> Matrix9x9<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all elements clamped to the specified range.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::from_element(-2.0);
    ///
    /// let n = m.clamp(7.0, 17.0);
    ///
    /// assert_eq!(n, Matrix9x9f32::from_element(7.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn clamp(self, min: T, max: T) -> Self {
        let mut a = self.a;
        for it in &mut a {
            *it = it.clamp(min, max);
        }
        Self { a }
    }

    /// Clamp all elements of the matrix to the specified range.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let mut m = Matrix9x9f32::from_element(-2.0);
    /// m.clamp_in_place(7.0, 17.0);
    ///
    /// assert_eq!(m, Matrix9x9f32::from_element(7.0));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Return the transpose of this matrix.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let mut m = Matrix9x9f32::from_element(2.0);
    /// let n = m.transpose();
    ///
    /// assert_eq!(n, Matrix9x9f32::from_element(2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transpose(&mut self) -> Self {
        // In-place transpose of the 9x9 submatrix
        // LLVM easily vectorizes this because the bounds and strides are power-of-two friendly
        for ii in 0..8 {
            for jj in (ii + 1)..8 {
                let idx_a = ii * 9 + jj;
                let idx_b = jj * 9 + ii;
                self.a.swap(idx_a, idx_b);
            }
        }

        // In-place swap of the 9th row and 9th column tail elements
        // (Excluding the very last corner element matrix[80] which stays put)
        for ii in 0..8 {
            let row_tail = ii * 9 + 8; // Element in the 9th column
            let col_tail = 8 * 9 + ii; // Element in the 9th row
            self.a.swap(row_tail, col_tail);
        }
        *self
    }

    /// Transpose matrix, in-place.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let mut m = Matrix9x9f32::from_element(2.0);
    /// m.transpose_in_place();
    ///
    /// assert_eq!(m, Matrix9x9f32::from_element(2.0));
    /// ```
    #[inline]
    pub fn transpose_in_place(&mut self) -> &mut Self {
        *self = self.transpose();
        self
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// Return trace of matrix.
    #[inline]
    pub fn trace(self) -> T {
        T::m9x9_trace(self)
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy + Zero + One + Matrix9x9Math + MathConstants + PartialOrd + FloatCore,
{
    /// Return the sum of all elements of the matrix.
    #[inline]
    pub fn sum(self) -> T {
        T::m9x9_sum(self)
    }

    /// Return the mean of all elements of the matrix.
    #[inline]
    pub fn mean(self) -> T {
        T::m9x9_mean(self)
    }

    /// Return the product of all elements of the matrix.
    #[inline]
    pub fn product(self) -> T {
        T::m9x9_product(self)
    }
}

// **** Symmetry ****

impl<T> Matrix9x9<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Enforces strict mathematical symmetry on the matrix in-place.
    /// Used so that rounding errors do not erode the symmetry of the matrix.
    /// Formula: `M = (M + Mᵀ) / 2`.
    pub fn enforce_symmetry(&mut self) {
        let half = T::one() / (T::one() + T::one());

        // --- Column 1 Cross-terms ---
        self.a[Self::M12] = (self.a[Self::M12] + self.a[Self::M21]) * half;
        self.a[Self::M21] = self.a[Self::M12];

        // --- Column 2 Cross-terms ---
        self.a[18] = (self.a[18] + self.a[2]) * half;
        self.a[2] = self.a[18];
        self.a[19] = (self.a[19] + self.a[11]) * half;
        self.a[11] = self.a[19];

        // --- Column 3 Cross-terms ---
        self.a[27] = (self.a[27] + self.a[3]) * half;
        self.a[3] = self.a[27];
        self.a[28] = (self.a[28] + self.a[12]) * half;
        self.a[12] = self.a[28];
        self.a[29] = (self.a[29] + self.a[21]) * half;
        self.a[21] = self.a[29];

        // --- Column 4 Cross-terms ---
        self.a[36] = (self.a[36] + self.a[4]) * half;
        self.a[4] = self.a[36];
        self.a[37] = (self.a[37] + self.a[13]) * half;
        self.a[13] = self.a[37];
        self.a[38] = (self.a[38] + self.a[22]) * half;
        self.a[22] = self.a[38];
        self.a[39] = (self.a[39] + self.a[31]) * half;
        self.a[31] = self.a[39];

        // --- Column 5 Cross-terms ---
        self.a[45] = (self.a[45] + self.a[5]) * half;
        self.a[5] = self.a[45];
        self.a[46] = (self.a[46] + self.a[14]) * half;
        self.a[14] = self.a[46];
        self.a[47] = (self.a[47] + self.a[23]) * half;
        self.a[23] = self.a[47];
        self.a[48] = (self.a[48] + self.a[32]) * half;
        self.a[32] = self.a[48];
        self.a[49] = (self.a[49] + self.a[41]) * half;
        self.a[41] = self.a[49];

        // --- Column 6 Cross-terms ---
        self.a[54] = (self.a[54] + self.a[6]) * half;
        self.a[6] = self.a[54];
        self.a[55] = (self.a[55] + self.a[15]) * half;
        self.a[15] = self.a[55];
        self.a[56] = (self.a[56] + self.a[24]) * half;
        self.a[24] = self.a[56];
        self.a[57] = (self.a[57] + self.a[33]) * half;
        self.a[33] = self.a[57];
        self.a[58] = (self.a[58] + self.a[42]) * half;
        self.a[42] = self.a[58];
        self.a[59] = (self.a[59] + self.a[51]) * half;
        self.a[51] = self.a[59];

        // --- Column 7 Cross-terms ---
        self.a[63] = (self.a[63] + self.a[7]) * half;
        self.a[7] = self.a[63];
        self.a[64] = (self.a[64] + self.a[16]) * half;
        self.a[16] = self.a[64];
        self.a[65] = (self.a[65] + self.a[25]) * half;
        self.a[25] = self.a[65];
        self.a[66] = (self.a[66] + self.a[34]) * half;
        self.a[34] = self.a[66];
        self.a[67] = (self.a[67] + self.a[43]) * half;
        self.a[43] = self.a[67];
        self.a[68] = (self.a[68] + self.a[52]) * half;
        self.a[52] = self.a[68];
        self.a[69] = (self.a[69] + self.a[61]) * half;
        self.a[61] = self.a[69];

        // --- Column 8 Cross-terms ---
        self.a[72] = (self.a[72] + self.a[8]) * half;
        self.a[8] = self.a[72];
        self.a[73] = (self.a[73] + self.a[17]) * half;
        self.a[17] = self.a[73];
        self.a[74] = (self.a[74] + self.a[26]) * half;
        self.a[26] = self.a[74];
        self.a[75] = (self.a[75] + self.a[35]) * half;
        self.a[35] = self.a[75];
        self.a[76] = (self.a[76] + self.a[44]) * half;
        self.a[44] = self.a[76];
        self.a[77] = (self.a[77] + self.a[53]) * half;
        self.a[53] = self.a[77];
        self.a[78] = (self.a[78] + self.a[62]) * half;
        self.a[62] = self.a[78];
        self.a[79] = (self.a[79] + self.a[71]) * half;
        self.a[71] = self.a[79];
    }
}

// **** Iterators ****

impl<T> Matrix9x9<T> {
    /// Returns an iterator over the rows of the matrix as slices of 9 elements.
    #[inline]
    pub fn rows(&self) -> ChunksExact<'_, T> {
        self.chunks_exact(9)
    }
}

impl<T> Matrix9x9<T> {
    /// Returns an iterator over the rows of the matrix as mutable slices of 9 elements.
    #[inline]
    pub fn rows_mut(&mut self) -> ChunksExactMut<'_, T> {
        self.chunks_exact_mut(9)
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Consumes the matrix and returns an array of its 9 rows.
    #[inline]
    pub fn into_rows(self) -> [[T; 9]; 9] {
        // Build the nested 2D array matrix safely in a single unrolled pass
        core::array::from_fn(|r| core::array::from_fn(|c| self.a[r * 9 + c]))
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Returns an iterator over the columns of the matrix as owned 9-element arrays.
    #[inline]
    pub fn cols(&self) -> impl Iterator<Item = [T; 9]> + '_ {
        // Create an iterator over the column indices (0, 1, 2, ..)
        (0..9).map(|c| {
            // Collect the strided elements for the current column
            [
                self.a[c],
                self.a[c + 9],
                self.a[c + 18],
                self.a[c + 27],
                self.a[c + 36],
                self.a[c + 45],
                self.a[c + 54],
                self.a[c + 63],
                self.a[c + 72],
            ]
        })
    }
}

impl<'a, T> IntoIterator for &'a Matrix9x9<T> {
    type Item = &'a [T];
    type IntoIter = ChunksExact<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages the Deref trait automatically to get 9-element rows
        self.chunks_exact(9)
    }
}

impl<'a, T> IntoIterator for &'a mut Matrix9x9<T> {
    type Item = &'a mut [T];
    type IntoIter = ChunksExactMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages the DerefMut trait automatically.
        self.chunks_exact_mut(9)
    }
}

impl<T> IntoIterator for Matrix9x9<T>
where
    T: Copy,
{
    type Item = [T; 9];
    type IntoIter = core::array::IntoIter<[T; 9], 9>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Construct the 9 rows of 9 elements safely in one direct pass
        let rows = core::array::from_fn(|r| core::array::from_fn(|c| self.a[r * 9 + c]));
        rows.into_iter()
    }
}

// **** Column Iterators ****

impl<T> Matrix9x9<T> {
    /// Exposes the matrix as a read-only reference to 9 contiguous columns.
    /// Each sub-array `[T; 9]` represents one full column in memory.
    #[inline]
    pub fn columns(&self) -> &[[T; 9]] {
        // SAFETY:
        // `self.a` contains 81 contiguous `T`s, so it can be viewed as 9 contiguous `[T; 9]` values.
        //`[T; 9]` has the same alignment as `T`.
        unsafe { core::slice::from_raw_parts(self.a.as_ptr().cast::<[T; 9]>(), 9) }
    }

    /// Exposes the matrix as a mutable reference to 9 contiguous columns.
    /// Each sub-array `[T; 9]` represents one full column in memory.
    #[inline]
    pub fn columns_mut(&mut self) -> &mut [[T; 9]] {
        // SAFETY: `self.a` contains 81 contiguous `T`s, which are 9 contiguous `[T; 9]` values.
        // `[T; 9]` has the same alignment as `T`, and the returned reference is uniquely borrowed from `self.a`.
        unsafe { core::slice::from_raw_parts_mut(self.a.as_mut_ptr().cast::<[T; 9]>(), 9) }
    }

    #[inline]
    pub fn iter_columns(&self) -> Matrix9x9Columns<'_, T> {
        Matrix9x9Columns { inner: self.columns().iter() }
    }

    #[inline]
    pub fn iter_columns_mut(&mut self) -> Matrix9x9ColumnsMut<'_, T> {
        Matrix9x9ColumnsMut { inner: self.columns_mut().iter_mut() }
    }
}

// **** Iterator Pairs ****

/// A custom iterator over the read-only columns of a 9x9 matrix.
#[derive(Debug, Default)]
pub struct Matrix9x9Columns<'a, T> {
    inner: Iter<'a, [T; 9]>,
}

impl<'a, T> Iterator for Matrix9x9Columns<'a, T> {
    type Item = &'a [T; 9];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// Support optimization traits identically to the mutable companion
impl<T> ExactSizeIterator for Matrix9x9Columns<'_, T> {}

impl<T> DoubleEndedIterator for Matrix9x9Columns<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

/// A custom iterator over the mutable columns of a 9x9 matrix.
#[derive(Debug)]
pub struct Matrix9x9ColumnsMut<'a, T> {
    inner: IterMut<'a, [T; 9]>,
}

impl<'a, T> Iterator for Matrix9x9ColumnsMut<'a, T> {
    type Item = &'a mut [T; 9];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// Ensure the standard ExactSizeIterator trait is supported for zip/enumerate optimization.
impl<T> ExactSizeIterator for Matrix9x9ColumnsMut<'_, T> {}

impl<T> DoubleEndedIterator for Matrix9x9ColumnsMut<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

// **** From ****

// **** From Array ****

impl<T> From<[T; 81]> for Matrix9x9<T> {
    /// Matrix from 1D array.
    #[inline]
    fn from(input: [T; 81]) -> Self {
        Self { a: input }
    }
}

impl<T: Copy> From<Matrix9x9<T>> for Matrix2x2<T> {
    /// Matrix2x2 from Matrix9x9. Takes top left of m9x9, discarding other values.
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix9x9<T>) -> Self {
        Self { a: [
            m.a[0],  m.a[1],
            m.a[9],  m.a[10],
        ] }
    }
}

impl<T: Copy> From<Matrix9x9<T>> for Matrix3x3<T> {
    /// Matrix3x3 from Matrix9x9. Takes top left of m9x9, discarding other values.
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix9x9<T>) -> Self {
        Self { a: [
            m.a[0],  m.a[1],  m.a[2],
            m.a[9],  m.a[10], m.a[11],
            m.a[18], m.a[19], m.a[20]
        ] }
    }
}

impl<T: Copy> From<Matrix9x9<T>> for Matrix4x4<T> {
    /// Matrix4x4 from Matrix9x9. Takes top left of m9x9, discarding other values.
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix9x9<T>) -> Self {
        Self { a: [
            m.a[0],  m.a[1],  m.a[2],  m.a[3],
            m.a[9],  m.a[10], m.a[11], m.a[12],
            m.a[18], m.a[19], m.a[20], m.a[21],
            m.a[27], m.a[28], m.a[29], m.a[30],
        ] }
    }
}
