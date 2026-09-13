use core::fmt;
use core::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, RangeFull,
    RangeInclusive, Sub, SubAssign,
};
use core::slice::{ChunksExact, ChunksExactMut, Iter, IterMut};
use num_traits::{ConstOne, ConstZero, MulAdd, MulAddAssign, One, Zero, float::FloatCore};
#[cfg(feature = "serde")]
use {
    postcard::experimental::max_size::MaxSize,
    sequential_storage::map::PostcardValue,
    serde::{Deserialize, Serialize},
};

use crate::{MathConstants, Matrix2x2, Matrix3x3, Matrix4x4Math, Quaternion, Vector4};

/// 4x4 matrix of `f32` values<br>
pub type Matrix4x4f32 = Matrix4x4<f32>;
/// 4x4 matrix of `f64` values<br><br>
pub type Matrix4x4f64 = Matrix4x4<f64>;

// **** Define ****

/// `Matrix4x4<T>`: 4x4 Matrix of type `T`.<br>
/// Aliases `Matrix4x4f32` and `Matrix4x4f64` are provided.<br>
/// Internal implementation is a flattened 4x4 matrix: an array of 16 elements stored in column-major order.
/// That is the element `m[row][col]` is at array position `[col * 4 + row]`.<br><br>
#[derive(Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", allow(clippy::unsafe_derive_deserialize))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize, MaxSize))]
#[repr(C, align(64))]
pub struct Matrix4x4<T> {
    // Flattened 4x4 matrix: 16 elements in column-major order
    pub(crate) a: [T; 16],
}

#[cfg(feature = "serde")]
impl<T> PostcardValue<'_> for Matrix4x4<T> where T: Serialize + for<'de> Deserialize<'de> {}

impl<T> fmt::Debug for Matrix4x4<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Start the struct block wrapper
        writeln!(f, "Matrix4x4 [")?;

        // Loop over rows using the Deref slice chunking behavior we added earlier
        for row in self.chunks_exact(4) {
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
impl<T> Matrix4x4<T> {
    pub const SIZE: usize = 16;
    pub const ROW_COUNT: usize = 4;
    pub const COL_COUNT: usize = 4;
    // Column 1
    pub const M11: usize = 0;
    pub const M21: usize = 1;
    pub const M31: usize = 2;
    pub const M41: usize = 3;
    // Column 2
    pub const M12: usize = 4;
    pub const M22: usize = 5;
    pub const M32: usize = 6;
    pub const M42: usize = 7;
    // Column 3
    pub const M13: usize = 8;
    pub const M23: usize = 9;
    pub const M33: usize = 10;
    pub const M43: usize = 11;
    // Column 4
    pub const M14: usize = 12;
    pub const M24: usize = 13;
    pub const M34: usize = 14;
    pub const M44: usize = 15;
}

// **** New ****

impl<T> Matrix4x4<T>
where
    T: Copy,
{
    /// Create a matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// assert_eq!(m, Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 11.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 41.0, 103.0,
    ///                                   67.0, 73.0, 83.0,  97.0]));
    /// ```
    #[inline]
    pub const fn new(a: [T; 16]) -> Self {
        Self::from_row_array(a)
    }
}

// **** Other constructors ****

impl<T> Matrix4x4<T>
where
    T: Copy,
{
    /// Create a matrix with all its elements set to a single value.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from_element(2.0);
    /// assert_eq!(m, Matrix4x4f32::new([  2.0, 2.0, 2.0, 2.0,
    ///                                    2.0, 2.0, 2.0, 2.0,
    ///                                    2.0, 2.0, 2.0, 2.0,
    ///                                    2.0, 2.0, 2.0, 2.0]));
    /// ```
    #[inline]
    pub const fn from_element(value: T) -> Self {
        Self { a: [value; 16] }
    }

    /// Matrix from array of row vectors.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4f32};
    /// let m = Matrix4x4f32::from_rows([ Vector4f32::new( 2.0, 17.0, 59.0, 127.0),
    ///                                   Vector4f32::new( 5.0, 11.0, 47.0, 109.0),
    ///                                   Vector4f32::new(23.0, 31.0, 41.0, 103.0),
    ///                                   Vector4f32::new(67.0, 73.0, 83.0,  97.0) ]);
    /// assert_eq!(m, Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 11.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 41.0, 103.0,
    ///                                   67.0, 73.0, 83.0,  97.0]));
    /// ```
    #[inline]
    pub const fn from_rows(v: [Vector4<T>; 4]) -> Self {
        Self {
            a: [
                v[0].x, v[1].x, v[2].x, v[3].x, //
                v[0].y, v[1].y, v[2].y, v[3].y, //
                v[0].z, v[1].z, v[2].z, v[3].z, //
                v[0].t, v[1].t, v[2].t, v[3].t, //
            ],
        }
    }

    /// Matrix from array of column vectors.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4f32};
    /// let m = Matrix4x4f32::from_columns([ Vector4f32::new( 2.0, 17.0, 59.0, 127.0),
    ///                                      Vector4f32::new( 5.0, 11.0, 47.0, 109.0),
    ///                                      Vector4f32::new(23.0, 31.0, 41.0, 103.0),
    ///                                      Vector4f32::new(67.0, 73.0, 83.0,  97.0) ]);
    /// assert_eq!(m, Matrix4x4f32::new([  2.0,   5.0,  23.0,  67.0,
    ///                                   17.0,  11.0,  31.0,  73.0,
    ///                                   59.0,  47.0,  41.0,  83.0,
    ///                                  127.0, 109.0, 103.0,  97.0]));
    /// ```
    #[inline]
    pub const fn from_columns(v: [Vector4<T>; 4]) -> Self {
        Self {
            a: [
                v[0].x, v[0].y, v[0].z, v[0].t, //
                v[1].x, v[1].y, v[1].z, v[1].t, //
                v[2].x, v[2].y, v[2].z, v[2].t, //
                v[3].x, v[3].y, v[3].z, v[3].t, //
            ],
        }
    }

    /// Matrix from 1D row array.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from_row_array([  2.0, 17.0, 59.0, 127.0,
    ///                                         5.0, 11.0, 47.0, 109.0,
    ///                                        23.0, 31.0, 41.0, 103.0,
    ///                                        67.0, 73.0, 83.0,  97.0]);
    /// assert_eq!(m, Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 11.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 41.0, 103.0,
    ///                                   67.0, 73.0, 83.0,  97.0]));
    /// ```
    #[inline]
    pub const fn from_row_array(a: [T; 16]) -> Self {
        Self {
            a: [
                a[0], a[4], a[8], a[12], //
                a[1], a[5], a[9], a[13], //
                a[2], a[6], a[10], a[14], //
                a[3], a[7], a[11], a[15], //
            ],
        }
    }

    /// Matrix from 1D column array.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from_column_array([  2.0, 17.0, 59.0, 127.0,
    ///                                            5.0, 11.0, 47.0, 109.0,
    ///                                           23.0, 31.0, 41.0, 103.0,
    ///                                           67.0, 73.0, 83.0,  97.0]);
    /// assert_eq!(m, Matrix4x4f32::new([  2.0,   5.0,  23.0,  67.0,
    ///                                   17.0,  11.0,  31.0,  73.0,
    ///                                   59.0,  47.0,  41.0,  83.0,
    ///                                  127.0, 109.0, 103.0,  97.0]));
    /// ```
    #[inline]
    pub const fn from_column_array(a: [T; 16]) -> Self {
        Self { a }
    }

    /// Matrix from 2D row array.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from_2d_row_array([[  2.0, 17.0, 59.0, 127.0],
    ///                                          [  5.0, 11.0, 47.0, 109.0],
    ///                                          [ 23.0, 31.0, 41.0, 103.0],
    ///                                          [ 67.0, 73.0, 83.0,  97.0]]);
    /// assert_eq!(m, Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 11.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 41.0, 103.0,
    ///                                   67.0, 73.0, 83.0,  97.0]));
    /// ```
    #[inline]
    pub const fn from_2d_row_array(a: [[T; 4]; 4]) -> Self {
        Self {
            a: [
                a[0][0], a[1][0], a[2][0], a[3][0], //
                a[0][1], a[1][1], a[2][1], a[3][1], //
                a[0][2], a[1][2], a[2][2], a[3][2], //
                a[0][3], a[1][3], a[2][3], a[3][3], //
            ],
        }
    }

    /// Matrix from 2D column array.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from_2d_column_array([[  2.0, 17.0, 59.0, 127.0],
    ///                                             [  5.0, 11.0, 47.0, 109.0],
    ///                                             [ 23.0, 31.0, 41.0, 103.0],
    ///                                             [ 67.0, 73.0, 83.0,  97.0]]);
    /// assert_eq!(m, Matrix4x4f32::new([  2.0,   5.0,  23.0,  67.0,
    ///                                   17.0,  11.0,  31.0,  73.0,
    ///                                   59.0,  47.0,  41.0,  83.0,
    ///                                  127.0, 109.0, 103.0,  97.0]));
    /// ```
    #[inline]
    pub const fn from_2d_column_array(a: [[T; 4]; 4]) -> Self {
        Self {
            a: [
                a[0][0], a[0][1], a[0][2], a[0][3], //
                a[1][0], a[1][1], a[1][2], a[1][3], //
                a[2][0], a[2][1], a[2][2], a[2][3], //
                a[3][0], a[3][1], a[3][2], a[3][3], //
            ],
        }
    }

    /// Try to create a matrix from a row slice.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let valid_data = [2.0; 16];
    /// let invalid_data = [2.0; 3];
    /// let Some(m) = Matrix4x4f32::try_from_row_slice(&valid_data) else {
    ///     panic!("Expected Some(Matrix3x3), but got None");
    /// };
    /// assert_eq!(2.0, m[0]);
    /// let None = Matrix4x4f32::try_from_row_slice(&invalid_data) else {
    ///     panic!("Expected None for invalid data, but got Some");
    /// };
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn try_from_row_slice(slice: &[T]) -> Option<Self> {
        if slice.len() != 16 {
            return None;
        }
        let a = [
            slice[0], slice[4], slice[8],  slice[12],
            slice[1], slice[5], slice[9],  slice[13],
            slice[2], slice[6], slice[10], slice[14],
            slice[3], slice[7], slice[11], slice[15],
        ];
        Some(Self { a })
    }
    /// Try to create a matrix from a column slice.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let valid_data = [2.0; 16];
    /// let invalid_data = [2.0; 3];
    /// let Some(m) = Matrix4x4f32::try_from_column_slice(&valid_data) else {
    ///     panic!("Expected Some(Matrix4x4), but got None");
    /// };
    /// assert_eq!(2.0, m[0]);
    /// let None = Matrix4x4f32::try_from_column_slice(&invalid_data) else {
    ///     panic!("Expected None for invalid data, but got Some");
    /// };
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn try_from_column_slice(slice: &[T]) -> Option<Self> {
        if slice.len() != 16 {
            return None;
        }
        let mut a = [slice[0]; 16];
        a.copy_from_slice(&slice[0..16]);
        Some(Self { a })
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy + ConstZero,
{
    /// Create a matrix with the diagonal set to a single value.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from_diagonal_element(2.0);
    /// assert_eq!(m, Matrix4x4f32::new([ 2.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 2.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 2.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 2.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_element(value: T) -> Self {
        Self {
            a: [
                value,   T::ZERO, T::ZERO, T::ZERO,
                T::ZERO, value,   T::ZERO, T::ZERO,
                T::ZERO, T::ZERO, value,   T::ZERO,
                T::ZERO, T::ZERO, T::ZERO, value,
            ]
        }
    }
    /// Create a matrix with the diagonal set to an array.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from_diagonal_array([ 2.0, 3.0, 5.0, 7.0 ]);
    /// assert_eq!(m, Matrix4x4f32::new([ 2.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 3.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 5.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 7.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_array(a: [T;4]) -> Self {
        Self {
            a: [
                a[0],    T::ZERO, T::ZERO, T::ZERO,
                T::ZERO, a[1],    T::ZERO, T::ZERO,
                T::ZERO, T::ZERO, a[2],    T::ZERO,
                T::ZERO, T::ZERO, T::ZERO, a[3],
            ]
        }
    }
    /// Create a matrix with the diagonal set to a vector.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4f32};
    /// let m = Matrix4x4f32::from_diagonal_vector(Vector4f32::new(2.0, 3.0, 5.0, 7.0));
    /// assert_eq!(m, Matrix4x4f32::new([ 2.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 3.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 5.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 7.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_vector(v: Vector4<T>) -> Self {
        Self {
            a: [
                v.x,     T::ZERO, T::ZERO, T::ZERO,
                T::ZERO, v.y,     T::ZERO, T::ZERO,
                T::ZERO, T::ZERO, v.z,     T::ZERO,
                T::ZERO, T::ZERO, T::ZERO, v.t,
            ]
        }
    }
}

// **** Zero ****

impl<T> Zero for Matrix4x4<T>
where
    T: Copy + Zero + PartialEq + Matrix4x4Math,
{
    /// Zero matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::Zero;
    /// let z = Matrix4x4f32::zero();
    ///
    /// assert_eq!(z, Matrix4x4f32::new([ 0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 0.0]));
    /// ```
    #[inline]
    fn zero() -> Self {
        Self { a: [T::zero(); 16] }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<T> ConstZero for Matrix4x4<T>
where
    T: Copy + ConstZero + PartialEq + Matrix4x4Math,
{
    /// Const zero matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::{zero,Zero,ConstZero};
    /// let m = Matrix4x4f32::ZERO;
    /// assert!(m.is_zero());
    /// ```
    const ZERO: Self = Self { a: [T::ZERO; 16] };
}

impl<T> Matrix4x4<T>
where
    T: Copy + FloatCore,
{
    /// Return true if matrix is near zero.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::Zero;
    /// let z = Matrix4x4f32::zero();
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

impl<T> One for Matrix4x4<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix4x4Math,
{
    /// Identity matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::One;
    /// let i = Matrix4x4f32::one();
    ///
    /// assert!(i.is_one());
    /// assert_eq!(i, Matrix4x4f32::new([ 1.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 1.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 1.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 1.0]));
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

impl<T> ConstOne for Matrix4x4<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix4x4Math,
{
    /// Const identity matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::ConstOne;
    /// let i = Matrix4x4f32::ONE;
    ///
    /// assert_eq!(i, Matrix4x4f32::new([ 1.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 1.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 1.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 1.0]));
    /// ```
    #[rustfmt::skip]
    const ONE: Self = Self {
        a: [
            T::ONE,  T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ONE,  T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ONE,  T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ONE,
        ]
    };
}

impl<T> Matrix4x4<T>
where
    T: Copy + Zero + One,
{
    /// Identity matrix.
    /// Alias for `one()` that does not require `num_traits::One`.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let i = Matrix4x4f32::identity();
    ///
    /// assert_eq!(i, Matrix4x4f32::new([ 1.0, 0.0, 0.0, 0.0,
    ///                                   0.0, 1.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 1.0, 0.0,
    ///                                   0.0, 0.0, 0.0, 1.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        Self {
            a: [
                T::one(),  T::zero(), T::zero(), T::zero(),
                T::zero(), T::one(),  T::zero(), T::zero(),
                T::zero(), T::zero(), T::one(),  T::zero(),
                T::zero(), T::zero(), T::zero(), T::one()
            ],
        }
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy + One + FloatCore,
{
    /// Return true if matrix is near identity.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::One;
    /// let i = Matrix4x4f32::one();
    /// assert!(i.is_near_identity(1e-5));
    /// ```
    pub fn is_near_identity(self, epsilon: T) -> bool {
        if self.a[Self::M21].abs() > epsilon
            || self.a[Self::M31].abs() > epsilon
            || self.a[Self::M41].abs() > epsilon
            || self.a[Self::M12].abs() > epsilon
            || self.a[Self::M32].abs() > epsilon
            || self.a[Self::M42].abs() > epsilon
            || self.a[Self::M13].abs() > epsilon
            || self.a[Self::M23].abs() > epsilon
            || self.a[Self::M43].abs() > epsilon
            || self.a[Self::M14].abs() > epsilon
            || self.a[Self::M24].abs() > epsilon
            || self.a[Self::M34].abs() > epsilon
        {
            return false;
        }
        if (self.a[Self::M11] - T::one()).abs() > epsilon
            || (self.a[Self::M22] - T::one()).abs() > epsilon
            || (self.a[Self::M33] - T::one()).abs() > epsilon
            || (self.a[Self::M44] - T::one()).abs() > epsilon
        {
            return false;
        }
        true
    }
}

// **** Neg ****

impl<T> Neg for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    /// Negate matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// m = - m;
    ///
    /// assert_eq!(m, Matrix4x4f32::new([ -2.0, -17.0, -59.0, -127.0,
    ///                                   -5.0, -11.0, -47.0, -109.0,
    ///                                  -23.0, -31.0, -41.0, -103.0,
    ///                                  -67.0, -73.0, -83.0,  -97.0]));
    /// ```
    #[inline]
    fn neg(self) -> Self {
        T::m4x4_neg(self)
    }
}

// **** Add ****

impl<T> Add for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    /// Add two matrices.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let n = Matrix4x4f32::new([  3.0, 19.0, 61.0, 131.0,
    ///                              7.0, 13.0, 53.0, 113.0,
    ///                             29.0, 37.0, 43.0, 107.0,
    ///                             71.0, 79.0, 89.0, 101.0]);
    /// let r = m + n;
    ///
    /// assert_eq!(r, Matrix4x4f32::new([  5.0, 36.0, 120.0, 258.0,
    ///                                   12.0, 24.0, 100.0, 222.0,
    ///                                   52.0, 68.0,  84.0, 210.0,
    ///                                  138.0,152.0, 172.0, 198.0]));
    ///
    /// # use num_traits::Zero;
    ///
    /// let z = Matrix4x4f32::zero();
    /// let r2 = m + z;
    ///
    /// assert_eq!(r2, m);
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        T::m4x4_add(self, other)
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy + One + FloatCore + AddAssign,
{
    /// Add a diagonal matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let n = Matrix4x4f32::new([  3.0,  0.0,  0.0,  0.0,
    ///                              0.0, 13.0,  0.0,  0.0,
    ///                              0.0,  0.0, 43.0,  0.0,
    ///                              0.0,  0.0,  0.0, 101.0]);
    /// let p = m + n;
    /// let r = m.add_diagonal(n);
    ///
    /// assert_eq!(r, p);
    /// assert_eq!(r, Matrix4x4f32::new([  5.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 24.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 84.0, 103.0,
    ///                                   67.0, 73.0, 83.0, 198.0]));
    /// ```
    #[must_use]
    pub fn add_diagonal(self, other: Self) -> Self {
        let mut ret = self.a;
        ret[Self::M11] += other.a[Self::M11];
        ret[Self::M22] += other.a[Self::M22];
        ret[Self::M33] += other.a[Self::M33];
        ret[Self::M44] += other.a[Self::M44];
        Self { a: ret }
    }

    /// Add a diagonal matrix, in-place.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// let n = Matrix4x4f32::new([  3.0,  0.0,  0.0,  0.0,
    ///                              0.0, 13.0,  0.0,  0.0,
    ///                              0.0,  0.0, 43.0,  0.0,
    ///                              0.0,  0.0,  0.0, 101.0]);
    /// let p = m + n;
    /// m.add_diagonal_in_place(n);
    ///
    /// assert_eq!(m, p);
    /// assert_eq!(m, Matrix4x4f32::new([  5.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 24.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 84.0, 103.0,
    ///                                   67.0, 73.0, 83.0, 198.0]));
    /// ```
    pub fn add_diagonal_in_place(&mut self, other: Self) {
        self.a[Self::M11] += other.a[Self::M11];
        self.a[Self::M22] += other.a[Self::M22];
        self.a[Self::M33] += other.a[Self::M33];
        self.a[Self::M44] += other.a[Self::M44];
    }

    /// Add a scalar that represents a diagonal matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let s = 3.0;
    /// let n = Matrix4x4f32::new([  s,  0.0,  0.0,  0.0,
    ///                              0.0,  s,  0.0,  0.0,
    ///                              0.0,  0.0,  s,  0.0,
    ///                              0.0,  0.0,  0.0,  s]);
    /// let p = m + n;
    /// let r = m.add_diagonal_scalar(s);
    ///
    /// assert_eq!(r, p);
    /// assert_eq!(r, Matrix4x4f32::new([  5.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 14.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 44.0, 103.0,
    ///                                   67.0, 73.0, 83.0, 100.0]));
    /// ```
    #[must_use]
    pub fn add_diagonal_scalar(self, other: T) -> Self {
        let mut ret = self.a;
        ret[Self::M11] += other;
        ret[Self::M22] += other;
        ret[Self::M33] += other;
        ret[Self::M44] += other;
        Self { a: ret }
    }

    /// Add a scalar that represents a diagonal matrix, in place.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let s = 3.0;
    /// let n = Matrix4x4f32::new([  s,  0.0,  0.0,  0.0,
    ///                              0.0,  s,  0.0,  0.0,
    ///                              0.0,  0.0,  s,  0.0,
    ///                              0.0,  0.0,  0.0,  s]);
    /// let p = m + n;
    /// m.add_diagonal_scalar_in_place(s);
    ///
    /// assert_eq!(m, p);
    /// assert_eq!(m, Matrix4x4f32::new([  5.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 14.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 44.0, 103.0,
    ///                                   67.0, 73.0, 83.0, 100.0]));
    /// ```
    pub fn add_diagonal_scalar_in_place(&mut self, other: T) {
        self.a[Self::M11] += other;
        self.a[Self::M22] += other;
        self.a[Self::M33] += other;
        self.a[Self::M44] += other;
    }

    /// Add a vector that represents a diagonal matrix.
    /// ```
    /// # use vqm::{Matrix4x4f32, Vector4f32};
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let v = Vector4f32 { x: 3.0, y: 13.0, z: 43.0, t: 101.0 };
    /// let n = Matrix4x4f32::new([  v.x,  0.0,  0.0,  0.0,
    ///                              0.0,  v.y,  0.0,  0.0,
    ///                              0.0,  0.0,  v.z,  0.0,
    ///                              0.0,  0.0,  0.0,  v.t]);
    /// let p = m + n;
    /// let r = m.add_diagonal_vector(v);
    ///
    /// assert_eq!(r, p);
    /// assert_eq!(r    , Matrix4x4f32::new([  5.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 24.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 84.0, 103.0,
    ///                                   67.0, 73.0, 83.0, 198.0]));
    /// ```
    #[must_use]
    pub fn add_diagonal_vector(self, other: Vector4<T>) -> Self {
        let mut ret = self.a;
        ret[Self::M11] += other.x;
        ret[Self::M22] += other.y;
        ret[Self::M33] += other.z;
        ret[Self::M44] += other.t;
        Self { a: ret }
    }

    /// Add a vector that represents a diagonal matrix, in-place.
    /// ```
    /// # use vqm::{Matrix4x4f32, Vector4f32};
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// let v = Vector4f32 { x: 3.0, y: 13.0, z: 43.0, t: 101.0 };
    /// let n = Matrix4x4f32::new([  v.x,  0.0,  0.0,  0.0,
    ///                              0.0,  v.y,  0.0,  0.0,
    ///                              0.0,  0.0,  v.z,  0.0,
    ///                              0.0,  0.0,  0.0,  v.t]);
    /// let p = m + n;
    /// m.add_diagonal_vector_in_place(v);
    ///
    /// assert_eq!(m, p);
    /// assert_eq!(m, Matrix4x4f32::new([  5.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 24.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 84.0, 103.0,
    ///                                   67.0, 73.0, 83.0, 198.0]));
    /// ```
    pub fn add_diagonal_vector_in_place(&mut self, other: Vector4<T>) {
        self.a[Self::M11] += other.x;
        self.a[Self::M22] += other.y;
        self.a[Self::M33] += other.z;
        self.a[Self::M44] += other.t;
    }

    /// Add an array that represents a diagonal matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let a = [3.0, 13.0, 43.0, 101.0 ];
    /// let n = Matrix4x4f32::new([  a[0], 0.0,  0.0,  0.0,
    ///                              0.0,  a[1], 0.0,  0.0,
    ///                              0.0,  0.0,  a[2], 0.0,
    ///                              0.0,  0.0,  0.0,  a[3]]);
    /// let s = m + n;
    /// let r = m.add_diagonal_array(a);
    ///
    /// assert_eq!(r, s);
    /// assert_eq!(r, Matrix4x4f32::new([  5.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 24.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 84.0, 103.0,
    ///                                   67.0, 73.0, 83.0, 198.0]));
    /// ```
    #[must_use]
    pub fn add_diagonal_array(self, other: [T; 4]) -> Self {
        let mut ret = self.a;
        ret[Self::M11] += other[0];
        ret[Self::M22] += other[1];
        ret[Self::M33] += other[2];
        ret[Self::M44] += other[3];
        Self { a: ret }
    }

    /// Add an array that represents a diagonal matrix, in-place.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// let a = [3.0, 13.0, 43.0, 101.0 ];
    /// let n = Matrix4x4f32::new([  a[0], 0.0,  0.0,  0.0,
    ///                              0.0,  a[1], 0.0,  0.0,
    ///                              0.0,  0.0,  a[2], 0.0,
    ///                              0.0,  0.0,  0.0,  a[3]]);
    /// let s = m + n;
    /// m.add_diagonal_array_in_place(a);
    ///
    /// assert_eq!(m, s);
    /// assert_eq!(m, Matrix4x4f32::new([  5.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 24.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 84.0, 103.0,
    ///                                   67.0, 73.0, 83.0, 198.0]));
    /// ```
    pub fn add_diagonal_array_in_place(&mut self, other: [T; 4]) {
        self.a[Self::M11] += other[0];
        self.a[Self::M22] += other[1];
        self.a[Self::M33] += other[2];
        self.a[Self::M44] += other[3];
    }
}

// **** AddAssign ****

impl<T> AddAssign for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// Add one matrix to another.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// let n = Matrix4x4f32::new([  3.0, 19.0, 61.0, 131.0,
    ///                              7.0, 13.0, 53.0, 113.0,
    ///                             29.0, 37.0, 43.0, 107.0,
    ///                             71.0, 79.0, 89.0, 101.0]);
    /// m += n;
    ///
    /// assert_eq!(m, Matrix4x4f32::new([  5.0, 36.0, 120.0, 258.0,
    ///                                   12.0, 24.0, 100.0, 222.0,
    ///                                   52.0, 68.0,  84.0, 210.0,
    ///                                  138.0,152.0, 172.0, 198.0]));
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

impl<T> MulAdd<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    /// Multiply matrix by constant and add another matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::MulAdd;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let n = Matrix4x4f32::new([  3.0, 19.0, 61.0, 131.0,
    ///                              7.0, 13.0, 53.0, 113.0,
    ///                             29.0, 37.0, 43.0, 107.0,
    ///                             71.0, 79.0, 89.0, 101.0]);
    /// let k = 137.0;
    /// let r = m.mul_add(k, n);
    /// assert_eq!(r, Matrix4x4f32::new([  277.0,  2348.0,  8144.0, 17530.0,
    ///                                    692.0,  1520.0,  6492.0, 15046.0,
    ///                                   3180.0,  4284.0,  5660.0, 14218.0,
    ///                                   9250.0, 10080.0, 11460.0, 13390.0]));
    /// ```
    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::m4x4_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

impl<T> MulAddAssign<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// Multiply matrix by constant and add another matrix in place.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::MulAddAssign;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// let n = Matrix4x4f32::new([  3.0, 19.0, 61.0, 131.0,
    ///                              7.0, 13.0, 53.0, 113.0,
    ///                             29.0, 37.0, 43.0, 107.0,
    ///                             71.0, 79.0, 89.0, 101.0]);
    /// let k = 137.0;
    /// m.mul_add_assign(k, n);
    ///
    /// assert_eq!(m, Matrix4x4f32::new([  277.0,  2348.0,  8144.0, 17530.0,
    ///                                    692.0,  1520.0,  6492.0, 15046.0,
    ///                                   3180.0,  4284.0,  5660.0, 14218.0,
    ///                                   9250.0, 10080.0, 11460.0, 13390.0]));
    /// ```
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

impl<T> Sub for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    /// Subtract two matrices.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    ///
    /// let n = Matrix4x4f32::new([  3.0, 13.0, 61.0, 131.0,
    ///                              7.0, 19.0, 53.0, 113.0,
    ///                             29.0, 37.0, 43.0, 107.0,
    ///                             71.0, 79.0, 89.0, 101.0]);
    /// let r = m - n;
    ///
    /// assert_eq!(r, Matrix4x4f32::new([  -1.0,  4.0, -2.0, -4.0,
    ///                                    -2.0, -8.0, -6.0, -4.0,
    ///                                    -6.0, -6.0, -2.0, -4.0,
    ///                                    -4.0, -6.0, -6.0, -4.0]));
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

impl<T> SubAssign for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// Subtract one matrix from another.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// let n = Matrix4x4f32::new([  3.0, 13.0, 43.0, 131.0,
    ///                              7.0, 19.0, 37.0, 113.0,
    ///                             29.0, 61.0, 53.0, 107.0,
    ///                             71.0, 79.0, 89.0, 101.0]);
    /// m -= n;
    ///
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Pre-multiply ****

impl Mul<Matrix4x4<f32>> for f32 {
    type Output = Matrix4x4<f32>;

    /// Pre-multiply a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let r = 2.0 * m;
    ///
    /// assert_eq!(r, Matrix4x4f32::new([  4.0, 34.0, 118.0, 254.0,
    ///                                   10.0, 22.0,  94.0, 218.0,
    ///                                   46.0, 62.0,  82.0, 206.0,
    ///                                  134.0,146.0, 166.0, 194.0]));
    /// ```
    #[inline]
    fn mul(self, other: Matrix4x4<f32>) -> Matrix4x4<f32> {
        f32::m4x4_mul_scalar(other, self)
    }
}

impl Mul<Matrix4x4<f64>> for f64 {
    type Output = Matrix4x4<f64>;
    #[inline]
    fn mul(self, other: Matrix4x4<f64>) -> Matrix4x4<f64> {
        f64::m4x4_mul_scalar(other, self)
    }
}

// **** Mul Scalar ****

impl<T> Mul<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    /// Multiply a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let r = m * 2.0;
    ///
    /// ```
    #[inline]
    fn mul(self, other: T) -> Self {
        T::m4x4_mul_scalar(self, other)
    }
}

// **** MulAssign ****

impl<T> MulAssign<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// In-place multiply a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// m *= 2.0;
    ///
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

impl<T> Mul<Vector4<T>> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Vector4<T>;

    /// Multiply a vector by a matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use vqm::Vector4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let v = Vector4f32{x:3.0, y:7.0, z:13.0, t:17.0};
    /// let r = m * v;
    /// /// assert_eq!(r, Vector4f32{x:3051.0, y:2556.0, z:2570.0, t:3440.0});
    /// ```
    #[inline]
    fn mul(self, other: Vector4<T>) -> Vector4<T> {
        T::m4x4_mul_vector(self, other)
    }
}

#[cfg(not(feature = "uom"))]
impl<T> Mul<Matrix4x4<T>> for Vector4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    /// Pre-multiply a vector by a matrix.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4f32};
    /// let m = Matrix4x4f32::new([  2.0,   3.0,   5.0,   7.0,
    ///                             11.0,  13.0,  17.0,  19.0,
    ///                             23.0,  29.0,  31.0,  37.0,
    ///                             41.0,  43.0,  47.0,  53.0]);
    /// let v = Vector4f32{x:3.0, y:7.0,  z:13.0, t:17.0};
    /// let r = v * m;
    ///
    /// assert_eq!(r, Vector4f32{x:3.0*2.0 + 7.0*11.0 + 13.0*23.0 + 17.0*41.0,
    ///                           y:3.0*3.0 + 7.0*13.0 + 13.0*29.0 + 17.0*43.0,
    ///                           z:3.0*5.0 + 7.0*17.0 + 13.0*31.0 + 17.0*47.0,
    ///                           t:3.0*7.0 + 7.0*19.0 + 13.0*37.0 + 17.0*53.0});
    /// ```
    #[inline]
    fn mul(self, other: Matrix4x4<T>) -> Self {
        T::m4x4_vector_mul(self, other)
    }
}

// **** Mul ****

impl<T> Mul<Matrix4x4<T>> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    /// Multiply two matrices.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    ///
    /// let n = Matrix4x4f32::new([  3.0, 19.0, 61.0, 131.0,
    ///                              7.0, 13.0, 53.0, 113.0,
    ///                             29.0, 37.0, 43.0, 107.0,
    ///                             71.0, 79.0, 89.0, 101.0]);
    /// let r = m * n;
    ///
    /// assert_eq!(r, Matrix4x4f32::new([
    ///    2.0*  3.0 + 17.0*  7.0 + 59.0* 29.0 + 127.0* 71.0,
    ///    2.0* 19.0 + 17.0* 13.0 + 59.0* 37.0 + 127.0* 79.0,
    ///    2.0* 61.0 + 17.0* 53.0 + 59.0* 43.0 + 127.0* 89.0,
    ///    2.0*131.0 + 17.0*113.0 + 59.0*107.0 + 127.0*101.0,
    ///
    ///    5.0*  3.0 + 11.0*  7.0 + 47.0* 29.0 + 109.0* 71.0,
    ///    5.0* 19.0 + 11.0* 13.0 + 47.0* 37.0 + 109.0* 79.0,
    ///    5.0* 61.0 + 11.0* 53.0 + 47.0* 43.0 + 109.0* 89.0,
    ///    5.0*131.0 + 11.0*113.0 + 47.0*107.0 + 109.0*101.0,
    ///
    ///   23.0*  3.0 + 31.0*  7.0 + 41.0* 29.0 + 103.0* 71.0,
    ///   23.0* 19.0 + 31.0* 13.0 + 41.0* 37.0 + 103.0* 79.0,
    ///   23.0* 61.0 + 31.0* 53.0 + 41.0* 43.0 + 103.0* 89.0,
    ///   23.0*131.0 + 31.0*113.0 + 41.0*107.0 + 103.0*101.0,
    ///
    ///   67.0*  3.0 + 73.0*  7.0 + 83.0* 29.0 +  97.0* 71.0,
    ///   67.0* 19.0 + 73.0* 13.0 + 83.0* 37.0 +  97.0* 79.0,
    ///   67.0* 61.0 + 73.0* 53.0 + 83.0* 43.0 +  97.0* 89.0,
    ///   67.0*131.0 + 73.0*113.0 + 83.0*107.0 +  97.0*101.0,
    /// ]));
    ///
    /// # use num_traits::{One,one};
    ///
    /// let i = Matrix4x4f32::one();
    /// let r2 = m * i;
    ///
    /// assert_eq!(r2, m);
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        T::m4x4_mul(self, other)
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy + One + FloatCore,
{
    /// Multiply by a diagonal matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    ///
    /// let n = Matrix4x4f32::new([  3.0, 0.0,  0.0,   0.0,
    ///                              0.0, 9.0,  0.0,   0.0,
    ///                              0.0, 0.0, 43.0,   0.0,
    ///                              0.0, 0.0,  0.0, 101.0]);
    /// let r = m.mul_diagonal(n);
    /// let s = m * n;
    ///
    /// assert_eq!(r, s);
    /// ```
    #[must_use]
    pub fn mul_diagonal(self, other: Self) -> Self {
        let ret = [
            self.a[Self::M11] * other.a[Self::M11],
            self.a[Self::M21] * other.a[Self::M11],
            self.a[Self::M31] * other.a[Self::M11],
            self.a[Self::M41] * other.a[Self::M11],
            self.a[Self::M12] * other.a[Self::M22],
            self.a[Self::M22] * other.a[Self::M22],
            self.a[Self::M32] * other.a[Self::M22],
            self.a[Self::M42] * other.a[Self::M22],
            self.a[Self::M13] * other.a[Self::M33],
            self.a[Self::M23] * other.a[Self::M33],
            self.a[Self::M33] * other.a[Self::M33],
            self.a[Self::M43] * other.a[Self::M33],
            self.a[Self::M14] * other.a[Self::M44],
            self.a[Self::M24] * other.a[Self::M44],
            self.a[Self::M34] * other.a[Self::M44],
            self.a[Self::M44] * other.a[Self::M44],
        ];
        Self { a: ret }
    }

    /// Multiply by a vector that represents a diagonal matrix.
    /// ```
    /// # use vqm::{Matrix4x4f32, Vector4f32};
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let v = Vector4f32 { x: 3.0, y: 13.0, z: 43.0, t: 101.0 };
    /// let n = Matrix4x4f32::new([  v.x, 0.0,  0.0, 0.0,
    ///                              0.0, v.y,  0.0, 0.0,
    ///                              0.0, 0.0,  v.z, 0.0,
    ///                              0.0, 0.0,  0.0, v.t]);
    /// let r = m.mul_diagonal_vector(v);
    /// let s = m * n;
    /// let t = m.mul_diagonal(n);
    ///
    /// assert_eq!(r, s);
    /// assert_eq!(r, t);
    /// ```
    #[must_use]
    pub fn mul_diagonal_vector(self, other: Vector4<T>) -> Self {
        let ret = [
            self.a[Self::M11] * other.x,
            self.a[Self::M21] * other.x,
            self.a[Self::M31] * other.x,
            self.a[Self::M41] * other.x,
            self.a[Self::M12] * other.y,
            self.a[Self::M22] * other.y,
            self.a[Self::M32] * other.y,
            self.a[Self::M42] * other.y,
            self.a[Self::M13] * other.z,
            self.a[Self::M23] * other.z,
            self.a[Self::M33] * other.z,
            self.a[Self::M43] * other.z,
            self.a[Self::M14] * other.t,
            self.a[Self::M24] * other.t,
            self.a[Self::M34] * other.t,
            self.a[Self::M44] * other.t,
        ];
        Self { a: ret }
    }

    /// Multiply by a vector that represents a diagonal matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let a = [ 3.0, 13.0, 43.0, 101.0 ];
    /// let n = Matrix4x4f32::new([  a[0], 0.0,  0.0,  0.0,
    ///                              0.0,  a[1], 0.0,  0.0,
    ///                              0.0,  0.0,  a[2], 0.0,
    ///                              0.0,  0.0,  0.0,  a[3]]);
    /// let r = m.mul_diagonal_array(a);
    /// let s = m * n;
    /// let t = m.mul_diagonal(n);
    ///
    /// assert_eq!(r, s);
    /// assert_eq!(r, t);
    /// ```
    #[must_use]
    pub fn mul_diagonal_array(self, other: [T; 4]) -> Self {
        let ret = [
            self.a[Self::M11] * other[0],
            self.a[Self::M21] * other[0],
            self.a[Self::M31] * other[0],
            self.a[Self::M41] * other[0],
            self.a[Self::M12] * other[1],
            self.a[Self::M22] * other[1],
            self.a[Self::M32] * other[1],
            self.a[Self::M42] * other[1],
            self.a[Self::M13] * other[2],
            self.a[Self::M23] * other[2],
            self.a[Self::M33] * other[2],
            self.a[Self::M43] * other[2],
            self.a[Self::M14] * other[3],
            self.a[Self::M24] * other[3],
            self.a[Self::M34] * other[3],
            self.a[Self::M44] * other[3],
        ];
        Self { a: ret }
    }
}

impl<T> MulAssign<Matrix4x4<T>> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// Multiply one matrix by another.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// let n = Matrix4x4f32::new([  3.0, 19.0, 61.0, 131.0,
    ///                              7.0, 13.0, 53.0, 113.0,
    ///                             29.0, 37.0, 43.0, 107.0,
    ///                             71.0, 79.0, 89.0, 101.0]);
    /// m *= n;
    ///
    /// assert_eq!(m, Matrix4x4f32::new([
    ///    2.0*  3.0 + 17.0*  7.0 + 59.0* 29.0 + 127.0* 71.0,
    ///    2.0* 19.0 + 17.0* 13.0 + 59.0* 37.0 + 127.0* 79.0,
    ///    2.0* 61.0 + 17.0* 53.0 + 59.0* 43.0 + 127.0* 89.0,
    ///    2.0*131.0 + 17.0*113.0 + 59.0*107.0 + 127.0*101.0,
    ///
    ///    5.0*  3.0 + 11.0*  7.0 + 47.0* 29.0 + 109.0* 71.0,
    ///    5.0* 19.0 + 11.0* 13.0 + 47.0* 37.0 + 109.0* 79.0,
    ///    5.0* 61.0 + 11.0* 53.0 + 47.0* 43.0 + 109.0* 89.0,
    ///    5.0*131.0 + 11.0*113.0 + 47.0*107.0 + 109.0*101.0,
    ///
    ///   23.0*  3.0 + 31.0*  7.0 + 41.0* 29.0 + 103.0* 71.0,
    ///   23.0* 19.0 + 31.0* 13.0 + 41.0* 37.0 + 103.0* 79.0,
    ///   23.0* 61.0 + 31.0* 53.0 + 41.0* 43.0 + 103.0* 89.0,
    ///   23.0*131.0 + 31.0*113.0 + 41.0*107.0 + 103.0*101.0,
    ///
    ///   67.0*  3.0 + 73.0*  7.0 + 83.0* 29.0 +  97.0* 71.0,
    ///   67.0* 19.0 + 73.0* 13.0 + 83.0* 37.0 +  97.0* 79.0,
    ///   67.0* 61.0 + 73.0* 53.0 + 83.0* 43.0 +  97.0* 89.0,
    ///   67.0*131.0 + 73.0*113.0 + 83.0*107.0 +  97.0*101.0,
    /// ]));
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Matrix4x4<T>) {
        *self = *self * other;
    }
}

// **** Outer Product ****

impl<T> Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// Calculates the outer product of a column vector and a row vector to give a matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use vqm::Vector4f32;
    /// let row = Vector4f32{x:2.0, y:5.0, z:11.0, t:17.0};
    /// let col = Vector4f32{x:3.0, y:7.0, z:13.0, t:19.0};
    /// let m = Matrix4x4f32::outer_product(col, row);
    /// assert_eq!(m, Matrix4x4f32::new([ 6.0,  15.0,  33.0, 51.0,
    ///                                  14.0,  35.0,  77.0, 119.0,
    ///                                  26.0,  65.0, 143.0, 221.0,
    ///                                  38.0,  95.0, 209.0, 323.0]));
    ///```
    #[inline]
    pub fn outer_product(col: Vector4<T>, row: Vector4<T>) -> Self {
        T::m4x4_vector_outer_product(col, row)
    }
}

impl<T> Vector4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// Calculates the outer product with another vector to give a matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use vqm::Vector4f32;
    /// let row = Vector4f32{x:2.0, y:5.0, z:11.0, t:17.0};
    /// let col = Vector4f32{x:3.0, y:7.0, z:13.0, t:19.0};
    /// let m = col.outer_product(row);
    /// assert_eq!(m, Matrix4x4f32::new([ 6.0,  15.0,  33.0,  51.0,
    ///                                  14.0,  35.0,  77.0, 119.0,
    ///                                  26.0,  65.0, 143.0, 221.0,
    ///                                  38.0,  95.0, 209.0, 323.0]));
    ///```
    #[inline]
    pub fn outer_product(self, row: Vector4<T>) -> Matrix4x4<T> {
        T::m4x4_vector_outer_product(self, row)
    }
}

impl<T> Quaternion<T>
where
    T: Copy + Matrix4x4Math,
{
    /// Quaternion outer product `q * q^T` resulting in a symmetric 4x4 matrix.
    /// ```
    /// # use vqm::{Quaternionf32,Matrix4x4f32};
    /// let q = Quaternionf32::new(2.0, 5.0, 11.0, 17.0);
    ///
    /// let m = q.outer_product();
    ///
    /// assert_eq!(m, Matrix4x4f32::new([ 4.0,  10.0,  22.0,  34.0,
    ///                                  10.0,  25.0,  55.0,  85.0,
    ///                                  22.0,  55.0, 121.0, 187.0,
    ///                                  34.0,  85.0, 187.0, 289.0]));
    /// ```
    #[inline]
    pub fn outer_product(self) -> Matrix4x4<T> {
        T::m4x4_quaternion_outer_product(self)
    }
}

// **** Div ****

impl<T> Div<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    /// Divide a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let r = m / 2.0;
    ///
    /// assert_eq!(r, Matrix4x4f32::new([ 1.0,  8.5, 29.5, 63.5,
    ///                                   2.5,  5.5, 23.5, 54.5,
    ///                                  11.5, 15.5, 20.5, 51.5,
    ///                                  33.5, 36.5, 41.5, 48.5]));
    /// ```
    #[inline]
    fn div(self, other: T) -> Self {
        T::m4x4_div_scalar(self, other)
    }
}

// **** DivAssign ****

impl<T> DivAssign<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// In-place divide a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// m /= 2.0;
    ///
    /// assert_eq!(m, Matrix4x4f32::new([ 1.0,  8.5, 29.5, 63.5,
    ///                                   2.5,  5.5, 23.5, 54.5,
    ///                                  11.5, 15.5, 20.5, 51.5,
    ///                                  33.5, 36.5, 41.5, 48.5]));
    /// ```
    #[inline]
    fn div_assign(&mut self, other: T) {
        *self = *self / other;
    }
}

// **** AsRef ****

impl<T> AsRef<[T; 16]> for Matrix4x4<T> {
    /// Immutable reference to the raw array.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let a: &[f32; 16] = m.as_ref();
    /// assert_eq!(5.0, a[Matrix4x4f32::M21]);
    /// ```
    #[inline]
    fn as_ref(&self) -> &[T; 16] {
        &self.a
    }
}

impl<T> AsMut<[T; 16]> for Matrix4x4<T> {
    /// Mutable reference to the raw array.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// let a: &mut [f32; 16] = m.as_mut();
    /// a[4] = 7.0;
    /// assert_eq!(7.0, m[4]);
    /// ```
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 16] {
        &mut self.a
    }
}

// **** Deref ****

impl<T> Deref for Matrix4x4<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        &self.a
    }
}

impl<T> DerefMut for Matrix4x4<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.a
    }
}

// **** Index ****

impl<T> Index<usize> for Matrix4x4<T> {
    type Output = T;

    /// Access matrix element by index.
    /// ```
    /// # use vqm::Matrix4x4f32;
    ///
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    ///
    /// assert_eq!(m[Matrix4x4f32::M11], 2.0);
    /// assert_eq!(m[Matrix4x4f32::M12], 17.0);
    /// assert_eq!(m[Matrix4x4f32::M13], 59.0);
    /// assert_eq!(m[Matrix4x4f32::M14], 127.0);
    /// assert_eq!(m[Matrix4x4f32::M21], 5.0);
    /// assert_eq!(m[Matrix4x4f32::M22], 11.0);
    /// assert_eq!(m[Matrix4x4f32::M23], 47.0);
    /// assert_eq!(m[Matrix4x4f32::M24], 109.0);
    /// assert_eq!(m[Matrix4x4f32::M31], 23.0);
    /// assert_eq!(m[Matrix4x4f32::M32], 31.0);
    /// assert_eq!(m[Matrix4x4f32::M33], 41.0);
    /// assert_eq!(m[Matrix4x4f32::M34], 103.0);
    /// assert_eq!(m[Matrix4x4f32::M41], 67.0);
    /// assert_eq!(m[Matrix4x4f32::M42], 73.0);
    /// assert_eq!(m[Matrix4x4f32::M43], 83.0);
    /// assert_eq!(m[Matrix4x4f32::M44], 97.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

impl<T> Index<Range<usize>> for Matrix4x4<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: Range<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<RangeFull> for Matrix4x4<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _index: RangeFull) -> &[T] {
        &self.a
    }
}

impl<T> Index<RangeInclusive<usize>> for Matrix4x4<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeInclusive<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<(usize, usize)> for Matrix4x4<T> {
    type Output = T;

    /// Access matrix element by ordered pair (row, column).
    /// ```
    /// # use vqm::Matrix4x4f32;
    ///
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    ///
    /// assert_eq!(m[(0,0)], 2.0);
    /// assert_eq!(m[(0,1)], 17.0);
    /// assert_eq!(m[(0,2)], 59.0);
    /// assert_eq!(m[(0,3)], 127.0);
    /// assert_eq!(m[(1,0)], 5.0);
    /// assert_eq!(m[(1,1)], 11.0);
    /// assert_eq!(m[(1,2)], 47.0);
    /// assert_eq!(m[(1,3)], 109.0);
    /// assert_eq!(m[(2,0)], 23.0);
    /// assert_eq!(m[(2,1)], 31.0);
    /// assert_eq!(m[(2,2)], 41.0);
    /// assert_eq!(m[(2,3)], 103.0);
    /// assert_eq!(m[(3,0)], 67.0);
    /// assert_eq!(m[(3,1)], 73.0);
    /// assert_eq!(m[(3,2)], 83.0);
    /// assert_eq!(m[(3,3)], 97.0);
    /// ```
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < 4 && col < 4, "Matrix index out of bounds: row={row}, col={col}");
        &self.a[col * 4 + row]
    }
}

// **** IndexMut ****

impl<T> IndexMut<usize> for Matrix4x4<T> {
    /// Set matrix element by index.
    /// ```
    /// # use vqm::Matrix4x4f32;
    ///
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    ///
    /// m[Matrix4x4f32::M11] = 3.0;
    /// m[Matrix4x4f32::M12] = 19.0;
    /// m[Matrix4x4f32::M13] = 61.0;
    /// m[Matrix4x4f32::M14] = 131.0;
    /// m[Matrix4x4f32::M21] = 7.0;
    /// m[Matrix4x4f32::M22] = 13.0;
    /// m[Matrix4x4f32::M23] = 53.0;
    /// m[Matrix4x4f32::M24] = 113.0;
    /// m[Matrix4x4f32::M31] = 29.0;
    /// m[Matrix4x4f32::M32] = 37.0;
    /// m[Matrix4x4f32::M33] = 43.0;
    /// m[Matrix4x4f32::M34] = 107.0;
    /// m[Matrix4x4f32::M41] = 71.0;
    /// m[Matrix4x4f32::M42] = 79.0;
    /// m[Matrix4x4f32::M43] = 89.0;
    /// m[Matrix4x4f32::M44] = 101.0;
    ///
    /// assert_eq!(m, Matrix4x4f32::new([  3.0, 19.0, 61.0, 131.0,
    ///                                    7.0, 13.0, 53.0, 113.0,
    ///                                   29.0, 37.0, 43.0, 107.0,
    ///                                   71.0, 79.0, 89.0, 101.0]));
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.a[index]
    }
}

impl<T> IndexMut<Range<usize>> for Matrix4x4<T> {
    #[inline]
    fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<RangeFull> for Matrix4x4<T> {
    #[inline]
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        &mut self.a
    }
}

impl<T> IndexMut<RangeInclusive<usize>> for Matrix4x4<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeInclusive<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix4x4<T> {
    /// Set matrix element by ordered pair (row, column).
    /// ```
    /// # use vqm::Matrix4x4f32;
    ///
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    ///
    /// m[(0,0)] = 3.0;
    /// m[(0,1)] = 19.0;
    /// m[(0,2)] = 61.0;
    /// m[(0,3)] = 131.0;
    /// m[(1,0)] = 7.0;
    /// m[(1,1)] = 13.0;
    /// m[(1,2)] = 53.0;
    /// m[(1,3)] = 113.0;
    /// m[(2,0)] = 29.0;
    /// m[(2,1)] = 37.0;
    /// m[(2,2)] = 43.0;
    /// m[(2,3)] = 107.0;
    /// m[(3,0)] = 71.0;
    /// m[(3,1)] = 79.0;
    /// m[(3,2)] = 89.0;
    /// m[(3,3)] = 101.0;
    ///
    /// assert_eq!(m, Matrix4x4f32::new([  3.0, 19.0, 61.0, 131.0,
    ///                                    7.0, 13.0, 53.0, 113.0,
    ///                                   29.0, 37.0, 43.0, 107.0,
    ///                                   71.0, 79.0, 89.0, 101.0]));
    /// ```
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(row < 4 && col < 4, "Matrix index out of bounds: row={row}, col={col}");
        &mut self.a[col * 4 + row]
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy,
{
    /// Set matrix row from a vector.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4f32};
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// m.set_row(1, Vector4f32::new(7.0, 13.0, 19.0, 29.0));
    /// assert_eq!(Vector4f32{ x: 7.0, y: 13.0, z: 19.0, t: 29.0 }, m.row(1));
    /// ```
    pub fn set_row(&mut self, row: usize, value: Vector4<T>) {
        if row >= 4 {
            return;
        }
        self.a[row] = value.x;
        self.a[row + 4] = value.y;
        self.a[row + 8] = value.z;
        self.a[row + 12] = value.t;
    }

    /// Return matrix row as a vector.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4f32};
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let v = m.row(0);
    ///
    /// assert_eq!(v, Vector4f32{ x: 2.0, y: 17.0, z: 59.0, t:127.0 });
    /// assert_eq!(m.row(1), Vector4f32{ x: 5.0, y: 11.0, z: 47.0, t:109.0 });
    /// assert_eq!(m.row(2), Vector4f32{ x: 23.0, y: 31.0, z: 41.0, t:103.0 });
    /// assert_eq!(m.row(3), Vector4f32{ x: 67.0, y: 73.0, z: 83.0, t:97.0 });
    /// ```
    pub fn row(self, row: usize) -> Vector4<T> {
        let r = row.min(3);
        Vector4 { x: self.a[r], y: self.a[r + 4], z: self.a[r + 8], t: self.a[r + 12] }
    }

    /// Return matrix row as a tuple.
    pub fn row_tuple(self, row: usize) -> (T, T, T, T) {
        let r = row.min(3);
        ( self.a[r], self.a[r + 4], self.a[r + 8], self.a[r + 12] )
    }

    /// Set matrix column from a vector.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4f32};
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// m.set_column(1, Vector4f32::new(7.0, 13.0, 19.0, 29.0));
    /// assert_eq!(Vector4f32{ x: 7.0, y: 13.0, z: 19.0, t: 29.0 }, m.column(1));
    /// ```
    pub fn set_column(&mut self, column: usize, value: Vector4<T>) {
        if column >= 4 {
            return;
        }
        let start = column * 4;
        // Extract a 4-element slice.
        // Because row < 4, start + 4 will never exceed 16.
        let column_slice = &mut self.a[start..start + 4];
        column_slice[0] = value.x;
        column_slice[1] = value.y;
        column_slice[2] = value.z;
        column_slice[3] = value.t;
    }

    /// Return matrix column as a vector.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4f32};
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let v = m.column(0);
    ///
    /// assert_eq!(v, Vector4f32{ x: 2.0, y: 5.0, z: 23.0, t: 67.0 });
    /// assert_eq!(m.column(1), Vector4f32{ x: 17.0, y: 11.0, z: 31.0, t: 73.0 });
    /// assert_eq!(m.column(2), Vector4f32{ x: 59.0, y: 47.0, z: 41.0, t: 83.0 });
    /// assert_eq!(m.column(3), Vector4f32{ x: 127.0, y: 109.0, z: 103.0, t: 97.0 });
    /// ```
    pub fn column(self, column: usize) -> Vector4<T> {
        // Branchless clamp: restricts r to 0..=3
        let base = column.min(3) * 4;
        let chunk = &self.a[base..];
        Vector4 { x: chunk[0], y: chunk[1], z: chunk[2], t: chunk[3] }
    }

    /// Return matrix column as a tuple.
    pub fn column_tuple(self, column: usize) -> (T, T, T, T) {
        // Branchless clamp: restricts r to 0..=3
        let base = column.min(3) * 4;
        let chunk = &self.a[base..];
        (chunk[0], chunk[1], chunk[2], chunk[3])
    }

    /// Return matrix diagonal as an array.
    /// ```
    /// # use vqm::Matrix4x4f32;
    ///
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let a = m.diagonal_as_array();
    ///
    /// assert_eq!(a, [ 2.0, 11.0, 41.0, 97.0 ]);
    /// ```
    pub fn diagonal_as_array(self) -> [T; 4] {
        [self.a[Self::M11], self.a[Self::M22], self.a[Self::M33], self.a[Self::M44]]
    }

    /// Return matrix diagonal as a vector.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4f32};
    ///
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let v = m.diagonal_as_vector();
    ///
    /// assert_eq!(v, Vector4f32{ x: 2.0, y: 11.0, z: 41.0, t: 97.0 });
    /// ```
    pub fn diagonal_as_vector(self) -> Vector4<T> {
        Vector4 { x: self.a[Self::M11], y: self.a[Self::M22], z: self.a[Self::M33], t: self.a[Self::M44] }
    }
}

// **** abs ****

impl<T> Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// Return a copy of the matrix with all elements set to their absolute values.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, -17.0,  59.0,  127.0,
    ///                              5.0, -11.0,  47.0,  109.0,
    ///                             23.0,  31.0, -41.0, -103.0,
    ///                             67.0,  73.0, -83.0,  97.0]);
    /// let n = m.abs();
    ///
    /// assert_eq!(n, Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 11.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 41.0, 103.0,
    ///                                   67.0, 73.0, 83.0,  97.0]));
    /// ```
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        T::m4x4_abs(self)
    }

    /// Set all elements of the matrix to their absolute values.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, -17.0,  59.0,  127.0,
    ///                                  5.0, -11.0,  47.0,  109.0,
    ///                                 23.0,  31.0,  41.0, -103.0,
    ///                                 67.0,  73.0, -83.0,   97.0]);
    /// m.abs_in_place();
    ///
    /// assert_eq!(m, Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                    5.0, 11.0, 47.0, 109.0,
    ///                                   23.0, 31.0, 41.0, 103.0,
    ///                                   67.0, 73.0, 83.0,  97.0]));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = T::m4x4_abs(*self);
        self
    }
}

// **** clamp ****

impl<T> Matrix4x4<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all elements clamped to the specified range.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, -59.0, 127.0,
    ///                              5.0, 11.0,  47.0, 109.0,
    ///                             23.0, 31.0, -41.0, 103.0,
    ///                             67.0, 73.0,  83.0,  97.0]);
    /// let n = m.clamp(7.0, 17.0);
    ///
    /// assert_eq!(n, Matrix4x4f32::new([ 7.0, 17.0,  7.0, 17.0,
    ///                                   7.0, 11.0, 17.0, 17.0,
    ///                                  17.0, 17.0,  7.0, 17.0,
    ///                                  17.0, 17.0, 17.0, 17.0]));
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
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, -59.0, 127.0,
    ///                                  5.0, 11.0,  47.0, 109.0,
    ///                                 23.0, 31.0, -41.0, 103.0,
    ///                                 67.0, 73.0,  83.0,  97.0]);
    /// m.clamp_in_place(7.0, 17.0);
    ///
    /// assert_eq!(m, Matrix4x4f32::new([ 7.0, 17.0,  7.0, 17.0,
    ///                                   7.0, 11.0, 17.0, 17.0,
    ///                                  17.0, 17.0,  7.0, 17.0,
    ///                                  17.0, 17.0, 17.0, 17.0]));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy,
{
    /// Return the transpose of this matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let n = m.transpose();
    ///
    /// assert_eq!(n, Matrix4x4f32::new([  2.0,  5.0, 23.0, 67.0,
    ///                                   17.0, 11.0, 31.0, 73.0,
    ///                                   59.0, 47.0, 41.0, 83.0,
    ///                                  127.0,109.0,103.0, 97.0]));
    /// ```
    #[inline]
    #[must_use]
    pub fn transpose(self) -> Self {
        Self {
            a: [
                self.a[0], self.a[4], self.a[8], self.a[12], //
                self.a[1], self.a[5], self.a[9], self.a[13], //
                self.a[2], self.a[6], self.a[10], self.a[14], //
                self.a[3], self.a[7], self.a[11], self.a[15], //
            ],
        }
    }

    /// Transpose matrix, in-place.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// m.transpose_in_place();
    ///
    /// assert_eq!(m, Matrix4x4f32::new([  2.0,  5.0, 23.0, 67.0,
    ///                                   17.0, 11.0, 31.0, 73.0,
    ///                                   59.0, 47.0, 41.0, 83.0,
    ///                                  127.0,109.0,103.0, 97.0]));
    /// ```
    #[inline]
    pub fn transpose_in_place(&mut self) -> &mut Self {
        *self = self.transpose();
        self
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// Return the adjugate of this matrix, ie the transpose of the cofactor matrix.
    /// Equivalent to the inverse but without dividing by the determinant of the matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::One;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let (n, d) = m.adjugate();
    /// assert_eq!(n, Matrix4x4f32::new([  115992.0, -135144.0,  10428.0, -11076.0,
    ///                                   -105288.0,  159960.0, -46068.0,   7020.0,
    ///                                    -14572.0,  -19820.0,  54882.0, -16926.0,
    ///                                     11588.0,  -10076.0, -19494.0,   7098.0]));
    /// assert_eq!(-945984.0, d);
    /// assert_eq!(-945984.0, m.determinant());
    ///
    /// assert!((n * m / m.determinant()).is_near_identity(1e-5));
    /// assert_eq!(Matrix4x4f32::one(), n*m/m.determinant());
    /// assert_eq!(n * m / d, Matrix4x4f32::new([ 1.0, 0.0, 0.0, 0.0,
    ///                                           0.0, 1.0, 0.0, 0.0,
    ///                                           0.0, 0.0, 1.0, 0.0,
    ///                                           0.0, 0.0, 0.0, 1.0]));
    /// ```
    #[inline]
    pub fn adjugate(self) -> (Self, T) {
        let (adjugate, determinant) = T::m4x4_adjugate(self);
        (adjugate, determinant)
    }

    /// Adjugate matrix, in-place.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// let mut n = m;
    /// n.adjugate_in_place();
    ///
    /// assert_eq!(m.adjugate().0, n);
    /// ```
    #[inline]
    pub fn adjugate_in_place(&mut self) -> &mut Self {
        *self = self.adjugate().0;
        self
    }

    /// Matrix determinant.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let d = m.determinant();
    ///
    /// assert_eq!(-945984.0, d);
    ///
    /// ```
    #[inline]
    pub fn determinant(self) -> T {
        T::m4x4_determinant(self)
    }

    /// Return trace of matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let t = m.trace();
    ///
    /// assert_eq!(t, 151.0);
    /// ```
    #[inline]
    pub fn trace(self) -> T {
        T::m4x4_trace(self)
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy + Matrix4x4Math + FloatCore + MathConstants,
{
    /// Return the inverse of this matrix. Returns self if the determinant is zero.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let n = m.inverse();
    ///
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        let (adjugate, determinant) = T::m4x4_adjugate(self);
        if determinant.abs() < T::EPSILON {
            return self;
        }
        adjugate / determinant
    }

    /// Invert this matrix, in-place. Returns self if the determinant is zero.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                                  5.0, 11.0, 47.0, 109.0,
    ///                                 23.0, 31.0, 41.0, 103.0,
    ///                                 67.0, 73.0, 83.0,  97.0]);
    /// m.invert_in_place();
    /// ```
    #[inline]
    pub fn invert_in_place(&mut self) -> &mut Self {
        let (adjugate, determinant) = T::m4x4_adjugate(*self);
        if determinant.abs() < T::EPSILON {
            return self;
        }
        *self = adjugate / determinant;
        self
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy + Zero + One + Matrix4x4Math + MathConstants + PartialOrd + FloatCore,
{
    /// Return inverse of matrix or `T::zero()` if not invertible.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::Zero;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              2.0, 17.0, 59.0, 127.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let n = m.inverse_or_zero();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(Matrix4x4f32::zero(), n);
    ///
    /// ```
    #[must_use]
    pub fn inverse_or_zero(self) -> Self {
        let (adjugate, determinant) = self.adjugate();
        if determinant.abs() < T::EPSILON {
            return Self::zero();
        }
        adjugate / determinant
    }

    /// Return inverse of matrix or `None` if not invertible.
    /// ```
    /// # use vqm::{Matrix4x4f32};
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              2.0, 17.0, 59.0, 127.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let n = m.try_inverse();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(None, n);
    ///
    /// ```
    pub fn try_inverse(self) -> Option<Self> {
        let (adjugate, determinant) = self.adjugate();
        if determinant.abs() < T::EPSILON {
            return None;
        }
        Some(adjugate / determinant)
    }

    /// Return the sum of all elements of the matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let s = m.sum();
    ///
    /// assert_eq!(s, 895.0);
    /// ```
    #[inline]
    pub fn sum(self) -> T {
        T::m4x4_sum(self)
    }

    /// Return the mean of all elements of the matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let mean = m.mean();
    ///
    /// assert_eq!(mean, 895.0 / 16.0);
    /// ```
    #[inline]
    pub fn mean(self) -> T {
        T::m4x4_mean(self)
    }

    /// Return the product of all elements of the matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let product = m.product();
    ///
    /// assert_eq!(product, 8.510985e24);
    /// ```
    #[inline]
    pub fn product(self) -> T {
        T::m4x4_product(self)
    }

    /// Return the sum of the squares of the trace of the matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// let t = m.trace_sum_squares();
    ///
    /// assert_eq!(t, 2.0 * 2.0 + 11.0 *11.0 + 41.0 * 41.0 + 97.0 * 97.0);
    /// ```
    #[inline]
    pub fn trace_sum_squares(self) -> T {
        T::m4x4_trace_sum_squares(self)
    }
}

// **** Symmetry ****

impl<T> Matrix4x4<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Enforces strict mathematical symmetry on the matrix in-place.
    /// Used so that rounding errors do not erode the symmetry of the matrix.
    /// Formula: `M = (M + Mᵀ) / 2`.
    #[inline]
    pub fn enforce_symmetry(&mut self) {
        let half = T::one() / (T::one() + T::one());

        self.a[Self::M12] = (self.a[Self::M12] + self.a[Self::M21]) * half;
        self.a[Self::M21] = self.a[Self::M12];
        self.a[Self::M13] = (self.a[Self::M13] + self.a[Self::M31]) * half;
        self.a[Self::M31] = self.a[Self::M13];
        self.a[Self::M14] = (self.a[Self::M14] + self.a[Self::M41]) * half;
        self.a[Self::M41] = self.a[Self::M14];

        self.a[Self::M23] = (self.a[Self::M23] + self.a[Self::M32]) * half;
        self.a[Self::M32] = self.a[Self::M23];
        self.a[Self::M24] = (self.a[Self::M24] + self.a[Self::M42]) * half;
        self.a[Self::M42] = self.a[Self::M24];

        self.a[Self::M34] = (self.a[Self::M34] + self.a[Self::M43]) * half;
        self.a[Self::M43] = self.a[Self::M34];
    }
}

// **** Iterators ****

impl<T> Matrix4x4<T> {
    /// Returns an iterator over the rows of the matrix as slices of 4 elements.
    #[inline]
    pub fn rows(&self) -> ChunksExact<'_, T> {
        self.chunks_exact(4)
    }
}

impl<T> Matrix4x4<T> {
    /// Returns an iterator over the rows of the matrix as mutable slices of 4 elements.
    #[inline]
    pub fn rows_mut(&mut self) -> ChunksExactMut<'_, T> {
        self.chunks_exact_mut(4)
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy,
{
    /// Consumes the matrix and returns an array of its 4 rows.
    #[inline]
    pub fn into_rows(self) -> [[T; 4]; 4] {
        // Build the nested 2D array matrix safely in a single unrolled pass
        core::array::from_fn(|r| core::array::from_fn(|c| self.a[r * 4 + c]))
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy,
{
    /// Returns an iterator over the columns of the matrix as owned 4-element arrays.
    #[inline]
    pub fn cols(&self) -> impl Iterator<Item = [T; 4]> {
        // Create an iterator over the column indices (0, 1, 2, 3)
        (0..4).map(|c| {
            // Collect the strided elements for the current column
            [self.a[c], self.a[c + 4], self.a[c + 8], self.a[c + 12]]
        })
    }
}

impl<'a, T> IntoIterator for &'a Matrix4x4<T> {
    type Item = &'a [T];
    type IntoIter = ChunksExact<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages the Deref trait automatically to get 4-element rows
        self.chunks_exact(4)
    }
}

impl<'a, T> IntoIterator for &'a mut Matrix4x4<T> {
    type Item = &'a mut [T];
    type IntoIter = ChunksExactMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages the DerefMut trait automatically.
        self.chunks_exact_mut(4)
    }
}

impl<T> IntoIterator for Matrix4x4<T>
where
    T: Copy,
{
    type Item = [T; 4];
    type IntoIter = core::array::IntoIter<[T; 4], 4>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Construct the 4 rows of 4 elements safely in one direct pass
        let rows = core::array::from_fn(|r| core::array::from_fn(|c| self.a[r * 4 + c]));
        rows.into_iter()
    }
}

// **** Column Iterators ****

impl<T> Matrix4x4<T> {
    /// Exposes the matrix as a read-only reference to 4 contiguous columns.
    /// Each sub-array `[T; 4]` represents one full column in memory.
    #[inline]
    pub fn columns(&self) -> &[[T; 4]] {
        // remainder is empty.
        let (chunks, _remainder) = self.a.as_chunks::<4>();
        chunks
    }

    /// Exposes the matrix as a mutable reference to 4 contiguous columns.
    /// Each sub-array `[T; 4]` represents one full column in memory.
    #[inline]
    pub fn columns_mut(&mut self) -> &mut [[T; 4]] {
        // remainder is empty.
        let (chunks, _remainder) = self.a.as_chunks_mut::<4>();
        chunks
    }

    #[inline]
    pub fn iter_columns(&self) -> Matrix4x4Columns<'_, T> {
        let (chunks, _remainder) = self.a.as_chunks::<4>();
        // remainder is empty.
        Matrix4x4Columns { inner: chunks.iter() }
    }

    #[inline]
    pub fn iter_columns_mut(&mut self) -> Matrix4x4ColumnsMut<'_, T> {
        // remainder is empty.
        let (chunks, _remainder) = self.a.as_chunks_mut::<4>();
        Matrix4x4ColumnsMut { inner: chunks.iter_mut() }
    }
}

// **** Iterator Pairs ****

/// A custom iterator over the read-only columns of a 4x4 matrix.
#[derive(Debug, Default)]
pub struct Matrix4x4Columns<'a, T> {
    inner: Iter<'a, [T; 4]>,
}

impl<'a, T> Iterator for Matrix4x4Columns<'a, T> {
    type Item = &'a [T; 4];

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
impl<T> ExactSizeIterator for Matrix4x4Columns<'_, T> {}

impl<T> DoubleEndedIterator for Matrix4x4Columns<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

/// A custom iterator over the mutable columns of a 4x4 matrix.
#[derive(Debug, Default)]
pub struct Matrix4x4ColumnsMut<'a, T> {
    inner: IterMut<'a, [T; 4]>,
}

impl<'a, T> Iterator for Matrix4x4ColumnsMut<'a, T> {
    type Item = &'a mut [T; 4];

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
impl<T> ExactSizeIterator for Matrix4x4ColumnsMut<'_, T> {}

impl<T> DoubleEndedIterator for Matrix4x4ColumnsMut<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

// **** From ****

// **** From Array ****

impl<T> From<[T; 16]> for Matrix4x4<T> {
    /// Matrix from 1D array.
    #[inline]
    fn from(input: [T; 16]) -> Self {
        Self { a: input }
    }
}

// **** From Matrix ****

impl<T: Copy + Zero> From<Matrix2x2<T>> for Matrix4x4<T> {
    /// Matrix4x4 from Matrix2x2.
    /// ```
    /// # use vqm::{Matrix2x2f32,Matrix4x4f32};
    /// let m2 = Matrix2x2f32::new([ 2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let n2 = Matrix2x2f32::new([ 3.0, 19.0,
    ///                              7.0, 13.0]);
    /// let m3: Matrix4x4f32 = m2.into();
    /// let n3 = Matrix4x4f32::from(m2);
    ///
    /// assert_eq!(m3, Matrix4x4f32::new([ 2.0, 17.0, 0.0, 0.0,
    ///                                    5.0, 11.0, 0.0, 0.0,
    ///                                    0.0,  0.0, 0.0, 0.0,
    ///                                    0.0,  0.0, 0.0, 0.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix2x2<T>) -> Self {
        Self { a: [
            m[0],      m[1],      T::zero(), T::zero(),
            m[2],      m[3],      T::zero(), T::zero(),
            T::zero(), T::zero(), T::zero(), T::zero(),
            T::zero(), T::zero(), T::zero(), T::zero(),
        ] }
    }
}

impl<T: Copy> From<Matrix4x4<T>> for Matrix2x2<T> {
    /// Matrix2x2 from Matrix4x4. Takes top left of m4x4, discarding other values.
    /// ```
    /// # use vqm::{Matrix2x2f32,Matrix4x4f32};
    /// let m2 = Matrix2x2f32::new([ 2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let m4 = Matrix4x4f32::new([ 2.0, 17.0, 59.0, 127.0,
    ///                              5.0, 11.0, 47.0, 109.0,
    ///                             23.0, 31.0, 41.0, 103.0,
    ///                             67.0, 73.0, 83.0,  97.0]);
    /// assert_eq!(m2, Matrix2x2f32::from(m4));
    /// ```
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix4x4<T>) -> Self {
        Self { a: [
            m.a[0], m.a[1],
            m.a[4], m.a[5]
        ] }
    }
}

impl<T: Copy + Zero> From<Matrix3x3<T>> for Matrix4x4<T> {
    /// Matrix4x4 from Matrix3x3.
    /// ```
    /// # use vqm::{Matrix3x3f32, Matrix4x4f32};
    /// let m3x3 = Matrix3x3f32::new([ 2.0, 17.0, 59.0,
    ///                                5.0, 11.0, 47.0,
    ///                               23.0, 31.0, 41.0]);
    /// let m4x4 = Matrix4x4f32::from(m3x3);
    /// assert_eq!(m4x4, Matrix4x4f32::new([ 2.0, 17.0, 59.0, 0.0,
    ///                                      5.0, 11.0, 47.0, 0.0,
    ///                                     23.0, 31.0, 41.0, 0.0,
    ///                                      0.0,  0.0,  0.0, 0.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix3x3<T>) -> Self {
        Self { a: [
            m[0],      m[1],      m[2],      T::zero(),
            m[3],      m[4],      m[5],      T::zero(),
            m[6],      m[7],      m[8],      T::zero(),
            T::zero(), T::zero(), T::zero(), T::zero(),
        ] }
    }
}

impl<T: Copy> From<Matrix4x4<T>> for Matrix3x3<T> {
    /// Matrix3x3 from Matrix4x4. Takes top left of m4x4, discarding other values.
    /// ```
    /// # use vqm::{Matrix3x3f32,Matrix4x4f32};
    /// let m3x3 = Matrix3x3f32::new([ 2.0, 17.0, 59.0,
    ///                                 5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// let m4x4 = Matrix4x4f32::new([ 2.0, 17.0, 59.0, 127.0,
    ///                                 5.0, 11.0, 47.0, 109.0,
    ///                                23.0, 31.0, 41.0, 103.0,
    ///                                67.0, 73.0, 83.0,  97.0]);
    /// assert_eq!(m3x3, Matrix3x3f32::from(m4x4));
    /// ```
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix4x4<T>) -> Self {
        Self { a: [
            m.a[0], m.a[1], m.a[2],
            m.a[4], m.a[5], m.a[6],
            m.a[8], m.a[9], m.a[10]
        ] }
    }
}
