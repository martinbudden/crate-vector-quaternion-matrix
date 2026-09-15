use core::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, RangeFull,
    RangeInclusive, Sub, SubAssign,
};
use core::slice::{ChunksExact, ChunksExactMut, Iter, IterMut};
use num_traits::{ConstOne, ConstZero, MulAdd, MulAddAssign, One, Zero, float::FloatCore};
#[cfg(feature = "storage")]
use sequential_storage::map::PostcardValue;
#[cfg(feature = "serde")]
use {
    postcard::experimental::max_size::MaxSize,
    serde::{Deserialize, Serialize},
};

use crate::{MathConstants, Matrix2x2Math, Vector2};

/// 2x2 matrix of `f32` values<br>
pub type Matrix2x2f32 = Matrix2x2<f32>;
/// 2x2 matrix of `f64` values<br><br>
pub type Matrix2x2f64 = Matrix2x2<f64>;

// **** Define ****

/// `Matrix2x2<T>`: 2x2 Matrix of type `T`.<br>
/// Aliases `Matrix2x2f32` and `Matrix2x2f64` are provided.<br>
/// Internal implementation is using a flattened 1-dimensional array: an array of 4 elements stored in column-major order.
/// That is the element `m[row][col]` is at array position `[col * 2 + row]`.<br><br>
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", allow(clippy::unsafe_derive_deserialize))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize, MaxSize))]
#[repr(C, align(16))]
pub struct Matrix2x2<T> {
    // Flattened 2x2 matrix: 4 elements in column-major order
    pub(crate) a: [T; 4],
}

#[cfg(feature = "storage")]
impl<T> PostcardValue<'_> for Matrix2x2<T> where T: Serialize + for<'de> Deserialize<'de> {}

/// Constants to index matrix elements.
impl<T> Matrix2x2<T> {
    pub const SIZE: usize = 4;
    pub const ROW_COUNT: usize = 2;
    pub const COL_COUNT: usize = 2;
    // Column 1
    pub const M11: usize = 0;
    pub const M21: usize = 1;
    // Column 2
    pub const M12: usize = 2;
    pub const M22: usize = 3;
}

// **** New ****

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Create a matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0,  17.0,
    ///                              5.0,  11.0]);
    /// assert_eq!(m, Matrix2x2f32::new([  2.0,  17.0,
    ///                                    5.0,  11.0]));
    /// ```
    #[inline]
    pub const fn new(a: [T; 4]) -> Self {
        Self::from_row_array(a)
    }
}

// **** Other constructors ****

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Create a matrix with all its elements set to a single value.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from_element(2.0);
    /// assert_eq!(m, Matrix2x2f32::new([  2.0, 2.0,
    ///                                    2.0, 2.0]));
    /// ```
    #[inline]
    pub const fn from_element(value: T) -> Self {
        Self { a: [value; 4] }
    }

    /// Matrix from array of row vectors.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let m = Matrix2x2f32::from_rows([ Vector2f32::new(2.0, 17.0),
    ///                                   Vector2f32::new(5.0, 11.0) ]);
    /// assert_eq!(m, Matrix2x2f32::new([  2.0, 17.0,
    ///                                    5.0, 11.0 ]));
    /// ```
    #[inline]
    pub const fn from_rows(v: [Vector2<T>; 2]) -> Self {
        Self {
            a: [
                v[0].x, v[1].x, //
                v[0].y, v[1].y, //
            ],
        }
    }

    /// Matrix from array of column vectors.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let m = Matrix2x2f32::from_columns([ Vector2f32::new(2.0, 17.0),
    ///                                      Vector2f32::new(5.0, 11.0) ]);
    /// assert_eq!(m, Matrix2x2f32::new([  2.0,   5.0,
    ///                                   17.0,  11.0 ]));
    /// ```
    #[inline]
    pub const fn from_columns(v: [Vector2<T>; 2]) -> Self {
        Self {
            a: [
                v[0].x, v[0].y, //
                v[1].x, v[1].y, //
            ],
        }
    }

    /// Matrix from 1D row array.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from_row_array([  2.0, 17.0,
    ///                                         5.0, 11.0]);
    /// assert_eq!(m, Matrix2x2f32::new([  2.0, 17.0,
    ///                                    5.0, 11.0 ]));
    /// ```
    #[inline]
    pub const fn from_row_array(a: [T; 4]) -> Self {
        Self {
            a: [
                a[0], a[2], //
                a[1], a[3], //
            ],
        }
    }

    /// Matrix from 1D column array.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from_column_array([  2.0, 17.0,
    ///                                            5.0, 11.0]);
    /// assert_eq!(m, Matrix2x2f32::new([  2.0,   5.0,
    ///                                   17.0,  11.0 ]));
    /// ```
    #[inline]
    pub const fn from_column_array(a: [T; 4]) -> Self {
        Self { a }
    }

    /// Matrix from 2D row array.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from_2d_row_array([[  2.0, 17.0],
    ///                                          [  5.0, 11.0]]);
    /// assert_eq!(m, Matrix2x2f32::new([  2.0, 17.0,
    ///                                    5.0, 11.0 ]));
    /// ```
    #[inline]
    pub const fn from_2d_row_array(a: [[T; 2]; 2]) -> Self {
        Self {
            a: [
                a[0][0], a[1][0], //
                a[0][1], a[1][1], //
            ],
        }
    }

    /// Matrix from 2D column array.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from_2d_column_array([[  2.0, 17.0],
    ///                                             [  5.0, 11.0]]);
    /// assert_eq!(m, Matrix2x2f32::new([  2.0,   5.0,
    ///                                   17.0,  11.0 ]));
    /// ```
    #[inline]
    pub const fn from_2d_column_array(a: [[T; 2]; 2]) -> Self {
        Self {
            a: [
                a[0][0], a[0][1], //
                a[1][0], a[1][1], //
            ],
        }
    }

    /// Try to create a matrix from a row slice.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let valid_data = [1.0; 4];
    /// let invalid_data = [1.0; 3];
    /// let Some(m) = Matrix2x2f32::try_from_row_slice(&valid_data) else {
    ///     panic!("Expected Some(Matrix2x2), but got None");
    /// };
    /// assert_eq!(1.0, m[0]);
    /// let None = Matrix2x2f32::try_from_row_slice(&invalid_data) else {
    ///     panic!("Expected None for invalid data, but got Some");
    /// };
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn try_from_row_slice(slice: &[T]) -> Option<Self> {
        if slice.len() != 4 {
            return None;
        }
        let a = [
            slice[0], slice[2],
            slice[1], slice[3],
        ];
        Some(Self { a })
    }

    /// Try to create a matrix from a column slice.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let valid_data = [2.0; 4];
    /// let invalid_data = [2.0; 3];
    /// let Some(m) = Matrix2x2f32::try_from_column_slice(&valid_data) else {
    ///     panic!("Expected Some(Matrix2x2), but got None");
    /// };
    /// assert_eq!(2.0, m[0]);
    /// let None = Matrix2x2f32::try_from_column_slice(&invalid_data) else {
    ///     panic!("Expected None for invalid data, but got Some");
    /// };
    /// ```
    #[inline]
    pub fn try_from_column_slice(slice: &[T]) -> Option<Self> {
        if slice.len() != 4 {
            return None;
        }
        let mut a = [slice[0]; 4];
        a.copy_from_slice(&slice[0..4]);
        Some(Self { a })
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + ConstZero,
{
    /// Create a matrix with the diagonal set to a single value.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from_diagonal_element(2.0);
    /// assert_eq!(m, Matrix2x2f32::new([ 2.0, 0.0,
    ///                                   0.0, 2.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_element(value: T) -> Self {
        Self {
            a: [
                value,   T::ZERO,
                T::ZERO, value,
            ]
        }
    }
    /// Create a matrix with the diagonal set to an array.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from_diagonal_array([2.0, 3.0]);
    /// assert_eq!(m, Matrix2x2f32::new([ 2.0, 0.0,
    ///                                   0.0, 3.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_array(a: [T;2]) -> Self {
        Self {
            a: [
                a[0],    T::ZERO,
                T::ZERO, a[1],
            ]
        }
    }
    /// Create a matrix with the diagonal set to a vector.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let m = Matrix2x2f32::from_diagonal_vector(Vector2f32::new(2.0, 3.0));
    /// assert_eq!(m, Matrix2x2f32::new([ 2.0, 0.0,
    ///                                   0.0, 3.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_vector(v: Vector2<T>) -> Self {
        Self {
            a: [
                v.x,     T::ZERO,
                T::ZERO, v.y,
            ]
        }
    }
}

// **** Zero ****

impl<T> Zero for Matrix2x2<T>
where
    T: Copy + Zero + PartialEq + Matrix2x2Math,
{
    /// Zero matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::{Zero,zero};
    /// let z = Matrix2x2f32::zero();
    ///
    /// assert_eq!(z, Matrix2x2f32::new([ 0.0, 0.0,
    ///                                   0.0, 0.0]));
    /// assert!(z.is_zero());
    /// ```
    #[inline]
    fn zero() -> Self {
        Self { a: [T::zero(); 4] }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<T> ConstZero for Matrix2x2<T>
where
    T: Copy + ConstZero + PartialEq + Matrix2x2Math,
{
    /// Const zero matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::{zero,Zero,ConstZero};
    /// let m = Matrix2x2f32::ZERO;
    /// assert!(m.is_zero());
    /// ```
    const ZERO: Self = Self { a: [T::ZERO; 4] };
}

impl<T> Matrix2x2<T>
where
    T: Copy + FloatCore,
{
    /// Return true if matrix is near zero.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::Zero;
    /// let z = Matrix2x2f32::zero();
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

impl<T> One for Matrix2x2<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix2x2Math,
{
    /// Identity matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::One;
    /// let i = Matrix2x2f32::one();
    ///
    /// assert_eq!(i, Matrix2x2f32::new([ 1.0, 0.0,
    ///                                   0.0, 1.0]));
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

impl<T> ConstOne for Matrix2x2<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix2x2Math,
{
    /// Const identity matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::ConstOne;
    /// let i = Matrix2x2f32::ONE;
    ///
    /// assert_eq!(i, Matrix2x2f32::new([ 1.0, 0.0,
    ///                                   0.0, 1.0]));
    /// ```
    #[rustfmt::skip]
    const ONE: Self = Self {
        a: [
            T::ONE,  T::ZERO,
            T::ZERO, T::ONE,
        ]
    };
}

impl<T> Matrix2x2<T>
where
    T: Copy + Zero + One,
{
    /// Identity matrix.
    /// Alias for `one()` that does not require `num_traits::One`.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let i = Matrix2x2f32::identity();
    ///
    /// assert_eq!(i, Matrix2x2f32::new([ 1.0, 0.0,
    ///                                   0.0, 1.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        Self {
            a: [
                T::one(),  T::zero(),
                T::zero(), T::one(),
            ],
        }
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + One + FloatCore,
{
    /// Return true if matrix is near identity.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::One;
    /// let i = Matrix2x2f32::one();
    /// assert!(i.is_near_identity(1e-5));
    /// ```
    pub fn is_near_identity(self, epsilon: T) -> bool {
        if self.a[Self::M21].abs() > epsilon || self.a[Self::M12].abs() > epsilon {
            return false;
        }
        if (self.a[Self::M11] - T::one()).abs() > epsilon || (self.a[Self::M22] - T::one()).abs() > epsilon {
            return false;
        }
        true
    }
}

// **** Neg ****

impl<T> Neg for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Negate matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// m = - m;
    ///
    /// assert_eq!(m, Matrix2x2f32::new([ -2.0, -17.0,
    ///                                   -5.0, -11.0]));
    /// ```
    #[inline]
    fn neg(self) -> Self {
        T::m2x2_neg(self)
    }
}

// **** Add ****

impl<T> Add for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Add two matrices.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let n = Matrix2x2f32::new([  3.0, 19.0,
    ///                              7.0, 13.0]);
    /// let r = m + n;
    ///
    /// assert_eq!(r, Matrix2x2f32::new([  5.0, 36.0,
    ///                                   12.0, 24.0]));
    ///
    /// # use num_traits::Zero;
    ///
    /// let z = Matrix2x2f32::zero();
    /// let r2 = m + z;
    ///
    /// assert_eq!(r2, m);
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        T::m2x2_add(self, other)
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + One + FloatCore + AddAssign,
{
    /// Add a diagonal matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let n = Matrix2x2f32::new([  3.0,  0.0,
    ///                              0.0, 13.0]);
    /// let s = m + n;
    /// let r = m.add_diagonal(n);
    ///
    /// assert_eq!(r, s);
    /// assert_eq!(r, Matrix2x2f32::new([  5.0, 17.0,
    ///                                    5.0, 24.0]));
    /// ```
    #[must_use]
    pub fn add_diagonal(self, other: Self) -> Self {
        let mut ret = self.a;
        ret[Self::M11] += other.a[Self::M11];
        ret[Self::M22] += other.a[Self::M22];
        Self { a: ret }
    }

    /// Add a diagonal matrix, in-place.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// let n = Matrix2x2f32::new([  3.0,  0.0,
    ///                              0.0, 13.0]);
    /// let s = m + n;
    /// m.add_diagonal_in_place(n);
    ///
    /// assert_eq!(m, s);
    /// assert_eq!(m, Matrix2x2f32::new([  5.0, 17.0,
    ///                                    5.0, 24.0]));
    /// ```
    pub fn add_diagonal_in_place(&mut self, other: Self) {
        self.a[Self::M11] += other.a[Self::M11];
        self.a[Self::M22] += other.a[Self::M22];
    }

    /// Add a scalar that represents a diagonal matrix.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let s = 3.0;
    /// let n = Matrix2x2f32::new([  3.0,  0.0,
    ///                              0.0,  3.0]);
    /// let p = m + n;
    /// let r = m.add_diagonal_scalar(s);
    ///
    /// assert_eq!(r, p);
    /// assert_eq!(r, Matrix2x2f32::new([  5.0, 17.0,
    ///                                    5.0, 14.0]));
    /// ```
    #[must_use]
    pub fn add_diagonal_scalar(self, other: T) -> Self {
        let mut ret = self.a;
        ret[Self::M11] += other;
        ret[Self::M22] += other;
        Self { a: ret }
    }

    /// Add a scalar that represents a diagonal matrix, in-place.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// let s = 3.0;
    /// let n = Matrix2x2f32::new([  3.0,  0.0,
    ///                              0.0,  3.0]);
    /// let p = m + n;
    /// m.add_diagonal_scalar_in_place(s);
    ///
    /// assert_eq!(m, p);
    /// assert_eq!(m, Matrix2x2f32::new([  5.0, 17.0,
    ///                                    5.0, 14.0]));
    /// ```
    pub fn add_diagonal_scalar_in_place(&mut self, other: T) {
        self.a[Self::M11] += other;
        self.a[Self::M22] += other;
    }

    /// Add a vector that represents a diagonal matrix.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let v = Vector2f32{x:3.0, y:13.0};
    /// let n = Matrix2x2f32::new([  3.0,  0.0,
    ///                              0.0, 13.0]);
    /// let p = m + n;
    /// let r = m.add_diagonal_vector(v);
    ///
    /// assert_eq!(r, p);
    /// assert_eq!(r, Matrix2x2f32::new([  5.0, 17.0,
    ///                                    5.0, 24.0]));
    /// ```
    #[must_use]
    pub fn add_diagonal_vector(self, other: Vector2<T>) -> Self {
        let mut ret = self.a;
        ret[Self::M11] += other.x;
        ret[Self::M22] += other.y;
        Self { a: ret }
    }

    /// Add a vector that represents a diagonal matrix, in-place.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// let v = Vector2f32{x:3.0, y:13.0};
    /// let n = Matrix2x2f32::new([  3.0,  0.0,
    ///                              0.0, 13.0]);
    /// let p = m + n;
    /// m.add_diagonal_vector_in_place(v);
    ///
    /// assert_eq!(m, p);
    /// assert_eq!(m, Matrix2x2f32::new([  5.0, 17.0,
    ///                                    5.0, 24.0]));
    /// ```
    pub fn add_diagonal_vector_in_place(&mut self, other: Vector2<T>) {
        self.a[Self::M11] += other.x;
        self.a[Self::M22] += other.y;
    }

    /// Add an array that represents a diagonal matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let a = [3.0, 13.0 ];
    /// let n = Matrix2x2f32::new([  a[0], 0.0,
    ///                              0.0,  a[1]]);
    /// let p = m + n;
    /// let r = m.add_diagonal_array(a);
    ///
    /// assert_eq!(r, p);
    /// assert_eq!(r, Matrix2x2f32::new([  5.0, 17.0,
    ///                                    5.0, 24.0]));
    /// ```
    #[must_use]
    pub fn add_diagonal_array(self, other: [T; 2]) -> Self {
        let mut ret = self.a;
        ret[Self::M11] += other[0];
        ret[Self::M22] += other[1];
        Self { a: ret }
    }
    /// Add an array that represents a diagonal matrix, in-place.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// let a = [3.0, 13.0 ];
    /// let n = Matrix2x2f32::new([  a[0], 0.0,
    ///                              0.0,  a[1]]);
    /// let p = m + n;
    /// m.add_diagonal_array_in_place(a);
    ///
    /// assert_eq!(m, p);
    /// assert_eq!(m, Matrix2x2f32::new([  5.0, 17.0,
    ///                                    5.0, 24.0]));
    /// ```
    pub fn add_diagonal_array_in_place(&mut self, other: [T; 2]) {
        self.a[Self::M11] += other[0];
        self.a[Self::M22] += other[1];
    }
}

// **** AddAssign ****

impl<T> AddAssign for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Add one matrix to another.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// let n = Matrix2x2f32::new([  3.0, 19.0,
    ///                              7.0, 13.0]);
    /// m += n;
    ///
    /// assert_eq!(m, Matrix2x2f32::new([  5.0, 36.0,
    ///                                   12.0, 24.0]));
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

impl<T> MulAdd<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Multiply matrix by constant and add another matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::MulAdd;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let n = Matrix2x2f32::new([  3.0, 19.0,
    ///                              7.0, 13.0]);
    /// let k = 137.0;
    /// let r = m.mul_add(k, n);
    ///
    /// assert_eq!(r, Matrix2x2f32::new([  277.0,  2348.0,
    ///                                    692.0,  1520.0]));
    /// ```
    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::m2x2_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

impl<T> MulAddAssign<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Multiply matrix by constant and add another matrix in place.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::MulAddAssign;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// let n = Matrix2x2f32::new([  3.0, 19.0,
    ///                              7.0, 13.0]);
    /// let k = 137.0;
    /// m.mul_add_assign(k, n);
    ///
    /// assert_eq!(m, Matrix2x2f32::new([  277.0,  2348.0,
    ///                                    692.0,  1520.0]));
    /// ```
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

impl<T> Sub for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Subtract two matrices.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let n = Matrix2x2f32::new([  3.0, 13.0,
    ///                              7.0, 19.0]);
    /// let r = m - n;
    ///
    /// assert_eq!(r, Matrix2x2f32::new([  -1.0,  4.0,
    ///                                    -2.0, -8.0]));
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

impl<T> SubAssign for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Subtract one matrix from another.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// let n = Matrix2x2f32::new([  3.0, 13.0,
    ///                              7.0, 19.0]);
    /// m -= n;
    ///
    /// assert_eq!(m, Matrix2x2f32::new([  -1.0,  4.0,
    ///                                    -2.0, -8.0]));
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Pre-multiply ****

impl Mul<Matrix2x2<f32>> for f32 {
    type Output = Matrix2x2<f32>;

    /// Pre-multiply a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let r = 2.0 * m;
    ///
    /// assert_eq!(r, Matrix2x2f32::new([  4.0, 34.0,
    ///                                   10.0, 22.0]));
    /// ```
    #[inline]
    fn mul(self, other: Matrix2x2<f32>) -> Matrix2x2<f32> {
        f32::m2x2_mul_scalar(other, self)
    }
}

impl Mul<Matrix2x2<f64>> for f64 {
    type Output = Matrix2x2<f64>;
    #[inline]
    fn mul(self, other: Matrix2x2<f64>) -> Matrix2x2<f64> {
        f64::m2x2_mul_scalar(other, self)
    }
}

// **** Mul Scalar ****

impl<T> Mul<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Multiply a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let r = m * 2.0;
    ///
    /// assert_eq!(r, Matrix2x2f32::new([  4.0, 34.0,
    ///                                   10.0, 22.0]));
    /// ```
    #[inline]
    fn mul(self, other: T) -> Self {
        T::m2x2_mul_scalar(self, other)
    }
}

// **** MulAssign ****

impl<T> MulAssign<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// In-place multiply a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// m *= 2.0;
    ///
    /// assert_eq!(m, Matrix2x2f32::new([  4.0, 34.0,
    ///                                   10.0, 22.0]));
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

impl<T> Mul<Vector2<T>> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Vector2<T>;

    /// Multiply a vector by a matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use vqm::Vector2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// let v = Vector2f32{x:3.0, y:7.0};
    /// let r = m * v;
    ///
    /// assert_eq!(r, Vector2f32{x:2.0*3.0 + 17.0*7.0, y:5.0*3.0 + 11.0*7.0});
    /// ```
    #[inline]
    fn mul(self, other: Vector2<T>) -> Vector2<T> {
        T::m2x2_mul_vector(self, other)
    }
}

#[cfg(not(feature = "uom"))]
impl<T> Mul<Matrix2x2<T>> for Vector2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Pre-multiply a vector by a matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use vqm::Vector2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// let v = Vector2f32{x:3.0, y:7.0};
    /// let r = v * m;
    ///
    /// assert_eq!(r, Vector2f32{x:3.0*2.0 + 7.0*5.0, y:3.0*17.0 + 7.0*11.0});
    /// ```
    #[inline]
    fn mul(self, other: Matrix2x2<T>) -> Self {
        T::m2x2_vector_mul(self, other)
    }
}

// **** Mul ****

impl<T> Mul<Matrix2x2<T>> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Multiply two matrices.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let n = Matrix2x2f32::new([  3.0, 13.0,
    ///                              7.0, 19.0]);
    /// let r = m * n;
    ///
    /// assert_eq!(r, Matrix2x2f32::new([
    ///    2.0 * 3.0 + 17.0 * 7.0,  2.0 * 13.0 + 17.0 * 19.0,
    ///    5.0 * 3.0 + 11.0 * 7.0,  5.0 * 13.0 + 11.0 * 19.0,
    /// ]));
    ///
    /// # use num_traits::One;
    ///
    /// let i = Matrix2x2f32::one();
    /// let r2 = m * i;
    ///
    /// assert_eq!(r2, m);
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        T::m2x2_mul(self, other)
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + One + FloatCore,
{
    /// Multiply by a diagonal matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let n = Matrix2x2f32::new([  3.0,  0.0,
    ///                              0.0, 13.0]);
    /// let r = m.mul_diagonal(n);
    /// let s = m * n;
    ///
    /// assert_eq!(r, s);
    /// assert_eq!(r, Matrix2x2f32::new([  6.0, 221.0,
    ///                                   15.0, 143.0]));
    /// ```
    #[must_use]
    pub fn mul_diagonal(self, other: Self) -> Self {
        let ret = [
            self.a[Self::M11] * other.a[Self::M11],
            self.a[Self::M21] * other.a[Self::M11],
            self.a[Self::M12] * other.a[Self::M22],
            self.a[Self::M22] * other.a[Self::M22],
        ];
        Self { a: ret }
    }

    /// Multiply by a vector that represents a diagonal matrix.
    /// ```
    /// # use vqm::{Matrix2x2f32, Vector2f32};
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let v = Vector2f32 { x: 3.0, y: 13.0 };
    /// let n = Matrix2x2f32::new([  v.x, 0.0,
    ///                              0.0, v.y]);
    /// let r = m.mul_diagonal_vector(v);
    /// let s = m * n;
    /// let t = m.mul_diagonal(n);
    ///
    /// assert_eq!(r, s);
    /// assert_eq!(r, t);
    /// assert_eq!(r, Matrix2x2f32::new([  6.0, 221.0,
    ///                                   15.0, 143.0]));
    /// ```
    #[must_use]
    pub fn mul_diagonal_vector(self, other: Vector2<T>) -> Self {
        let ret = [
            self.a[Self::M11] * other.x,
            self.a[Self::M21] * other.x,
            self.a[Self::M12] * other.y,
            self.a[Self::M22] * other.y,
        ];
        Self { a: ret }
    }

    /// Multiply by an array that represents a diagonal matrix.
    /// # Example
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let a = [ 3.0, 13.0 ];
    /// let n = Matrix2x2f32::new([  a[0], 0.0,
    ///                              0.0, a[1]]);
    /// let r = m.mul_diagonal_array(a);
    /// let s = m * n;
    /// let t = m.mul_diagonal(n);
    ///
    /// assert_eq!(r, s);
    /// assert_eq!(r, t);
    /// assert_eq!(r, Matrix2x2f32::new([  6.0, 221.0,
    ///                                   15.0, 143.0]));
    /// ```
    #[must_use]
    pub fn mul_diagonal_array(self, other: [T; 2]) -> Self {
        let ret = [
            self.a[Self::M11] * other[0],
            self.a[Self::M21] * other[0],
            self.a[Self::M12] * other[1],
            self.a[Self::M22] * other[1],
        ];
        Self { a: ret }
    }
}

impl<T> MulAssign<Matrix2x2<T>> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Multiply one matrix by another.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// let n = Matrix2x2f32::new([  3.0, 13.0,
    ///                              7.0, 19.0]);
    /// m *= n;
    ///
    /// assert_eq!(m, Matrix2x2f32::new([
    ///    2.0 * 3.0 + 17.0 * 7.0,  2.0 * 13.0 + 17.0 * 19.0,
    ///    5.0 * 3.0 + 11.0 * 7.0,  5.0 * 13.0 + 11.0 * 19.0,
    /// ]));
    ///
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Matrix2x2<T>) {
        *self = *self * other;
    }
}

// **** Outer Product ****

impl<T> Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Calculates the outer product of a column vector and a row vector to give a matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use vqm::Vector2f32;
    /// let row = Vector2f32{x:2.0, y:5.0};
    /// let col = Vector2f32{x:3.0, y:7.0};
    /// let m = Matrix2x2f32::outer_product(col, row);
    /// assert_eq!(m, Matrix2x2f32::new([ 6.0,  15.0,
    ///                                  14.0,  35.0]));
    ///```
    #[inline]
    pub fn outer_product(col: Vector2<T>, row: Vector2<T>) -> Self {
        T::m2x2_vector_outer_product(col, row)
    }
}

impl<T> Vector2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Calculates the outer product with another vector to give a matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use vqm::Vector2f32;
    /// let row = Vector2f32{x:2.0, y:5.0};
    /// let col = Vector2f32{x:3.0, y:7.0};
    /// let m = col.outer_product(row);
    /// assert_eq!(m, Matrix2x2f32::new([ 6.0,  15.0,
    ///                                  14.0,  35.0]));
    ///```
    #[inline]
    pub fn outer_product(self, row: Vector2<T>) -> Matrix2x2<T> {
        T::m2x2_vector_outer_product(self, row)
    }
}

// **** Div ****

impl<T> Div<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Divide a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let r = m / 2.0;
    ///
    /// assert_eq!(r, Matrix2x2f32::new([ 1.0,  8.5,
    ///                                   2.5,  5.5]));
    /// ```
    #[inline]
    fn div(self, other: T) -> Self {
        T::m2x2_div_scalar(self, other)
    }
}

// **** DivAssign ****

impl<T> DivAssign<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// In-place divide a matrix by a scalar.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// m /= 2.0;
    ///
    /// assert_eq!(m, Matrix2x2f32::new([ 1.0,  8.5,
    ///                                   2.5,  5.5]));
    /// ```
    #[inline]
    fn div_assign(&mut self, other: T) {
        *self = *self / other;
    }
}

// **** AsRef ****

impl<T> AsRef<[T; 4]> for Matrix2x2<T> {
    /// Immutable reference to the raw array.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let a: &[f32; 4] = m.as_ref();
    /// assert_eq!(5.0, a[Matrix2x2f32::M21]);
    /// ```
    #[inline]
    fn as_ref(&self) -> &[T; 4] {
        &self.a
    }
}

impl<T> AsMut<[T; 4]> for Matrix2x2<T> {
    /// Immutable reference to the raw array.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// let a: &mut [f32; 4] = m.as_mut();
    /// a[2] = 7.0;
    /// assert_eq!(7.0, m[2]);
    /// ```
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 4] {
        &mut self.a
    }
}

// **** Deref ****

impl<T> Deref for Matrix2x2<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        &self.a
    }
}

impl<T> DerefMut for Matrix2x2<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.a
    }
}

// **** Index ****

impl<T> Index<usize> for Matrix2x2<T> {
    type Output = T;

    /// Access matrix element by index.
    /// ```
    /// # use vqm::Matrix2x2f32;
    ///
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    ///
    /// assert_eq!(m[Matrix2x2f32::M11], 2.0);
    /// assert_eq!(m[Matrix2x2f32::M12], 17.0);
    /// assert_eq!(m[Matrix2x2f32::M21], 5.0);
    /// assert_eq!(m[Matrix2x2f32::M22], 11.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

impl<T> Index<Range<usize>> for Matrix2x2<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: Range<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<RangeFull> for Matrix2x2<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _index: RangeFull) -> &[T] {
        &self.a
    }
}

impl<T> Index<RangeInclusive<usize>> for Matrix2x2<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeInclusive<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<(usize, usize)> for Matrix2x2<T> {
    type Output = T;

    /// Access matrix element by ordered pair (row, column).
    /// ```
    /// # use vqm::Matrix2x2f32;
    ///
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    ///
    /// assert_eq!(m[(0,0)], 2.0);
    /// assert_eq!(m[(0,1)], 17.0);
    /// assert_eq!(m[(1,0)], 5.0);
    /// assert_eq!(m[(1,1)], 11.0);
    /// ```
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < 2 && col < 2, "Matrix index out of bounds: row={row}, col={col}");
        &self.a[col * 2 + row]
    }
}

// **** IndexMut ****

impl<T> IndexMut<usize> for Matrix2x2<T> {
    /// Set matrix element by index.
    /// ```
    /// # use vqm::Matrix2x2f32;
    ///
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    ///
    /// m[Matrix2x2f32::M11] = 3.0;
    /// m[Matrix2x2f32::M12] = 19.0;
    /// m[Matrix2x2f32::M21] = 7.0;
    /// m[Matrix2x2f32::M22] = 13.0;
    ///
    /// assert_eq!(m, Matrix2x2f32::new([  3.0, 19.0,
    ///                                    7.0, 13.0]));
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.a[index]
    }
}

impl<T> IndexMut<Range<usize>> for Matrix2x2<T> {
    #[inline]
    fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<RangeFull> for Matrix2x2<T> {
    #[inline]
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        &mut self.a
    }
}

impl<T> IndexMut<RangeInclusive<usize>> for Matrix2x2<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeInclusive<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix2x2<T> {
    /// Set matrix element by ordered pair (row, column).
    /// ```
    /// # use vqm::Matrix2x2f32;
    ///
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    ///
    /// m[(0,0)] = 3.0;
    /// m[(0,1)] = 19.0;
    /// m[(1,0)] = 7.0;
    /// m[(1,1)] = 13.0;
    ///
    /// assert_eq!(m, Matrix2x2f32::new([  3.0, 19.0,
    ///                                    7.0, 13.0]));
    /// ```
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(row < 2 && col < 2, "Matrix index out of bounds: row={row}, col={col}");
        &mut self.a[col * 2 + row]
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Set matrix row from a vector.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// m.set_row(1, Vector2f32::new(7.0, 13.0));
    /// assert_eq!(Vector2f32{ x: 7.0, y: 13.0 }, m.row(1));
    /// ```
    pub fn set_row(&mut self, row: usize, value: Vector2<T>) {
        if row == 0 {
            self.a[0] = value.x;
            self.a[2] = value.y;
        } else {
            self.a[1] = value.x;
            self.a[3] = value.y;
        }
    }

    /// Return matrix row as a vector.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let v = m.row(0);
    ///
    /// assert_eq!(v, Vector2f32{ x: 2.0, y: 17.0 });
    /// assert_eq!(m.row(1), Vector2f32{ x: 5.0, y: 11.0 });
    /// ```
    pub fn row(self, row: usize) -> Vector2<T> {
        let r = row.min(1);
        Vector2 { x: self.a[r], y: self.a[r + 2] }
    }

    /// Returns a row as tuple.
    #[inline]
    pub fn row_tuple(&self, row: usize) -> (T, T) {
        let r = row.min(1);
        (self.a[r], self.a[r + 2])
    }

    /// Set matrix column from a vector.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// m.set_column(1, Vector2f32::new(7.0, 13.0));
    /// assert_eq!(Vector2f32{ x: 7.0, y: 13.0 }, m.column(1));
    /// ```
    pub fn set_column(&mut self, column: usize, value: Vector2<T>) {
        if column >= 2 {
            return;
        }
        let start = column * 2;
        // Extract a 4-element slice.
        // Because row < 2, start + 2 will never exceed 3.
        let column_slice = &mut self.a[start..start + 2];
        column_slice[0] = value.x;
        column_slice[1] = value.y;
    }

    /// Return matrix column as a vector.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let v = m.column(0);
    ///
    /// assert_eq!(v, Vector2f32{ x: 2.0, y: 5.0 });
    /// assert_eq!(m.column(1), Vector2f32{ x: 17.0, y: 11.0 });
    /// ```
    pub fn column(self, column: usize) -> Vector2<T> {
        // Branchless clamp: restricts c to 0..=1
        let base = column.min(1) * 2;
        let chunk = &self.a[base..];
        Vector2 { x: chunk[0], y: chunk[1] }
    }

    /// Returns a column as a tuple.
    #[inline]
    pub fn column_tuple(self, column: usize) -> (T, T) {
        // Branchless clamp: restricts c to 0..=1
        let base = column.min(1) * 2;
        let chunk = &self.a[base..];
        (chunk[0], chunk[1])
    }

    /// Return matrix diagonal as a array.
    /// ```
    /// # use vqm::Matrix2x2f32;
    ///
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let a = m.diagonal_as_array();
    ///
    /// assert_eq!(a, [ 2.0, 11.0 ]);
    /// ```
    pub fn diagonal_as_array(self) -> [T; 2] {
        [self.a[Self::M11], self.a[Self::M22]]
    }

    /// Return matrix diagonal as a vector.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2f32};
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let v = m.diagonal_as_vector();
    ///
    /// assert_eq!(v, Vector2f32{ x: 2.0, y: 11.0 });
    /// ```
    pub fn diagonal_as_vector(self) -> Vector2<T> {
        Vector2 { x: self.a[Self::M11], y: self.a[Self::M22] }
    }
}

// **** abs ****

impl<T> Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Return a copy of the matrix with all elements set to their absolute values.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, -17.0,
    ///                              5.0, -11.0]);
    /// let n = m.abs();
    ///
    /// assert_eq!(n, Matrix2x2f32::new([  2.0, 17.0,
    ///                                    5.0, 11.0]));
    /// ```
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        T::m2x2_abs(self)
    }

    /// Set all elements of the matrix to their absolute values.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, -17.0,
    ///                                  5.0, -11.0]);
    /// m.abs_in_place();
    ///
    /// assert_eq!(m, Matrix2x2f32::new([  2.0, 17.0,
    ///                                    5.0, 11.0]));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = T::m2x2_abs(*self);
        self
    }
}

// **** clamp ****

impl<T> Matrix2x2<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all elements clamped to the specified range.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let n = m.clamp(2.5, 7.5);
    ///
    /// assert_eq!(n, Matrix2x2f32::new([ 2.5, 7.5,
    ///                                   5.0, 7.5]));
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
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0,  3.0,
    ///                                  7.0, 11.0]);
    /// m.clamp_in_place(2.5, 7.5);
    ///
    /// assert_eq!(m, Matrix2x2f32::new([ 2.5, 3.0,
    ///                                   7.0, 7.5]));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Return the transpose of this matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let n = m.transpose();
    ///
    /// assert_eq!(n, Matrix2x2f32::new([  2.0,  5.0,
    ///                                   17.0, 11.0]));
    /// ```
    #[inline]
    #[must_use]
    pub fn transpose(self) -> Self {
        Self { a: [self.a[0], self.a[2], self.a[1], self.a[3]] }
    }

    /// Transpose matrix, in-place.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// m.transpose_in_place();
    ///
    /// assert_eq!(m, Matrix2x2f32::new([  2.0,  5.0,
    ///                                   17.0, 11.0]));
    /// ```
    #[inline]
    pub fn transpose_in_place(&mut self) -> &mut Self {
        *self = self.transpose();
        self
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Return the adjugate of this matrix, ie the transpose of the cofactor matrix.
    /// Equivalent to the inverse but without dividing by the determinant of the matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let (n,d) = m.adjugate();
    ///
    /// assert_eq!(n, Matrix2x2f32::new([ 11.0, -17.0,
    ///                                   -5.0,   2.0]));
    /// assert!((n*m/d).is_near_identity(1e-5));
    /// ```
    #[inline]
    pub fn adjugate(self) -> (Self, T) {
        let (adjugate, determinant) = T::m2x2_adjugate(self);
        (adjugate, determinant)
    }

    /// Adjugate matrix, in-place.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
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
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let d = m.determinant();
    ///
    /// assert_eq!(2.0*11.0 - 17.0*5.0, d);
    ///
    /// ```
    #[inline]
    pub fn determinant(self) -> T {
        T::m2x2_determinant(self)
    }

    /// Return trace of matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let trace = m.trace();
    ///
    /// assert_eq!(trace, 13.0);
    /// ```
    #[inline]
    pub fn trace(self) -> T {
        T::m2x2_trace(self)
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + Matrix2x2Math + FloatCore + MathConstants,
{
    /// Return the inverse of this matrix. Returns self if the determinant is zero.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let n = m.inverse();
    ///
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        let (adjugate, determinant) = T::m2x2_adjugate(self);
        if determinant.abs() < T::EPSILON {
            return self;
        }
        adjugate / determinant
    }

    /// Invert this matrix, in-place. Returns self if the determinant is zero.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::new([  2.0, 17.0,
    ///                                  5.0, 11.0]);
    /// m.invert_in_place();
    /// ```
    #[inline]
    pub fn invert_in_place(&mut self) -> &mut Self {
        let (adjugate, determinant) = T::m2x2_adjugate(*self);
        if determinant.abs() < T::EPSILON {
            return self;
        }
        *self = adjugate / determinant;
        self
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + Zero + One + Matrix2x2Math + MathConstants + PartialOrd + FloatCore,
{
    /// Return inverse of matrix or `T::zero()` if not invertible.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::Zero;
    /// let m = Matrix2x2f32::new([  2.0,  3.0,
    ///                              7.0, 10.5]);
    /// let n = m.inverse_or_zero();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(Matrix2x2f32::zero(), n);
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
    /// # use vqm::{Matrix2x2f32};
    /// let m = Matrix2x2f32::new([  2.0,  3.0,
    ///                              7.0, 10.5]);
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
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let s = m.sum();
    ///
    /// assert_eq!(s, 35.0);
    /// ```
    #[inline]
    pub fn sum(self) -> T {
        T::m2x2_sum(self)
    }

    /// Return the mean of all elements of the matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let mean = m.mean();
    ///
    /// assert_eq!(mean, 35.0 / 4.0);
    /// ```
    #[inline]
    pub fn mean(self) -> T {
        T::m2x2_mean(self)
    }

    /// Return the product of all elements of the matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let product = m.product();
    ///
    /// assert_eq!(product, 1870.0);
    /// ```
    #[inline]
    pub fn product(self) -> T {
        T::m2x2_product(self)
    }

    /// Return the sum of the squares of the trace of the matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let t = m.trace_sum_squares();
    ///
    /// assert_eq!(t, 2.0 * 2.0 + 11.0 * 11.0);
    /// ```
    #[inline]
    pub fn trace_sum_squares(self) -> T {
        T::m2x2_trace_sum_squares(self)
    }
}

// **** Symmetry ****

impl<T> Matrix2x2<T>
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
    }
}

// **** Iterators ****

impl<T> Matrix2x2<T> {
    /// Returns an iterator over the rows of the matrix as slices of 2 elements.
    #[inline]
    pub fn rows(&self) -> &[[T; 2]] {
        let (chunks, _remainder) = self.as_chunks::<2>();
        chunks
    }
}

impl<T> Matrix2x2<T> {
    /// Returns an iterator over the rows of the matrix as mutable slices of 2 elements.
    #[inline]
    pub fn rows_mut(&mut self) -> &mut [[T; 2]] {
        let (chunks, _remainder) = self.as_chunks_mut::<2>();
        chunks
    }
}

impl<T> Matrix2x2<T> {
    /// Consumes the matrix and returns an array of its 2 rows.
    #[allow(clippy::many_single_char_names)]
    #[inline]
    pub fn into_rows(self) -> [[T; 2]; 2] {
        let [a, b, c, d] = self.a;
        [[a, b], [c, d]]
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Returns an iterator over the columns of the matrix as owned 2-element arrays.
    #[inline]
    pub fn cols(&self) -> impl Iterator<Item = [T; 2]> + '_ {
        // Create an iterator over the column indices (0, 1)
        (0..2).map(|c| {
            // Collect the strided elements for the current column
            [self.a[c], self.a[c + 2]]
        })
    }
}

impl<'a, T> IntoIterator for &'a Matrix2x2<T> {
    type Item = &'a [T];
    type IntoIter = ChunksExact<'a, T>;

    #[allow(clippy::chunks_exact_to_as_chunks)]
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages the Deref trait automatically to get slice chunks
        self.chunks_exact(2)
    }
}

impl<'a, T> IntoIterator for &'a mut Matrix2x2<T> {
    type Item = &'a mut [T];
    type IntoIter = ChunksExactMut<'a, T>;

    #[allow(clippy::chunks_exact_to_as_chunks)]
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages the DerefMut trait automatically
        self.chunks_exact_mut(2)
    }
}

impl<T> IntoIterator for Matrix2x2<T> {
    type Item = [T; 2];
    type IntoIter = core::array::IntoIter<[T; 2], 2>;

    #[allow(clippy::many_single_char_names)]
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let [a, b, c, d] = self.a;
        // Construct an array of rows, then convert that array into an iterator
        [[a, b], [c, d]].into_iter()
    }
}

// **** Column Iterators ****

impl<T> Matrix2x2<T> {
    /// Exposes the matrix as a read-only reference to 2 contiguous columns.
    /// Each sub-array `[T; 2]` represents one full column in memory.
    #[inline]
    pub fn columns(&self) -> &[[T; 2]] {
        // SAFETY:
        // `self.a` contains 4 contiguous `T`s, so it can be viewed as 2 contiguous `[T; 2]` values.
        //`[T; 2]` has the same alignment as `T`.
        unsafe { core::slice::from_raw_parts(self.a.as_ptr().cast::<[T; 2]>(), 2) }
    }

    /// Exposes the matrix as a mutable reference to 2 contiguous columns.
    /// Each sub-array `[T; 2]` represents one full column in memory.
    #[inline]
    pub fn columns_mut(&mut self) -> &mut [[T; 2]] {
        // SAFETY: `self.a` contains 4 contiguous `T`s, which are 2 contiguous `[T; 2]` values.
        // `[T; 2]` has the same alignment as `T`, and the returned reference is uniquely borrowed from `self.a`.
        unsafe { core::slice::from_raw_parts_mut(self.a.as_mut_ptr().cast::<[T; 2]>(), 2) }
    }

    #[inline]
    pub fn iter_columns(&self) -> Matrix2x2Columns<'_, T> {
        Matrix2x2Columns { inner: self.columns().iter() }
    }

    #[inline]
    pub fn iter_columns_mut(&mut self) -> Matrix2x2ColumnsMut<'_, T> {
        Matrix2x2ColumnsMut { inner: self.columns_mut().iter_mut() }
    }
}

// **** Iterator Pairs ****

/// A custom iterator over the read-only columns of a 2x2 matrix.
#[derive(Debug)]
pub struct Matrix2x2Columns<'a, T> {
    inner: Iter<'a, [T; 2]>,
}

impl<'a, T> Iterator for Matrix2x2Columns<'a, T> {
    type Item = &'a [T; 2];

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
impl<T> ExactSizeIterator for Matrix2x2Columns<'_, T> {}

impl<T> DoubleEndedIterator for Matrix2x2Columns<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

/// A custom iterator over the mutable columns of a 2x2 matrix.
#[derive(Debug, Default)]
pub struct Matrix2x2ColumnsMut<'a, T> {
    inner: IterMut<'a, [T; 2]>,
}

impl<'a, T> Iterator for Matrix2x2ColumnsMut<'a, T> {
    type Item = &'a mut [T; 2];

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
impl<T> ExactSizeIterator for Matrix2x2ColumnsMut<'_, T> {}

impl<T> DoubleEndedIterator for Matrix2x2ColumnsMut<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}
