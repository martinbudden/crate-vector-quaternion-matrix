use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};

use crate::{MathConstants, Matrix2x2Math, Matrix3x3, Vector2d};

/// 2x2 matrix of `f32` values<br>
pub type Matrix2x2f32 = Matrix2x2<f32>;
/// 2x2 matrix of `f64` values<br><br>
pub type Matrix2x2f64 = Matrix2x2<f64>;

/// Zero determinant error.
#[derive(Debug, PartialEq)]
pub enum MatrixError {
    ZeroDeterminant,
}

// **** Define ****

/// `Matrix2x2<T>`: 2x2 Matrix of type `T`.<br>
/// Aliases `Matrix2x2f32` and `Matrix2x2f64` are provided.<br><br>
/// `Matrix2x2f32` uses **SIMD** accelerations implemented in `Matrix2x2Math`.<br><br>
/// Internal implementation is using a flattened 1-dimensional array: an array of 4 elements stored in row-major order.
/// That is the element `m[row][col]` is at array position `[row * 2 + col]`, so element `m01` is at `a[1]`.<br><br>
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix2x2<T> {
    // Flattened 2x2 matrix: 4 elements in row-major order
    pub(crate) a: [T; 4],
}

// **** New ****
impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Create a matrix
    #[inline(always)]
    pub const fn new(input: [T; 4]) -> Self {
        Self { a: input }
    }
}

// **** Zero ****

/// Zero matrix
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// # use num_traits::Zero;
/// let z = Matrix2x2f32::zero();
///
/// assert_eq!(z, Matrix2x2f32::from([ 0.0, 0.0,
///                                    0.0, 0.0]));
/// ```
impl<T> Zero for Matrix2x2<T>
where
    T: Copy + Zero + PartialEq + Matrix2x2Math,
{
    #[inline(always)]
    fn zero() -> Self {
        Self { a: [T::zero(), T::zero(), T::zero(), T::zero()] }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.a.iter().all(|&x| x == T::zero())
    }
}

// **** One ****

/// Identity matrix
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// # use num_traits::One;
/// let i = Matrix2x2f32::one();
///
/// assert_eq!(i, Matrix2x2f32::from([ 1.0, 0.0,
///                                    0.0, 1.0]));
/// ```
impl<T> One for Matrix2x2<T>
where
    T: Copy + Zero + One + PartialEq + Matrix2x2Math,
{
    #[inline(always)]
    fn one() -> Self {
        Self { a: [T::one(), T::zero(), T::zero(), T::one()] }
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.a == [T::one(), T::zero(), T::zero(), T::one()]
    }
}

// **** Neg ****

/// Negate matrix
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
///                                  7.0, 11.0]);
/// m = - m;
///
/// assert_eq!(m, Matrix2x2f32::from([ -2.0,  -3.0,
///                                    -7.0, -11.0]));
/// ```
impl<T> Neg for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        T::m2x2_neg(self)
    }
}

// **** Add ****

/// Add two matrices
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
/// let n = Matrix2x2f32::from([29.0, 31.0,
///                             41.0, 43.0]);
/// let r = m + n;
///
/// assert_eq!(r, Matrix2x2f32::from([31.0, 34.0,
///                                   48.0, 54.0]));
///
/// # use num_traits::Zero;
///
/// let z = Matrix2x2f32::zero();
/// let r2 = m + z;
///
/// assert_eq!(r2, m);
/// ```
impl<T> Add for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        T::m2x2_add(self, other)
    }
}

// **** AddAssign ****

/// Add one matrix to another
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
///                                  7.0, 11.0]);
/// let n = Matrix2x2f32::from([29.0, 31.0,
///                             41.0, 43.0]);
/// m += n;
///
/// assert_eq!(m, Matrix2x2f32::from([31.0, 34.0,
///                                   48.0, 54.0]));
/// ```
impl<T> AddAssign for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

/// Multiply vector by constant and add another vector
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// # use num_traits::MulAdd;
/// let m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
/// let n = Matrix2x2f32::from([29.0, 31.0,
///                             41.0, 43.0]);
/// let k = 47.0;
/// let r = m.mul_add(k, n);
///
/// assert_eq!(r, Matrix2x2f32::from([123.0, 172.0,
///                                   370.0, 560.0]));
/// ```
impl<T> MulAdd<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::m2x2_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

/// Multiply vector by constant and add another vector in place
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// # use num_traits::MulAddAssign;
/// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
/// let n = Matrix2x2f32::from([29.0, 31.0,
///                             41.0, 43.0]);
/// let k = 47.0;
/// m.mul_add_assign(k, n);
///
/// assert_eq!(m, Matrix2x2f32::from([123.0, 172.0,
///                                   370.0, 560.0]));
/// ```
impl<T> MulAddAssign<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    #[inline(always)]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

/// Subtract two matrices
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
/// let n = Matrix2x2f32::from([29.0, 31.0,
///                             41.0, 43.0]);
/// let r = m - n;
///
/// assert_eq!(r, Matrix2x2f32::from([-27.0, -28.0,
///                                   -34.0, -32.0]));
/// ```
impl<T> Sub for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

/// Subtract one matrix from another
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
///                                  7.0, 11.0]);
/// let n = Matrix2x2f32::from([29.0, 31.0,
///                             41.0, 43.0]);
/// m -= n;
///
/// assert_eq!(m, Matrix2x2f32::from([-27.0, -28.0,
///                                   -34.0, -32.0]));
/// ```
impl<T> SubAssign for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Pre-multiply ****

/// Pre-multiply a matrix by a constant
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
/// let r = 2.0 * m;
///
/// assert_eq!(r, Matrix2x2f32::from([ 4.0,  6.0,
///                                   14.0, 22.0]));
/// ```
impl Mul<Matrix2x2<f32>> for f32 {
    type Output = Matrix2x2<f32>;

    #[inline(always)]
    fn mul(self, other: Matrix2x2<f32>) -> Matrix2x2<f32> {
        f32::m2x2_mul_scalar(other, self)
    }
}

impl Mul<Matrix2x2<f64>> for f64 {
    type Output = Matrix2x2<f64>;
    #[inline(always)]
    fn mul(self, other: Matrix2x2<f64>) -> Matrix2x2<f64> {
        f64::m2x2_mul_scalar(other, self)
    }
}

// **** Mul ****

/// Multiply a matrix by a constant
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
/// let r = m * 2.0;
///
/// assert_eq!(r, Matrix2x2f32::from([ 4.0,  6.0,
///                                   14.0, 22.0]));
/// ```
impl<T> Mul<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: T) -> Self {
        T::m2x2_mul_scalar(self, other)
    }
}

// **** MulAssign ****

/// In-place multiply a matrix by a constant
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
///                                  7.0, 11.0]);
/// m *= 2.0;
///
/// assert_eq!(m, Matrix2x2f32::from([ 4.0,  6.0,
///                                   14.0, 22.0]));
/// ```
impl<T> MulAssign<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    #[inline(always)]
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

/// Multiply a vector by a matrix
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// # use vector_quaternion_matrix::Vector2d;
/// let m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
/// let v = Vector2d::<f32>{x:29.0, y:31.0};
/// let r = m * v;
///
/// assert_eq!(r, Vector2d::<f32>{x:2.0*29.0 + 3.0*31.0, y:7.0*29.0 + 11.0*31.0});
/// ```
impl<T> Mul<Vector2d<T>> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Vector2d<T>;

    #[inline(always)]
    fn mul(self, other: Vector2d<T>) -> Vector2d<T> {
        T::m2x2_mul_vector(self, other)
    }
}

/// Pre-multiply a vector by a matrix
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// # use vector_quaternion_matrix::Vector2df32;
/// let m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
/// let v = Vector2df32{x:29.0, y:31.0};
/// let r = v * m;
///
/// assert_eq!(r, Vector2df32{x:29.0*2.0 + 31.0*7.0, y:29.0*3.0 +31.0*11.0});
/// ```
impl<T> Mul<Matrix2x2<T>> for Vector2d<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: Matrix2x2<T>) -> Self {
        T::m2x2_vector_mul(self, other)
    }
}

/// Multiply two matrices
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
/// let n = Matrix2x2f32::from([29.0, 31.0,
///                             41.0, 43.0]);
/// let r = m * n;
///
/// assert_eq!(r, Matrix2x2f32::from([
///    2.0 * 29.0 +  3.0 * 41.0,   2.0 * 31.0 +  3.0 * 43.0,
///    7.0 * 29.0 + 11.0 * 41.0,   7.0 * 31.0 + 11.0 * 43.0,
/// ]));
///
/// # use num_traits::One;
///
/// let i = Matrix2x2f32::one();
/// let r2 = m * i;
///
/// assert_eq!(r2, m);
/// ```
impl<T> Mul<Matrix2x2<T>> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        T::m2x2_mul(self, other)
    }
}

/// Multiply one matrix by another
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
///
/// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
///                                  7.0, 11.0]);
/// let n = Matrix2x2f32::from([29.0, 31.0,
///                             41.0, 43.0]);
/// m *= n;
///
/// assert_eq!(m, Matrix2x2f32::from([
///    2.0 * 29.0 +  3.0 * 41.0,   2.0 * 31.0 +  3.0 * 43.0,
///    7.0 * 29.0 + 11.0 * 41.0,   7.0 * 31.0 + 11.0 * 43.0,
/// ]));
///
/// ```
impl<T> MulAssign<Matrix2x2<T>> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    #[inline(always)]
    fn mul_assign(&mut self, other: Matrix2x2<T>) {
        *self = *self * other;
    }
}
// **** Div ****

/// Divide a matrix by a constant
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
/// let r = m / 2.0;
///
/// assert_eq!(r, Matrix2x2f32::from([ 1.0, 1.5,
///                                    3.5, 5.5]));
/// ```
impl<T> Div<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, other: T) -> Self {
        T::m2x2_div_scalar(self, other)
    }
}

// **** DivAssign ****

/// In-place divide a matrix by a constant
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
///                                  7.0, 11.0]);
/// m /= 2.0;
///
/// assert_eq!(m, Matrix2x2f32::from([ 1.0, 1.5,
///                                    3.5, 5.5]));
/// ```
impl<T> DivAssign<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    #[inline(always)]
    fn div_assign(&mut self, other: T) {
        *self = *self / other;
    }
}

// **** Index ****

/// Access matrix element by index
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
///
/// let m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
///
/// assert_eq!(m[0], 2.0);
/// assert_eq!(m[1], 3.0);
/// assert_eq!(m[2], 7.0);
/// assert_eq!(m[3], 11.0);
/// ```
impl<T> Index<usize> for Matrix2x2<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

// **** IndexMut ****

/// Set matrix element by index
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
///
/// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
///                                  7.0, 11.0]);
///
/// m[0] = 29.0;
/// m[1] = 31.0;
/// m[2] = 37.0;
/// m[3] = 41.0;
///
/// assert_eq!(m, Matrix2x2f32::from([29.0, 31.0,
///                                       37.0, 41.0]));
/// ```
impl<T> IndexMut<usize> for Matrix2x2<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.a[index]
    }
}

/// Access matrix element by ordered pair (row, column)
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
///
/// let m = Matrix2x2f32::from([ 2.0,  3.0,
///                              7.0, 11.0]);
///
/// assert_eq!(m[(0,0)], 2.0);
/// assert_eq!(m[(0,1)], 3.0);
/// assert_eq!(m[(1,0)], 7.0);
/// assert_eq!(m[(1,1)], 11.0);
/// ```
impl<T> Index<(usize, usize)> for Matrix2x2<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.a[row * 2 + col]
    }
}

/// Set matrix element by ordered pair (row, column)
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
///
/// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
///                                  7.0, 11.0]);
///
/// m[(0,0)] = 29.0;
/// m[(0,1)] = 31.0;
/// m[(1,0)] = 41.0;
/// m[(1,1)] = 43.0;
///
/// assert_eq!(m, Matrix2x2f32::from([29.0, 31.0,
///                                   41.0, 43.0]));
/// ```
impl<T> IndexMut<(usize, usize)> for Matrix2x2<T> {
    #[inline(always)]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.a[row * 2 + col]
    }
}

// **** abs ****
impl<T> Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Return a copy of the matrix with all components set to their absolute values
    #[inline(always)]
    pub fn abs(self) -> Self {
        T::m2x2_abs(self)
    }

    /// Set all components of the matrix to their absolute values
    #[inline(always)]
    pub fn abs_in_place(&mut self) {
        *self = self.abs();
    }
}

/// Set components whose absolute value is less than EPSILON to zero.
/// ```
/// # use vector_quaternion_matrix::{Matrix2x2f32,Vector2df32,MathConstants};
///
/// let mut m = Matrix2x2f32::from([ 2.0,  f32::EPSILON /2.0,
///                              7.0, 11.0]);
/// m.epsilonize();
///
/// assert_eq!(m, Matrix2x2f32::from([ 2.0,  0.0,
///                                    7.0, 11.0]));
/// ```
impl<T> Matrix2x2<T>
where
    T: Copy + Signed + PartialOrd + MathConstants,
{
    /// If any component of the matrix is less than EPSILON in absolute value, set it to Zero.
    pub fn epsilonize(&mut self) {
        for a in self.a.iter_mut() {
            if a.abs() <= T::EPSILON {
                *a = T::zero();
            }
        }
    }
}

// **** clamp ****
impl<T> Matrix2x2<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all components clamped to the specified range
    #[inline(always)]
    pub fn clamp(self, min: T, max: T) -> Self {
        let mut a = self.a;
        for a in a.iter_mut() {
            *a = a.clamp(min, max);
        }
        Self { a }
    }

    /// Clamp all components of the matrix to the specified range
    #[inline(always)]
    pub fn clamp_in_place(&mut self, min: T, max: T) {
        for a in self.a.iter_mut() {
            *a = a.clamp(min, max);
        }
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    pub fn set_row(&mut self, row: usize, value: Vector2d<T>) {
        match row {
            0 => {
                self.a[0] = value.x;
                self.a[1] = value.y;
            }
            _ => {
                self.a[2] = value.x;
                self.a[3] = value.y;
            }
        }
    }

    /// Return matrix row as a vector
    /// ```
    /// # use vector_quaternion_matrix::{Matrix2x2f32,Vector2df32};
    ///
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let v = m.row(0);
    ///
    /// assert_eq!(v, Vector2df32{ x: 2.0, y: 3.0 });
    /// assert_eq!(m.row(1), Vector2df32{ x: 7.0, y: 11.0 });
    /// ```
    pub fn row(self, row: usize) -> Vector2d<T> {
        match row {
            0 => Vector2d::<T> { x: self.a[0], y: self.a[1] },
            // default to row 1 if row out of range
            _ => Vector2d::<T> { x: self.a[2], y: self.a[3] },
        }
    }

    pub fn set_column(&mut self, column: usize, value: Vector2d<T>) {
        match column {
            0 => {
                self.a[0] = value.x;
                self.a[3] = value.y;
            }
            _ => {
                self.a[1] = value.x;
                self.a[2] = value.y;
            }
        }
    }

    /// Return matrix column as a vector
    /// ```
    /// # use vector_quaternion_matrix::{Matrix2x2f32,Vector2df32};
    ///
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let v = m.column(0);
    ///
    /// assert_eq!(v, Vector2df32{ x: 2.0, y: 7.0 });
    /// assert_eq!(m.column(1), Vector2df32{ x: 3.0, y: 11.0 });
    /// ```
    pub fn column(self, column: usize) -> Vector2d<T> {
        match column {
            0 => Vector2d::<T> { x: self.a[0], y: self.a[2] },
            // default to column 1 if column out of range
            _ => Vector2d::<T> { x: self.a[1], y: self.a[3] },
        }
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Transpose matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let n = m.transpose();
    ///
    /// assert_eq!(n, Matrix2x2f32::from([ 2.0,  7.0,
    ///                                    3.0, 11.0]));
    /// ```
    #[inline(always)]
    pub fn transpose(self) -> Self {
        Self { a: [self.a[0], self.a[2], self.a[1], self.a[3]] }
    }

    /// Transpose matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                                  7.0, 11.0]);
    /// m.transpose_in_place();
    ///
    /// assert_eq!(m, Matrix2x2f32::from([ 2.0,  7.0,
    ///                                    3.0, 11.0]));
    /// ```
    #[inline(always)]
    pub fn transpose_in_place(&mut self) {
        *self = self.transpose();
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + Matrix2x2Math + One + Neg<Output = T> + Add<Output = T>,
{
    /// Adjugate matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let n = m.adjugate();
    ///
    /// assert!((n*m/m.determinant()).is_near_identity());
    /// ```
    #[inline(always)]
    pub fn adjugate(self) -> Self {
        T::m2x2_adjugate(self)
    }

    /// Adjugate matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let mut n = m;
    /// n.adjugate_in_place();
    ///
    /// assert_eq!(m.adjugate(), n);
    /// ```
    #[inline(always)]
    pub fn adjugate_in_place(&mut self) {
        *self = self.adjugate();
    }
    /// Add vector to diagonal of matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::{Matrix2x2f32,Vector2df32};
    /// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                                  7.0, 11.0]);
    ///
    /// let v = Vector2df32{ x: 10.0, y: 20.0 };
    ///
    /// m.add_to_diagonal_in_place(v);
    ///
    /// assert_eq!(m, Matrix2x2f32::from([ 12.0,  3.0,
    ///                                     7.0, 31.0]));
    /// ```
    #[inline(always)]
    pub fn add_to_diagonal_in_place(&mut self, v: Vector2d<T>) {
        self.a[0] = self.a[0] + v.x;
        self.a[3] = self.a[3] + v.y;
    }

    /// Subtract vector from diagonal of matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::{Matrix2x2f32,Vector2df32};
    /// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                                  7.0, 11.0]);
    ///
    /// let v = Vector2df32{ x: 10.0, y: 20.0 };
    ///
    /// m.subtract_from_diagonal_in_place(v);
    ///
    /// assert_eq!(m, Matrix2x2f32::from([ -8.0,  3.0,
    ///                                     7.0, -9.0]));
    /// ```
    #[inline(always)]
    pub fn subtract_from_diagonal_in_place(&mut self, v: Vector2d<T>) {
        self.a[0] = self.a[0] + (-v.x);
        self.a[3] = self.a[3] + (-v.y);
    }

    /// Matrix determinant
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let d = m.determinant();
    ///
    /// assert_eq!(2.0*11.0 - 3.0*7.0, d);
    ///
    /// ```
    #[inline(always)]
    pub fn determinant(self) -> T {
        T::m2x2_determinant(self)
    }

    /// Matrix top right determinant
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let d = m.top_right_determinant();
    ///
    /// assert_eq!(2.0*11.0 - 3.0*3.0, d);
    ///
    /// ```
    #[inline(always)]
    pub fn top_right_determinant(self) -> T {
        T::m2x2_top_right_determinant(self)
    }

    /// Return the sum of the squares of the top right elements
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let d = m.top_right_sum_squares();
    ///
    /// assert_eq!(3.0*3.0, d);
    ///
    /// ```
    #[inline(always)]
    pub fn top_right_sum_squares(self) -> T {
        T::m2x2_top_right_sum_squares(self)
    }

    /// Return the sum of all components of the matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let s = m.sum();
    ///
    /// assert_eq!(s, 23.0);
    /// ```
    #[inline(always)]
    pub fn sum(self) -> T {
        T::m2x2_sum(self)
    }

    /// Return the mean of all components of the matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let mean = m.mean();
    ///
    /// assert_eq!(mean, 23.0 / 4.0);
    /// ```
    #[inline(always)]
    pub fn mean(self) -> T {
        T::m2x2_mean(self)
    }

    /// Return the product of all components of the matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let product = m.product();
    ///
    /// assert_eq!(product, 462.0);
    /// ```
    #[inline(always)]
    pub fn product(self) -> T {
        T::m2x2_product(self)
    }

    /// Return trace of matrix.
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let trace = m.trace();
    ///
    /// assert_eq!(trace, 13.0);
    /// ```
    #[inline(always)]
    pub fn trace(self) -> T {
        T::m2x2_trace(self)
    }

    /// Return the sum of the squares of the trace of the matrix.
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let t = m.trace_sum_squares();
    ///
    /// assert_eq!(t, 2.0 * 2.0 + 11.0 * 11.0);
    /// ```
    #[inline(always)]
    pub fn trace_sum_squares(self) -> T {
        T::m2x2_trace_sum_squares(self)
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + Zero + One + Matrix2x2Math + MathConstants + PartialOrd + Signed,
{
    /// Invert matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                                  7.0, 11.0]);
    /// m.invert_in_place();
    ///
    /// ```
    #[inline(always)]
    pub fn invert_in_place(&mut self) -> bool {
        let adjugate = self.adjugate();
        let determinant = self.determinant();
        if determinant.abs() <= T::EPSILON {
            return false;
        }
        *self = adjugate / determinant;
        true
    }

    /// Return inverse of matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let n = m.inverse();
    ///
    /// ```
    #[inline(always)]
    pub fn inverse(self) -> Self {
        let adjugate = self.adjugate();
        let determinant = self.determinant();
        adjugate / determinant
    }
    /// Return inverse of matrix or `T::zero()` if not invertible.
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// # use num_traits::Zero;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 10.5]);
    /// let n = m.inverse_or_zero();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(Matrix2x2f32::zero(), n);
    ///
    /// ```
    pub fn inverse_or_zero(self) -> Self {
        let determinant = self.determinant();
        if determinant.abs() < T::EPSILON {
            return Self::zero();
        }
        let adjugate = self.adjugate();
        adjugate / determinant
    }
    /// Return inverse of matrix or `None` if not invertible.
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 10.5]);
    /// let n = m.try_inverse();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(None, n);
    ///
    /// ```
    pub fn try_inverse(self) -> Option<Self> {
        let determinant = self.determinant();
        if determinant.abs() < T::EPSILON {
            return None;
        }
        let adjugate = self.adjugate();
        Some(adjugate / determinant)
    }
    /// Return true if matrix is near zero
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// # use num_traits::Zero;
    /// let z = Matrix2x2f32::zero();
    /// assert!(z.is_near_zero());
    /// ```
    pub fn is_near_zero(self) -> bool {
        for a in self.a.iter() {
            if a.abs() > T::EPSILON {
                return false;
            }
        }
        true
    }

    /// Return true if matrix is near identity
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// # use num_traits::One;
    /// let i = Matrix2x2f32::one();
    /// assert!(i.is_near_identity());
    /// ```
    pub fn is_near_identity(self) -> bool {
        if self.a[1].abs() > T::EPSILON || self.a[2].abs() > T::EPSILON {
            return false;
        }
        if (self.a[0] - T::one()).abs() > T::EPSILON || (self.a[3] - T::one()).abs() > T::EPSILON {
            return false;
        }
        true
    }
}

// **** From ****

/// Matrix2x2 from array
impl<T> From<[T; 4]> for Matrix2x2<T>
where
    T: Copy,
{
    #[inline(always)]
    fn from(input: [T; 4]) -> Self {
        Self { a: input }
    }
}

/// Matrix2x2 from array of 2 vectors
impl<T> From<[Vector2d<T>; 2]> for Matrix2x2<T>
where
    T: Copy,
{
    #[inline(always)]
    fn from(v: [Vector2d<T>; 2]) -> Self {
        Self { a: [v[0].x, v[0].y, v[1].x, v[1].y] }
    }
}

/// Matrix2x2 from 2 vectors
impl<T> From<(Vector2d<T>, Vector2d<T>)> for Matrix2x2<T> {
    #[inline(always)]
    fn from(v: (Vector2d<T>, Vector2d<T>)) -> Self {
        Self { a: [v.0.x, v.0.y, v.1.x, v.1.y] }
    }
}

/// Matrix2x2 from Matrix3x3. Takes top left of m3x3, discarding other values.
/// ```
/// # use vector_quaternion_matrix::{Matrix2x2f32,Matrix3x3f32};
/// let m2 = Matrix2x2f32::from([ 2.0,  3.0,
///                               7.0, 11.0]);
/// let m3 = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                               7.0, 11.0, 13.0,
///                              17.0, 19.0, 23.0]);
/// assert_eq!(m2, Matrix2x2f32::from(m3));
/// ```
impl<T> From<Matrix3x3<T>> for Matrix2x2<T>
where
    T: Copy,
{
    #[inline(always)]
    fn from(m: Matrix3x3<T>) -> Self {
        Self { a: [m.a[0], m.a[1], m.a[3], m.a[4]] }
    }
}
