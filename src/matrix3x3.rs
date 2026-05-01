use cfg_if::cfg_if;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};

use crate::{MathConstants, Matrix2x2, Matrix3x3Math, Quaternion, QuaternionMath, SqrtMethods, Vector3d};

/// 3x3 matrix of `f32` values<br>
pub type Matrix3x3f32 = Matrix3x3<f32>;
/// 3x3 matrix of `f64` values<br><br>
pub type Matrix3x3f64 = Matrix3x3<f64>;

// **** Define ****

cfg_if! {
if #[cfg(feature = "no_align")] {
// Compact 36-byte version

/// `Matrix3x3<T>`: 3x3 Matrix of type `T`.<br>
/// Aliases `Matrix3x3f32` and `Matrix3x3f64` are provided.<br>
/// Internal implementation is a flattened 3x3 matrix: an array of 9 elements stored in row-major order<br>
/// That is the element `m[row][col]` is at array position `[row * 3 + col]`, so element `m12` is at `a[5]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix3x3<T> {
    // Flattened 3x3 matrix: 9 elements in row-major order
    pub(crate) a: [T; 9],
}

} else {
// High-performance 32-byte aligned version, allows use of SIMD and f32x8

/// `Matrix3x3<T>`: 3x3 Matrix of type `T`.<br>
/// Aliases `Matrix3x3f32` and `Matrix3x3f64` are provided.<br><br>
/// `Matrix3x3f32` uses **SIMD** accelerations implemented in `Matrix3x3Math`.<br><br>
/// Internal implementation is using a flattened 1-dimensional array: an array of 9 elements stored in row-major order.
/// That is the element `m[row][col]` is at array position `[row * 3 + col]`, so element `m12` is at `a[5]`.<br><br>
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix3x3<T> {
    // Flattened 3x3 matrix: 9 elements in row-major order
    pub(crate) a: [T; 9],
}
}
}

// **** New ****

/// Create a matrix.
/// ```
/// # use vqm::Matrix3x3f32;
/// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
///                              5.0, 11.0, 47.0,
///                             23.0, 31.0, 41.0]);
/// assert_eq!(m, Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                                     5.0, 11.0, 47.0,
///                                    23.0, 31.0, 41.0]));
/// ```
impl<T> Matrix3x3<T>
where
    T: Copy,
{
    /// Create a matrix.
    #[inline]
    pub const fn new(input: [T; 9]) -> Self {
        Self { a: input }
    }
}

// **** Zero ****

/// Zero matrix.
/// ```
/// # use vqm::Matrix3x3f32;
/// # use num_traits::{Zero,zero};
/// let z = Matrix3x3f32::zero();
///
/// assert_eq!(z, Matrix3x3f32::from([ 0.0, 0.0, 0.0,
///                                    0.0, 0.0, 0.0,
///                                    0.0, 0.0, 0.0]));
/// assert!(z.is_zero());
/// ```
impl<T> Zero for Matrix3x3<T>
where
    T: Copy + Zero + PartialEq + Matrix3x3Math,
{
    #[inline]
    fn zero() -> Self {
        Self { a: [T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero()] }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.a.iter().all(|&x| x == T::zero())
    }
}

// **** One ****

/// Identity matrix.
/// ```
/// # use vqm::Matrix3x3f32;
/// # use num_traits::One;
/// let i = Matrix3x3f32::one();
///
/// assert_eq!(i, Matrix3x3f32::from([ 1.0, 0.0, 0.0,
///                                    0.0, 1.0, 0.0,
///                                    0.0, 0.0, 1.0]));
/// ```
impl<T> One for Matrix3x3<T>
where
    T: Copy + Zero + One + PartialEq + Matrix3x3Math,
{
    #[inline]
    fn one() -> Self {
        Self { a: [T::one(), T::zero(), T::zero(), T::zero(), T::one(), T::zero(), T::zero(), T::zero(), T::one()] }
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.a == [T::one(), T::zero(), T::zero(), T::zero(), T::one(), T::zero(), T::zero(), T::zero(), T::one()]
    }
}

// **** Neg ****

/// Negate matrix.
/// ```
/// # use vqm::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                                   5.0, 11.0, 47.0,
///                                  23.0, 31.0, 41.0]);
/// m = - m;
///
/// assert_eq!(m, Matrix3x3f32::from([ -2.0, -17.0, -59.0,
///                                    -5.0, -11.0, -47.0,
///                                   -23.0, -31.0, -41.0]));
/// ```
impl<T> Neg for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        T::m3x3_neg(self)
    }
}

// **** Add ****

/// Add two matrices.
/// ```
/// # use vqm::Matrix3x3f32;
/// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                               5.0, 11.0, 47.0,
///                              23.0, 31.0, 41.0]);
/// let n = Matrix3x3f32::from([  3.0, 19.0, 61.0,
///                               7.0, 13.0, 53.0,
///                              29.0, 37.0, 43.0]);
/// let r = m + n;
///
/// assert_eq!(r, Matrix3x3f32::from([  5.0, 36.0, 120.0,
///                                    12.0, 24.0, 100.0,
///                                    52.0, 68.0,  84.0]));
///
/// # use num_traits::Zero;
///
/// let z = Matrix3x3f32::zero();
/// let r2 = m + z;
///
/// assert_eq!(r2, m);
/// ```
impl<T> Add for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        T::m3x3_add(self, other)
    }
}

// **** AddAssign ****

/// Add one matrix to another.
/// ```
/// # use vqm::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                                   5.0, 11.0, 47.0,
///                                  23.0, 31.0, 41.0]);
/// let n = Matrix3x3f32::from([  3.0, 19.0, 61.0,
///                               7.0, 13.0, 53.0,
///                              29.0, 37.0, 43.0]);
/// m += n;
///
/// assert_eq!(m, Matrix3x3f32::from([  5.0, 36.0, 120.0,
///                                    12.0, 24.0, 100.0,
///                                    52.0, 68.0,  84.0]));
/// ```
impl<T> AddAssign for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

/// Multiply vector by constant and add another vector.
/// ```
/// # use vqm::Matrix3x3f32;
/// # use num_traits::MulAdd;
/// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                               5.0, 11.0, 47.0,
///                               23.0, 31.0, 41.0]);
/// let n = Matrix3x3f32::from([  3.0, 19.0, 61.0,
///                               7.0, 13.0, 53.0,
///                              29.0, 37.0, 43.0]);
/// let k = 137.0;
/// let r = m.mul_add(k, n);
///
/// assert_eq!(r, Matrix3x3f32::from([  277.0, 2348.0, 8144.0,
///                                     692.0, 1520.0, 6492.0,
///                                    3180.0, 4284.0, 5660.0]));
/// ```
impl<T> MulAdd<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::m3x3_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

/// Multiply vector by constant and add another vector in place.
/// ```
/// # use vqm::Matrix3x3f32;
/// # use num_traits::MulAddAssign;
/// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                                   5.0, 11.0, 47.0,
///                                  23.0, 31.0, 41.0]);
/// let n = Matrix3x3f32::from([  3.0, 19.0, 61.0,
///                               7.0, 13.0, 53.0,
///                              29.0, 37.0, 43.0]);
/// let k = 137.0;
/// m.mul_add_assign(k, n);
///
/// assert_eq!(m, Matrix3x3f32::from([  277.0, 2348.0, 8144.0,
///                                     692.0, 1520.0, 6492.0,
///                                    3180.0, 4284.0, 5660.0]));
/// ```
impl<T> MulAddAssign<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

/// Subtract two matrices.
/// ```
/// # use vqm::Matrix3x3f32;
/// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                               5.0, 11.0, 47.0,
///                              23.0, 31.0, 41.0]);
/// let n = Matrix3x3f32::from([  3.0, 13.0, 43.0,
///                               7.0, 19.0, 37.0,
///                              29.0, 61.0, 53.0]);
/// let r = m - n;
///
/// assert_eq!(r, Matrix3x3f32::from([  -1.0,  4.0, 16.0,
///                                     -2.0, -8.0, 10.0,
///                                     -6.0,-30.0,-12.0]));
/// ```
impl<T> Sub for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

/// Subtract one matrix from another.
/// ```
/// # use vqm::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                                   5.0, 11.0, 47.0,
///                                  23.0, 31.0, 41.0]);
/// let n = Matrix3x3f32::from([  3.0, 13.0, 43.0,
///                               7.0, 19.0, 37.0,
///                              29.0, 61.0, 53.0]);
/// m -= n;
///
/// assert_eq!(m, Matrix3x3f32::from([  -1.0,  4.0, 16.0,
///                                     -2.0, -8.0, 10.0,
///                                     -6.0,-30.0,-12.0]));
/// ```
impl<T> SubAssign for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Pre-multiply ****

/// Pre-multiply a matrix by a constant.
/// ```
/// # use vqm::Matrix3x3f32;
/// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                               5.0, 11.0, 47.0,
///                              23.0, 31.0, 41.0]);
/// let r = 2.0 * m;
///
/// assert_eq!(r, Matrix3x3f32::from([  4.0, 34.0, 118.0,
///                                    10.0, 22.0,  94.0,
///                                    46.0, 62.0,  82.0]));
/// ```
impl Mul<Matrix3x3<f32>> for f32 {
    type Output = Matrix3x3<f32>;
    #[inline]
    fn mul(self, other: Matrix3x3<f32>) -> Matrix3x3<f32> {
        f32::m3x3_mul_scalar(other, self)
    }
}

impl Mul<Matrix3x3<f64>> for f64 {
    type Output = Matrix3x3<f64>;
    #[inline]
    fn mul(self, other: Matrix3x3<f64>) -> Matrix3x3<f64> {
        f64::m3x3_mul_scalar(other, self)
    }
}

// **** Mul ****

/// Multiply a matrix by a constant.
/// ```
/// # use vqm::Matrix3x3f32;
/// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                               5.0, 11.0, 47.0,
///                              23.0, 31.0, 41.0]);
/// let r = m * 2.0;
///
/// assert_eq!(r, Matrix3x3f32::from([  4.0, 34.0, 118.0,
///                                    10.0, 22.0,  94.0,
///                                    46.0, 62.0,  82.0]));
/// ```
impl<T> Mul<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: T) -> Self {
        T::m3x3_mul_scalar(self, other)
    }
}

// **** MulAssign ****

/// In-place multiply a matrix by a constant.
/// ```
/// # use vqm::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                                   5.0, 11.0, 47.0,
///                                  23.0, 31.0, 41.0]);
/// m *= 2.0;
///
/// assert_eq!(m, Matrix3x3f32::from([  4.0, 34.0, 118.0,
///                                    10.0, 22.0,  94.0,
///                                    46.0, 62.0,  82.0]));
/// ```
impl<T> MulAssign<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    #[inline]
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

/// Multiply a vector by a matrix.
/// ```
/// # use vqm::Matrix3x3f32;
/// # use vqm::Vector3df32;
/// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                               5.0, 11.0, 47.0,
///                              23.0, 31.0, 41.0]);
/// let v = Vector3df32{x:3.0, y:7.0, z:13.0};
/// let r = m * v;
///
/// assert_eq!(r, Vector3df32{x:892.0, y:703.0, z:819.0});
/// ```
impl<T> Mul<Vector3d<T>> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Vector3d<T>;
    #[inline]
    fn mul(self, other: Vector3d<T>) -> Vector3d<T> {
        T::m3x3_mul_vector(self, other)
    }
}

/// Pre-multiply a vector by a matrix.
/// ```
/// # use vqm::{Matrix3x3f32,Vector3df32};
/// let m = Matrix3x3f32::from([  2.0,   3.0,   5.0,
///                              11.0,  13.0,  17.0,
///                              23.0,  29.0,  31.0]);
/// let v = Vector3df32{x:59.0, y:61.0, z:67.0};
/// let r = v * m;
///
/// assert_eq!(r, Vector3df32{x:2330.0, y:2913.0, z:3409.0});
/// ```
impl<T> Mul<Matrix3x3<T>> for Vector3d<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Matrix3x3<T>) -> Self {
        T::m3x3_vector_mul(self, other)
    }
}

/// Multiply two matrices.
/// ```
/// # use vqm::Matrix3x3f32;
/// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                               5.0, 11.0, 47.0,
///                              23.0, 31.0, 41.0]);
/// let n = Matrix3x3f32::from([  3.0, 19.0, 61.0,
///                               7.0, 13.0, 53.0,
///                              29.0, 37.0, 43.0]);
/// let r = m * n;
///
/// assert_eq!(r, Matrix3x3f32::from([
///    2.0*3.0 + 17.0*7.0 + 59.0*29.0,   2.0*19.0 + 17.0*13.0 + 59.0*37.0,   2.0*61.0 + 17.0*53.0 + 59.0*43.0,
///    5.0*3.0 + 11.0*7.0 + 47.0*29.0,   5.0*19.0 + 11.0*13.0 + 47.0*37.0,   5.0*61.0 + 11.0*53.0 + 47.0*43.0,
///   23.0*3.0 + 31.0*7.0 + 41.0*29.0,  23.0*19.0 + 31.0*13.0 + 41.0*37.0,  23.0*61.0 + 31.0*53.0 + 41.0*43.0,
/// ]));
///
/// # use num_traits::One;
///
/// let i = Matrix3x3f32::one();
/// let r2 = m * i;
///
/// assert_eq!(r2, m);
/// ```
impl<T> Mul<Matrix3x3<T>> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        T::m3x3_mul(self, other)
    }
}

/// Multiply one matrix by another.
/// ```
/// # use vqm::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                                   5.0, 11.0, 47.0,
///                                  23.0, 31.0, 41.0]);
/// let n = Matrix3x3f32::from([  3.0, 19.0, 61.0,
///                               7.0, 13.0, 53.0,
///                              29.0, 37.0, 43.0]);
/// m *= n;
///
/// assert_eq!(m, Matrix3x3f32::from([
///    2.0*3.0 + 17.0*7.0 + 59.0*29.0,   2.0*19.0 + 17.0*13.0 + 59.0*37.0,   2.0*61.0 + 17.0*53.0 + 59.0*43.0,
///    5.0*3.0 + 11.0*7.0 + 47.0*29.0,   5.0*19.0 + 11.0*13.0 + 47.0*37.0,   5.0*61.0 + 11.0*53.0 + 47.0*43.0,
///   23.0*3.0 + 31.0*7.0 + 41.0*29.0,  23.0*19.0 + 31.0*13.0 + 41.0*37.0,  23.0*61.0 + 31.0*53.0 + 41.0*43.0,
/// ]));
/// ```
impl<T> MulAssign<Matrix3x3<T>> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    #[inline]
    fn mul_assign(&mut self, other: Matrix3x3<T>) {
        *self = *self * other;
    }
}
// **** Div ****

/// Divide a matrix by a constant.
/// ```
/// # use vqm::Matrix3x3f32;
/// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                               5.0, 11.0, 47.0,
///                              23.0, 31.0, 41.0]);
/// let r = m / 2.0;
///
/// assert_eq!(r, Matrix3x3f32::from([ 1.0,  8.5, 29.5,
///                                    2.5,  5.5, 23.5,
///                                   11.5, 15.5, 20.5]));
/// ```
impl<T> Div<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    #[inline]
    fn div(self, other: T) -> Self {
        T::m3x3_div_scalar(self, other)
    }
}

// **** DivAssign ****

/// In-place divide a matrix by a constant.
/// ```
/// # use vqm::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                                   5.0, 11.0, 47.0,
///                                  23.0, 31.0, 41.0]);
/// m /= 2.0;
///
/// assert_eq!(m, Matrix3x3f32::from([ 1.0,  8.5, 29.5,
///                                    2.5,  5.5, 23.5,
///                                   11.5, 15.5, 20.5]));
/// ```
impl<T> DivAssign<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    #[inline]
    fn div_assign(&mut self, other: T) {
        *self = *self / other;
    }
}

// **** Index ****

/// Access matrix element by index.
/// ```
/// # use vqm::Matrix3x3f32;
///
/// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                               5.0, 11.0, 47.0,
///                              23.0, 31.0, 41.0]);
///
/// assert_eq!(m[0], 2.0);
/// assert_eq!(m[1], 17.0);
/// assert_eq!(m[2], 59.0);
/// assert_eq!(m[3], 5.0);
/// assert_eq!(m[4], 11.0);
/// assert_eq!(m[5], 47.0);
/// assert_eq!(m[6], 23.0);
/// assert_eq!(m[7], 31.0);
/// assert_eq!(m[8], 41.0);
/// ```
impl<T> Index<usize> for Matrix3x3<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

// **** IndexMut ****

/// Set matrix element by index.
/// ```
/// # use vqm::Matrix3x3f32;
///
/// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                                   5.0, 11.0, 47.0,
///                                  23.0, 31.0, 41.0]);
///
/// m[0] = 3.0;
/// m[1] = 19.0;
/// m[2] = 61.0;
/// m[3] = 7.0;
/// m[4] = 13.0;
/// m[5] = 53.0;
/// m[6] = 29.0;
/// m[7] = 37.0;
/// m[8] = 43.0;
///
/// assert_eq!(m, Matrix3x3f32::from([  3.0, 19.0, 61.0,
///                                     7.0, 13.0, 53.0,
///                                    29.0, 37.0, 43.0]));
/// ```
impl<T> IndexMut<usize> for Matrix3x3<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.a[index]
    }
}

/// Access matrix element by ordered pair (row, column).
/// ```
/// # use vqm::Matrix3x3f32;
///
/// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                                   5.0, 11.0, 47.0,
///                                  23.0, 31.0, 41.0]);
///
/// assert_eq!(m[(0,0)], 2.0);
/// assert_eq!(m[(0,1)], 17.0);
/// assert_eq!(m[(0,2)], 59.0);
/// assert_eq!(m[(1,0)], 5.0);
/// assert_eq!(m[(1,1)], 11.0);
/// assert_eq!(m[(1,2)], 47.0);
/// assert_eq!(m[(2,0)], 23.0);
/// assert_eq!(m[(2,1)], 31.0);
/// assert_eq!(m[(2,2)], 41.0);
/// ```
impl<T> Index<(usize, usize)> for Matrix3x3<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.a[row * 3 + col]
    }
}

/// Set matrix element by ordered pair (row, column).
/// ```
/// # use vqm::Matrix3x3f32;
///
/// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
///                                   5.0, 11.0, 47.0,
///                                  23.0, 31.0, 41.0]);
///
/// m[(0,0)] = 3.0;
/// m[(0,1)] = 19.0;
/// m[(0,2)] = 61.0;
/// m[(1,0)] = 7.0;
/// m[(1,1)] = 13.0;
/// m[(1,2)] = 53.0;
/// m[(2,0)] = 29.0;
/// m[(2,1)] = 37.0;
/// m[(2,2)] = 43.0;
///
/// assert_eq!(m, Matrix3x3f32::from([  3.0, 19.0, 61.0,
///                                     7.0, 13.0, 53.0,
///                                    29.0, 37.0, 43.0]));
/// ```
impl<T> IndexMut<(usize, usize)> for Matrix3x3<T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.a[row * 3 + col]
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy,
{
    pub fn set_row(&mut self, row: usize, value: Vector3d<T>) {
        match row {
            0 => {
                self.a[0] = value.x;
                self.a[1] = value.y;
                self.a[2] = value.z;
            }
            1 => {
                self.a[3] = value.x;
                self.a[4] = value.y;
                self.a[5] = value.z;
            }
            _ => {
                self.a[6] = value.x;
                self.a[7] = value.y;
                self.a[8] = value.z;
            }
        }
    }

    /// Return matrix row as a vector.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let v = m.row(0);
    ///
    /// assert_eq!(v, Vector3df32{ x: 2.0, y: 17.0, z: 59.0 });
    /// assert_eq!(m.row(1), Vector3df32{ x: 5.0, y: 11.0, z: 47.0 });
    /// assert_eq!(m.row(2), Vector3df32{ x: 23.0, y: 31.0, z: 41.0 });
    /// ```
    pub fn row(self, row: usize) -> Vector3d<T> {
        match row {
            0 => Vector3d::<T> { x: self.a[0], y: self.a[1], z: self.a[2] },
            1 => Vector3d::<T> { x: self.a[3], y: self.a[4], z: self.a[5] },
            // default to row 2 if row out of range
            _ => Vector3d::<T> { x: self.a[6], y: self.a[7], z: self.a[8] },
        }
    }

    pub fn set_column(&mut self, column: usize, value: Vector3d<T>) {
        match column {
            0 => {
                self.a[0] = value.x;
                self.a[3] = value.y;
                self.a[6] = value.z;
            }
            1 => {
                self.a[1] = value.x;
                self.a[4] = value.y;
                self.a[7] = value.z;
            }
            _ => {
                self.a[2] = value.x;
                self.a[5] = value.y;
                self.a[8] = value.z;
            }
        }
    }

    /// Return matrix column as a vector.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let v = m.column(0);
    ///
    /// assert_eq!(v, Vector3df32{ x: 2.0, y: 5.0, z: 23.0 });
    /// assert_eq!(m.column(1), Vector3df32{ x: 17.0, y: 11.0, z: 31.0 });
    /// assert_eq!(m.column(2), Vector3df32{ x: 59.0, y: 47.0, z: 41.0 });
    /// ```
    pub fn column(self, column: usize) -> Vector3d<T> {
        match column {
            0 => Vector3d::<T> { x: self.a[0], y: self.a[3], z: self.a[6] },
            1 => Vector3d::<T> { x: self.a[1], y: self.a[4], z: self.a[7] },
            // default to column 2 if column out of range
            _ => Vector3d::<T> { x: self.a[2], y: self.a[5], z: self.a[8] },
        }
    }

    /// Return matrix diagonal as a vector.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    ///
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let v = m.diagonal();
    ///
    /// assert_eq!(v, Vector3df32{ x: 2.0, y: 11.0, z: 41.0 });
    /// ```
    pub fn diagonal(self) -> Vector3d<T> {
        Vector3d::<T> { x: self.a[0], y: self.a[4], z: self.a[8] }
    }
}

// **** abs ****

impl<T> Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    /// Return a copy of the matrix with all components set to their absolute values.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, -17.0,  59.0,
    ///                               5.0, -11.0,  47.0,
    ///                              23.0,  31.0, -41.0]);
    /// let n = m.abs();
    ///
    /// assert_eq!(n, Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                                     5.0, 11.0, 47.0,
    ///                                    23.0, 31.0, 41.0]));
    /// ```
    #[inline]
    pub fn abs(self) -> Self {
        T::m3x3_abs(self)
    }

    /// Set all components of the matrix to their absolute values.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::from([  2.0, -17.0, 59.0,
    ///                                   5.0, -11.0, 47.0,
    ///                                  23.0, 31.0, -41.0]);
    /// m.abs_in_place();
    ///
    /// assert_eq!(m, Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                                     5.0, 11.0, 47.0,
    ///                                    23.0, 31.0, 41.0]));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = T::m3x3_abs(*self);
        self
    }
}

// **** clamp ****

impl<T> Matrix3x3<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all components clamped to the specified range.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, -59.0,
    ///                               5.0, 11.0,  47.0,
    ///                              23.0, 31.0, -41.0]);
    /// let n = m.clamp(7.0, 17.0);
    ///
    /// assert_eq!(n, Matrix3x3f32::from([ 7.0, 17.0,  7.0,
    ///                                    7.0, 11.0, 17.0,
    ///                                   17.0, 17.0,  7.0]));
    /// ```
    #[inline]
    pub fn clamp(self, min: T, max: T) -> Self {
        let mut a = self.a;
        for it in &mut a {
            *it = it.clamp(min, max);
        }
        Self { a }
    }

    /// Clamp all components of the matrix to the specified range.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::from([  2.0, 17.0, -59.0,
    ///                                   5.0, 11.0,  47.0,
    ///                                  23.0, 31.0, -41.0]);
    /// m.clamp_in_place(7.0, 17.0);
    ///
    /// assert_eq!(m, Matrix3x3f32::from([ 7.0, 17.0,  7.0,
    ///                                    7.0, 11.0, 17.0,
    ///                                   17.0, 17.0,  7.0]));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy,
{
    /// Return the transpose of this matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let n = m.transpose();
    ///
    /// assert_eq!(n, Matrix3x3f32::from([  2.0,  5.0, 23.0,
    ///                                    17.0, 11.0, 31.0,
    ///                                    59.0, 47.0, 41.0]));
    /// ```
    #[inline]
    pub fn transpose(self) -> Self {
        Self { a: [self.a[0], self.a[3], self.a[6], self.a[1], self.a[4], self.a[7], self.a[2], self.a[5], self.a[8]] }
    }

    /// Transpose matrix, in-place.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                                   5.0, 11.0, 47.0,
    ///                                  23.0, 31.0, 41.0]);
    /// m.transpose_in_place();
    ///
    /// assert_eq!(m, Matrix3x3f32::from([  2.0,  5.0, 23.0,
    ///                                    17.0, 11.0, 31.0,
    ///                                    59.0, 47.0, 41.0]));
    /// ```
    #[inline]
    pub fn transpose_in_place(&mut self) -> &mut Self {
        *self = self.transpose();
        self
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    /// Return the adjugate of this matrix, ie the transpose of the cofactor matrix.
    /// Equivalent to the inverse but without dividing by the determinant of the matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::One;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let (n,d) = m.adjugate();
    ///
    /// assert_eq!(m.determinant(), d);
    /// assert!((n*m/m.determinant()).is_near_identity());
    /// assert_eq!(Matrix3x3f32::one(), n*m/(m.determinant()));
    /// ```
    #[inline]
    pub fn adjugate(self) -> (Self, T) {
        let (adjugate, determinant) = T::m3x3_adjugate(self);
        (adjugate, determinant)
    }

    /// Adjugate matrix, in-place.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
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
    /// Return the inverse of this matrix. Does not check if the determinant is non-zero before inverting.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let n = m.inverse();
    ///
    /// ```
    #[inline]
    pub fn inverse(self) -> Self {
        let (adjugate, determinant) = T::m3x3_adjugate(self);
        adjugate / determinant
    }

    /// Invert this matrix, in-place. Does not check if the determinant is non-zero before inverting.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                                   5.0, 11.0, 47.0,
    ///                                  23.0, 31.0, 41.0]);
    /// m.invert_in_place();
    ///
    /// ```
    #[inline]
    pub fn invert_in_place(&mut self) -> &mut Self {
        let (adjugate, determinant) = T::m3x3_adjugate(*self);
        *self = adjugate / determinant;
        self
    }

    /// Matrix determinant.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let d = m.determinant();
    ///
    /// assert_eq!(7098.0, d);
    ///
    /// ```
    #[inline]
    pub fn determinant(self) -> T {
        T::m3x3_determinant(self)
    }

    /// Return trace of matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let t = m.trace();
    ///
    /// assert_eq!(t, 54.0);
    /// ```
    #[inline]
    pub fn trace(self) -> T {
        T::m3x3_trace(self)
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy + Zero + One + Matrix3x3Math + MathConstants + PartialOrd + Signed,
{
    /// Return inverse of matrix or `T::zero()` if not invertible.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::Zero;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               2.0, 17.0, 59.0,
    ///                              23.0, 31.0, 41.0]);
    /// let n = m.inverse_or_zero();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(Matrix3x3f32::zero(), n);
    ///
    /// ```
    pub fn inverse_or_zero(self) -> Self {
        let (adjugate, determinant) = self.adjugate();
        if determinant.abs() < T::EPSILON {
            return Self::zero();
        }
        adjugate / determinant
    }

    /// Return inverse of matrix or `None` if not invertible.
    /// ```
    /// # use vqm::{Matrix3x3f32};
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               2.0, 17.0, 59.0,
    ///                              23.0, 31.0, 41.0]);
    /// let n = m.try_invert();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(None, n);
    ///
    /// ```
    pub fn try_invert(self) -> Option<Self> {
        let (adjugate, determinant) = self.adjugate();
        if determinant.abs() < T::EPSILON {
            return None;
        }
        Some(adjugate / determinant)
    }

    /// Return the sum of all components of the matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let s = m.sum();
    ///
    /// assert_eq!(s, 236.0);
    /// ```
    #[inline]
    pub fn sum(self) -> T {
        T::m3x3_sum(self)
    }

    /// Return the mean of all components of the matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let mean = m.mean();
    ///
    /// assert_eq!(mean, 236.0 / 9.0);
    /// ```
    #[inline]
    pub fn mean(self) -> T {
        T::m3x3_mean(self)
    }

    /// Return the product of all components of the matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let product = m.product();
    ///
    /// assert_eq!(product, 151_588_013_830.0);
    /// ```
    #[inline]
    pub fn product(self) -> T {
        T::m3x3_product(self)
    }

    /// Return the sum of the squares of the trace of the matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let t = m.trace_sum_squares();
    ///
    /// assert_eq!(t, 2.0 * 2.0 + 11.0 *11.0 + 41.0 * 41.0);
    /// ```
    #[inline]
    pub fn trace_sum_squares(self) -> T {
        T::m3x3_trace_sum_squares(self)
    }

    /// Return true if matrix is near zero.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::Zero;
    /// let z = Matrix3x3f32::zero();
    /// assert!(z.is_near_zero());
    /// ```
    pub fn is_near_zero(self) -> bool {
        for a in &self.a {
            if a.abs() > T::EPSILON {
                return false;
            }
        }
        true
    }

    /// Return true if matrix is near identity.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::One;
    /// let i = Matrix3x3f32::one();
    /// assert!(i.is_near_identity());
    /// ```
    pub fn is_near_identity(self) -> bool {
        if self.a[1].abs() > T::EPSILON
            || self.a[2].abs() > T::EPSILON
            || self.a[3].abs() > T::EPSILON
            || self.a[5].abs() > T::EPSILON
            || self.a[6].abs() > T::EPSILON
        {
            return false;
        }
        if (self.a[0] - T::one()).abs() > T::EPSILON
            || (self.a[4] - T::one()).abs() > T::EPSILON
            || (self.a[8] - T::one()).abs() > T::EPSILON
        {
            return false;
        }
        true
    }
}

// **** From ****

// **** From Array ****

/// Matrix from 1D array.
/// ```
/// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([  2.0, 17.0, 59.0,
    ///                               5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
/// assert_eq!(m, Matrix3x3f32::new([  2.0, 17.0, 59.0,
///                                    5.0, 11.0, 47.0,
///                                   23.0, 31.0, 41.0]));
/// ```
impl<T> From<[T; 9]> for Matrix3x3<T>
where
    T: Copy,
{
    #[inline]
    fn from(input: [T; 9]) -> Self {
        Self { a: input }
    }
}

/// Matrix from 2D array.
/// ```
/// # use vqm::Matrix3x3f32;
/// let m = Matrix3x3f32::from([[  2.0, 17.0, 59.0],
///                             [  5.0, 11.0, 47.0],
///                             [ 23.0, 31.0, 41.0]]);
/// assert_eq!(m, Matrix3x3f32::new([  2.0, 17.0, 59.0,
///                                    5.0, 11.0, 47.0,
///                                   23.0, 31.0, 41.0]));
/// ```
impl<T> From<[[T; 3]; 3]> for Matrix3x3<T>
where
    T: Copy,
{
    #[inline]
    fn from(a: [[T; 3]; 3]) -> Self {
        Self { a: [a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2]] }
    }
}

/// Matrix from array of vectors.
/// ```
/// # use vqm::{Matrix3x3f32,Vector3df32};
/// let m = Matrix3x3f32::from([ Vector3df32::new( 2.0, 17.0, 59.0),
///                              Vector3df32::new( 5.0, 11.0, 47.0),
///                              Vector3df32::new(23.0, 31.0, 41.0) ]);
/// assert_eq!(m, Matrix3x3f32::new([  2.0, 17.0, 59.0,
///                                    5.0, 11.0, 47.0,
///                                   23.0, 31.0, 41.0]));
/// ```
impl<T> From<[Vector3d<T>; 3]> for Matrix3x3<T>
where
    T: Copy,
{
    #[inline]
    fn from(v: [Vector3d<T>; 3]) -> Self {
        Self { a: [v[0].x, v[0].y, v[0].z, v[1].x, v[1].y, v[1].z, v[2].x, v[2].y, v[2].z] }
    }
}

/// Matrix from tuple of vectors.
/// ```
/// # use vqm::{Matrix3x3f32,Vector3df32};
/// let m = Matrix3x3f32::from(( Vector3df32::new( 2.0, 17.0, 59.0),
///                              Vector3df32::new( 5.0, 11.0, 47.0),
///                              Vector3df32::new(23.0, 31.0, 41.0) ));
/// assert_eq!(m, Matrix3x3f32::new([  2.0, 17.0, 59.0,
///                                    5.0, 11.0, 47.0,
///                                   23.0, 31.0, 41.0]));
/// ```
impl<T> From<(Vector3d<T>, Vector3d<T>, Vector3d<T>)> for Matrix3x3<T> {
    #[inline]
    fn from(v: (Vector3d<T>, Vector3d<T>, Vector3d<T>)) -> Self {
        Self { a: [v.0.x, v.0.y, v.0.z, v.1.x, v.1.y, v.1.z, v.2.x, v.2.y, v.2.z] }
    }
}

// **** From Matrix ****

/// Matrix3x3 from Matrix2x2.
/// ```
/// # use vqm::{Matrix2x2f32,Matrix3x3f32};
/// let m2 = Matrix2x2f32::from([ 2.0, 17.0,
///                               5.0, 11.0]);
/// let n2 = Matrix2x2f32::from([ 3.0, 19.0,
///                               7.0, 13.0]);
/// let m3: Matrix3x3f32 = m2.into();
/// let n3 = Matrix3x3f32::from(m2);
///
/// assert_eq!(m3, Matrix3x3f32::from([ 2.0, 17.0, 0.0,
///                                     5.0, 11.0, 0.0,
///                                     0.0,  0.0, 0.0]));
/// ```
impl<T> From<Matrix2x2<T>> for Matrix3x3<T>
where
    T: Copy + Zero,
{
    #[inline]
    fn from(m: Matrix2x2<T>) -> Self {
        Self { a: [m[0], m[1], T::zero(), m[2], m[3], T::zero(), T::zero(), T::zero(), T::zero()] }
    }
}

/// Matrix2x2 from Matrix3x3. Takes top left of m3x3, discarding other values.
/// ```
/// # use vqm::{Matrix2x2f32,Matrix3x3f32};
/// let m2 = Matrix2x2f32::from([ 2.0, 17.0,
///                               5.0, 11.0]);
/// let m3 = Matrix3x3f32::from([ 2.0, 17.0, 59.0,
///                               5.0, 11.0, 47.0,
///                              23.0, 31.0, 41.0]);
/// assert_eq!(m2, Matrix2x2f32::from(m3));
/// ```
impl<T> From<Matrix3x3<T>> for Matrix2x2<T>
where
    T: Copy,
{
    #[inline]
    fn from(m: Matrix3x3<T>) -> Self {
        Self { a: [m.a[0], m.a[1], m.a[3], m.a[4]] }
    }
}

// **** From Quaternion ****

/// Create rotation matrix from quaternion.
///
/// see [Quaternion-derived rotation matrix](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix),
/// uses Hamilton convention.
impl<T> From<Quaternion<T>> for Matrix3x3<T>
where
    T: Copy + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    #[inline]
    fn from(q: Quaternion<T>) -> Self {
        let two = T::one() + T::one();
        Self {
            a: [
                T::one() - (q.y * q.y + q.z * q.z) * two,
                (q.x * q.y - q.w * q.z) * two,
                (q.w * q.y + q.x * q.z) * two,
                (q.w * q.z + q.x * q.y) * two,
                T::one() - (q.x * q.x + q.z * q.z) * two,
                (q.y * q.z - q.w * q.x) * two,
                (q.x * q.z - q.w * q.y) * two,
                (q.w * q.x + q.y * q.z) * two,
                T::one() - (q.x * q.x + q.y * q.y) * two,
            ],
        }
    }
}

/// Create quaternion from a rotation matrix.
///
/// Adapted from [Converting a Rotation Matrix to a Quaternion](https://d3cw3dd2w33x3b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf) by Mike Day.
/// Note that Day's paper uses the [Shuster multiplication convention](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Alternative_conventions),
/// rather than the Hamilton multiplication convention used by the Quaternion class.
impl<T> From<Matrix3x3<T>> for Quaternion<T>
where
    T: Copy + One + FloatCore + SqrtMethods + QuaternionMath,
{
    fn from(m: Matrix3x3<T>) -> Self {
        let half = T::one() / (T::one() + T::one());
        // Choose largest scale factor from 4w, 4x, 4y, and 4z, to avoid a scale factor of zero, or numerical instabilities caused by division of a small scale factor.
        if m.a[8] < T::zero() {
            // |(x,y)| is bigger than |(z,w)|?
            if m.a[0] > m.a[4] {
                // |x| bigger than |y|, so use x-form
                let t = T::one() + (m.a[0] - m.a[4]) - m.a[8]; // 1 + 2(xx - yy) - 1 + 2(xx + yy) = 4xx
                let q = Self { w: m.a[7] - m.a[5], x: t, y: m.a[1] + m.a[3], z: m.a[6] + m.a[2] };
                return q * t.sqrt_reciprocal() * half;
            }
            // |y| bigger than |x|, so use y-form
            let t = T::one() - (m.a[0] - m.a[4]) - m.a[8]; // 1 - 2(xx - yy) - 1 + 2(xx + yy) = 4yy
            let q = Self { w: m.a[2] - m.a[6], x: m.a[1] + m.a[3], y: t, z: m.a[5] + m.a[7] };
            return q * t.sqrt_reciprocal() * half;
        }

        // |(z,w)| bigger than |(x,y)|
        if m.a[0] < -m.a[4] {
            // |z| bigger than |w|, so use z-form
            let t = T::one() - m.a[0] - (m.a[4] - m.a[8]); // 1 - (1 - 2*(yy + zz)) - (2(yy - zz)) = 4zz
            let q = Self { w: m.a[3] - m.a[1], x: m.a[2] + m.a[6], y: m.a[5] + m.a[7], z: t };
            return q * t.sqrt_reciprocal() * half;
        }

        // |w| bigger than |z|, so use w-form
        // ww + xx + yy + zz = 1, since unit quaternion, so xx + yy + zz =  1 - ww
        let t = T::one() + m.a[0] + m.a[4] + m.a[8]; // 1 + 1 - 2*(yy + zz) + 1 - 2(xx + zz) + 1 - 2(xx + yy) =  4 - 4(xx + yy + zz) = 4 - 4(1 - ww) = 4ww
        let q = Self { w: t, x: m.a[7] - m.a[5], y: m.a[2] - m.a[6], z: m.a[3] - m.a[1] };
        q * t.sqrt_reciprocal() * half
    }
}
