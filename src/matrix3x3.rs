use cfg_if::cfg_if;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{One, Signed, Zero, float::FloatCore};

use crate::{MathConstants, Matrix2x2, MatrixError, Quaternion, QuaternionOps, SqrtMethods, Vector3d};

/// 3x3 matrix of `f32` values
pub type Matrix3x3f32 = Matrix3x3<f32>;
/// 3x3 matrix of `f64` values
pub type Matrix3x3f64 = Matrix3x3<f64>;

// **** Align ****
// ensure vectors are aligned on 16 byte boundaries.
#[cfg(feature = "align")]
const _: () = assert!(core::mem::size_of::<Matrix3x3f32>() == 48);
#[cfg(feature = "align")]
const _: () = assert!(core::mem::align_of::<Matrix3x3f32>() == 16);
#[cfg(not(feature = "align"))]
const _: () = assert!(core::mem::size_of::<Matrix3x3f32>() == 36);
#[cfg(not(feature = "align"))]
const _: () = assert!(core::mem::align_of::<Matrix3x3f32>() == 4);

// **** Define ****
cfg_if! {
if #[cfg(feature = "align")] {
// High-performance 16-byte aligned version
/// `Matrix3x3<T>`: 3x3 Matrix of type `T`.<br>
/// Aliases `Matrix3x3f32` and `Matrix3x3f64` are provided.<br>
/// Internal implementation is a flattened 3x3 matrix: an array of 9 elements stored in row-major order<br>
/// That is the element `m[row][col]` is at array position `[row * 3 + col]`, so element `m12` is at `a[5]`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix3x3<T> {
    // Flattened 3x3 matrix: 9 elements in row-major order
    a: [T; 9],
}
} else {
// Compact 36-byte version
/// `Matrix3x3<T>`: 3x3 Matrix of type `T`.<br>
/// Aliases `Matrix3x3f32` and `Matrix3x3f64` are provided.<br>
/// Internal implementation is a flattened 3x3 matrix: an array of 9 elements stored in row-major order<br>
/// That is the element `m[row][col]` is at array position `[row * 3 + col]`, so element `m12` is at `a[5]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix3x3<T> {
    // Flattened 3x3 matrix: 9 elements in row-major order
    a: [T; 9],
}
}
}

// **** Zero ****
/// Zero matrix
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// # use num_traits::Zero;
/// let z = Matrix3x3f32::zero();
///
/// assert_eq!(z, Matrix3x3f32::from([ 0.0, 0.0, 0.0,
///                                    0.0, 0.0, 0.0,
///                                    0.0, 0.0, 0.0]));
/// ```
impl<T> Zero for Matrix3x3<T>
where
    T: Copy + Zero + PartialEq,
{
    fn zero() -> Self {
        Self { a: [T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero()] }
    }
    fn is_zero(&self) -> bool {
        self.a.iter().all(|&x| x == T::zero())
    }
}

// **** One ****
/// Identity matrix
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// # use num_traits::One;
/// let i = Matrix3x3f32::one();
///
/// assert_eq!(i, Matrix3x3f32::from([ 1.0, 0.0, 0.0,
///                                    0.0, 1.0, 0.0,
///                                    0.0, 0.0, 1.0]));
/// ```
impl<T> One for Matrix3x3<T>
where
    T: Copy + Zero + One + PartialEq + Sub<Output = T> + Mul<Output = T>,
{
    fn one() -> Self {
        Self { a: [T::one(), T::zero(), T::zero(), T::zero(), T::one(), T::zero(), T::zero(), T::zero(), T::one()] }
    }

    fn is_one(&self) -> bool {
        self.a == [T::one(), T::zero(), T::zero(), T::zero(), T::one(), T::zero(), T::zero(), T::zero(), T::one()]
    }
}

// **** Neg ****
/// Negate matrix
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                                  7.0, 11.0, 13.0,
///                                 17.0, 19.0, 23.0]);
/// m = - m;
///
/// assert_eq!(m, Matrix3x3f32::from([ -2.0,  -3.0,  -5.0,
///                                    -7.0, -11.0, -13.0,
///                                   -17.0, -19.0, -23.0]));
/// ```
impl<T> Neg for Matrix3x3<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut a = self.a;
        for r in a.iter_mut() {
            *r = -*r;
        }
        Self { a }
    }
}

// **** Add ****
/// Add two matrices
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                              7.0, 11.0, 13.0,
///                             17.0, 19.0, 23.0]);
/// let n = Matrix3x3f32::from([29.0, 31.0, 37.0,
///                             41.0, 43.0, 47.0,
///                             53.0, 59.0, 61.0]);
/// let r = m + n;
///
/// assert_eq!(r, Matrix3x3f32::from([31.0, 34.0, 42.0,
///                                   48.0, 54.0, 60.0,
///                                   70.0, 78.0, 84.0]));
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
    T: Copy + Add<Output = T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut a = self.a;
        for (ii, r) in a.iter_mut().enumerate() {
            *r = *r + rhs.a[ii];
        }
        Self { a }
    }
}

// **** AddAssign ****
/// Add one matrix to another
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                                  7.0, 11.0, 13.0,
///                                 17.0, 19.0, 23.0]);
/// let n = Matrix3x3f32::from([29.0, 31.0, 37.0,
///                             41.0, 43.0, 47.0,
///                             53.0, 59.0, 61.0]);
/// m += n;
///
/// assert_eq!(m, Matrix3x3f32::from([31.0, 34.0, 42.0,
///                                   48.0, 54.0, 60.0,
///                                   70.0, 78.0, 84.0]));
/// ```
impl<T> AddAssign for Matrix3x3<T>
where
    T: Copy + Add<Output = T>,
{
    fn add_assign(&mut self, rhs: Self) {
        for (ii, r) in self.a.iter_mut().enumerate() {
            *r = *r + rhs.a[ii];
        }
    }
}

// **** Sub ****
/// Subtract two matrices
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                              7.0, 11.0, 13.0,
///                             17.0, 19.0, 23.0]);
/// let n = Matrix3x3f32::from([29.0, 31.0, 37.0,
///                             41.0, 43.0, 47.0,
///                             53.0, 59.0, 61.0]);
/// let r = m - n;
///
/// assert_eq!(r, Matrix3x3f32::from([-27.0, -28.0, -32.0,
///                                   -34.0, -32.0, -34.0,
///                                   -36.0, -40.0, -38.0]));
/// ```
impl<T> Sub for Matrix3x3<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut a = self.a;
        for (ii, r) in a.iter_mut().enumerate() {
            *r = *r - rhs.a[ii];
        }
        Self { a }
    }
}

// **** SubAssign ****
/// Subtract one matrix from another
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                                  7.0, 11.0, 13.0,
///                                 17.0, 19.0, 23.0]);
/// let n = Matrix3x3f32::from([29.0, 31.0, 37.0,
///                             41.0, 43.0, 47.0,
///                             53.0, 59.0, 61.0]);
/// m -= n;
///
/// assert_eq!(m, Matrix3x3f32::from([-27.0, -28.0, -32.0,
///                                   -34.0, -32.0, -34.0,
///                                   -36.0, -40.0, -38.0]));
/// ```
impl<T> SubAssign for Matrix3x3<T>
where
    T: Copy + Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: Self) {
        for (ii, r) in self.a.iter_mut().enumerate() {
            *r = *r - rhs.a[ii];
        }
    }
}

// **** Pre-multiply ****
/// Pre-multiply a matrix by a constant
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                               7.0, 11.0, 13.0,
///                              17.0, 19.0, 23.0]);
/// let r = 2.0 * m;
///
/// assert_eq!(r, Matrix3x3f32::from([ 4.0,  6.0, 10.0,
///                                   14.0, 22.0, 26.0,
///                                   34.0, 38.0, 46.0]));
/// ```
impl Mul<Matrix3x3<f32>> for f32 {
    type Output = Matrix3x3<f32>;
    fn mul(self, rhs: Matrix3x3<f32>) -> Matrix3x3<f32> {
        let mut a = rhs.a;
        for r in a.iter_mut() {
            *r *= self;
        }
        Matrix3x3::<f32> { a }
    }
}

impl Mul<Matrix3x3<f64>> for f64 {
    type Output = Matrix3x3<f64>;
    fn mul(self, rhs: Matrix3x3<f64>) -> Matrix3x3<f64> {
        let mut a = rhs.a;
        for r in a.iter_mut() {
            *r *= self;
        }
        Matrix3x3::<f64> { a }
    }
}

// **** Mul ****
/// Multiply a matrix by a constant
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                              7.0, 11.0, 13.0,
///                             17.0, 19.0, 23.0]);
/// let r = m * 2.0;
///
/// assert_eq!(r, Matrix3x3f32::from([ 4.0,  6.0, 10.0,
///                                   14.0, 22.0, 26.0,
///                                   34.0, 38.0, 46.0]));
/// ```
impl<T> Mul<T> for Matrix3x3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, k: T) -> Self {
        let mut a = self.a;
        for r in a.iter_mut() {
            *r = *r * k;
        }
        Self { a }
    }
}

// **** MulAssign ****
/// In-place multiply a matrix by a constant
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                                  7.0, 11.0, 13.0,
///                                 17.0, 19.0, 23.0]);
/// m *= 2.0;
///
/// assert_eq!(m, Matrix3x3f32::from([ 4.0,  6.0, 10.0,
///                                   14.0, 22.0, 26.0,
///                                   34.0, 38.0, 46.0]));
/// ```
impl<T> MulAssign<T> for Matrix3x3<T>
where
    T: Copy + Mul<Output = T>,
{
    fn mul_assign(&mut self, k: T) {
        #[allow(clippy::assign_op_pattern)]
        for r in self.a.iter_mut() {
            *r = *r * k;
        }
    }
}

/// Multiply a vector by a matrix
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// # use vector_quaternion_matrix::Vector3d;
/// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                              7.0, 11.0, 13.0,
///                             17.0, 19.0, 23.0]);
/// let v = Vector3d::<f32>{x:29.0, y:31.0, z:37.0};
/// let r = m * v;
///
/// assert_eq!(r, Vector3d::<f32>{x:336.0, y:1025.0, z:1933.0});
/// ```
impl<T> Mul<Vector3d<T>> for Matrix3x3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector3d<T>;
    fn mul(self, v: Vector3d<T>) -> Vector3d<T> {
        Vector3d::<T> {
            x: self.a[0] * v.x + self.a[1] * v.y + self.a[2] * v.z,
            y: self.a[3] * v.x + self.a[4] * v.y + self.a[5] * v.z,
            z: self.a[6] * v.x + self.a[7] * v.y + self.a[8] * v.z,
        }
    }
}

/// Pre-multiply a vector by a matrix
/// ```
/// # use vector_quaternion_matrix::{Matrix3x3f32,Vector3df32};
/// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                              7.0, 11.0, 13.0,
///                             17.0, 19.0, 23.0]);
/// let v = Vector3df32{x:29.0, y:31.0, z:37.0};
/// let r = v * m;
///
/// assert_eq!(r, Vector3df32{x:29.0*2.0 + 31.0*7.0 + 37.0*17.0, y:1131.0, z:1399.0});
/// ```
impl<T> Mul<Matrix3x3<T>> for Vector3d<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, rhs: Matrix3x3<T>) -> Self {
        Self {
            x: self.x * rhs.a[0] + self.y * rhs.a[3] + self.z * rhs.a[6],
            y: self.x * rhs.a[1] + self.y * rhs.a[4] + self.z * rhs.a[7],
            z: self.x * rhs.a[2] + self.y * rhs.a[5] + self.z * rhs.a[8],
        }
    }
}

/// Multiply two matrices
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                              7.0, 11.0, 13.0,
///                             17.0, 19.0, 23.0]);
/// let n = Matrix3x3f32::from([29.0, 31.0, 37.0,
///                             41.0, 43.0, 47.0,
///                             53.0, 59.0, 61.0]);
/// let r = m * n;
///
/// assert_eq!(r, Matrix3x3f32::from([
///    2.0 * 29.0 +  3.0 * 41.0 +  5.0 * 53.0,   2.0 * 31.0 +  3.0 * 43.0 +  5.0 * 59.0,   2.0 * 37.0 +  3.0 * 47.0 +  5.0 * 61.0,
///    7.0 * 29.0 + 11.0 * 41.0 + 13.0 * 53.0,   7.0 * 31.0 + 11.0 * 43.0 + 13.0 * 59.0,   7.0 * 37.0 + 11.0 * 47.0 + 13.0 * 61.0,
///   17.0 * 29.0 + 19.0 * 41.0 + 23.0 * 53.0,  17.0 * 31.0 + 19.0 * 43.0 + 23.0 * 59.0,  17.0 * 37.0 + 19.0 * 47.0 + 23.0 * 61.0,
/// ]));
///
/// # use num_traits::One;
///
/// let I = Matrix3x3f32::one();
/// let r2 = m * I;
///
/// assert_eq!(r2, m);
/// ```
impl<T> Mul<Matrix3x3<T>> for Matrix3x3<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let a = [
            self.a[0] * rhs.a[0] + self.a[1] * rhs.a[3] + self.a[2] * rhs.a[6],
            self.a[0] * rhs.a[1] + self.a[1] * rhs.a[4] + self.a[2] * rhs.a[7],
            self.a[0] * rhs.a[2] + self.a[1] * rhs.a[5] + self.a[2] * rhs.a[8],
            self.a[3] * rhs.a[0] + self.a[4] * rhs.a[3] + self.a[5] * rhs.a[6],
            self.a[3] * rhs.a[1] + self.a[4] * rhs.a[4] + self.a[5] * rhs.a[7],
            self.a[3] * rhs.a[2] + self.a[4] * rhs.a[5] + self.a[5] * rhs.a[8],
            self.a[6] * rhs.a[0] + self.a[7] * rhs.a[3] + self.a[8] * rhs.a[6],
            self.a[6] * rhs.a[1] + self.a[7] * rhs.a[4] + self.a[8] * rhs.a[7],
            self.a[6] * rhs.a[2] + self.a[7] * rhs.a[5] + self.a[8] * rhs.a[8],
        ];
        Matrix3x3::<T> { a }
    }
}

/// Multiply one matrix by another
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                                  7.0, 11.0, 13.0,
///                                 17.0, 19.0, 23.0]);
/// let n = Matrix3x3f32::from([29.0, 31.0, 37.0,
///                             41.0, 43.0, 47.0,
///                             53.0, 59.0, 61.0]);
/// m *= n;
///
/// assert_eq!(m, Matrix3x3f32::from([
///    2.0 * 29.0 +  3.0 * 41.0 +  5.0 * 53.0,   2.0 * 31.0 +  3.0 * 43.0 +  5.0 * 59.0,   2.0 * 37.0 +  3.0 * 47.0 +  5.0 * 61.0,
///    7.0 * 29.0 + 11.0 * 41.0 + 13.0 * 53.0,   7.0 * 31.0 + 11.0 * 43.0 + 13.0 * 59.0,   7.0 * 37.0 + 11.0 * 47.0 + 13.0 * 61.0,
///   17.0 * 29.0 + 19.0 * 41.0 + 23.0 * 53.0,  17.0 * 31.0 + 19.0 * 43.0 + 23.0 * 59.0,  17.0 * 37.0 + 19.0 * 47.0 + 23.0 * 61.0,
/// ]));
/// ```
impl<T> MulAssign<Matrix3x3<T>> for Matrix3x3<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: Matrix3x3<T>) {
        *self = *self * rhs;
    }
}
// **** Div ****
/// Divide a matrix by a constant
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                              7.0, 11.0, 13.0,
///                             17.0, 19.0, 23.0]);
/// let r = m / 2.0;
///
/// assert_eq!(r, Matrix3x3f32::from([ 1.0, 1.5, 2.5,
///                                    3.5, 5.5, 6.5,
///                                    8.5, 9.5, 11.5]));
/// ```
impl<T> Div<T> for Matrix3x3<T>
where
    T: Copy + One + Div<Output = T>,
{
    type Output = Self;
    fn div(self, k: T) -> Self {
        let reciprocal: T = T::one() / k;
        /*let mut a = self.a;
        for r in a.iter_mut() {
            *r = *r * reciprocal;
        }
        Matrix3x3::<T> { a }*/
        self * reciprocal
    }
}

// **** DivAssign ****
/// In-place divide a matrix by a constant
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
/// let mut m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                                  7.0, 11.0, 13.0,
///                                 17.0, 19.0, 23.0]);
/// m /= 2.0;
///
/// assert_eq!(m, Matrix3x3f32::from([ 1.0, 1.5, 2.5,
///                                    3.5, 5.5, 6.5,
///                                    8.5, 9.5, 11.5]));
/// ```
impl<T> DivAssign<T> for Matrix3x3<T>
where
    T: Copy + One + Div<Output = T>,
{
    fn div_assign(&mut self, rhs: T) {
        let reciprocal: T = T::one() / rhs;
        for r in self.a.iter_mut() {
            *r = *r * reciprocal;
        }
    }
}

// **** Index ****
/// Access matrix element by index
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
///
/// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                              7.0, 11.0, 13.0,
///                             17.0, 19.0, 23.0]);
///
/// assert_eq!(m[0], 2.0);
/// assert_eq!(m[1], 3.0);
/// assert_eq!(m[2], 5.0);
/// assert_eq!(m[3], 7.0);
/// assert_eq!(m[4], 11.0);
/// assert_eq!(m[5], 13.0);
/// assert_eq!(m[6], 17.0);
/// assert_eq!(m[7], 19.0);
/// assert_eq!(m[8], 23.0);
/// ```
impl<T> Index<usize> for Matrix3x3<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

// **** IndexMut ****
/// Set matrix element by index
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
///
/// let mut m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                                  7.0, 11.0, 13.0,
///                                  17.0, 19.0, 23.0]);
///
/// m[0] = 29.0;
/// m[1] = 31.0;
/// m[2] = 37.0;
/// m[3] = 41.0;
/// m[4] = 43.0;
/// m[5] = 47.0;
/// m[6] = 53.0;
/// m[7] = 59.0;
/// m[8] = 61.0;
///
/// assert_eq!(m, Matrix3x3f32::from([29.0, 31.0, 37.0,
///                                   41.0, 43.0, 47.0,
///                                   53.0, 59.0, 61.0]));
/// ```
impl<T> IndexMut<usize> for Matrix3x3<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.a[index]
    }
}

/// Access matrix element by ordered pair (row, column)
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
///
/// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                              7.0, 11.0, 13.0,
///                             17.0, 19.0, 23.0]);
///
/// assert_eq!(m[(0,0)], 2.0);
/// assert_eq!(m[(0,1)], 3.0);
/// assert_eq!(m[(0,2)], 5.0);
/// assert_eq!(m[(1,0)], 7.0);
/// assert_eq!(m[(1,1)], 11.0);
/// assert_eq!(m[(1,2)], 13.0);
/// assert_eq!(m[(2,0)], 17.0);
/// assert_eq!(m[(2,1)], 19.0);
/// assert_eq!(m[(2,2)], 23.0);
/// ```
impl<T> Index<(usize, usize)> for Matrix3x3<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.a[row * 3 + col]
    }
}

/// Set matrix element by ordered pair (row, column)
/// ```
/// # use vector_quaternion_matrix::Matrix3x3f32;
///
/// let mut m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
///                                  7.0, 11.0, 13.0,
///                                 17.0, 19.0, 23.0]);
///
/// m[(0,0)] = 29.0;
/// m[(0,1)] = 31.0;
/// m[(0,2)] = 37.0;
/// m[(1,0)] = 41.0;
/// m[(1,1)] = 43.0;
/// m[(1,2)] = 47.0;
/// m[(2,0)] = 53.0;
/// m[(2,1)] = 59.0;
/// m[(2,2)] = 61.0;
///
/// assert_eq!(m, Matrix3x3f32::from([29.0, 31.0, 37.0,
///                                   41.0, 43.0, 47.0,
///                                   53.0, 59.0, 61.0]));
/// ```
impl<T> IndexMut<(usize, usize)> for Matrix3x3<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.a[row * 3 + col]
    }
}

// **** impl new ****
impl<T> Matrix3x3<T>
where
    T: Copy,
{
    /// Create a matrix
    pub const fn new(input: [T; 9]) -> Self {
        Self { a: input }
    }
}

// **** impl abs ****
impl<T> Matrix3x3<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the matrix with all components set to their absolute values
    pub fn abs(&self) -> Self {
        let mut data = self.a;
        for d in data.iter_mut() {
            *d = d.abs();
        }
        Self { a: data }
    }

    /// Set all components of the matrix to their absolute values
    pub fn abs_in_place(&mut self) {
        for d in self.a.iter_mut() {
            *d = d.abs();
        }
    }
}

// **** impl clamp ****
impl<T> Matrix3x3<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all components clamped to the specified range
    pub fn clamp(&self, min: T, max: T) -> Self {
        let mut data = self.a;
        for d in data.iter_mut() {
            *d = d.clamp(min, max);
        }
        Self { a: data }
    }

    /// Clamp all components of the matrix to the specified range
    pub fn clamp_in_place(&mut self, min: T, max: T) {
        for d in self.a.iter_mut() {
            *d = d.clamp(min, max);
        }
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

    /// Return matrix row as a vector
    /// ```
    /// # use vector_quaternion_matrix::{Matrix3x3f32,Vector3df32};
    /// let A = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let v = A.row(0);
    ///
    /// assert_eq!(v, Vector3df32{ x: 2.0, y: 3.0, z: 5.0 });
    /// assert_eq!(A.row(1), Vector3df32{ x: 7.0, y: 11.0, z: 13.0 });
    /// assert_eq!(A.row(2), Vector3df32{ x: 17.0, y: 19.0, z: 23.0 });
    /// ```
    pub fn row(&self, row: usize) -> Vector3d<T> {
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

    /// Return matrix column as a vector
    /// ```
    /// # use vector_quaternion_matrix::{Matrix3x3f32,Vector3df32};
    /// let A = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let v = A.column(0);
    ///
    /// assert_eq!(v, Vector3df32{ x: 2.0, y: 7.0, z: 17.0 });
    /// assert_eq!(A.column(1), Vector3df32{ x: 3.0, y: 11.0, z: 19.0 });
    /// assert_eq!(A.column(2), Vector3df32{ x: 5.0, y: 13.0, z: 23.0 });
    /// ```
    pub fn column(&self, column: usize) -> Vector3d<T> {
        match column {
            0 => Vector3d::<T> { x: self.a[0], y: self.a[3], z: self.a[6] },
            1 => Vector3d::<T> { x: self.a[1], y: self.a[4], z: self.a[7] },
            // default to column 2 if column out of range
            _ => Vector3d::<T> { x: self.a[2], y: self.a[5], z: self.a[8] },
        }
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy,
{
    /// Transpose matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let A = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let B = A.transpose();
    ///
    /// assert_eq!(B, Matrix3x3f32::from([ 2.0,  7.0, 17.0,
    ///                                    3.0, 11.0, 19.0,
    ///                                    5.0, 13.0, 23.0]));
    /// ```
    pub fn transpose(&self) -> Self {
        Self { a: [self.a[0], self.a[3], self.a[6], self.a[1], self.a[4], self.a[7], self.a[2], self.a[5], self.a[8]] }
    }

    /// Transpose matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let mut A = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                                  7.0, 11.0, 13.0,
    ///                                 17.0, 19.0, 23.0]);
    /// A.transpose_in_place();
    ///
    /// assert_eq!(A, Matrix3x3f32::from([ 2.0,  7.0, 17.0,
    ///                                    3.0, 11.0, 19.0,
    ///                                    5.0, 13.0, 23.0]));
    /// ```
    pub fn transpose_in_place(&mut self) {
        *self = self.transpose();
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy + Zero + One + Neg<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    /// Adjugate matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let A = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let B = A.adjugate();
    ///
    /// assert!((B*A/A.determinant()).is_near_identity());
    /// ```
    pub fn adjugate(&self) -> Self {
        Self {
            a: [
                self.a[4] * self.a[8] - self.a[5] * self.a[7],    //  (e*i - f*h)
                -(self.a[1] * self.a[8] - self.a[2] * self.a[7]), // -(b*i - c*h)
                self.a[1] * self.a[5] - self.a[2] * self.a[4],    //  (b*f - c*e)
                -(self.a[3] * self.a[8] - self.a[5] * self.a[6]), // -(d*i - f*g)
                self.a[0] * self.a[8] - self.a[2] * self.a[6],    //  (a*i - c*g)
                -(self.a[0] * self.a[5] - self.a[2] * self.a[3]), // -(a*f - c*d)
                self.a[3] * self.a[7] - self.a[4] * self.a[6],    //  (d*h - e*g)
                -(self.a[0] * self.a[7] - self.a[1] * self.a[6]), // -(a*h - b*g)
                self.a[0] * self.a[4] - self.a[1] * self.a[3],    //  (a*e - b*d)
            ],
        }
    }

    /// Adjugate matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let A = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let mut B = A;
    /// B.adjugate_in_place();
    ///
    /// assert_eq!(A.adjugate(), B);
    /// ```
    pub fn adjugate_in_place(&mut self) {
        *self = self.adjugate();
    }
    /// Add vector to diagonal of matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::{Matrix3x3f32,Vector3df32};
    /// let mut A = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                                  7.0, 11.0, 13.0,
    ///                                 17.0, 19.0, 23.0]);
    ///
    /// let v = Vector3df32{ x: 10.0, y: 20.0, z: 30.0 };
    ///
    /// A.add_to_diagonal_in_place(v);
    ///
    /// assert_eq!(A, Matrix3x3f32::from([ 12.0,  3.0,  5.0,
    ///                                     7.0, 31.0, 13.0,
    ///                                    17.0, 19.0, 53.0]));
    /// ```
    pub fn add_to_diagonal_in_place(&mut self, v: Vector3d<T>) {
        self.a[0] = self.a[0] + v.x;
        self.a[4] = self.a[4] + v.y;
        self.a[8] = self.a[8] + v.z;
    }

    /// Subtract vector from diagonal of matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::{Matrix3x3f32,Vector3df32};
    /// let mut A = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                                  7.0, 11.0, 13.0,
    ///                                 17.0, 19.0, 23.0]);
    ///
    /// let v = Vector3df32{ x: 10.0, y: 20.0, z: 30.0 };
    ///
    /// A.subtract_from_diagonal_in_place(v);
    ///
    /// assert_eq!(A, Matrix3x3f32::from([ -8.0,  3.0,   5.0,
    ///                                     7.0, -9.0,  13.0,
    ///                                    17.0, 19.0,  -7.0]));
    /// ```
    pub fn subtract_from_diagonal_in_place(&mut self, v: Vector3d<T>) {
        self.a[0] = self.a[0] - v.x;
        self.a[4] = self.a[4] - v.y;
        self.a[8] = self.a[8] - v.z;
    }

    /// Invert matrix in-place, assuming it is a diagonal matrix
    pub fn invert_in_place_assuming_diagonal(&mut self) {
        self.a[0] = T::one() / self.a[0];
        self.a[4] = T::one() / self.a[4];
        self.a[8] = T::one() / self.a[8];
    }

    /// Return inverse of matrix, assuming it is diagonal
    pub fn inverse_assuming_diagonal(&self) -> Self {
        let mut ret = *self;
        ret.invert_in_place_assuming_diagonal();
        ret
    }

    /// Matrix determinant
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let d = m.determinant();
    ///
    /// assert_eq!(-78.0, d);
    ///
    /// ```
    pub fn determinant(&self) -> T {
        self.a[0] * (self.a[4] * self.a[8] - self.a[5] * self.a[7])
            - self.a[1] * (self.a[3] * self.a[8] - self.a[5] * self.a[6])
            + self.a[2] * (self.a[3] * self.a[7] - self.a[4] * self.a[6])
    }

    /// Matrix top right determinant
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let A = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let d = A.top_right_determinant();
    ///
    /// assert_eq!(76.0, d);
    ///
    /// ```
    pub fn top_right_determinant(&self) -> T {
        //let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
        //             0     4     8     5      5      1      1     8    5     2        2     1    5      4     2
        self.a[0] * (self.a[4] * self.a[8] - self.a[5] * self.a[5])
            - self.a[1] * (self.a[1] * self.a[8] - self.a[5] * self.a[2])
            + self.a[2] * (self.a[1] * self.a[5] - self.a[4] * self.a[2])
    }

    /// Return the sum of all components of the matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let s = m.sum();
    ///
    /// assert_eq!(s, 100.0);
    /// ```
    pub fn sum(&self) -> T {
        self.a[0] + self.a[1] + self.a[2] + self.a[3] + self.a[4] + self.a[5] + self.a[6] + self.a[7] + self.a[8]
    }

    /// Return the mean of all components of the matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let mean = m.mean();
    ///
    /// assert_eq!(mean, 100.0 / 9.0);
    /// ```
    pub fn mean(&self) -> T {
        let nine = T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one();
        self.sum() / nine
    }

    /// Return the product of all components of the matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let product = m.product();
    ///
    /// assert_eq!(product, 223092860.0);
    /// ```
    pub fn product(&self) -> T {
        self.a[0] * self.a[1] * self.a[2] * self.a[3] * self.a[4] * self.a[5] * self.a[6] * self.a[7] * self.a[8]
    }

    /// Return trace of matrix.
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let A = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let t = A.trace();
    ///
    /// assert_eq!(t, 36.0);
    /// ```
    pub fn trace(&self) -> T {
        self.a[0] + self.a[4] + self.a[8]
    }
    /// Return the sum of the squares of the trace of the matrix.
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let A = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let t = A.trace_sum_squares();
    ///
    /// assert_eq!(t, 2.0 * 2.0 + 11.0 *11.0 + 23.0 * 23.0);
    /// ```
    pub fn trace_sum_squares(&self) -> T {
        self.a[0] * self.a[0] + self.a[4] * self.a[4] + self.a[8] * self.a[8]
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy
        + Zero
        + One
        + MathConstants
        + PartialOrd
        + Signed
        + Neg<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    /// Invert matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let mut a = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                                  7.0, 11.0, 13.0,
    ///                                 17.0, 19.0, 23.0]);
    /// a.invert_in_place();
    ///
    /// ```
    pub fn invert_in_place(&mut self) -> bool {
        let adjugate = self.adjugate();
        let determinant = self.a[0] * adjugate.a[0] + self.a[1] * adjugate.a[3] + self.a[2] * adjugate.a[6];
        if determinant.abs() <= T::EPSILON {
            return false;
        }
        *self = adjugate / determinant;
        true
    }

    /// Return inverse of matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              7.0, 11.0, 13.0,
    ///                             17.0, 19.0, 23.0]);
    /// let n = m.inverse();
    ///
    /// ```
    pub fn inverse(&self) -> Self {
        let adjugate = self.adjugate();
        let determinant = self.a[0] * adjugate.a[0] + self.a[1] * adjugate.a[3] + self.a[2] * adjugate.a[6];
        adjugate / determinant
    }
    /// Return inverse of matrix or zero if not invertible.
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// # use num_traits::Zero;
    /// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              2.0,  3.0,  5.0,
    ///                             17.0, 19.0, 23.0]);
    /// let n = m.inverse_or_zero();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(Matrix3x3f32::zero(), n);
    ///
    /// ```
    pub fn inverse_or_zero(&self) -> Self {
        let determinant = self.determinant();
        if determinant.abs() < T::EPSILON {
            return Self::zero();
        }
        let adjugate = self.adjugate();
        adjugate / determinant
    }
    /// Return inverse of matrix or `None` if not invertible.
    /// ```
    /// # use vector_quaternion_matrix::{Matrix3x3f32,MatrixError};
    /// let m = Matrix3x3f32::from([ 2.0,  3.0,  5.0,
    ///                              2.0,  3.0,  5.0,
    ///                             17.0, 19.0, 23.0]);
    /// let n = m.try_inverse();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(Err(MatrixError::ZeroDeterminant), n);
    ///
    /// ```
    pub fn try_inverse(&self) -> Result<Self, MatrixError> {
        let determinant = self.determinant();
        if determinant.abs() < T::EPSILON {
            return Err(MatrixError::ZeroDeterminant);
        }
        let adjugate = self.adjugate();
        Ok(adjugate / determinant)
    }
    /// Return true if matrix is near zero
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// # use num_traits::Zero;
    ///
    /// let Z = Matrix3x3f32::zero();
    /// assert!(Z.is_near_zero());
    /// ```
    pub fn is_near_zero(&self) -> bool {
        for a in self.a.iter() {
            if a.abs() > T::EPSILON {
                return false;
            }
        }
        true
    }

    /// Return true if matrix is near identity
    /// ```
    /// # use vector_quaternion_matrix::Matrix3x3f32;
    /// # use num_traits::One;
    ///
    /// let I = Matrix3x3f32::one();
    /// assert!(I.is_near_identity());
    /// ```
    pub fn is_near_identity(&self) -> bool {
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
/// Matrix3x3 from Matrix2x2
/// ```
/// # use vector_quaternion_matrix::{Matrix2x2f32,Matrix3x3f32};
/// let m2 = Matrix2x2f32::from([ 2.0,  3.0,
///                               7.0, 11.0]);
/// let n2 = Matrix2x2f32::from([ 13.0, 17.0,
///                               19.0, 21.0]);
/// let m3: Matrix3x3f32 = m2.into();
/// let n3 = Matrix3x3f32::from(m2);
///
/// assert_eq!(m3, Matrix3x3f32::from([ 2.0,  3.0, 0.0,
///                                     7.0, 11.0, 0.0,
///                                     0.0,  0.0, 0.0]));
/// ```
impl<T> From<Matrix2x2<T>> for Matrix3x3<T>
where
    T: Zero + Copy,
{
    fn from(m: Matrix2x2<T>) -> Self {
        Self { a: [m[0], m[1], T::zero(), m[2], m[3], T::zero(), T::zero(), T::zero(), T::zero()] }
    }
}

// **** From Array ****
/// Matrix from array
impl<T> From<[T; 9]> for Matrix3x3<T>
where
    T: Copy,
{
    fn from(input: [T; 9]) -> Self {
        Self { a: input }
    }
}

impl<T> From<[Vector3d<T>; 3]> for Matrix3x3<T>
where
    T: Copy,
{
    fn from(v: [Vector3d<T>; 3]) -> Self {
        Self { a: [v[0].x, v[0].y, v[0].z, v[1].x, v[1].y, v[1].z, v[2].x, v[2].y, v[2].z] }
    }
}

impl<T> From<(Vector3d<T>, Vector3d<T>, Vector3d<T>)> for Matrix3x3<T> {
    fn from(v: (Vector3d<T>, Vector3d<T>, Vector3d<T>)) -> Self {
        Self { a: [v.0.x, v.0.y, v.0.z, v.1.x, v.1.y, v.1.z, v.2.x, v.2.y, v.2.z] }
    }
}

/// Create rotation matrix from quaternion.
///
/// see [Quaternion-derived rotation matrix](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix),
/// uses Hamilton convention.
impl<T> From<Quaternion<T>> for Matrix3x3<T>
where
    T: Copy + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
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
/// Adapted from [Converting a Rotation Matrix to a Quaternion](https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf) by Mike Day.
/// Note that Day's paper uses the [Shuster multiplication convention](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Alternative_conventions),
/// rather than the Hamilton multiplication convention used by the Quaternion class.
impl<T> From<Matrix3x3<T>> for Quaternion<T>
where
    T: Copy + One + FloatCore + SqrtMethods + QuaternionOps,
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
                return q * t.reciprocal_sqrt() * half;
            }
            // |y| bigger than |x|, so use y-form
            let t = T::one() - (m.a[0] - m.a[4]) - m.a[8]; // 1 - 2(xx - yy) - 1 + 2(xx + yy) = 4yy
            let q = Self { w: m.a[2] - m.a[6], x: m.a[1] + m.a[3], y: t, z: m.a[5] + m.a[7] };
            return q * t.reciprocal_sqrt() * half;
        }

        // |(z,w)| bigger than |(x,y)|
        if m.a[0] < -m.a[4] {
            // |z| bigger than |w|, so use z-form
            let t = T::one() - m.a[0] - (m.a[4] - m.a[8]); // 1 - (1 - 2*(yy + zz)) - (2(yy - zz)) = 4zz
            let q = Self { w: m.a[3] - m.a[1], x: m.a[2] + m.a[6], y: m.a[5] + m.a[7], z: t };
            return q * t.reciprocal_sqrt() * half;
        }

        // |w| bigger than |z|, so use w-form
        // ww + xx + yy + zz = 1, since unit quaternion, so xx + yy + zz =  1 - ww
        let t = T::one() + m.a[0] + m.a[4] + m.a[8]; // 1 + 1 - 2*(yy + zz) + 1 - 2(xx + zz) + 1 - 2(xx + yy) =  4 - 4(xx + yy + zz) = 4 - 4(1 - ww) = 4ww
        let q = Self { w: t, x: m.a[7] - m.a[5], y: m.a[2] - m.a[6], z: m.a[3] - m.a[1] };
        q * t.reciprocal_sqrt() * half
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    const A: Matrix3x3<f32> = Matrix3x3::<f32> { a: [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0] };
    const B: Matrix3x3<f32> = Matrix3x3::<f32> { a: [29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 61.0] };

    #[test]
    fn normal_types() {
        is_normal::<Matrix3x3<f32>>();
    }
    #[test]
    fn default() {
        let a: Matrix3x3<f32> = Matrix3x3f32::default();
        assert_eq!(a, Matrix3x3f32 { a: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] });
        let z = Matrix3x3f32::zero();
        //let z: Matrix3x3 = zero();
        assert_eq!(a, z);
        assert!(z.is_zero());
        assert!(!z.is_one());
        assert!(z.is_near_zero());

        let i = Matrix3x3f32::one();
        //let i: Matrix3x3 = one();
        assert!(i.is_one());
        assert!(!i.is_zero());
        assert!(i.is_near_identity());
    }
    #[test]
    fn neg() {
        assert_eq!(-A, Matrix3x3::<f32> { a: [-2.0, -3.0, -5.0, -7.0, -11.0, -13.0, -17.0, -19.0, -23.0] });

        let b = -A;
        assert_eq!(b, Matrix3x3::<f32> { a: [-2.0, -3.0, -5.0, -7.0, -11.0, -13.0, -17.0, -19.0, -23.0] });
    }
    #[test]
    fn add() {
        let a_plus_b = Matrix3x3::<f32> {
            a: [
                2.0 + 29.0,
                3.0 + 31.0,
                5.0 + 37.0,
                7.0 + 41.0,
                11.0 + 43.0,
                13.0 + 47.0,
                17.0 + 53.0,
                19.0 + 59.0,
                23.0 + 61.0,
            ],
        };
        assert_eq!(A + B, a_plus_b);
    }
    #[test]
    fn sub() {
        let a_minus_b = Matrix3x3::from([
            2.0 - 29.0,
            3.0 - 31.0,
            5.0 - 37.0,
            7.0 - 41.0,
            11.0 - 43.0,
            13.0 - 47.0,
            17.0 - 53.0,
            19.0 - 59.0,
            23.0 - 61.0,
        ]);
        assert_eq!(A - B, a_minus_b);
    }
    #[test]
    fn mul() {
        let a_times_b = Matrix3x3::from([
            2.0 * 29.0 + 3.0 * 41.0 + 5.0 * 53.0,
            2.0 * 31.0 + 3.0 * 43.0 + 5.0 * 59.0,
            2.0 * 37.0 + 3.0 * 47.0 + 5.0 * 61.0,
            7.0 * 29.0 + 11.0 * 41.0 + 13.0 * 53.0,
            7.0 * 31.0 + 11.0 * 43.0 + 13.0 * 59.0,
            7.0 * 37.0 + 11.0 * 47.0 + 13.0 * 61.0,
            17.0 * 29.0 + 19.0 * 41.0 + 23.0 * 53.0,
            17.0 * 31.0 + 19.0 * 43.0 + 23.0 * 59.0,
            17.0 * 37.0 + 19.0 * 47.0 + 23.0 * 61.0,
        ]);

        assert_eq!(A * B, a_times_b);
    }
    #[test]
    fn new() {
        let a = Matrix3x3::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        assert_eq!(A, a);
        let b = Matrix3x3::from([
            Vector3d { x: 2.0, y: 3.0, z: 5.0 },
            Vector3d { x: 7.0, y: 11.0, z: 13.0 },
            Vector3d { x: 17.0, y: 19.0, z: 23.0 },
        ]);
        assert_eq!(A, b);
        let c = Matrix3x3::from((
            Vector3d { x: 2.0, y: 3.0, z: 5.0 },
            Vector3d { x: 7.0, y: 11.0, z: 13.0 },
            Vector3d { x: 17.0, y: 19.0, z: 23.0 },
        ));
        assert_eq!(A, c);
        let d: Matrix3x3<f32> = Matrix3x3::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        assert_eq!(A, d);
    }
    #[test]
    fn from_array() {
        let a = Matrix3x3::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        assert_eq!(A, a)
    }
    #[test]
    fn adjugate() {
        let a: Matrix3x3<f32> = Matrix3x3::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = a.adjugate();
        let c = a * b;
        let determinant = a.determinant();
        assert!((c / determinant).is_near_identity());
    }
    #[test]
    fn inverse() {
        let a: Matrix3x3<f32> = Matrix3x3::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = a.inverse();
        let c = a * b;
        assert!((c[0] - 1.0).abs() < f32::EPSILON);
        assert!((c[4] - 1.0).abs() < f32::EPSILON * 3.0);
        assert!((c[8] - 1.0).abs() < f32::EPSILON);
        assert!(c[1].abs() < f32::EPSILON);
        assert!(c[2].abs() < f32::EPSILON);
        assert!(c[3].abs() < f32::EPSILON);
        assert!(c[5].abs() < f32::EPSILON);
        assert!(c[6].abs() < f32::EPSILON);
        assert!(c[7].abs() < f32::EPSILON * 5.0);

        assert!(((c - Matrix3x3::one()) / 5.0).is_near_zero());
    }
}
