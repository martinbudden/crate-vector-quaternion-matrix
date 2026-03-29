use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{One, Signed, Zero, float::FloatCore};

use crate::{MathConstants, Vector2d};

/// 2x2 matrix of `f32` values
pub type Matrix2x2f32 = Matrix2x2<f32>;
/// 2x2 matrix of `f64` values
pub type Matrix2x2f64 = Matrix2x2<f64>;

/// Zero determinant error.
#[derive(Debug, PartialEq)]
pub enum MatrixError {
    ZeroDeterminant,
}

// **** Define ****
/// `Matrix2x2<T>`: 2x2 Matrix of type `T`.<br>
/// Aliases `Matrix2x2f32` and `Matrix2x2f64` are provided.<br>
/// Internal implementation is a flattened 1x2 matrix: an array of 4 elements stored in row-major order<br>
/// That is the element `m[row][col]` is at array position `[row * 2 + col]`, so element `m01` is at `a[1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix2x2<T> {
    // Flattened 2x2 matrix: 4 elements in row-major order
    a: [T; 4],
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
    T: Copy + Zero + PartialEq,
{
    fn zero() -> Self {
        Self { a: [T::zero(), T::zero(), T::zero(), T::zero()] }
    }
    fn is_zero(&self) -> bool {
        self.a.iter().all(|&x| x == T::zero())
    }
}

// **** One ****
/// Identity matrix
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// # use num_traits::One;
/// let I = Matrix2x2f32::one();
///
/// assert_eq!(I, Matrix2x2f32::from([ 1.0, 0.0,
///                                    0.0, 1.0]));
/// ```
impl<T> One for Matrix2x2<T>
where
    T: Copy + Zero + One + PartialEq + Sub<Output = T> + Mul<Output = T>,
{
    fn one() -> Self {
        Self { a: [T::one(), T::zero(), T::zero(), T::one()] }
    }

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

// **** NegReference ****
/// Negate quaternion reference
/// ```
/// # use vector_quaternion_matrix::Matrix2x2f32;
/// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
///                                  7.0, 11.0]);
/// m = - m;
///
/// assert_eq!(m, Matrix2x2f32::from([ -2.0,  -3.0,
///                                    -7.0, -11.0]));
/// ```
impl<T> Neg for &Matrix2x2<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Matrix2x2<T>;
    fn neg(self) -> Self::Output {
        let mut a = self.a;
        for r in a.iter_mut() {
            *r = -*r;
        }
        Matrix2x2 { a }
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
    fn mul(self, rhs: Matrix2x2<f32>) -> Matrix2x2<f32> {
        let mut a = rhs.a;
        for r in a.iter_mut() {
            *r *= self;
        }
        Matrix2x2f32 { a }
    }
}

impl Mul<Matrix2x2<f64>> for f64 {
    type Output = Matrix2x2<f64>;
    fn mul(self, rhs: Matrix2x2<f64>) -> Matrix2x2<f64> {
        let mut a = rhs.a;
        for r in a.iter_mut() {
            *r *= self;
        }
        Matrix2x2::<f64> { a }
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
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector2d<T>;
    fn mul(self, v: Vector2d<T>) -> Vector2d<T> {
        Vector2d::<T> { x: self.a[0] * v.x + self.a[1] * v.y, y: self.a[2] * v.x + self.a[3] * v.y }
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
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, rhs: Matrix2x2<T>) -> Self {
        Self { x: self.x * rhs.a[0] + self.y * rhs.a[2], y: self.x * rhs.a[1] + self.y * rhs.a[3] }
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
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let a = [
            self.a[0] * rhs.a[0] + self.a[1] * rhs.a[2],
            self.a[0] * rhs.a[1] + self.a[1] * rhs.a[3],
            self.a[2] * rhs.a[0] + self.a[3] * rhs.a[2],
            self.a[2] * rhs.a[1] + self.a[3] * rhs.a[3],
        ];
        Matrix2x2::<T> { a }
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
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: Matrix2x2<T>) {
        *self = *self * rhs;
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
    T: Copy + One + Div<Output = T>,
{
    type Output = Self;
    fn div(self, k: T) -> Self {
        let reciprocal: T = T::one() / k;
        let mut a = self.a;
        for r in a.iter_mut() {
            *r = *r * reciprocal;
        }
        Matrix2x2::<T> { a }
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
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.a[row * 2 + col]
    }
}

// **** impl new ****
impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Create a matrix
    pub const fn new(input: [T; 4]) -> Self {
        Self { a: input }
    }
}

// **** impl abs ****
impl<T> Matrix2x2<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the matrix with all components set to their absolute values
    pub fn abs(&self) -> Self {
        let mut a = self.a;
        for a in a.iter_mut() {
            *a = a.abs();
        }
        Self { a }
    }

    /// Set all components of the matrix to their absolute values
    pub fn abs_in_place(&mut self) {
        for a in self.a.iter_mut() {
            *a = a.abs();
        }
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
// **** impl clamp ****
impl<T> Matrix2x2<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all components clamped to the specified range
    pub fn clamp(&self, min: T, max: T) -> Self {
        let mut a = self.a;
        for a in a.iter_mut() {
            *a = a.clamp(min, max);
        }
        Self { a }
    }

    /// Clamp all components of the matrix to the specified range
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
    /// let A = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let v = A.row(0);
    ///
    /// assert_eq!(v, Vector2df32{ x: 2.0, y: 3.0 });
    /// assert_eq!(A.row(1), Vector2df32{ x: 7.0, y: 11.0 });
    /// ```
    pub fn row(&self, row: usize) -> Vector2d<T> {
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
    /// let A = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let v = A.column(0);
    ///
    /// assert_eq!(v, Vector2df32{ x: 2.0, y: 7.0 });
    /// assert_eq!(A.column(1), Vector2df32{ x: 3.0, y: 11.0 });
    /// ```
    pub fn column(&self, column: usize) -> Vector2d<T> {
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
    pub fn transpose(&self) -> Self {
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
    pub fn transpose_in_place(&mut self) {
        *self = self.transpose();
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + Zero + One + Neg<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
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
    pub fn adjugate(&self) -> Self {
        Self { a: [self.a[3], -self.a[1], -self.a[2], self.a[0]] }
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
    pub fn adjugate_in_place(&mut self) {
        *self = self.adjugate();
    }
    /// Add vector to diagonal of matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::{Matrix2x2f32,Vector2df32};
    /// let mut A = Matrix2x2f32::from([ 2.0,  3.0,
    ///                                  7.0, 11.0]);
    ///
    /// let v = Vector2df32{ x: 10.0, y: 20.0 };
    ///
    /// A.add_to_diagonal_in_place(v);
    ///
    /// assert_eq!(A, Matrix2x2f32::from([ 12.0,  3.0,
    ///                                     7.0, 31.0]));
    /// ```
    pub fn add_to_diagonal_in_place(&mut self, v: Vector2d<T>) {
        self.a[0] = self.a[0] + v.x;
        self.a[3] = self.a[3] + v.y;
    }

    /// Subtract vector from diagonal of matrix, in-place
    /// ```
    /// # use vector_quaternion_matrix::{Matrix2x2f32,Vector2df32};
    /// let mut A = Matrix2x2f32::from([ 2.0,  3.0,
    ///                                  7.0, 11.0]);
    ///
    /// let v = Vector2df32{ x: 10.0, y: 20.0 };
    ///
    /// A.subtract_from_diagonal_in_place(v);
    ///
    /// assert_eq!(A, Matrix2x2f32::from([ -8.0,  3.0,
    ///                                     7.0, -9.0]));
    /// ```
    pub fn subtract_from_diagonal_in_place(&mut self, v: Vector2d<T>) {
        self.a[0] = self.a[0] - v.x;
        self.a[3] = self.a[3] - v.y;
    }

    /// Invert matrix in-place, assuming it is a diagonal matrix
    pub fn invert_in_place_assuming_diagonal(&mut self) {
        self.a[0] = T::one() / self.a[0];
        self.a[3] = T::one() / self.a[3];
    }

    /// Return inverse of matrix, assuming it is diagonal
    pub fn inverse_assuming_diagonal(&self) -> Self {
        let mut ret = *self;
        ret.invert_in_place_assuming_diagonal();
        ret
    }

    /// Matrix determinant
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let A = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let d = A.determinant();
    ///
    /// assert_eq!(2.0*11.0 - 3.0*7.0, d);
    ///
    /// ```
    pub fn determinant(&self) -> T {
        self.a[0] * self.a[3] - self.a[1] * self.a[2]
    }

    /// Return the sum of all components of the matrix
    /// ```
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let A = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 11.0]);
    /// let s = A.sum();
    ///
    /// assert_eq!(s, 23.0);
    /// ```
    pub fn sum(&self) -> T {
        self.a[0] + self.a[1] + self.a[2] + self.a[3]
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
    pub fn mean(&self) -> T {
        self.sum() / (T::one() + T::one() + T::one() + T::one())
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
    pub fn product(&self) -> T {
        self.a[0] * self.a[1] * self.a[2] * self.a[3]
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
    pub fn trace(&self) -> T {
        self.a[0] + self.a[3]
    }
}

impl<T> Matrix2x2<T>
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
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                                  7.0, 11.0]);
    /// m.invert_in_place();
    ///
    /// ```
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
    pub fn inverse(&self) -> Self {
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
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0,  3.0,
    ///                              7.0, 10.5]);
    /// let n = m.try_inverse();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(None, n);
    ///
    /// ```
    pub fn try_inverse(&self) -> Option<Self> {
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
    /// # use vector_quaternion_matrix::Matrix2x2f32;
    /// # use num_traits::One;
    /// let I = Matrix2x2f32::one();
    /// assert!(I.is_near_identity());
    /// ```
    pub fn is_near_identity(&self) -> bool {
        if self.a[1].abs() > T::EPSILON || self.a[2].abs() > T::EPSILON {
            return false;
        }
        if (self.a[0] - T::one()).abs() > T::EPSILON || (self.a[3] - T::one()).abs() > T::EPSILON {
            return false;
        }
        true
    }
}

// **** From Array ****
/// Matrix from array
impl<T> From<[T; 4]> for Matrix2x2<T>
where
    T: Copy,
{
    fn from(input: [T; 4]) -> Self {
        Self { a: input }
    }
}

impl<T> From<[Vector2d<T>; 2]> for Matrix2x2<T>
where
    T: Copy,
{
    fn from(v: [Vector2d<T>; 2]) -> Self {
        Self { a: [v[0].x, v[0].y, v[1].x, v[1].y] }
    }
}

impl<T> From<(Vector2d<T>, Vector2d<T>)> for Matrix2x2<T> {
    fn from(v: (Vector2d<T>, Vector2d<T>)) -> Self {
        Self { a: [v.0.x, v.0.y, v.1.x, v.1.y] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    const A: Matrix2x2<f32> = Matrix2x2f32 { a: [2.0, 3.0, 5.0, 7.0] };
    const B: Matrix2x2<f32> = Matrix2x2f32 { a: [29.0, 31.0, 37.0, 41.0] };

    #[test]
    fn normal_types() {
        is_normal::<Matrix2x2<f32>>();
    }
    #[test]
    fn default() {
        let a: Matrix2x2<f32> = Matrix2x2f32::default();
        assert_eq!(a, Matrix2x2f32 { a: [0.0, 0.0, 0.0, 0.0] });
        let z = Matrix2x2f32::zero();
        //let z: Matrix2x2 = zero();
        assert_eq!(a, z);
        assert!(z.is_zero());
        assert!(!z.is_one());
        assert!(z.is_near_zero());

        let i = Matrix2x2f32::one();
        //let i: Matrix2x2 = one();
        assert!(i.is_one());
        assert!(!i.is_zero());
        assert!(i.is_near_identity());
    }
    #[test]
    fn neg() {
        assert_eq!(-A, Matrix2x2f32 { a: [-2.0, -3.0, -5.0, -7.0] });

        let b = -A;
        assert_eq!(b, Matrix2x2f32 { a: [-2.0, -3.0, -5.0, -7.0] });
    }
    #[test]
    fn add() {
        let a_plus_b = Matrix2x2f32 { a: [2.0 + 29.0, 3.0 + 31.0, 5.0 + 37.0, 7.0 + 41.0] };
        assert_eq!(A + B, a_plus_b);
    }
    #[test]
    fn sub() {
        let a_minus_b = Matrix2x2::from([2.0 - 29.0, 3.0 - 31.0, 5.0 - 37.0, 7.0 - 41.0]);
        assert_eq!(A - B, a_minus_b);
    }
    #[test]
    fn mul() {
        let a_times_b = Matrix2x2::from([
            2.0 * 29.0 + 3.0 * 37.0,
            2.0 * 31.0 + 3.0 * 41.0,
            5.0 * 29.0 + 7.0 * 37.0,
            5.0 * 31.0 + 7.0 * 41.0,
        ]);

        assert_eq!(A * B, a_times_b);
    }
    #[test]
    fn new() {
        let a = Matrix2x2::from([2.0, 3.0, 5.0, 7.0]);
        assert_eq!(A, a);
        let b = Matrix2x2::from([Vector2d { x: 2.0, y: 3.0 }, Vector2d { x: 5.0, y: 7.0 }]);
        assert_eq!(A, b);
        let c = Matrix2x2::from((Vector2d { x: 2.0, y: 3.0 }, Vector2d { x: 5.0, y: 7.0 }));
        assert_eq!(A, c);
        let d: Matrix2x2<f32> = Matrix2x2::new([2.0, 3.0, 5.0, 7.0]);
        assert_eq!(A, d);
    }
    #[test]
    fn from_array() {
        let a = Matrix2x2::from([2.0, 3.0, 5.0, 7.0]);
        assert_eq!(A, a)
    }
    #[test]
    fn adjugate() {
        let a: Matrix2x2<f32> = Matrix2x2::from([2.0, 3.0, 5.0, 7.0]);
        let b = a.adjugate();
        let c = a * b;
        let determinant = a.determinant();
        assert!((c / determinant).is_near_identity());
    }
    #[test]
    fn inverse() {
        let a: Matrix2x2<f32> = Matrix2x2::from([2.0, 3.0, 5.0, 7.0]);
        let b = a.inverse();
        let c = a * b;
        assert_eq!(1.0, c[0]);
        assert_eq!(0.0, c[1]);
        assert_eq!(0.0, c[2]);
        assert_eq!(1.0, c[3]);
        //assert!((c[0] - 1.0).abs() < f32::EPSILON*3.0);
        //assert!((c[3] - 1.0).abs() < f32::EPSILON * 3.0);
        //assert!(c[1].abs() < f32::EPSILON);
        //assert!(c[2].abs() < f32::EPSILON);

        //assert!(((c - Matrix2x2::one()) / 5.0).is_near_zero());
    }
}
