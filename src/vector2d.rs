#[allow(unused)]
use core::convert::TryFrom;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{One, Signed, Zero, float::FloatCore};

use crate::{SqrtMethods, Vector2dMath, Vector3d};

/// 2-dimensional `{x, y}` vector of `i8` values
pub type Vector2di8 = Vector2d<i8>;
/// 2-dimensional `{x, y}` vector of `i16` values
pub type Vector2di16 = Vector2d<i16>;
/// 2-dimensional `{x, y}` vector of `i32` values
pub type Vector2di32 = Vector2d<i32>;
/// 2-dimensional `{x, y}` vector of `f32` values
pub type Vector2df32 = Vector2d<f32>;
/// 2-dimensional `{x, y}` vector of `f64` values
pub type Vector2df64 = Vector2d<f64>;

// **** Define ****
/// `Vector2d<T>`: 2D vector of type `T`.<br>
/// `Vector2d32` and `Vector2df64` and several integer aliases are provided.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector2d<T> {
    pub x: T,
    pub y: T,
}

// **** Zero ****
/// Zero vector
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// # use num_traits::zero;
/// let z: Vector2df32 = zero();
///
/// assert_eq!(z, Vector2df32 { x: 0.0, y: 0.0 });
/// ```
impl<T> Zero for Vector2d<T>
where
    T: Zero + PartialEq + Vector2dMath,
{
    fn zero() -> Self {
        Self { x: T::zero(), y: T::zero() }
    }

    fn is_zero(&self) -> bool {
        self.x == T::zero() && self.y == T::zero()
    }
}

// **** Neg ****
/// Negate vector
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let v = Vector2df32 { x: 2.0, y: 3.0 };
/// let r = -v;
///
/// assert_eq!(r, Vector2df32 { x: -2.0, y: -3.0 });
/// ```
impl<T> Neg for Vector2d<T>
where
    T: Vector2dMath,
{
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        T::v2_neg(self)
    }
}

// **** Add ****
/// Add two vectors
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let u = Vector2df32::new(2.0, 3.0);
/// let v = Vector2df32::new(7.0, 11.0);
/// let r = u + v;
///
/// assert_eq!(r, Vector2df32 { x: 9.0, y: 14.0 });
/// ```
impl<T> Add for Vector2d<T>
where
    T: Vector2dMath,
{
    type Output = Vector2d<T>;
    fn add(self, rhs: Self) -> Self {
        T::v2_add(self, rhs)
    }
}

// **** AddAssign ****
/// Add one vector to another
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let mut r = Vector2df32::new(2.0, 3.0);
/// let u = Vector2df32::new(7.0, 11.0);
/// r += u;
///
/// assert_eq!(r, Vector2df32 { x: 9.0, y: 14.0 });
///
/// # use num_traits::zero;
/// let z: Vector2df32 = zero();
/// let r = u + z;
/// assert_eq!(r, u);
/// ```
impl<T> AddAssign for Vector2d<T>
where
    T: Copy + Vector2dMath,
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** Sub ****
/// Subtract two vectors
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let u = Vector2df32::new(2.0, 3.0);
/// let v = Vector2df32::new(7.0, 11.0);
/// let r = u - v;
///
/// assert_eq!(r, Vector2df32 { x: -5.0, y: -8.0 });
/// ```
impl<T> Sub for Vector2d<T>
where
    T: Add<Output = T> + Vector2dMath,
{
    type Output = Vector2d<T>;
    fn sub(self, rhs: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-rhs)
    }
}

// **** SubAssign ****
/// Subtract one vector from another
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let mut r = Vector2df32::new(2.0, 3.0);
/// let v = Vector2df32::new(7.0, 11.0);
/// r -= v;
///
/// assert_eq!(r, Vector2df32 { x: -5.0, y: -8.0 });
/// ```
impl<T> SubAssign for Vector2d<T>
where
    T: Copy + Add<Output = T> + Vector2dMath,
{
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Mul Scalar ****
/// Pre-multiply vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let v = Vector2df32::new(2.0, 3.0);
/// let r = 2.0 * v;
///
/// assert_eq!(r, Vector2df32 { x: 4.0, y: 6.0 });
/// ```
impl Mul<Vector2d<f32>> for f32 {
    type Output = Vector2d<f32>;
    fn mul(self, rhs: Vector2d<f32>) -> Vector2d<f32> {
        Vector2d { x: self * rhs.x, y: self * rhs.y }
    }
}

impl Mul<Vector2d<f64>> for f64 {
    type Output = Vector2d<f64>;
    fn mul(self, rhs: Vector2d<f64>) -> Vector2d<f64> {
        Vector2d { x: self * rhs.x, y: self * rhs.y }
    }
}

// **** Mul ****
/// Multiply vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let v = Vector2df32::new(2.0, 3.0);
/// let r = v * 2.0;
///
/// assert_eq!(r, Vector2df32 { x: 4.0, y: 6.0 });
/// ```
impl<T> Mul<T> for Vector2d<T>
where
    T: Copy + Vector2dMath,
{
    type Output = Self;
    fn mul(self, k: T) -> Self {
        T::v2_mul_scalar(self, k)
    }
}

impl Mul<f32> for Vector2d<i8> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i8, y: (self.y as f32 * k) as i8 }
    }
}

impl Mul<f32> for Vector2d<i16> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i16, y: (self.y as f32 * k) as i16 }
    }
}

impl Mul<f32> for Vector2d<i32> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i32, y: (self.y as f32 * k) as i32 }
    }
}

// **** MulAssign ****
/// In-place multiply a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let mut v = Vector2df32::new(2.0, 3.0);
/// v *= 2.0;
///
/// assert_eq!(v, Vector2df32 { x: 4.0, y: 6.0 });
/// ```
impl<T> MulAssign<T> for Vector2d<T>
where
    T: Copy + Vector2dMath,
{
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

// **** Div by scalar ****
/// Divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let v = Vector2df32::new(2.0, 3.0);
/// let r = v / 2.0;
///
/// assert_eq!(r, Vector2df32 { x: 1.0, y: 1.5 });
/// ```
impl<T> Div<T> for Vector2d<T>
where
    T: Copy + Vector2dMath,
{
    type Output = Self;
    fn div(self, k: T) -> Self {
        T::v2_div_scalar(self, k)
    }
}

/// In-place divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let mut v = Vector2df32::new(2.0, 3.0);
/// v /= 2.0;
///
/// assert_eq!(v, Vector2df32 { x: 1.0, y: 1.5 });
/// ```
impl<T> DivAssign<T> for Vector2d<T>
where
    T: Copy + Div<Output = T> + Vector2dMath,
{
    fn div_assign(&mut self, k: T) {
        *self = self.div(k);
    }
}

// **** Index ****
/// Access vector component by index
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let v = Vector2df32::new(2.0, 3.0);
///
/// assert_eq!(v[0], 2.0);
/// assert_eq!(v[1], 3.0);
/// ```
impl<T> Index<usize> for Vector2d<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => &self.y, // default to z component if index out of range
        }
    }
}

// **** IndexMut ****
// Set vector component by index
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let mut v = Vector2df32::new(2.0, 3.0);
/// v[0] = 7.0;
/// v[1] = 11.0;
///
/// assert_eq!(v, Vector2df32 { x:7.0, y:11.0 });
/// ```
impl<T> IndexMut<usize> for Vector2d<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.y, // default to z component if index out of range
        }
    }
}

// **** impl new ****
impl<T> Vector2d<T>
where
    T: Copy,
{
    /// Create a vector
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

// **** impl abs ****
impl<T> Vector2d<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the vector with all components set to their absolute values
    pub fn abs(self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs() }
    }

    /// Set all components of the vector to their absolute values
    pub fn abs_in_place(&mut self) {
        *self = self.abs();
    }
}

// **** impl clamp ****
impl<T> Vector2d<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the vector with all components clamped to the specified range
    pub fn clamp(self, min: T, max: T) -> Self {
        Self { x: self.x.clamp(min, max), y: self.y.clamp(min, max) }
    }

    /// Clamp all components of the vector to the specified range
    pub fn clamp_in_place(&mut self, min: T, max: T) {
        self.x = self.x.clamp(min, max);
        self.y = self.y.clamp(min, max);
    }
}

// **** impl dot and cross ****
impl<T> Vector2d<T>
where
    T: Vector2dMath + Copy,
{
    /// Vector dot product
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(2.0, 3.0);
    /// let w = Vector2df32::new(7.0, 11.0);
    ///
    /// let x = v.dot(w);
    ///
    /// assert_eq!(x, 47.0);
    /// ```
    #[inline(always)]
    pub fn dot(self, other: Self) -> T {
        T::v2_dot(self, other)
    }
}

impl<T> Vector2d<T>
where
    T: Copy + Sub<Output = T> + Mul<Output = T>,
{
    /// Z component of vector cross product of self and rhs extended to 3D
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(2.0, 3.0);
    /// let w = Vector2df32::new(7.0, 11.0);
    ///
    /// let x = v.cross(w);
    ///
    /// assert_eq!(x, 1.0);
    /// ```
    #[inline(always)]
    pub fn cross(self, rhs: Self) -> T {
        self.x * rhs.y - self.y * rhs.x
    }
}

// **** impl norm_squared ****
impl<T> Vector2d<T>
where
    T: Copy + Add<Output = T> + Vector2dMath + Vector2dMath,
{
    /// Return square of Euclidean norm
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }

    /// Return distance between two points, squared
    pub fn distance_squared(self, rhs: Self) -> T {
        (self - rhs).norm_squared()
    }
}

impl<T> Vector2d<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Return the sum of all components of the vector
    pub fn sum(self) -> T {
        self.x + self.y
    }

    /// Return the product of all components of the vector
    pub fn product(self) -> T {
        self.x * self.y
    }
}

// **** impl mean ****
impl<T> Vector2d<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the vector
    pub fn mean(self) -> T {
        (self.x + self.y) / (T::one() + T::one())
    }
}

// **** impl norm ****
impl<T> Vector2d<T>
where
    T: Copy + Add<Output = T> + SqrtMethods + Vector2dMath + Vector2dMath,
{
    /// Return Euclidean norm
    pub fn norm(self) -> T {
        Self::norm_squared(self).sqrt()
    }
}

impl<T> Vector2d<T>
where
    T: Copy + Zero + PartialEq + SqrtMethods + Vector2dMath + Vector2dMath,
{
    /// Return normalized form of the vector
    pub fn normalized(self) -> Self {
        let norm = self.norm();
        // If norm == 0.0 then the vector is already normalized
        if norm == T::zero() {
            return self;
        }
        self * T::v2_reciprocal(norm)
    }

    /// Normalize the vector in place
    pub fn normalize(&mut self) -> Self {
        let norm = self.norm();
        #[allow(clippy::assign_op_pattern)]
        // If norm == 0.0 then the vector is already normalized
        if norm != T::zero() {
            *self *= T::v2_reciprocal(norm);
        }
        *self
    }
}

impl<T> Vector2d<T>
where
    T: Copy + Zero + SqrtMethods + Vector2dMath + Vector2dMath,
{
    // Return distance between two points
    pub fn distance(self, rhs: Self) -> T {
        self.distance_squared(rhs).sqrt()
    }
}

// **** From ****
/// Vector2d from Vector3d, discarding z value.
/// ```
/// # use vector_quaternion_matrix::{Vector2df32,Vector3df32};
/// let v: Vector2df32 = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 }.into();
/// let u = Vector2df32::from(Vector3df32{ x: 7.0, y: 11.0, z: 13.0 });
///
/// assert_eq!(v, Vector2df32 { x: 2.0, y: 3.0 });
/// assert_eq!(u, Vector2df32 { x: 7.0, y: 11.0 });
impl<T> From<Vector3d<T>> for Vector2d<T>
where
    T: Zero,
{
    fn from(v: Vector3d<T>) -> Self {
        Vector2d::<T> { x: v.x, y: v.y }
    }
}

/*
/// Non-zero z component error.
#[derive(Debug, PartialEq)]
pub enum VectorError {
    NonZeroZ,
}

/// Vector2d try_from Vector3d
/// ```
/// # use vector_quaternion_matrix::{Vector2df32,Vector3df32};
/// let result: Result<Vector2df32, _> = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 }.try_into();
/// match result {
///     Ok(v2) => assert!(false),
///     Err(_) => assert!(true),
/// }
/// let result: Result<Vector2df32, _> = Vector2df32::try_from(Vector3df32{ x: 2.0, y: 3.0, z: 5.0 });
/// match result {
///     Ok(v2) => assert!(false),
///     Err(_) => assert!(true),
/// }
/// let result: Result<Vector2df32, _> = Vector3df32 { x: 2.0, y: 3.0, z: 0.0 }.try_into();
/// match result {
///     Ok(v2d) => assert_eq!(v2d, Vector2df32 { x: 2.0, y: 3.0 }),
///     Err(_) => assert!(false),
/// }
/// let result: Result<Vector2df32, _> = Vector2df32::try_from(Vector3df32{ x: 2.0, y: 3.0, z: 0.0 });
/// match result {
///     Ok(v2d) => assert_eq!(v2d, Vector2df32 { x: 2.0, y: 3.0 }),
///     Err(_) => assert!(false),
/// }
/// ```
impl<T> TryFrom<Vector3d<T>> for Vector2d<T>
where
    T: Zero + PartialEq,
{
    type Error = VectorError;

    fn try_from(v: Vector3d<T>) -> Result<Self, Self::Error> {
        // In embedded/control systems, exact float comparison (== 0.0)
        // is usually fine for a "pure" check, but you can also use
        // a small epsilon if the Z comes from a calculation.
        if v.z == T::zero() { Ok(Vector2d::<T> { x: v.x, y: v.y }) } else { Err(VectorError::NonZeroZ) }
    }
}
*/

// **** From Tuple ****
/// Vector from tuple
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let v = Vector2df32::from((2.0, 3.0));
/// let w: Vector2df32 = (7.0, 11.0).into();
///
/// assert_eq!(v, Vector2df32 { x: 2.0, y: 3.0 });
/// assert_eq!(w, Vector2df32 { x: 7.0, y: 11.0 });
/// ```
impl<T> From<(T, T)> for Vector2d<T> {
    fn from((x, y): (T, T)) -> Self {
        Self { x, y }
    }
}

/// Vector from array
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let v = Vector2df32::from([2.0, 3.0]);
/// let w: Vector2df32 = [7.0, 11.0].into();
///
/// assert_eq!(v, Vector2df32 { x: 2.0, y: 3.0 });
/// assert_eq!(w, Vector2df32 { x: 7.0, y: 11.0 });
/// ```
impl<T> From<[T; 2]> for Vector2d<T>
where
    T: Copy,
{
    fn from(v: [T; 2]) -> Self {
        Self { x: v[0], y: v[1] }
    }
}

/// Array from vector
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let v = Vector2df32 { x: 2.0, y: 3.0 };
///
/// let a = <[f32; 2]>::from(v);
/// let b: [f32; 2] = v.into();
///
/// assert_eq!(a, [2.0, 3.0]);
/// assert_eq!(b, [2.0, 3.0]);
/// ```
impl<T> From<Vector2d<T>> for [T; 2] {
    fn from(v: Vector2d<T>) -> Self {
        [v.x, v.y]
    }
}

/// `Vector2d<f32>` from `Vector2d<i16>`
/// ```
/// # use vector_quaternion_matrix::{Vector2df32,Vector2di16,Vector2di32};
/// let v_i16 = Vector2di16{x: 2, y: 3};
/// let v_f32 = Vector2df32::from(v_i16);
///
/// let w_f32 = Vector2df32{x: 7.0, y: 11.0};
/// let w_i16 : Vector2di16 = w_f32.into();
///
/// let u_i32 = Vector2di32{x: 17, y: 19};
/// let u_f32 : Vector2df32 = u_i32.into();
///
/// assert_eq!(v_f32, Vector2df32 { x: 2.0, y: 3.0 });
/// assert_eq!(w_i16, Vector2di16 { x: 7, y: 11 });
/// assert_eq!(u_f32, Vector2df32 { x: 17.0, y: 19.0 });
/// ```
impl From<Vector2d<i16>> for Vector2d<f32> {
    fn from(v: Vector2d<i16>) -> Self {
        Self { x: v.x as f32, y: v.y as f32 }
    }
}

impl From<Vector2d<f32>> for Vector2d<i16> {
    fn from(v: Vector2d<f32>) -> Self {
        Self { x: v.x as i16, y: v.y as i16 }
    }
}
impl From<Vector2d<i32>> for Vector2d<f32> {
    fn from(v: Vector2d<i32>) -> Self {
        Self { x: v.x as f32, y: v.y as f32 }
    }
}

impl From<Vector2d<f32>> for Vector2d<i32> {
    fn from(v: Vector2d<f32>) -> Self {
        Self { x: v.x as i32, y: v.y as i32 }
    }
}
