use cfg_if::cfg_if;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{One, Signed, Zero, float::FloatCore};

use crate::{SqrtMethods, Vector2d, Vector3dMath, Vector3dOps};

/// 3-dimensional `{x, y, z}` vector of `i8` values
pub type Vector3di8 = Vector3d<i8>;
/// 3-dimensional `{x, y, z}` vector of `i16` values
pub type Vector3di16 = Vector3d<i16>;
/// 3-dimensional `{x, y, z}` vector of `i32` values
pub type Vector3di32 = Vector3d<i32>;
/// 3-dimensional `{x, y, z}` vector of `f32` values
pub type Vector3df32 = Vector3d<f32>;
/// 3-dimensional `{x, y, z}` vector of `f64` values
pub type Vector3df64 = Vector3d<f64>;

// **** Define ****
cfg_if! {
if #[cfg(feature = "align")] {
// High-performance 16-byte aligned version
/// `Vector3d<T>`: 3D vector of type `T`.<br>
/// `Vector3d32` and `Vector3df64` and several integer aliases are provided.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector3d<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}
} else {
// Compact 12-byte version
/// `Vector3d<T>`: 3D vector of type `T`.<br>
/// `Vector3d32` and `Vector3df64` and several integer aliases are provided.
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector3d<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}
}
}

// **** Zero ****
/// Zero vector
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// # use num_traits::zero;
/// let z: Vector3df32 = zero();
///
/// assert_eq!(z, Vector3df32 { x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl<T> Zero for Vector3d<T>
where
    T: Zero + PartialEq + Vector3dOps,
{
    fn zero() -> Self {
        Self { x: T::zero(), y: T::zero(), z: T::zero() }
    }

    fn is_zero(&self) -> bool {
        self.x == T::zero() && self.y == T::zero() && self.z == T::zero()
    }
}

// **** Neg ****
/// Negate vector
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let v = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
/// let r = -v;
///
/// assert_eq!(r, Vector3df32 { x: -2.0, y: -3.0, z: -5.0 });
/// ```
impl<T> Neg for Vector3d<T>
where
    T: Vector3dOps,
{
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        T::neg(self)
    }
}

// **** Add ****
/// Add two vectors
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let u = Vector3df32::new(2.0, 3.0, 5.0);
/// let v = Vector3df32::new(7.0, 11.0, 13.0);
/// let r = u + v;
///
/// assert_eq!(r, Vector3df32 { x: 9.0, y: 14.0, z: 18.0 });
/// ```
impl<T> Add for Vector3d<T>
where
    T: Vector3dOps,
{
    type Output = Vector3d<T>;
    fn add(self, rhs: Self) -> Self {
        T::add(self, rhs)
    }
}

// **** AddAssign ****
/// Add one vector to another
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let mut r = Vector3df32::new(2.0, 3.0, 5.0);
/// let u = Vector3df32::new(7.0, 11.0, 13.0);
/// r += u;
///
/// assert_eq!(r, Vector3df32 { x: 9.0, y: 14.0, z: 18.0 });
///
/// # use num_traits::zero;
/// let z: Vector3df32 = zero();
/// let r = u + z;
/// assert_eq!(r, u);
/// ```
impl<T> AddAssign for Vector3d<T>
where
    T: Copy + Vector3dOps,
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** Sub ****
/// Subtract two vectors
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let u = Vector3df32::new(2.0, 3.0, 5.0);
/// let v = Vector3df32::new(7.0, 11.0, 13.0);
/// let r = u - v;
///
/// assert_eq!(r, Vector3df32 { x: -5.0, y: -8.0, z: -8.0 });
/// ```
impl<T> Sub for Vector3d<T>
where
    T: Add<Output = T> + Vector3dOps,
{
    type Output = Vector3d<T>;
    fn sub(self, rhs: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-rhs)
    }
}

// **** SubAssign ****
/// Subtract one vector from another
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let mut r = Vector3df32::new(2.0, 3.0, 5.0);
/// let v = Vector3df32::new(7.0, 11.0, 17.0);
/// r -= v;
///
/// assert_eq!(r, Vector3df32 { x: -5.0, y: -8.0, z: -12.0 });
/// ```
impl<T> SubAssign for Vector3d<T>
where
    T: Copy + Add<Output = T> + Vector3dOps,
{
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Mul Scalar ****
/// Pre-multiply vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let v = Vector3df32::new(2.0, 3.0, 5.0);
/// let r = 2.0 * v;
///
/// assert_eq!(r, Vector3df32 { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl Mul<Vector3d<f32>> for f32 {
    type Output = Vector3d<f32>;
    fn mul(self, rhs: Vector3d<f32>) -> Vector3d<f32> {
        Vector3d { x: self * rhs.x, y: self * rhs.y, z: self * rhs.z }
    }
}

impl Mul<Vector3d<f64>> for f64 {
    type Output = Vector3d<f64>;
    fn mul(self, rhs: Vector3d<f64>) -> Vector3d<f64> {
        Vector3d { x: self * rhs.x, y: self * rhs.y, z: self * rhs.z }
    }
}

// **** Mul ****
/// Multiply vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let v = Vector3df32::new(2.0, 3.0, 5.0);
/// let r = v * 2.0;
///
/// assert_eq!(r, Vector3df32 { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl<T> Mul<T> for Vector3d<T>
where
    T: Copy + Vector3dOps,
{
    type Output = Self;
    fn mul(self, k: T) -> Self {
        T::mul_scalar(self, k)
    }
}

impl Mul<f32> for Vector3d<i8> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i8, y: (self.y as f32 * k) as i8, z: (self.z as f32 * k) as i8 }
    }
}

impl Mul<f32> for Vector3d<i16> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i16, y: (self.y as f32 * k) as i16, z: (self.z as f32 * k) as i16 }
    }
}

impl Mul<f32> for Vector3d<i32> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i32, y: (self.y as f32 * k) as i32, z: (self.z as f32 * k) as i32 }
    }
}

// **** MulAssign ****
/// In-place multiply a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let mut v = Vector3df32::new(2.0, 3.0, 5.0);
/// v *= 2.0;
///
/// assert_eq!(v, Vector3df32 { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl<T> MulAssign<T> for Vector3d<T>
where
    T: Copy + Vector3dOps,
{
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

// **** Div by scalar ****
/// Divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let v = Vector3df32::new(2.0, 3.0, 5.0);
/// let r = v / 2.0;
///
/// assert_eq!(r, Vector3df32 { x: 1.0, y: 1.5, z: 2.5 });
/// ```
impl<T> Div<T> for Vector3d<T>
where
    T: Copy + Vector3dOps,
{
    type Output = Self;
    fn div(self, k: T) -> Self {
        T::div_scalar(self, k)
    }
}

/// In-place divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let mut v = Vector3df32::new(2.0, 3.0, 5.0);
/// v /= 2.0;
///
/// assert_eq!(v, Vector3df32 { x: 1.0, y: 1.5, z: 2.5 });
/// ```
impl<T> DivAssign<T> for Vector3d<T>
where
    T: Copy + Div<Output = T> + Vector3dOps,
{
    fn div_assign(&mut self, k: T) {
        *self = self.div(k);
    }
}

// **** Index ****
/// Access vector component by index
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let v = Vector3df32::new(2.0, 3.0, 5.0);
///
/// assert_eq!(v[0], 2.0);
/// assert_eq!(v[1], 3.0);
/// assert_eq!(v[2], 5.0);
/// ```
impl<T> Index<usize> for Vector3d<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => &self.z, // default to z component if index out of range
        }
    }
}

// **** IndexMut ****
// Set vector component by index
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let mut v = Vector3df32::new(2.0, 3.0, 5.0);
/// v[0] = 7.0;
/// v[1] = 11.0;
/// v[2] = 13.0;
///
/// assert_eq!(v, Vector3df32 { x:7.0, y:11.0, z:13.0 });
/// ```
impl<T> IndexMut<usize> for Vector3d<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => &mut self.z, // default to z component if index out of range
        }
    }
}

// **** impl new ****
impl<T> Vector3d<T>
where
    T: Copy,
{
    /// Create a vector
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

// **** impl abs ****
impl<T> Vector3d<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the vector with all components set to their absolute values
    pub fn abs(self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() }
    }

    /// Set all components of the vector to their absolute values
    pub fn abs_in_place(&mut self) {
        *self = self.abs();
    }
}

// **** impl clamp ****
impl<T> Vector3d<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the vector with all components clamped to the specified range
    pub fn clamp(self, min: T, max: T) -> Self {
        Self { x: self.x.clamp(min, max), y: self.y.clamp(min, max), z: self.z.clamp(min, max) }
    }

    /// Clamp all components of the vector to the specified range
    pub fn clamp_in_place(&mut self, min: T, max: T) {
        self.x = self.x.clamp(min, max);
        self.y = self.y.clamp(min, max);
        self.z = self.z.clamp(min, max);
    }
}

// **** impl dot and cross ****
impl<T> Vector3d<T>
where
    T: Vector3dMath + Copy,
{
    /// Vector dot product
    /// ```
    /// # use vector_quaternion_matrix::Vector3df32;
    /// let v = Vector3df32::new(2.0, 3.0, 5.0);
    /// let w = Vector3df32::new(7.0, 11.0, 13.0);
    ///
    /// let x = v.dot(w);
    ///
    /// assert_eq!(x, 112.0);
    /// ```
    #[inline(always)]
    pub fn dot(self, other: Self) -> T {
        T::dot(self, other)
    }

    /// Vector cross product
    /// ```
    /// # use vector_quaternion_matrix::Vector3df32;
    /// let v = Vector3df32::new(2.0, 3.0, 5.0);
    /// let w = Vector3df32::new(7.0, 11.0, 13.0);
    ///
    /// let x = v.cross(w);
    ///
    /// assert_eq!(x, Vector3df32::new(-16.0, 9.0, 1.0));
    /// ```
    #[inline(always)]
    pub fn cross(self, other: Self) -> Vector3d<T> {
        T::cross(self, other)
    }
}

// **** impl norm_squared ****
impl<T> Vector3d<T>
where
    T: Copy + Add<Output = T> + Vector3dOps + Vector3dMath,
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

impl<T> Vector3d<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Return the sum of all components of the vector
    pub fn sum(self) -> T {
        self.x + self.y + self.z
    }

    /// Return the product of all components of the vector
    pub fn product(self) -> T {
        self.x * self.y * self.z
    }
}

// **** impl mean ****
impl<T> Vector3d<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the vector
    pub fn mean(self) -> T {
        let three = T::one() + T::one() + T::one();
        (self.x + self.y + self.z) / three
    }
}

// **** impl norm ****
impl<T> Vector3d<T>
where
    T: Copy + Add<Output = T> + SqrtMethods + Vector3dMath + Vector3dOps,
{
    /// Return Euclidean norm
    pub fn norm(self) -> T {
        Self::norm_squared(self).sqrt()
    }
}

impl<T> Vector3d<T>
where
    T: Copy + Zero + PartialEq + SqrtMethods + Vector3dOps + Vector3dMath,
{
    /// Return normalized form of the vector
    pub fn normalized(self) -> Self {
        let norm = self.norm();
        // If norm == 0.0 then the vector is already normalized
        if norm == T::zero() {
            return self;
        }
        self * T::reciprocal(norm)
    }

    /// Normalize the vector in place
    pub fn normalize(&mut self) -> Self {
        let norm = self.norm();
        //#[allow(clippy::assign_op_pattern)]
        // If norm == 0.0 then the vector is already normalized
        if norm != T::zero() {
            *self *= T::reciprocal(norm);
        }
        *self
    }
}

impl<T> Vector3d<T>
where
    T: Copy + Zero + SqrtMethods + Vector3dMath + Vector3dOps,
{
    // Return distance between two points
    pub fn distance(self, rhs: Self) -> T {
        self.distance_squared(rhs).sqrt()
    }
}

// **** From ****
/// Vector3d from Vector2d
/// ```
/// # use vector_quaternion_matrix::{Vector2df32,Vector3df32};
/// let v = Vector3df32::from(Vector2df32 { x: 2.0, y: 3.0 });
/// let w: Vector3df32 = Vector2df32 { x: 7.0, y: 11.0 }.into();
///
/// assert_eq!(v, Vector3df32 { x: 2.0, y: 3.0, z: 0.0 });
/// assert_eq!(w, Vector3df32 { x: 7.0, y: 11.0, z: 0.0 });
/// ```
impl<T> From<Vector2d<T>> for Vector3d<T>
where
    T: Zero,
{
    fn from(vec: Vector2d<T>) -> Self {
        Self { x: vec.x, y: vec.y, z: T::zero() }
    }
}

// **** From Tuple ****
/// Vector from tuple
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let v = Vector3df32::from((2.0, 3.0, 5.0));
/// let w: Vector3df32 = (7.0, 11.0, 13.0).into();
///
/// assert_eq!(v, Vector3df32 { x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w, Vector3df32 { x: 7.0, y: 11.0, z: 13.0 });
/// ```
impl<T> From<(T, T, T)> for Vector3d<T> {
    fn from((x, y, z): (T, T, T)) -> Self {
        Self { x, y, z }
    }
}

/// Vector from array
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let v = Vector3df32::from([2.0, 3.0, 5.0]);
/// let w: Vector3df32 = [7.0, 11.0, 13.0].into();
///
/// assert_eq!(v, Vector3df32 { x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w, Vector3df32 { x: 7.0, y: 11.0, z: 13.0 });
/// ```
impl<T> From<[T; 3]> for Vector3d<T>
where
    T: Copy,
{
    fn from(v: [T; 3]) -> Self {
        Self { x: v[0], y: v[1], z: v[2] }
    }
}

/// Array from vector
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let v = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
///
/// let a = <[f32; 3]>::from(v);
/// let b: [f32; 3] = v.into();
///
/// assert_eq!(a, [2.0, 3.0, 5.0]);
/// assert_eq!(b, [2.0, 3.0, 5.0]);
/// ```
impl<T> From<Vector3d<T>> for [T; 3] {
    fn from(v: Vector3d<T>) -> Self {
        [v.x, v.y, v.z]
    }
}

/// `Vector3d<f32>` from `Vector3d<i16>`
/// ```
/// # use vector_quaternion_matrix::{Vector3df32,Vector3di16,Vector3di32};
/// let v_i16 = Vector3di16{x: 2, y: 3, z: 5};
/// let v_f32 = Vector3df32::from(v_i16);
///
/// let w_f32 = Vector3df32{x: 7.0, y: 11.0, z: 13.0};
/// let w_i16 : Vector3di16 = w_f32.into();
///
/// let u_i32 = Vector3di32{x: 17, y: 19, z: 23};
/// let u_f32 : Vector3df32 = u_i32.into();
///
/// assert_eq!(v_f32, Vector3df32 { x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w_i16, Vector3di16 { x: 7, y: 11, z: 13 });
/// assert_eq!(u_f32, Vector3df32 { x: 17.0, y: 19.0, z: 23.0 });
/// ```
impl From<Vector3d<i16>> for Vector3d<f32> {
    fn from(v: Vector3d<i16>) -> Self {
        Self { x: v.x as f32, y: v.y as f32, z: v.z as f32 }
    }
}

impl From<Vector3d<f32>> for Vector3d<i16> {
    fn from(v: Vector3d<f32>) -> Self {
        Self { x: v.x as i16, y: v.y as i16, z: v.z as i16 }
    }
}
impl From<Vector3d<i32>> for Vector3d<f32> {
    fn from(v: Vector3d<i32>) -> Self {
        Self { x: v.x as f32, y: v.y as f32, z: v.z as f32 }
    }
}

impl From<Vector3d<f32>> for Vector3d<i32> {
    fn from(v: Vector3d<f32>) -> Self {
        Self { x: v.x as i32, y: v.y as i32, z: v.z as i32 }
    }
}
