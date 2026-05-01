use cfg_if::cfg_if;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};

use crate::{MathConstants, Quaternion, QuaternionMath, SqrtMethods, Vector2d, Vector3dMath};

/// 3-dimensional `{x, y, z}` vector of `f32` values<br>
pub type Vector3df32 = Vector3d<f32>;
/// 3-dimensional `{x, y, z}` vector of `f64` values<br><br>
pub type Vector3df64 = Vector3d<f64>;

// **** Define ****

cfg_if! {
if #[cfg(feature = "no_align")] {
// Compact 12-byte version

/// `Vector3d<T>`: 3D vector of type `T`.<br>
/// Aliases `Vector3df32`, `Vector3df64`, `Vector3di16`, and `Vector3di32` are provided.<br><br>
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector3d<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}
} else {
// High-performance 16-byte aligned version, enables use of SIMD

/// `Vector3d<T>`: 3D vector of type `T`.<br>
/// Aliases `Vector3df32`, `Vector3df64`, `Vector3di16`, and `Vector3di32`  are provided.<br><br>
/// `Vector3df32` uses **SIMD** accelerations implemented in `Vector3dMath`.<br><br>
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector3d<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}
}
}

// **** New ****

/// Create a vector.
/// ```
/// # use vqm::Vector3df32;
/// let v = Vector3df32::new(2.0,  3.0, 7.0);
/// assert_eq!(v, Vector3df32 { x:2.0, y:3.0, z: 7.0 });
/// ```
impl<T> Vector3d<T>
where
    T: Copy,
{
    /// Create a vector.
    #[inline]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

// **** Zero ****

/// Zero vector.
/// ```
/// # use vqm::Vector3df32;
/// # use num_traits::{zero,Zero};
/// let z: Vector3df32 = zero();
/// assert!(z.is_zero());
/// assert_eq!(z, Vector3df32 { x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl<T> Zero for Vector3d<T>
where
    T: Copy + Zero + PartialEq + Vector3dMath,
{
    #[inline]
    fn zero() -> Self {
        Self { x: T::zero(), y: T::zero(), z: T::zero() }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.x == T::zero() && self.y == T::zero() && self.z == T::zero()
    }
}

// **** Neg ****

/// Negate vector.
/// ```
/// # use vqm::Vector3df32;
/// let v = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
/// let r = -v;
///
/// assert_eq!(r, Vector3df32 { x: -2.0, y: -3.0, z: -5.0 });
/// ```
impl<T> Neg for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        T::v3_neg(self)
    }
}

// **** Add ****

/// Add two vectors.
/// ```
/// # use vqm::Vector3df32;
/// let u = Vector3df32::new(2.0, 5.0, 11.0);
/// let v = Vector3df32::new(3.0, 7.0, 13.0);
/// let r = u + v;
///
/// assert_eq!(r, Vector3df32 { x: 5.0, y: 12.0, z: 24.0 });
/// ```
impl<T> Add for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        T::v3_add(self, other)
    }
}

// **** AddAssign ****

/// Add one vector to another.
/// ```
/// # use vqm::Vector3df32;
/// let mut r = Vector3df32::new(2.0, 5.0, 11.0);
/// let u = Vector3df32::new(3.0, 7.0, 13.0);
/// r += u;
///
/// assert_eq!(r, Vector3df32 { x: 5.0, y: 12.0, z: 24.0 });
///
/// # use num_traits::zero;
/// let z: Vector3df32 = zero();
/// let r = u + z;
/// assert_eq!(r, u);
/// ```
impl<T> AddAssign for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

/// Multiply vector by constant and add another vector.
/// ```
/// # use vqm::Vector3df32;
/// # use num_traits::MulAdd;
/// let mut v = Vector3df32::new(2.0, 5.0, 11.0);
/// let w = Vector3df32::new(3.0, 7.0, 13.0);
/// let k = 23.0;
/// let r = v.mul_add(k, w);
///
/// assert_eq!(r, Vector3df32 { x: 49.0, y: 122.0, z: 266.0 });
/// ```
impl<T> MulAdd<T> for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    type Output = Self;

    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::v3_mul_add(self, k, other)
    }
}

impl MulAdd<i32> for Vector3d<i16> {
    type Output = Self;

    #[inline]
    fn mul_add(self, k: i32, other: Self) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Vector3d {
            x: self.x * (k as i16) + other.x,
            y: self.y * (k as i16) + other.y,
            z: self.z * (k as i16) + other.z,
        }
    }
}

impl MulAdd<f32> for Vector3d<i16> {
    type Output = Self;

    #[inline]
    fn mul_add(self, k: f32, other: Self) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Vector3d {
            x: self.x * (k as i16) + other.x,
            y: self.y * (k as i16) + other.y,
            z: self.z * (k as i16) + other.z,
        }
    }
}

// **** MulAddAssign ****

/// Multiply vector by constant and add another vector in place.
/// ```
/// # use vqm::Vector3df32;
/// # use num_traits::MulAddAssign;
/// let mut v = Vector3df32::new(2.0, 5.0, 11.0);
/// let w = Vector3df32::new(3.0, 7.0, 13.0);
/// let k = 23.0;
/// v.mul_add_assign(k, w);
///
/// assert_eq!(v, Vector3df32 { x: 49.0, y: 122.0, z: 266.0 });
/// ```
impl<T> MulAddAssign<T> for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

/// Subtract two vectors.
/// ```
/// # use vqm::Vector3df32;
/// let u = Vector3df32::new(2.0, 5.0, 13.0);
/// let v = Vector3df32::new(3.0, 7.0, 11.0);
/// let r = u - v;
///
/// assert_eq!(r, Vector3df32 { x: -1.0, y: -2.0, z: 2.0 });
/// ```
impl<T> Sub for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

/// Subtract one vector from another.
/// ```
/// # use vqm::Vector3df32;
/// let mut r = Vector3df32::new(2.0, 5.0, 13.0);
/// let     v = Vector3df32::new(3.0, 7.0, 11.0);
/// r -= v;
///
/// assert_eq!(r, Vector3df32 { x: -1.0, y: -2.0, z: 2.0 });
/// ```
impl<T> SubAssign for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Scalar Mul ****

/// Pre-multiply vector by a constant.
/// ```
/// # use vqm::Vector3df32;
/// let v = Vector3df32::new(2.0, 3.0, 5.0);
/// let r = 2.0 * v;
///
/// assert_eq!(r, Vector3df32 { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl Mul<Vector3d<f32>> for f32 {
    type Output = Vector3d<f32>;
    #[inline]
    fn mul(self, other: Vector3d<f32>) -> Vector3d<f32> {
        f32::v3_mul_scalar(other, self)
    }
}

impl Mul<Vector3d<f64>> for f64 {
    type Output = Vector3d<f64>;
    #[inline]
    fn mul(self, other: Vector3d<f64>) -> Vector3d<f64> {
        f64::v3_mul_scalar(other, self)
    }
}

// **** Mul Scalar ****

/// Multiply vector by a constant.
/// ```
/// # use vqm::Vector3df32;
/// let v = Vector3df32::new(2.0, 3.0, 5.0);
/// let r = v * 2.0;
///
/// assert_eq!(r, Vector3df32 { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl<T> Mul<T> for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    type Output = Self;

    #[inline]
    fn mul(self, k: T) -> Self {
        T::v3_mul_scalar(self, k)
    }
}

impl Mul<i32> for Vector3d<i16> {
    type Output = Self;

    #[inline]
    fn mul(self, k: i32) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Vector3d { x: self.x * (k as i16), y: self.y * (k as i16), z: self.z * (k as i16) }
    }
}

impl Mul<i16> for Vector3d<f32> {
    type Output = Self;

    #[inline]
    fn mul(self, k: i16) -> Self {
        Vector3d { x: self.x * f32::from(k), y: self.y * f32::from(k), z: self.z * f32::from(k) }
    }
}

impl Mul<i32> for Vector3d<f32> {
    type Output = Self;

    #[inline]
    fn mul(self, k: i32) -> Self {
        #[allow(clippy::cast_precision_loss)]
        Vector3d { x: self.x * (k as f32), y: self.y * (k as f32), z: self.z * (k as f32) }
    }
}

// **** MulAssign ****

/// In-place multiply a vector by a constant.
/// ```
/// # use vqm::Vector3df32;
/// let mut v = Vector3df32::new(2.0, 3.0, 5.0);
/// v *= 2.0;
///
/// assert_eq!(v, Vector3df32 { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl<T> MulAssign<T> for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    #[inline]
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

// **** Mul Elementwise ****

/// Elementwise multiply a vector by another vector.
/// ```
/// # use vqm::Vector3df32;
/// let v = Vector3df32::new(2.0, 5.0, 11.0);
/// let u = Vector3df32::new(3.0, 7.0, 13.0);
/// let r = v * u;
///
/// assert_eq!(r, Vector3df32 { x: 6.0, y: 35.0, z: 143.0 });
/// ```
impl<T> Mul<Vector3d<T>> for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        T::v3_mul_elementwise(self, other)
    }
}

// **** Div by scalar ****

/// Divide a vector by a constant.
/// ```
/// # use vqm::Vector3df32;
/// let v = Vector3df32::new(2.0, 3.0, 5.0);
/// let r = v / 2.0;
///
/// assert_eq!(r, Vector3df32 { x: 1.0, y: 1.5, z: 2.5 });
/// ```
impl<T> Div<T> for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    type Output = Self;

    #[inline]
    fn div(self, k: T) -> Self {
        T::v3_div_scalar(self, k)
    }
}

/// In-place divide a vector by a constant.
/// ```
/// # use vqm::Vector3df32;
/// let mut v = Vector3df32::new(2.0, 3.0, 5.0);
/// v /= 2.0;
///
/// assert_eq!(v, Vector3df32 { x: 1.0, y: 1.5, z: 2.5 });
/// ```
impl<T> DivAssign<T> for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    #[inline]
    fn div_assign(&mut self, k: T) {
        *self = self.div(k);
    }
}

// **** Div Elementwise ****

/// Elementwise divide a vector by another vector.
/// ```
/// # use vqm::Vector3df32;
/// let v = Vector3df32::new(3.0, 7.0, 13.0);
/// let u = Vector3df32::new(2.0, 5.0, 11.0);
/// let r = v / u;
///
/// assert_eq!(r, Vector3df32 { x: 1.5, y: 1.4, z: 13.0 / 11.0 });
/// ```
impl<T> Div<Vector3d<T>> for Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        T::v3_div_elementwise(self, other)
    }
}

// **** Index ****

/// Access vector component by index.
/// ```
/// # use vqm::Vector3df32;
/// let v = Vector3df32::new(2.0, 3.0, 5.0);
///
/// assert_eq!(v[0], 2.0);
/// assert_eq!(v[1], 3.0);
/// assert_eq!(v[2], 5.0);
/// ```
impl<T> Index<usize> for Vector3d<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z, // default to z component if index out of range
        }
    }
}

// **** IndexMut ****

// Set vector component by index.
/// ```
/// # use vqm::Vector3df32;
/// let mut v = Vector3df32::new(2.0, 5.0, 11.0);
/// v[0] = 3.0;
/// v[1] = 7.0;
/// v[2] = 13.0;
///
/// assert_eq!(v, Vector3df32 { x:3.0, y:7.0, z:13.0 });
/// ```
impl<T> IndexMut<usize> for Vector3d<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.z, // default to z component if index out of range
        }
    }
}

// **** abs ****

impl<T> Vector3d<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the vector with all components set to their absolute values.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, -3.0, -5.0);
    /// let u = v.abs();
    ///
    /// assert_eq!(u, Vector3df32::new(2.0, 3.0, 5.0));
    /// ```
    #[inline]
    pub fn abs(self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() }
    }

    /// Set all components of the vector to their absolute values.
    /// ```
    /// # use vqm::Vector3df32;
    /// let mut v = Vector3df32::new(2.0, -3.0, -5.0);
    /// v.abs_in_place();
    ///
    /// assert_eq!(v, Vector3df32::new(2.0, 3.0, 5.0));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = self.abs();
        self
    }
}

// **** clamp ****

impl<T> Vector3d<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the vector with all components clamped to the specified range.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, 3.0, 11.0);
    /// let u = v.clamp(2.5, 7.5);
    ///
    /// assert_eq!(u, Vector3df32::new(2.5, 3.0, 7.5));
    /// ```
    #[inline]
    pub fn clamp(self, min: T, max: T) -> Self {
        Self { x: self.x.clamp(min, max), y: self.y.clamp(min, max), z: self.z.clamp(min, max) }
    }

    /// Clamp all components of the vector to the specified range.
    /// ```
    /// # use vqm::Vector3df32;
    /// let mut v = Vector3df32::new(2.0, 3.0, 11.0);
    /// v.clamp_in_place(2.5, 7.5);
    ///
    /// assert_eq!(v, Vector3df32::new(2.5, 3.0, 7.5));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

// **** dot ****

impl<T> Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    /// Vector dot product.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, 5.0, 11.0);
    /// let w = Vector3df32::new(3.0, 7.0, 13.0);
    ///
    /// let x = v.dot(w);
    ///
    /// assert_eq!(x, 184.0);
    /// ```
    #[inline]
    pub fn dot(self, other: Self) -> T {
        T::v3_dot(self, other)
    }
}

// **** cross ****

impl<T> Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    /// Vector cross product.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, 5.0, 11.0);
    /// let w = Vector3df32::new(3.0, 7.0, 13.0);
    ///
    /// let x = v.cross(w);
    ///
    /// assert_eq!(x, Vector3df32::new(-12.0, 7.0, -1.0));
    /// ```
    #[inline]
    pub fn cross(self, other: Self) -> Vector3d<T> {
        T::v3_cross(self, other)
    }
}

// **** norm_squared ****

impl<T> Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    /// Return square of Euclidean norm.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, 3.0, 5.0);
    /// assert_eq!(38.0, v.norm_squared());
    /// ```
    #[inline]
    pub fn norm_squared(self) -> T {
        T::v3_norm_squared(self)
    }

    /// Return distance between two points, squared.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, 5.0, 11.0);
    /// let w = Vector3df32::new(3.0, 7.0, 17.0);
    /// assert_eq!(41.0, v.distance_squared(w));
    /// ```
    #[inline]
    pub fn distance_squared(self, other: Self) -> T {
        (self - other).norm_squared()
    }
}

// **** norm ****

impl<T> Vector3d<T>
where
    T: Copy + SqrtMethods + Vector3dMath,
{
    /// Return Euclidean norm.
    #[inline]
    pub fn norm(self) -> T {
        Self::norm_squared(self).sqrt()
    }
}

impl<T> Vector3d<T>
where
    T: Copy + Zero + PartialEq + SqrtMethods + Vector3dMath,
{
    /// Return normalized form of the vector, checking if the norm is zero.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(0.0, 0.0, 0.0);
    /// let n = v.normalize();
    /// assert_eq!(Vector3df32 { x: 0.0, y: 0.0, z: 0.0 }, n);
    /// ```
    #[inline]
    pub fn normalize(self) -> Self {
        let norm_squared = self.norm_squared();
        // If norm == 0.0 then the vector is already normalized
        if norm_squared == T::zero() {
            return self;
        }
        self * norm_squared.sqrt_reciprocal()
    }

    /// Normalize the vector in place, checking if the norm is zero.
    /// ```
    /// # use vqm::Vector3df32;
    /// let mut v = Vector3df32::new(0.0, 0.0, 0.0);
    /// v.normalize_in_place();
    /// assert_eq!(Vector3df32 { x: 0.0, y: 0.0, z: 0.0 }, v);
    /// ```
    #[inline]
    pub fn normalize_in_place(&mut self) -> &mut Self {
        *self = self.normalize();
        self
    }

    /// Return normalized form of the vector, not checking if the norm is zero.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(1.0, 4.0, 8.0);
    /// let n = v.normalize_unchecked();
    /// assert_eq!(Vector3df32 { x: 0.11111111, y: 0.44444445, z: 0.8888889 }, n);
    /// ```
    #[inline]
    pub fn normalize_unchecked(self) -> Self {
        let norm_squared = self.norm_squared();
        self * norm_squared.sqrt_reciprocal()
    }

    /// Normalize the vector in place, not checking if the norm is zero.
    /// ```
    /// # use vqm::Vector3df32;
    /// let mut v = Vector3df32::new(1.0, 4.0, 8.0);
    /// v.normalize_unchecked_in_place();
    /// assert_eq!(Vector3df32 { x: 0.11111111, y: 0.44444445, z: 0.8888889 }, v);
    /// ```
    #[inline]
    pub fn normalize_unchecked_in_place(&mut self) -> &mut Self {
        *self = self.normalize_unchecked();
        self
    }
}

impl<T> Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    // Return true if the vector is normalized.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, 3.0, 5.0);
    /// let n = v.normalize();
    /// assert!(n.is_normalized());
    /// ```
    #[inline]
    pub fn is_normalized(self) -> bool {
        T::v3_is_normalized(self)
    }
}

impl<T> Vector3d<T>
where
    T: Copy + Zero + SqrtMethods + Vector3dMath,
{
    // Return distance between two points
    #[inline]
    pub fn distance(self, other: Self) -> T {
        self.distance_squared(other).sqrt()
    }
}

// **** to_degrees ****

impl<T> Vector3d<T>
where
    T: Copy + Mul<Output = T> + MathConstants,
{
    /// Convert the vector to degrees, assuming it is in radians.
    /// ```
    /// # use vqm::{Vector3df32, MathConstants};
    /// let v = Vector3df32::new(f32::FRAC_PI_2, f32::FRAC_PI_4, f32::FRAC_PI_6);
    /// assert_eq!(Vector3df32::new(90.0, 45.0, 30.0), v.to_degrees());
    /// ```
    #[inline]
    pub fn to_degrees(self) -> Self {
        Self { x: self.x * T::RADIANS_TO_DEGREES, y: self.y * T::RADIANS_TO_DEGREES, z: self.z * T::RADIANS_TO_DEGREES }
    }

    /// Convert the vector to radians, assuming it is in degrees.
    /// ```
    /// # use vqm::{Vector3df32, MathConstants};
    /// let v = Vector3df32::new(90.0, 45.0, 30.0);
    /// assert_eq!(Vector3df32::new(f32::FRAC_PI_2, f32::FRAC_PI_4, f32::FRAC_PI_6), v.to_radians());
    /// ```
    #[inline]
    pub fn to_radians(self) -> Self {
        Self { x: self.x * T::DEGREES_TO_RADIANS, y: self.y * T::DEGREES_TO_RADIANS, z: self.z * T::DEGREES_TO_RADIANS }
    }
}

// **** sum ****

impl<T> Vector3d<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    /// Return the sum of all components of the vector.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, 3.0, 5.0);
    /// assert_eq!(10.0, v.sum());
    /// ```
    #[inline]
    pub fn sum(self) -> T {
        self.x + self.y + self.z
    }

    /// Return the product of all components of the vector.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, 3.0, 5.0);
    /// assert_eq!(30.0, v.product());
    /// ```
    #[inline]
    pub fn product(self) -> T {
        self.x * self.y * self.z
    }
}

// **** mean ****

impl<T> Vector3d<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the vector.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, 3.0, 7.0);
    /// assert_eq!(4.0, v.mean());
    /// ```
    #[inline]
    pub fn mean(self) -> T {
        let three = T::one() + T::one() + T::one();
        self.sum() / three
    }
}

// **** max ****

impl<T> Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    /// Return the max element in the vector.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, 3.0, 5.0);
    /// let w = Vector3df32::new(3.0, 5.0, 2.0);
    /// let x = Vector3df32::new(5.0, 3.0, 2.0);
    /// assert_eq!(5.0, v.max());
    /// assert_eq!(5.0, w.max());
    /// assert_eq!(5.0, x.max());
    /// ```
    #[inline]
    pub fn max(self) -> T {
        T::v3_max(self)
    }

    /// Return the min element in the vector.
    /// ```
    /// # use vqm::Vector3df32;
    /// let v = Vector3df32::new(2.0, 3.0, 5.0);
    /// let w = Vector3df32::new(3.0, 5.0, 2.0);
    /// let x = Vector3df32::new(5.0, 3.0, 2.0);
    /// assert_eq!(2.0, v.min());
    /// assert_eq!(2.0, w.min());
    /// assert_eq!(2.0, x.min());
    /// ```
    #[inline]
    pub fn min(self) -> T {
        T::v3_min(self)
    }
}

impl<T> Vector3d<T>
where
    T: Copy + Zero + One + SqrtMethods + Vector3dMath + QuaternionMath,
{
    #[inline]
    pub fn rotate_by(self, q: Quaternion<T>) -> Self {
        // Extract the vector part of the quaternion (x, y, z)
        let q_xyz = Vector3d { x: q.x, y: q.y, z: q.z };

        // 1. uv = 2 * (q_xyz cross v)
        let uv = q_xyz.cross(self) * (T::one() + T::one());

        // 2. res = v + w * uv + (q_xyz cross t)
        // This is the optimized Rodrigues form
        self + (uv * q.w) + q_xyz.cross(uv)
    }
    #[inline]
    pub fn rotate_back_by(self, q: Quaternion<T>) -> Self {
        // Rotating 'back' is just rotating by the inverse (conjugate)
        self.rotate_by(q.conjugate())
    }
}

// **** From ****

// **** From Tuple ****

/// Vector from tuple.
/// ```
/// # use vqm::Vector3df32;
/// let v = Vector3df32::from((2.0, 3.0, 5.0));
/// let w: Vector3df32 = (7.0, 11.0, 13.0).into();
///
/// assert_eq!(v, Vector3df32 { x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w, Vector3df32 { x: 7.0, y: 11.0, z: 13.0 });
/// ```
impl<T> From<(T, T, T)> for Vector3d<T> {
    #[inline]
    fn from((x, y, z): (T, T, T)) -> Self {
        Self { x, y, z }
    }
}

// **** From Array ****

/// Vector from array.
/// ```
/// # use vqm::Vector3df32;
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
    #[inline]
    fn from(v: [T; 3]) -> Self {
        Self { x: v[0], y: v[1], z: v[2] }
    }
}

/// Array from vector.
/// ```
/// # use vqm::Vector3df32;
/// let v = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
///
/// let a = <[f32; 3]>::from(v);
/// let b: [f32; 3] = v.into();
///
/// assert_eq!(a, [2.0, 3.0, 5.0]);
/// assert_eq!(b, [2.0, 3.0, 5.0]);
/// ```
impl<T> From<Vector3d<T>> for [T; 3] {
    #[inline]
    fn from(v: Vector3d<T>) -> Self {
        [v.x, v.y, v.z]
    }
}

// **** From Vector ****

/// Vector3d from Vector2d.
/// ```
/// # use vqm::{Vector2df32,Vector3df32};
/// let v = Vector3df32::from(Vector2df32 { x: 2.0, y: 3.0 });
/// let w: Vector3df32 = Vector2df32 { x: 7.0, y: 11.0 }.into();
///
/// assert_eq!(v, Vector3df32 { x: 2.0, y: 3.0, z: 0.0 });
/// assert_eq!(w, Vector3df32 { x: 7.0, y: 11.0, z: 0.0 });
/// ```
impl<T> From<Vector2d<T>> for Vector3d<T>
where
    T: Copy + Zero,
{
    #[inline]
    fn from(other: Vector2d<T>) -> Self {
        Self { x: other.x, y: other.y, z: T::zero() }
    }
}

// **** From Vector ****

/// Vector2d from Vector3d, discarding z value.
/// ```
/// # use vqm::{Vector2df32,Vector3df32};
/// let v: Vector2df32 = Vector3df32 { x: 2.0, y: 5.0, z: 11.0 }.into();
/// let u = Vector2df32::from(Vector3df32{ x: 3.0, y: 7.0, z: 13.0 });
///
/// assert_eq!(v, Vector2df32 { x: 2.0, y: 5.0 });
/// assert_eq!(u, Vector2df32 { x: 3.0, y: 7.0 });
impl<T> From<Vector3d<T>> for Vector2d<T>
where
    T: Copy + Zero,
{
    #[inline]
    fn from(v: Vector3d<T>) -> Self {
        Vector2d::<T> { x: v.x, y: v.y }
    }
}
/// 3-dimensional `{x, y, z}` vector of `i16` values<br>
pub type Vector3di16 = Vector3d<i16>;

// **** Vector3di16 ****

impl Mul<f32> for Vector3d<i16> {
    type Output = Self;

    #[inline]
    fn mul(self, k: f32) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self { x: (f32::from(self.x) * k) as i16, y: (f32::from(self.y) * k) as i16, z: (f32::from(self.z) * k) as i16 }
    }
}

/// `Vector3d<f32>` from `Vector3d<i16>`.
/// ```
/// # use vqm::{Vector3df32,Vector3di16};
/// let v_i16 = Vector3di16{x: 2, y: 3, z: 5};
/// let v_f32 = Vector3df32::from(v_i16);
///
/// let w_f32 = Vector3df32{x: 7.0, y: 11.0, z: 13.0};
/// let w_i16 : Vector3di16 = w_f32.into();
///
/// assert_eq!(v_f32, Vector3df32 { x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w_i16, Vector3di16 { x: 7, y: 11, z: 13 });
/// ```
impl From<Vector3d<i16>> for Vector3d<f32> {
    #[inline]
    fn from(v: Vector3d<i16>) -> Self {
        Self { x: f32::from(v.x), y: f32::from(v.y), z: f32::from(v.z) }
    }
}

impl From<Vector3d<f32>> for Vector3d<i16> {
    #[inline]
    fn from(v: Vector3d<f32>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self { x: v.x as i16, y: v.y as i16, z: v.z as i16 }
    }
}

impl From<[i16; 3]> for Vector3d<f32> {
    #[inline]
    fn from(v: [i16; 3]) -> Self {
        Self { x: f32::from(v[0]), y: f32::from(v[1]), z: f32::from(v[2]) }
    }
}

impl From<Vector3d<f32>> for [i16; 3] {
    #[inline]
    fn from(v: Vector3d<f32>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        [v.x as i16, v.y as i16, v.z as i16]
    }
}

// **** Vector3di32 ****

/// 3-dimensional `{x, y, z}` vector of `i32` values<br><br>
pub type Vector3di32 = Vector3d<i32>;

impl Mul<f32> for Vector3d<i32> {
    type Output = Self;

    #[inline]
    fn mul(self, k: f32) -> Self {
        #[allow(clippy::cast_precision_loss)]
        #[allow(clippy::cast_possible_truncation)]
        Self { x: ((self.x as f32) * k) as i32, y: ((self.y as f32) * k) as i32, z: ((self.z as f32) * k) as i32 }
    }
}

/// `Vector3d<f32>` from `Vector3d<i32>`.
/// ```
/// # use vqm::{Vector3df32,Vector3di32};
/// let v_i32 = Vector3di32{x: 2, y: 3, z: 5};
/// let v_f32 = Vector3df32::from(v_i32);
///
/// let w_f32 = Vector3df32{x: 7.0, y: 11.0, z: 13.0};
/// let w_i32 : Vector3di32 = w_f32.into();
///
/// assert_eq!(v_f32, Vector3df32 { x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w_i32, Vector3di32 { x: 7, y: 11, z: 13 });
/// ```
impl From<Vector3d<i32>> for Vector3d<f32> {
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn from(v: Vector3d<i32>) -> Self {
        Self { x: v.x as f32, y: v.y as f32, z: v.z as f32 }
    }
}

impl From<Vector3d<f32>> for Vector3d<i32> {
    #[inline]
    fn from(v: Vector3d<f32>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self { x: v.x as i32, y: v.y as i32, z: v.z as i32 }
    }
}

impl From<[i32; 3]> for Vector3d<f32> {
    #[inline]
    fn from(v: [i32; 3]) -> Self {
        #[allow(clippy::cast_precision_loss)]
        Self { x: v[0] as f32, y: v[1] as f32, z: v[2] as f32 }
    }
}

impl From<Vector3d<f32>> for [i32; 3] {
    #[inline]
    fn from(v: Vector3d<f32>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        [v.x as i32, v.y as i32, v.z as i32]
    }
}
