use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};

use crate::{MathConstants, SqrtMethods, Vector2d, Vector3d, Vector4dMath};

/// 4-dimensional `{x, y, z, t}` vector of `f32` values<br>
pub type Vector4df32 = Vector4d<f32>;
/// 4-dimensional `{x, y, z, t}` vector of `f64` values<br><br>
pub type Vector4df64 = Vector4d<f64>;

// **** Define ****

/// `Vector4d<T>`: 3D vector of type `T`.<br>
/// Aliases `Vector4df32` and `Vector4df64` are provided.<br><br>
/// `Vector4df32` uses **SIMD** accelerations implemented in `Vector4dMath`.<br><br>
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector4d<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub t: T,
}

// **** New ****

/// Create a vector.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32::new(2.0,  3.0, 7.0, 11.0);
/// assert_eq!(v, Vector4df32 { x:2.0, y:3.0, z: 7.0, t: 11.0 });
/// ```
impl<T> Vector4d<T>
where
    T: Copy,
{
    /// Create a vector.
    #[inline]
    pub const fn new(x: T, y: T, z: T, t: T) -> Self {
        Self { x, y, z, t }
    }
}

// **** Zero ****

/// Zero vector.
/// ```
/// # use vqm::Vector4df32;
/// # use num_traits::{zero,Zero};
/// let z: Vector4df32 = zero();
/// assert!(z.is_zero());
/// assert_eq!(z, Vector4df32 { x: 0.0, y: 0.0, z: 0.0, t: 0.0 });
/// ```
impl<T> Zero for Vector4d<T>
where
    T: Copy + Zero + PartialEq + Vector4dMath,
{
    #[inline]
    fn zero() -> Self {
        Self { x: T::zero(), y: T::zero(), z: T::zero(), t: T::zero() }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.x == T::zero() && self.y == T::zero() && self.z == T::zero() && self.z == T::zero()
    }
}

// **** Neg ****

/// Negate vector.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32 { x: 2.0, y: 3.0, z: 5.0, t: 7.0 };
/// let r = -v;
///
/// assert_eq!(r, Vector4df32 { x: -2.0, y: -3.0, z: -5.0, t: -7.0 });
/// ```
impl<T> Neg for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        T::v4_neg(self)
    }
}

// **** Add ****

/// Add two vectors.
/// ```
/// # use vqm::Vector4df32;
/// let u = Vector4df32::new(2.0, 5.0, 11.0, 17.0);
/// let v = Vector4df32::new(3.0, 7.0, 13.0, 19.0);
/// let r = u + v;
///
/// assert_eq!(r, Vector4df32 { x: 5.0, y: 12.0, z: 24.0, t: 36.0 });
/// ```
impl<T> Add for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        T::v4_add(self, other)
    }
}

// **** AddAssign ****

/// Add one vector to another.
/// ```
/// # use vqm::Vector4df32;
/// let mut r = Vector4df32::new(2.0, 5.0, 11.0, 17.0);
/// let u = Vector4df32::new(3.0, 7.0, 13.0, 19.0);
/// r += u;
///
/// assert_eq!(r, Vector4df32 { x: 5.0, y: 12.0, z: 24.0, t: 36.0 });
///
/// # use num_traits::zero;
/// let z: Vector4df32 = zero();
/// let r = u + z;
/// assert_eq!(r, u);
/// ```
impl<T> AddAssign for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

/// Multiply vector by constant and add another vector.
/// ```
/// # use vqm::Vector4df32;
/// # use num_traits::MulAdd;
/// let v = Vector4df32::new(2.0, 5.0, 11.0, 17.0);
/// let w = Vector4df32::new(3.0, 7.0, 13.0, 19.0);
/// let k = 23.0;
/// let r = v.mul_add(k, w);
///
/// assert_eq!(r, Vector4df32 { x: 49.0, y: 122.0, z: 266.0, t: 410.0 });
/// ```
impl<T> MulAdd<T> for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    type Output = Self;

    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::v4_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

/// Multiply vector by constant and add another vector in place.
/// ```
/// # use vqm::Vector4df32;
/// # use num_traits::MulAddAssign;
/// let mut v = Vector4df32::new(2.0, 5.0, 11.0, 17.0);
/// let w = Vector4df32::new(3.0, 7.0, 13.0, 19.0);
/// let k = 23.0;
/// v.mul_add_assign(k, w);
///
/// assert_eq!(v, Vector4df32 { x: 49.0, y: 122.0, z: 266.0, t: 410.0 });
/// ```
impl<T> MulAddAssign<T> for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

/// Subtract two vectors.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32::new(2.0, 5.0, 13.0, 17.0);
/// let w = Vector4df32::new(3.0, 7.0, 11.0, 23.0);
/// let r = v - w;
///
/// assert_eq!(r, Vector4df32 { x: -1.0, y: -2.0, z: 2.0, t: -6.0 });
/// ```
impl<T> Sub for Vector4d<T>
where
    T: Copy + Vector4dMath,
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
/// # use vqm::Vector4df32;
/// let mut r = Vector4df32::new(2.0, 5.0, 13.0, 17.0);
/// let     v = Vector4df32::new(3.0, 7.0, 11.0, 23.0);
/// r -= v;
///
/// assert_eq!(r, Vector4df32 { x: -1.0, y: -2.0, z: 2.0, t: -6.0 });
/// ```
impl<T> SubAssign for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Scalar Mul ****

/// Pre-multiply vector by a constant.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// let r = 2.0 * v;
///
/// assert_eq!(r, Vector4df32 { x: 4.0, y: 6.0, z: 10.0, t: 14.0 });
/// ```
impl Mul<Vector4d<f32>> for f32 {
    type Output = Vector4d<f32>;
    #[inline]
    fn mul(self, other: Vector4d<f32>) -> Vector4d<f32> {
        f32::v4_mul_scalar(other, self)
    }
}

impl Mul<Vector4d<f64>> for f64 {
    type Output = Vector4d<f64>;
    #[inline]
    fn mul(self, other: Vector4d<f64>) -> Vector4d<f64> {
        f64::v4_mul_scalar(other, self)
    }
}

// **** Mul Scalar ****

/// Multiply vector by a constant.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// let r = v * 2.0;
///
/// assert_eq!(r, Vector4df32 { x: 4.0, y: 6.0, z: 10.0, t: 14.0 });
/// ```
impl<T> Mul<T> for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    type Output = Self;

    #[inline]
    fn mul(self, k: T) -> Self {
        T::v4_mul_scalar(self, k)
    }
}

// **** MulAssign ****

/// In-place multiply a vector by a constant.
/// ```
/// # use vqm::Vector4df32;
/// let mut v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// v *= 2.0;
///
/// assert_eq!(v, Vector4df32 { x: 4.0, y: 6.0, z: 10.0, t: 14.0 });
/// ```
impl<T> MulAssign<T> for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    #[inline]
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

// **** Mul Elementwise ****

/// Elementwise multiply a vector by another vector.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32::new(2.0, 5.0, 11.0, 17.0);
/// let u = Vector4df32::new(3.0, 7.0, 13.0, 19.0);
/// let r = v * u;
///
/// assert_eq!(r, Vector4df32 { x: 6.0, y: 35.0, z: 143.0, t: 323.0 });
/// ```
impl<T> Mul<Vector4d<T>> for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        T::v4_mul_elementwise(self, other)
    }
}

// **** Div by scalar ****

/// Divide a vector by a constant.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// let r = v / 2.0;
///
/// assert_eq!(r, Vector4df32 { x: 1.0, y: 1.5, z: 2.5, t: 3.5 });
/// ```
impl<T> Div<T> for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    type Output = Self;

    #[inline]
    fn div(self, k: T) -> Self {
        T::v4_div_scalar(self, k)
    }
}

/// In-place divide a vector by a constant.
/// ```
/// # use vqm::Vector4df32;
/// let mut v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// v /= 2.0;
///
/// assert_eq!(v, Vector4df32 { x: 1.0, y: 1.5, z: 2.5, t: 3.5 });
/// ```
impl<T> DivAssign<T> for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    #[inline]
    fn div_assign(&mut self, k: T) {
        *self = self.div(k);
    }
}

// **** Div Elementwise ****

/// Elementwise divide a vector by another vector.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32::new(3.0, 7.0, 13.0, 19.0);
/// let u = Vector4df32::new(2.0, 5.0, 11.0, 17.0);
/// let r = v / u;
///
/// assert_eq!(r, Vector4df32 { x: 1.5, y: 1.4, z: 13.0 / 11.0, t: 19.0 / 17.0 });
/// ```
impl<T> Div<Vector4d<T>> for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        T::v4_div_elementwise(self, other)
    }
}

// **** Index ****

/// Access vector component by index.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
///
/// assert_eq!(v[0], 2.0);
/// assert_eq!(v[1], 3.0);
/// assert_eq!(v[2], 5.0);
/// assert_eq!(v[3], 7.0);
/// ```
impl<T> Index<usize> for Vector4d<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => &self.t, // default to t component if index out of range
        }
    }
}

// **** IndexMut ****

// Set vector component by index.
/// ```
/// # use vqm::Vector4df32;
/// let mut v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// v[0] = 7.0;
/// v[1] = 11.0;
/// v[2] = 13.0;
/// v[3] = 17.0;
///
/// assert_eq!(v, Vector4df32 { x:7.0, y:11.0, z:13.0, t: 17.0 });
/// ```
impl<T> IndexMut<usize> for Vector4d<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => &mut self.t, // default to t component if index out of range
        }
    }
}

// **** abs ****

impl<T> Vector4d<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the vector with all components set to their absolute values.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, -3.0, -5.0, 7.0);
    /// let u = v.abs();
    ///
    /// assert_eq!(u, Vector4df32::new(2.0, 3.0, 5.0, 7.0));
    /// ```
    #[inline]
    pub fn abs(self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs(), t: self.t.abs() }
    }

    /// Set all components of the vector to their absolute values.
    /// ```
    /// # use vqm::Vector4df32;
    /// let mut v = Vector4df32::new(2.0, -3.0, -5.0, 7.0);
    /// v.abs_in_place();
    ///
    /// assert_eq!(v, Vector4df32::new(2.0, 3.0, 5.0, 7.0));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = self.abs();
        self
    }
}

// **** clamp ****

impl<T> Vector4d<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the vector with all components clamped to the specified range.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 7.0, 11.0);
    /// let u = v.clamp(2.5, 7.5);
    ///
    /// assert_eq!(u, Vector4df32::new(2.5, 3.0, 7.0, 7.5));
    /// ```
    #[inline]
    pub fn clamp(self, min: T, max: T) -> Self {
        Self {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
            t: self.t.clamp(min, max),
        }
    }

    /// Clamp all components of the vector to the specified range.
    /// ```
    /// # use vqm::Vector4df32;
    /// let mut v = Vector4df32::new(2.0, 3.0, 7.0, 11.0);
    /// v.clamp_in_place(2.5, 7.5);
    ///
    /// assert_eq!(v, Vector4df32::new(2.5, 3.0, 7.0, 7.5));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

// **** dot ****

impl<T> Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    /// Vector dot product.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Vector4df32::new(11.0, 13.0, 17.0, 19.0);
    ///
    /// let x = v.dot(w);
    ///
    /// assert_eq!(x, 279.0);
    /// ```
    #[inline]
    pub fn dot(self, other: Self) -> T {
        T::v4_dot(self, other)
    }
}

// **** norm_squared ****

impl<T> Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    /// Return square of Euclidean norm.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// assert_eq!(87.0, v.norm_squared());
    /// ```
    #[inline]
    pub fn norm_squared(self) -> T {
        T::v4_norm_squared(self)
    }

    /// Return distance between two points, squared.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Vector4df32::new(11.0, 13.0, 17.0, 19.0);
    /// assert_eq!(469.0, v.distance_squared(w));
    /// ```
    #[inline]
    pub fn distance_squared(self, other: Self) -> T {
        (self - other).norm_squared()
    }
}

// **** norm ****

impl<T> Vector4d<T>
where
    T: Copy + SqrtMethods + Vector4dMath,
{
    /// Return Euclidean norm.
    #[inline]
    pub fn norm(self) -> T {
        Self::norm_squared(self).sqrt()
    }
}

impl<T> Vector4d<T>
where
    T: Copy + Zero + PartialEq + SqrtMethods + Vector4dMath,
{
    /// Return normalized form of the vector, checking if the norm is zero.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(0.0, 0.0, 0.0, 0.0);
    /// let n = v.normalize();
    /// assert_eq!(Vector4df32 { x: 0.0, y: 0.0, z: 0.0, t: 0.0 }, n);
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
    /// # use vqm::Vector4df32;
    /// let mut v = Vector4df32::new(0.0, 0.0, 0.0, 0.0);
    /// v.normalize_in_place();
    /// assert_eq!(Vector4df32 { x: 0.0, y: 0.0, z: 0.0, t: 0.0 }, v);
    /// ```
    #[inline]
    pub fn normalize_in_place(&mut self) -> &mut Self {
        *self = self.normalize();
        self
    }

    /// Return normalized form of the vector, not checking if the norm is zero.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// let n = v.normalize_unchecked();
    /// assert_eq!(Vector4df32 { x: 0.21442251, y: 0.32163376, z: 0.5360563, t: 0.7504788 }, n);
    /// ```
    #[inline]
    pub fn normalize_unchecked(self) -> Self {
        let norm_squared = self.norm_squared();
        self * norm_squared.sqrt_reciprocal()
    }

    /// Normalize the vector in place, not checking if the norm is zero.
    /// ```
    /// # use vqm::Vector4df32;
    /// let mut v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// v.normalize_unchecked_in_place();
    /// assert_eq!(Vector4df32 { x: 0.21442251, y: 0.32163376, z: 0.5360563, t: 0.7504788 }, v);
    /// ```
    #[inline]
    pub fn normalize_unchecked_in_place(&mut self) -> &mut Self {
        *self = self.normalize_unchecked();
        self
    }
}

impl<T> Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    // Return true if the vector is normalized.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// let n = v.normalize();
    /// let s = n.norm_squared();
    /// assert_eq!(1.0, s);
    /// assert!(n.is_normalized());
    /// ```
    #[inline]
    pub fn is_normalized(self) -> bool {
        T::v4_is_normalized(self)
    }
}

impl<T> Vector4d<T>
where
    T: Copy + Zero + SqrtMethods + Vector4dMath,
{
    // Return distance between two points
    #[inline]
    pub fn distance(self, other: Self) -> T {
        self.distance_squared(other).sqrt()
    }
}

// **** to_degrees ****

impl<T> Vector4d<T>
where
    T: Copy + Mul<Output = T> + MathConstants,
{
    /// Convert the vector to degrees, assuming it is in radians.
    /// ```
    /// # use vqm::{Vector4df32, MathConstants};
    /// let v = Vector4df32::new(f32::FRAC_PI_2, f32::FRAC_PI_4, f32::FRAC_PI_6, f32::FRAC_PI_8);
    /// assert_eq!(Vector4df32::new(90.0, 45.0, 30.0, 22.5), v.to_degrees());
    /// ```
    #[inline]
    pub fn to_degrees(self) -> Self {
        Self {
            x: self.x * T::RADIANS_TO_DEGREES,
            y: self.y * T::RADIANS_TO_DEGREES,
            z: self.z * T::RADIANS_TO_DEGREES,
            t: self.t * T::RADIANS_TO_DEGREES,
        }
    }

    /// Convert the vector to radians, assuming it is in degrees.
    /// ```
    /// # use vqm::{Vector4df32, MathConstants};
    /// let v = Vector4df32::new(90.0, 45.0, 30.0, 22.5);
    /// assert_eq!(Vector4df32::new(f32::FRAC_PI_2, f32::FRAC_PI_4, f32::FRAC_PI_6, f32::FRAC_PI_8), v.to_radians());
    /// ```
    #[inline]
    pub fn to_radians(self) -> Self {
        Self {
            x: self.x * T::DEGREES_TO_RADIANS,
            y: self.y * T::DEGREES_TO_RADIANS,
            z: self.z * T::DEGREES_TO_RADIANS,
            t: self.t * T::DEGREES_TO_RADIANS,
        }
    }
}

// **** sum ****

impl<T> Vector4d<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    /// Return the sum of all components of the vector.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// assert_eq!(17.0, v.sum());
    /// ```
    #[inline]
    pub fn sum(self) -> T {
        self.x + self.y + self.z + self.t
    }

    /// Return the product of all components of the vector.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// assert_eq!(210.0, v.product());
    /// ```
    #[inline]
    pub fn product(self) -> T {
        self.x * self.y * self.z * self.t
    }
}

// **** mean ****

impl<T> Vector4d<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the vector.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// assert_eq!(4.25, v.mean());
    /// ```
    #[inline]
    pub fn mean(self) -> T {
        let four = T::one() + T::one() + T::one() + T::one();
        self.sum() / four
    }
}

// **** max ****

impl<T> Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    /// Return the max element in the vector.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Vector4df32::new(3.0, 5.0, 7.0, 2.0);
    /// let x = Vector4df32::new(5.0, 7.0, 3.0, 2.0);
    /// assert_eq!(7.0, v.max());
    /// assert_eq!(7.0, w.max());
    /// assert_eq!(7.0, x.max());
    /// ```
    #[inline]
    pub fn max(self) -> T {
        T::v4_max(self)
    }

    /// Return the max element in the vector.
    /// ```
    /// # use vqm::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Vector4df32::new(3.0, 5.0, 7.0, 2.0);
    /// let x = Vector4df32::new(5.0, 7.0, 3.0, 2.0);
    /// assert_eq!(2.0, v.min());
    /// assert_eq!(2.0, w.min());
    /// assert_eq!(2.0, x.min());
    /// ```
    #[inline]
    pub fn min(self) -> T {
        T::v4_min(self)
    }
}

// **** From ****

// **** From Tuple ****

/// Vector from tuple.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32::from((2.0, 3.0, 5.0, 7.0));
/// let w: Vector4df32 = (11.0, 13.0, 17.0, 19.0).into();
///
/// assert_eq!(v, Vector4df32 { x: 2.0, y: 3.0, z: 5.0, t: 7.0 });
/// assert_eq!(w, Vector4df32 { x: 11.0, y: 13.0, z: 17.0, t: 19.0 });
/// ```
impl<T> From<(T, T, T, T)> for Vector4d<T>
where
    T: Copy,
{
    #[inline]
    fn from((x, y, z, t): (T, T, T, T)) -> Self {
        Self { x, y, z, t }
    }
}

// **** From Array ****

/// Vector from array.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32::from([2.0, 3.0, 5.0, 7.0]);
/// let w: Vector4df32 = [11.0, 13.0, 17.0, 19.0].into();
///
/// assert_eq!(v, Vector4df32 { x: 2.0, y: 3.0, z: 5.0, t: 7.0 });
/// assert_eq!(w, Vector4df32 { x: 11.0, y: 13.0, z: 17.0, t: 19.0 });
/// ```
impl<T> From<[T; 4]> for Vector4d<T>
where
    T: Copy,
{
    #[inline]
    fn from(v: [T; 4]) -> Self {
        Self { x: v[0], y: v[1], z: v[2], t: v[3] }
    }
}

/// Array from vector.
/// ```
/// # use vqm::Vector4df32;
/// let v = Vector4df32 { x: 2.0, y: 3.0, z: 5.0, t: 7.0 };
///
/// let a = <[f32; 4]>::from(v);
/// let b: [f32; 4] = v.into();
///
/// assert_eq!(a, [2.0, 3.0, 5.0, 7.0]);
/// assert_eq!(b, [2.0, 3.0, 5.0, 7.0]);
/// ```
impl<T> From<Vector4d<T>> for [T; 4] {
    #[inline]
    fn from(v: Vector4d<T>) -> Self {
        [v.x, v.y, v.z, v.t]
    }
}

// **** From Vector ****

/// Vector4d from Vector3d.
/// ```
/// # use vqm::{Vector3df32,Vector4df32};
/// let v = Vector4df32::from(Vector3df32 { x: 2.0, y: 3.0, z: 5.0 });
/// let w: Vector4df32 = Vector3df32 { x: 7.0, y: 11.0, z: 13.0 }.into();
///
/// assert_eq!(v, Vector4df32 { x: 2.0, y: 3.0, z: 5.0, t: 0.0 });
/// assert_eq!(w, Vector4df32 { x: 7.0, y: 11.0, z: 13.0, t: 0.0 });
/// ```
impl<T> From<Vector3d<T>> for Vector4d<T>
where
    T: Copy + Zero,
{
    #[inline]
    fn from(other: Vector3d<T>) -> Self {
        Self { x: other.x, y: other.y, z: other.z, t: T::zero() }
    }
}

// **** From Vector ****

/// Vector4d from Vector2d.
/// ```
/// # use vqm::{Vector2df32,Vector4df32};
/// let v = Vector4df32::from(Vector2df32 { x: 2.0, y: 3.0 });
/// let w: Vector4df32 = Vector2df32 { x: 11.0, y: 13.0 }.into();
///
/// assert_eq!(v, Vector4df32 { x: 2.0, y: 3.0, z: 0.0, t: 0.0 });
/// assert_eq!(w, Vector4df32 { x: 11.0, y: 13.0, z: 0.0, t: 0.0 });
/// ```
impl<T> From<Vector2d<T>> for Vector4d<T>
where
    T: Copy + Zero,
{
    #[inline]
    fn from(other: Vector2d<T>) -> Self {
        Self { x: other.x, y: other.y, z: T::zero(), t: T::zero() }
    }
}
