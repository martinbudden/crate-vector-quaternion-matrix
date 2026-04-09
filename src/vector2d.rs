use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};

use crate::{SqrtMethods, Vector2dMath};

/// 2-dimensional `{x, y}` vector of `f32` values<br>
pub type Vector2df32 = Vector2d<f32>;
/// 2-dimensional `{x, y}` vector of `f64` values<br><br>
pub type Vector2df64 = Vector2d<f64>;

// **** Define ****

/// `Vector2d<T>`: 2D vector of type `T`.<br>
/// Aliases `Vector2df32` and `Vector2df64` are provided.<br><br>
/// `Vector2df32` uses **SIMD** accelerations implemented in `Vector2dMath`.<br><br>
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector2d<T> {
    pub x: T,
    pub y: T,
}

// **** New ****

/// Create a vector.
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// let v = Vector2df32::new(2.0,  3.0);
/// assert_eq!(v, Vector2df32 { x: 2.0, y: 3.0 });
/// ```
impl<T> Vector2d<T>
where
    T: Copy,
{
    /// Create a vector
    #[inline(always)]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

// **** Zero ****

/// Zero vector
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// # use num_traits::{zero,Zero};
/// let z: Vector2df32 = zero();
/// assert!(z.is_zero());
/// assert_eq!(z, Vector2df32 { x: 0.0, y: 0.0 });
/// ```
impl<T> Zero for Vector2d<T>
where
    T: Copy + Zero + PartialEq + Vector2dMath,
{
    #[inline(always)]
    fn zero() -> Self {
        Self { x: T::zero(), y: T::zero() }
    }

    #[inline(always)]
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
    T: Copy + Vector2dMath,
{
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
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
    T: Copy + Vector2dMath,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        T::v2_add(self, other)
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
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

/// Multiply vector by constant and add another vector
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// # use num_traits::MulAdd;
/// let mut v = Vector2df32::new(2.0, 3.0);
/// let w = Vector2df32::new(5.0, 7.0);
/// let k = 11.0;
/// let r = v.mul_add(k, w);
///
/// assert_eq!(r, Vector2df32 { x: 27.0, y: 40.0 });
/// ```
impl<T> MulAdd<T> for Vector2d<T>
where
    T: Copy + Vector2dMath,
{
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::v2_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

/// Multiply vector by constant and add another vector in place
/// ```
/// # use vector_quaternion_matrix::Vector2df32;
/// # use num_traits::MulAddAssign;
/// let mut v = Vector2df32::new(2.0, 3.0);
/// let w = Vector2df32::new(5.0, 7.0);
/// let k = 11.0;
/// v.mul_add_assign(k, w);
///
/// assert_eq!(v, Vector2df32 { x: 27.0, y: 40.0 });
/// ```
impl<T> MulAddAssign<T> for Vector2d<T>
where
    T: Copy + Vector2dMath,
{
    #[inline(always)]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
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
    T: Copy + Vector2dMath,
{
    type Output = Self;

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-other)
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
    T: Copy + Vector2dMath,
{
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Scalar Mul ****

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
    #[inline(always)]
    fn mul(self, other: Vector2d<f32>) -> Vector2d<f32> {
        f32::v2_mul_scalar(other, self)
    }
}

impl Mul<Vector2d<f64>> for f64 {
    type Output = Vector2d<f64>;
    #[inline(always)]
    fn mul(self, other: Vector2d<f64>) -> Vector2d<f64> {
        f64::v2_mul_scalar(other, self)
    }
}

// **** Mul Scalar ****

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

    #[inline(always)]
    fn mul(self, k: T) -> Self {
        T::v2_mul_scalar(self, k)
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
    #[inline(always)]
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

// **** Div scalar ****

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

    #[inline(always)]
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
    T: Copy + Vector2dMath,
{
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.y, // default to z component if index out of range
        }
    }
}

// **** abs ****

impl<T> Vector2d<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the vector with all components set to their absolute values
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(2.0, -3.0);
    /// let u = v.abs();
    ///
    /// assert_eq!(u, Vector2df32::new(2.0, 3.0));
    /// ```
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs() }
    }

    /// Set all components of the vector to their absolute values
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let mut v = Vector2df32::new(2.0, -3.0);
    /// v.abs_mut();
    ///
    /// assert_eq!(v, Vector2df32::new(2.0, 3.0));
    /// ```
    #[inline(always)]
    pub fn abs_mut(&mut self) -> &mut Self {
        *self = self.abs();
        self
    }
}

// **** clamp ****

impl<T> Vector2d<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all components clamped to the specified range
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(2.0, 3.0);
    /// let u = v.clamped(2.5, 7.5);
    ///
    /// assert_eq!(u, Vector2df32::new(2.5, 3.0));
    /// ```
    #[inline(always)]
    pub fn clamped(self, min: T, max: T) -> Self {
        Self { x: self.x.clamp(min, max), y: self.y.clamp(min, max) }
    }

    /// Clamp all components of the vector to the specified range
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let mut v = Vector2df32::new(2.0, 3.0);
    /// v.clamp(2.5, 7.5);
    ///
    /// assert_eq!(v, Vector2df32::new(2.5, 3.0));
    /// ```
    #[inline(always)]
    pub fn clamp(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamped(min, max);
        self
    }
}

// **** dot ****

impl<T> Vector2d<T>
where
    T: Copy + Vector2dMath,
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

// **** cross ****

impl<T> Vector2d<T>
where
    T: Copy + Vector2dMath,
{
    /// Z component of vector cross product of self and other extended to 3D
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
    pub fn cross(self, other: Self) -> T {
        T::v2_cross(self, other)
    }
}

// **** norm_squared ****

impl<T> Vector2d<T>
where
    T: Copy + Vector2dMath,
{
    /// Return square of Euclidean norm
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(2.0, 3.0);
    /// assert_eq!(13.0, v.norm_squared());
    /// ```
    #[inline(always)]
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }

    /// Return distance between two points, squared
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(2.0, 3.0);
    /// let w = Vector2df32::new(7.0, 11.0);
    /// assert_eq!(89.0, v.distance_squared(w));
    /// ```
    #[inline(always)]
    pub fn distance_squared(self, other: Self) -> T {
        (self - other).norm_squared()
    }
}

// **** norm ****

impl<T> Vector2d<T>
where
    T: Copy + SqrtMethods + Vector2dMath,
{
    /// Return Euclidean norm
    #[inline(always)]
    pub fn norm(self) -> T {
        Self::norm_squared(self).sqrt()
    }
}

impl<T> Vector2d<T>
where
    T: Copy + Zero + PartialEq + SqrtMethods + Vector2dMath,
{
    /// Return normalized form of the vector, checking if the norm is zero.
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(0.0, 0.0);
    /// let n = v.normalized_checked();
    /// assert_eq!(Vector2df32 { x: 0.0, y: 0.0 }, n);
    /// ```
    #[inline(always)]
    pub fn normalized_checked(self) -> Self {
        let norm_squared = self.norm_squared();
        // If norm == 0.0 then the vector is already normalized
        if norm_squared == T::zero() {
            return self;
        }
        self * norm_squared.sqrt_reciprocal()
    }

    /// Normalize the vector in place, checking if the norm is zero.
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let mut v = Vector2df32::new(3.0, 4.0);
    /// v.normalize_checked();
    /// assert_eq!(Vector2df32 { x: 0.6, y: 0.8 }, v);
    /// ```
    #[inline(always)]
    pub fn normalize_checked(&mut self) -> &mut Self {
        *self = self.normalized_checked();
        self
    }

    /// Return normalized form of the vector
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(3.0, 4.0);
    /// let n = v.normalized();
    /// assert_eq!(Vector2df32 { x: 0.6, y: 0.8 }, n);
    /// ```
    #[inline(always)]
    pub fn normalized(self) -> Self {
        let norm_squared = self.norm_squared();
        self * norm_squared.sqrt_reciprocal()
    }

    /// Normalize the vector in place
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let mut v = Vector2df32::new(3.0, 4.0);
    /// v.normalize();
    /// assert_eq!(Vector2df32 { x: 0.6, y: 0.8 }, v);
    /// ```
    #[inline(always)]
    pub fn normalize(&mut self) -> &mut Self {
        *self = self.normalized();
        self
    }
}

impl<T> Vector2d<T>
where
    T: Copy + Zero + SqrtMethods + Vector2dMath,
{
    // Return distance between two points
    #[inline(always)]
    pub fn distance(self, other: Self) -> T {
        self.distance_squared(other).sqrt()
    }
}

impl<T> Vector2d<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Return the sum of all components of the vector
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(2.0, 3.0);
    /// assert_eq!(5.0, v.sum());
    /// ```
    #[inline(always)]
    pub fn sum(self) -> T {
        self.x + self.y
    }

    /// Return the product of all components of the vector
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(2.0, 3.0);
    /// assert_eq!(6.0, v.product());
    /// ```
    #[inline(always)]
    pub fn product(self) -> T {
        self.x * self.y
    }
}

// **** mean ****

impl<T> Vector2d<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the vector
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(2.0, 3.0);
    /// assert_eq!(2.5, v.mean());
    /// ```
    #[inline(always)]
    pub fn mean(self) -> T {
        (self.x + self.y) / (T::one() + T::one())
    }
}

// **** max ****

impl<T> Vector2d<T>
where
    T: Copy + Vector2dMath,
{
    /// Return the max element in the vector
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(2.0, 3.0);
    /// let w = Vector2df32::new(3.0, 2.0);
    /// assert_eq!(3.0, v.max());
    /// assert_eq!(3.0, w.max());
    /// ```
    #[inline(always)]
    pub fn max(self) -> T {
        T::v2_max(self)
    }

    /// Return the min element in the vector
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    /// let v = Vector2df32::new(2.0, 3.0);
    /// let w = Vector2df32::new(3.0, 2.0);
    /// assert_eq!(2.0, v.min());
    /// assert_eq!(2.0, w.min());
    /// ```
    #[inline(always)]
    pub fn min(self) -> T {
        T::v2_min(self)
    }
}

// **** From ****

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
    #[inline(always)]
    fn from((x, y): (T, T)) -> Self {
        Self { x, y }
    }
}

// **** From Array ****

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
    #[inline(always)]
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
    #[inline(always)]
    fn from(v: Vector2d<T>) -> Self {
        [v.x, v.y]
    }
}
