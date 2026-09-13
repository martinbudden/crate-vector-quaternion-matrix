#[cfg(feature = "uom")]
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{ConstZero, MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};
#[cfg(feature = "serde")]
use {
    postcard::experimental::max_size::MaxSize,
    sequential_storage::map::PostcardValue,
    serde::{Deserialize, Serialize},
};

use crate::{MathConstants, MathMethods, Vector2, Vector3, vector4_math::Vector4Math};

/// 4-dimensional `{x, y, z, t}` vector of `f32` values<br>
pub type Vector4f32 = Vector4<f32>;
/// 4-dimensional `{x, y, z, t}` vector of `f64` values<br><br>
pub type Vector4f64 = Vector4<f64>;

// **** Define ****

/// `Vector4<T>`: 4D vector of type `T`.<br>
/// Aliases `Vector4f32` and `Vector4f64` are provided.<br><br>
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "std", derive(derive_more::Display))]
#[cfg_attr(feature = "std", display("V{{x:{x}, y:{y}, z:{z}, t:{t}}}"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize, MaxSize))]
#[repr(C, align(16))]
#[allow(missing_docs)]
pub struct Vector4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub t: T,
}

#[cfg(feature = "serde")]
impl<T> PostcardValue<'_> for Vector4<T> where T: Serialize + for<'de> Deserialize<'de> {}

// **** New ****

impl<T> Vector4<T>
where
    T: Copy,
{
    /// Create a vector.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0,  3.0, 7.0, 11.0);
    /// assert_eq!(v, Vector4f32 { x:2.0, y:3.0, z: 7.0, t: 11.0 });
    /// ```
    #[inline]
    pub const fn new(x: T, y: T, z: T, t: T) -> Self {
        Self { x, y, z, t }
    }
}

// **** Zero ****

impl<T> Zero for Vector4<T>
where
    T: Copy + ConstZero + PartialEq,
{
    /// Zero vector.
    /// ```
    /// # use vqm::Vector4f32;
    /// # use num_traits::{zero,Zero};
    /// let z: Vector4f32 = zero();
    /// assert!(z.is_zero());
    /// assert_eq!(z, Vector4f32 { x: 0.0, y: 0.0, z: 0.0, t: 0.0 });
    /// ```
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

/// Const zero vector.
/// ```
/// # use vqm::Vector4f32;
/// # use num_traits::{zero,Zero,ConstZero};
/// let z = Vector4f32::ZERO;
/// assert!(z.is_zero());
/// assert_eq!(z, Vector4f32 { x: 0.0, y: 0.0, z: 0.0, t: 0.0 });
/// ```
impl<T> ConstZero for Vector4<T>
where
    T: Copy + Zero + ConstZero + PartialEq,
{
    const ZERO: Self = Self { x: T::ZERO, y: T::ZERO, z: T::ZERO, t: T::ZERO };
}

impl<T> Vector4<T>
where
    T: Copy + FloatCore,
{
    /// Return true if vector is near zero.
    /// ```
    /// # use vqm::Vector4f32;
    /// # use num_traits::Zero;
    /// let z = Vector4f32::zero();
    /// assert!(z.is_near_zero(1e-5));
    /// ```
    pub fn is_near_zero(self, epsilon: T) -> bool {
        if self.x.abs() > epsilon && self.y.abs() > epsilon && self.z.abs() > epsilon && self.t.abs() > epsilon {
            return false;
        }
        true
    }
}

// **** Neg ****

impl<T> Neg for Vector4<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;

    /// Negate vector.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32 { x: 2.0, y: 3.0, z: 5.0, t: 7.0 };
    /// let r = -v;
    ///
    /// assert_eq!(r, Vector4f32 { x: -2.0, y: -3.0, z: -5.0, t: -7.0 });
    /// ```
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z, t: -self.t }
    }
}

// **** Add ****

impl<T> Add for Vector4<T>
where
    T: Copy + Add<T, Output = T>,
{
    type Output = Self;

    /// Add two vectors.
    /// ```
    /// # use vqm::Vector4f32;
    /// let u = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// let v = Vector4f32::new(3.0, 7.0, 13.0, 19.0);
    /// let r = u + v;
    ///
    /// assert_eq!(r, Vector4f32 { x: 5.0, y: 12.0, z: 24.0, t: 36.0 });
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z, t: self.t + other.t }
    }
}

// **** AddAssign ****

impl<T> AddAssign for Vector4<T>
where
    T: Copy + Add<T, Output = T>,
{
    /// Add one vector to another.
    /// ```
    /// # use vqm::Vector4f32;
    /// let mut r = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// let u = Vector4f32::new(3.0, 7.0, 13.0, 19.0);
    /// r += u;
    ///
    /// assert_eq!(r, Vector4f32 { x: 5.0, y: 12.0, z: 24.0, t: 36.0 });
    ///
    /// # use num_traits::zero;
    /// let z: Vector4f32 = zero();
    /// let r = u + z;
    /// assert_eq!(r, u);
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

impl<T> MulAdd<T> for Vector4<T>
where
    T: Copy + Mul<T, Output = T> + Add<T, Output = T>,
{
    type Output = Self;

    /// Multiply vector by constant and add another vector.
    /// ```
    /// # use vqm::Vector4f32;
    /// # use num_traits::MulAdd;
    /// let v = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// let w = Vector4f32::new(3.0, 7.0, 13.0, 19.0);
    /// let k = 23.0;
    /// let r = v.mul_add(k, w);
    ///
    /// assert_eq!(r, Vector4f32 { x: 49.0, y: 122.0, z: 266.0, t: 410.0 });
    /// ```
    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        Self { x: self.x * k + other.x, y: self.y * k + other.y, z: self.z * k + other.z, t: self.t * k + other.t }
    }
}

// **** MulAddAssign ****

impl<T> MulAddAssign<T> for Vector4<T>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
{
    /// Multiply vector by constant and add another vector in place.
    /// ```
    /// # use vqm::Vector4f32;
    /// # use num_traits::MulAddAssign;
    /// let mut v = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// let w = Vector4f32::new(3.0, 7.0, 13.0, 19.0);
    /// let k = 23.0;
    /// v.mul_add_assign(k, w);
    ///
    /// assert_eq!(v, Vector4f32 { x: 49.0, y: 122.0, z: 266.0, t: 410.0 });
    /// ```
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

impl<T> Sub for Vector4<T>
where
    T: Copy + Add<T, Output = T> + Neg<Output = T>,
{
    type Output = Self;

    /// Subtract two vectors.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 5.0, 13.0, 17.0);
    /// let w = Vector4f32::new(3.0, 7.0, 11.0, 23.0);
    /// let r = v - w;
    ///
    /// assert_eq!(r, Vector4f32 { x: -1.0, y: -2.0, z: 2.0, t: -6.0 });
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse our existing Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

impl<T> SubAssign for Vector4<T>
where
    T: Copy + Neg<Output = T> + Add<T, Output = T> + Sub<T, Output = T>,
{
    /// Subtract one vector from another.
    /// ```
    /// # use vqm::Vector4f32;
    /// let mut r = Vector4f32::new(2.0, 5.0, 13.0, 17.0);
    /// let     v = Vector4f32::new(3.0, 7.0, 11.0, 23.0);
    /// r -= v;
    ///
    /// assert_eq!(r, Vector4f32 { x: -1.0, y: -2.0, z: 2.0, t: -6.0 });
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Scalar Mul ****

impl Mul<Vector4<f32>> for f32 {
    type Output = Vector4<f32>;

    /// Pre-multiply vector by a scalar.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// let r = 2.0 * v;
    ///
    /// assert_eq!(r, Vector4f32 { x: 4.0, y: 10.0, z: 22.0, t: 34.0 });
    /// ```
    #[inline]
    fn mul(self, other: Vector4<f32>) -> Vector4<f32> {
        f32::v4_mul_scalar(other, self)
    }
}

impl Mul<Vector4<f64>> for f64 {
    type Output = Vector4<f64>;
    #[inline]
    fn mul(self, other: Vector4<f64>) -> Vector4<f64> {
        Vector4 { x: other.x * self, y: other.y * self, z: other.z * self, t: other.t * self }
    }
}

// **** Mul Scalar ****

#[cfg(not(feature = "uom"))]
impl<T> Mul<T> for Vector4<T>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
{
    type Output = Self;

    /// Multiply vector by a scalar.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 3.0, 5.0, 7.0);
    /// let r = v * 2.0;
    ///
    /// assert_eq!(r, Vector4f32 { x: 4.0, y: 6.0, z: 10.0, t: 14.0 });
    /// ```
    #[inline]
    fn mul(self, k: T) -> Self {
        Self { x: self.x * k, y: self.y * k, z: self.z * k, t: self.t * k }
    }
}

#[cfg(feature = "uom")]
impl<T, Rhs, Out> Mul<Rhs> for Vector4<T>
where
    T: Copy + Mul<Rhs, Output = Out>,
    Rhs: Copy,
{
    type Output = Vector4<Out>;

    /// Multiply a vector by a scalar.
    #[inline]
    fn mul(self, rhs: Rhs) -> Self::Output {
        Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs, t: self.t * rhs }
    }
}

// **** MulAssign ****

impl<T> MulAssign<T> for Vector4<T>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
{
    /// In-place multiply a vector by a scalar.
    /// ```
    /// # use vqm::Vector4f32;
    /// let mut v = Vector4f32::new(2.0, 3.0, 5.0, 7.0);
    /// v *= 2.0;
    ///
    /// assert_eq!(v, Vector4f32 { x: 4.0, y: 6.0, z: 10.0, t: 14.0 });
    /// ```
    #[inline]
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

// **** Mul Elementwise ****

#[cfg(not(feature = "uom"))]
impl<T> Mul<Vector4<T>> for Vector4<T>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
{
    type Output = Self;

    /// Elementwise multiply a vector by another vector.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// let u = Vector4f32::new(3.0, 7.0, 13.0, 19.0);
    /// let r = v * u;
    ///
    /// assert_eq!(r, Vector4f32 { x: 6.0, y: 35.0, z: 143.0, t: 323.0 });
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self { x: self.x * other.x, y: self.y * other.y, z: self.z * other.z, t: self.t * other.t }
    }
}

// **** Div by scalar ****

#[cfg(not(feature = "uom"))]
impl<T> Div<T> for Vector4<T>
where
    T: Copy + One + Add<T, Output = T> + Div<T, Output = T> + Mul<T, Output = T>,
{
    type Output = Self;

    /// Divide a vector by a scalar.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 3.0, 5.0, 7.0);
    /// let r = v / 2.0;
    ///
    /// assert_eq!(r, Vector4f32 { x: 1.0, y: 1.5, z: 2.5, t: 3.5 });
    /// ```
    #[inline]
    fn div(self, k: T) -> Self {
        self.mul(T::one() / k)
    }
}

#[cfg(feature = "uom")]
impl<T, Rhs, Out> Div<Rhs> for Vector4<T>
where
    T: Copy + Div<Rhs, Output = Out>,
    Rhs: Copy,
{
    type Output = Vector4<Out>;

    /// Divide a vector by a scalar.
    #[inline]
    fn div(self, rhs: Rhs) -> Vector4<Out> {
        Vector4::<Out> { x: self.x / rhs, y: self.y / rhs, z: self.z / rhs, t: self.t / rhs }
    }
}

impl<T> DivAssign<T> for Vector4<T>
where
    T: Copy + One + Add<T, Output = T> + Div<T, Output = T> + Mul<T, Output = T>,
{
    /// In-place divide a vector by a scalar.
    /// ```
    /// # use vqm::Vector4f32;
    /// let mut v = Vector4f32::new(2.0, 3.0, 5.0, 7.0);
    /// v /= 2.0;
    ///
    /// assert_eq!(v, Vector4f32 { x: 1.0, y: 1.5, z: 2.5, t: 3.5 });
    /// ```
    #[inline]
    fn div_assign(&mut self, k: T) {
        let k_reciprocal = T::one() / k;
        *self = self.mul(k_reciprocal);
    }
}

// **** Div Elementwise ****

#[cfg(not(feature = "uom"))]
impl<T> Div<Vector4<T>> for Vector4<T>
where
    T: Copy + Div<T, Output = T>,
{
    type Output = Self;

    /// Elementwise divide a vector by another vector.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(3.0, 7.0, 13.0, 19.0);
    /// let u = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// let r = v / u;
    ///
    /// assert_eq!(r, Vector4f32 { x: 1.5, y: 1.4, z: 13.0 / 11.0, t: 19.0 / 17.0 });
    /// ```
    #[inline]
    fn div(self, other: Self) -> Self {
        Self { x: self.x / other.x, y: self.y / other.y, z: self.z / other.z, t: self.t / other.t }
    }
}

// **** Index ****

impl<T> Index<usize> for Vector4<T> {
    type Output = T;

    /// Access vector component by index.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 3.0, 5.0, 7.0);
    ///
    /// assert_eq!(v[0], 2.0);
    /// assert_eq!(v[1], 3.0);
    /// assert_eq!(v[2], 5.0);
    /// assert_eq!(v[3], 7.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &T {
        // make safe by using index = 0 if index out of range
        let safe_index = if index < 4 { index } else { 0 };
        unsafe {
            let ptr = core::ptr::from_ref::<Self>(self).cast::<T>();
            &*ptr.add(safe_index)
        }
    }
}

// **** IndexMut ****

impl<T> IndexMut<usize> for Vector4<T> {
    // Set vector component by index.
    /// ```
    /// # use vqm::Vector4f32;
    /// let mut v = Vector4f32::new(2.0, 3.0, 5.0, 7.0);
    /// v[0] = 7.0;
    /// v[1] = 11.0;
    /// v[2] = 13.0;
    /// v[3] = 17.0;
    ///
    /// assert_eq!(v, Vector4f32 { x:7.0, y:11.0, z:13.0, t: 17.0 });
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        // make safe by using index = 0 if index out of range
        let safe_index = if index < 4 { index } else { 0 };
        unsafe {
            let ptr = core::ptr::from_mut::<Self>(self).cast::<T>();
            &mut *ptr.add(safe_index)
        }
    }
}

// **** lerp ****

impl<T> Vector4<T>
where
    T: Copy,
    Vector4<T>: Mul<T, Output = Vector4<T>> + Add<Output = Vector4<T>> + Sub<Output = Vector4<T>>,
{
    /// Linear interpolation between two vectors.
    /// Calculates `self * (1 - t) + other * t`.
    /// ```
    /// # use vqm::Vector4f32;
    /// let u = Vector4f32::new(2.0, 5.0, 11.0, 13.0);
    /// let v = Vector4f32::new(3.0, 7.0, 17.0, 23.0);
    /// let w = u.lerp(v, 0.25);
    ///
    /// assert_eq!(w, Vector4f32::new(2.25, 5.5, 12.5, 15.5));
    /// ```
    #[inline]
    #[must_use]
    pub fn lerp(self, other: Self, t: T) -> Self {
        self + (other - self) * t
    }
}

// **** approx_eq ****

impl<T> Vector4<T>
where
    T: FloatCore,
{
    /// Compare two vectors with a tolerance.
    #[inline]
    pub fn approx_eq(&self, other: &Self, epsilon: T) -> bool
    where
        T: FloatCore,
    {
        (self.x - other.x).abs() <= epsilon
            && (self.y - other.y).abs() <= epsilon
            && (self.z - other.z).abs() <= epsilon
            && (self.t - other.t).abs() <= epsilon
    }
}

// **** abs ****

impl<T> Vector4<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the vector with all components set to their absolute values.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, -3.0, -5.0, 7.0);
    /// let u = v.abs();
    ///
    /// assert_eq!(u, Vector4f32::new(2.0, 3.0, 5.0, 7.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs(), t: self.t.abs() }
    }

    /// Set all components of the vector to their absolute values.
    /// ```
    /// # use vqm::Vector4f32;
    /// let mut v = Vector4f32::new(2.0, -3.0, -5.0, 7.0);
    /// v.abs_in_place();
    ///
    /// assert_eq!(v, Vector4f32::new(2.0, 3.0, 5.0, 7.0));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = self.abs();
        self
    }
}

// **** clamp ****

impl<T> Vector4<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the vector with all components clamped to the specified range.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 3.0, 7.0, 11.0);
    /// let u = v.clamp(2.5, 7.5);
    ///
    /// assert_eq!(u, Vector4f32::new(2.5, 3.0, 7.0, 7.5));
    /// ```
    #[inline]
    #[must_use]
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
    /// # use vqm::Vector4f32;
    /// let mut v = Vector4f32::new(2.0, 3.0, 7.0, 11.0);
    /// v.clamp_in_place(2.5, 7.5);
    ///
    /// assert_eq!(v, Vector4f32::new(2.5, 3.0, 7.0, 7.5));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

// **** dot ****

impl<T> Vector4<T>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
{
    /// Vector dot product.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// let w = Vector4f32::new(3.0, 7.0, 13.0, 19.0);
    ///
    /// let x = v.dot(w);
    ///
    /// assert_eq!(x, 507.0);
    /// ```
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z + self.t * other.t
    }
}

#[cfg(feature = "uom")]
impl<T> Vector4<T> {
    /// Calculates the dot product of two vectors.
    pub fn dot_uom<Rhs, Out>(self, rhs: Vector4<Rhs>) -> Out
    where
        T: Mul<Rhs, Output = Out>,
        Out: Add<Output = Out>,
    {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.t * rhs.t
    }
}

// **** norm_squared ****

impl<T> Vector4<T>
where
    T: Copy + Neg<Output = T> + Add<Output = T> + Mul<T, Output = T>,
{
    /// Return square of Euclidean norm.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// assert_eq!(439.0, v.norm_squared());
    /// ```
    #[inline]
    pub fn norm_squared(self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z + self.t * self.t
    }

    /// Return distance between two points, squared.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 5.0, 11.0, 13.0);
    /// let w = Vector4f32::new(3.0, 7.0, 17.0, 23.0);
    /// assert_eq!(141.0, v.distance_squared(w));
    /// ```
    #[inline]
    pub fn distance_squared(self, other: Self) -> T {
        (self - other).norm_squared()
    }
}

#[cfg(feature = "uom")]
impl<D, U, V> Vector4<uom::si::Quantity<D, U, V>>
where
    D: uom::si::Dimension + ?Sized,
    U: uom::si::Units<V> + ?Sized,
    V: Copy + num_traits::Float + uom::Conversion<V>,
    uom::si::Quantity<D, U, V>: Copy,
{
    /// Calculates the squared norm of the vector.
    #[inline]
    pub fn norm_squared_uom<Out>(self) -> Out
    where
        uom::si::Quantity<D, U, V>: Mul<uom::si::Quantity<D, U, V>, Output = Out>,
        Out: Add<Output = Out>,
    {
        self.x * self.x + self.y * self.y + self.z * self.z + self.t * self.t
    }

    /// Calculates the Euclidean norm (length) of the vector.
    #[inline]
    pub fn norm_uom<Intermediate>(self) -> uom::si::Quantity<D, U, V>
    where
        uom::si::Quantity<D, U, V>: Mul<uom::si::Quantity<D, U, V>, Output = Intermediate>,
        Intermediate: Add<Output = Intermediate>,
    {
        // Extract raw scalar primitive values
        let x = self.x.value;
        let y = self.y.value;
        let z = self.z.value;
        let t = self.t.value;
        let norm = (x * x + y * y + z * z + t * t).sqrt();

        uom::si::Quantity { dimension: PhantomData, units: PhantomData, value: norm }
    }

    #[inline]
    /// Normalizes the vector, returning a unit vector pointing in the same direction.
    pub fn normalize_uom<Intermediate>(self) -> Vector4<uom::si::ratio::Ratio<U, V>>
    where
        uom::si::Quantity<D, U, V>: Mul<uom::si::Quantity<D, U, V>, Output = Intermediate>,
        Intermediate: Add<Output = Intermediate>,
    {
        let x = self.x.value;
        let y = self.y.value;
        let z = self.z.value;
        let t = self.t.value;
        let norm = (x * x + y * y + z * z + t * t).sqrt();
        let norm_reciprocal = V::one() / norm;

        Self {
            x: uom::si::Quantity { dimension: PhantomData, units: PhantomData, value: x * norm_reciprocal },
            y: uom::si::Quantity { dimension: PhantomData, units: PhantomData, value: y * norm_reciprocal },
            z: uom::si::Quantity { dimension: PhantomData, units: PhantomData, value: z * norm_reciprocal },
            t: uom::si::Quantity { dimension: PhantomData, units: PhantomData, value: t * norm_reciprocal },
        }
    }
}

// **** norm ****

impl<T> Vector4<T>
where
    T: Copy + Neg<Output = T> + Add<Output = T> + Mul<T, Output = T> + MathMethods,
{
    /// Return Euclidean norm.
    #[inline]
    pub fn norm(self) -> T {
        Self::norm_squared(self).sqrt()
    }
}

impl<T> Vector4<T>
where
    T: Copy + Zero + Neg<Output = T> + Add<Output = T> + Mul<T, Output = T> + PartialEq + MathMethods,
{
    /// Return normalized form of the vector, checking if the norm is zero.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(0.0, 0.0, 0.0, 0.0);
    /// let n = v.normalize();
    /// assert_eq!(Vector4f32 { x: 0.0, y: 0.0, z: 0.0, t: 0.0 }, n);
    /// ```
    #[inline]
    #[must_use]
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
    /// # use vqm::Vector4f32;
    /// let mut v = Vector4f32::new(0.0, 0.0, 0.0, 0.0);
    /// v.normalize_in_place();
    /// assert_eq!(Vector4f32 { x: 0.0, y: 0.0, z: 0.0, t: 0.0 }, v);
    /// ```
    #[inline]
    pub fn normalize_in_place(&mut self) -> &mut Self {
        *self = self.normalize();
        self
    }

    /// Return normalized form of the vector, not checking if the norm is zero.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 3.0, 5.0, 7.0);
    /// let n = v.normalize_unchecked();
    /// assert!((Vector4f32 { x: 0.21442251, y: 0.32163376, z: 0.5360563, t: 0.7504788 } - n).is_near_zero(4.0e-4));
    /// ```
    #[inline]
    #[must_use]
    pub fn normalize_unchecked(self) -> Self {
        let norm_squared = self.norm_squared();
        self * norm_squared.sqrt_reciprocal()
    }

    /// Normalize the vector in place, not checking if the norm is zero.
    /// ```
    /// # use vqm::Vector4f32;
    /// let mut v = Vector4f32::new(2.0, 3.0, 5.0, 7.0);
    /// v.normalize_unchecked_in_place();
    /// assert!((Vector4f32 { x: 0.21442251, y: 0.32163376, z: 0.5360563, t: 0.7504788 } - v).is_near_zero(4.0e-4));
    /// ```
    #[inline]
    pub fn normalize_unchecked_in_place(&mut self) -> &mut Self {
        *self = self.normalize_unchecked();
        self
    }
}

impl<T> Vector4<T>
where
    T: Copy + Zero + Neg<Output = T> + Add<Output = T> + Mul<T, Output = T> + MathMethods,
{
    /// Return distance between two points.
    #[inline]
    pub fn distance(self, other: Self) -> T {
        self.distance_squared(other).sqrt()
    }
}

// **** to_degrees ****

impl<T> Vector4<T>
where
    T: Copy + FloatCore,
{
    /// Convert the vector to degrees, assuming it is in radians.
    /// ```
    /// # use vqm::Vector4f32;
    /// # use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8};
    /// let v = Vector4f32::new(FRAC_PI_2, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8);
    /// let w = v.to_degrees();
    /// assert!((w.x - 90.0).abs() < 2e-6);
    /// assert!((w.y - 45.0).abs() < 2e-6);
    /// assert!((w.z - 30.0).abs() < 2e-6);
    /// assert!((w.t - 22.5).abs() < 2e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_degrees(self) -> Self {
        Self { x: self.x.to_degrees(), y: self.y.to_degrees(), z: self.z.to_degrees(), t: self.t.to_degrees() }
    }

    /// Convert the vector to radians, assuming it is in degrees.
    /// ```
    /// # use vqm::Vector4f32;
    /// # use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8};
    /// let v = Vector4f32::new(90.0, 45.0, 30.0, 22.5);
    /// assert_eq!(Vector4f32::new(FRAC_PI_2, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8), v.to_radians());
    /// ```
    #[inline]
    #[must_use]
    pub fn to_radians(self) -> Self {
        Self { x: self.x.to_radians(), y: self.y.to_radians(), z: self.z.to_radians(), t: self.t.to_radians() }
    }
}

impl<T> Vector4<T>
where
    T: Copy + Mul<Output = T> + MathConstants,
{
    /// Convert the vector to meters per second squared, assuming it is in earth gravity units.
    /// ```
    /// # use vqm::{Vector4f32, MathConstants};
    /// let v = Vector4f32::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(Vector4f32::new(9.806_65, 19.613_3, 29.419_95, 39.226_6), v.g_to_mps2());
    /// ```
    #[inline]
    #[must_use]
    pub fn g_to_mps2(self) -> Self {
        Self { x: self.x * T::G0, y: self.y * T::G0, z: self.z * T::G0, t: self.t * T::G0 }
    }

    /// Convert the vector to earth gravity units, assuming it is in meters per second squared.
    /// ```
    /// # use vqm::{Vector4f32, MathConstants};
    /// let v = Vector4f32::new(9.806_65, 19.613_3, 29.419_95, 39.226_6);
    /// assert_eq!(Vector4f32::new(1.0, 2.0, 3.0, 4.0), v.mps2_to_g());
    /// ```
    #[inline]
    #[must_use]
    pub fn mps2_to_g(self) -> Self {
        Self {
            x: self.x * T::G0_RECIPROCAL,
            y: self.y * T::G0_RECIPROCAL,
            z: self.z * T::G0_RECIPROCAL,
            t: self.t * T::G0_RECIPROCAL,
        }
    }
}

// **** sum ****

impl<T> Vector4<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    /// Return the sum of all components of the vector.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// assert_eq!(35.0, v.sum());
    /// ```
    #[inline]
    pub fn sum(self) -> T {
        self.x + self.y + self.z + self.t
    }

    /// Return the product of all components of the vector.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// assert_eq!(1870.0, v.product());
    /// ```
    #[inline]
    pub fn product(self) -> T {
        self.x * self.y * self.z * self.t
    }
}

// **** mean ****

impl<T> Vector4<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the vector.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 5.0, 11.0, 17.0);
    /// assert_eq!(8.75, v.mean());
    /// ```
    #[inline]
    pub fn mean(self) -> T {
        let four = T::one() + T::one() + T::one() + T::one();
        self.sum() / four
    }
}

// **** max ****

impl<T> Vector4<T>
where
    T: Copy + Vector4Math,
{
    /// Return the max element in the vector.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Vector4f32::new(3.0, 5.0, 7.0, 2.0);
    /// let x = Vector4f32::new(5.0, 7.0, 3.0, 2.0);
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
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Vector4f32::new(3.0, 5.0, 7.0, 2.0);
    /// let x = Vector4f32::new(5.0, 7.0, 3.0, 2.0);
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

impl<T> From<(T, T, T, T)> for Vector4<T> {
    /// Vector from tuple.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::from((2.0, 5.0, 11.0, 17.0));
    /// let w: Vector4f32 = (3.0, 7.0, 13.0, 19.0).into();
    ///
    /// assert_eq!(v, Vector4f32 { x: 2.0, y: 5.0, z: 11.0, t: 17.0 });
    /// assert_eq!(w, Vector4f32 { x: 3.0, y: 7.0, z: 13.0, t: 19.0 });
    /// ```
    #[inline]
    fn from((x, y, z, t): (T, T, T, T)) -> Self {
        Self { x, y, z, t }
    }
}

// **** From Array ****

impl<T: Copy> From<[T; 4]> for Vector4<T> {
    /// Vector from array.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32::from([2.0, 5.0, 11.0, 17.0]);
    /// let w: Vector4f32 = [3.0, 7.0, 13.0, 19.0].into();
    ///
    /// assert_eq!(v, Vector4f32 { x: 2.0, y: 5.0, z: 11.0, t: 17.0 });
    /// assert_eq!(w, Vector4f32 { x: 3.0, y: 7.0, z: 13.0, t: 19.0 });
    /// ```
    #[inline]
    fn from(v: [T; 4]) -> Self {
        Self { x: v[0], y: v[1], z: v[2], t: v[3] }
    }
}

impl<T> From<Vector4<T>> for [T; 4] {
    /// Array from vector.
    /// ```
    /// # use vqm::Vector4f32;
    /// let v = Vector4f32 { x: 2.0, y: 5.0, z: 11.0, t: 17.0 };
    ///
    /// let a = <[f32; 4]>::from(v);
    /// let b: [f32; 4] = v.into();
    ///
    /// assert_eq!(a, [2.0, 5.0, 11.0, 17.0]);
    /// assert_eq!(b, [2.0, 5.0, 11.0, 17.0]);
    /// ```
    #[inline]
    fn from(v: Vector4<T>) -> Self {
        [v.x, v.y, v.z, v.t]
    }
}

impl From<[i16; 4]> for Vector4<f32> {
    #[inline]
    fn from(v: [i16; 4]) -> Self {
        Self { x: f32::from(v[0]), y: f32::from(v[1]), z: f32::from(v[2]), t: f32::from(v[3]) }
    }
}

impl From<Vector4<f32>> for [i16; 4] {
    #[inline]
    fn from(v: Vector4<f32>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        [v.x as i16, v.y as i16, v.z as i16, v.t as i16]
    }
}

impl From<[i32; 4]> for Vector4<f32> {
    #[inline]
    fn from(v: [i32; 4]) -> Self {
        #[allow(clippy::cast_precision_loss)]
        Self { x: v[0] as f32, y: v[1] as f32, z: v[2] as f32, t: v[3] as f32 }
    }
}

impl From<Vector4<f32>> for [i32; 4] {
    #[inline]
    fn from(v: Vector4<f32>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        [v.x as i32, v.y as i32, v.z as i32, v.t as i32]
    }
}

// **** From Vector ****

impl<T: Zero> From<Vector2<T>> for Vector4<T> {
    /// Vector4 from Vector2.
    /// ```
    /// # use vqm::{Vector2f32,Vector4f32};
    /// let v = Vector4f32::from(Vector2f32 { x: 2.0, y: 5.0 });
    /// let w: Vector4f32 = Vector2f32 { x: 3.0, y: 7.0 }.into();
    ///
    /// assert_eq!(v, Vector4f32 { x: 2.0, y: 5.0, z: 0.0, t: 0.0 });
    /// assert_eq!(w, Vector4f32 { x: 3.0, y: 7.0, z: 0.0, t: 0.0 });
    /// ```
    #[inline]
    fn from(other: Vector2<T>) -> Self {
        Self { x: other.x, y: other.y, z: T::zero(), t: T::zero() }
    }
}

impl<T: Zero> From<Vector3<T>> for Vector4<T> {
    /// Vector4 from Vector3.
    /// ```
    /// # use vqm::{Vector3f32,Vector4f32};
    /// let v = Vector4f32::from(Vector3f32 { x: 2.0, y: 5.0, z: 11.0 });
    /// let w: Vector4f32 = Vector3f32 { x: 3.0, y: 7.0, z: 13.0 }.into();
    ///
    /// assert_eq!(v, Vector4f32 { x: 2.0, y: 5.0, z: 11.0, t: 0.0 });
    /// assert_eq!(w, Vector4f32 { x: 3.0, y: 7.0, z: 13.0, t: 0.0 });
    /// ```
    #[inline]
    fn from(other: Vector3<T>) -> Self {
        Self { x: other.x, y: other.y, z: other.z, t: T::zero() }
    }
}
