#[cfg(feature = "uom")]
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{ConstZero, MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};
#[cfg(feature = "storage")]
use sequential_storage::map::PostcardValue;
#[cfg(feature = "serde")]
use {
    postcard::experimental::max_size::MaxSize,
    serde::{Deserialize, Serialize},
};

use crate::{MathConstants, MathMethods, Quaternion, QuaternionMath, Vector2, vector3_math::Vector3Math};

/// 3-dimensional `{x, y, z}` vector of `f32` values<br>
pub type Vector3f32 = Vector3<f32>;
/// 3-dimensional `{x, y, z}` vector of `f64` values<br><br>
pub type Vector3f64 = Vector3<f64>;

// **** Define ****

/// `Vector3<T>`: 3D vector of type `T`.<br>
/// Aliases `Vector3f32` and `Vector2f64` are provided.<br><br>
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "std", derive(derive_more::Display))]
#[cfg_attr(feature = "std", display("V{{x:{x}, y:{y}, z:{z}}}"))]
// Conditionally derive serde traits
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize, MaxSize))]
// Conditionally apply alignment based on "align" feature
#[cfg_attr(feature = "align", repr(C, align(16)))]
#[cfg_attr(not(feature = "align"), repr(C))]
#[allow(missing_docs)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[cfg(feature = "storage")]
impl<T> PostcardValue<'_> for Vector3<T> where T: Serialize + for<'de> Deserialize<'de> {}

// **** New ****

impl<T> Vector3<T>
where
    T: Copy,
{
    /// Create a vector.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0,  3.0, 7.0);
    /// assert_eq!(v, Vector3f32 { x:2.0, y:3.0, z: 7.0 });
    /// ```
    #[inline]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

// **** Zero ****

impl<T> Zero for Vector3<T>
where
    T: Copy + ConstZero + PartialEq,
{
    /// Zero vector.
    /// ```
    /// # use vqm::Vector3f32;
    /// # use num_traits::{zero,Zero};
    /// let z: Vector3f32 = zero();
    /// assert!(z.is_zero());
    /// assert_eq!(z, Vector3f32 { x: 0.0, y: 0.0, z: 0.0 });
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
/// # use vqm::Vector3f32;
/// # use num_traits::{zero,Zero,ConstZero};
/// let z = Vector3f32::ZERO;
/// assert!(z.is_zero());
/// assert_eq!(z, Vector3f32 { x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl<T> ConstZero for Vector3<T>
where
    T: Copy + Zero + ConstZero + PartialEq,
{
    const ZERO: Self = Self { x: T::ZERO, y: T::ZERO, z: T::ZERO };
}

impl<T> Vector3<T>
where
    T: Copy + FloatCore,
{
    /// Return true if vector is near zero.
    /// ```
    /// # use vqm::Vector3f32;
    /// # use num_traits::Zero;
    /// let z = Vector3f32::zero();
    /// assert!(z.is_near_zero(1e-5));
    /// ```
    pub fn is_near_zero(self, epsilon: T) -> bool {
        if self.x.abs() > epsilon && self.y.abs() > epsilon && self.z.abs() > epsilon {
            return false;
        }
        true
    }
}

// **** Neg ****

impl<T> Neg for Vector3<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;

    /// Negate vector.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32 { x: 2.0, y: 3.0, z: 5.0 };
    /// let r = -v;
    ///
    /// assert_eq!(r, Vector3f32 { x: -2.0, y: -3.0, z: -5.0 });
    /// ```
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

// **** Add ****

impl<T> Add for Vector3<T>
where
    T: Copy + Add<T, Output = T>,
{
    type Output = Self;

    /// Add two vectors.
    /// ```
    /// # use vqm::Vector3f32;
    /// let u = Vector3f32::new(2.0, 5.0, 11.0);
    /// let v = Vector3f32::new(3.0, 7.0, 13.0);
    /// let r = u + v;
    ///
    /// assert_eq!(r, Vector3f32 { x: 5.0, y: 12.0, z: 24.0 });
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
}

// **** AddAssign ****

impl<T> AddAssign for Vector3<T>
where
    T: Copy + Add<T, Output = T>,
{
    /// Add one vector to another.
    /// ```
    /// # use vqm::Vector3f32;
    /// let mut r = Vector3f32::new(2.0, 5.0, 11.0);
    /// let u = Vector3f32::new(3.0, 7.0, 13.0);
    /// r += u;
    ///
    /// assert_eq!(r, Vector3f32 { x: 5.0, y: 12.0, z: 24.0 });
    ///
    /// # use num_traits::zero;
    /// let z: Vector3f32 = zero();
    /// let r = u + z;
    /// assert_eq!(r, u);
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

impl<T> MulAdd<T> for Vector3<T>
where
    T: Copy + Mul<T, Output = T> + Add<T, Output = T>,
{
    type Output = Self;

    /// Multiply vector by constant and add another vector.
    /// ```
    /// # use vqm::Vector3f32;
    /// # use num_traits::MulAdd;
    /// let mut v = Vector3f32::new(2.0, 5.0, 11.0);
    /// let w = Vector3f32::new(3.0, 7.0, 13.0);
    /// let k = 23.0;
    /// let r = v.mul_add(k, w);
    ///
    /// assert_eq!(r, Vector3f32 { x: 49.0, y: 122.0, z: 266.0 });
    /// ```
    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        Self { x: self.x * k + other.x, y: self.y * k + other.y, z: self.z * k + other.z }
    }
}

// **** MulAddAssign ****

impl<T> MulAddAssign<T> for Vector3<T>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
{
    /// Multiply vector by constant and add another vector in place.
    /// ```
    /// # use vqm::Vector3f32;
    /// # use num_traits::MulAddAssign;
    /// let mut v = Vector3f32::new(2.0, 5.0, 11.0);
    /// let w = Vector3f32::new(3.0, 7.0, 13.0);
    /// let k = 23.0;
    /// v.mul_add_assign(k, w);
    ///
    /// assert_eq!(v, Vector3f32 { x: 49.0, y: 122.0, z: 266.0 });
    /// ```
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

impl<T> Sub for Vector3<T>
where
    T: Copy + Add<T, Output = T> + Neg<Output = T>,
{
    type Output = Self;

    /// Subtract two vectors.
    /// ```
    /// # use vqm::Vector3f32;
    /// let u = Vector3f32::new(2.0, 5.0, 13.0);
    /// let v = Vector3f32::new(3.0, 7.0, 11.0);
    /// let r = u - v;
    ///
    /// assert_eq!(r, Vector3f32 { x: -1.0, y: -2.0, z: 2.0 });
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse our existing Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

impl<T> SubAssign for Vector3<T>
where
    T: Copy + Neg<Output = T> + Add<T, Output = T> + Sub<T, Output = T>,
{
    /// Subtract one vector from another.
    /// ```
    /// # use vqm::Vector3f32;
    /// let mut r = Vector3f32::new(2.0, 5.0, 13.0);
    /// let     v = Vector3f32::new(3.0, 7.0, 11.0);
    /// r -= v;
    ///
    /// assert_eq!(r, Vector3f32 { x: -1.0, y: -2.0, z: 2.0 });
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Scalar Mul ****

impl Mul<Vector3<f32>> for f32 {
    type Output = Vector3<f32>;

    /// Pre-multiply vector by a scalar.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    /// let r = 2.0 * v;
    ///
    /// assert_eq!(r, Vector3f32 { x: 4.0, y: 10.0, z: 22.0 });
    /// ```
    #[inline]
    fn mul(self, other: Vector3<f32>) -> Vector3<f32> {
        f32::v3_mul_scalar(other, self)
    }
}

impl Mul<Vector3<f64>> for f64 {
    type Output = Vector3<f64>;

    #[inline]
    fn mul(self, other: Vector3<f64>) -> Vector3<f64> {
        Vector3 { x: other.x * self, y: other.y * self, z: other.z * self }
    }
}

// **** Mul Scalar ****

#[cfg(not(feature = "uom"))]
impl<T> Mul<T> for Vector3<T>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
    // #[cfg(feature = "simd")] T: crate::Vector3Math, placeholder code for when this feature becomes stable
{
    type Output = Self;

    /// Multiply vector by a scalar.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    /// let r = v * 2.0;
    ///
    /// assert_eq!(r, Vector3f32 { x: 4.0, y: 10.0, z: 22.0 });
    /// ```
    #[inline]
    fn mul(self, k: T) -> Self {
        /* placeholder code for when attributes in where clause becomes stable
        #[cfg(feature = "simd")]
        {
            T::v3_mul_scalar(self, k)
        }
        #[cfg(not(feature = "simd"))]*/
        Self { x: self.x * k, y: self.y * k, z: self.z * k }
    }
}

#[cfg(feature = "uom")]
impl<T, Rhs, Out> Mul<Rhs> for Vector3<T>
where
    T: Copy + Mul<Rhs, Output = Out>,
    Rhs: Copy,
{
    type Output = Vector3<Out>;

    /// Multiply a vector by a scalar.
    #[inline]
    fn mul(self, rhs: Rhs) -> Self::Output {
        Vector3 { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs }
    }
}

// **** MulAssign ****

impl<T> MulAssign<T> for Vector3<T>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
{
    /// In-place multiply a vector by a scalar.
    /// ```
    /// # use vqm::Vector3f32;
    /// let mut v = Vector3f32::new(2.0, 5.0, 11.0);
    /// v *= 2.0;
    ///
    /// assert_eq!(v, Vector3f32 { x: 4.0, y: 10.0, z: 22.0 });
    /// ```
    #[inline]
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

// **** Mul Elementwise ****

#[cfg(not(feature = "uom"))]
impl<T> Mul<Vector3<T>> for Vector3<T>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
{
    type Output = Self;

    /// Elementwise multiply a vector by another vector.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    /// let u = Vector3f32::new(3.0, 7.0, 13.0);
    /// let r = v * u;
    ///
    /// assert_eq!(r, Vector3f32 { x: 6.0, y: 35.0, z: 143.0 });
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self { x: self.x * other.x, y: self.y * other.y, z: self.z * other.z }
    }
}

// **** Div by scalar ****

#[cfg(not(feature = "uom"))]
impl<T> Div<T> for Vector3<T>
where
    T: Copy + One + Add<T, Output = T> + Div<T, Output = T> + Mul<T, Output = T>,
{
    type Output = Self;

    /// Divide a vector by a scalar.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 3.0, 5.0);
    /// let r = v / 2.0;
    ///
    /// assert_eq!(r, Vector3f32 { x: 1.0, y: 1.5, z: 2.5 });
    /// ```
    #[inline]
    fn div(self, k: T) -> Self {
        self.mul(T::one() / k)
    }
}

#[cfg(feature = "uom")]
impl<T, Rhs, Out> Div<Rhs> for Vector3<T>
where
    T: Copy + Div<Rhs, Output = Out>,
    Rhs: Copy,
{
    type Output = Vector3<Out>;

    /// Divide a vector by a scalar.
    #[inline]
    fn div(self, rhs: Rhs) -> Vector3<Out> {
        Vector3::<Out> { x: self.x / rhs, y: self.y / rhs, z: self.z / rhs }
    }
}

impl<T> DivAssign<T> for Vector3<T>
where
    T: Copy + One + Add<T, Output = T> + Div<T, Output = T> + Mul<T, Output = T>,
{
    /// In-place divide a vector by a scalar.
    /// ```
    /// # use vqm::Vector3f32;
    /// let mut v = Vector3f32::new(2.0, 3.0, 5.0);
    /// v /= 2.0;
    ///
    /// assert_eq!(v, Vector3f32 { x: 1.0, y: 1.5, z: 2.5 });
    /// ```
    #[inline]
    fn div_assign(&mut self, k: T) {
        let k_reciprocal = T::one() / k;
        *self = self.mul(k_reciprocal);
    }
}

// **** Div Elementwise ****

#[cfg(not(feature = "uom"))]
impl<T> Div<Vector3<T>> for Vector3<T>
where
    T: Copy + Div<T, Output = T>,
{
    type Output = Self;

    /// Elementwise divide a vector by another vector.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(3.0, 7.0, 13.0);
    /// let u = Vector3f32::new(2.0, 5.0, 11.0);
    /// let r = v / u;
    ///
    /// assert_eq!(r, Vector3f32 { x: 1.5, y: 1.4, z: 13.0 / 11.0 });
    /// ```
    #[inline]
    fn div(self, other: Self) -> Self {
        Self { x: self.x / other.x, y: self.y / other.y, z: self.z / other.z }
    }
}

// **** Index ****

impl<T> Index<usize> for Vector3<T> {
    type Output = T;

    /// Access vector component by index.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    ///
    /// assert_eq!(v[0], 2.0);
    /// assert_eq!(v[1], 5.0);
    /// assert_eq!(v[2], 11.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}

// **** IndexMut ****

impl<T> IndexMut<usize> for Vector3<T> {
    // Set vector component by index.
    /// ```
    /// # use vqm::Vector3f32;
    /// let mut v = Vector3f32::new(2.0, 5.0, 11.0);
    /// v[0] = 3.0;
    /// v[1] = 7.0;
    /// v[2] = 13.0;
    ///
    /// assert_eq!(v, Vector3f32 { x:3.0, y:7.0, z:13.0 });
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.z,
        }
    }
}

// **** lerp ****

impl<T> Vector3<T>
where
    T: Copy,
    Vector3<T>: Mul<T, Output = Vector3<T>> + Add<Output = Vector3<T>> + Sub<Output = Vector3<T>>,
{
    /// Linear interpolation between two vectors.
    /// Calculates `self * (1 - t) + other * t`.
    /// ```
    /// # use vqm::Vector3f32;
    /// let u = Vector3f32::new(2.0, 5.0, 11.0);
    /// let v = Vector3f32::new(3.0, 7.0, 17.0);
    /// let w = u.lerp(v, 0.25);
    ///
    /// assert_eq!(w, Vector3f32::new(2.25, 5.5, 12.5));
    /// ```
    #[inline]
    #[must_use]
    pub fn lerp(self, other: Self, t: T) -> Self {
        self + (other - self) * t
    }
}

// **** approx_eq ****

impl<T> Vector3<T>
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
    }
}

// **** abs ****

impl<T> Vector3<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the vector with all components set to their absolute values.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, -5.0, -11.0);
    /// let u = v.abs();
    ///
    /// assert_eq!(u, Vector3f32::new(2.0, 5.0, 11.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() }
    }

    /// Set all components of the vector to their absolute values.
    /// ```
    /// # use vqm::Vector3f32;
    /// let mut v = Vector3f32::new(2.0, -5.0, -11.0);
    /// v.abs_in_place();
    ///
    /// assert_eq!(v, Vector3f32::new(2.0, 5.0, 11.0));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = self.abs();
        self
    }
}

// **** clamp ****

impl<T> Vector3<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the vector with all components clamped to the specified range.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    /// let u = v.clamp(2.5, 7.5);
    ///
    /// assert_eq!(u, Vector3f32::new(2.5, 5.0, 7.5));
    /// ```
    #[inline]
    #[must_use]
    pub fn clamp(self, min: T, max: T) -> Self {
        Self { x: self.x.clamp(min, max), y: self.y.clamp(min, max), z: self.z.clamp(min, max) }
    }

    /// Clamp all components of the vector to the specified range.
    /// ```
    /// # use vqm::Vector3f32;
    /// let mut v = Vector3f32::new(2.0, 5.0, 11.0);
    /// v.clamp_in_place(2.5, 7.5);
    ///
    /// assert_eq!(v, Vector3f32::new(2.5, 5.0, 7.5));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

// **** dot ****

impl<T> Vector3<T>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
{
    /// Vector dot product.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    /// let w = Vector3f32::new(3.0, 7.0, 13.0);
    ///
    /// let x = v.dot(w);
    ///
    /// assert_eq!(x, 184.0);
    /// ```
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

#[cfg(feature = "uom")]
impl<T> Vector3<T> {
    /// Calculates the dot product of two vectors.
    ///
    /// When using `uom`, the output quantity automatically scales its dimensions
    /// dynamically based on the input quantities (e.g., Length * Force = Energy).
    ///
    /// # Example
    ///
    /// ```
    /// # use uom::si::f32::{Length, Force, Energy};
    /// # use uom::si::{length::meter,force::newton,energy::joule};
    /// # use vqm::Vector3;
    ///
    /// // Create a displacement vector (Length)
    /// let displacement = Vector3 {
    ///     x: Length::new::<meter>(2.0),
    ///     y: Length::new::<meter>(3.0),
    ///     z: Length::new::<meter>(4.0),
    /// };
    ///
    /// // Create a constant pushing force vector (Force)
    /// let force = Vector3 {
    ///     x: Force::new::<newton>(10.0),
    ///     y: Force::new::<newton>(5.0),
    ///     z: Force::new::<newton>(0.0),
    /// };
    ///
    /// // Calculate mechanical work done: (2*10) + (3*5) + (4*0) = 35 Joules
    /// let work_done: Energy = displacement.dot_uom(force);
    ///
    /// assert_eq!(work_done, Energy::new::<joule>(35.0));
    /// ```
    pub fn dot_uom<Rhs, Out>(self, rhs: Vector3<Rhs>) -> Out
    where
        T: Mul<Rhs, Output = Out>,
        Out: Add<Output = Out>,
    {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

// **** cross ****

impl<T> Vector3<T>
where
    T: Copy + Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T>,
{
    /// Vector cross product.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    /// let w = Vector3f32::new(3.0, 7.0, 13.0);
    ///
    /// let x = v.cross(w);
    ///
    /// assert_eq!(x, Vector3f32::new(-12.0, 7.0, -1.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn cross(self, other: Self) -> Vector3<T> {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

#[cfg(feature = "uom")]
impl<T> Vector3<T> {
    /// Calculates the cross product of two 3D vectors.
    ///
    /// When using `uom`, the dimensions of the output vector adapt dynamically
    /// based on the input quantities (e.g., Length × Force = Torque).
    ///
    /// # Example
    ///
    /// ```
    /// # use uom::si::f32::{Length, Area};
    /// # use uom::si::{length::meter,area::square_meter};
    /// # use vqm::Vector3;
    ///
    /// let a = Vector3 {
    ///     x: Length::new::<meter>(2.0),
    ///     y: Length::new::<meter>(5.0),
    ///     z: Length::new::<meter>(11.0),
    /// };
    /// let b = Vector3 {
    ///     x: Length::new::<meter>(3.0),
    ///     y: Length::new::<meter>(7.0),
    ///     z: Length::new::<meter>(13.0),
    /// };
    /// // Length * Length results in Area
    /// let area_vector: Vector3<Area> = a.cross_uom(b);
    /// assert_eq!(area_vector.x, Area::new::<square_meter>(-12.0));
    /// assert_eq!(area_vector.y, Area::new::<square_meter>(7.0));
    /// assert_eq!(area_vector.z, Area::new::<square_meter>(-1.0));
    /// ```
    #[inline]
    pub fn cross_uom<Rhs, Out>(self, rhs: Vector3<Rhs>) -> Vector3<Out>
    where
        T: Copy + Mul<Rhs, Output = Out>,
        Rhs: Copy,
        Out: Sub<Output = Out>,
    {
        Vector3 {
            x: (self.y * rhs.z) - (self.z * rhs.y),
            y: (self.z * rhs.x) - (self.x * rhs.z),
            z: (self.x * rhs.y) - (self.y * rhs.x),
        }
    }
}

// **** norm_squared ****

impl<T> Vector3<T>
where
    T: Copy + Neg<Output = T> + Add<Output = T> + Mul<T, Output = T>,
{
    /// Return square of Euclidean norm.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    /// assert_eq!(150.0, v.norm_squared());
    /// ```
    #[inline]
    pub fn norm_squared(self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Return distance between two points, squared.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    /// let w = Vector3f32::new(3.0, 7.0, 17.0);
    /// assert_eq!(41.0, v.distance_squared(w));
    /// ```
    #[inline]
    pub fn distance_squared(self, other: Self) -> T {
        (self - other).norm_squared()
    }
}

#[cfg(feature = "uom")]
impl<D, U, V> Vector3<uom::si::Quantity<D, U, V>>
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
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Calculates the Euclidean norm (length) of the vector.
    ///
    /// # Example
    /// ```
    /// # use uom::si::f32::{Length, Area};
    /// # use uom::si::{length::meter,area::square_meter};
    /// # use vqm::Vector3;
    ///
    /// let v = Vector3 {
    ///     x: Length::new::<meter>(3.0),
    ///     y: Length::new::<meter>(4.0),
    ///     z: Length::new::<meter>(12.0),
    /// };
    ///
    /// let norm: Length = v.norm_uom();
    /// assert_eq!(norm, Length::new::<meter>(13.0));
    /// ```
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
        let norm = (x * x + y * y + z * z).sqrt();

        uom::si::Quantity { dimension: PhantomData, units: PhantomData, value: norm }
    }

    #[inline]
    /// Normalizes the vector, returning a unit vector pointing in the same direction.
    ///
    /// # Example
    /// ```
    /// # use uom::si::f32::{Length, Ratio};
    /// # use uom::si::{length::meter,ratio::ratio};
    /// # use vqm::Vector3;
    ///
    /// let v = Vector3 {
    ///     x: Length::new::<meter>(3.0),
    ///     y: Length::new::<meter>(4.0),
    ///     z: Length::new::<meter>(12.0),
    /// };
    ///
    /// let unit_vector: Vector3<Ratio> = v.normalize_uom();
    ///
    /// assert!((unit_vector.x - Ratio::new::<ratio>(3.0 / 13.0)).value.abs() < 1e-7);
    /// assert!((unit_vector.y - Ratio::new::<ratio>(4.0 / 13.0)).value.abs() < 1e-7);
    /// assert!((unit_vector.z - Ratio::new::<ratio>(12.0 / 13.0)).value.abs() < 1e-7);
    /// ```
    pub fn normalize_uom<Intermediate>(self) -> Vector3<uom::si::ratio::Ratio<U, V>>
    where
        uom::si::Quantity<D, U, V>: Mul<uom::si::Quantity<D, U, V>, Output = Intermediate>,
        Intermediate: Add<Output = Intermediate>,
    {
        let x = self.x.value;
        let y = self.y.value;
        let z = self.z.value;
        let norm = (x * x + y * y + z * z).sqrt();
        let norm_reciprocal = V::one() / norm;

        Vector3 {
            x: uom::si::Quantity { dimension: PhantomData, units: PhantomData, value: x * norm_reciprocal },
            y: uom::si::Quantity { dimension: PhantomData, units: PhantomData, value: y * norm_reciprocal },
            z: uom::si::Quantity { dimension: PhantomData, units: PhantomData, value: z * norm_reciprocal },
        }
    }
}

// **** norm ****

impl<T> Vector3<T>
where
    T: Copy + Neg<Output = T> + Add<Output = T> + Mul<T, Output = T> + MathMethods,
{
    /// Return Euclidean norm.
    #[inline]
    pub fn norm(self) -> T {
        Self::norm_squared(self).sqrt()
    }
}

impl<T> Vector3<T>
where
    T: Copy + Zero + Neg<Output = T> + Add<Output = T> + Mul<T, Output = T> + PartialEq + MathMethods,
{
    /// Return normalized form of the vector, checking if the norm is zero.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(0.0, 0.0, 0.0);
    /// let n = v.normalize();
    /// assert_eq!(Vector3f32 { x: 0.0, y: 0.0, z: 0.0 }, n);
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
    /// # use vqm::Vector3f32;
    /// let mut v = Vector3f32::new(0.0, 0.0, 0.0);
    /// v.normalize_in_place();
    /// assert_eq!(Vector3f32 { x: 0.0, y: 0.0, z: 0.0 }, v);
    /// ```
    #[inline]
    pub fn normalize_in_place(&mut self) -> &mut Self {
        *self = self.normalize();
        self
    }

    /// Return normalized form of the vector, not checking if the norm is zero.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(1.0, 4.0, 8.0);
    /// let n = v.normalize_unchecked();
    /// assert!((Vector3f32 { x: 0.11111111, y: 0.44444445, z: 0.8888889 } - n).is_near_zero(4.0e-4));
    /// ```
    #[inline]
    #[must_use]
    pub fn normalize_unchecked(self) -> Self {
        let norm_squared = self.norm_squared();
        self * norm_squared.sqrt_reciprocal()
    }

    /// Normalize the vector in place, not checking if the norm is zero.
    /// ```
    /// # use vqm::Vector3f32;
    /// let mut v = Vector3f32::new(1.0, 4.0, 8.0);
    /// v.normalize_unchecked_in_place();
    /// assert!((Vector3f32 { x: 0.11111111, y: 0.44444445, z: 0.8888889 } - v).is_near_zero(4.0e-4));
    /// ```
    #[inline]
    pub fn normalize_unchecked_in_place(&mut self) -> &mut Self {
        *self = self.normalize_unchecked();
        self
    }
}

impl<T> Vector3<T>
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

impl<T> Vector3<T>
where
    T: Copy + FloatCore,
{
    /// Convert the vector to degrees, assuming it is in radians.
    /// ```
    /// # use vqm::Vector3f32;
    /// # use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_6};
    /// let v = Vector3f32::new(FRAC_PI_2, FRAC_PI_4, FRAC_PI_6);
    /// let w = v.to_degrees();
    /// assert!((w.x - 90.0).abs() < 2e-6);
    /// assert!((w.y - 45.0).abs() < 2e-6);
    /// assert!((w.z - 30.0).abs() < 2e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_degrees(self) -> Self {
        Self { x: self.x.to_degrees(), y: self.y.to_degrees(), z: self.z.to_degrees() }
    }

    /// Convert the vector to radians, assuming it is in degrees.
    /// ```
    /// # use vqm::Vector3f32;
    /// # use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_6};
    /// let v = Vector3f32::new(90.0, 45.0, 30.0);
    /// assert_eq!(Vector3f32::new(FRAC_PI_2, FRAC_PI_4, FRAC_PI_6), v.to_radians());
    /// ```
    #[inline]
    #[must_use]
    pub fn to_radians(self) -> Self {
        Self { x: self.x.to_radians(), y: self.y.to_radians(), z: self.z.to_radians() }
    }
}

impl<T> Vector3<T>
where
    T: Copy + Mul<Output = T> + MathConstants,
{
    /// Convert the vector to meters per second squared, assuming it is in earth gravity units.
    /// ```
    /// # use vqm::{Vector3f32, MathConstants};
    /// let v = Vector3f32::new(1.0, 2.0, 3.0);
    /// assert_eq!(Vector3f32::new(9.806_65, 19.613_3, 29.419_95), v.g_to_mps2());
    /// ```
    #[inline]
    #[must_use]
    pub fn g_to_mps2(self) -> Self {
        Self { x: self.x * T::G0, y: self.y * T::G0, z: self.z * T::G0 }
    }

    /// Convert the vector to earth gravity units, assuming it is in meters per second squared.
    /// ```
    /// # use vqm::{Vector3f32, MathConstants};
    /// let v = Vector3f32::new(9.806_65, 19.613_3, 29.419_95);
    /// assert_eq!(Vector3f32::new(1.0, 2.0, 3.0), v.mps2_to_g());
    /// ```
    #[inline]
    #[must_use]
    pub fn mps2_to_g(self) -> Self {
        Self { x: self.x * T::G0_RECIPROCAL, y: self.y * T::G0_RECIPROCAL, z: self.z * T::G0_RECIPROCAL }
    }
}

// **** sum ****

impl<T> Vector3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    /// Return the sum of all components of the vector.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    /// assert_eq!(18.0, v.sum());
    /// ```
    #[inline]
    pub fn sum(self) -> T {
        self.x + self.y + self.z
    }

    /// Return the product of all components of the vector.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    /// assert_eq!(110.0, v.product());
    /// ```
    #[inline]
    pub fn product(self) -> T {
        self.x * self.y * self.z
    }
}

// **** mean ****

impl<T> Vector3<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the vector.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 5.0, 11.0);
    /// assert_eq!(6.0, v.mean());
    /// ```
    #[inline]
    pub fn mean(self) -> T {
        let three = T::one() + T::one() + T::one();
        self.sum() / three
    }
}

// **** max ****

impl<T> Vector3<T>
where
    T: Copy + Vector3Math,
{
    /// Return the max element in the vector.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 3.0, 5.0);
    /// let w = Vector3f32::new(3.0, 5.0, 2.0);
    /// let x = Vector3f32::new(5.0, 3.0, 2.0);
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
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::new(2.0, 3.0, 5.0);
    /// let w = Vector3f32::new(3.0, 5.0, 2.0);
    /// let x = Vector3f32::new(5.0, 3.0, 2.0);
    /// assert_eq!(2.0, v.min());
    /// assert_eq!(2.0, w.min());
    /// assert_eq!(2.0, x.min());
    /// ```
    #[inline]
    pub fn min(self) -> T {
        T::v3_min(self)
    }
}

impl<T> Vector3<T>
where
    T: Copy + Zero + One + Sub<Output = T> + Vector3Math + MathMethods + QuaternionMath,
{
    #[inline]
    #[must_use]
    /// Rotate vector using quaternion.
    pub fn rotate_by(self, q: Quaternion<T>) -> Self {
        // Extract the vector part of the quaternion (x, y, z)
        let q_xyz = Vector3 { x: q.x, y: q.y, z: q.z };

        // 1. uv = 2 * (q_xyz cross v)
        let uv = q_xyz.cross(self) * (T::one() + T::one());

        // 2. res = v + w * uv + (q_xyz cross t)
        // This is the optimized Rodrigues form
        self + (uv * q.w) + q_xyz.cross(uv)
    }
    #[inline]
    #[must_use]
    /// Rotate vector using quaternion conjugate.
    pub fn rotate_back_by(self, q: Quaternion<T>) -> Self {
        // Rotating 'back' is just rotating by the inverse (conjugate)
        self.rotate_by(q.conjugate())
    }
}

// **** From ****

// **** From Tuple ****

impl<T> From<(T, T, T)> for Vector3<T> {
    /// Vector from tuple.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::from((2.0, 5.0, 11.0));
    /// let w: Vector3f32 = (3.0, 7.0, 13.0).into();
    ///
    /// assert_eq!(v, Vector3f32 { x: 2.0, y: 5.0, z: 11.0 });
    /// assert_eq!(w, Vector3f32 { x: 3.0, y: 7.0, z: 13.0 });
    /// ```
    #[inline]
    fn from((x, y, z): (T, T, T)) -> Self {
        Self { x, y, z }
    }
}

// **** From Array ****

impl<T: Copy> From<[T; 3]> for Vector3<T> {
    /// Vector from array.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32::from([2.0, 5.0, 11.0]);
    /// let w: Vector3f32 = [3.0, 7.0, 13.0].into();
    ///
    /// assert_eq!(v, Vector3f32 { x: 2.0, y: 5.0, z: 11.0 });
    /// assert_eq!(w, Vector3f32 { x: 3.0, y: 7.0, z: 13.0 });
    /// ```
    #[inline]
    fn from(v: [T; 3]) -> Self {
        Self { x: v[0], y: v[1], z: v[2] }
    }
}

impl<T> From<Vector3<T>> for [T; 3] {
    /// Array from vector.
    /// ```
    /// # use vqm::Vector3f32;
    /// let v = Vector3f32 { x: 2.0, y: 5.0, z: 11.0 };
    ///
    /// let a = <[f32; 3]>::from(v);
    /// let b: [f32; 3] = v.into();
    ///
    /// assert_eq!(a, [2.0, 5.0, 11.0]);
    /// assert_eq!(b, [2.0, 5.0, 11.0]);
    /// ```
    #[inline]
    fn from(v: Vector3<T>) -> Self {
        [v.x, v.y, v.z]
    }
}

impl From<[i16; 3]> for Vector3<f32> {
    #[inline]
    fn from(v: [i16; 3]) -> Self {
        Self { x: f32::from(v[0]), y: f32::from(v[1]), z: f32::from(v[2]) }
    }
}

impl From<Vector3<f32>> for [i16; 3] {
    #[inline]
    fn from(v: Vector3<f32>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        [v.x as i16, v.y as i16, v.z as i16]
    }
}

impl From<[i32; 3]> for Vector3<f32> {
    #[inline]
    fn from(v: [i32; 3]) -> Self {
        #[allow(clippy::cast_precision_loss)]
        Self { x: v[0] as f32, y: v[1] as f32, z: v[2] as f32 }
    }
}

impl From<Vector3<f32>> for [i32; 3] {
    #[inline]
    fn from(v: Vector3<f32>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        [v.x as i32, v.y as i32, v.z as i32]
    }
}

// **** From Vector ****

impl<T: Zero> From<Vector2<T>> for Vector3<T> {
    /// Vector3 from Vector2.
    /// ```
    /// # use vqm::{Vector2f32,Vector3f32};
    /// let v = Vector3f32::from(Vector2f32 { x: 2.0, y: 5.0 });
    /// let w: Vector3f32 = Vector2f32 { x: 3.0, y: 7.0 }.into();
    ///
    /// assert_eq!(v, Vector3f32 { x: 2.0, y: 5.0, z: 0.0 });
    /// assert_eq!(w, Vector3f32 { x: 3.0, y: 7.0, z: 0.0 });
    /// ```
    #[inline]
    fn from(other: Vector2<T>) -> Self {
        Self { x: other.x, y: other.y, z: T::zero() }
    }
}

impl<T> From<Vector3<T>> for Vector2<T> {
    /// Vector2 from Vector3, discarding z value.
    /// ```
    /// # use vqm::{Vector2f32,Vector3f32};
    /// let v: Vector2f32 = Vector3f32 { x: 2.0, y: 5.0, z: 11.0 }.into();
    /// let u = Vector2f32::from(Vector3f32{ x: 3.0, y: 7.0, z: 13.0 });
    ///
    /// assert_eq!(v, Vector2f32 { x: 2.0, y: 5.0 });
    /// assert_eq!(u, Vector2f32 { x: 3.0, y: 7.0 });
    /// ```
    #[inline]
    fn from(v: Vector3<T>) -> Self {
        Vector2::<T> { x: v.x, y: v.y }
    }
}
