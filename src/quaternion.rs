use core::convert::From;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{ConstOne, ConstZero};
use num_traits::{MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};
#[cfg(feature = "storage")]
use sequential_storage::map::PostcardValue;
#[cfg(feature = "serde")]
use {
    postcard::experimental::max_size::MaxSize,
    serde::{Deserialize, Serialize},
};

use crate::math_methods::MathMethods;
use crate::{QuaternionMath, Vector3};

/// Quaternion of `f32` values<br>
pub type Quaternionf32 = Quaternion<f32>;
/// Quaternion of `f64` values<br><br>
pub type Quaternionf64 = Quaternion<f64>;

// **** Define ****

/// `Quaternion<T>`: quaternion type `T`.<br>
/// Implementations use the Hamilton convention.
/// Aliases `Quaternionf32` and `Quaternionf64` are provided.<br><br>
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(derive_more::Display))]
#[cfg_attr(feature = "std", display("Q{{w:{w}, x:{x}, y:{y}, z:{z}}}"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize, MaxSize))]
#[allow(missing_docs)]
#[repr(C, align(16))]
pub struct Quaternion<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

#[cfg(feature = "storage")]
impl<T> PostcardValue<'_> for Quaternion<T> where T: Serialize + for<'de> Deserialize<'de> {}

// **** Default ****

/// Default quaternion.
/// ```
/// # use vqm::Quaternionf32;
/// # use num_traits::Zero;
///
/// let d = Quaternionf32::default();
///
/// assert_eq!(d, Quaternionf32 { w:1.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl<T> Default for Quaternion<T>
where
    T: Copy + Zero + One,
{
    #[inline]
    fn default() -> Self {
        Self { w: T::one(), x: T::zero(), y: T::zero(), z: T::zero() }
    }
}

// **** New ****

impl<T> Quaternion<T>
where
    T: Copy,
{
    /// Create a quaternion.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0,  3.0, 7.0, 11.0);
    /// assert_eq!(q, Quaternionf32 { w:2.0, x:3.0, y: 7.0, z: 11.0 });
    /// ```
    #[inline]
    pub const fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }
}

// **** Zero ****

impl<T> Zero for Quaternion<T>
where
    T: Copy + ConstZero + PartialEq + QuaternionMath,
{
    /// Zero quaternion.
    /// ```
    /// # use vqm::Quaternionf32;
    /// # use num_traits::{zero,Zero};
    /// let z = Quaternionf32::zero();
    /// assert!(z.is_zero());
    /// assert_eq!(z, Quaternionf32 { w:0.0, x: 0.0, y: 0.0, z: 0.0 });
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

/// Const zero quaternion.
/// ```
/// # use vqm::Quaternionf32;
/// # use num_traits::{Zero,ConstZero};
/// let z = Quaternionf32::ZERO;
/// assert!(z.is_zero());
/// assert_eq!(z, Quaternionf32 { w:0.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl<T> ConstZero for Quaternion<T>
where
    T: Copy + ConstZero + PartialEq + QuaternionMath,
{
    const ZERO: Self = Self { w: T::ZERO, x: T::ZERO, y: T::ZERO, z: T::ZERO };
}

impl<T> Quaternion<T>
where
    T: Copy + FloatCore,
{
    /// Return true if vector is near zero.
    /// ```
    /// # use vqm::Quaternionf32;
    /// # use num_traits::Zero;
    /// let z = Quaternionf32::zero();
    /// assert!(z.is_near_zero(1e-5));
    /// ```
    pub fn is_near_zero(self, epsilon: T) -> bool {
        if self.w.abs() > epsilon && self.x.abs() > epsilon && self.y.abs() > epsilon && self.z.abs() > epsilon {
            return false;
        }
        true
    }
}

// **** One ****

impl<T> One for Quaternion<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Sub<Output = T> + Mul<Output = T> + QuaternionMath,
{
    /// Unit quaternion.
    /// ```
    /// # use vqm::Quaternionf32;
    /// # use num_traits::One;
    ///
    /// let i = Quaternionf32::one();
    ///
    /// assert_eq!(i, Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
    /// ```
    #[inline]
    fn one() -> Self {
        Self::ONE
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

/// Const unit quaternion.
/// ```
/// # use vqm::Quaternionf32;
/// # use num_traits::ConstOne;
///
/// let i = Quaternionf32::ONE;
///
/// assert_eq!(i, Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl<T> ConstOne for Quaternion<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Sub<Output = T> + QuaternionMath,
{
    const ONE: Self = Self { w: T::ONE, x: T::ZERO, y: T::ZERO, z: T::ZERO };
}

impl<T> Quaternion<T>
where
    T: Copy + Zero + One,
{
    /// Unit quaternion.
    /// Alias for `one()` that does not require `num_traits::One`.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let i = Quaternionf32::identity();
    ///
    /// assert_eq!(i, Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
    /// ```
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        Self { w: T::one(), x: T::zero(), y: T::zero(), z: T::zero() }
    }
}
// **** Neg ****

impl<T> Neg for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    /// Negate quaternion.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut q = Quaternionf32 { w: 2.0, x: -3.0, y: -5.0, z: 7.0 };
    /// q = -q;
    ///
    /// assert_eq!(q, Quaternionf32 { w: -2.0, x: 3.0, y: 5.0, z: -7.0 });
    /// ```
    #[inline]
    fn neg(self) -> Self {
        T::q_neg(self)
    }
}

// **** Add ****

impl<T> Add for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    /// Add two quaternions.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let u = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let v = Quaternionf32::new(11.0, 13.0, 17.0, 19.0);
    /// let r = u + v;
    ///
    /// assert_eq!(r, Quaternionf32 { w: 13.0, x: 16.0, y: 22.0, z: 26.0 });
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        T::q_add(self, other)
    }
}

// **** AddAssign ****

impl<T> AddAssign for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    /// Add one quaternion to another.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut r = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Quaternionf32::new(11.0, 13.0, 17.0, 19.0);
    /// r += w;
    ///
    /// assert_eq!(r, Quaternionf32 { w: 13.0, x: 16.0, y: 22.0, z: 26.0 });
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        // This mutates 'self' in place.
        // On RP2350, this avoids a stack copy of the current orientation.
        *self = *self + other;
    }
}

// **** MulAdd ****

impl<T> MulAdd<T> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    /// Multiply quaternion by constant and add another quaternion.
    /// ```
    /// # use vqm::Quaternionf32;
    /// # use num_traits::MulAdd;
    /// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Quaternionf32::new(11.0, 13.0, 17.0, 19.0);
    /// let k = 23.0;
    /// let r = q.mul_add(k, w);
    ///
    /// assert_eq!(r, Quaternionf32 { w: 57.0, x: 82.0, y: 132.0, z: 180.0 });
    /// ```
    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::q_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

impl<T> MulAddAssign<T> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    /// Multiply quaternion by constant and add another quaternion in place.
    /// ```
    /// # use vqm::Quaternionf32;
    /// # use num_traits::MulAddAssign;
    /// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Quaternionf32::new(11.0, 13.0, 17.0, 19.0);
    /// let k = 23.0;
    /// q.mul_add_assign(k, w);
    ///
    /// assert_eq!(q, Quaternionf32 { w: 57.0, x: 82.0, y: 132.0, z: 180.0 });
    /// ```
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

impl<T> Sub for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    /// Subtract two quaternions.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Quaternionf32::new(11.0, 13.0, 17.0, 19.0);
    /// let r = q - w;
    ///
    /// assert_eq!(r, Quaternionf32 { w: -9.0, x: -10.0, y: -12.0, z: -12.0 });
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

impl<T> SubAssign for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    /// Subtract one quaternion from another.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut r = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Quaternionf32::new(11.0, 13.0, 17.0, 19.0);
    /// r -= w;
    ///
    /// assert_eq!(r, Quaternionf32 { w: -9.0, x: -10.0, y: -12.0, z: -12.0 });
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Scalar Mul ****

impl Mul<Quaternion<f32>> for f32 {
    type Output = Quaternion<f32>;

    /// Pre-multiply quaternion by a scalar.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let r = 2.0 * q;
    ///
    /// assert_eq!(r, Quaternionf32 { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
    /// ```
    #[inline]
    fn mul(self, other: Quaternion<f32>) -> Quaternion<f32> {
        f32::q_mul_scalar(other, self)
    }
}

impl Mul<Quaternion<f64>> for f64 {
    type Output = Quaternion<f64>;
    #[inline]
    fn mul(self, other: Quaternion<f64>) -> Quaternion<f64> {
        f64::q_mul_scalar(other, self)
    }
}

// **** Mul Scalar ****

impl<T> Mul<T> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    /// Multiply quaternion by a scalar.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let r = q * 2.0;
    ///
    /// assert_eq!(r, Quaternionf32 { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
    /// ```
    #[inline]
    fn mul(self, k: T) -> Self {
        T::q_mul_scalar(self, k)
    }
}

// **** MulAssign ****

impl<T> MulAssign<T> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    /// In-place multiply a quaternion by a scalar.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let r = q * 2.0;
    ///
    /// assert_eq!(r, Quaternionf32 { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
    /// ```
    #[inline]
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

// **** Div by scalar ****

impl<T> Div<T> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    /// Divide a quaternion by a scalar.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let r = q / 2.0;
    ///
    /// assert_eq!(r, Quaternionf32 { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
    /// ```
    #[inline]
    fn div(self, k: T) -> Self {
        T::q_div_scalar(self, k)
    }
}

impl<T> DivAssign<T> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    /// In-place divide a quaternion by a scalar.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// q /= 2.0;
    ///
    /// assert_eq!(q, Quaternionf32 { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
    /// ```
    #[inline]
    fn div_assign(&mut self, k: T) {
        *self = self.div(k);
    }
}

// **** Mul ****

impl<T> Mul<Quaternion<T>> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    /// Multiply two quaternions.
    #[inline]
    fn mul(self, other: Self) -> Self {
        T::q_mul(self, other)
    }
}

// **** MulAssign ****

impl<T> MulAssign<Quaternion<T>> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    /// Multiply one quaternion by another.
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = self.mul(other);
    }
}

// **** Index ****

impl<T> Index<usize> for Quaternion<T> {
    type Output = T;

    /// Access quaternion component by index.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    ///
    /// assert_eq!(q[0], 2.0);
    /// assert_eq!(q[1], 3.0);
    /// assert_eq!(q[2], 5.0);
    /// assert_eq!(q[3], 7.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.w,
            1 => &self.x,
            2 => &self.y,
            _ => &self.z,
        }
    }
}
// **** IndexMut ****

impl<T> IndexMut<usize> for Quaternion<T> {
    // Set quaternion component by index.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 6.0);
    /// q[0] = 7.0;
    /// q[1] = 11.0;
    /// q[2] = 13.0;
    /// q[3] = 17.0;
    ///
    /// assert_eq!(q, Quaternionf32 { w:7.0, x:11.0, y:13.0, z: 17.0 });
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.w,
            1 => &mut self.x,
            2 => &mut self.y,
            _ => &mut self.z,
        }
    }
}

// **** lerp ****

impl<T> Quaternion<T>
where
    T: Copy,
    Quaternion<T>: Mul<T, Output = Quaternion<T>> + Add<Output = Quaternion<T>> + Sub<Output = Quaternion<T>>,
{
    /// Linear interpolation between two quaternions.
    /// Calculates `self * (1 - t) + other * t`.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 5.0, 11.0, 13.0);
    /// let r = Quaternionf32::new(3.0, 7.0, 17.0, 23.0);
    /// let s = q.lerp(r, 0.25);
    ///
    /// assert_eq!(s, Quaternionf32::new(2.25, 5.5, 12.5, 15.5));
    /// ```
    #[inline]
    #[must_use]
    pub fn lerp(self, other: Self, t: T) -> Self {
        self + (other - self) * t
    }
}

// **** approx_eq ****

impl<T> Quaternion<T>
where
    T: FloatCore,
{
    /// Compare two quaternions with a tolerance.
    #[inline]
    pub fn approx_eq(&self, other: &Self, epsilon: T) -> bool
    where
        T: FloatCore,
    {
        (self.w - other.w).abs() <= epsilon
            && (self.x - other.x).abs() <= epsilon
            && (self.y - other.y).abs() <= epsilon
            && (self.z - other.z).abs() <= epsilon
    }
}

// **** abs ****

impl<T> Quaternion<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the quaternion with all components set to their absolute values.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, -3.0, -5.0, 7.0);
    /// let r = q.abs();
    ///
    /// assert_eq!(r, Quaternionf32::new(2.0, 3.0, 5.0, 7.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self { w: self.w.abs(), x: self.x.abs(), y: self.y.abs(), z: self.z.abs() }
    }

    /// Set all components of the quaternion to their absolute values.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut q = Quaternionf32::new(2.0, -3.0, -5.0, 7.0);
    /// q.abs_in_place();
    ///
    /// assert_eq!(q, Quaternionf32::new(2.0, 3.0, 5.0, 7.0));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = self.abs();
        self
    }
}

// **** clamp ****

impl<T> Quaternion<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the quaternion with all components clamped to the specified range.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 3.0, 7.0, 11.0);
    /// let r = q.clamp(2.5, 7.5);
    ///
    /// assert_eq!(r, Quaternionf32::new(2.5, 3.0, 7.0, 7.5));
    /// ```
    #[inline]
    #[must_use]
    pub fn clamp(self, min: T, max: T) -> Self {
        Self {
            w: self.w.clamp(min, max),
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
        }
    }

    /// Clamp all components of the quaternion to the specified range.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut q = Quaternionf32::new(2.0, 3.0, 7.0, 11.0);
    /// q.clamp_in_place(2.5, 7.5);
    ///
    /// assert_eq!(q, Quaternionf32::new(2.5, 3.0, 7.0, 7.5));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

// **** dot ****

impl<T> Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    /// Quaternion dot product.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let v = Quaternionf32::new(2.0, 5.0, 11.0, 17.0);
    /// let w = Quaternionf32::new(3.0, 7.0, 13.0, 19.0);
    ///
    /// let x = v.dot(w);
    ///
    /// assert_eq!(x, 507.0);
    /// ```
    #[inline]
    pub fn dot(self, other: Self) -> T {
        T::q_dot(self, other)
    }
}

// **** norm_squared ****

impl<T> Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    /// Return square of Euclidean norm.
    #[inline]
    pub fn norm_squared(self) -> T {
        T::q_norm_squared(self)
    }
}

// **** norm ****

impl<T> Quaternion<T>
where
    T: Copy + MathMethods + QuaternionMath,
{
    /// Return Euclidean norm.
    #[inline]
    pub fn norm(self) -> T {
        Self::norm_squared(self).sqrt()
    }
}

impl<T> Quaternion<T>
where
    T: Copy + Zero + PartialEq + MathMethods + QuaternionMath,
{
    /// Return normalized form of the quaternion, checking if the magnitude is zero.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(0.0, 0.0, 0.0, 0.0);
    /// let r = q.normalize();
    /// assert_eq!(Quaternionf32 { w: 0.0, x: 0.0, y: 0.0, z: 0.0 }, r);
    /// ```
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        let norm_squared = self.norm_squared();
        // If norm == 0.0 then the quaternion is already normalized
        if norm_squared == T::zero() {
            return self;
        }
        self * norm_squared.sqrt_reciprocal()
    }

    /// Normalize the quaternion in place, checking if the magnitude is zero.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut q = Quaternionf32::new(0.0, 0.0, 0.0, 0.0);
    /// q.normalize_in_place();
    /// assert_eq!(Quaternionf32 { w: 0.0, x: 0.0, y: 0.0, z: 0.0 }, q);
    /// ```
    #[inline]
    pub fn normalize_in_place(&mut self) -> &mut Self {
        *self = self.normalize();
        self
    }

    /// Return normalized form of the quaternion, not checking if the magnitude is zero.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let r = q.normalized_unchecked();
    /// assert!((Quaternionf32 { w: 0.21442251, x: 0.32163376, y: 0.5360563, z: 0.7504788 } - r).is_near_zero(1.0e-4));
    /// ```
    #[inline]
    #[must_use]
    pub fn normalized_unchecked(self) -> Self {
        let norm_squared = self.norm_squared();
        self * norm_squared.sqrt_reciprocal()
    }

    /// Normalize the quaternion in place, not checking if the magnitude is zero.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// q.normalize_unchecked_in_place();
    /// assert!((Quaternionf32 { w: 0.21442251, x: 0.32163376, y: 0.5360563, z: 0.7504788 } - q).is_near_zero(1.0e-4));
    /// ```
    #[inline]
    pub fn normalize_unchecked_in_place(&mut self) -> &mut Self {
        *self = self.normalized_unchecked();
        self
    }
}

impl<T> Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    // Return true if the quaternion is normalized.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let n = q.normalize();
    /// let s = n.norm_squared();
    /// assert!((1.0 - s).abs() < 8.0e-4);
    /// assert!(n.is_normalized(8.0e-4));
    /// ```
    #[inline]
    pub fn is_normalized(self, epsilon: T) -> bool {
        T::q_is_normalized(self, epsilon)
    }
}

impl<T> Quaternion<T>
where
    T: Copy + One + Zero + QuaternionMath + Div<T, Output = T> + Neg<Output = T>,
{
    // Return the inverse of the quaternion.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let r = q.inverse();
    /// let p = q * r;
    /// assert_eq!(1.0, p.w);
    /// assert!(p.x.abs() < 1e-7);
    /// assert!(p.y.abs() < 1e-7);
    /// assert!(p.z.abs() < 1e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        let n = self.norm_squared();
        if n.is_zero() {
            Self { w: T::one(), x: T::zero(), y: T::zero(), z: T::zero() }
        } else {
            let r = T::one() / n;
            Self { w: self.w * r, x: -self.x * r, y: -self.y * r, z: -self.z * r }
        }
    }
    /// Inverts this quaternion if it is not zero.
    /// # Example
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(1.0, 2.0, 3.0, 4.0);
    /// let q_inv = q.try_inverse().unwrap_or_default();
    /// let r = q * q_inv;
    ///
    /// assert_eq!(1.0, r.w);
    /// assert!(r.x.abs() < 1e-7);
    /// assert!(r.y.abs() < 1e-7);
    /// assert!(r.z.abs() < 1e-7);
    ///
    /// let z = Quaternionf32::new(0.0, 0.0, 0.0, 0.0);
    /// let z_inv = z.try_inverse();
    ///
    /// assert!(z_inv.is_none());
    /// ```
    #[inline]
    #[must_use]
    pub fn try_inverse(&self) -> Option<Self> {
        let n = self.norm_squared();
        if n.is_zero() {
            None
        } else {
            let r = T::one() / n;
            Some(Self { w: self.w * r, x: -self.x * r, y: -self.y * r, z: -self.z * r })
        }
    }
}

impl<T> Quaternion<T>
where
    T: Copy + ConstOne + Add<Output = T> + Div<Output = T>,
{
    // Return the half of the quaternion.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// let r = q.half();
    /// assert_eq!(Quaternionf32 { w: 1.0, x: 1.5, y: 2.5, z: 3.5 }, r);
    /// ```
    #[inline]
    #[must_use]
    pub fn half(self) -> Self {
        let r = T::ONE / (T::ONE + T::ONE);
        Self { w: self.w * r, x: self.x * r, y: self.y * r, z: self.z * r }
    }
}

// **** sum ****

impl<T> Quaternion<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    /// Return the sum of all components of the quaternion.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// assert_eq!(17.0, q.sum());
    /// ```
    #[inline]
    pub fn sum(self) -> T {
        self.w + self.x + self.y + self.z
    }

    /// Return the product of all components of the quaternion.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
    /// assert_eq!(210.0, q.product());
    /// ```
    #[inline]
    pub fn product(self) -> T {
        self.w * self.x * self.y * self.z
    }
}

// **** mean ****

impl<T> Quaternion<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the quaternion.
    #[inline]
    pub fn mean(self) -> T {
        let four = T::one() + T::one() + T::one() + T::one();
        (self.w + self.x + self.y + self.z) / four
    }
}

// **** rotate ***

impl<T> Quaternion<T>
where
    T: Copy + FloatCore + MathMethods,
{
    pub fn rotate(self, v: &Vector3<T>) -> Vector3<T> {
        let two: T = T::one() + T::one();
        let half = T::one() / two;

        let x2: T = self.x * self.x;
        let y2: T = self.y * self.y;
        let z2: T = self.z * self.z;

        Vector3::<T> {
            x: (v.x * (half - y2 - z2)
                + v.y * (self.x * self.y - self.w * self.z)
                + v.z * (self.w * self.y + self.x * self.z))
                * two,
            y: (v.x * (self.w * self.z + self.x * self.y)
                + v.y * (half - x2 - z2)
                + v.z * (self.y * self.z - self.w * self.x))
                * two,
            z: (v.x * (self.w * self.y + self.x * self.z)
                + v.y * (self.w * self.x + self.y * self.z)
                + v.z * (half - x2 - y2))
                * two,
        }
    }
    pub fn cos_roll(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.w * self.x + self.y * self.z;
        let b: T = half - self.x * self.x - self.y * self.y;
        b * (a * a + b * b).sqrt_reciprocal()
    }

    pub fn sin_pitch(self) -> T {
        let two: T = T::one() + T::one();
        (self.w * self.y - self.x * self.z) * two
    }

    pub fn cos_pitch(self) -> T {
        let s: T = self.sin_pitch();
        let sq = T::one() - s * s;
        if sq < T::zero() { T::zero() } else { sq.sqrt() }
    }

    pub fn tan_pitch(self) -> T {
        let s: T = self.sin_pitch();
        let sq = T::one() - s * s;
        if sq < T::zero() { T::zero() } else { s * sq.sqrt_reciprocal() }
    }

    pub fn cos_yaw(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.w * self.z + self.x * self.y;
        let b: T = half - self.y * self.y - self.z * self.z;
        b * (a * a + b * b).sqrt_reciprocal()
    }

    pub fn sin_yaw(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.w * self.z + self.x * self.y;
        let b: T = half - self.y * self.y - self.z * self.z;
        a * (a * a + b * b).sqrt_reciprocal()
    }

    pub fn sin_roll(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.w * self.x + self.y * self.z;
        let b: T = half - self.x * self.x - self.y * self.y;
        a * (a * a + b * b).sqrt_reciprocal()
    }

    /// cos of the total tilt angle (direct Z-axis projection from the rotation matrix).
    pub fn cos_tilt(self) -> T {
        T::one() - (T::one() + T::one()) * (self.x * self.x + self.y * self.y)
    }

    /// sin of the total tilt angle (direct Z-axis projection from the rotation matrix).
    pub fn sin_tilt(self) -> T {
        let c: T = self.cos_tilt();
        let sq = T::one() - c * c;
        if sq < T::zero() { T::zero() } else { sq.sqrt() }
    }
}

impl<T> Quaternion<T>
where
    T: Copy + FloatCore + MathMethods,
{
    /// clip `sin(roll_angle)` to +/-1.0 when roll angle outside range [-90 degrees, 90 degrees].
    pub fn sin_roll_clipped(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.w * self.x + self.y * self.z;
        let b: T = half - self.x * self.x - self.y * self.y;
        if b < T::zero() {
            if a < T::zero() {
                return -T::one();
            }
            return T::one();
        }
        a * (a * a + b * b).sqrt_reciprocal()
    }

    pub fn sin_pitch_clipped(self) -> T {
        let d = self.w * self.w - self.y * self.y;
        let half_sin_pitch = self.w * self.y - self.x * self.z;

        // if d < 0.0, then self is outside the range [-90, 90] degrees, so we return 1.0 or 1.0 according to the sign of sin_pitch
        if d < T::zero() {
            if half_sin_pitch < T::zero() {
                return -T::one();
            }
            return T::one();
        }
        // self is in the range [-90, 90] degrees, so we can just return sin(self)
        let two: T = T::one() + T::one();
        two * half_sin_pitch
    }
}

impl<T> Quaternion<T>
where
    T: Copy + FloatCore + MathMethods,
{
    /// Rotate about the x-axis,
    /// equivalent to *= Quaternion(cos(theta/2), sin(theta/2), 0, 0).
    pub fn rotate_x(&mut self, theta: T) -> &mut Self {
        let two = T::one() + T::one();
        let (sin, cos) = (theta / two).sin_cos();
        let wt: T = self.w * cos - self.x * sin;
        self.x = self.w * sin + self.x * cos;
        let yt: T = self.y * cos + self.z * sin;
        self.z = self.z * cos - self.y * sin;
        self.w = wt;
        self.y = yt;
        self
    }

    /// Rotate about the y-axis,
    /// equivalent to *= Quaternion(cos(theta/2), 0, sin(theta/2), 0).
    pub fn rotate_y(&mut self, theta: T) -> &mut Self {
        let two = T::one() + T::one();
        let (sin, cos) = (theta / two).sin_cos();
        let wt: T = self.w * cos - self.y * sin;
        let xt: T = self.x * cos - self.z * sin;
        self.y = self.w * sin + self.y * cos;
        self.z = self.x * sin - self.z * cos;
        self.w = wt;
        self.x = xt;
        self
    }

    /// Rotate about the z-axis,
    /// equivalent to *= Quaternion(cos(theta/2), 0, 0, sin(theta/2)).
    pub fn rotate_z(&mut self, theta: T) -> &mut Self {
        let two = T::one() + T::one();
        let (sin, cos) = (theta / two).sin_cos();
        let wt: T = self.w * cos - self.z * sin;
        let xt: T = self.x * cos - self.y * sin;
        self.y = self.x * sin + self.y * cos;
        self.z = self.z * cos - self.w * sin;
        self.w = wt;
        self.x = xt;
        self
    }

    #[inline]
    pub fn calculate_roll_radians(self) -> T {
        let half = T::one() / (T::one() + T::one());
        (self.w * self.x + self.y * self.z).atan2(half - self.x * self.x - self.y * self.y)
    }

    #[inline]
    pub fn calculate_pitch_radians(self) -> T {
        let two = T::one() + T::one();
        (two * (self.w * self.y - self.x * self.z)).asin()
    }

    #[inline]
    pub fn calculate_yaw_radians(self) -> T {
        let half = T::one() / (T::one() + T::one());
        (self.w * self.z + self.x * self.y).atan2(half - self.y * self.y - self.z * self.z)
    }

    /// Calculate a quaternion's Euler angles in radians.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let orientation = Quaternionf32::default();
    /// let (roll, pitch, yaw) = orientation.calculate_euler_angles_radians();
    /// ```
    #[inline]
    pub fn calculate_euler_angles_radians(self) -> (T, T, T) {
        (self.calculate_roll_radians(), self.calculate_pitch_radians(), self.calculate_yaw_radians())
    }

    /// Create a Quaternion from roll, pitch, and yaw Euler angles (in radians).
    /// See: <https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_(in_3-2-1_sequence)_to_quaternion_conversion>.
    pub fn from_roll_pitch_yaw_radians(roll_radians: T, pitch_radians: T, yaw_radians: T) -> Self {
        let half: T = T::one() / (T::one() + T::one());
        let (sin_half_roll, cos_half_roll) = (roll_radians * half).sin_cos();
        let (sin_half_pitch, cos_half_pitch) = (pitch_radians * half).sin_cos();
        let (sin_half_yaw, cos_half_yaw) = (yaw_radians * half).sin_cos();
        Self {
            w: cos_half_roll * cos_half_pitch * cos_half_yaw + sin_half_roll * sin_half_pitch * sin_half_yaw,
            x: sin_half_roll * cos_half_pitch * cos_half_yaw - cos_half_roll * sin_half_pitch * sin_half_yaw,
            y: cos_half_roll * sin_half_pitch * cos_half_yaw + sin_half_roll * cos_half_pitch * sin_half_yaw,
            z: cos_half_roll * cos_half_pitch * sin_half_yaw - sin_half_roll * sin_half_pitch * cos_half_yaw,
        }
    }

    /// Create a Quaternion from roll, pitch, and yaw Euler angles (in radians).
    /// See: <https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_(in_3-2-1_sequence)_to_quaternion_conversion>.
    pub fn from_roll_pitch_yaw_degrees(roll_degrees: T, pitch_degrees: T, yaw_degrees: T) -> Self {
        Self::from_roll_pitch_yaw_radians(
            roll_degrees.to_radians(),
            pitch_degrees.to_radians(),
            yaw_degrees.to_radians(),
        )
    }

    /// Create a Quaternion from roll and pitch Euler angles (in radians), assumes yaw angle is zero.
    pub fn from_roll_pitch_radians(roll_radians: T, pitch_radians: T) -> Self {
        let half: T = T::one() / (T::one() + T::one());
        let (sin_half_roll, cos_half_roll) = (roll_radians * half).sin_cos();
        let (sin_half_pitch, cos_half_pitch) = (pitch_radians * half).sin_cos();

        Self {
            w: cos_half_roll * cos_half_pitch,
            x: sin_half_roll * cos_half_pitch,
            y: cos_half_roll * sin_half_pitch,
            z: -sin_half_roll * sin_half_pitch,
        }
    }

    /// Create a Quaternion from roll Euler angle (in radians), assumes pitch and roll angles are zero.
    pub fn from_roll_radians(roll_radians: T) -> Self {
        let half: T = T::one() / (T::one() + T::one());
        let (sin_half_roll, cos_half_roll) = (roll_radians * half).sin_cos();

        Self { w: cos_half_roll, x: sin_half_roll, y: T::zero(), z: T::zero() }
    }

    /// Create a Quaternion from roll Euler angle (in degrees), assumes pitch and roll angles are zero.
    pub fn from_roll_degrees(roll_degrees: T) -> Self {
        Self::from_roll_radians(roll_degrees.to_radians())
    }

    /// Create a Quaternion from pitch Euler angle (in radians), assumes roll and yaw angles are zero.
    pub fn from_pitch_radians(pitch_radians: T) -> Self {
        let half: T = T::one() / (T::one() + T::one());
        let (sin_half_pitch, cos_half_pitch) = (pitch_radians * half).sin_cos();

        Self { w: cos_half_pitch, x: T::zero(), y: sin_half_pitch, z: T::zero() }
    }

    /// Create a Quaternion from pitch Euler angle (in degrees), assumes roll and yaw angles are zero.
    pub fn from_pitch_degrees(pitch_degrees: T) -> Self {
        Self::from_pitch_radians(pitch_degrees.to_radians())
    }

    /// Create a Quaternion from yaw Euler angle (in radians), assumes roll and pitch angles are zero.
    pub fn from_yaw_radians(yaw_radians: T) -> Self {
        let half: T = T::one() / (T::one() + T::one());
        let (sin_half_yaw, cos_half_yaw) = (yaw_radians * half).sin_cos();

        Self { w: cos_half_yaw, x: T::zero(), y: T::zero(), z: sin_half_yaw }
    }

    /// Create a Quaternion from yaw Euler angle (in degrees), assumes roll and pitch angles are zero.
    pub fn from_yaw_degrees(yaw_degrees: T) -> Self {
        Self::from_yaw_radians(yaw_degrees.to_radians())
    }
}

impl<T> Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    // Return the conjugate of the quaternion.
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self {
        T::q_conjugate(self)
    }
}

impl<T> Quaternion<T>
where
    T: Copy + One + Neg<Output = T> + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    /// Return the scalar part of the quaternion.
    #[inline]
    pub fn scalar(self) -> T {
        self.w
    }

    /// Return the imaginary part of the quaternion.
    #[inline]
    pub fn imaginary(self) -> Vector3<T> {
        Vector3::<T> { x: self.x, y: self.y, z: self.z }
    }

    /// Return the last column of the equivalent rotation matrix, but calculated more efficiently than a full conversion.
    #[inline]
    pub fn direction_cosine_matrix_z(self) -> Vector3<T> {
        let two = T::one() + T::one();
        Vector3::<T> {
            x: (self.w * self.y + self.x * self.z) * two,
            y: (self.y * self.z - self.w * self.x) * two,
            z: self.w * self.w,
        }
    }

    #[inline]
    pub fn gravity(self) -> Vector3<T> {
        let two = T::one() + T::one();
        Vector3::<T> {
            x: (self.x * self.z - self.w * self.y) * two,
            y: (self.w * self.x + self.y * self.z) * two,
            z: (self.w * self.w + self.z * self.z) * two - T::one(),
        }
    }

    #[inline]
    pub fn half_gravity(self) -> Vector3<T> {
        let half: T = T::one() / (T::one() + T::one());
        Vector3::<T> {
            x: self.x * self.z - self.w * self.y,
            y: self.w * self.x + self.y * self.z,
            z: self.w * self.w + self.z * self.z - half,
        }
    }
}

impl<T> Quaternion<T>
where
    T: Copy + MathMethods + FloatCore,
{
    #[inline]
    pub fn calculate_roll_degrees(self) -> T {
        self.calculate_roll_radians().to_degrees()
    }

    #[inline]
    pub fn calculate_pitch_degrees(self) -> T {
        self.calculate_pitch_radians().to_degrees()
    }

    #[inline]
    pub fn calculate_yaw_degrees(self) -> T {
        self.calculate_yaw_radians().to_degrees()
    }

    /// Calculate a quaternion's Euler angles in degrees.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let orientation = Quaternionf32::default();
    /// let (roll, pitch, yaw) = orientation.calculate_euler_angles_degrees();
    /// ```
    #[inline]
    pub fn calculate_euler_angles_degrees(self) -> (T, T, T) {
        (self.calculate_roll_degrees(), self.calculate_pitch_degrees(), self.calculate_yaw_degrees())
    }

    /// Create a quaternion from roll, pitch, and yaw Euler angles (in degrees).
    #[inline]
    pub fn from_roll_pitch_yaw_angles_degrees(roll_degrees: T, pitch_degrees: T, yaw_degrees: T) -> Self {
        Self::from_roll_pitch_yaw_radians(
            roll_degrees.to_radians(),
            pitch_degrees.to_radians(),
            yaw_degrees.to_radians(),
        )
    }

    /// Create a Quaternion from roll and pitch Euler angles (in degrees), assumes yaw angle is zero.
    #[inline]
    pub fn from_roll_pitch_angles_degrees(roll_degrees: T, pitch_degrees: T) -> Self {
        Self::from_roll_pitch_radians(roll_degrees.to_radians(), pitch_degrees.to_radians())
    }
}

// **** From ****

// **** From Tuple ****

impl<T> From<(T, T, T, T)> for Quaternion<T>
where
    T: Copy,
{
    /// Quaternion from tuple.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let v = Quaternionf32::from((2.0, 3.0, 5.0, 7.0));
    /// let w: Quaternionf32 = (11.0, 13.0, 17.0, 19.0).into();
    ///
    /// assert_eq!(v, Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 });
    /// assert_eq!(w, Quaternionf32 { w: 11.0, x: 13.0, y: 17.0, z: 19.0 });
    /// ```
    #[inline]
    fn from((w, x, y, z): (T, T, T, T)) -> Self {
        Self { w, x, y, z }
    }
}

// **** From Array ****

impl<T: Copy> From<[T; 4]> for Quaternion<T> {
    /// Quaternion from array.
    /// ```
    /// # use vqm::Quaternionf32;
    ///
    /// let v = Quaternionf32::from([2.0, 3.0, 5.0, 6.0]);
    /// let w: Quaternionf32 = [7.0, 11.0, 13.0, 17.0].into();
    ///
    /// assert_eq!(v, Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 6.0 });
    /// assert_eq!(w, Quaternionf32 { w: 7.0, x: 11.0, y: 13.0, z: 17.0 });
    /// ```
    #[inline]
    fn from(q: [T; 4]) -> Self {
        Self { w: q[0], x: q[1], y: q[2], z: q[3] }
    }
}

impl<T> From<Quaternion<T>> for [T; 4] {
    /// Array from quaternion.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let q = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
    ///
    /// let a = <[f32; 4]>::from(q);
    /// let b: [f32; 4] = q.into();
    ///
    /// assert_eq!(a, [2.0, 3.0, 5.0, 7.0]);
    /// assert_eq!(b, [2.0, 3.0, 5.0, 7.0]);
    /// ```
    #[inline]
    fn from(q: Quaternion<T>) -> Self {
        [q.w, q.x, q.y, q.z]
    }
}

impl<T> From<(T, T)> for Quaternion<T>
where
    T: MathMethods + FloatCore,
{
    #[inline]
    fn from((roll_radians, pitch_radians): (T, T)) -> Self {
        Quaternion::from_roll_pitch_radians(roll_radians, pitch_radians)
    }
}

impl<T> From<(T, T, T)> for Quaternion<T>
where
    T: MathMethods + FloatCore,
{
    #[inline]
    fn from((roll_radians, pitch_radians, yaw_radians): (T, T, T)) -> Self {
        Quaternion::from_roll_pitch_yaw_radians(roll_radians, pitch_radians, yaw_radians)
    }
}
