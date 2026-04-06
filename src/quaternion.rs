use core::convert::From;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};

use crate::math_methods::TrigonometricMethods;
use crate::sqrt_methods::SqrtMethods;
use crate::{QuaternionMath, Vector3d};

/// Quaternion of `f32` values<br>
pub type Quaternionf32 = Quaternion<f32>;
/// Quaternion of `f64` values<br><br>
pub type Quaternionf64 = Quaternion<f64>;

/// RollPitchYaw `struct { roll: f32, pitch: f32, yaw: f32 }`<br>
pub type RollPitchYawf32 = RollPitchYaw<f32>;
/// RollPitchYaw `struct { roll: f64, pitch: f64, yaw: f64 }`<br>
pub type RollPitchYawf64 = RollPitchYaw<f64>;

/// RollPitch `struct { roll: f32, pitch: f32 }`<br>
pub type RollPitchf32 = RollPitch<f32>;
/// RollPitch `struct { roll: f64, pitch: f64 }`<br><br>
pub type RollPitchf64 = RollPitch<f64>;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RollPitchYaw<T> {
    pub roll: T,
    pub pitch: T,
    pub yaw: T,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RollPitch<T> {
    pub roll: T,
    pub pitch: T,
}

// **** Define ****

/// `Quaternion<T>`: quaternion type `T`.<br>
/// Aliases `Quaternion32` and `Quaternionf64` are provided.<br><br>
/// `Quaternionf32` uses **SIMD** accelerations implemented in `QuaternionMath`.<br><br>
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

// **** Default ****

/// Default quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
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
    #[inline(always)]
    fn default() -> Self {
        Self { w: T::one(), x: T::zero(), y: T::zero(), z: T::zero() }
    }
}

// **** New ****

impl<T> Quaternion<T>
where
    T: Copy,
{
    /// Create a quaternion
    #[inline(always)]
    pub const fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }
}

// **** Zero ****

/// Zero quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
/// # use num_traits::Zero;
///
/// let z = Quaternionf32::zero();
///
/// assert_eq!(z, Quaternionf32 { w:0.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl<T> Zero for Quaternion<T>
where
    T: Copy + Zero + PartialEq + QuaternionMath,
{
    #[inline(always)]
    fn zero() -> Self {
        Self { w: T::zero(), x: T::zero(), y: T::zero(), z: T::zero() }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.w == T::zero() && self.x == T::zero() && self.y == T::zero() && self.z == T::zero()
    }
}

// **** One ****

/// Unit quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
/// # use num_traits::One;
///
/// let i = Quaternionf32::one();
///
/// assert_eq!(i, Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl<T> One for Quaternion<T>
where
    T: Copy + Zero + One + PartialEq + Sub<Output = T> + Mul<Output = T> + QuaternionMath,
{
    #[inline(always)]
    fn one() -> Self {
        Self { w: T::one(), x: T::zero(), y: T::zero(), z: T::zero() }
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.w == T::one() && self.x == T::zero() && self.y == T::zero() && self.z == T::zero()
    }
}

// **** Neg ****

/// Negate quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
/// let mut q = Quaternionf32 { w: 2.0, x: -3.0, y: -5.0, z: 7.0 };
/// q = -q;
///
/// assert_eq!(q, Quaternionf32 { w: -2.0, x: 3.0, y: 5.0, z: -7.0 });
/// ```
impl<T> Neg for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        T::q_neg(self)
    }
}

// **** Add ****

/// Add two quaternions
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
/// let u = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
/// let v = Quaternionf32::new(11.0, 13.0, 17.0, 19.0);
/// let r = u + v;
///
/// assert_eq!(r, Quaternionf32 { w: 13.0, x: 16.0, y: 22.0, z: 26.0 });
/// ```
impl<T> Add for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        T::q_add(self, rhs)
    }
}

// **** AddAssign ****

/// Add one quaternion to another
impl<T> AddAssign for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        // This mutates 'self' in place.
        // On RP2350, this avoids a stack copy of the current orientation.
        *self = *self + rhs;
    }
}

// **** MulAdd ****

/// Multiply quaternion by constant and add another quaternion in place
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
/// # use num_traits::MulAdd;
/// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
/// let w = Quaternionf32::new(11.0, 13.0, 17.0, 19.0);
/// let k = 23.0;
/// let r = q.mul_add(k, w);
///
/// assert_eq!(r, Quaternionf32 { w: 57.0, x: 82.0, y: 132.0, z: 180.0 });
/// ```
impl<T> MulAdd<T> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::q_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

/// Multiply quaternion by constant and add another quaternion in place
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
/// # use num_traits::MulAddAssign;
/// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
/// let w = Quaternionf32::new(11.0, 13.0, 17.0, 19.0);
/// let k = 23.0;
/// q.mul_add_assign(k, w);
///
/// assert_eq!(q, Quaternionf32 { w: 57.0, x: 82.0, y: 132.0, z: 180.0 });
/// ```
impl<T> MulAddAssign<T> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    #[inline(always)]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

/// Subtract two quaternions
impl<T> Sub for Quaternion<T>
where
    T: Copy + Add<Output = T> + QuaternionMath,
{
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-rhs)
    }
}

// **** SubAssign ****

/// Subtract one quaternion from another
impl<T> SubAssign for Quaternion<T>
where
    T: Copy + Add<Output = T> + QuaternionMath,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

// **** Scalar Mul ****

/// Pre-multiply quaternion by a constant
impl Mul<Quaternion<f32>> for f32 {
    type Output = Quaternion<f32>;
    #[inline(always)]
    fn mul(self, other: Quaternion<f32>) -> Quaternion<f32> {
        f32::q_mul_scalar(other, self)
    }
}

impl Mul<Quaternion<f64>> for f64 {
    type Output = Quaternion<f64>;
    #[inline(always)]
    fn mul(self, other: Quaternion<f64>) -> Quaternion<f64> {
        f64::q_mul_scalar(other, self)
    }
}

// **** Mul Scalar ****

/// Multiply quaternion by a constant
impl<T> Mul<T> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, k: T) -> Self {
        T::q_mul_scalar(self, k)
    }
}

// **** MulAssign ****

/// In-place multiply a quaternion by a constant
impl<T> MulAssign<T> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    #[inline(always)]
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

// **** Div Scalar ****

/// Divide a quaternion by a constant
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
/// let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
/// let r = q / 2.0;
///
/// assert_eq!(r, Quaternionf32 { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
/// ```
impl<T> Div<T> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, k: T) -> Self {
        T::q_div_scalar(self, k)
    }
}

/// In-place divide a quaternion by a constant
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
/// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
/// q /= 2.0;
///
/// assert_eq!(q, Quaternionf32 { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
/// ```
impl<T> DivAssign<T> for Quaternion<T>
where
    T: Copy + Div<Output = T> + QuaternionMath,
{
    #[inline(always)]
    fn div_assign(&mut self, k: T) {
        *self = self.div(k);
    }
}

// **** Mul ****

// **** MulAdd ****

/// Multiply two quaternions
impl<T> Mul<Quaternion<T>> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        T::q_mul(self, rhs)
    }
}

// **** MulAssign ****

/// Multiply one quaternion by another
impl<T> MulAssign<Quaternion<T>> for Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs);
    }
}

// **** Index ****

/// Access quaternion component by index
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
/// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
///
/// assert_eq!(q[0], 2.0);
/// assert_eq!(q[1], 3.0);
/// assert_eq!(q[2], 5.0);
/// assert_eq!(q[3], 7.0);
/// ```
impl<T> Index<usize> for Quaternion<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.w,
            1 => &self.x,
            2 => &self.y,
            _ => &self.z, // default to z component if index out of range
        }
    }
}

// **** IndexMut ****

// Set quaternion component by index
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
/// let mut q = Quaternionf32::new(2.0, 3.0, 5.0, 6.0);
/// q[0] = 7.0;
/// q[1] = 11.0;
/// q[2] = 13.0;
/// q[3] = 17.0;
///
/// assert_eq!(q, Quaternionf32 { w:7.0, x:11.0, y:13.0, z: 17.0 });
/// ```
impl<T> IndexMut<usize> for Quaternion<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.w,
            1 => &mut self.x,
            2 => &mut self.y,
            _ => &mut self.z, // default to z component if index out of range
        }
    }
}

// **** abs ****

impl<T> Quaternion<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the quaternion with all components set to their absolute values
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self { w: self.w.abs(), x: self.x.abs(), y: self.y.abs(), z: self.z.abs() }
    }

    /// Set all components of the quaternion to their absolute values
    #[inline(always)]
    pub fn abs_in_place(&mut self) {
        *self = self.abs();
    }
}

// **** clamp ****

impl<T> Quaternion<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the quaternion with all components clamped to the specified range
    #[inline(always)]
    pub fn clamp(self, min: T, max: T) -> Self {
        Self {
            w: self.w.clamp(min, max),
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
        }
    }

    /// Clamp all components of the quaternion to the specified range
    #[inline(always)]
    pub fn clamp_in_place(&mut self, min: T, max: T) {
        self.w = self.w.clamp(min, max);
        self.x = self.x.clamp(min, max);
        self.y = self.y.clamp(min, max);
        self.z = self.z.clamp(min, max);
    }
}

// **** mean ****

impl<T> Quaternion<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the quaternion
    #[inline(always)]
    pub fn mean(self) -> T {
        let four = T::one() + T::one() + T::one() + T::one();
        (self.w + self.x + self.y + self.z) / four
    }
}

// **** norm_squared ****

impl<T> Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    /// Return square of Euclidean norm
    #[inline(always)]
    pub fn norm_squared(self) -> T {
        T::q_norm_squared(self)
    }
}

// **** norm ****

impl<T> Quaternion<T>
where
    T: Copy + SqrtMethods + QuaternionMath,
{
    /// Return Euclidean norm
    #[inline(always)]
    pub fn norm(self) -> T {
        Self::norm_squared(self).sqrt()
    }
}

impl<T> Quaternion<T>
where
    T: Copy + Zero + PartialOrd + SqrtMethods + QuaternionMath,
{
    /// Return normalized form of the quaternion
    #[inline(always)]
    pub fn normalized(self) -> Self {
        let norm: T = self.norm();
        // If norm == 0.0 then the quaternion is already normalized
        if norm == T::zero() {
            return self;
        }
        self * T::q_reciprocal(norm)
    }

    /// Normalize the quaternion in place
    #[inline(always)]
    pub fn normalize(&mut self) {
        let norm: T = self.norm();
        // If norm == 0.0 then the quaternion is already normalized
        if norm != T::zero() {
            *self *= T::q_reciprocal(norm);
        }
    }
}

impl<T> Quaternion<T>
where
    T: Copy + Zero + One + PartialOrd + Neg<Output = T> + Sub<Output = T> + Div<Output = T> + SqrtMethods,
{
    pub fn rotate(self, v: &Vector3d<T>) -> Vector3d<T> {
        let two: T = T::one() + T::one();
        let half = T::one() / two;
        let x2: T = self.x * self.x;
        let y2: T = self.y * self.y;
        let z2: T = self.z * self.z;
        Vector3d::<T> {
            x: v.x * (half - y2 - z2)
                + v.y * (self.x * self.y - self.w * self.z)
                + v.z * (self.w * self.y + self.x * self.z),
            y: v.x * (self.w * self.z + self.x * self.y)
                + v.y * (half - x2 - z2)
                + v.z * (self.y * self.z - self.w * self.x),
            z: v.x * (self.x * self.z - self.w * self.y)
                + v.y * (self.w * self.x + self.y * self.z)
                + v.z * (half - x2 - y2),
        } //* two
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
        (T::one() - s * s).sqrt()
    }

    pub fn tan_pitch(self) -> T {
        let s: T = self.sin_pitch();
        s * (T::one() - s * s).sqrt_reciprocal()
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

    /// clip sin(roll_angle) to +/-1.0 when roll angle outside range [-90 degrees, 90 degrees]
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
}

impl<T> Quaternion<T>
where
    T: Copy + Zero + One + Neg<Output = T> + Sub<Output = T> + Div<Output = T> + TrigonometricMethods,
{
    /// Rotate about the x-axis,
    /// equivalent to *= Quaternion(cos(theta/2), sin(theta/2), 0, 0)
    pub fn rotate_x(&mut self, theta: T) -> Self {
        let two = T::one() + T::one();
        let (sin, cos) = (theta / two).sin_cos();
        let wt: T = self.w * cos - self.x * sin;
        self.x = self.w * sin + self.x * cos;
        let yt: T = self.y * cos + self.z * sin;
        self.z = self.z * cos - self.y * sin;
        self.w = wt;
        self.y = yt;
        *self
    }
    /// Rotate about the y-axis,
    /// equivalent to *= Quaternion(cos(theta/2), 0, sin(theta/2), 0)
    pub fn rotate_y(&mut self, theta: T) -> Self {
        let two = T::one() + T::one();
        let (sin, cos) = (theta / two).sin_cos();
        let wt: T = self.w * cos - self.y * sin;
        let xt: T = self.x * cos - self.z * sin;
        self.y = self.w * sin + self.y * cos;
        self.z = self.x * sin - self.z * cos;
        self.w = wt;
        self.x = xt;
        *self
    }

    /// Rotate about the z-axis,
    /// equivalent to *= Quaternion(cos(theta/2), 0, 0, sin(theta/2))
    pub fn rotate_z(&mut self, theta: T) -> Self {
        let two = T::one() + T::one();
        let (sin, cos) = (theta / two).sin_cos();
        let wt: T = self.w * cos - self.z * sin;
        let xt: T = self.x * cos - self.y * sin;
        self.y = self.x * sin + self.y * cos;
        self.z = self.z * cos - self.w * sin;
        self.w = wt;
        self.x = xt;
        *self
    }

    #[inline(always)]
    pub fn calculate_roll_radians(self) -> T {
        let half = T::one() / (T::one() + T::one());
        (self.w * self.x + self.y * self.z).atan2(half - self.x * self.x - self.y * self.y)
    }

    #[inline(always)]
    pub fn calculate_pitch_radians(self) -> T {
        (self.w * self.y - self.x * self.z).asin()
    }

    #[inline(always)]
    pub fn calculate_yaw_radians(self) -> T {
        let half = T::one() / (T::one() + T::one());
        (self.w * self.z + self.x * self.y).atan2(half - self.y * self.y - self.z * self.z)
    }

    /// Create a Quaternion from roll, pitch, and yaw Euler angles (in radians).
    /// See: <https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_(in_3-2-1_sequence)_to_quaternion_conversion>
    pub fn from_roll_pitch_yaw_angles_radians(roll_radians: T, pitch_radians: T, yaw_radians: T) -> Self {
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

    /// Create a Quaternion from roll and pitch Euler angles (in radians), assumes yaw angle is zero.
    pub fn from_roll_pitch_angles_radians(roll_radians: T, pitch_radians: T) -> Self {
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
}

impl<T> Quaternion<T>
where
    T: Copy + QuaternionMath,
{
    // Return the conjugate of the quaternion

    #[inline(always)]
    pub fn conjugate(self) -> Self {
        T::q_conjugate(self)
    }
}

impl<T> Quaternion<T>
where
    T: Copy + One + Neg<Output = T> + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    /// Return the imaginary part of the quaternion
    #[inline(always)]
    pub fn imaginary(self) -> Vector3d<T> {
        Vector3d::<T> { x: self.x, y: self.y, z: self.z }
    }
    /// Return the last column of the equivalent rotation matrix, but calculated more efficiently than a full conversion
    #[inline(always)]
    pub fn direction_cosine_matrix_z(self) -> Vector3d<T> {
        let two = T::one() + T::one();
        Vector3d::<T> {
            x: (self.w * self.y + self.x * self.z) * two,
            y: (self.y * self.z - self.w * self.x) * two,
            z: self.w * self.w,
        }
    }

    #[inline(always)]
    pub fn gravity(self) -> Vector3d<T> {
        let two = T::one() + T::one();
        Vector3d::<T> {
            x: (self.x * self.z - self.w * self.y) * two,
            y: (self.w * self.x + self.y * self.z) * two,
            z: (self.w * self.w + self.z * self.z) * two - T::one(),
        }
    }

    #[inline(always)]
    pub fn half_gravity(self) -> Vector3d<T> {
        let half: T = T::one() / (T::one() + T::one());
        Vector3d::<T> {
            x: self.x * self.z - self.w * self.y,
            y: self.w * self.x + self.y * self.z,
            z: self.w * self.w + self.z * self.z - half,
        }
    }
}

impl<T> Quaternion<T>
where
    T: Copy
        + Zero
        + One
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + SqrtMethods
        + TrigonometricMethods
        + FloatCore,
{
    #[inline(always)]
    pub fn calculate_roll_degrees(self) -> T {
        self.calculate_roll_radians().to_degrees()
    }

    #[inline(always)]
    pub fn calculate_pitch_degrees(self) -> T {
        self.calculate_pitch_radians().to_degrees()
    }

    #[inline(always)]
    pub fn calculate_yaw_degrees(self) -> T {
        self.calculate_yaw_radians().to_degrees()
    }

    /// Create a Quaternion from roll, pitch, and yaw Euler angles (in degrees).
    #[inline(always)]
    pub fn from_roll_pitch_yaw_angles_degrees(roll_degrees: T, pitch_degrees: T, yaw_degrees: T) -> Self {
        Self::from_roll_pitch_yaw_angles_radians(
            roll_degrees.to_radians(),
            pitch_degrees.to_radians(),
            yaw_degrees.to_radians(),
        )
    }

    /// Create a Quaternion from roll and pitch Euler angles (in degrees), assumes yaw angle is zero.
    #[inline(always)]
    pub fn from_roll_pitch_angles_degrees(roll_degrees: T, pitch_degrees: T) -> Self {
        Self::from_roll_pitch_angles_radians(roll_degrees.to_radians(), pitch_degrees.to_radians())
    }
}

// **** From ****

/// Quaternion from array
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
///
/// let v = Quaternionf32::from([2.0, 3.0, 5.0, 6.0]);
/// let w: Quaternionf32 = [7.0, 11.0, 13.0, 17.0].into();
///
/// assert_eq!(v, Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 6.0 });
/// assert_eq!(w, Quaternionf32 { w: 7.0, x: 11.0, y: 13.0, z: 17.0 });
/// ```
impl<T> From<[T; 4]> for Quaternion<T>
where
    T: Copy,
{
    #[inline(always)]
    fn from(q: [T; 4]) -> Self {
        Self { w: q[0], x: q[1], y: q[2], z: q[3] }
    }
}

/// Array from quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternionf32;
///
/// let q = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
///
/// let a = <[f32; 4]>::from(q);
/// let b: [f32; 4] = q.into();
///
/// assert_eq!(a, [2.0, 3.0, 5.0, 7.0]);
/// assert_eq!(b, [2.0, 3.0, 5.0, 7.0]);
/// ```
impl<T> From<Quaternion<T>> for [T; 4] {
    #[inline(always)]
    fn from(q: Quaternion<T>) -> Self {
        [q.w, q.x, q.y, q.z]
    }
}

impl<T> From<(T, T)> for Quaternion<T>
where
    T: Copy
        + Zero
        + One
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + SqrtMethods
        + TrigonometricMethods
        + FloatCore,
{
    #[inline(always)]
    fn from((roll_radians, pitch_radians): (T, T)) -> Self {
        Quaternion::from_roll_pitch_angles_radians(roll_radians, pitch_radians)
    }
}

impl<T> From<(T, T, T)> for Quaternion<T>
where
    T: Copy
        + Zero
        + One
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + SqrtMethods
        + TrigonometricMethods
        + FloatCore,
{
    #[inline(always)]
    fn from((roll_radians, pitch_radians, yaw_radians): (T, T, T)) -> Self {
        Quaternion::from_roll_pitch_yaw_angles_radians(roll_radians, pitch_radians, yaw_radians)
    }
}

impl<T> From<RollPitchYaw<T>> for Quaternion<T>
where
    T: Copy
        + Zero
        + One
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + SqrtMethods
        + TrigonometricMethods
        + FloatCore,
{
    fn from(angles: RollPitchYaw<T>) -> Self {
        Quaternion::from_roll_pitch_yaw_angles_radians(angles.roll, angles.pitch, angles.yaw)
    }
}

impl<T> From<RollPitch<T>> for Quaternion<T>
where
    T: Copy
        + Zero
        + One
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + SqrtMethods
        + TrigonometricMethods
        + FloatCore,
{
    #[inline(always)]
    fn from(angles: RollPitch<T>) -> Self {
        Quaternion::from_roll_pitch_angles_radians(angles.roll, angles.pitch)
    }
}
