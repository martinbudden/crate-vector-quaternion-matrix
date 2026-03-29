use core::convert::From;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{One, Signed, Zero, float::FloatCore};

use crate::Vector3d;
use crate::math_methods::MathMethods;

/// quaternion of `f32` values
pub type Quaternionf32 = Quaternion<f32>;
/// quaternion of `f64` values
pub type Quaternionf64 = Quaternion<f64>;

/// RollPitchYaw `struct { roll: f32, pitch: f32, yaw: f32 }`
pub type RollPitchYawf32 = RollPitchYaw<f32>;
/// RollPitchYaw `struct { roll: f64, pitch: f64, yaw: f64 }`
pub type RollPitchYawf64 = RollPitchYaw<f64>;

/// RollPitch `struct { roll: f32, pitch: f32 }`
pub type RollPitchf32 = RollPitch<f32>;
/// RollPitch `struct { roll: f64, pitch: f64 }`
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
/// Aliases `Quaternion32` and `Quaternionf64` are provided.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

// **** Default ****
impl<T> Default for Quaternion<T>
where
    T: Zero + One,
{
    fn default() -> Self {
        Self { w: T::one(), x: T::zero(), y: T::zero(), z: T::zero() }
    }
}

// **** Zero ****
/// Zero quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternion;
/// # use num_traits::Zero;
///
/// let z = Quaternion::<f32>::zero();
///
/// assert_eq!(z, Quaternion::<f32> { w:0.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl<T> Zero for Quaternion<T>
where
    T: Zero + PartialEq,
{
    fn zero() -> Self {
        Self { w: T::zero(), x: T::zero(), y: T::zero(), z: T::zero() }
    }

    fn is_zero(&self) -> bool {
        self.w == T::zero() && self.x == T::zero() && self.y == T::zero() && self.z == T::zero()
    }
}

// **** One ****
/// Unit quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternion;
/// # use num_traits::One;
///
/// let i = Quaternion::<f32>::one();
///
/// assert_eq!(i, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl<T> One for Quaternion<T>
where
    T: Copy + Zero + One + PartialEq + Sub<Output = T> + Mul<Output = T>,
{
    fn one() -> Self {
        Self { w: T::one(), x: T::zero(), y: T::zero(), z: T::zero() }
    }

    fn is_one(&self) -> bool {
        self.w == T::one() && self.x == T::zero() && self.y == T::zero() && self.z == T::zero()
    }
}

// **** Neg ****
/// Negate quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternion;
/// let mut q = Quaternion{ w: 2.0, x: -3.0, y: -5.0, z: 7.0 };
/// q = -q;
///
/// assert_eq!(q, Quaternion { w: -2.0, x: 3.0, y: 5.0, z: -7.0 });
/// ```
impl<T> Neg for Quaternion<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self { w: -self.w, x: -self.x, y: -self.y, z: -self.z }
    }
}

// **** NegReference ****
/// Negate vector reference
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let q = Quaternion { w: 2.0, x: -3.0, y: -5.0, z: 7.0 };
/// let r = -q;
///
/// assert_eq!(r, Quaternion { w: -2.0, x: 3.0, y: 5.0, z: -7.0 });
/// assert_eq!(q, Quaternion { w: 2.0, x: -3.0, y: -5.0, z: 7.0 });
/// ```
impl<T> Neg for &Quaternion<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Quaternion<T>;
    fn neg(self) -> Self::Output {
        Quaternion { w: -self.w, x: -self.x, y: -self.y, z: -self.z }
    }
}

// **** Add ****
/// Add two quaternions
/// ```
/// # use vector_quaternion_matrix::Quaternion;
/// let u = Quaternion::new(2.0, 3.0, 5.0, 7.0);
/// let v = Quaternion::new(11.0, 13.0, 17.0, 19.0);
/// let r = u + v;
///
/// assert_eq!(r, Quaternion { w: 13.0, x: 16.0, y: 22.0, z: 26.0 });
/// ```
impl<T> Add for Quaternion<T>
where
    T: Add<Output = T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { w: self.w + rhs.w, x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

// **** AddAssign ****
/// Add one quaternion to another
impl<T> AddAssign for Quaternion<T>
where
    T: Copy + Add<Output = T>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

// **** Sub ****
/// Subtract two quaternions
impl<T> Sub for Quaternion<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { w: self.w - rhs.w, x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

// **** SubAssign ****
/// Subtract one quaternion from another
impl<T> SubAssign for Quaternion<T>
where
    T: Copy + Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

// **** Pre-multiply ****
/// Pre-multiply quaternion by a constant
impl Mul<Quaternion<f32>> for f32 {
    type Output = Quaternion<f32>;
    fn mul(self, rhs: Quaternion<f32>) -> Quaternion<f32> {
        Quaternion { w: self * rhs.w, x: self * rhs.x, y: self * rhs.y, z: self * rhs.z }
    }
}

impl Mul<Quaternion<f64>> for f64 {
    type Output = Quaternion<f64>;
    fn mul(self, rhs: Quaternion<f64>) -> Quaternion<f64> {
        Quaternion { w: self * rhs.w, x: self * rhs.x, y: self * rhs.y, z: self * rhs.z }
    }
}

// **** Mul ****
/// Multiply quaternion by a constant
impl<T> Mul<T> for Quaternion<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self { w: self.w * k, x: self.x * k, y: self.y * k, z: self.z * k }
    }
}

// **** MulAssign ****
/// In-place multiply a quaternion by a constant
impl<T> MulAssign<T> for Quaternion<T>
where
    T: Copy + Mul<Output = T>,
{
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

/// Multiply two quaternions
impl<T> Mul<Quaternion<T>> for Quaternion<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

/// Multiply one quaternion by another
impl<T> MulAssign<Quaternion<T>> for Quaternion<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// **** Div ****
/// Divide a quaternion by a constant
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let q = Quaternion::new(2.0, 3.0, 5.0, 7.0);
/// let r = q / 2.0;
///
/// assert_eq!(r, Quaternion { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
/// ```
impl<T> Div<T> for Quaternion<T>
where
    T: Copy + One + Div<Output = T>,
{
    type Output = Self;
    fn div(self, k: T) -> Self {
        let r: T = T::one() / k;
        Self { w: self.w * r, x: self.x * r, y: self.y * r, z: self.z * r }
    }
}

// **** DivAssign ****
/// In-place divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let mut q = Quaternion::new(2.0, 3.0, 5.0, 7.0);
/// q /= 2.0;
///
/// assert_eq!(q, Quaternion { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
/// ```
impl<T> DivAssign<T> for Quaternion<T>
where
    T: Copy + One + Div<Output = T>,
{
    fn div_assign(&mut self, k: T) {
        *self = *self / k;
    }
}

// **** Index ****
/// Access quaternion component by index
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let mut q = Quaternion::new(2.0, 3.0, 5.0, 7.0);
///
/// assert_eq!(q[0], 2.0);
/// assert_eq!(q[1], 3.0);
/// assert_eq!(q[2], 5.0);
/// assert_eq!(q[3], 7.0);
/// ```
impl<T> Index<usize> for Quaternion<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.w,
            1 => &self.x,
            2 => &self.y,
            3 => &self.z,
            _ => &self.z, // default to z component if index out of range
        }
    }
}

// **** IndexMut ****
// Set quaternion component by index
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let mut q = Quaternion::new(2.0, 3.0, 5.0, 6.0);
/// q[0] = 7.0;
/// q[1] = 11.0;
/// q[2] = 13.0;
/// q[3] = 17.0;
///
/// assert_eq!(q, Quaternion { w:7.0, x:11.0, y:13.0, z: 17.0 });
/// ```
impl<T> IndexMut<usize> for Quaternion<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.w,
            1 => &mut self.x,
            2 => &mut self.y,
            3 => &mut self.z,
            _ => &mut self.z, // default to z component if index out of range
        }
    }
}

// **** impl new ****
impl<T> Quaternion<T>
where
    T: Copy,
{
    /// Create a quaternion
    pub const fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }
}

// **** impl abs ****
impl<T> Quaternion<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the quaternion with all components set to their absolute values
    pub fn abs(&self) -> Self {
        Self { w: self.w.abs(), x: self.x.abs(), y: self.y.abs(), z: self.z.abs() }
    }

    /// Set all components of the quaternion to their absolute values
    pub fn abs_in_place(&mut self) {
        *self = self.abs();
    }
}

// **** impl clamp ****
impl<T> Quaternion<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the quaternion with all components clamped to the specified range
    pub fn clamp(&self, min: T, max: T) -> Self {
        Self {
            w: self.w.clamp(min, max),
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
        }
    }

    /// Clamp all components of the quaternion to the specified range
    pub fn clamp_in_place(&mut self, min: T, max: T) {
        self.w = self.w.clamp(min, max);
        self.x = self.x.clamp(min, max);
        self.y = self.y.clamp(min, max);
        self.z = self.z.clamp(min, max);
    }
}

// **** impl squared_norm ****
impl<T> Quaternion<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    /// Return square of Euclidean norm
    pub fn squared_norm(&self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }
}

// **** impl mean ****
impl<T> Quaternion<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the quaternion
    pub fn mean(&self) -> T {
        let four = T::one() + T::one() + T::one() + T::one();
        (self.w + self.x + self.y + self.z) / four
    }
}
// **** impl norm ****
impl<T> Quaternion<T>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T> + MathMethods,
{
    /// Return Euclidean norm
    pub fn norm(&self) -> T {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl<T> Quaternion<T>
where
    T: Copy + Zero + One + PartialOrd + Div<Output = T> + MathMethods,
{
    /// Return normalized form of the quaternion
    pub fn normalized(&self) -> Self {
        let norm: T = self.norm();
        // If norm == 0.0 then the quaternion is already normalized
        if norm == T::zero() {
            return *self;
        }
        *self / norm
    }

    /// Normalize the quaternion in place
    pub fn normalize(&mut self) {
        let norm: T = self.norm();
        #[allow(clippy::assign_op_pattern)]
        // If norm == 0.0 then the quaternion is already normalized
        if norm != T::zero() {
            *self = *self / self.norm();
        }
    }
}

impl<T> Quaternion<T>
where
    T: Copy + Zero + One + PartialOrd + Neg<Output = T> + Sub<Output = T> + Div<Output = T> + MathMethods,
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
        } * two
    }

    pub fn calculate_roll_radians(self) -> T {
        let half = T::one() / (T::one() + T::one());
        (self.w * self.x + self.y * self.z).atan2(half - self.x * self.x - self.y * self.y)
    }

    pub fn calculate_pitch_radians(self) -> T {
        (self.w * self.y - self.x * self.z).asin()
    }

    pub fn calculate_yaw_radians(self) -> T {
        let half = T::one() / (T::one() + T::one());
        (self.w * self.z + self.x * self.y).atan2(half - self.y * self.y - self.z * self.z)
    }

    pub fn sin_roll(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.w * self.x + self.y * self.z;
        let b: T = half - self.x * self.x - self.y * self.y;
        a * (a * a + b * b).reciprocal_sqrt()
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
        a * (a * a + b * b).reciprocal_sqrt()
    }

    pub fn cos_roll(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.w * self.x + self.y * self.z;
        let b: T = half - self.x * self.x - self.y * self.y;
        b * (a * a + b * b).reciprocal_sqrt()
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
        s * (T::one() - s * s).reciprocal_sqrt()
    }

    pub fn cos_yaw(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.w * self.z + self.x * self.y;
        let b: T = half - self.y * self.y - self.z * self.z;
        b * (a * a + b * b).reciprocal_sqrt()
    }

    pub fn sin_yaw(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.w * self.z + self.x * self.y;
        let b: T = half - self.y * self.y - self.z * self.z;
        a * (a * a + b * b).reciprocal_sqrt()
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
    T: Copy + One + Neg<Output = T> + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    // Return the conjugate of the quaternion
    pub fn conjugate(self) -> Self {
        Self { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    /// Return the imaginary part of the quaternion
    pub fn imaginary(self) -> Vector3d<T> {
        Vector3d::<T> { x: self.x, y: self.y, z: self.z }
    }
    /// Return the last column of the equivalent rotation matrix, but calculated more efficiently than a full conversion
    pub fn direction_cosine_matrix_z(self) -> Vector3d<T> {
        let two = T::one() + T::one();
        Vector3d::<T> {
            x: (self.w * self.y + self.x * self.z) * two,
            y: (self.y * self.z - self.w * self.x) * two,
            z: self.w * self.w,
        }
    }

    pub fn gravity(&self) -> Vector3d<T> {
        let two = T::one() + T::one();
        Vector3d::<T> {
            x: (self.x * self.z - self.w * self.y) * two,
            y: (self.w * self.x + self.y * self.z) * two,
            z: (self.w * self.w + self.z * self.z) * two - T::one(),
        }
    }

    pub fn half_gravity(&self) -> Vector3d<T> {
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
        + MathMethods
        + FloatCore,
{
    pub fn calculate_roll_degrees(self) -> T {
        self.calculate_roll_radians().to_degrees()
    }

    pub fn calculate_pitch_degrees(self) -> T {
        self.calculate_pitch_radians().to_degrees()
    }

    pub fn calculate_yaw_degrees(self) -> T {
        self.calculate_yaw_radians().to_degrees()
    }

    /// Create a Quaternion from roll, pitch, and yaw Euler angles (in degrees).
    pub fn from_roll_pitch_yaw_angles_degrees(roll_degrees: T, pitch_degrees: T, yaw_degrees: T) -> Self {
        Self::from_roll_pitch_yaw_angles_radians(
            roll_degrees.to_radians(),
            pitch_degrees.to_radians(),
            yaw_degrees.to_radians(),
        )
    }

    /// Create a Quaternion from roll and pitch Euler angles (in degrees), assumes yaw angle is zero.
    pub fn from_roll_pitch_angles_degrees(roll_degrees: T, pitch_degrees: T) -> Self {
        Self::from_roll_pitch_angles_radians(roll_degrees.to_radians(), pitch_degrees.to_radians())
    }
}

// **** From ****
/// Quaternion from array
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let v = Quaternion::<f32>::from([2.0, 3.0, 5.0, 6.0]);
/// let w: Quaternion::<f32> = [7.0, 11.0, 13.0, 17.0].into();
///
/// assert_eq!(v, Quaternion::<f32> { w: 2.0, x: 3.0, y: 5.0, z: 6.0 });
/// assert_eq!(w, Quaternion::<f32> { w: 7.0, x: 11.0, y: 13.0, z: 17.0 });
/// ```
impl<T> From<[T; 4]> for Quaternion<T>
where
    T: Copy,
{
    fn from(q: [T; 4]) -> Self {
        Self { w: q[0], x: q[1], y: q[2], z: q[3] }
    }
}

/// Array from quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let q = Quaternion::<f32> { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
///
/// let a = <[f32; 4]>::from(q);
/// let b: [f32; 4] = q.into();
///
/// assert_eq!(a, [2.0, 3.0, 5.0, 7.0]);
/// assert_eq!(b, [2.0, 3.0, 5.0, 7.0]);
/// ```
impl<T> From<Quaternion<T>> for [T; 4] {
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
        + MathMethods
        + FloatCore,
{
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
        + MathMethods
        + FloatCore,
{
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
        + MathMethods
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
        + MathMethods
        + FloatCore,
{
    fn from(angles: RollPitch<T>) -> Self {
        Quaternion::from_roll_pitch_angles_radians(angles.roll, angles.pitch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<Quaternionf32>();
    }
    #[test]
    fn default() {
        let a = Quaternionf32::default();
        assert_eq!(a, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
        assert!(a.is_one());
        let z = Quaternionf32::zero();
        assert!(z.is_zero());
        let i = Quaternionf32::one();
        assert!(i.is_one());
    }
    #[test]
    fn from() {
        let a = Quaternionf32::from((0.0, 0.0, 0.0));
        let b = Quaternionf32::from_roll_pitch_yaw_angles_radians(0.0, 0.0, 0.0);
        assert_eq!(a, b);
        let c = Quaternionf32::from((0.0, 0.0));
        let d = Quaternionf32::from_roll_pitch_angles_radians(0.0, 0.0);
        assert_eq!(c, d);
    }
    #[test]
    fn neg() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(-a, Quaternion { w: -2.0, x: -3.0, y: -5.0, z: -7.0 });

        let b = -a;
        assert_eq!(b, Quaternion { w: -2.0, x: -3.0, y: -5.0, z: -7.0 });
    }
    #[test]
    fn add() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternion { w: 11.0, x: 13.0, y: 17.0, z: 19.0 };
        assert_eq!(a + b, Quaternion { w: 13.0, x: 16.0, y: 22.0, z: 26.0 });
    }
    #[test]
    fn add_assign() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternion { w: 11.0, x: 13.0, y: 17.0, z: 19.0 };
        let mut c = a;
        c += b;
        assert_eq!(c, Quaternion { w: 13.0, x: 16.0, y: 22.0, z: 26.0 });
    }
    #[test]
    fn sub() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternion { w: 11.0, x: 13.0, y: 17.0, z: 23.0 };
        let c = a - b;
        assert_eq!(c, Quaternion { w: -9.0, x: -10.0, y: -12.0, z: -16.0 });
    }
    #[test]
    fn sub_assign() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternion { w: 11.0, x: 13.0, y: 17.0, z: 23.0 };
        let mut c = a;
        c -= b;
        assert_eq!(c, Quaternion { w: -9.0, x: -10.0, y: -12.0, z: -16.0 });
    }
    #[test]
    fn mul() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(a * 2.0, Quaternion { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
        assert_eq!(2.0 * a, Quaternion { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
    }
    #[test]
    fn mul_assign() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let mut b = a;
        b *= 2.0;
        assert_eq!(b, Quaternion { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
    }
    #[test]
    fn div() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(a / 2.0, Quaternion { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
    }
    #[test]
    fn div_assign() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let mut b = a;
        b /= 2.0;
        assert_eq!(b, Quaternion { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
    }
    #[test]
    fn new() {
        let a = Quaternion::new(2.0, 3.0, 5.0, 7.0);
        assert_eq!(a, Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 });
    }
    #[test]
    fn squared_norm() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(a.squared_norm(), 87.0);
    }
    #[test]
    fn norm() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(a.norm(), 87.0_f32.sqrt());
        let z = Quaternion { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        assert_eq!(z.norm(), 0.0);
    }
    #[test]
    fn normalized() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = a / 87.0_f32.sqrt();
        assert_eq!(a.normalized(), b);
        let z = Quaternion { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        assert_eq!(z.normalized(), z);
    }
    #[test]
    fn normalize() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let a_normalized = a.normalized();
        let mut b = a;
        b.normalize();
        assert_eq!(b, a_normalized);
        let z = Quaternion { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        let mut y = z;
        y.normalize();
        assert_eq!(z, y);
    }
    #[test]
    fn abs() {
        let a = Quaternion { w: -2.0, x: 3.0, y: -5.0, z: -7.0 };
        assert_eq!(a.abs(), Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 });
    }
    #[test]
    fn abs_in_place() {
        let a = Quaternion { w: -2.0, x: -3.0, y: 5.0, z: 7.0 };
        let mut b = a;
        b.abs_in_place();
        assert_eq!(b, a.abs());
    }
    #[test]
    fn clamp() {
        let a = Quaternion { w: -5.0, x: -2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.clamp(-1.0, 4.0), Quaternion { w: -1.0, x: -1.0, y: 3.0, z: 4.0 });
    }
    #[test]
    fn clamp_in_place() {
        let a = Quaternion { w: -5.0, x: -2.0, y: 3.0, z: 5.0 };
        let mut b = a;
        b.clamp_in_place(-1.0, 4.0);
        assert_eq!(b, a.clamp(-1.0, 4.0));
    }
}
