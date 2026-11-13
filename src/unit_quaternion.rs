use core::ops::{Mul, Neg};

use num_traits::{ConstOne, ConstZero, One, Signed, Zero, float::FloatCore};

use super::{IsScalar, MathMethods, Matrix3x3, Quaternion, QuaternionMath, Vector3};

/// `UnitQuaternion` of `f32` values<br>
pub type UnitQuaternionf32 = UnitQuaternion<f32>;
/// `UnitQuaternion` of `f64` values<br><br>
pub type UnitQuaternionf64 = UnitQuaternion<f64>;

// **** Define ****

/// Normalized quaternion.
/// Aliases `UnitQuaternionf32` and `UnitQuaternionf64` are provided.<br><br>
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitQuaternion<T> {
    q: Quaternion<T>,
}

// **** Default ****

/// Default `UnitQuaternion`.
impl<T> Default for UnitQuaternion<T>
where
    T: Copy + Zero + One,
{
    #[inline]
    fn default() -> Self {
        Self { q: Quaternion::default() }
    }
}

// **** New ****

impl<T> UnitQuaternion<T> {
    /// Create a `UnitQuaternion` normalizing it if it is normalizable, otherwise return `None`.
    #[inline]
    pub fn try_new(w: T, x: T, y: T, z: T) -> Option<Self>
    where
        T: Copy + FloatCore + MathMethods + QuaternionMath,
    {
        Self::try_from_quaternion(Quaternion::new(w, x, y, z))
    }

    /// Create a `UnitQuaternion` from a Quaternion `q`.
    /// # Preconditions
    /// `q` is finite and non-zero.
    pub fn new(q: Quaternion<T>) -> Self
    where
        T: FloatCore + MathMethods + QuaternionMath,
    {
        let norm = q.norm();
        debug_assert!(norm.is_finite() && norm > T::epsilon());
        Self { q: q / norm }
    }

    /// Create a `UnitQuaternion` from a normalized Quaternion `q` if `q` is normalizable, otherwise return `None`.
    #[inline]
    pub fn try_from_quaternion(q: Quaternion<T>) -> Option<Self>
    where
        T: Copy + FloatCore + MathMethods + QuaternionMath,
    {
        let norm = q.norm();
        if !norm.is_finite() || norm <= T::epsilon() {
            return None;
        }
        Some(Self { q: q / norm })
    }

    /// Create a `UnitQuaternion` from a Quaternion `q`.
    /// # Preconditions
    /// `q` is finite and non-zero.
    pub fn from_quaternion(q: Quaternion<T>) -> Self
    where
        T: FloatCore + MathMethods + QuaternionMath,
    {
        let norm = q.norm();
        debug_assert!(norm.is_finite() && norm > T::epsilon());
        Self { q: q / norm }
    }

    /// Return underlying quaternion.
    #[inline]
    pub fn q(self) -> Quaternion<T> {
        self.q
    }
}

impl<T> From<UnitQuaternion<T>> for Quaternion<T> {
    fn from(q: UnitQuaternion<T>) -> Self {
        q.q
    }
}

// **** One ****

impl<T> One for UnitQuaternion<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + QuaternionMath + MathMethods,
{
    /// Unit quaternion.
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
impl<T> ConstOne for UnitQuaternion<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + QuaternionMath + MathMethods,
{
    const ONE: Self = Self { q: Quaternion::ONE };
}

impl<T> UnitQuaternion<T>
where
    T: Zero + One,
{
    /// Unit quaternion.
    /// Alias for `one()` that does not require `num_traits::One`.
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        Self { q: Quaternion::identity() }
    }
}

impl<T> UnitQuaternion<T>
where
    T: Copy + One + Signed + PartialOrd,
{
    /// Return true if quaternion is near the unit quaternion.
    pub fn is_near_one(self, epsilon: T) -> bool {
        self.q.is_near_one(epsilon)
    }
}

// **** Neg ****

impl<T> Neg for UnitQuaternion<T>
where
    T: QuaternionMath,
{
    type Output = Self;

    /// Negate quaternion.
    #[inline]
    fn neg(self) -> Self {
        Self { q: -self.q }
    }
}

// **** Mul ****

// Mul is required for the `One` trait.
impl<T> Mul<UnitQuaternion<T>> for UnitQuaternion<T>
where
    T: Copy + MathMethods + QuaternionMath,
{
    type Output = Self;

    /// Multiply two quaternions and normalize.
    #[inline]
    fn mul(self, other: Self) -> Self {
        let q = self.q * other.q;
        let norm_squared = q.norm_squared();
        Self { q: q * norm_squared.sqrt_reciprocal() }
    }
}

// **** approx_eq ****

impl<T> UnitQuaternion<T>
where
    T: Copy + Signed + PartialOrd,
{
    /// Compare two unit quaternions with a tolerance.
    #[inline]
    pub fn approx_eq(&self, other: Self, epsilon: T) -> bool {
        self.q.approx_eq(other.q, epsilon)
    }
}

// **** abs ****

impl<T> UnitQuaternion<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the quaternion with all components set to their absolute values.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self { q: self.q.abs() }
    }

    /// Set all components of the quaternion to their absolute values.
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = self.abs();
        self
    }
}

// **** dot ****

impl<T> UnitQuaternion<T>
where
    T: QuaternionMath,
{
    /// Quaternion dot product.
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.q.dot(other.q)
    }
}

impl<T> UnitQuaternion<T> {
    /// Return the inverse of the unit quaternion.
    /// For a unit quaternion this is the same as the conjugate.
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self
    where
        T: Neg<Output = T>,
    {
        self.conjugate()
    }
}

impl<T> UnitQuaternion<T> {
    /// Return the conjugate of the quaternion.
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self
    where
        T: Neg<Output = T>,
    {
        Self { q: Quaternion { w: self.q.w, x: -self.q.x, y: -self.q.y, z: -self.q.z } }
    }

    /// Set quaternion to its conjugate.
    #[inline]
    pub fn conjugate_in_place(&mut self) -> &mut Self
    where
        T: Copy + Neg<Output = T>,
    {
        *self = self.conjugate();
        self
    }
}

impl<T> UnitQuaternion<T> {
    /// Return the scalar part of the quaternion.
    #[inline]
    pub fn scalar(self) -> T {
        self.q.w
    }

    /// Return the imaginary part of the quaternion.
    #[inline]
    pub fn imaginary(self) -> Vector3<T> {
        self.q.imaginary()
    }
}

// **** rotate ***

impl<T> UnitQuaternion<T> {
    /// Rotates a 3D vector by this quaternion.
    /// Uses an active rotation in a right-handed coordinate system.
    pub fn rotate(self, v: Vector3<T>) -> Vector3<T>
    where
        T: FloatCore + MathMethods,
    {
        let two: T = T::one() + T::one();
        let half = T::one() / two;

        let x2: T = self.q.x * self.q.x;
        let y2: T = self.q.y * self.q.y;
        let z2: T = self.q.z * self.q.z;

        Vector3::<T> {
            x: (v.x * (half - y2 - z2)
                + v.y * (self.q.x * self.q.y - self.q.w * self.q.z)
                + v.z * (self.q.w * self.q.y + self.q.x * self.q.z))
                * two,
            y: (v.x * (self.q.w * self.q.z + self.q.x * self.q.y)
                + v.y * (half - x2 - z2)
                + v.z * (self.q.y * self.q.z - self.q.w * self.q.x))
                * two,
            z: (v.x * (self.q.x * self.q.z - self.q.w * self.q.y)
                + v.y * (self.q.w * self.q.x + self.q.y * self.q.z)
                + v.z * (half - x2 - y2))
                * two,
        }
    }

    /// Rotate about the x-axis,
    /// equivalent to *= Quaternion(cos(theta/2), sin(theta/2), 0, 0).
    pub fn rotate_x(&mut self, theta: T) -> &mut Self
    where
        T: FloatCore + MathMethods,
    {
        self.q = *self.q.rotate_x(theta);
        self
    }

    /// Rotate about the y-axis,
    /// equivalent to *= Quaternion(cos(theta/2), 0, sin(theta/2), 0).
    pub fn rotate_y(&mut self, theta: T) -> &mut Self
    where
        T: FloatCore + MathMethods,
    {
        self.q = *self.q.rotate_y(theta);
        self
    }
    /// Rotate about the z-axis,
    /// equivalent to *= Quaternion(cos(theta/2), 0, 0, sin(theta/2)).
    pub fn rotate_z(&mut self, theta: T) -> &mut Self
    where
        T: FloatCore + MathMethods,
    {
        self.q = *self.q.rotate_z(theta);
        self
    }
}

impl<T> UnitQuaternion<T>
where
    T: FloatCore + MathMethods,
{
    /// Returns the cosine of the Euler roll angle.
    ///
    /// Uses the aerospace roll-pitch-yaw convention.
    pub fn cos_roll(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.q.w * self.q.x + self.q.y * self.q.z;
        let b: T = half - self.q.x * self.q.x - self.q.y * self.q.y;

        let sq = a * a + b * b;
        if sq == T::zero() { T::zero() } else { b * sq.sqrt_reciprocal() }
    }

    /// Returns the sine of the Euler pitch angle.
    ///
    /// Uses the aerospace roll-pitch-yaw convention.
    pub fn sin_pitch(self) -> T {
        let two: T = T::one() + T::one();
        (self.q.w * self.q.y - self.q.x * self.q.z) * two
    }

    /// Returns the non-negative cosine of the Euler pitch angle.
    ///
    /// Uses the aerospace roll-pitch-yaw convention.
    pub fn cos_pitch(self) -> T {
        let s: T = self.sin_pitch();
        let sq = T::one() - s * s;

        if sq <= T::zero() { T::zero() } else { sq.sqrt() }
    }

    /// Returns the tan of the Euler pitch angle.
    ///
    /// Uses the aerospace roll-pitch-yaw convention.
    /// `tan_pitch` is undefined at ±90° pitch.
    pub fn tan_pitch(self) -> T {
        let s: T = self.sin_pitch();
        let sq = T::one() - s * s;

        if sq <= T::zero() { T::zero() } else { s * sq.sqrt_reciprocal() }
    }

    /// Returns the cosine of the Euler pitch angle.
    ///
    /// Uses the aerospace roll-pitch-yaw convention.
    pub fn cos_yaw(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.q.w * self.q.z + self.q.x * self.q.y;
        let b: T = half - self.q.y * self.q.y - self.q.z * self.q.z;

        let sq = a * a + b * b;
        if sq == T::zero() { T::zero() } else { b * sq.sqrt_reciprocal() }
    }

    /// Returns the sin of the Euler yaw angle.
    ///
    /// Uses the aerospace roll-pitch-yaw convention.
    pub fn sin_yaw(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.q.w * self.q.z + self.q.x * self.q.y;
        let b: T = half - self.q.y * self.q.y - self.q.z * self.q.z;

        let sq = a * a + b * b;
        if sq == T::zero() { T::zero() } else { a * sq.sqrt_reciprocal() }
    }

    /// Returns the sin of the Euler roll angle.
    ///
    /// Uses the aerospace roll-pitch-yaw convention.
    pub fn sin_roll(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.q.w * self.q.x + self.q.y * self.q.z;
        let b: T = half - self.q.x * self.q.x - self.q.y * self.q.y;

        let sq = a * a + b * b;
        if sq == T::zero() { T::zero() } else { a * sq.sqrt_reciprocal() }
    }

    /// cos of the total tilt angle (direct Z-axis projection from the rotation matrix).
    ///
    /// Uses the aerospace roll-pitch-yaw convention.
    pub fn cos_tilt(self) -> T {
        T::one() - (T::one() + T::one()) * (self.q.x * self.q.x + self.q.y * self.q.y)
    }

    /// sin of the total tilt angle (direct Z-axis projection from the rotation matrix).
    ///
    /// Uses the aerospace roll-pitch-yaw convention.
    pub fn sin_tilt(self) -> T {
        let c: T = self.cos_tilt();
        let sq = T::one() - c * c;
        if sq <= T::zero() { T::zero() } else { sq.sqrt() }
    }
}

impl<T> UnitQuaternion<T>
where
    T: FloatCore + MathMethods,
{
    /// clip `sin(roll_angle)` to +/-1.0 when roll angle outside range [-90 degrees, 90 degrees].
    ///
    /// Uses the aerospace roll-pitch-yaw convention.
    pub fn sin_roll_clipped(self) -> T {
        let half = T::one() / (T::one() + T::one());
        let a: T = self.q.w * self.q.x + self.q.y * self.q.z;
        let b: T = half - self.q.x * self.q.x - self.q.y * self.q.y;
        if b < T::zero() {
            if a < T::zero() {
                return -T::one();
            }
            return T::one();
        }
        let sq = a * a + b * b;
        if sq == T::zero() { T::zero() } else { a * sq.sqrt_reciprocal() }
    }

    /// clip `sin(pitch_angle)` to +/-1.0 when pitch angle outside range [-90 degrees, 90 degrees].
    ///
    /// Uses the aerospace roll-pitch-yaw convention.
    pub fn sin_pitch_clipped(self) -> T {
        let two: T = T::one() + T::one();
        let sin_pitch = two * (self.q.w * self.q.y - self.q.x * self.q.z);
        let d = self.q.w * self.q.w - self.q.y * self.q.y;

        // if d < 0.0, then self is outside the range [-90, 90] degrees, so we return 1.0 or 1.0 according to the sign of sin_pitch
        if d < T::zero() {
            if sin_pitch < T::zero() {
                return -T::one();
            }
            return T::one();
        }
        // self is in the range [-90, 90] degrees, so we can just return sin(self)
        sin_pitch
    }
}

impl<T> UnitQuaternion<T>
where
    T: FloatCore + MathMethods,
{
    /// Calculate roll angle of quaternion, in radians.
    #[inline]
    pub fn calculate_roll_radians(self) -> T {
        let half = T::one() / (T::one() + T::one());
        (self.q.w * self.q.x + self.q.y * self.q.z).atan2(half - self.q.x * self.q.x - self.q.y * self.q.y)
    }

    /// Calculate pitch angle of quaternion, in radians.
    #[inline]
    pub fn calculate_pitch_radians(self) -> T {
        let two = T::one() + T::one();
        (two * (self.q.w * self.q.y - self.q.x * self.q.z)).asin()
    }

    /// Calculate yaw angle of quaternion, in radians.
    #[inline]
    pub fn calculate_yaw_radians(self) -> T {
        let half = T::one() / (T::one() + T::one());
        (self.q.w * self.q.z + self.q.x * self.q.y).atan2(half - self.q.y * self.q.y - self.q.z * self.q.z)
    }

    /// Calculate a quaternion's Euler angles in radians.
    /// ```
    /// # use vqm::UnitQuaternionf32;
    /// let orientation = UnitQuaternionf32::default();
    /// let (roll, pitch, yaw) = orientation.calculate_euler_angles_radians();
    /// ```
    #[inline]
    pub fn calculate_euler_angles_radians(self) -> (T, T, T) {
        (self.calculate_roll_radians(), self.calculate_pitch_radians(), self.calculate_yaw_radians())
    }

    /// Create a `UnitQuaternion` from roll, pitch, and yaw Euler angles (in radians).
    /// See: <https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_(in_3-2-1_sequence)_to_quaternion_conversion>.
    pub fn from_roll_pitch_yaw_radians(roll_radians: T, pitch_radians: T, yaw_radians: T) -> Self {
        let half: T = T::one() / (T::one() + T::one());
        let (sin_half_roll, cos_half_roll) = (roll_radians * half).sin_cos();
        let (sin_half_pitch, cos_half_pitch) = (pitch_radians * half).sin_cos();
        let (sin_half_yaw, cos_half_yaw) = (yaw_radians * half).sin_cos();
        Self {
            q: Quaternion {
                w: cos_half_roll * cos_half_pitch * cos_half_yaw + sin_half_roll * sin_half_pitch * sin_half_yaw,
                x: sin_half_roll * cos_half_pitch * cos_half_yaw - cos_half_roll * sin_half_pitch * sin_half_yaw,
                y: cos_half_roll * sin_half_pitch * cos_half_yaw + sin_half_roll * cos_half_pitch * sin_half_yaw,
                z: cos_half_roll * cos_half_pitch * sin_half_yaw - sin_half_roll * sin_half_pitch * cos_half_yaw,
            },
        }
    }

    /// Create a `UnitQuaternion` from roll, pitch, and yaw Euler angles (in radians).
    /// See: <https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_(in_3-2-1_sequence)_to_quaternion_conversion>.
    pub fn from_roll_pitch_yaw_degrees(roll_degrees: T, pitch_degrees: T, yaw_degrees: T) -> Self {
        Self::from_roll_pitch_yaw_radians(
            roll_degrees.to_radians(),
            pitch_degrees.to_radians(),
            yaw_degrees.to_radians(),
        )
    }

    /// Create a `UnitQuaternion` from roll and pitch Euler angles (in radians), assumes yaw angle is zero.
    pub fn from_roll_pitch_radians(roll_radians: T, pitch_radians: T) -> Self {
        let half: T = T::one() / (T::one() + T::one());
        let (sin_half_roll, cos_half_roll) = (roll_radians * half).sin_cos();
        let (sin_half_pitch, cos_half_pitch) = (pitch_radians * half).sin_cos();

        Self {
            q: Quaternion {
                w: cos_half_roll * cos_half_pitch,
                x: sin_half_roll * cos_half_pitch,
                y: cos_half_roll * sin_half_pitch,
                z: -sin_half_roll * sin_half_pitch,
            },
        }
    }

    /// Create a `UnitQuaternion` from roll and pitch Euler angles (in degrees), assumes yaw angle is zero.
    pub fn from_roll_pitch_degrees(roll_degrees: T, pitch_degrees: T) -> Self {
        Self::from_roll_pitch_radians(roll_degrees.to_radians(), pitch_degrees.to_radians())
    }

    /// Create a `UnitQuaternion` from roll Euler angle (in radians), assumes pitch and roll angles are zero.
    pub fn from_roll_radians(roll_radians: T) -> Self {
        let half: T = T::one() / (T::one() + T::one());
        let (sin_half_roll, cos_half_roll) = (roll_radians * half).sin_cos();

        Self { q: Quaternion { w: cos_half_roll, x: sin_half_roll, y: T::zero(), z: T::zero() } }
    }

    /// Create a `UnitQuaternion` from roll Euler angle (in degrees), assumes pitch and roll angles are zero.
    pub fn from_roll_degrees(roll_degrees: T) -> Self {
        Self::from_roll_radians(roll_degrees.to_radians())
    }

    /// Create a `UnitQuaternion` from pitch Euler angle (in radians), assumes roll and yaw angles are zero.
    pub fn from_pitch_radians(pitch_radians: T) -> Self {
        let half: T = T::one() / (T::one() + T::one());
        let (sin_half_pitch, cos_half_pitch) = (pitch_radians * half).sin_cos();

        Self { q: Quaternion { w: cos_half_pitch, x: T::zero(), y: sin_half_pitch, z: T::zero() } }
    }

    /// Create a `UnitQuaternion` from pitch Euler angle (in degrees), assumes roll and yaw angles are zero.
    pub fn from_pitch_degrees(pitch_degrees: T) -> Self {
        Self::from_pitch_radians(pitch_degrees.to_radians())
    }

    /// Create a `UnitQuaternion` from yaw Euler angle (in radians), assumes roll and pitch angles are zero.
    pub fn from_yaw_radians(yaw_radians: T) -> Self {
        let half: T = T::one() / (T::one() + T::one());
        let (sin_half_yaw, cos_half_yaw) = (yaw_radians * half).sin_cos();

        Self { q: Quaternion { w: cos_half_yaw, x: T::zero(), y: T::zero(), z: sin_half_yaw } }
    }

    /// Create a `UnitQuaternion` from yaw Euler angle (in degrees), assumes roll and pitch angles are zero.
    pub fn from_yaw_degrees(yaw_degrees: T) -> Self {
        Self::from_yaw_radians(yaw_degrees.to_radians())
    }
}

impl<T> UnitQuaternion<T>
where
    T: MathMethods + FloatCore,
{
    /// Calculate roll angle of quaternion, in degrees.
    #[inline]
    pub fn calculate_roll_degrees(self) -> T {
        self.calculate_roll_radians().to_degrees()
    }

    /// Calculate pitch angle of quaternion, in degrees.
    #[inline]
    pub fn calculate_pitch_degrees(self) -> T {
        self.calculate_pitch_radians().to_degrees()
    }

    /// Calculate yaw angle of quaternion, in degrees.
    #[inline]
    pub fn calculate_yaw_degrees(self) -> T {
        self.calculate_yaw_radians().to_degrees()
    }

    /// Calculate a quaternion's Euler angles in degrees.
    /// ```
    /// # use vqm::UnitQuaternionf32;
    /// let orientation = UnitQuaternionf32::default();
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

    /// Create a `UnitQuaternion` from roll and pitch Euler angles (in degrees), assumes yaw angle is zero.
    #[inline]
    pub fn from_roll_pitch_angles_degrees(roll_degrees: T, pitch_degrees: T) -> Self {
        Self::from_roll_pitch_radians(roll_degrees.to_radians(), pitch_degrees.to_radians())
    }
}

impl<T> UnitQuaternion<T> {
    /// Create a `UnitQuaternion` from a rotation axis and an angle in radians.
    /// Rotation axis must be normalized.
    pub fn from_axis_angle_radians(axis: Vector3<T>, angle_radians: T) -> Self
    where
        T: FloatCore + MathMethods,
    {
        let half: T = T::one() / (T::one() + T::one());
        let (sin_half_angle, cos_half_angle) = (angle_radians * half).sin_cos();
        Self {
            q: Quaternion {
                w: cos_half_angle,
                x: axis.x * sin_half_angle,
                y: axis.y * sin_half_angle,
                z: axis.z * sin_half_angle,
            },
        }
    }

    /// Create a `UnitQuaternion` from a rotation axis and an angle in degrees.
    /// Rotation axis must be normalized.
    pub fn from_axis_angle_degrees(axis: Vector3<T>, angle_degrees: T) -> Self
    where
        T: FloatCore + MathMethods,
    {
        Self::from_axis_angle_radians(axis, angle_degrees.to_radians())
    }

    /// Converts the quaternion into an axis-angle representation.
    ///
    /// Returns a tuple containing the normalized rotation axis and the rotation angle in radians.
    ///
    /// For an identity quaternion (zero rotation), an arbitrary unit axis is returned (1, 0, 0).
    ///
    /// # Returns
    /// * `(axis, angle)` - The unit rotation axis and rotation angle in radians.
    pub fn to_axis_angle(&self) -> (Vector3<T>, T)
    where
        T: FloatCore + MathMethods,
    {
        let w = self.q.w.clamp(-T::one(), T::one());

        let two: T = T::one() + T::one();
        let angle = two * w.acos();

        let s = (self.q.x * self.q.x + self.q.y * self.q.y + self.q.z * self.q.z).sqrt();

        let axis = if s > T::epsilon() {
            Vector3::new(self.q.x / s, self.q.y / s, self.q.z / s)
        } else {
            Vector3::new(T::one(), T::zero(), T::zero())
        };

        (axis, angle)
    }

    /// Calculate the quaternion representing the shortest rotation that transforms vector `a` into vector `b`.
    pub fn from_rotation_between(a: Vector3<T>, b: Vector3<T>) -> Option<Self>
    where
        T: FloatCore + MathMethods + IsScalar,
    {
        let a_norm = a.norm();
        let b_norm = b.norm();

        // Reject zero-length vectors.
        if a_norm <= T::epsilon() || b_norm <= T::epsilon() {
            return None;
        }

        // Normalize input vectors.
        let a = a / a_norm;
        let b = b / b_norm;

        let dot = a.dot(b).clamp(-T::one(), T::one());

        // Parallel vectors: identity rotation.
        if dot >= T::one() - T::epsilon() {
            return Some(Self::identity());
        }

        // Antiparallel vectors: 180-degree rotation.
        if dot <= -T::one() + T::epsilon() {
            // Select a vector that is not parallel to a.
            let perpendicular = if a.x.abs() < a.y.abs() && a.x.abs() < a.z.abs() {
                Vector3::new(T::one(), T::zero(), T::zero())
            } else if a.y.abs() < a.z.abs() {
                Vector3::new(T::zero(), T::one(), T::zero())
            } else {
                Vector3::new(T::zero(), T::zero(), T::one())
            };

            let axis = a.cross(perpendicular).normalize();

            return Some(Self { q: Quaternion { w: T::zero(), x: axis.x, y: axis.y, z: axis.z } });
        }

        // General case.
        let cross = a.cross(b);

        let two = T::one() + T::one();
        let s = (two * (T::one() + dot)).sqrt_reciprocal();

        Some(Self { q: Quaternion { w: s / two, x: cross.x * s, y: cross.y * s, z: cross.z * s } })
    }

    /// Calculates the shortest angular distance between two quaternions.
    ///
    /// Returns the smallest rotation angle required to transform the
    /// orientation represented by this quaternion into the orientation
    /// represented by `other`.
    ///
    /// The absolute value of the dot product ensures that equivalent quaternion
    /// representations (q and -q) produce an angle of zero.
    ///
    /// The returned angle is in radians and is in the range [0, π].
    ///
    /// # Returns
    /// The angular distance between the two orientations in radians.
    pub fn angle_to(self, other: Self) -> T
    where
        T: FloatCore + MathMethods + QuaternionMath,
    {
        let dot = self.dot(other).abs().clamp(T::zero(), T::one());

        let two = T::one() + T::one();
        two * dot.acos()
    }
}

impl<T> UnitQuaternion<T>
where
    T: FloatCore,
{
    /// Return the last column of the equivalent rotation matrix, but calculated more efficiently than a full conversion.
    #[inline]
    pub fn direction_cosine_matrix_z(self) -> Vector3<T> {
        let two = T::one() + T::one();
        Vector3::<T> {
            x: (self.q.w * self.q.y + self.q.x * self.q.z) * two,
            y: (self.q.y * self.q.z - self.q.w * self.q.x) * two,
            z: self.q.w * self.q.w,
        }
    }

    /// Estimate the gravity vector, ie direction of vertical in the sensor frame.
    #[inline]
    pub fn gravity(self) -> Vector3<T> {
        let two = T::one() + T::one();
        Vector3::<T> {
            x: (self.q.x * self.q.z - self.q.w * self.q.y) * two,
            y: (self.q.w * self.q.x + self.q.y * self.q.z) * two,
            z: (self.q.w * self.q.w + self.q.z * self.q.z) * two - T::one(),
        }
    }

    /// Convenience function to return half of gravity vector of quaternion.
    #[inline]
    pub fn half_gravity(self) -> Vector3<T> {
        let half: T = T::one() / (T::one() + T::one());
        Vector3::<T> {
            x: self.q.x * self.q.z - self.q.w * self.q.y,
            y: self.q.w * self.q.x + self.q.y * self.q.z,
            z: self.q.w * self.q.w + self.q.z * self.q.z - half,
        }
    }
}

// **** From ****

impl<T> From<(T, T)> for UnitQuaternion<T>
where
    T: MathMethods + FloatCore,
{
    #[inline]
    fn from((roll_radians, pitch_radians): (T, T)) -> Self {
        UnitQuaternion::from_roll_pitch_radians(roll_radians, pitch_radians)
    }
}

impl<T> From<(T, T, T)> for UnitQuaternion<T>
where
    T: MathMethods + FloatCore,
{
    #[inline]
    fn from((roll_radians, pitch_radians, yaw_radians): (T, T, T)) -> Self {
        UnitQuaternion::from_roll_pitch_yaw_radians(roll_radians, pitch_radians, yaw_radians)
    }
}

impl<T> From<Vector3<T>> for UnitQuaternion<T>
where
    T: Copy + Zero + One + PartialEq + MathMethods + QuaternionMath,
{
    /// Quaternion from Vector3.
    /// The vector forms the imaginary part of the quaternion, which is then normalized.
    /// ```
    /// # use vqm::{Quaternionf32, UnitQuaternionf32, Vector3f32};
    /// let v = Vector3f32 { x: 2.0, y: 3.0, z: 6.0 };
    /// let q = UnitQuaternionf32::from(v).q();
    ///
    /// assert_eq!(q, Quaternionf32 { w: 0.14142136, x: 0.28284273, y: 0.42426407, z: 0.84852815 });
    /// assert!(q.is_near(Quaternionf32 { w: 0.14142136, x: 0.28284273, y: 0.42426407, z: 0.84852815 }, 1e-3));
    /// ```
    #[inline]
    fn from(v: Vector3<T>) -> Self {
        let q = Quaternion { w: T::one(), x: v.x, y: v.y, z: v.z };
        Self { q: q.normalize() }
    }
}

impl<T> From<Matrix3x3<T>> for Quaternion<T>
where
    T: Copy + FloatCore + MathMethods + QuaternionMath,
{
    /// Create quaternion from a rotation matrix.
    ///
    /// Adapted from [Converting a Rotation Matrix to a Quaternion](https://d3cw3dd2w33x3b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf) by Mike Day.
    /// Note that Day's paper uses the [Shuster multiplication convention](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Alternative_conventions),
    /// rather than the Hamilton multiplication convention used by the Quaternion class.
    fn from(m: Matrix3x3<T>) -> Self {
        let half = T::one() / (T::one() + T::one());
        // Choose largest scale factor from 4w, 4x, 4y, and 4z, to avoid a scale factor of zero, or numerical instabilities caused by division of a small scale factor.
        if m.a[8] < T::zero() {
            // |(x,y)| is bigger than |(z,w)|?
            if m.a[0] > m.a[4] {
                // |x| bigger than |y|, so use x-form
                let t = T::one() + (m.a[0] - m.a[4]) - m.a[8]; // 1 + 2(xx - yy) - 1 + 2(xx + yy) = 4xx
                let q = Self { w: m.a[7] - m.a[5], x: t, y: m.a[1] + m.a[3], z: m.a[6] + m.a[2] };
                return q * t.sqrt_reciprocal() * half;
            }
            // |y| bigger than |x|, so use y-form
            let t = T::one() - (m.a[0] - m.a[4]) - m.a[8]; // 1 - 2(xx - yy) - 1 + 2(xx + yy) = 4yy
            let q = Self { w: m.a[2] - m.a[6], x: m.a[1] + m.a[3], y: t, z: m.a[5] + m.a[7] };
            return q * t.sqrt_reciprocal() * half;
        }

        // |(z,w)| bigger than |(x,y)|
        if m.a[0] < -m.a[4] {
            // |z| bigger than |w|, so use z-form
            let t = T::one() - m.a[0] - (m.a[4] - m.a[8]); // 1 - (1 - 2*(yy + zz)) - (2(yy - zz)) = 4zz
            let q = Self { w: m.a[3] - m.a[1], x: m.a[2] + m.a[6], y: m.a[5] + m.a[7], z: t };
            return q * t.sqrt_reciprocal() * half;
        }

        // |w| bigger than |z|, so use w-form
        // ww + xx + yy + zz = 1, since unit quaternion, so xx + yy + zz =  1 - ww
        let t = T::one() + m.a[0] + m.a[4] + m.a[8]; // 1 + 1 - 2*(yy + zz) + 1 - 2(xx + zz) + 1 - 2(xx + yy) =  4 - 4(xx + yy + zz) = 4 - 4(1 - ww) = 4ww
        let q = Self { w: t, x: m.a[7] - m.a[5], y: m.a[2] - m.a[6], z: m.a[3] - m.a[1] };
        q * t.sqrt_reciprocal() * half
    }
}

impl<T> From<UnitQuaternion<T>> for Matrix3x3<T>
where
    T: Copy + Zero + One + FloatCore,
{
    /// Create rotation matrix from quaternion.
    ///
    /// see [Quaternion-derived rotation matrix](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix),
    /// uses Hamilton convention.
    #[inline]
    fn from(q: UnitQuaternion<T>) -> Self {
        let two = T::one() + T::one();
        let q = q.q();
        Self {
            a: [
                T::one() - (q.y * q.y + q.z * q.z) * two,
                (q.x * q.y - q.w * q.z) * two,
                (q.w * q.y + q.x * q.z) * two,
                (q.w * q.z + q.x * q.y) * two,
                T::one() - (q.x * q.x + q.z * q.z) * two,
                (q.y * q.z - q.w * q.x) * two,
                (q.x * q.z - q.w * q.y) * two,
                (q.w * q.x + q.y * q.z) * two,
                T::one() - (q.x * q.x + q.y * q.y) * two,
            ],
        }
    }
}
