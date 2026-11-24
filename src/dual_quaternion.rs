use crate::{IsScalar, MathMethods, QuaternionMath, Vector3};
use core::convert::From;
use core::ops::{Add, Div, Mul, MulAssign, Neg};
use num_traits::Signed;
use num_traits::{ConstOne, ConstZero, One, Zero, float::FloatCore};
#[cfg(feature = "storage")]
use sequential_storage::map::PostcardValue;

#[cfg(feature = "serde")]
use {
    postcard::experimental::max_size::MaxSize,
    serde::{Deserialize, Serialize},
};

use super::{Quaternion, UnitQuaternion};

/// `DualQuaternion` of `f32` values<br>
pub type DualQuaternionf32 = DualQuaternion<f32>;
/// `DualQuaternion` of `f64` values<br><br>
pub type DualQuaternionf64 = DualQuaternion<f64>;

// **** Define ****

/// `DualQuaternion<T>`: quaternion type `T`.<br>
/// Aliases `Quaternionf32` and `Quaternionf64` are provided.<br><br>
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize, MaxSize))]
pub struct DualQuaternion<T>(UnitQuaternion<T>, Quaternion<T>);

#[cfg(feature = "storage")]
impl<T> PostcardValue<'_> for DualQuaternion<T> where T: Serialize + for<'de> Deserialize<'de> {}

// **** Default ****

/// Default dual quaternion.
impl<T> Default for DualQuaternion<T>
where
    T: Copy + Zero + One + ConstZero + ConstOne + PartialEq + FloatCore + MathMethods + QuaternionMath,
{
    #[inline]
    fn default() -> Self {
        Self(UnitQuaternion::one(), Quaternion::zero())
    }
}

// **** New ****

impl<T> DualQuaternion<T> {
    /// Create a dual quaternion, from a unit quaternion and a quaternion.
    #[inline]
    pub const fn new(r: UnitQuaternion<T>, t: Quaternion<T>) -> Self {
        Self(r, t)
    }

    /// Create a `DualQuaternion` from a rotation unit quaternion.
    #[inline]
    pub fn from_rotation(rotation: UnitQuaternion<T>) -> Self
    where
        T: Copy + Zero + ConstZero + QuaternionMath + PartialEq,
    {
        Self(rotation, Quaternion::zero())
    }

    /// Create a `DualQuaternion` from a translation vector.
    ///```
    /// # use vqm::{DualQuaternionf32, UnitQuaternionf32, Vector3f32};
    /// let r = UnitQuaternionf32::from_roll_pitch_yaw_degrees(45.0, 60.0, 120.0);
    /// let t = Vector3f32::new(2.0, 3.0, 5.0);
    /// let dq = DualQuaternionf32::from_rotation_translation(r, t);
    /// ```
    #[inline]
    pub fn from_translation(translation: Vector3<T>) -> Self
    where
        T: Copy + One + ConstZero + ConstOne + FloatCore + MathMethods + QuaternionMath,
    {
        let half = T::one() / (T::one() + T::one());
        let dual =
            Quaternion::<T> { w: T::zero(), x: translation.x * half, y: translation.y * half, z: translation.z * half };

        Self(UnitQuaternion::one(), dual)
    }

    /// Create a `DualQuaternion` from  a rotation unit quaternion and translation vector.
    ///```
    /// # use vqm::{DualQuaternionf32, UnitQuaternionf32, Vector3f32};
    /// let r = UnitQuaternionf32::from_roll_pitch_yaw_degrees(45.0, 60.0, 120.0);
    /// let t = Vector3f32::new(2.0, 3.0, 5.0);
    /// let dq = DualQuaternionf32::from_rotation_translation(r, t);
    /// ```
    #[inline]
    pub fn from_rotation_translation(rotation: UnitQuaternion<T>, translation: Vector3<T>) -> Self
    where
        T: Copy + Zero + One + Div<Output = T> + QuaternionMath,
    {
        let half = T::one() / (T::one() + T::one());
        Self(
            rotation,
            Quaternion::<T> { w: T::zero(), x: translation.x, y: translation.y, z: translation.z }
                * rotation.q()
                * half,
        )
    }
}

impl<T> DualQuaternion<T>
where
    T: Copy + Signed + PartialOrd + QuaternionMath,
{
    /// Return true if dual quaternion is approximately other.
    pub fn is_approx(self, other: DualQuaternion<T>, epsilon: T) -> bool {
        self.0.is_approx(other.0, epsilon) && self.1.is_approx(other.1, epsilon)
    }
}

// **** One ****

impl<T> One for DualQuaternion<T>
where
    T: Copy + FloatCore + ConstZero + ConstOne + PartialEq + QuaternionMath + MathMethods,
{
    /// Unit quaternion.
    #[inline]
    fn one() -> Self {
        Self(UnitQuaternion::ONE, Quaternion::ZERO)
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

/// Const unit dual quaternion.
impl<T> ConstOne for DualQuaternion<T>
where
    T: Copy + FloatCore + ConstZero + ConstOne + PartialEq + QuaternionMath + MathMethods,
{
    const ONE: Self = Self(UnitQuaternion::ONE, Quaternion::ZERO);
}

impl<T> DualQuaternion<T>
where
    T: Copy + Zero + One + ConstZero + ConstOne + QuaternionMath + PartialEq,
{
    /// Unit quaternion.
    /// Alias for `one()` that does not require `num_traits::One`.
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        Self(UnitQuaternion::identity(), Quaternion::zero())
    }
}

impl<T> DualQuaternion<T>
where
    T: Copy + One + Signed + PartialOrd,
{
    /// Return true if dual quaternion is approximately equal the unit dual quaternion.
    pub fn is_approx_one(self, epsilon: T) -> bool {
        self.0.is_approx_one(epsilon) && self.1.is_approx_one(epsilon)
    }
}

// **** Neg ****

impl<T> Neg for DualQuaternion<T>
where
    T: QuaternionMath,
{
    type Output = Self;

    /// Negate dual quaternion.
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }
}

// **** Mul ****

impl<T> Mul<DualQuaternion<T>> for DualQuaternion<T>
where
    T: Copy + FloatCore + MathMethods + QuaternionMath,
{
    type Output = Self;

    /// Multiply two dual quaternions.
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0, self.0.q() * other.1 + self.1 * other.0.q())
    }
}

// **** MulAssign ****

impl<T> MulAssign<DualQuaternion<T>> for DualQuaternion<T>
where
    T: Copy + FloatCore + MathMethods + QuaternionMath,
{
    /// Multiply one dual quaternion by another.
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = self.mul(other);
    }
}

// **** dot ****

impl<T> DualQuaternion<T>
where
    T: QuaternionMath,
{
    /// Dual quaternion dot product.
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.0.dot(other.0)
    }
}

// **** norm_squared ****

impl<T> DualQuaternion<T>
where
    T: Copy + QuaternionMath,
{
    /// Return square of Euclidean norm.
    #[inline]
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }
}

// **** norm ****

impl<T> DualQuaternion<T>
where
    T: Copy + MathMethods + QuaternionMath,
{
    /// Return Euclidean norm.
    #[inline]
    pub fn norm(self) -> T {
        Self::norm_squared(self).sqrt()
    }
}

impl<T> DualQuaternion<T> {
    /// Return normalized form of the dual quaternion.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self
    where
        T: Copy + Zero + PartialEq + MathMethods + QuaternionMath + FloatCore,
    {
        let (r, norm_reciprocal) = UnitQuaternion::quaternion_norm_reciprocal(self.0.q());
        Self(r, self.1 * norm_reciprocal + self.0.q() * self.0.q().dot(self.1))
    }

    /// Normalize the quaternion in place, checking if the magnitude is zero.
    /// ```
    /// # use vqm::Quaternionf32;
    /// let mut q = Quaternionf32::new(0.0, 0.0, 0.0, 0.0);
    /// q.normalize_in_place();
    /// assert_eq!(Quaternionf32 { w: 0.0, x: 0.0, y: 0.0, z: 0.0 }, q);
    /// ```
    #[inline]
    pub fn normalize_in_place(&mut self) -> &mut Self
    where
        T: Copy + Zero + PartialEq + MathMethods + QuaternionMath + FloatCore,
    {
        *self = self.normalize();
        self
    }
}

impl<T> DualQuaternion<T> {
    /// Return the conjugate of the dual quaternion.
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self
    where
        T: Neg<Output = T> + QuaternionMath,
    {
        Self(self.0.conjugate(), self.1.conjugate())
    }

    /// Set dual quaternion to its conjugate.
    #[inline]
    pub fn conjugate_in_place(&mut self) -> &mut Self
    where
        T: Copy + Neg<Output = T> + QuaternionMath,
    {
        *self = self.conjugate();
        self
    }
}

impl<T> DualQuaternion<T> {
    /// Calculate the dual quaternion to the power of exponent.Compute the power of the `DualQuaternion`.
    #[must_use]
    pub fn pow(self, exponent: T) -> Self
    where
        T: Copy
            + One
            + ConstZero
            + ConstOne
            + Neg<Output = T>
            + Add<Output = T>
            + Div<Output = T>
            + IsScalar
            + FloatCore
            + MathMethods
            + QuaternionMath
            + Signed,
    {
        let one = T::one();
        let two = one + one;
        let half = one / two;
        let theta = two * self.0.scalar().acos();
        let (s, c) = (theta / two).sin_cos();
        if theta.abs() <= T::epsilon() {
            Self::from_translation(self.translation() * exponent)
        } else {
            let s0 = self.0.imaginary() / s;
            let d = self.1.scalar() * -two / s;
            let se = (self.1.imaginary() - s0 * (d * half) * c) / s;

            let (s_exp, c_exp) = (exponent * theta / two).sin_cos();
            let q_real = UnitQuaternion::from_scalar_vector_unchecked(c_exp, s0 * s_exp);
            let q_dual = Quaternion::from_scalar_vector(
                s_exp * -exponent * d / two,
                s0 * c_exp * exponent * d / two + se * s_exp,
            );
            Self::new(q_real, q_dual)
        }
    }
}

// **** Accessors ****

impl<T> DualQuaternion<T> {
    /// Returns the rotation unit quaternion of a dual-quaternion.
    pub fn rotation(self) -> UnitQuaternion<T> {
        self.0
    }

    /// Returns the translation vector of a dual-quaternion.
    pub fn translation(self) -> Vector3<T>
    where
        T: One + Neg<Output = T> + Add<Output = T> + QuaternionMath,
    {
        let two = T::one() + T::one();
        let t = self.1 * self.0.q().conjugate() * two;
        Vector3 { x: t.x, y: t.y, z: t.z }
    }
}

// **** From ****

impl<T> From<(UnitQuaternion<T>, Quaternion<T>)> for DualQuaternion<T>
where
    T: Copy + Zero + One + Div<Output = T> + QuaternionMath,
{
    /// `DualQuaternion` from unit quaternion and quaternion.
    #[inline]
    fn from(rhs: (UnitQuaternion<T>, Quaternion<T>)) -> Self {
        Self(rhs.0, rhs.1)
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<DualQuaternion<f32>>();
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::{UnitQuaternionf32, Vector3f32};
    const EPSILON: f32 = 0.000_001;

    #[test]
    fn from_rotation_translation_round_trip() {
        let r = UnitQuaternionf32::from_roll_pitch_yaw_degrees(45.0, 60.0, 120.0);
        let t = Vector3f32::new(2.0, 3.0, 5.0);

        let dq = DualQuaternionf32::from_rotation_translation(r, t);

        let r_prime = dq.rotation();
        let t_prime = dq.translation();

        assert!(r_prime.is_approx(r, EPSILON));
        assert!(t_prime.is_approx(t, EPSILON));
    }

    #[test]
    fn rotations() {
        // rotate 90 degrees about X
        let rx = UnitQuaternionf32::from_roll_pitch_yaw_degrees(90.0, 0.0, 0.0);

        // rotate 90 degrees about Y
        let ry = UnitQuaternionf32::from_roll_pitch_yaw_degrees(0.0, 90.0, 0.0);

        // rotate 90 degrees about Z
        let rz = UnitQuaternionf32::from_roll_pitch_yaw_degrees(0.0, 0.0, 90.0);

        let v_zero = Vector3f32::zero();

        let dqx = DualQuaternionf32::from_rotation_translation(rx, v_zero);
        let dqy = DualQuaternionf32::from_rotation_translation(ry, v_zero);
        let dqz = DualQuaternionf32::from_rotation_translation(rz, v_zero);

        let dq = dqy * dqz * dqx;

        let r = dq.rotation();
        let t = dq.translation();

        assert_eq!(t, v_zero);

        let rotated = r.rotate(Vector3f32::new(1.0, 0.0, 0.0));
        assert!(rotated.is_approx(Vector3f32::new(0.0, 1.0, 0.0), EPSILON));
    }

    #[test]
    fn translations() {
        let t1 = Vector3f32::new(1.0, 2.0, 3.0);
        let t2 = Vector3f32::new(1.0, -2.0, 0.0);

        let dq1 = DualQuaternionf32::from_rotation_translation(UnitQuaternionf32::identity(), t1);
        let dq2 = DualQuaternionf32::from_rotation_translation(UnitQuaternionf32::identity(), t2);
        let dq3 = dq1 * dq2;

        let r_prime = dq3.rotation();
        let t_prime = dq3.translation();

        let r_expected = UnitQuaternionf32::identity();
        let t_expected = Vector3f32::new(2.0, 0.0, 3.0);

        assert!(t_prime.is_approx(t_expected, EPSILON));

        let rotate1 = r_prime.rotate(Vector3f32::new(1.0, 1.0, 1.0));
        let rotate2 = r_expected.rotate(Vector3f32::new(1.0, 1.0, 1.0));

        assert!(rotate1.is_approx(rotate2, EPSILON));
    }

    #[test]
    fn conjugate() {
        let r = UnitQuaternionf32::from_roll_pitch_yaw_degrees(45.0, 60.0, 120.0);
        let t = Vector3f32::new(2.0, 3.0, 5.0);

        let dq = DualQuaternionf32::from_rotation_translation(r, t);
        let dq_conjugate = dq.conjugate();

        let dq_prime = dq * dq_conjugate;
        let r_prime = dq_prime.rotation();
        let t_prime = dq_prime.translation();

        assert!(t_prime.is_approx_zero(EPSILON));

        assert!(r_prime.is_approx_one(EPSILON));
    }
}
