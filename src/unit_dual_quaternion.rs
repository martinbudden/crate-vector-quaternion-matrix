use super::{MathMethods, Quaternion, QuaternionMath, UnitQuaternion, Vector3};
use core::ops::{Add, Div, Mul, MulAssign, Neg};
use num_traits::{ConstOne, ConstZero, One, Signed, Zero, float::FloatCore};

/// `UnitDualQuaternion` of `f32` values<br>
pub type UnitDualQuaternionf32 = UnitDualQuaternion<f32>;
/// `UnitDualQuaternion` of `f64` values<br><br>
pub type UnitDualQuaternionf64 = UnitDualQuaternion<f64>;

// **** Define ****

/// `UnitDualQuaternion<T>`: quaternion type `T`.<br>
/// Aliases `UnitDualQuaternionf32` and `UnitDualQuaternionf64` are provided.<br><br>
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitDualQuaternion<T> {
    /// Real (rotation) component.
    real: UnitQuaternion<T>,
    /// Dual component.
    dual: Quaternion<T>,
}

// **** Default ****

/// Default `UnitDualQuaternion`.
impl<T> Default for UnitDualQuaternion<T>
where
    T: Copy + Zero + One + ConstZero + ConstOne + PartialEq + FloatCore + MathMethods + QuaternionMath,
{
    #[inline]
    fn default() -> Self {
        Self { real: UnitQuaternion::one(), dual: Quaternion::zero() }
    }
}

// **** New ****

impl<T> UnitDualQuaternion<T> {
    /// Create a unit dual quaternion, from a unit quaternion and a quaternion,
    /// enforcing the Study condition, ie
    /// `|real · real - 1|  < epsilon` and `|real · dual| < epsilon`.
    #[inline]
    pub fn new(real: UnitQuaternion<T>, dual: Quaternion<T>) -> Self
    where
        T: Copy + Neg<Output = T> + QuaternionMath,
    {
        let dual = dual - real.q() * real.q().dot(dual);
        Self { real, dual }
    }

    /// Create a unit dual quaternion, from a unit quaternion and a quaternion.
    /// Returns `None` unless the Study condition is satisfied to a tolerance of `epsilon`.
    #[inline]
    pub fn try_new(real: UnitQuaternion<T>, dual: Quaternion<T>, epsilon: T) -> Option<Self>
    where
        T: Copy + Zero + Neg<Output = T> + PartialOrd + QuaternionMath,
    {
        if real.q().dot(dual) < epsilon { Some(Self { real, dual }) } else { None }
    }

    /// Create a `UnitDualQuaternion` from a rotation unit quaternion.
    ///```
    /// # use vqm::{UnitDualQuaternionf32, UnitQuaternionf32};
    /// let rotation = UnitQuaternionf32::from_roll_pitch_yaw_degrees(45.0, 60.0, 120.0);
    /// let dq = UnitDualQuaternionf32::from_rotation(rotation);
    /// ```
    #[inline]
    pub fn from_rotation(rotation: UnitQuaternion<T>) -> Self
    where
        T: Copy + Zero + ConstZero + QuaternionMath + PartialEq,
    {
        Self { real: rotation, dual: Quaternion::zero() }
    }

    /// Create a `UnitDualQuaternion` from a translation vector.
    ///```
    /// # use vqm::{UnitDualQuaternionf32, Vector3f32};
    /// let translation = Vector3f32::new(2.0, 3.0, 5.0);
    /// let dq = UnitDualQuaternionf32::from_translation(translation);
    /// ```
    #[inline]
    pub fn from_translation(translation: Vector3<T>) -> Self
    where
        T: Copy + One + ConstZero + ConstOne + FloatCore + MathMethods + QuaternionMath,
    {
        let half = T::one() / (T::one() + T::one());
        let dual =
            Quaternion::<T> { w: T::zero(), x: translation.x * half, y: translation.y * half, z: translation.z * half };

        Self { real: UnitQuaternion::one(), dual }
    }

    /// Create a `UnitDualQuaternion` from  a rotation unit quaternion and translation vector.
    ///```
    /// # use vqm::{UnitDualQuaternionf32, UnitQuaternionf32, Vector3f32};
    /// let r = UnitQuaternionf32::from_roll_pitch_yaw_degrees(45.0, 60.0, 120.0);
    /// let t = Vector3f32::new(2.0, 3.0, 5.0);
    /// let dq = UnitDualQuaternionf32::from_rotation_translation(r, t);
    /// ```
    #[inline]
    pub fn from_rotation_translation(rotation: UnitQuaternion<T>, translation: Vector3<T>) -> Self
    where
        T: Copy + Zero + One + Div<Output = T> + QuaternionMath,
    {
        let half = T::one() / (T::one() + T::one());
        Self {
            real: rotation,
            dual: Quaternion::<T> { w: T::zero(), x: translation.x, y: translation.y, z: translation.z }
                * rotation.q()
                * half,
        }
    }
}

impl<T> UnitDualQuaternion<T> {
    /// Return real component.
    #[inline]
    pub fn real(self) -> UnitQuaternion<T> {
        self.real
    }

    /// Returns the rotation unit quaternion of a dual-quaternion.
    pub fn rotation(self) -> UnitQuaternion<T> {
        self.real
    }

    /// Return dual component.
    #[inline]
    pub fn dual(self) -> Quaternion<T> {
        self.dual
    }

    /// Returns the translation vector of a dual-quaternion.
    pub fn translation(self) -> Vector3<T>
    where
        T: One + Neg<Output = T> + Add<Output = T> + QuaternionMath,
    {
        let two = T::one() + T::one();
        let t = self.dual * self.real.q().conjugate() * two;
        Vector3 { x: t.x, y: t.y, z: t.z }
    }
}

impl<T> UnitDualQuaternion<T> {
    /// Returns true if unit dual quaternion satisfies the Study condition, ie
    /// `|real · real - 1|  < epsilon` and `|real · dual| < epsilon`.
    pub fn is_valid(self, epsilon: T) -> bool
    where
        T: Copy + Signed + PartialOrd + QuaternionMath,
    {
        (self.real.q().dot(self.real.q()) - T::one()).abs() < epsilon
            && ((self.real.q().dot(self.dual)).abs() < epsilon)
    }

    /// Return true if dual quaternion is approximately equal to other.
    pub fn is_approx(self, other: UnitDualQuaternion<T>, epsilon: T) -> bool
    where
        T: Copy + Signed + PartialOrd + QuaternionMath,
    {
        self.real.is_approx(other.real, epsilon) && self.dual.is_approx(other.dual, epsilon)
    }

    /// Return true if self and other represent the same transformations.
    pub fn approx_eq_transform(&self, other: Self, epsilon: T) -> bool
    where
        T: Copy + FloatCore + MathMethods + QuaternionMath + Signed,
    {
        self.real.orientation_approx_eq(other.real, epsilon)
            && self.translation().is_approx(other.translation(), epsilon)
    }
}

impl<T> UnitDualQuaternion<T>
where
    T: FloatCore,
{
    /// Return true if all the dual quaternion's components are finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.real.is_finite() && self.dual.is_finite()
    }
}

// **** One ****

impl<T> One for UnitDualQuaternion<T>
where
    T: Copy + FloatCore + ConstZero + ConstOne + PartialEq + QuaternionMath + MathMethods,
{
    /// Unit quaternion.
    #[inline]
    fn one() -> Self {
        Self { real: UnitQuaternion::ONE, dual: Quaternion::ZERO }
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

/// Const unit dual quaternion.
impl<T> ConstOne for UnitDualQuaternion<T>
where
    T: Copy + FloatCore + ConstZero + ConstOne + PartialEq + QuaternionMath + MathMethods,
{
    const ONE: Self = Self { real: UnitQuaternion::ONE, dual: Quaternion::ZERO };
}

impl<T> UnitDualQuaternion<T>
where
    T: Copy + Zero + One + ConstZero + ConstOne + QuaternionMath + PartialEq,
{
    /// Unit quaternion.
    /// Alias for `one()` that does not require `num_traits::One`.
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        Self { real: UnitQuaternion::identity(), dual: Quaternion::zero() }
    }
}

impl<T> UnitDualQuaternion<T>
where
    T: Copy + One + Signed + PartialOrd,
{
    /// Return true if dual quaternion is approximately equal the unit dual quaternion.
    pub fn is_approx_one(self, epsilon: T) -> bool {
        self.real.is_approx_one(epsilon) && self.dual.is_approx_zero(epsilon)
    }
}

// **** Neg ****

impl<T> Neg for UnitDualQuaternion<T>
where
    T: QuaternionMath,
{
    type Output = Self;

    /// Negate dual quaternion.
    #[inline]
    fn neg(self) -> Self {
        Self { real: -self.real, dual: -self.dual }
    }
}

// **** Mul ****

impl<T> Mul<UnitDualQuaternion<T>> for UnitDualQuaternion<T>
where
    T: Copy + FloatCore + MathMethods + QuaternionMath,
{
    type Output = Self;

    /// Multiply two dual quaternions.
    /// For computational efficiency the result is not renormalized:
    /// The caller should `renormalize` after accumulated floating point operations.
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self { real: self.real * other.real, dual: self.real.q() * other.dual + self.dual * other.real.q() }
    }
}

// **** MulAssign ****

impl<T> MulAssign<UnitDualQuaternion<T>> for UnitDualQuaternion<T>
where
    T: Copy + FloatCore + MathMethods + QuaternionMath,
{
    /// Multiply one dual quaternion by another.
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = self.mul(other);
    }
}

// **** approx_eq ****

impl<T> UnitDualQuaternion<T> {
    /// Compare two unit dual quaternions componentwise with a tolerance.
    #[inline]
    pub fn approx_eq(&self, other: Self, epsilon: T) -> bool
    where
        T: Copy + Signed + PartialOrd,
    {
        self.real.approx_eq(other.real, epsilon) && self.dual.approx_eq(other.dual, epsilon)
    }
}

// **** dot ****

impl<T> UnitDualQuaternion<T>
where
    T: QuaternionMath,
{
    /// Quaternion dot product of real component.
    #[inline]
    pub fn real_dot(self, other: Self) -> T {
        self.real.dot(other.real)
    }
}

impl<T> UnitDualQuaternion<T> {
    /// Return renormalized unit dual quaternion, for dealing with rounding errors.
    #[inline]
    #[must_use]
    pub fn renormalize(self) -> Self
    where
        T: Copy + Zero + PartialEq + MathMethods + QuaternionMath + FloatCore,
    {
        let (real, norm_reciprocal) = UnitQuaternion::quaternion_norm_reciprocal(self.real.q());

        let dual = self.dual * norm_reciprocal;
        let dual = dual - real.q() * real.q().dot(dual);

        Self { real, dual }
    }

    /// Return renormalized unit dual quaternion in-place, for dealing with rounding errors.
    #[inline]
    pub fn renormalize_in_place(&mut self) -> &mut Self
    where
        T: Copy + Zero + PartialEq + MathMethods + QuaternionMath + FloatCore,
    {
        *self = self.renormalize();
        self
    }
}

impl<T> UnitDualQuaternion<T> {
    /// Return the quaternion conjugate of the dual quaternion.
    ///
    /// Applies quaternion conjugation independently to the real and dual components.
    /// This operation is not the dual quaternion inverse.
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self
    where
        T: Neg<Output = T> + QuaternionMath,
    {
        Self { real: self.real.conjugate(), dual: self.dual.conjugate() }
    }

    /// Quaternion conjugate of the dual quaternion, in-place.
    ///
    /// Applies quaternion conjugation independently to the real and dual components.
    /// This operation is not the dual quaternion inverse.
    #[inline]
    pub fn conjugate_in_place(&mut self) -> &mut Self
    where
        T: Copy + Neg<Output = T> + QuaternionMath,
    {
        *self = self.conjugate();
        self
    }

    /// Return the dual quaternion inverse, ie quantity that satisfies: `identity = dq * dq.inverse()`.
    #[must_use]
    pub fn inverse(self) -> Self
    where
        T: Copy + Neg<Output = T> + QuaternionMath,
    {
        let real = self.real.conjugate();

        let dual = -(real.q() * self.dual * real.q());

        Self { real, dual }
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<UnitDualQuaternion<f32>>();
    }
}
