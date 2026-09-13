use core::ops::{Mul, Neg};
use num_traits::float::FloatCore;
#[cfg(feature = "serde")]
use {
    postcard::experimental::max_size::MaxSize,
    sequential_storage::map::PostcardValue,
    serde::{Deserialize, Serialize},
};

use crate::{MathConstants, MathMethods, Quaternion, Vector2, Vector3};

/// `RollPitchYaw` struct `{ roll: f32, pitch: f32, yaw: f32 }`<br>
pub type RollPitchYawf32 = RollPitchYaw<f32>;
/// `RollPitchYaw` struct `{ roll: f64, pitch: f64, yaw: f64 }`<br>
pub type RollPitchYawf64 = RollPitchYaw<f64>;

/// `RollPitch` struct `{ roll: f32, pitch: f32 }`<br>
pub type RollPitchf32 = RollPitch<f32>;
/// `RollPitch` struct `{ roll: f64, pitch: f64 }`<br><br>
pub type RollPitchf64 = RollPitch<f64>;

/// Roll and Pitch bundled for convenience.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "std", derive(derive_more::Display))]
#[cfg_attr(feature = "std", display("RP{{roll:{roll}, pitch:{pitch}}}"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize, MaxSize))]
#[allow(missing_docs)]
pub struct RollPitch<T> {
    pub roll: T,
    pub pitch: T,
}

#[cfg(feature = "serde")]
impl<T> PostcardValue<'_> for RollPitch<T> where T: Serialize + for<'de> Deserialize<'de> {}

impl<T> RollPitch<T>
where
    T: Copy,
{
    /// Create a `RollPitch`.
    #[inline]
    pub const fn new(roll: T, pitch: T) -> Self {
        Self { roll, pitch }
    }

    /// Create a `RollPitch` from a `Vector2` assuming the North East Down (NED) convention.
    #[inline]
    pub fn from_vector_ned(v: Vector2<T>) -> Self {
        Self { roll: v.y, pitch: v.x }
    }
}

impl<T> RollPitch<T>
where
    T: Copy + Mul<Output = T> + MathConstants,
{
    #[inline]
    #[must_use]
    pub fn to_degrees(self) -> Self {
        Self { roll: self.roll * T::RADIANS_TO_DEGREES, pitch: self.pitch * T::RADIANS_TO_DEGREES }
    }
    #[inline]
    #[must_use]
    pub fn to_radians(self) -> Self {
        Self { roll: self.roll * T::DEGREES_TO_RADIANS, pitch: self.pitch * T::DEGREES_TO_RADIANS }
    }
}

impl<T> RollPitch<T>
where
    T: Copy + FloatCore,
{
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self { roll: self.roll.abs(), pitch: self.pitch.abs() }
    }
    #[inline]
    #[must_use]
    pub fn clamp(self, min: T, max: T) -> Self {
        Self { roll: self.roll.clamp(min, max), pitch: self.pitch.clamp(min, max) }
    }
}

impl<T> From<RollPitch<T>> for Quaternion<T>
where
    T: Copy + MathMethods + FloatCore,
{
    #[inline]
    fn from(angles: RollPitch<T>) -> Self {
        Quaternion::from_roll_pitch_radians(angles.roll, angles.pitch)
    }
}

/// Roll, Pitch, and Yaw bundled for convenience.<br><br>
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "std", derive(derive_more::Display))]
#[cfg_attr(feature = "std", display("RPY{{roll:{roll}, pitch:{pitch}, yaw:{yaw}}}"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize, MaxSize))]
#[allow(missing_docs)]
pub struct RollPitchYaw<T> {
    pub roll: T,
    pub pitch: T,
    pub yaw: T,
}

#[cfg(feature = "serde")]
impl<T> PostcardValue<'_> for RollPitchYaw<T> where T: Serialize + for<'de> Deserialize<'de> {}

impl<T> RollPitchYaw<T>
where
    T: Copy + Neg<Output = T>,
{
    /// Create a `RollPitchYaw`.
    #[inline]
    pub const fn new(roll: T, pitch: T, yaw: T) -> Self {
        Self { roll, pitch, yaw }
    }

    /// Create a `RollPitchYaw` from a `Vector3` assuming the North East Down (NED) convention.
    #[inline]
    pub fn from_vector_ned(v: Vector3<T>) -> Self {
        Self { roll: v.y, pitch: v.x, yaw: -v.z }
    }
}

impl<T> RollPitchYaw<T>
where
    T: Copy + Mul<Output = T> + MathConstants,
{
    #[inline]
    #[must_use]
    pub fn to_degrees(self) -> Self {
        Self {
            roll: self.roll * T::RADIANS_TO_DEGREES,
            pitch: self.pitch * T::RADIANS_TO_DEGREES,
            yaw: self.yaw * T::RADIANS_TO_DEGREES,
        }
    }
    #[inline]
    #[must_use]
    pub fn to_radians(self) -> Self {
        Self {
            roll: self.roll * T::DEGREES_TO_RADIANS,
            pitch: self.pitch * T::DEGREES_TO_RADIANS,
            yaw: self.yaw * T::RADIANS_TO_DEGREES,
        }
    }
}

impl<T> RollPitchYaw<T>
where
    T: Copy + FloatCore,
{
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self { roll: self.roll.abs(), pitch: self.pitch.abs(), yaw: self.yaw.abs() }
    }
    #[inline]
    #[must_use]
    pub fn clamp(self, min: T, max: T) -> Self {
        Self { roll: self.roll.clamp(min, max), pitch: self.pitch.clamp(min, max), yaw: self.yaw.clamp(min, max) }
    }
}

impl<T> From<RollPitchYaw<T>> for Quaternion<T>
where
    T: Copy + MathMethods + FloatCore,
{
    #[inline]
    fn from(angles: RollPitchYaw<T>) -> Self {
        Quaternion::from_roll_pitch_yaw_radians(angles.roll, angles.pitch, angles.yaw)
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}
    #[cfg(feature = "serde")]
    fn is_config<T: Serialize + MaxSize + for<'a> Deserialize<'a> + for<'a> PostcardValue<'a>>() {}

    #[test]
    fn normal_types() {
        is_full::<RollPitchf32>();
        is_full::<RollPitchYawf32>();
        #[cfg(feature = "serde")]
        is_config::<RollPitchf32>();
        #[cfg(feature = "serde")]
        is_config::<RollPitchYawf32>();
    }
}
