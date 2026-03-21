#![no_std]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]

mod math_methods;
mod math_constants;
mod matrix3x3;
mod quaternion;
mod vector3d;

pub use math_methods::MathMethods;
pub use math_constants::MathConstants;

pub use matrix3x3::Matrix3x3;
pub use matrix3x3::Matrix3x3f32;
pub use matrix3x3::Matrix3x3f64;

pub use quaternion::Quaternion;
pub use quaternion::Quaternionf32;
pub use quaternion::Quaternionf64;
pub use quaternion::RollPitchYawf32;
pub use quaternion::RollPitchYawf64;
pub use quaternion::RollPitchf32;
pub use quaternion::RollPitchf64;

pub use vector3d::Vector3d;
pub use vector3d::Vector3df32;
pub use vector3d::Vector3df64;
pub use vector3d::Vector3di8;
pub use vector3d::Vector3di16;
pub use vector3d::Vector3di32;
