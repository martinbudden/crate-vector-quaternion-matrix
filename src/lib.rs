#![feature(portable_simd)]
#![doc = include_str!("../README.md")]
#![no_std]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]

//mod eigen3x3;
mod math_constants;
mod math_methods;
mod sqrt_methods;

mod matrix2x2;
mod matrix2x2_math;

mod matrix3x3;
mod matrix3x3_math;

mod quaternion;
mod quaternion_math;

mod vector2d;
mod vector2d_math;
mod vector2di;

mod vector3d;
mod vector3d_math;
mod vector3di;

pub use math_constants::MathConstants;
pub use math_methods::TrigonometricMethods;
pub use sqrt_methods::SqrtMethods;

pub use matrix2x2::MatrixError;
pub use matrix2x2::{Matrix2x2, Matrix2x2f32, Matrix2x2f64};
pub use matrix2x2_math::Matrix2x2Math;

pub use matrix3x3::{Matrix3x3, Matrix3x3f32, Matrix3x3f64};
pub use matrix3x3_math::Matrix3x3Math;

pub use quaternion::{
    Quaternion, Quaternionf32, Quaternionf64, RollPitchYawf32, RollPitchYawf64, RollPitchf32, RollPitchf64,
};
pub use quaternion_math::QuaternionMath;

pub use vector2d::{Vector2d, Vector2df32, Vector2df64};
pub use vector2d_math::Vector2dMath;
pub use vector2di::{Vector2di8, Vector2di16, Vector2di32};

pub use vector3d::{Vector3d, Vector3df32, Vector3df64};
pub use vector3d_math::Vector3dMath;
pub use vector3di::{Vector3di8, Vector3di16, Vector3di32};
