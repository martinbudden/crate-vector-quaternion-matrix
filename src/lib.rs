#![cfg_attr(feature = "simd", feature(portable_simd))]
#![doc = include_str!("../README.md")]
#![no_std]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]
#![warn(unused_results)]
#![warn(clippy::pedantic)]
#![allow(clippy::inline_always)]
#![allow(clippy::return_self_not_must_use)]

pub mod math_constants;
pub mod math_methods;
pub mod sqrt_methods;

pub mod vector2d;
pub mod vector2d_math;

pub mod vector3d;
pub mod vector3d_math;

pub mod vector4d;
pub mod vector4d_math;

pub mod matrix2x2;
pub mod matrix2x2_math;

pub mod matrix3x3;
pub mod matrix3x3_math;

pub mod quaternion;
pub mod quaternion_math;

pub use math_constants::MathConstants;
pub use math_methods::TrigonometricMethods;
pub use math_methods::{sin_approx, cos_approx, sin_cos_approx};
pub use sqrt_methods::SqrtMethods;

pub use vector2d::{Vector2d, Vector2df32, Vector2df64};
pub use vector2d_math::Vector2dMath;

pub use vector3d::{Vector3d, Vector3df32, Vector3df64, Vector3di16};
pub use vector3d_math::Vector3dMath;

pub use vector4d::{Vector4d, Vector4df32, Vector4df64};
pub use vector4d_math::Vector4dMath;

pub use quaternion::{
    Quaternion, Quaternionf32, Quaternionf64, RollPitchYawf32, RollPitchYawf64, RollPitchf32, RollPitchf64,
};
pub use quaternion_math::QuaternionMath;

pub use matrix2x2::{Matrix2x2, Matrix2x2f32, Matrix2x2f64};
pub use matrix2x2_math::Matrix2x2Math;

pub use matrix3x3::{Matrix3x3, Matrix3x3f32, Matrix3x3f64};
pub use matrix3x3_math::Matrix3x3Math;
