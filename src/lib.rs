#![cfg_attr(feature = "uom", doc = include_str!("../README.md"))]
#![cfg_attr(feature = "simd", feature(portable_simd, min_specialization))]
#![no_std]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
//#![deny(missing_docs)]
#![deny(
    missing_copy_implementations,
    missing_debug_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unused_must_use,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications,
    unused_results
)]
#![warn(unused_results)]
#![warn(clippy::pedantic, clippy::doc_paragraphs_missing_punctuation)]

mod math_approximations;
mod math_constants;
mod math_methods;
mod sqrt_approximations;

mod vector2;
mod vector2_math;

mod vector3;
mod vector3_buf;
mod vector3_math;

mod vector4;
mod vector4_math;

mod matrix2x2;
mod matrix2x2_math;

mod matrix3x3;
mod matrix3x3_math;

mod matrix3x3_m2x2;
mod matrix3x3_m3x3;

mod matrix4x4;
mod matrix4x4_math;

mod quaternion;
mod quaternion_math;
mod roll_pitch_yaw;

pub use math_constants::MathConstants;
pub use math_methods::MathMethods;

// The trigonometric approximation functions need to be `pub` for benchmarking, but are not documented for general use.
#[doc(hidden)]
pub use math_approximations::{cos_approx_f32, sin_approx_f32, sin_cos_approx_f32}; // needed for benchmarks

pub use vector2::{Vector2, Vector2f32, Vector2f64};

pub use vector3::{Vector3, Vector3f32, Vector3f64};
pub use vector3_buf::SliceTooShortError;

pub use vector4::{Vector4, Vector4f32, Vector4f64};

pub use quaternion::{Quaternion, Quaternionf32, Quaternionf64};
pub use quaternion_math::QuaternionMath;
pub use roll_pitch_yaw::{RollPitch, RollPitchYaw, RollPitchYawf32, RollPitchYawf64, RollPitchf32, RollPitchf64};

pub use matrix2x2::{Matrix2x2, Matrix2x2f32, Matrix2x2f64};
pub use matrix2x2_math::Matrix2x2Math;

pub use matrix3x3::{Matrix3x3, Matrix3x3f32, Matrix3x3f64};
pub use matrix3x3_math::Matrix3x3Math;

pub use matrix3x3_m2x2::{Matrix3x3xM2x2, Matrix3x3xM2x2f32, Matrix3x3xM2x2f64};
pub use matrix3x3_m3x3::{Matrix3x3xM3x3, Matrix3x3xM3x3f32, Matrix3x3xM3x3f64};

pub use matrix4x4::{Matrix4x4, Matrix4x4f32, Matrix4x4f64};
pub use matrix4x4_math::Matrix4x4Math;
