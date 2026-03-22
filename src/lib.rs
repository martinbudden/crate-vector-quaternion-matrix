#![no_std]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]

mod eigen3x3;
mod math_constants;
mod math_methods;
mod matrix2x2;
mod matrix3x3;
mod quaternion;
mod vector2d;
mod vector3d;

pub use math_constants::MathConstants;
pub use math_methods::MathMethods;

pub use eigen3x3::{EigenResult, EigenResult3x3, EigenResult3x3f32, EigenResult3x3f64};
pub use matrix2x2::{Matrix2x2, Matrix2x2f32, Matrix2x2f64};
pub use matrix3x3::{Matrix3x3, Matrix3x3f32, Matrix3x3f64};

pub use quaternion::{
    Quaternion, Quaternionf32, Quaternionf64, RollPitchYawf32, RollPitchYawf64, RollPitchf32, RollPitchf64,
};

pub use vector2d::{Vector2d, Vector2df32, Vector2df64, Vector2di8, Vector2di16, Vector2di32};
pub use vector3d::{Vector3d, Vector3df32, Vector3df64, Vector3di8, Vector3di16, Vector3di32};
