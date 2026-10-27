#![allow(clippy::inline_always)]
use core::mem::{align_of, size_of};

#[cfg(feature = "simd")]
use core::{
    mem::transmute,
    simd::{f32x4, num::SimdFloat},
};

const _: () = assert!(size_of::<Quaternion<f32>>() == 16);
const _: () = assert!(align_of::<Quaternion<f32>>() == 16);

#[allow(unused)]
use crate::{Quaternion, math_methods::MathMethods}; // for acos

// **** From ****

#[cfg(feature = "simd")]
impl From<Quaternion<f32>> for f32x4 {
    #[inline(always)]
    fn from(v: Quaternion<f32>) -> Self {
        // SAFETY: assert f32x4 and Quaternion<f32> have same size and alignment
        const _: () = assert!(size_of::<f32x4>() == size_of::<Quaternion<f32>>());
        const _: () = assert!(size_of::<f32x4>() == align_of::<Quaternion<f32>>());
        unsafe { transmute(v) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x4> for Quaternion<f32> {
    #[inline(always)]
    fn from(simd: f32x4) -> Self {
        // SAFETY: assert f32x4 and Quaternion<f32> have same size and alignment
        const _: () = assert!(size_of::<f32x4>() == size_of::<Quaternion<f32>>());
        const _: () = assert!(size_of::<f32x4>() == align_of::<Quaternion<f32>>());
        unsafe { transmute(simd) }
    }
}

// **** Math ****

/// Math functions for Quaternion.<br><br>
#[allow(missing_docs)]
pub trait QuaternionMath: Sized {
    fn q_neg(this: Quaternion<Self>) -> Quaternion<Self>;
    fn q_add(this: Quaternion<Self>, this: Quaternion<Self>) -> Quaternion<Self>;
    fn q_mul_scalar(this: Quaternion<Self>, a: Self) -> Quaternion<Self>;
    fn q_div_scalar(this: Quaternion<Self>, a: Self) -> Quaternion<Self>;
    fn q_mul_add(this: Quaternion<Self>, k: Self, other: Quaternion<Self>) -> Quaternion<Self>;
    fn q_norm_squared(this: Quaternion<Self>) -> Self;
    fn q_is_normalized(this: Quaternion<Self>, epsilon: Self) -> bool;
    fn q_mul(other: Quaternion<Self>, other: Quaternion<Self>) -> Quaternion<Self>;
    fn q_conjugate(this: Quaternion<Self>) -> Quaternion<Self>;
    fn q_dot(this: Quaternion<Self>, other: Quaternion<Self>) -> Self;
    fn q_slerp(this: Quaternion<Self>, other: Quaternion<Self>, t: Self) -> Quaternion<Self>;
}

// **** SIMD-accelerated implementation for f32 ****

impl QuaternionMath for f32 {
    #[inline(always)]
    fn q_neg(this: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            (-f32x4::from(this)).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: -this.w, x: -this.x, y: -this.y, z: -this.z }
        }
    }

    #[inline(always)]
    fn q_add(this: Quaternion<Self>, other: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);

            // Add all 4 lanes (w, x, y, z) in one cycle
            (this_simd + other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: this.w + other.w, x: this.x + other.x, y: this.y + other.y, z: this.z + other.z }
        }
    }

    #[inline(always)]
    fn q_mul_scalar(this: Quaternion<Self>, k: Self) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let k_simd = f32x4::splat(k);

            (this_simd * k_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: this.w * k, x: this.x * k, y: this.y * k, z: this.z * k }
        }
    }

    #[inline(always)]
    fn q_div_scalar(this: Quaternion<Self>, k: Self) -> Quaternion<Self> {
        Self::q_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn q_mul_add(this: Quaternion<Self>, k: Self, other: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);
            let k_simd = f32x4::splat(k);

            // This maps to the Vector Fused Multiply-Add instruction
            ((this_simd * k_simd) + other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion {
                w: this.w * k + other.w,
                x: this.x * k + other.x,
                y: this.y * k + other.y,
                z: this.z * k + other.z,
            }
        }
    }

    #[inline(always)]
    fn q_mul(this: Quaternion<Self>, other: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            Quaternion {
                w: this.w * other.w - this.x * other.x - this.y * other.y - this.z * other.z,
                x: this.w * other.x + this.x * other.w + this.y * other.z - this.z * other.y,
                y: this.w * other.y - this.x * other.z + this.y * other.w + this.z * other.x,
                z: this.w * other.z + this.x * other.y - this.y * other.x + this.z * other.w,
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion {
                w: this.w * other.w - this.x * other.x - this.y * other.y - this.z * other.z,
                x: this.w * other.x + this.x * other.w + this.y * other.z - this.z * other.y,
                y: this.w * other.y - this.x * other.z + this.y * other.w + this.z * other.x,
                z: this.w * other.z + this.x * other.y - this.y * other.x + this.z * other.w,
            }
        }
    }

    #[inline(always)]
    fn q_norm_squared(this: Quaternion<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);

            (this_simd * this_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.w * this.w + this.x * this.x + this.y * this.y + this.z * this.z
        }
    }

    #[inline(always)]
    fn q_is_normalized(this: Quaternion<Self>, epsilon: Self) -> bool {
        let norm_squared = Self::q_norm_squared(this);
        (norm_squared - 1.0).abs() < epsilon
    }

    #[inline(always)]
    fn q_conjugate(this: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            // Negate x, y, z but keep w positive
            let ret_simd = this_simd * f32x4::from_array([1.0, -1.0, -1.0, -1.0]);

            ret_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: this.w, x: -this.x, y: -this.y, z: -this.z }
        }
    }

    #[inline(always)]
    fn q_dot(this: Quaternion<Self>, other: Quaternion<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);

            (this_simd * other_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.w * other.w + this.x * other.x + this.y * other.y + this.z * other.z
        }
    }

    /// Spherical Linear Interpolation (SLERP) matching vqm's API conventions.
    /// Interpolates smoothly between `q0` and `q1` by amount `t` [0.0 to 1.0].
    #[inline(always)]
    fn q_slerp(this: Quaternion<Self>, other: Quaternion<Self>, t: Self) -> Quaternion<Self> {
        // Clamp interpolation factor to safe boundaries
        let t = t.clamp(0.0, 1.0);

        // Calculate the cosine of the angle between quaternions via their dot product
        // Hamilton convention: q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z
        let mut dot = this.w * other.w + this.x * other.x + this.y * other.y + this.z * other.z;

        // Enforce shortest-path tracking.
        // If the dot product is negative, the quaternions have opposite polarities,
        // so negate one of the quaternions to loop over the short arc.
        let mut other = other;
        if dot < 0.0 {
            dot = -dot;
            other.w = -other.w;
            other.x = -other.x;
            other.y = -other.y;
            other.z = -other.z;
        }

        // Handle identical/parallel inputs (Safe guard against division-by-zero)
        // If the inputs are incredibly close, sin(theta) approaches zero.
        // We fallback to a standard linear interpolation (LERP) and normalize.
        if dot > 0.9995 {
            let w0 = 1.0 - t;
            let w1 = t;
            return Quaternion::new(
                this.w * w0 + other.w * w1,
                this.x * w0 + other.x * w1,
                this.y * w0 + other.y * w1,
                this.z * w0 + other.z * w1,
            )
            .normalize();
        }

        // Standard SLERP calculation
        // Calculate the half-angle theta between orientations
        let theta_0 = dot.acos(); // Angle between input quaternions
        let theta = theta_0 * t; // Angle to the target interpolation step

        let (sin_theta, cos_theta) = theta.sin_cos();
        let sin_theta_0 = theta_0.sin(); // Only 1 final sin call for the denominator

        // Calculate Davis scale factors
        let s1 = sin_theta / sin_theta_0;
        let s0 = cos_theta - dot * s1;
        Quaternion::new(
            this.w * s0 + other.w * s1,
            this.x * s0 + other.x * s1,
            this.y * s0 + other.y * s1,
            this.z * s0 + other.z * s1,
        )
    }
}

// **** f64 ****

impl QuaternionMath for f64 {
    #[inline(always)]
    fn q_neg(this: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: -this.w, x: -this.x, y: -this.y, z: -this.z }
    }

    #[inline(always)]
    fn q_add(this: Quaternion<Self>, other: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: this.w + other.w, x: this.x + other.x, y: this.y + other.y, z: this.z + other.z }
    }

    #[inline(always)]
    fn q_mul_scalar(this: Quaternion<Self>, k: Self) -> Quaternion<Self> {
        Quaternion { w: this.w * k, x: this.x * k, y: this.y * k, z: this.z * k }
    }

    #[inline(always)]
    fn q_div_scalar(this: Quaternion<Self>, k: Self) -> Quaternion<Self> {
        Self::q_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn q_mul_add(this: Quaternion<Self>, k: Self, other: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion {
            w: this.w * k + other.w,
            x: this.x * k + other.x,
            y: this.y * k + other.y,
            z: this.z * k + other.z,
        }
    }

    #[inline(always)]
    fn q_mul(this: Quaternion<Self>, other: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion {
            w: this.w * other.w - this.x * other.x - this.y * other.y - this.z * other.z,
            x: this.w * other.x + this.x * other.w + this.y * other.z - this.z * other.y,
            y: this.w * other.y - this.x * other.z + this.y * other.w + this.z * other.x,
            z: this.w * other.z + this.x * other.y - this.y * other.x + this.z * other.w,
        }
    }

    #[inline(always)]
    fn q_norm_squared(this: Quaternion<Self>) -> Self {
        this.w * this.w + this.x * this.x + this.y * this.y + this.z * this.z
    }

    #[inline(always)]
    fn q_is_normalized(this: Quaternion<Self>, epsilon: Self) -> bool {
        let norm_squared = Self::q_norm_squared(this);
        (norm_squared - 1.0).abs() < epsilon
    }

    #[inline(always)]
    fn q_conjugate(this: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: this.w, x: -this.x, y: -this.y, z: -this.z }
    }

    #[inline(always)]
    fn q_dot(this: Quaternion<Self>, other: Quaternion<Self>) -> Self {
        this.w * other.w + this.x * other.x + this.y * other.y + this.z * other.z
    }
    /// Spherical Linear Interpolation (SLERP) matching vqm's API conventions.
    /// Interpolates smoothly between `q0` and `q1` by amount `t` [0.0 to 1.0].
    #[inline(always)]
    fn q_slerp(this: Quaternion<Self>, other: Quaternion<Self>, t: Self) -> Quaternion<Self> {
        // Clamp interpolation factor to safe boundaries
        let t = t.clamp(0.0, 1.0);

        // Calculate the cosine of the angle between quaternions via their dot product
        // Hamilton convention: q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z
        let mut dot = this.w * other.w + this.x * other.x + this.y * other.y + this.z * other.z;

        // Enforce shortest-path tracking.
        // If the dot product is negative, the quaternions have opposite polarities,
        // so negate one of the quaternions to loop over the short arc.
        let mut other = other;
        if dot < 0.0 {
            dot = -dot;
            other.w = -other.w;
            other.x = -other.x;
            other.y = -other.y;
            other.z = -other.z;
        }

        // Handle identical/parallel inputs (Safe guard against division-by-zero)
        // If the inputs are incredibly close, sin(theta) approaches zero.
        // We fallback to a standard linear interpolation (LERP) and normalize.
        if dot > 0.9995 {
            let w0 = 1.0 - t;
            let w1 = t;
            return Quaternion::new(
                this.w * w0 + other.w * w1,
                this.x * w0 + other.x * w1,
                this.y * w0 + other.y * w1,
                this.z * w0 + other.z * w1,
            )
            .normalize();
        }

        // Standard SLERP calculation
        // Calculate the half-angle theta between orientations
        let theta_0 = dot.acos(); // Angle between input quaternions
        let theta = theta_0 * t; // Angle to the target interpolation step

        let (sin_theta, cos_theta) = theta.sin_cos();
        let sin_theta_0 = theta_0.sin(); // Only 1 final sin call for the denominator

        // Calculate Davis scale factors
        let s1 = sin_theta / sin_theta_0;
        let s0 = cos_theta - dot * s1;
        Quaternion::new(
            this.w * s0 + other.w * s1,
            this.x * s0 + other.x * s1,
            this.y * s0 + other.y * s1,
            this.z * s0 + other.z * s1,
        )
    }
}
