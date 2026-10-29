#![allow(clippy::inline_always)]

use core::mem::{align_of, size_of};

#[cfg(feature = "simd")]
use core::{
    mem::transmute,
    simd::{f32x4, num::SimdFloat, simd_swizzle},
};

#[cfg(any(feature = "align", feature = "simd"))]
const _: () = assert!(size_of::<Vector3<f32>>() == 16 && align_of::<Vector3<f32>>() == 16);

#[cfg(not(any(feature = "align", feature = "simd")))]
const _: () = assert!(size_of::<Vector3<f32>>() == 12 && align_of::<Vector3<f32>>() == 4);

#[allow(unused)]
use crate::{Vector3, math_methods::MathMethods};

// **** From ****

#[cfg(feature = "simd")]
impl From<Vector3<f32>> for f32x4 {
    #[inline(always)]
    fn from(this: Vector3<f32>) -> Self {
        // SAFETY: assert f32x4 and Vector3<f32> have same size and alignment
        const _: () = assert!(size_of::<f32x4>() == size_of::<Vector3<f32>>());
        const _: () = assert!(size_of::<f32x4>() == align_of::<Vector3<f32>>());
        // The 'filler' 4th float in the SIMD lane will be whatever was in the padding (usually 0.0 if set by Default).
        unsafe { transmute(this) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x4> for Vector3<f32> {
    #[inline(always)]
    fn from(simd: f32x4) -> Self {
        // SAFETY: assert f32x4 and Vector3<f32> have same size and alignment
        const _: () = assert!(size_of::<f32x4>() == size_of::<Vector3<f32>>());
        const _: () = assert!(size_of::<f32x4>() == align_of::<Vector3<f32>>());
        unsafe { transmute(simd) }
    }
}

// **** Math ****

/// Math functions for Vector3.<br>
pub trait Vector3Math: Sized {
    fn v3_neg(this: Vector3<Self>) -> Vector3<Self>;
    fn v3_add(this: Vector3<Self>, this: Vector3<Self>) -> Vector3<Self>;
    fn v3_mul_scalar(this: Vector3<Self>, k: Self) -> Vector3<Self>;
    fn v3_div_scalar(this: Vector3<Self>, k: Self) -> Vector3<Self>;
    fn v3_mul_elementwise(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self>;
    fn v3_div_elementwise(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self>;
    fn v3_mul_add(this: Vector3<Self>, k: Self, other: Vector3<Self>) -> Vector3<Self>;
    fn v3_norm_squared(this: Vector3<Self>) -> Self;
    fn v3_is_normalized(this: Vector3<Self>) -> bool;
    fn v3_max_element(this: Vector3<Self>) -> Self;
    fn v3_min_element(this: Vector3<Self>) -> Self;
    fn v3_dot(this: Vector3<Self>, other: Vector3<Self>) -> Self;
    fn v3_cross(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self>;
    fn v3_slerp(this: Vector3<Self>, other: Vector3<Self>, t: Self) -> Vector3<Self>;
}

// **** SIMD-accelerated implementation for f32 ****

impl Vector3Math for f32 {
    #[inline(always)]
    fn v3_neg(this: Vector3<Self>) -> Vector3<Self> {
        #[cfg(feature = "simd")]
        {
            (-f32x4::from(this)).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3 { x: -this.x, y: -this.y, z: -this.z }
        }
    }

    #[inline(always)]
    fn v3_add(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);

            (this_simd + other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3 { x: this.x + other.x, y: this.y + other.y, z: this.z + other.z }
        }
    }

    #[inline(always)]
    fn v3_mul_scalar(this: Vector3<Self>, k: Self) -> Vector3<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let k_simd = f32x4::splat(k);

            (this_simd * k_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3 { x: this.x * k, y: this.y * k, z: this.z * k }
        }
    }

    #[inline(always)]
    fn v3_div_scalar(this: Vector3<Self>, k: Self) -> Vector3<Self> {
        Self::v3_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn v3_mul_elementwise(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);
            (this_simd * other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3 { x: this.x * other.x, y: this.y * other.y, z: this.z * other.z }
        }
    }

    #[inline(always)]
    fn v3_div_elementwise(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);
            (this_simd / other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3 { x: this.x / other.x, y: this.y / other.y, z: this.z / other.z }
        }
    }

    #[inline(always)]
    fn v3_mul_add(this: Vector3<Self>, k: Self, other: Vector3<Self>) -> Vector3<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);
            let k_simd = f32x4::splat(k) * f32x4::from_array([1.0, 1.0, 1.0, 0.0]);

            // This maps to the Vector Fused Multiply-Add instruction
            ((this_simd * k_simd) + other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3 { x: this.x * k + other.x, y: this.y * k + other.y, z: this.z * k + other.z }
        }
    }

    #[inline(always)]
    fn v3_norm_squared(this: Vector3<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this) * f32x4::from_array([1.0, 1.0, 1.0, 0.0]);
            (this_simd * this_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.x * this.x + this.y * this.y + this.z * this.z
        }
    }

    #[inline(always)]
    fn v3_is_normalized(this: Vector3<Self>) -> bool {
        let norm_squared = Self::v3_norm_squared(this);
        (norm_squared - 1.0).abs() < 4e-6
    }

    #[inline(always)]
    fn v3_max_element(this: Vector3<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            // repeat this.z in final lane to allow reduce_max to work correctly
            let this_simd = f32x4::from_array([this.x, this.y, this.z, this.z]);
            this_simd.reduce_max()
        }
        #[cfg(not(feature = "simd"))]
        {
            if this.x > this.y {
                if this.x > this.z { this.x } else { this.z }
            } else {
                if this.y > this.z { this.y } else { this.z }
            }
        }
    }

    #[inline(always)]
    fn v3_min_element(this: Vector3<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            // repeat this.z in final lane to allow reduce_min to work correctly
            let this_simd = f32x4::from_array([this.x, this.y, this.z, this.z]);
            this_simd.reduce_min()
        }
        #[cfg(not(feature = "simd"))]
        {
            if this.x < this.y {
                if this.x < this.z { this.x } else { this.z }
            } else {
                if this.y < this.z { this.y } else { this.z }
            }
        }
    }

    // **** dot ****
    #[inline(always)]
    fn v3_dot(this: Vector3<Self>, other: Vector3<Self>) -> Self {
        //this.x * other.x + this.y * other.y + this.z * other.z
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);

            // Multiply the vectors, masking 4 lane to 0.0
            let product = (this_simd * other_simd) * f32x4::from_array([1.0, 1.0, 1.0, 0.0]);

            product.reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.x * other.x + this.y * other.y + this.z * other.z
        }
    }

    #[inline(always)]
    fn v3_cross(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);

            // Swizzle: [y, z, x, w]
            let this_yzx = simd_swizzle!(this_simd, [1, 2, 0, 3]);
            // Swizzle: [z, x, y, w]
            let other_zxy = simd_swizzle!(other_simd, [2, 0, 1, 3]);

            // Swizzle: [z, x, y, w]
            let this_zxy = simd_swizzle!(this_simd, [2, 0, 1, 3]);
            // Swizzle: [y, z, x, w]
            let other_yzx = simd_swizzle!(other_simd, [1, 2, 0, 3]);

            // Result = (a_yzx * b_zxy) - (a_zxy * b_yzx)
            let ret_simd = this_yzx * other_zxy - this_zxy * other_yzx;

            // Transmute back to our Vector3 struct
            ret_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3 {
                x: this.y * other.z - this.z * other.y,
                y: this.z * other.x - this.x * other.z,
                z: this.x * other.y - this.y * other.x,
            }
        }
    }

    fn v3_slerp(this: Vector3<Self>, other: Vector3<Self>, t: Self) -> Vector3<Self> {
        // Calculate the cosine of the angle between the two vectors via the dot product
        let dot = this.x * other.x + this.y * other.y + this.z * other.z;
        // Clamp the dot product to avoid precision errors outside the domain of acos().
        let cos_omega = dot.clamp(-1.0, 1.0);
        // Calculate the absolute angle between the vectors (omega).
        let omega = cos_omega.acos();

        // Handle edge cases where vectors are near colinear to avoid division-by-zero.
        // If the angle is near zero, fallback to linear interpolation (Lerp).
        if omega.abs() < 1e-5 {
            let one_minus_t = 1.0 - t;
            return Vector3 {
                x: this.x * one_minus_t + other.x * t,
                y: this.y * one_minus_t + other.y * t,
                z: this.z * one_minus_t + other.z * t,
            };
        }

        // Calculate Slerp weighting factors.
        let sin_omega = omega.sin();
        let sin_omega_reciprocal = 1.0 / sin_omega;
        let (sin_t_omega, cos_t_omega) = (t * omega).sin_cos();

        let s0 = (sin_omega * cos_t_omega - cos_omega * sin_t_omega) * sin_omega_reciprocal;
        let s1 = sin_t_omega * sin_omega_reciprocal;

        // Scale and combine components.
        Vector3 { x: this.x * s0 + other.x * s1, y: this.y * s0 + other.y * s1, z: this.z * s0 + other.z * s1 }
    }
}

// **** f64 ****

impl Vector3Math for f64 {
    #[inline(always)]
    fn v3_neg(this: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: -this.x, y: -this.y, z: -this.z }
    }

    #[inline(always)]
    fn v3_add(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: this.x + other.x, y: this.y + other.y, z: this.z + other.z }
    }

    #[inline(always)]
    fn v3_mul_scalar(this: Vector3<Self>, k: Self) -> Vector3<Self> {
        Vector3 { x: this.x * k, y: this.y * k, z: this.z * k }
    }

    #[inline(always)]
    fn v3_div_scalar(this: Vector3<Self>, k: Self) -> Vector3<Self> {
        Self::v3_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn v3_mul_elementwise(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: this.x * other.x, y: this.y * other.y, z: this.z * other.z }
    }

    #[inline(always)]
    fn v3_div_elementwise(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: this.x / other.x, y: this.y / other.y, z: this.z / other.z }
    }

    #[inline(always)]
    fn v3_mul_add(this: Vector3<Self>, k: Self, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: this.x * k + other.x, y: this.y * k + other.y, z: this.z * k + other.z }
    }

    #[inline(always)]
    fn v3_norm_squared(this: Vector3<Self>) -> Self {
        this.x * this.x + this.y * this.y + this.z * this.z
    }

    #[inline(always)]
    fn v3_is_normalized(this: Vector3<Self>) -> bool {
        let norm_squared = Self::v3_norm_squared(this);
        (norm_squared - 1.0).abs() < 4e-6
    }

    #[inline(always)]
    fn v3_max_element(this: Vector3<Self>) -> Self {
        if this.x > this.y {
            if this.x > this.z { this.x } else { this.z }
        } else {
            if this.y > this.z { this.y } else { this.z }
        }
    }

    #[inline(always)]
    fn v3_min_element(this: Vector3<Self>) -> Self {
        if this.x < this.y {
            if this.x < this.z { this.x } else { this.z }
        } else {
            if this.y < this.z { this.y } else { this.z }
        }
    }

    // **** dot ****
    #[inline(always)]
    fn v3_dot(this: Vector3<Self>, other: Vector3<Self>) -> Self {
        this.x * other.x + this.y * other.y + this.z * other.z
    }

    #[inline(always)]
    fn v3_cross(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 {
            x: this.y * other.z - this.z * other.y,
            y: this.z * other.x - this.x * other.z,
            z: this.x * other.y - this.y * other.x,
        }
    }
    fn v3_slerp(this: Vector3<Self>, other: Vector3<Self>, t: Self) -> Vector3<Self> {
        // Calculate the cosine of the angle between the two vectors via the dot product
        let dot = this.x * other.x + this.y * other.y + this.z * other.z;
        // Clamp the dot product to avoid precision errors outside the domain of acos().
        let cos_omega = dot.clamp(-1.0, 1.0);
        // Calculate the absolute angle between the vectors (omega).
        let omega = cos_omega.acos();

        // Handle edge cases where vectors are near colinear to avoid division-by-zero.
        // If the angle is near zero, fallback to linear interpolation (Lerp).
        if omega.abs() < 1e-5 {
            let one_minus_t = 1.0 - t;
            return Vector3 {
                x: this.x * one_minus_t + other.x * t,
                y: this.y * one_minus_t + other.y * t,
                z: this.z * one_minus_t + other.z * t,
            };
        }

        // Calculate Slerp weighting factors.
        let sin_omega = omega.sin();
        let sin_omega_reciprocal = 1.0 / sin_omega;
        let (sin_t_omega, cos_t_omega) = (t * omega).sin_cos();

        let s0 = (sin_omega * cos_t_omega - cos_omega * sin_t_omega) * sin_omega_reciprocal;
        let s1 = sin_t_omega * sin_omega_reciprocal;

        // Scale and combine components.
        Vector3 { x: this.x * s0 + other.x * s1, y: this.y * s0 + other.y * s1, z: this.z * s0 + other.z * s1 }
    }
}

impl Vector3Math for i16 {
    #[inline(always)]
    fn v3_neg(this: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: -this.x, y: -this.y, z: -this.z }
    }

    #[inline(always)]
    fn v3_add(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: this.x + other.x, y: this.y + other.y, z: this.z + other.z }
    }

    #[inline(always)]
    fn v3_mul_scalar(this: Vector3<Self>, k: Self) -> Vector3<Self> {
        Vector3 { x: this.x * k, y: this.y * k, z: this.z * k }
    }

    #[inline(always)]
    fn v3_div_scalar(this: Vector3<Self>, k: Self) -> Vector3<Self> {
        #[allow(clippy::cast_possible_truncation)]
        Self::v3_mul_scalar(this, (1.0 / f32::from(k)) as i16)
    }

    #[inline(always)]
    fn v3_mul_elementwise(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: this.x * other.x, y: this.y * other.y, z: this.z * other.z }
    }

    #[inline(always)]
    fn v3_div_elementwise(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        #[allow(clippy::cast_possible_truncation)]
        Vector3 { x: this.x / other.x, y: this.y / other.y, z: this.z / other.z }
    }

    #[inline(always)]
    fn v3_mul_add(this: Vector3<Self>, k: Self, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: this.x * k + other.x, y: this.y * k + other.y, z: this.z * k + other.z }
    }

    #[inline(always)]
    fn v3_norm_squared(this: Vector3<Self>) -> Self {
        this.x * this.x + this.y * this.y + this.z * this.z
    }

    #[inline(always)]
    fn v3_is_normalized(this: Vector3<Self>) -> bool {
        let norm_squared = Self::v3_norm_squared(this);
        norm_squared == 1
    }

    #[inline(always)]
    fn v3_max_element(this: Vector3<Self>) -> Self {
        if this.x > this.y {
            if this.x > this.z { this.x } else { this.z }
        } else {
            if this.y > this.z { this.y } else { this.z }
        }
    }

    #[inline(always)]
    fn v3_min_element(this: Vector3<Self>) -> Self {
        if this.x < this.y {
            if this.x < this.z { this.x } else { this.z }
        } else {
            if this.y < this.z { this.y } else { this.z }
        }
    }

    #[inline(always)]
    fn v3_dot(this: Vector3<Self>, other: Vector3<Self>) -> Self {
        this.x * other.x + this.y * other.y + this.z * other.z
    }

    #[inline(always)]
    fn v3_cross(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 {
            x: this.y * other.z - this.z * other.y,
            y: this.z * other.x - this.x * other.z,
            z: this.x * other.y - this.y * other.x,
        }
    }
    fn v3_slerp(this: Vector3<Self>, other: Vector3<Self>, t: Self) -> Vector3<Self> {
        debug_assert!(false, "slerp not meaningful for i16 vectors");
        _ = other;
        _ = t;
        this
    }
}

impl Vector3Math for i32 {
    #[inline(always)]
    fn v3_neg(this: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: -this.x, y: -this.y, z: -this.z }
    }

    #[inline(always)]
    fn v3_add(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: this.x + other.x, y: this.y + other.y, z: this.z + other.z }
    }

    #[inline(always)]
    fn v3_mul_scalar(this: Vector3<Self>, k: Self) -> Vector3<Self> {
        Vector3 { x: this.x * k, y: this.y * k, z: this.z * k }
    }

    #[inline(always)]
    fn v3_div_scalar(this: Vector3<Self>, k: Self) -> Vector3<Self> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        Self::v3_mul_scalar(this, (1.0 / (k as f32)) as i32)
    }

    #[inline(always)]
    fn v3_mul_elementwise(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: this.x * other.x, y: this.y * other.y, z: this.z * other.z }
    }

    #[inline(always)]
    fn v3_div_elementwise(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        #[allow(clippy::cast_possible_truncation)]
        Vector3 { x: this.x / other.x, y: this.y / other.y, z: this.z / other.z }
    }

    #[inline(always)]
    fn v3_mul_add(this: Vector3<Self>, k: Self, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 { x: this.x * k + other.x, y: this.y * k + other.y, z: this.z * k + other.z }
    }

    #[inline(always)]
    fn v3_norm_squared(this: Vector3<Self>) -> Self {
        this.x * this.x + this.y * this.y + this.z * this.z
    }

    #[inline(always)]
    fn v3_is_normalized(this: Vector3<Self>) -> bool {
        let norm_squared = Self::v3_norm_squared(this);
        norm_squared == 1
    }

    #[inline(always)]
    fn v3_max_element(this: Vector3<Self>) -> Self {
        if this.x > this.y {
            if this.x > this.z { this.x } else { this.z }
        } else {
            if this.y > this.z { this.y } else { this.z }
        }
    }

    #[inline(always)]
    fn v3_min_element(this: Vector3<Self>) -> Self {
        if this.x < this.y {
            if this.x < this.z { this.x } else { this.z }
        } else {
            if this.y < this.z { this.y } else { this.z }
        }
    }

    #[inline(always)]
    fn v3_dot(this: Vector3<Self>, other: Vector3<Self>) -> Self {
        this.x * other.x + this.y * other.y + this.z * other.z
    }

    #[inline(always)]
    fn v3_cross(this: Vector3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 {
            x: this.y * other.z - this.z * other.y,
            y: this.z * other.x - this.x * other.z,
            z: this.x * other.y - this.y * other.x,
        }
    }
    fn v3_slerp(this: Vector3<Self>, other: Vector3<Self>, t: Self) -> Vector3<Self> {
        debug_assert!(false, "slerp not meaningful for i16 vectors");
        _ = other;
        _ = t;
        this
    }
}
