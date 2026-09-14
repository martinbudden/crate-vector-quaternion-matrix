#![allow(clippy::inline_always)]
use cfg_if::cfg_if;
use core::mem::{align_of, size_of};
cfg_if! {
    if #[cfg(feature = "simd")] {
        use core::mem::transmute;
        use core::simd::{f32x2,num::SimdFloat};
    }
}

const _: () = assert!(size_of::<Vector2<f32>>() == 8);
const _: () = assert!(align_of::<Vector2<f32>>() == 8);

use crate::Vector2;

// **** From ****

#[cfg(feature = "simd")]
impl From<Vector2<f32>> for f32x2 {
    #[inline(always)]
    fn from(v: Vector2<f32>) -> Self {
        // SAFETY: assert f32x2 and Vector2<f32> have same size and alignment
        const _: () = assert!(size_of::<f32x2>() == size_of::<Vector2<f32>>());
        const _: () = assert!(size_of::<f32x2>() == align_of::<Vector2<f32>>());
        unsafe { transmute(v) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x2> for Vector2<f32> {
    #[inline(always)]
    fn from(simd: f32x2) -> Self {
        // SAFETY: assert f32x2 and Vector2<f32> have same size and alignment
        const _: () = assert!(size_of::<f32x2>() == size_of::<Vector2<f32>>());
        const _: () = assert!(size_of::<f32x2>() == align_of::<Vector2<f32>>());
        unsafe { transmute(simd) }
    }
}

// **** Math ****

/// Math functions for Vector2.<br>
pub trait Vector2Math: Sized {
    fn v2_neg(this: Vector2<Self>) -> Vector2<Self>;
    fn v2_add(this: Vector2<Self>, this: Vector2<Self>) -> Vector2<Self>;
    fn v2_mul_scalar(this: Vector2<Self>, k: Self) -> Vector2<Self>;
    fn v2_div_scalar(this: Vector2<Self>, k: Self) -> Vector2<Self>;
    fn v2_mul_elementwise(this: Vector2<Self>, other: Vector2<Self>) -> Vector2<Self>;
    fn v2_div_elementwise(this: Vector2<Self>, other: Vector2<Self>) -> Vector2<Self>;
    fn v2_mul_add(this: Vector2<Self>, k: Self, other: Vector2<Self>) -> Vector2<Self>;
    fn v2_norm_squared(this: Vector2<Self>) -> Self;
    fn v2_is_normalized(this: Vector2<Self>) -> bool;
    fn v2_max(this: Vector2<Self>) -> Self;
    fn v2_min(this: Vector2<Self>) -> Self;
    fn v2_dot(this: Vector2<Self>, other: Vector2<Self>) -> Self;
    fn v2_cross(this: Vector2<Self>, other: Vector2<Self>) -> Self;
}

// **** SIMD-accelerated implementation for f32 ****

impl Vector2Math for f32 {
    #[inline(always)]
    fn v2_neg(this: Vector2<Self>) -> Vector2<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);

            (-this_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2 { x: -this.x, y: -this.y }
        }
    }

    #[inline(always)]
    fn v2_add(this: Vector2<Self>, other: Vector2<Self>) -> Vector2<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);
            let other_simd = f32x2::from(other);

            (this_simd + other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2 { x: this.x + other.x, y: this.y + other.y }
        }
    }

    #[inline(always)]
    fn v2_mul_scalar(this: Vector2<Self>, k: Self) -> Vector2<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);
            let k_simd = f32x2::splat(k);

            (this_simd * k_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2 { x: this.x * k, y: this.y * k }
        }
    }

    #[inline(always)]
    fn v2_div_scalar(this: Vector2<Self>, k: Self) -> Vector2<Self> {
        Self::v2_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn v2_mul_elementwise(this: Vector2<Self>, other: Vector2<Self>) -> Vector2<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);
            let other_simd = f32x2::from(other);
            (this_simd * other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2 { x: this.x * other.x, y: this.y * other.y }
        }
    }

    #[inline(always)]
    fn v2_div_elementwise(this: Vector2<Self>, other: Vector2<Self>) -> Vector2<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);
            let other_simd = f32x2::from(other);
            (this_simd / other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2 { x: this.x / other.x, y: this.y / other.y }
        }
    }

    #[inline(always)]
    fn v2_mul_add(this: Vector2<Self>, k: Self, other: Vector2<Self>) -> Vector2<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);
            let other_simd = f32x2::from(other);
            let k_simd = f32x2::splat(k);

            // This maps to the Vector Fused Multiply-Add instruction
            ((this_simd * k_simd) + other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2 { x: this.x * k + other.x, y: this.y * k + other.y }
        }
    }

    #[inline(always)]
    fn v2_norm_squared(this: Vector2<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);

            (this_simd * this_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.x * this.x + this.y * this.y
        }
    }

    #[inline(always)]
    fn v2_is_normalized(this: Vector2<Self>) -> bool {
        let norm_squared = Self::v2_norm_squared(this);
        (norm_squared - 1.0).abs() < 4e-6
    }

    #[inline(always)]
    fn v2_max(this: Vector2<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);
            this_simd.reduce_max()
        }
        #[cfg(not(feature = "simd"))]
        {
            if this.x > this.y { this.x } else { this.y }
        }
    }

    #[inline(always)]
    fn v2_min(this: Vector2<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);
            this_simd.reduce_min()
        }
        #[cfg(not(feature = "simd"))]
        {
            if this.x < this.y { this.x } else { this.y }
        }
    }

    // **** dot ****
    #[inline(always)]
    fn v2_dot(this: Vector2<Self>, other: Vector2<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);
            let other_simd = f32x2::from(other);

            (this_simd * other_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.x * other.x + this.y * other.y
        }
    }

    #[inline(always)]
    fn v2_cross(this: Vector2<Self>, other: Vector2<Self>) -> Self {
        this.x * other.y - this.y * other.x
    }
}

// **** f64 ****

impl Vector2Math for f64 {
    #[inline(always)]
    fn v2_neg(this: Vector2<Self>) -> Vector2<Self> {
        Vector2 { x: -this.x, y: -this.y }
    }

    #[inline(always)]
    fn v2_add(this: Vector2<Self>, other: Vector2<Self>) -> Vector2<Self> {
        Vector2 { x: this.x + other.x, y: this.y + other.y }
    }

    #[inline(always)]
    fn v2_mul_scalar(this: Vector2<Self>, k: Self) -> Vector2<Self> {
        Vector2 { x: this.x * k, y: this.y * k }
    }

    #[inline(always)]
    fn v2_div_scalar(this: Vector2<Self>, k: Self) -> Vector2<Self> {
        Self::v2_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn v2_mul_elementwise(this: Vector2<Self>, other: Vector2<Self>) -> Vector2<Self> {
        Vector2 { x: this.x * other.x, y: this.y * other.y }
    }

    #[inline(always)]
    fn v2_div_elementwise(this: Vector2<Self>, other: Vector2<Self>) -> Vector2<Self> {
        Vector2 { x: this.x / other.x, y: this.y / other.y }
    }

    #[inline(always)]
    fn v2_mul_add(this: Vector2<Self>, k: Self, other: Vector2<Self>) -> Vector2<Self> {
        Vector2 { x: this.x * k + other.x, y: this.y * k + other.y }
    }

    #[inline(always)]
    fn v2_norm_squared(this: Vector2<Self>) -> Self {
        this.x * this.x + this.y * this.y
    }

    #[inline(always)]
    fn v2_is_normalized(q: Vector2<Self>) -> bool {
        let norm_squared = Self::v2_norm_squared(q);
        (norm_squared - 1.0).abs() < 4e-6
    }

    #[inline(always)]
    fn v2_max(this: Vector2<Self>) -> Self {
        if this.x > this.y { this.x } else { this.y }
    }

    #[inline(always)]
    fn v2_min(this: Vector2<Self>) -> Self {
        if this.x < this.y { this.x } else { this.y }
    }

    #[inline(always)]
    fn v2_dot(this: Vector2<Self>, other: Vector2<Self>) -> Self {
        this.x * other.x + this.y * other.y
    }

    #[inline(always)]
    fn v2_cross(this: Vector2<Self>, other: Vector2<Self>) -> Self {
        this.x * other.y - this.y * other.x
    }
}
