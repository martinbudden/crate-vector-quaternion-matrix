use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "simd")] {
        use core::{mem::transmute};
        use core::simd::{f32x2,num::SimdFloat};
    }
}

const _: () = assert!(core::mem::size_of::<Vector2d<f32>>() == 8);
const _: () = assert!(core::mem::align_of::<Vector2d<f32>>() == 8);

use crate::{SqrtMethods, Vector2d};

// **** From ****

#[cfg(feature = "simd")]
impl From<Vector2d<f32>> for f32x2 {
    #[inline(always)]
    fn from(v: Vector2d<f32>) -> Self {
        // SAFETY: assert f32x2 and Vector2d<f32> have same size and alignment
        const _: () = assert!(core::mem::size_of::<f32x2>() == core::mem::size_of::<Vector2d<f32>>());
        const _: () = assert!(core::mem::size_of::<f32x2>() == core::mem::align_of::<Vector2d<f32>>());
        unsafe { transmute(v) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x2> for Vector2d<f32> {
    #[inline(always)]
    fn from(simd: f32x2) -> Self {
        // SAFETY: assert f32x2 and Vector2d<f32> have same size and alignment
        const _: () = assert!(core::mem::size_of::<f32x2>() == core::mem::size_of::<Vector2d<f32>>());
        const _: () = assert!(core::mem::size_of::<f32x2>() == core::mem::align_of::<Vector2d<f32>>());
        unsafe { transmute(simd) }
    }
}

// **** Math ****

/// Math functions for Vector2d, using **SIMD** accelerations for `f32`.<br>
pub trait Vector2dMath: Sized {
    fn v2_reciprocal(self) -> Self;
    fn v2_neg(this: Vector2d<Self>) -> Vector2d<Self>;
    fn v2_add(this: Vector2d<Self>, this: Vector2d<Self>) -> Vector2d<Self>;
    fn v2_mul_scalar(this: Vector2d<Self>, k: Self) -> Vector2d<Self>;
    fn v2_div_scalar(this: Vector2d<Self>, k: Self) -> Vector2d<Self>;
    fn v2_mul_add(this: Vector2d<Self>, k: Self, other: Vector2d<Self>) -> Vector2d<Self>;
    fn v2_norm_squared(this: Vector2d<Self>) -> Self;
    fn v2_normalize(this: Vector2d<Self>) -> Vector2d<Self>;
    fn v2_is_normalized(this: Vector2d<Self>) -> bool;
    fn v2_max(this: Vector2d<Self>) -> Self;
    fn v2_min(this: Vector2d<Self>) -> Self;
    fn v2_dot(this: Vector2d<Self>, other: Vector2d<Self>) -> Self;
    fn v2_cross(this: Vector2d<Self>, other: Vector2d<Self>) -> Self;
}

// **** SIMD-accelerated implementation for f32 ****

impl Vector2dMath for f32 {
    #[inline(always)]
    fn v2_reciprocal(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn v2_neg(this: Vector2d<Self>) -> Vector2d<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);

            (-this_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2d { x: -this.x, y: -this.y }
        }
    }

    #[inline(always)]
    fn v2_add(this: Vector2d<Self>, other: Vector2d<Self>) -> Vector2d<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);
            let other_simd = f32x2::from(other);

            (this_simd + other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2d { x: this.x + other.x, y: this.y + other.y }
        }
    }

    #[inline(always)]
    fn v2_mul_scalar(this: Vector2d<Self>, k: Self) -> Vector2d<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x2::from(this);
            let k_simd = f32x2::splat(k);

            (this_simd * k_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2d { x: this.x * k, y: this.y * k }
        }
    }

    #[inline(always)]
    fn v2_div_scalar(this: Vector2d<Self>, k: Self) -> Vector2d<Self> {
        Self::v2_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn v2_mul_add(this: Vector2d<Self>, k: Self, other: Vector2d<Self>) -> Vector2d<Self> {
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
            Vector2d { x: this.x * k + other.x, y: this.y * k + other.y }
        }
    }

    #[inline(always)]
    fn v2_norm_squared(this: Vector2d<Self>) -> Self {
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
    fn v2_normalize(this: Vector2d<Self>) -> Vector2d<Self> {
        #[cfg(feature = "simd")]
        {
            let norm_squared = Self::v2_norm_squared(this);
            // If norm_squared is zero, then this must be zero (default) vector
            if norm_squared == 0.0 {
                return Vector2d::default();
            }

            let this_simd = f32x2::from(this);
            let norm_reciprocal = norm_squared.sqrt_reciprocal(); // Uses hardware vrsqrt
            let scale = f32x2::splat(norm_reciprocal);

            (this_simd * scale).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let norm_squared = Self::v2_norm_squared(this);
            if norm_squared == 0.0 {
                return Vector2d::default();
            }
            let norm_reciprocal = norm_squared.sqrt_reciprocal();
            Vector2d { x: this.x * norm_reciprocal, y: this.y * norm_reciprocal }
        }
    }

    #[inline(always)]
    fn v2_is_normalized(this: Vector2d<Self>) -> bool {
        let norm_squared = Self::v2_norm_squared(this);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn v2_max(this: Vector2d<Self>) -> Self {
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
    fn v2_min(this: Vector2d<Self>) -> Self {
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
    fn v2_dot(this: Vector2d<Self>, other: Vector2d<Self>) -> Self {
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
    fn v2_cross(this: Vector2d<Self>, other: Vector2d<Self>) -> Self {
        this.x * other.y - this.y * other.x
    }
}

// **** f64 ****

impl Vector2dMath for f64 {
    #[inline(always)]
    fn v2_reciprocal(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn v2_neg(this: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: -this.x, y: -this.y }
    }

    #[inline(always)]
    fn v2_add(this: Vector2d<Self>, other: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: this.x + other.x, y: this.y + other.y }
    }

    #[inline(always)]
    fn v2_mul_scalar(this: Vector2d<Self>, k: Self) -> Vector2d<Self> {
        Vector2d { x: this.x * k, y: this.y * k }
    }

    #[inline(always)]
    fn v2_div_scalar(this: Vector2d<Self>, k: Self) -> Vector2d<Self> {
        Self::v2_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn v2_mul_add(this: Vector2d<Self>, k: Self, other: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: this.x * k + other.x, y: this.y * k + other.y }
    }

    #[inline(always)]
    fn v2_norm_squared(this: Vector2d<Self>) -> Self {
        this.x * this.x + this.y * this.y
    }

    #[inline(always)]
    fn v2_normalize(this: Vector2d<Self>) -> Vector2d<Self> {
        let norm_squared = Self::v2_norm_squared(this);
        if norm_squared == 0.0 {
            return Vector2d::default();
        }
        let norm_reciprocal = norm_squared.sqrt_reciprocal();
        Vector2d { x: this.x * norm_reciprocal, y: this.y * norm_reciprocal }
    }

    #[inline(always)]
    fn v2_is_normalized(q: Vector2d<Self>) -> bool {
        let norm_squared = Self::v2_norm_squared(q);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn v2_max(this: Vector2d<Self>) -> Self {
        if this.x > this.y { this.x } else { this.y }
    }

    #[inline(always)]
    fn v2_min(this: Vector2d<Self>) -> Self {
        if this.x < this.y { this.x } else { this.y }
    }

    #[inline(always)]
    fn v2_dot(this: Vector2d<Self>, other: Vector2d<Self>) -> Self {
        this.x * other.x + this.y * other.y
    }

    #[inline(always)]
    fn v2_cross(this: Vector2d<Self>, other: Vector2d<Self>) -> Self {
        this.x * other.y - this.y * other.x
    }
}
