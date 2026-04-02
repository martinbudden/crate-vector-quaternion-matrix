#[cfg(feature = "simd")]
use cfg_if::cfg_if;
cfg_if! {
    if #[cfg(feature = "align")] {
        use core::{mem::transmute, simd::f32x2};
        use core::simd::{num::SimdFloat};
    }
}

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

/// Math functions for Vector2d, using SIMD accelerations for f32.
pub trait Vector2dMath: Sized {
    fn v2_reciprocal(x: Self) -> Self;
    fn v2_neg(v: Vector2d<Self>) -> Vector2d<Self>;
    fn v2_add(lhs: Vector2d<Self>, lhs: Vector2d<Self>) -> Vector2d<Self>;
    fn v2_mul_scalar(lhs: Vector2d<Self>, a: Self) -> Vector2d<Self>;
    fn v2_div_scalar(lhs: Vector2d<Self>, a: Self) -> Vector2d<Self>;
    fn v2_mul_add(lhs: Vector2d<Self>, a: Self, b: Vector2d<Self>) -> Vector2d<Self>;
    fn v2_norm_squared(q: Vector2d<Self>) -> Self;
    fn v2_normalize(v: Vector2d<Self>) -> Vector2d<Self>;
    fn v2_is_normalized(q: Vector2d<Self>) -> bool;
    fn v2_dot(a: Vector2d<Self>, b: Vector2d<Self>) -> Self;
    fn v2_cross(this: Vector2d<Self>, other: Vector2d<Self>) -> Self;
}

// **** SIMD-accelerated implementation for f32 ****

impl Vector2dMath for f32 {
    #[inline(always)]
    fn v2_reciprocal(x: Self) -> Self {
        1.0 / x
    }

    #[inline(always)]
    fn v2_neg(v: Vector2d<Self>) -> Vector2d<Self> {
        #[cfg(feature = "simd")]
        {
            // Transmute the 16-byte aligned struct to a SIMD register
            let v_simd = f32x2::from(v);
            // Negate all 4 lanes (x, y, z, w) simultaneously
            let ret_simd = -v_simd;
            ret_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2d { x: -v.x, y: -v.y }
        }
    }

    #[inline(always)]
    fn v2_add(lhs: Vector2d<Self>, rhs: Vector2d<Self>) -> Vector2d<Self> {
        #[cfg(feature = "simd")]
        {
            let lhs_simd = f32x2::from(lhs);
            let rhs_simd = f32x2::from(rhs);

            // Add all 4 lanes (w, x, y, z) in one cycle
            let ret_simd = lhs_simd + rhs_simd;

            ret_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2d { x: lhs.x + rhs.x, y: lhs.y + rhs.y }
        }
    }

    #[inline(always)]
    fn v2_mul_scalar(lhs: Vector2d<Self>, rhs: Self) -> Vector2d<Self> {
        #[cfg(feature = "simd")]
        {
            // 1. Transmute to SIMD
            let lhs_simd = f32x2::from(lhs);
            // 2. "Splat" the scalar: [s, s, s, s]
            let rhs_simd = f32x2::splat(rhs);
            // 3. Multiply all 4 lanes in 1 cycle (x*s, y*s, z*s, padding*s)
            let ret = lhs_simd * rhs_simd;
            ret.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2d { x: lhs.x * a, y: lhs.y * a }
        }
    }

    #[inline(always)]
    fn v2_div_scalar(lhs: Vector2d<Self>, a: Self) -> Vector2d<Self> {
        Self::v2_mul_scalar(lhs, 1.0 / a)
    }

    #[inline(always)]
    fn v2_mul_add(lhs: Vector2d<Self>, a: Self, b: Vector2d<Self>) -> Vector2d<Self> {
        #[cfg(feature = "simd")]
        {
            let v_lhs = f32x2::from(lhs);
            let v_b = f32x2::from(b);
            let v_a = f32x2::splat(a);

            // This maps to the Vector Fused Multiply-Add instruction
            let ret = (v_lhs * v_a) + v_b;
            ret.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector2d { x: lhs.x * a + b.x, y: lhs.y * a + b.y }
        }
    }

    #[inline(always)]
    fn v2_norm_squared(v: Vector2d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let v_simd = f32x2::from(v);
            (v_simd * v_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            v.x * v.x + v.y * v.y + v.z * v.z
        }
    }

    #[inline(always)]
    fn v2_normalize(v: Vector2d<Self>) -> Vector2d<Self> {
        #[cfg(feature = "simd")]
        {
            // 1. Calculate magnitude squared using our SIMD Dot Product
            let norm_squared = Self::v2_dot(v, v);

            // 2. Guard against division by zero (Important for sensor glitches!)
            if norm_squared == 0.0 {
                return Vector2d::default(); // Return zero vector if magnitude is 0
            }

            let norm_reciprocal = norm_squared.reciprocal_sqrt(); // Uses hardware vrsqrt

            // 3. Load vector into SIMD and "Splat" the inverse magnitude
            let mut v_simd = f32x2::from(v);
            let scale = f32x2::splat(norm_reciprocal);

            // 4. Multiply all lanes at once
            v_simd *= scale;

            v_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let norm_squared = v.x * v.x + v.y * v.y + v.z * v.z;
            if norm_squared == 0.0 {
                return Vector2d::default();
            }
            let norm_reciprocal = norm_squared.reciprocal_sqrt();
            Vector2d { x: v.x * norm_reciprocal, y: v.y * norm_reciprocal }
        }
    }

    #[inline(always)]
    fn v2_is_normalized(v: Vector2d<Self>) -> bool {
        let norm_squared = Self::v2_norm_squared(v);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn v2_dot(a: Vector2d<Self>, b: Vector2d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let va = f32x2::from(a);
            let vb = f32x2::from(b);

            // Multiply the vectors
            let prod = va * vb;

            prod.reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            (a.x * b.x) + (a.y * b.y)
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
    fn v2_reciprocal(x: Self) -> Self {
        1.0 / x
    }

    #[inline(always)]
    fn v2_neg(v: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: -v.x, y: -v.y }
    }

    #[inline(always)]
    fn v2_add(lhs: Vector2d<Self>, rhs: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: lhs.x + rhs.x, y: lhs.y + rhs.y }
    }

    #[inline(always)]
    fn v2_mul_scalar(lhs: Vector2d<Self>, a: Self) -> Vector2d<Self> {
        Vector2d { x: lhs.x * a, y: lhs.y * a }
    }

    #[inline(always)]
    fn v2_div_scalar(lhs: Vector2d<Self>, a: Self) -> Vector2d<Self> {
        Self::v2_mul_scalar(lhs, 1.0 / a)
    }

    #[inline(always)]
    fn v2_mul_add(lhs: Vector2d<Self>, a: Self, b: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: lhs.x * a + b.x, y: lhs.y * a + b.y }
    }

    #[inline(always)]
    fn v2_norm_squared(v: Vector2d<Self>) -> Self {
        v.x * v.x + v.y * v.y
    }

    #[inline(always)]
    fn v2_normalize(v: Vector2d<Self>) -> Vector2d<Self> {
        let norm_squared = v.x * v.x + v.y * v.y;
        if norm_squared == 0.0 {
            return Vector2d::default();
        }
        let norm_reciprocal = norm_squared.reciprocal_sqrt();
        Vector2d { x: v.x * norm_reciprocal, y: v.y * norm_reciprocal }
    }

    #[inline(always)]
    fn v2_is_normalized(q: Vector2d<Self>) -> bool {
        let norm_squared = Self::v2_norm_squared(q);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn v2_dot(a: Vector2d<Self>, b: Vector2d<Self>) -> Self {
        (a.x * b.x) + (a.y * b.y)
    }

    #[inline(always)]
    fn v2_cross(this: Vector2d<Self>, other: Vector2d<Self>) -> Self {
        this.x * other.y - this.y * other.x
    }
}
