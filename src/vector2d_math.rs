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
        // SAFETY: Both types are 16 bytes and aligned to 16 bytes.
        // The 'dummy' 4th float in the SIMD lane will be whatever
        // was in the padding (usually 0.0 if you use Default).
        unsafe { transmute(v) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x2> for Vector2d<f32> {
    #[inline(always)]
    fn from(simd: f32x2) -> Self {
        // SAFETY: Same size and alignment.
        unsafe { transmute(simd) }
    }
}

// **** Ops ****

pub trait Vector2dOps: Sized {
    fn reciprocal(x: Self) -> Self;
    fn norm_squared(q: Vector2d<Self>) -> Self;
    fn neg(v: Vector2d<Self>) -> Vector2d<Self>;
    fn add(lhs: Vector2d<Self>, lhs: Vector2d<Self>) -> Vector2d<Self>;
    fn mul_scalar(lhs: Vector2d<Self>, a: Self) -> Vector2d<Self>;
    fn div_scalar(lhs: Vector2d<Self>, a: Self) -> Vector2d<Self>;
    fn mul_add(lhs: Vector2d<Self>, a: Self, b: Vector2d<Self>) -> Vector2d<Self>;
}

impl Vector2dOps for f64 {
    #[inline(always)]
    fn reciprocal(x: Self) -> Self {
        1.0 / x
    }

    #[inline(always)]
    fn norm_squared(v: Vector2d<Self>) -> Self {
        v.x * v.x + v.y * v.y
    }

    #[inline(always)]
    fn neg(v: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: -v.x, y: -v.y }
    }

    #[inline(always)]
    fn add(lhs: Vector2d<Self>, rhs: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: lhs.x + rhs.x, y: lhs.y + rhs.y }
    }

    #[inline(always)]
    fn mul_scalar(lhs: Vector2d<Self>, a: Self) -> Vector2d<Self> {
        Vector2d { x: lhs.x * a, y: lhs.y * a }
    }

    #[inline(always)]
    fn div_scalar(lhs: Vector2d<Self>, a: Self) -> Vector2d<Self> {
        Self::mul_scalar(lhs, 1.0 / a)
    }
    #[inline(always)]
    fn mul_add(lhs: Vector2d<Self>, a: Self, b: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: lhs.x * a + b.x, y: lhs.y * a + b.y }
    }
}

// SIMD-accelerated implementation for f32
impl Vector2dOps for f32 {
    #[inline(always)]
    fn reciprocal(x: Self) -> Self {
        1.0 / x
    }

    #[inline(always)]
    fn norm_squared(v: Vector2d<Self>) -> Self {
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
    fn neg(v: Vector2d<Self>) -> Vector2d<Self> {
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
    fn add(lhs: Vector2d<Self>, rhs: Vector2d<Self>) -> Vector2d<Self> {
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
    fn mul_scalar(lhs: Vector2d<Self>, rhs: Self) -> Vector2d<Self> {
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
            use core::simd::f32x2;
            Vector2d { x: lhs.x * a, y: lhs.y * a }
        }
    }

    #[inline(always)]
    fn div_scalar(lhs: Vector2d<Self>, a: Self) -> Vector2d<Self> {
        Self::mul_scalar(lhs, 1.0 / a)
    }

    #[inline(always)]
    fn mul_add(lhs: Vector2d<Self>, a: Self, b: Vector2d<Self>) -> Vector2d<Self> {
        #[cfg(feature = "simd")]
        {
            //let v_lhs: f32x2 = lhs.into();
            //let v_b: f32x2 = b.into();
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
}

// **** Math ****

pub trait Vector2dMath: Sized {
    fn normalize(v: Vector2d<Self>) -> Vector2d<Self>;
    fn is_normalized(q: Vector2d<Self>) -> bool;
    fn dot(a: Vector2d<Self>, b: Vector2d<Self>) -> Self;
    fn cross(a: Vector2d<Self>, b: Vector2d<Self>) -> Vector2d<Self>;
}

impl Vector2dMath for f64 {
    #[inline(always)]
    fn normalize(v: Vector2d<Self>) -> Vector2d<Self> {
        let norm_squared = v.x * v.x + v.y * v.y;
        if norm_squared == 0.0 {
            return Vector2d::default();
        }
        let norm_reciprocal = norm_squared.reciprocal_sqrt();
        Vector2d { x: v.x * norm_reciprocal, y: v.y * norm_reciprocal }
    }

    #[inline(always)]
    fn is_normalized(q: Vector2d<Self>) -> bool {
        let norm_squared = Self::norm_squared(q);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn dot(a: Vector2d<Self>, b: Vector2d<Self>) -> Self {
        (a.x * b.x) + (a.y * b.y)
    }

    #[inline(always)]
    fn cross(_a: Vector2d<Self>, _b: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: 0.0, y: 0.0 }
    }
}

// SIMD-accelerated implementation for f32
impl Vector2dMath for f32 {
    #[inline(always)]
    fn normalize(v: Vector2d<Self>) -> Vector2d<Self> {
        #[cfg(feature = "simd")]
        {
            use core::simd::f32x2;

            // 1. Calculate magnitude squared using our SIMD Dot Product
            let norm_squared = Self::dot(v, v);

            // 2. Guard against division by zero (Important for sensor glitches!)
            if norm_squared == 0.0 {
                return Vector2d::default(); // Return zero vector if magnitude is 0
            }
            use crate::SqrtMethods;

            let norm_reciprocal = norm_squared.reciprocal_sqrt(); // Uses hardware vrsqrt

            // 3. Load vector into SIMD and "Splat" the inverse magnitude
            let mut v_simd: f32x2 = unsafe { core::mem::transmute(v) };
            let scale = f32x2::splat(norm_reciprocal);

            // 4. Multiply all lanes at once
            v_simd *= scale;

            unsafe { core::mem::transmute(v_simd) }
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
    fn is_normalized(q: Vector2d<Self>) -> bool {
        let norm_squared = Self::norm_squared(q);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }
    #[inline(always)]
    fn dot(a: Vector2d<Self>, b: Vector2d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            use core::simd::f32x2;
            use core::simd::num::SimdFloat;

            let va: f32x2 = unsafe { core::mem::transmute(a) };
            let vb: f32x2 = unsafe { core::mem::transmute(b) };

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
    fn cross(_a: Vector2d<Self>, _b: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: 0.0, y: 0.0 }
    }
}
