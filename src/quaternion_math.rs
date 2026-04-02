#[cfg(feature = "simd")]
use cfg_if::cfg_if;
cfg_if! {
    if #[cfg(feature = "align")] {
        use core::{mem::transmute, simd::f32x4};
        use core::simd::{num::SimdFloat};
    }
}

use crate::{Quaternion, SqrtMethods};

// **** From ****

#[cfg(feature = "simd")]
impl From<Quaternion<f32>> for f32x4 {
    #[inline(always)]
    fn from(v: Quaternion<f32>) -> Self {
        // SAFETY: assert f32x4 and Quaternion<f32> have same size and alignment
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::size_of::<Quaternion<f32>>());
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::align_of::<Quaternion<f32>>());
        unsafe { transmute(v) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x4> for Quaternion<f32> {
    #[inline(always)]
    fn from(simd: f32x4) -> Self {
        // SAFETY: assert f32x4 and Quaternion<f32> have same size and alignment
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::size_of::<Quaternion<f32>>());
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::align_of::<Quaternion<f32>>());
        unsafe { transmute(simd) }
    }
}

// **** Math ****

/// Math functions for Quaternion, using SIMD accelerations for f32.
pub trait QuaternionMath: Sized {
    fn q_reciprocal(x: Self) -> Self;
    fn q_neg(q: Quaternion<Self>) -> Quaternion<Self>;
    fn q_add(lhs: Quaternion<Self>, lhs: Quaternion<Self>) -> Quaternion<Self>;
    fn q_mul_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self>;
    fn q_div_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self>;
    fn q_mul(rhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self>;
    fn q_mul_add(lhs: Quaternion<Self>, a: Self, b: Quaternion<Self>) -> Quaternion<Self>;
    fn q_norm_squared(q: Quaternion<Self>) -> Self;
    fn q_normalize(q: Quaternion<Self>) -> Quaternion<Self>;
    fn q_is_normalized(q: Quaternion<Self>) -> bool;
    fn q_conjugate(q: Quaternion<Self>) -> Quaternion<Self>;
}

// **** SIMD-accelerated implementation for f32 ****

impl QuaternionMath for f32 {
    #[inline(always)]
    fn q_reciprocal(x: Self) -> Self {
        1.0 / x
    }

    #[inline(always)]
    fn q_neg(q: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            // Transmute the 16-byte aligned struct to a SIMD register
            let q_simd = f32x4::from(q);
            // Negate all 4 lanes (x, y, z, w) simultaneously
            (-q_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: -q.w, x: -q.x, y: -q.y, z: -q.z }
        }
    }

    #[inline(always)]
    fn q_add(lhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let lhs_simd = f32x4::from(lhs);
            let rhs_simd = f32x4::from(rhs);

            // Add all 4 lanes (w, x, y, z) in one cycle
            (lhs_simd + rhs_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: lhs.w + rhs.w, x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z }
        }
    }

    #[inline(always)]
    fn q_mul_scalar(lhs: Quaternion<Self>, rhs: Self) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let lhs_simd = f32x4::from(lhs);
            let rhs_simd = f32x4::splat(rhs);
            (lhs_simd * rhs_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: lhs.w * a, x: lhs.x * a, y: lhs.y * a, z: lhs.z * a }
        }
    }

    #[inline(always)]
    fn q_div_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self> {
        Self::q_mul_scalar(lhs, 1.0 / a)
    }

    #[inline(always)]
    fn q_mul_add(lhs: Quaternion<Self>, a: Self, b: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let v_lhs = f32x4::from(lhs);
            let v_b = f32x4::from(b);
            let v_a = f32x4::splat(a);

            // This maps to the Vector Fused Multiply-Add instruction
            ((v_lhs * v_a) + v_b).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: lhs.w * a + b.w, x: lhs.x * a + b.x, y: lhs.y * a + b.y, z: lhs.z * a + b.z }
        }
    }

    #[inline(always)]
    fn q_mul(lhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            Quaternion {
                w: lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
                x: lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
                y: lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
                z: lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion {
                w: lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
                x: lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
                y: lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
                z: lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
            }
        }
    }

    #[inline(always)]
    fn q_norm_squared(q: Quaternion<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let q_simd = f32x4::from(q);
            (q_simd * q_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
        }
    }

    #[inline(always)]
    fn q_normalize(q: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            // 1. Transmute to SIMD
            let q_simd = f32x4::from(q);

            // 2. Dot product (magnitude squared)
            let norm_squared = (q_simd * q_simd).reduce_sum();
            // 3. If norm_squared is zero, then this must be the unit quaternion
            if norm_squared == 0.0 {
                return Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
            }
            let norm_reciprocal = norm_squared.reciprocal_sqrt(); // Uses our hardware vrsqrt
            let ret_simd = q_simd * f32x4::splat(norm_reciprocal);
            ret_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let norm_squared = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
            if norm_squared == 0.0 {
                return Quaternion::default();
            }
            let norm_reciprocal = norm_squared.reciprocal_sqrt();
            Quaternion {
                w: q.x * norm_reciprocal,
                x: q.x * norm_reciprocal,
                y: q.y * norm_reciprocal,
                z: q.z * norm_reciprocal,
            }
        }
    }

    #[inline(always)]
    fn q_is_normalized(q: Quaternion<Self>) -> bool {
        let norm_squared = Self::q_norm_squared(q);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn q_conjugate(q: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let q_simd = f32x4::from(q);
            // Negate x, y, z but keep w positive
            // Mask: [1.0, -1.0, -1.0, -1.0]
            let mask = f32x4::from_array([1.0, -1.0, -1.0, -1.0]);
            let ret_simd = q_simd * mask;
            ret_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: q.w, x: -q.x, y: -q.y, z: -q.z }
        }
    }
}

// **** f64 ****

impl QuaternionMath for f64 {
    #[inline(always)]
    fn q_reciprocal(x: Self) -> Self {
        1.0 / x
    }

    #[inline(always)]
    fn q_neg(q: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: -q.w, x: -q.x, y: -q.y, z: -q.z }
    }

    #[inline(always)]
    fn q_add(lhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: lhs.w + rhs.w, x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z }
    }

    #[inline(always)]
    fn q_mul_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self> {
        Quaternion { w: lhs.w * a, x: lhs.x * a, y: lhs.y * a, z: lhs.z * a }
    }

    #[inline(always)]
    fn q_div_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self> {
        Self::q_mul_scalar(lhs, 1.0 / a)
    }

    #[inline(always)]
    fn q_mul_add(lhs: Quaternion<Self>, a: Self, b: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: lhs.w * a + b.w, x: lhs.x * a + b.x, y: lhs.y * a + b.y, z: lhs.z * a + b.z }
    }

    #[inline(always)]
    fn q_mul(lhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion {
            w: lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
            x: lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
            y: lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
            z: lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
        }
    }

    #[inline(always)]
    fn q_norm_squared(q: Quaternion<Self>) -> Self {
        q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
    }

    #[inline(always)]
    fn q_normalize(q: Quaternion<Self>) -> Quaternion<Self> {
        let norm_squared = Self::q_norm_squared(q);
        if norm_squared == 0.0 {
            return Quaternion::default();
        }
        let norm_reciprocal = norm_squared.reciprocal_sqrt();
        Quaternion {
            w: q.x * norm_reciprocal,
            x: q.x * norm_reciprocal,
            y: q.y * norm_reciprocal,
            z: q.z * norm_reciprocal,
        }
    }

    #[inline(always)]
    fn q_is_normalized(q: Quaternion<Self>) -> bool {
        let norm_squared = Self::q_norm_squared(q);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn q_conjugate(q: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: q.w, x: -q.x, y: -q.y, z: -q.z }
    }
}
