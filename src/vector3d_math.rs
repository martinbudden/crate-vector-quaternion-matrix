#[cfg(feature = "simd")]
use cfg_if::cfg_if;
cfg_if! {
    if #[cfg(feature = "align")] {
        use core::{mem::transmute, simd::f32x4};
        use core::simd::{simd_swizzle,num::SimdFloat};
    }
}

use crate::{SqrtMethods, Vector3d};

// **** From ****

#[cfg(feature = "simd")]
impl From<Vector3d<f32>> for f32x4 {
    #[inline(always)]
    fn from(v: Vector3d<f32>) -> Self {
        // SAFETY: assert f32x4 and Vector3d<f32> have same size and alignment
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::size_of::<Vector3d<f32>>());
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::align_of::<Vector3d<f32>>());
        // The 'filler' 4th float in the SIMD lane will be whatever was in the padding (usually 0.0 if set by Default).
        unsafe { transmute(v) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x4> for Vector3d<f32> {
    #[inline(always)]
    fn from(simd: f32x4) -> Self {
        // SAFETY: assert f32x4 and Vector3d<f32> have same size and alignment
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::size_of::<Vector3d<f32>>());
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::align_of::<Vector3d<f32>>());
        unsafe { transmute(simd) }
    }
}

// **** Math ****

/// Math functions for Vector3d, using SIMD accelerations for f32.
pub trait Vector3dMath: Sized {
    fn v3_reciprocal(x: Self) -> Self;
    fn v3_neg(v: Vector3d<Self>) -> Vector3d<Self>;
    fn v3_add(lhs: Vector3d<Self>, lhs: Vector3d<Self>) -> Vector3d<Self>;
    fn v3_mul_scalar(lhs: Vector3d<Self>, a: Self) -> Vector3d<Self>;
    fn v3_div_scalar(lhs: Vector3d<Self>, a: Self) -> Vector3d<Self>;
    fn v3_mul_add(lhs: Vector3d<Self>, a: Self, b: Vector3d<Self>) -> Vector3d<Self>;
    fn v3_norm_squared(v: Vector3d<Self>) -> Self;
    fn v3_normalize(v: Vector3d<Self>) -> Vector3d<Self>;
    fn v3_is_normalized(v: Vector3d<Self>) -> bool;
    fn v3_dot(a: Vector3d<Self>, b: Vector3d<Self>) -> Self;
    fn v3_cross(a: Vector3d<Self>, b: Vector3d<Self>) -> Vector3d<Self>;
}

// **** SIMD-accelerated implementation for f32 ****

impl Vector3dMath for f32 {
    #[inline(always)]
    fn v3_reciprocal(x: Self) -> Self {
        1.0 / x
    }

    #[inline(always)]
    fn v3_neg(v: Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            // Transmute the 16-byte aligned struct to a SIMD register
            let v_simd = f32x4::from(v);
            // Negate all 3 lanes (x, y, z, filler) simultaneously
            (-v_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3d { x: -v.x, y: -v.y, z: -v.z }
        }
    }

    #[inline(always)]
    fn v3_add(lhs: Vector3d<Self>, rhs: Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            let lhs_simd = f32x4::from(lhs);
            let rhs_simd = f32x4::from(rhs);

            // Add all 4 lanes (w, x, y, filler) in one cycle
            (lhs_simd + rhs_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3d { x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z }
        }
    }

    #[inline(always)]
    fn v3_mul_scalar(lhs: Vector3d<Self>, rhs: Self) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            // 1. Transmute to SIMD
            let lhs_simd = f32x4::from(lhs);
            // 2. "Splat" the scalar: [s, s, s, s]
            let rhs_simd = f32x4::splat(rhs);
            // 3. Multiply all 4 lanes in 1 cycle (x*s, y*s, z*s, padding*s)
            (lhs_simd * rhs_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3d { x: lhs.x * a, y: lhs.y * a, z: lhs.z * a }
        }
    }

    #[inline(always)]
    fn v3_div_scalar(lhs: Vector3d<Self>, a: Self) -> Vector3d<Self> {
        Self::v3_mul_scalar(lhs, 1.0 / a)
    }

    #[inline(always)]
    fn v3_mul_add(lhs: Vector3d<Self>, a: Self, b: Vector3d<Self>) -> Vector3d<Self> {
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
            Vector3d { x: lhs.x * a + b.x, y: lhs.y * a + b.y, z: lhs.z * a + b.z }
        }
    }

    #[inline(always)]
    fn v3_norm_squared(v: Vector3d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let v_simd = f32x4::from(v);
            (v_simd * v_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            v.x * v.x + v.y * v.y + v.z * v.z
        }
    }

    #[inline(always)]
    fn v3_normalize(v: Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            // 1. Calculate magnitude squared using our SIMD Dot Product
            let norm_squared = Self::v3_dot(v, v);
            // 2. If norm_squared is zero, then this must be zero (default) vector
            if norm_squared == 0.0 {
                return Vector3d::default(); // Return zero vector if magnitude is 0
            }

            let norm_reciprocal = norm_squared.reciprocal_sqrt(); // Uses hardware vrsqrt
            // 3. Load vector into SIMD and "Splat" the inverse magnitude
            let ret_simd = f32x4::from(v) * f32x4::splat(norm_reciprocal);
            ret_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let norm_squared = v.x * v.x + v.y * v.y + v.z * v.z;
            if norm_squared == 0.0 {
                return Vector3d::default();
            }
            let norm_reciprocal = norm_squared.reciprocal_sqrt();
            Vector3d { x: v.x * norm_reciprocal, y: v.y * norm_reciprocal, z: v.z * norm_reciprocal }
        }
    }

    #[inline(always)]
    fn v3_is_normalized(v: Vector3d<Self>) -> bool {
        let norm_squared = Self::v3_norm_squared(v);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn v3_dot(a: Vector3d<Self>, b: Vector3d<Self>) -> Self {
        (a.x * b.x) + (a.y * b.y) + (a.z * b.z)
        /*
        #[cfg(feature = "simd")]
        {
            let va = f32x4::from(a);
            let vb = f32x4::from(b);

            // Multiply the vectors, masking 4 lane to 0.0
            let prod = (va * vb) * f32x4::from_array([1.0, 1.0, 1.0, 0.0]);

            prod.reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            (a.x * b.x) + (a.y * b.y) + (a.z * b.z)
        }
        */
    }

    #[inline(always)]
    fn v3_cross(a: Vector3d<Self>, b: Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            let va = f32x4::from(a);
            let vb = f32x4::from(b);

            // Swizzle A: [y, z, x, w]
            let a_yzx = simd_swizzle!(va, [1, 2, 0, 3]);
            // Swizzle B: [z, x, y, w]
            let b_zxy = simd_swizzle!(vb, [2, 0, 1, 3]);

            // Swizzle A2: [z, x, y, w]
            let a_zxy = simd_swizzle!(va, [2, 0, 1, 3]);
            // Swizzle B2: [y, z, x, w]
            let b_yzx = simd_swizzle!(vb, [1, 2, 0, 3]);

            // Result = (a_yzx * b_zxy) - (a_zxy * b_yzx)
            let ret_simd = (a_yzx * b_zxy) - (a_zxy * b_yzx);

            // Transmute back to our Vector3d struct
            ret_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3d { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x }
        }
    }
}

// **** f64 ****

impl Vector3dMath for f64 {
    #[inline(always)]
    fn v3_reciprocal(x: Self) -> Self {
        1.0 / x
    }

    #[inline(always)]
    fn v3_neg(v: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d { x: -v.x, y: -v.y, z: -v.z }
    }

    #[inline(always)]
    fn v3_add(lhs: Vector3d<Self>, rhs: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d { x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z }
    }

    #[inline(always)]
    fn v3_mul_scalar(lhs: Vector3d<Self>, a: Self) -> Vector3d<Self> {
        Vector3d { x: lhs.x * a, y: lhs.y * a, z: lhs.z * a }
    }

    #[inline(always)]
    fn v3_div_scalar(lhs: Vector3d<Self>, a: Self) -> Vector3d<Self> {
        Self::v3_mul_scalar(lhs, 1.0 / a)
    }

    #[inline(always)]
    fn v3_mul_add(lhs: Vector3d<Self>, a: Self, b: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d { x: lhs.x * a + b.x, y: lhs.y * a + b.y, z: lhs.z * a + b.z }
    }

    #[inline(always)]
    fn v3_norm_squared(v: Vector3d<Self>) -> Self {
        v.x * v.x + v.y * v.y + v.z * v.z
    }

    #[inline(always)]
    fn v3_normalize(v: Vector3d<Self>) -> Vector3d<Self> {
        let norm_squared = v.x * v.x + v.y * v.y + v.z * v.z;
        if norm_squared == 0.0 {
            return Vector3d::default();
        }
        let norm_reciprocal = norm_squared.reciprocal_sqrt();
        Vector3d { x: v.x * norm_reciprocal, y: v.y * norm_reciprocal, z: v.z * norm_reciprocal }
    }

    #[inline(always)]
    fn v3_is_normalized(q: Vector3d<Self>) -> bool {
        let norm_squared = Self::v3_norm_squared(q);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn v3_dot(a: Vector3d<Self>, b: Vector3d<Self>) -> Self {
        (a.x * b.x) + (a.y * b.y) + (a.z * b.z)
    }

    #[inline(always)]
    fn v3_cross(this: Vector3d<Self>, other: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d {
            x: this.y * other.z - this.z * other.y,
            y: this.z * other.x - this.x * other.z,
            z: this.x * other.y - this.y * other.x,
        }
    }
}
