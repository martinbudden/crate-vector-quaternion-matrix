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

// **** Ops ****

pub trait QuaternionMath: Sized {
    fn q_reciprocal(x: Self) -> Self;
    fn q_norm_squared(q: Quaternion<Self>) -> Self;
    fn q_neg(q: Quaternion<Self>) -> Quaternion<Self>;
    fn q_add(lhs: Quaternion<Self>, lhs: Quaternion<Self>) -> Quaternion<Self>;
    fn q_mul_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self>;
    fn q_div_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self>;
    fn q_mul(rhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self>;
    fn q_mul_add(lhs: Quaternion<Self>, a: Self, b: Quaternion<Self>) -> Quaternion<Self>;
    fn q_normalize(q: Quaternion<Self>) -> Quaternion<Self>;
    fn q_is_normalized(q: Quaternion<Self>) -> bool;
    fn q_conjugate(q: Quaternion<Self>) -> Quaternion<Self>;
}

impl QuaternionMath for f64 {
    #[inline(always)]
    fn q_reciprocal(x: Self) -> Self {
        1.0 / x
    }

    #[inline(always)]
    fn q_norm_squared(q: Quaternion<Self>) -> Self {
        q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
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

// SIMD-accelerated implementation for f32
impl QuaternionMath for f32 {
    #[inline(always)]
    fn q_reciprocal(x: Self) -> Self {
        1.0 / x
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

    #[inline(always)]
    fn q_normalize(q: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            // 1. Transmute to SIMD
            let q_simd = f32x4::from(q);

            // 2. Dot product (magnitude squared)
            // No masking needed here because w is a valid lane in a Quat!
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
        // SIMD version only takes ~4-5 clock cycles on the RP2350.
        #[cfg(feature = "simd")]
        {
            // 1. Transmute to SIMD and calculate dot product
            let q_simd = f32x4::from(q);
            let norm_squared = (q_simd * q_simd).reduce_sum();

            // 2. Check if mag_sq is approx 1.0
            approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
        }

        #[cfg(not(feature = "simd"))]
        {
            let norm_squared = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
            approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
        }
    }
}

// **** Mul ****
/*
impl core::ops::Mul for Quaternion {
    type Output = Self;

    #[inline(always)]
    fn q_mul(self, rhs: Self) -> Self {
        #[cfg(feature = "simd")]
        {
            let a: f32x4 = unsafe { core::mem::transmute_copy(&self) };
            let b: f32x4 = unsafe { core::mem::transmute_copy(&rhs) };

            // 1. Initial product: [w1*x2, w1*y2, w1*z2, w1*w2]
            // We swizzle 'a' to broadcast 'w' into all lanes
            let a_wwww = simd_swizzle!(a, [3, 3, 3, 3]);
            let mut res = a_wwww * b;

            // 2. Add/Sub subsequent terms using swizzles and FMA
            // [x1*w2, y1*w2, z1*w2, -x1*x2]
            let a_xyzx = simd_swizzle!(a, [0, 1, 2, 0]);
            let b_wwxx = simd_swizzle!(b, [3, 3, 3, 0]);
            // Logic: res = res + (a_xyzx * b_wwxx) with sign flips..

            // Note: For brevity, most SIMD libs use a specific
            // set of 4 vector FMAs to complete the Hamilton product.

            // For now, let's look at the robust Scalar version that
            // the compiler can still auto-vectorize:
            self.scalar_mul(rhs)
        }
        #[cfg(not(feature = "simd"))]
        {
            self.scalar_mul(rhs)
        }
    }
}

    impl Quaternion {
    #[inline(always)]
    fn q_scalar_mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}
*/
