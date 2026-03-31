#[cfg(feature = "simd")]
use cfg_if::cfg_if;
cfg_if! {
    if #[cfg(feature = "align")] {
        use core::{mem::transmute, simd::f32x4};
        use core::simd::{num::SimdFloat};
        use core::simd::{simd_swizzle};
    }
}

use crate::{Quaternion, SqrtMethods, Vector3d};

#[cfg(feature = "simd")]
impl From<Quaternion<f32>> for f32x4 {
    #[inline(always)]
    fn from(v: Quaternion<f32>) -> Self {
        // SAFETY: Both types are 16 bytes and aligned to 16 bytes.
        // The 'dummy' 4th float in the SIMD lane will be whatever
        // was in the padding (usually 0.0 if you use Default).
        unsafe { transmute(v) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x4> for Quaternion<f32> {
    #[inline(always)]
    fn from(simd: f32x4) -> Self {
        // SAFETY: Same size and alignment.
        unsafe { transmute(simd) }
    }
}

pub trait QuaternionOps: Sized {
    fn reciprocal(x: Self) -> Self;
    fn norm_squared(q: Quaternion<Self>) -> Self;
    fn neg(q: Quaternion<Self>) -> Quaternion<Self>;
    fn add(lhs: Quaternion<Self>, lhs: Quaternion<Self>) -> Quaternion<Self>;
    fn mul_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self>;
    fn div_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self>;
    fn mul(rhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self>;
    fn mul_add(lhs: Quaternion<Self>, a: Self, b: Quaternion<Self>) -> Quaternion<Self>;
}

pub trait QuaternionMath: Sized {
    fn conjugate(q: Quaternion<Self>) -> Quaternion<Self>;
    fn normalize(q: Quaternion<Self>) -> Quaternion<Self>;
    fn is_normalized(q: Quaternion<Self>) -> bool;
    fn derivative(q: Quaternion<Self>, gyro: Vector3d<Self>) -> Quaternion<Self>;
}

impl QuaternionMath for f64 {
    #[inline(always)]
    fn conjugate(q: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: q.w, x: -q.x, y: -q.y, z: -q.z }
    }

    #[inline(always)]
    fn normalize(q: Quaternion<Self>) -> Quaternion<Self> {
        let norm_squared = Self::norm_squared(q);
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
    fn is_normalized(q: Quaternion<Self>) -> bool {
        let norm_squared = Self::norm_squared(q);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn derivative(q: Quaternion<Self>, gyro_rps: Vector3d<Self>) -> Quaternion<Self> {
        Quaternion {
            w: (q.x * gyro_rps.x - q.y * gyro_rps.y - q.z * gyro_rps.z) * 0.5,
            x: (q.w * gyro_rps.x + q.y * gyro_rps.z - q.z * gyro_rps.y) * 0.5,
            y: (q.w * gyro_rps.y - q.x * gyro_rps.z + q.z * gyro_rps.x) * 0.5,
            z: (q.w * gyro_rps.z + q.x * gyro_rps.y - q.y * gyro_rps.x) * 0.5,
        }
    }
}

impl QuaternionOps for f64 {
    #[inline(always)]
    fn reciprocal(x: Self) -> Self {
        1.0 / x
    }

    #[inline(always)]
    fn norm_squared(q: Quaternion<Self>) -> Self {
        q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
    }

    #[inline(always)]
    fn neg(q: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: -q.w, x: -q.x, y: -q.y, z: -q.z }
    }

    #[inline(always)]
    fn add(lhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: lhs.w + rhs.w, x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z }
    }

    #[inline(always)]
    fn mul_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self> {
        Quaternion { w: lhs.w * a, x: lhs.x * a, y: lhs.y * a, z: lhs.z * a }
    }

    #[inline(always)]
    fn div_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self> {
        Self::mul_scalar(lhs, 1.0 / a)
    }

    #[inline(always)]
    fn mul_add(lhs: Quaternion<Self>, a: Self, b: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: lhs.w * a + b.w, x: lhs.x * a + b.x, y: lhs.y * a + b.y, z: lhs.z * a + b.z }
    }

    #[inline(always)]
    fn mul(lhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion {
            w: lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
            x: lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
            y: lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
            z: lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
        }
    }
}

// SIMD-accelerated implementation for f32
impl QuaternionMath for f32 {
    #[inline(always)]
    fn conjugate(q: Quaternion<Self>) -> Quaternion<Self> {
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
    fn normalize(q: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            // 1. Transmute to SIMD
            let q_simd: f32x4 = unsafe { core::mem::transmute(q) };

            // 2. Dot product (magnitude squared)
            // No masking needed here because w is a valid lane in a Quat!
            let norm_squared = (q_simd * q_simd).reduce_sum();

            // 3. Guard against division by zero
            if norm_squared == 0.0 {
                return Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
            }
            let norm_reciprocal = norm_squared.reciprocal_sqrt(); // Uses our hardware vrsqrt
            let ret_simd = q_simd * f32x4::splat(norm_reciprocal);
            unsafe { core::mem::transmute(ret_simd) }
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
    fn is_normalized(q: Quaternion<Self>) -> bool {
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
    /*
    the lines:
    let w1 = simd_swizzle!(q_simd,); // [w, w, w, w]
    are giving the error
    unexpected end of macro invocation missing tokens in macro arguments
    */
    //=================

    #[inline(always)]
    fn derivative(q: Quaternion<Self>, gyro: Vector3d<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            /*// Load q: [w, x, y, z]
            let q_v: f32x4 = unsafe { core::mem::transmute(q) };

            // Load gyro: [x, y, z, padding] -> Swizzle to [0, x, y, z]
            let g_raw: f32x4 = unsafe { core::mem::transmute(gyro) };
            //let g_v = simd_swizzle!(g_raw, [f32x4::splat(0.0)], [4, 0, 1, 2]);

            // Efficiently shift: [x, y, z, 0] -> [0, x, y, z]
            // We rotate right and then zero out the 'w' (index 0)
            let g_v = simd_swizzle!(g_raw, [3, 0, 1, 2]) * f32x4::from_array([0.0, 1.0, 1.0, 1.0]);

            // Hamilton Product (q * g) for [w, x, y, z] layout:
            // w_out = -x1*gx - y1*gy - z1*gz
            // x_out =  w1*gx + y1*gz - z1*gy
            // y_out =  w1*gy - x1*gz + z1*gx
            // z_out =  w1*gz + x1*gy - y1*gx

            // Row A: [w, w, w, w] * [0, gx, gy, gz]
            let res = simd_swizzle!(q_v, [0, 0, 0, 0]) * g_v;

            // Row B: [x, -x, -x, -x] * [gx, 0, gz, gy]
            // (Using signs and swizzles to match the Hamilton rows)
            let x_part = simd_swizzle!(q_v, [1, 1, 1, 1])
                * simd_swizzle!(g_v, [1, 0, 3, 2])
                * f32x4::from_array([-1.0, 1.0, -1.0, 1.0]);

            // Row C: [y, y, -y, y] * [gy, gz, 0, gx]
            let y_part = simd_swizzle!(q_v, [2, 2, 2, 2])
                * simd_swizzle!(g_v, [2, 3, 0, 1])
                * f32x4::from_array([-1.0, 1.0, 1.0, -1.0]);

            // Row D: [z, z, z, -z] * [gz, gy, gx, 0]
            let z_part = simd_swizzle!(q_v, [3, 3, 3, 3])
                * simd_swizzle!(g_v, [3, 2, 1, 0])
                * f32x4::from_array([-1.0, -1.0, 1.0, 1.0]);

            let q_dot = (res + x_part + y_part + z_part) * f32x4::splat(0.5);

            unsafe { core::mem::transmute(q_dot) }*/
            let q_v: f32x4 = unsafe { core::mem::transmute(q) };
            let g_raw: f32x4 = unsafe { core::mem::transmute(gyro) };

            // Shift [x, y, z, pad] to [0, x, y, z] and zero the w lane
            let g_v = simd_swizzle!(g_raw, [3, 0, 1, 2]) * f32x4::from_array([0.0, 1.0, 1.0, 1.0]);

            // Parallel Hamilton Calculation
            let w1 = simd_swizzle!(q_v, [0, 0, 0, 0]);
            let x1 = simd_swizzle!(q_v, [1, 1, 1, 1]);
            let y1 = simd_swizzle!(q_v, [2, 2, 2, 2]);
            let z1 = simd_swizzle!(q_v, [3, 3, 3, 3]);

            let g_w = g_v; // [0, gx, gy, gz]
            let g_x = simd_swizzle!(g_v, [1, 0, 3, 2]); // [gx, 0, gz, gy]
            let g_y = simd_swizzle!(g_v, [2, 3, 0, 1]); // [gy, gz, 0, gx]
            let g_z = simd_swizzle!(g_v, [3, 2, 1, 0]); // [gz, gy, gx, 0]

            let res = (w1 * g_w)
                + (x1 * g_x * f32x4::from_array([-1.0, 1.0, -1.0, 1.0]))
                + (y1 * g_y * f32x4::from_array([-1.0, 1.0, 1.0, -1.0]))
                + (z1 * g_z * f32x4::from_array([-1.0, -1.0, 1.0, 1.0]));

            let q_dot = res * f32x4::splat(0.5);
            unsafe { core::mem::transmute(q_dot) }
        }
        #[cfg(not(feature = "simd"))]
        {
            Self {
                w: 0.5 * (-q.x * gyro.x - q.y * gyro.y - q.z * gyro.z),
                x: 0.5 * (q.w * gyro.x + q.y * gyro.z - q.z * gyro.y),
                y: 0.5 * (q.w * gyro.y - q.x * gyro.z + q.z * gyro.x),
                z: 0.5 * (q.w * gyro.z + q.x * gyro.y - q.y * gyro.x),
            }
        }
    }

    //==============
}

// SIMD-accelerated implementation for f32
impl QuaternionOps for f32 {
    #[inline(always)]
    fn reciprocal(x: Self) -> Self {
        1.0 / x
    }

    #[inline(always)]
    fn norm_squared(q: Quaternion<Self>) -> Self {
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
    fn neg(q: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            use core::simd::f32x4;
            // Transmute the 16-byte aligned struct to a SIMD register
            let q_simd: f32x4 = unsafe { core::mem::transmute(q) };

            // Negate all 4 lanes (x, y, z, w) simultaneously
            let res_simd = -q_simd;

            unsafe { core::mem::transmute(res_simd) }
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: -q.w, x: -q.x, y: -q.y, z: -q.z }
        }
    }

    #[inline(always)]
    fn add(lhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let lhs_simd = f32x4::from(lhs);
            let rhs_simd = f32x4::from(rhs);

            // Add all 4 lanes (w, x, y, z) in one cycle
            let ret_simd = lhs_simd + rhs_simd;

            ret_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: lhs.w + rhs.w, x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z }
        }
    }

    #[inline(always)]
    fn mul_scalar(lhs: Quaternion<Self>, rhs: Self) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let lhs_simd = f32x4::from(lhs);
            let rhs_simd = f32x4::splat(rhs);
            let ret_simd = lhs_simd * rhs_simd;
            ret_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: lhs.w * a, x: lhs.x * a, y: lhs.y * a, z: lhs.z * a }
        }
    }

    #[inline(always)]
    fn div_scalar(lhs: Quaternion<Self>, a: Self) -> Quaternion<Self> {
        Self::mul_scalar(lhs, 1.0 / a)
    }

    #[inline(always)]
    fn mul_add(lhs: Quaternion<Self>, a: Self, b: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let v_lhs = f32x4::from(lhs);
            let v_b = f32x4::from(b);
            let v_a = f32x4::splat(a);

            // This maps to the Vector Fused Multiply-Add instruction
            let ret = (v_lhs * v_a) + v_b;
            ret.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: lhs.w * a + b.w, x: lhs.x * a + b.x, y: lhs.y * a + b.y, z: lhs.z * a + b.z }
        }
    }

    #[inline(always)]
    fn mul(lhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self> {
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
}

// **** Mul ****
/*
impl core::ops::Mul for Quaternion {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(feature = "simd")]
        {
            use core::simd::{f32x4, simd_swizzle};

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
    fn scalar_mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}
*/
