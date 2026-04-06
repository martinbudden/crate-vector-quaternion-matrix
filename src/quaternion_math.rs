use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "simd")] {
        use core::{mem::transmute};
        use core::simd::{f32x4,num::SimdFloat};
    }
}

const _: () = assert!(core::mem::size_of::<Quaternion<f32>>() == 16);
const _: () = assert!(core::mem::align_of::<Quaternion<f32>>() == 16);

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

/// Math functions for Quaternion, using **SIMD** accelerations for `f32`.<br><br>
pub trait QuaternionMath: Sized {
    fn q_reciprocal(self) -> Self;
    fn q_neg(this: Quaternion<Self>) -> Quaternion<Self>;
    fn q_add(this: Quaternion<Self>, this: Quaternion<Self>) -> Quaternion<Self>;
    fn q_mul_scalar(this: Quaternion<Self>, a: Self) -> Quaternion<Self>;
    fn q_div_scalar(this: Quaternion<Self>, a: Self) -> Quaternion<Self>;
    fn q_mul_add(this: Quaternion<Self>, k: Self, other: Quaternion<Self>) -> Quaternion<Self>;
    fn q_norm_squared(this: Quaternion<Self>) -> Self;
    fn q_normalize(this: Quaternion<Self>) -> Quaternion<Self>;
    fn q_is_normalized(this: Quaternion<Self>) -> bool;
    fn q_mul(other: Quaternion<Self>, other: Quaternion<Self>) -> Quaternion<Self>;
    fn q_conjugate(this: Quaternion<Self>) -> Quaternion<Self>;
}

// **** SIMD-accelerated implementation for f32 ****

impl QuaternionMath for f32 {
    #[inline(always)]
    fn q_reciprocal(self) -> Self {
        1.0 / self
    }

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
    fn q_normalize(this: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let norm_squared = (this_simd * this_simd).reduce_sum();
            // If norm_squared is zero, then this must be the unit quaternion
            if norm_squared == 0.0 {
                return Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
            }
            let norm_reciprocal = norm_squared.sqrt_reciprocal(); // Uses our hardware vrsqrt
            let scale = f32x4::splat(norm_reciprocal);

            (this_simd * scale).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let norm_squared = this.w * this.w + this.x * this.x + this.y * this.y + this.z * this.z;
            if norm_squared == 0.0 {
                return Quaternion::default();
            }
            let norm_reciprocal = norm_squared.sqrt_reciprocal();
            Quaternion {
                w: this.x * norm_reciprocal,
                x: this.x * norm_reciprocal,
                y: this.y * norm_reciprocal,
                z: this.z * norm_reciprocal,
            }
        }
    }

    #[inline(always)]
    fn q_is_normalized(this: Quaternion<Self>) -> bool {
        let norm_squared = Self::q_norm_squared(this);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
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
}

// **** f64 ****

impl QuaternionMath for f64 {
    #[inline(always)]
    fn q_reciprocal(self) -> Self {
        1.0 / self
    }

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
    fn q_normalize(this: Quaternion<Self>) -> Quaternion<Self> {
        let norm_squared = Self::q_norm_squared(this);
        if norm_squared == 0.0 {
            return Quaternion::default();
        }
        let norm_reciprocal = norm_squared.sqrt_reciprocal();
        Quaternion {
            w: this.x * norm_reciprocal,
            x: this.x * norm_reciprocal,
            y: this.y * norm_reciprocal,
            z: this.z * norm_reciprocal,
        }
    }

    #[inline(always)]
    fn q_is_normalized(this: Quaternion<Self>) -> bool {
        let norm_squared = Self::q_norm_squared(this);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn q_conjugate(this: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: this.w, x: -this.x, y: -this.y, z: -this.z }
    }
}
