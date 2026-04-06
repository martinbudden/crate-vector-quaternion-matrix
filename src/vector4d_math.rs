use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "simd")] {
        use core::{mem::transmute};
        use core::simd::{f32x4,num::SimdFloat};
    }
}

const _: () = assert!(core::mem::size_of::<Vector4d<f32>>() == 16);
const _: () = assert!(core::mem::align_of::<Vector4d<f32>>() == 16);

use crate::{SqrtMethods, Vector4d};

// **** From ****

#[cfg(feature = "simd")]
impl From<Vector4d<f32>> for f32x4 {
    #[inline(always)]
    fn from(this: Vector4d<f32>) -> Self {
        // SAFETY: assert f32x4 and Vector4d<f32> have same size and alignment
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::size_of::<Vector4d<f32>>());
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::align_of::<Vector4d<f32>>());
        // The 'filler' 4th float in the SIMD lane will be whatever was in the padding (usually 0.0 if set by Default).
        unsafe { transmute(this) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x4> for Vector4d<f32> {
    #[inline(always)]
    fn from(simd: f32x4) -> Self {
        // SAFETY: assert f32x4 and Vector4d<f32> have same size and alignment
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::size_of::<Vector4d<f32>>());
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::align_of::<Vector4d<f32>>());
        unsafe { transmute(simd) }
    }
}

// **** Math ****

/// Math functions for Vector4d, using **SIMD** accelerations for `f32`.<br><br>
pub trait Vector4dMath: Sized {
    fn v4_reciprocal(self) -> Self;
    fn v4_neg(this: Vector4d<Self>) -> Vector4d<Self>;
    fn v4_add(this: Vector4d<Self>, this: Vector4d<Self>) -> Vector4d<Self>;
    fn v4_mul_scalar(this: Vector4d<Self>, k: Self) -> Vector4d<Self>;
    fn v4_div_scalar(this: Vector4d<Self>, k: Self) -> Vector4d<Self>;
    fn v4_mul_add(this: Vector4d<Self>, k: Self, other: Vector4d<Self>) -> Vector4d<Self>;
    fn v4_norm_squared(this: Vector4d<Self>) -> Self;
    fn v4_normalize(this: Vector4d<Self>) -> Vector4d<Self>;
    fn v4_is_normalized(this: Vector4d<Self>) -> bool;
    fn v4_max(this: Vector4d<Self>) -> Self;
    fn v4_min(this: Vector4d<Self>) -> Self;
    fn v4_dot(this: Vector4d<Self>, other: Vector4d<Self>) -> Self;
}

// **** SIMD-accelerated implementation for f32 ****

impl Vector4dMath for f32 {
    #[inline(always)]
    fn v4_reciprocal(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn v4_neg(this: Vector4d<Self>) -> Vector4d<Self> {
        #[cfg(feature = "simd")]
        {
            (-f32x4::from(this)).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector4d { x: -this.x, y: -this.y, z: -this.z, t: -this.t }
        }
    }

    #[inline(always)]
    fn v4_add(this: Vector4d<Self>, other: Vector4d<Self>) -> Vector4d<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);

            (this_simd + other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector4d { x: this.x + other.x, y: this.y + other.y, z: this.z + other.z, t: this.t + other.t }
        }
    }

    #[inline(always)]
    fn v4_mul_scalar(this: Vector4d<Self>, k: Self) -> Vector4d<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let k_simd = f32x4::splat(k);

            (this_simd * k_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector4d { x: this.x * k, y: this.y * k, z: this.z * k, t: this.t * k }
        }
    }

    #[inline(always)]
    fn v4_div_scalar(this: Vector4d<Self>, k: Self) -> Vector4d<Self> {
        Self::v4_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn v4_mul_add(this: Vector4d<Self>, k: Self, other: Vector4d<Self>) -> Vector4d<Self> {
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
            Vector4d {
                x: this.x * k + other.x,
                y: this.y * k + other.y,
                z: this.z * k + other.z,
                t: this.t * k + other.t,
            }
        }
    }

    #[inline(always)]
    fn v4_norm_squared(this: Vector4d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            (this_simd * this_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.x * this.x + this.y * this.y + this.z * this.z
        }
    }

    #[inline(always)]
    fn v4_normalize(this: Vector4d<Self>) -> Vector4d<Self> {
        #[cfg(feature = "simd")]
        {
            let norm_squared = Self::v4_dot(this, this);
            // If norm_squared is zero, then this must be zero (default) vector
            if norm_squared == 0.0 {
                return Vector4d::default();
            }

            let this_simd = f32x4::from(this);
            let norm_reciprocal = norm_squared.sqrt_reciprocal(); // Uses hardware vrsqrt
            let scale = f32x4::splat(norm_reciprocal);
            (this_simd * scale).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let norm_squared = Self::v4_norm_squared(this);
            if norm_squared == 0.0 {
                return Vector4d::default();
            }
            let norm_reciprocal = norm_squared.sqrt_reciprocal();
            Vector4d {
                x: this.x * norm_reciprocal,
                y: this.y * norm_reciprocal,
                z: this.z * norm_reciprocal,
                t: this.t * norm_reciprocal,
            }
        }
    }

    #[inline(always)]
    fn v4_is_normalized(this: Vector4d<Self>) -> bool {
        let norm_squared = Self::v4_norm_squared(this);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn v4_max(this: Vector4d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            this_simd.reduce_max()
        }
        #[cfg(not(feature = "simd"))]
        {
            if this.x > this.y {
                if this.x > this.z {
                    if this.x > this.t { this.x } else { this.t }
                } else {
                    if this.z > this.t { this.z } else { this.t }
                }
            } else {
                if this.y > this.z {
                    if this.y > this.t { this.y } else { this.t }
                } else {
                    if this.z > this.t { this.z } else { this.t }
                }
            }
        }
    }

    #[inline(always)]
    fn v4_min(this: Vector4d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            this_simd.reduce_min()
        }
        #[cfg(not(feature = "simd"))]
        {
            if this.x < this.y {
                if this.x < this.z {
                    if this.x < this.t { this.x } else { this.t }
                } else {
                    if this.z < this.t { this.z } else { this.t }
                }
            } else {
                if this.y < this.z {
                    if this.y < this.t { this.y } else { this.t }
                } else {
                    if this.z < this.t { this.z } else { this.t }
                }
            }
        }
    }

    // **** dot ****
    #[inline(always)]
    fn v4_dot(this: Vector4d<Self>, other: Vector4d<Self>) -> Self {
        //this.x * other.x + this.y * other.y + this.z * other.z
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);

            // Multiply the vectors, masking 4 lane to 0.0
            (this_simd * other_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.x * other.x + this.y * other.y + this.z * other.z + this.t * other.t
        }
    }
}

// **** f64 ****

impl Vector4dMath for f64 {
    #[inline(always)]
    fn v4_reciprocal(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn v4_neg(this: Vector4d<Self>) -> Vector4d<Self> {
        Vector4d { x: -this.x, y: -this.y, z: -this.z, t: -this.t }
    }

    #[inline(always)]
    fn v4_add(this: Vector4d<Self>, other: Vector4d<Self>) -> Vector4d<Self> {
        Vector4d { x: this.x + other.x, y: this.y + other.y, z: this.z + other.z, t: this.t + other.t }
    }

    #[inline(always)]
    fn v4_mul_scalar(this: Vector4d<Self>, k: Self) -> Vector4d<Self> {
        Vector4d { x: this.x * k, y: this.y * k, z: this.z * k, t: this.t * k }
    }

    #[inline(always)]
    fn v4_div_scalar(this: Vector4d<Self>, k: Self) -> Vector4d<Self> {
        Self::v4_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn v4_mul_add(this: Vector4d<Self>, k: Self, other: Vector4d<Self>) -> Vector4d<Self> {
        Vector4d { x: this.x * k + other.x, y: this.y * k + other.y, z: this.z * k + other.z, t: this.t * k + other.t }
    }

    #[inline(always)]
    fn v4_norm_squared(this: Vector4d<Self>) -> Self {
        this.x * this.x + this.y * this.y + this.z * this.z + this.t * this.t
    }

    #[inline(always)]
    fn v4_normalize(this: Vector4d<Self>) -> Vector4d<Self> {
        let norm_squared = Self::v4_norm_squared(this);
        if norm_squared == 0.0 {
            return Vector4d::default();
        }
        let norm_reciprocal = norm_squared.sqrt_reciprocal();
        Vector4d {
            x: this.x * norm_reciprocal,
            y: this.y * norm_reciprocal,
            z: this.z * norm_reciprocal,
            t: this.t * norm_reciprocal,
        }
    }

    #[inline(always)]
    fn v4_is_normalized(this: Vector4d<Self>) -> bool {
        let norm_squared = Self::v4_norm_squared(this);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn v4_max(this: Vector4d<Self>) -> Self {
        if this.x > this.y {
            if this.x > this.z {
                if this.x > this.t { this.x } else { this.t }
            } else {
                if this.z > this.t { this.z } else { this.t }
            }
        } else {
            if this.y > this.z {
                if this.y > this.t { this.y } else { this.t }
            } else {
                if this.z > this.t { this.z } else { this.t }
            }
        }
    }

    #[inline(always)]
    fn v4_min(this: Vector4d<Self>) -> Self {
        if this.x < this.y {
            if this.x < this.z {
                if this.x < this.t { this.x } else { this.t }
            } else {
                if this.z < this.t { this.z } else { this.t }
            }
        } else {
            if this.y < this.z {
                if this.y < this.t { this.y } else { this.t }
            } else {
                if this.z < this.t { this.z } else { this.t }
            }
        }
    }

    // **** dot ****
    #[inline(always)]
    fn v4_dot(this: Vector4d<Self>, other: Vector4d<Self>) -> Self {
        this.x * other.x + this.y * other.y + this.z * other.z + this.t * other.t
    }
}
