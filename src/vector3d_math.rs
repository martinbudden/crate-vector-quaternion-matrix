use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "simd")] {
        use core::{mem::transmute};
        use core::simd::{f32x4,num::SimdFloat,simd_swizzle};
        // must be aligned if using SIMD
        const _: () = assert!(core::mem::size_of::<Vector3d<f32>>() == 16);
        const _: () = assert!(core::mem::align_of::<Vector3d<f32>>() == 16);
    } else if #[cfg(feature = "no_align")] {
        const _: () = assert!(core::mem::size_of::<Vector3d<f32>>() == 12);
        const _: () = assert!(core::mem::align_of::<Vector3d<f32>>() == 4);
    } else {
        const _: () = assert!(core::mem::size_of::<Vector3d<f32>>() == 16);
        const _: () = assert!(core::mem::align_of::<Vector3d<f32>>() == 16);
    }
}

use crate::{SqrtMethods, Vector3d};

// **** From ****

#[cfg(feature = "simd")]
impl From<Vector3d<f32>> for f32x4 {
    #[inline(always)]
    fn from(this: Vector3d<f32>) -> Self {
        // SAFETY: assert f32x4 and Vector3d<f32> have same size and alignment
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::size_of::<Vector3d<f32>>());
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::align_of::<Vector3d<f32>>());
        // The 'filler' 4th float in the SIMD lane will be whatever was in the padding (usually 0.0 if set by Default).
        unsafe { transmute(this) }
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

/// Math functions for Vector3d, using **SIMD** accelerations for `f32`.<br><br>
pub trait Vector3dMath: Sized {
    fn v3_neg(this: Vector3d<Self>) -> Vector3d<Self>;
    fn v3_add(this: Vector3d<Self>, this: Vector3d<Self>) -> Vector3d<Self>;
    fn v3_mul_scalar(this: Vector3d<Self>, k: Self) -> Vector3d<Self>;
    fn v3_div_scalar(this: Vector3d<Self>, k: Self) -> Vector3d<Self>;
    fn v3_mul_add(this: Vector3d<Self>, k: Self, other: Vector3d<Self>) -> Vector3d<Self>;
    fn v3_norm_squared(this: Vector3d<Self>) -> Self;
    fn v3_normalize(this: Vector3d<Self>) -> Vector3d<Self>;
    fn v3_is_normalized(this: Vector3d<Self>) -> bool;
    fn v3_max(this: Vector3d<Self>) -> Self;
    fn v3_min(this: Vector3d<Self>) -> Self;
    fn v3_dot(this: Vector3d<Self>, other: Vector3d<Self>) -> Self;
    fn v3_cross(this: Vector3d<Self>, other: Vector3d<Self>) -> Vector3d<Self>;
}

// **** SIMD-accelerated implementation for f32 ****

impl Vector3dMath for f32 {
    #[inline(always)]
    fn v3_neg(this: Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            (-f32x4::from(this)).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3d { x: -this.x, y: -this.y, z: -this.z }
        }
    }

    #[inline(always)]
    fn v3_add(this: Vector3d<Self>, other: Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);

            (this_simd + other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3d { x: this.x + other.x, y: this.y + other.y, z: this.z + other.z }
        }
    }

    #[inline(always)]
    fn v3_mul_scalar(this: Vector3d<Self>, k: Self) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let k_simd = f32x4::splat(k);

            (this_simd * k_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3d { x: this.x * k, y: this.y * k, z: this.z * k }
        }
    }

    #[inline(always)]
    fn v3_div_scalar(this: Vector3d<Self>, k: Self) -> Vector3d<Self> {
        Self::v3_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn v3_mul_add(this: Vector3d<Self>, k: Self, other: Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);
            let k_simd = f32x4::splat(k) * f32x4::from_array([1.0, 1.0, 1.0, 0.0]);

            // This maps to the Vector Fused Multiply-Add instruction
            ((this_simd * k_simd) + other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3d { x: this.x * k + other.x, y: this.y * k + other.y, z: this.z * k + other.z }
        }
    }

    #[inline(always)]
    fn v3_norm_squared(this: Vector3d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this) * f32x4::from_array([1.0, 1.0, 1.0, 0.0]);
            (this_simd * this_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.x * this.x + this.y * this.y + this.z * this.z
        }
    }

    #[inline(always)]
    fn v3_normalize(this: Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            let norm_squared = Self::v3_norm_squared(this);
            // If norm_squared is zero, then this must be zero (default) vector
            if norm_squared == 0.0 {
                return Vector3d::default();
            }

            let this_simd = f32x4::from(this);
            let norm_reciprocal = norm_squared.sqrt_reciprocal(); // Uses hardware vrsqrt
            let scale = f32x4::splat(norm_reciprocal);
            (this_simd * scale).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let norm_squared = Self::v3_norm_squared(this);
            if norm_squared == 0.0 {
                return Vector3d::default();
            }
            let norm_reciprocal = norm_squared.sqrt_reciprocal();
            Vector3d { x: this.x * norm_reciprocal, y: this.y * norm_reciprocal, z: this.z * norm_reciprocal }
        }
    }

    #[inline(always)]
    fn v3_is_normalized(this: Vector3d<Self>) -> bool {
        let norm_squared = Self::v3_norm_squared(this);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn v3_max(this: Vector3d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            // repeat this.z in final lane to allow reduce_max to work correctly
            let this_simd = f32x4::from_array([this.x, this.y, this.z, this.z]);
            this_simd.reduce_max()
        }
        #[cfg(not(feature = "simd"))]
        {
            if this.x > this.y {
                if this.x > this.z { this.x } else { this.z }
            } else {
                if this.y > this.z { this.y } else { this.z }
            }
        }
    }

    #[inline(always)]
    fn v3_min(this: Vector3d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            // repeat this.z in final lane to allow reduce_min to work correctly
            let this_simd = f32x4::from_array([this.x, this.y, this.z, this.z]);
            this_simd.reduce_min()
        }
        #[cfg(not(feature = "simd"))]
        {
            if this.x < this.y {
                if this.x < this.z { this.x } else { this.z }
            } else {
                if this.y < this.z { this.y } else { this.z }
            }
        }
    }

    // **** dot ****
    #[inline(always)]
    fn v3_dot(this: Vector3d<Self>, other: Vector3d<Self>) -> Self {
        //this.x * other.x + this.y * other.y + this.z * other.z
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);

            // Multiply the vectors, masking 4 lane to 0.0
            let product = (this_simd * other_simd) * f32x4::from_array([1.0, 1.0, 1.0, 0.0]);

            product.reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.x * other.x + this.y * other.y + this.z * other.z
        }
    }

    #[inline(always)]
    fn v3_cross(this: Vector3d<Self>, other: Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);

            // Swizzle: [y, z, x, w]
            let this_yzx = simd_swizzle!(this_simd, [1, 2, 0, 3]);
            // Swizzle: [z, x, y, w]
            let other_zxy = simd_swizzle!(other_simd, [2, 0, 1, 3]);

            // Swizzle: [z, x, y, w]
            let this_zxy = simd_swizzle!(this_simd, [2, 0, 1, 3]);
            // Swizzle: [y, z, x, w]
            let other_yzx = simd_swizzle!(other_simd, [1, 2, 0, 3]);

            // Result = (a_yzx * b_zxy) - (a_zxy * b_yzx)
            let ret_simd = this_yzx * other_zxy - this_zxy * other_yzx;

            // Transmute back to our Vector3d struct
            ret_simd.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3d {
                x: this.y * other.z - this.z * other.y,
                y: this.z * other.x - this.x * other.z,
                z: this.x * other.y - this.y * other.x,
            }
        }
    }
}

// **** f64 ****

impl Vector3dMath for f64 {
    #[inline(always)]
    fn v3_neg(this: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d { x: -this.x, y: -this.y, z: -this.z }
    }

    #[inline(always)]
    fn v3_add(this: Vector3d<Self>, other: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d { x: this.x + other.x, y: this.y + other.y, z: this.z + other.z }
    }

    #[inline(always)]
    fn v3_mul_scalar(this: Vector3d<Self>, k: Self) -> Vector3d<Self> {
        Vector3d { x: this.x * k, y: this.y * k, z: this.z * k }
    }

    #[inline(always)]
    fn v3_div_scalar(this: Vector3d<Self>, k: Self) -> Vector3d<Self> {
        Self::v3_mul_scalar(this, 1.0 / k)
    }

    #[inline(always)]
    fn v3_mul_add(this: Vector3d<Self>, k: Self, other: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d { x: this.x * k + other.x, y: this.y * k + other.y, z: this.z * k + other.z }
    }

    #[inline(always)]
    fn v3_norm_squared(this: Vector3d<Self>) -> Self {
        this.x * this.x + this.y * this.y + this.z * this.z
    }

    #[inline(always)]
    fn v3_normalize(this: Vector3d<Self>) -> Vector3d<Self> {
        let norm_squared = Self::v3_norm_squared(this);
        if norm_squared == 0.0 {
            return Vector3d::default();
        }
        let norm_reciprocal = norm_squared.sqrt_reciprocal();
        Vector3d { x: this.x * norm_reciprocal, y: this.y * norm_reciprocal, z: this.z * norm_reciprocal }
    }

    #[inline(always)]
    fn v3_is_normalized(this: Vector3d<Self>) -> bool {
        let norm_squared = Self::v3_norm_squared(this);
        approx::abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-6)
    }

    #[inline(always)]
    fn v3_max(this: Vector3d<Self>) -> Self {
        if this.x > this.y {
            if this.x > this.z { this.x } else { this.z }
        } else {
            if this.y > this.z { this.y } else { this.z }
        }
    }

    #[inline(always)]
    fn v3_min(this: Vector3d<Self>) -> Self {
        if this.x < this.y {
            if this.x < this.z { this.x } else { this.z }
        } else {
            if this.y < this.z { this.y } else { this.z }
        }
    }

    // **** dot ****
    #[inline(always)]
    fn v3_dot(this: Vector3d<Self>, other: Vector3d<Self>) -> Self {
        this.x * other.x + this.y * other.y + this.z * other.z
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

impl Vector3dMath for i16 {
    #[inline(always)]
    fn v3_neg(this: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d { x: -this.x, y: -this.y, z: -this.z }
    }

    #[inline(always)]
    fn v3_add(this: Vector3d<Self>, other: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d { x: this.x + other.x, y: this.y + other.y, z: this.z + other.z }
    }

    #[inline(always)]
    fn v3_mul_scalar(this: Vector3d<Self>, k: Self) -> Vector3d<Self> {
        Vector3d { x: this.x * k, y: this.y * k, z: this.z * k }
    }

    #[inline(always)]
    fn v3_div_scalar(this: Vector3d<Self>, k: Self) -> Vector3d<Self> {
        Self::v3_mul_scalar(this, (1.0 / (k as f32)) as i16)
    }

    #[inline(always)]
    fn v3_mul_add(this: Vector3d<Self>, k: Self, other: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d { x: this.x * k + other.x, y: this.y * k + other.y, z: this.z * k + other.z }
    }

    #[inline(always)]
    fn v3_norm_squared(this: Vector3d<Self>) -> Self {
        this.x * this.x + this.y * this.y + this.z * this.z
    }

    #[inline(always)]
    fn v3_normalize(this: Vector3d<Self>) -> Vector3d<Self> {
        let norm_squared = Self::v3_norm_squared(this);
        if norm_squared == 0 {
            return Vector3d::default();
        }
        let norm_reciprocal = (norm_squared as f32).sqrt_reciprocal();
        Vector3d {
            x: ((this.x as f32) * norm_reciprocal) as i16,
            y: ((this.y as f32) * norm_reciprocal) as i16,
            z: ((this.z as f32) * norm_reciprocal) as i16,
        }
    }

    #[inline(always)]
    fn v3_is_normalized(this: Vector3d<Self>) -> bool {
        let norm_squared = Self::v3_norm_squared(this);
        norm_squared == 1
    }

    #[inline(always)]
    fn v3_max(this: Vector3d<Self>) -> Self {
        if this.x > this.y {
            if this.x > this.z { this.x } else { this.z }
        } else {
            if this.y > this.z { this.y } else { this.z }
        }
    }

    #[inline(always)]
    fn v3_min(this: Vector3d<Self>) -> Self {
        if this.x < this.y {
            if this.x < this.z { this.x } else { this.z }
        } else {
            if this.y < this.z { this.y } else { this.z }
        }
    }

    #[inline(always)]
    fn v3_dot(this: Vector3d<Self>, other: Vector3d<Self>) -> Self {
        this.x * other.x + this.y * other.y + this.z * other.z
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
