use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "simd")] {
        use core::{mem::transmute};
        use core::simd::{f32x2,f32x4,num::SimdFloat};
    }
}

const _: () = assert!(core::mem::size_of::<Matrix2x2<f32>>() == 16);
const _: () = assert!(core::mem::align_of::<Matrix2x2<f32>>() == 16);

use crate::{Matrix2x2, Vector2d};

// **** From ****

#[cfg(feature = "simd")]
impl From<Matrix2x2<f32>> for f32x4 {
    #[inline(always)]
    fn from(v: Matrix2x2<f32>) -> Self {
        // SAFETY: assert f32x4 and Matrix2x2<f32> have same size and alignment
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::size_of::<Matrix2x2<f32>>());
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::align_of::<Matrix2x2<f32>>());
        unsafe { transmute(v) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x4> for Matrix2x2<f32> {
    #[inline(always)]
    fn from(simd: f32x4) -> Self {
        // SAFETY: assert f32x4 and Matrix2x2<f32> have same size and alignment
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::size_of::<Matrix2x2<f32>>());
        const _: () = assert!(core::mem::size_of::<f32x4>() == core::mem::align_of::<Matrix2x2<f32>>());
        unsafe { transmute(simd) }
    }
}

// **** Math ****

/// Math functions for Matrix2x2, using **SIMD** accelerations for `f32`.
pub trait Matrix2x2Math: Sized {
    fn m2x2_neg(this: Matrix2x2<Self>) -> Matrix2x2<Self>;
    fn m2x2_abs(this: Matrix2x2<Self>) -> Matrix2x2<Self>;
    fn m2x2_add(this: Matrix2x2<Self>, this: Matrix2x2<Self>) -> Matrix2x2<Self>;
    fn m2x2_mul_scalar(this: Matrix2x2<Self>, other: Self) -> Matrix2x2<Self>;
    fn m2x2_div_scalar(this: Matrix2x2<Self>, other: Self) -> Matrix2x2<Self>;
    fn m2x2_mul_add(this: Matrix2x2<Self>, k: Self, other: Matrix2x2<Self>) -> Matrix2x2<Self>;
    fn m2x2_mul_vector(this: Matrix2x2<Self>, other: Vector2d<Self>) -> Vector2d<Self>;
    fn m2x2_vector_mul(this: Vector2d<Self>, other: Matrix2x2<Self>) -> Vector2d<Self>;
    fn m2x2_mul(this: Matrix2x2<Self>, other: Matrix2x2<Self>) -> Matrix2x2<Self>;
    fn m2x2_determinant(this: Matrix2x2<Self>) -> Self;
    fn m2x2_top_right_determinant(this: Matrix2x2<Self>) -> Self;
    fn m2x2_top_right_sum_squares(this: Matrix2x2<Self>) -> Self;
    fn m2x2_trace(this: Matrix2x2<Self>) -> Self;
    fn m2x2_trace_sum_squares(this: Matrix2x2<Self>) -> Self;
    fn m2x2_sum(this: Matrix2x2<Self>) -> Self;
    fn m2x2_mean(this: Matrix2x2<Self>) -> Self;
    fn m2x2_product(this: Matrix2x2<Self>) -> Self;
    fn m2x2_adjugate(this: Matrix2x2<Self>) -> Matrix2x2<Self>;
}

impl Matrix2x2Math for f32 {
    #[inline(always)]
    fn m2x2_neg(this: Matrix2x2<Self>) -> Matrix2x2<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);

            (-this_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut a = this.a;
            for r in a.iter_mut() {
                *r = -*r;
            }
            Matrix2x2::from(a)
        }
    }

    #[inline(always)]
    fn m2x2_abs(this: Matrix2x2<Self>) -> Matrix2x2<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix2x2::from(ret)
    }

    #[inline(always)]
    fn m2x2_add(this: Matrix2x2<Self>, other: Matrix2x2<Self>) -> Matrix2x2<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::from(other);

            // Add all 4 lanes (w, x, y, filler) in one cycle
            (this_simd + other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut a = this.a;
            for (ii, r) in a.iter_mut().enumerate() {
                *r += other.a[ii];
            }
            Matrix2x2::from(a)
        }
    }

    #[inline(always)]
    fn m2x2_mul_scalar(this: Matrix2x2<Self>, other: Self) -> Matrix2x2<Self> {
        #[cfg(feature = "simd")]
        {
            let this_simd = f32x4::from(this);
            let other_simd = f32x4::splat(other);

            (this_simd * other_simd).into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut a = this.a;
            for r in a.iter_mut() {
                *r *= other;
            }
            Matrix2x2::from(a)
        }
    }

    #[inline(always)]
    fn m2x2_div_scalar(this: Matrix2x2<Self>, other: Self) -> Matrix2x2<Self> {
        Self::m2x2_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m2x2_mul_add(this: Matrix2x2<Self>, k: Self, other: Matrix2x2<Self>) -> Matrix2x2<Self> {
        Self::m2x2_add(Self::m2x2_mul_scalar(this, k), other)
    }

    #[inline(always)]
    fn m2x2_mul_vector(this: Matrix2x2<Self>, other: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: this.a[0] * other.x + this.a[1] * other.y, y: this.a[2] * other.x + this.a[3] * other.y }
    }

    #[inline(always)]
    fn m2x2_vector_mul(this: Vector2d<Self>, other: Matrix2x2<Self>) -> Vector2d<Self> {
        Vector2d { x: this.x * other.a[0] + this.y * other.a[2], y: this.x * other.a[1] + this.y * other.a[3] }
    }

    #[inline(always)]
    fn m2x2_mul(this: Matrix2x2<Self>, other: Matrix2x2<Self>) -> Matrix2x2<Self> {
        #[cfg(feature = "simd")]
        {
            let a0_simd = f32x2::from_array([this.a[0], this.a[1]]);
            let a1_simd = f32x2::from_array([this.a[2], this.a[3]]);
            let b0_simd = f32x2::from_array([other.a[0], other.a[2]]);
            let b1_simd = f32x2::from_array([other.a[1], other.a[3]]);
            let a = [
                (a0_simd * b0_simd).reduce_sum(),
                (a0_simd * b1_simd).reduce_sum(),
                (a1_simd * b0_simd).reduce_sum(),
                (a1_simd * b1_simd).reduce_sum(),
            ];

            Matrix2x2::from(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            let a = [
                this.a[0] * other.a[0] + this.a[1] * other.a[2],
                this.a[0] * other.a[1] + this.a[1] * other.a[3],
                this.a[2] * other.a[0] + this.a[3] * other.a[2],
                this.a[2] * other.a[1] + this.a[3] * other.a[3],
            ];

            Matrix2x2::from(a)
        }
    }

    #[inline(always)]
    fn m2x2_determinant(this: Matrix2x2<Self>) -> Self {
        this.a[0] * this.a[3] - this.a[1] * this.a[2]
    }

    #[inline(always)]
    fn m2x2_top_right_determinant(this: Matrix2x2<Self>) -> Self {
        this.a[0] * this.a[3] - this.a[1] * this.a[1]
    }

    #[inline(always)]
    fn m2x2_top_right_sum_squares(this: Matrix2x2<Self>) -> Self {
        this.a[1] * this.a[1]
    }

    #[inline(always)]
    fn m2x2_trace(this: Matrix2x2<Self>) -> Self {
        this.a[0] + this.a[3]
    }

    #[inline(always)]
    fn m2x2_trace_sum_squares(this: Matrix2x2<Self>) -> Self {
        this.a[0] * this.a[0] + this.a[3] * this.a[3]
    }

    #[inline(always)]
    fn m2x2_sum(this: Matrix2x2<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m2x2_mean(this: Matrix2x2<Self>) -> Self {
        this.sum() / 4.0
    }

    #[inline(always)]
    fn m2x2_product(this: Matrix2x2<Self>) -> Self {
        this.a.iter().product()
    }
    #[inline(always)]
    fn m2x2_adjugate(this: Matrix2x2<Self>) -> Matrix2x2<Self> {
        Matrix2x2::from([this.a[3], -this.a[1], -this.a[2], this.a[0]])
    }
}

// **** f64 ****

impl Matrix2x2Math for f64 {
    #[inline(always)]
    fn m2x2_neg(this: Matrix2x2<Self>) -> Matrix2x2<Self> {
        let mut a = this.a;
        for r in a.iter_mut() {
            *r = -*r;
        }
        Matrix2x2::from(a)
    }

    #[inline(always)]
    fn m2x2_abs(this: Matrix2x2<Self>) -> Matrix2x2<Self> {
        let mut a = this.a;
        for r in a.iter_mut() {
            *r = r.abs();
        }
        Matrix2x2::from(a)
    }

    #[inline(always)]
    fn m2x2_add(this: Matrix2x2<Self>, other: Matrix2x2<Self>) -> Matrix2x2<Self> {
        let mut a = this.a;
        for (ii, r) in a.iter_mut().enumerate() {
            *r += other.a[ii];
        }
        Matrix2x2::from(a)
    }

    #[inline(always)]
    fn m2x2_mul_scalar(this: Matrix2x2<Self>, other: Self) -> Matrix2x2<Self> {
        let mut a = this.a;
        for r in a.iter_mut() {
            *r *= other;
        }
        Matrix2x2::from(a)
    }

    #[inline(always)]
    fn m2x2_div_scalar(this: Matrix2x2<Self>, other: Self) -> Matrix2x2<Self> {
        Self::m2x2_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m2x2_mul_add(this: Matrix2x2<Self>, k: Self, other: Matrix2x2<Self>) -> Matrix2x2<Self> {
        Self::m2x2_add(Self::m2x2_mul_scalar(this, k), other)
    }

    #[inline(always)]
    fn m2x2_mul_vector(this: Matrix2x2<Self>, other: Vector2d<Self>) -> Vector2d<Self> {
        Vector2d { x: this.a[0] * other.x + this.a[2] * other.y, y: this.a[1] * other.x + this.a[3] * other.y }
    }

    #[inline(always)]
    fn m2x2_vector_mul(this: Vector2d<Self>, other: Matrix2x2<Self>) -> Vector2d<Self> {
        Vector2d { x: this.x * other.a[0] + this.y * other.a[2], y: this.x * other.a[1] + this.y * other.a[3] }
    }

    #[inline(always)]
    fn m2x2_mul(this: Matrix2x2<Self>, other: Matrix2x2<Self>) -> Matrix2x2<Self> {
        let a = [
            this.a[0] * other.a[0] + this.a[1] * other.a[2],
            this.a[0] * other.a[1] + this.a[1] * other.a[3],
            this.a[2] * other.a[0] + this.a[3] * other.a[2],
            this.a[2] * other.a[1] + this.a[3] * other.a[3],
        ];
        Matrix2x2::from(a)
    }

    #[inline(always)]
    fn m2x2_determinant(this: Matrix2x2<Self>) -> Self {
        this.a[0] * this.a[3] - this.a[1] * this.a[2]
    }

    #[inline(always)]
    fn m2x2_top_right_determinant(this: Matrix2x2<Self>) -> Self {
        this.a[0] * this.a[3] - this.a[1] * this.a[1]
    }

    #[inline(always)]
    fn m2x2_top_right_sum_squares(this: Matrix2x2<Self>) -> Self {
        this.a[1] * this.a[1]
    }

    #[inline(always)]
    fn m2x2_trace(this: Matrix2x2<Self>) -> Self {
        this.a[0] + this.a[3]
    }

    #[inline(always)]
    fn m2x2_trace_sum_squares(this: Matrix2x2<Self>) -> Self {
        this.a[0] * this.a[0] + this.a[3] * this.a[3]
    }

    #[inline(always)]
    fn m2x2_sum(this: Matrix2x2<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m2x2_mean(this: Matrix2x2<Self>) -> Self {
        this.sum() / 4.0
    }

    #[inline(always)]
    fn m2x2_product(this: Matrix2x2<Self>) -> Self {
        this.a.iter().product()
    }
    #[inline(always)]
    fn m2x2_adjugate(this: Matrix2x2<Self>) -> Matrix2x2<Self> {
        Matrix2x2::from([this.a[3], -this.a[1], -this.a[2], this.a[0]])
    }
}
