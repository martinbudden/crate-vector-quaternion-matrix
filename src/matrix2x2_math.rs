#![allow(clippy::inline_always)]
use cfg_if::cfg_if;
use core::mem::{align_of, size_of};

cfg_if! {
    if #[cfg(feature = "simd")] {
        use core::mem::transmute;
        use core::simd::{f32x2,f32x4,num::SimdFloat};
    }
}

const _: () = assert!(size_of::<Matrix2x2<f32>>() == 16);
const _: () = assert!(align_of::<Matrix2x2<f32>>() == 16);

use crate::{Matrix2x2, Vector2};

// Column 1
const M11: usize = 0;
const M21: usize = 1;
// Column 2
const M12: usize = 2;
const M22: usize = 3;

// **** From ****

#[cfg(feature = "simd")]
impl From<Matrix2x2<f32>> for f32x4 {
    #[inline(always)]
    fn from(v: Matrix2x2<f32>) -> Self {
        // SAFETY: assert f32x4 and Matrix2x2<f32> have same size and alignment
        const _: () = assert!(size_of::<f32x4>() == size_of::<Matrix2x2<f32>>());
        const _: () = assert!(size_of::<f32x4>() == align_of::<Matrix2x2<f32>>());
        unsafe { transmute(v) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x4> for Matrix2x2<f32> {
    #[inline(always)]
    fn from(simd: f32x4) -> Self {
        // SAFETY: assert f32x4 and Matrix2x2<f32> have same size and alignment
        const _: () = assert!(size_of::<f32x4>() == size_of::<Matrix2x2<f32>>());
        const _: () = assert!(size_of::<f32x4>() == align_of::<Matrix2x2<f32>>());
        unsafe { transmute(simd) }
    }
}

// **** Math ****

/// Math functions for Matrix2x2.
pub trait Matrix2x2Math: Sized {
    fn m2x2_neg(this: Matrix2x2<Self>) -> Matrix2x2<Self>;
    fn m2x2_abs(this: Matrix2x2<Self>) -> Matrix2x2<Self>;
    fn m2x2_add(this: Matrix2x2<Self>, this: Matrix2x2<Self>) -> Matrix2x2<Self>;
    fn m2x2_mul_scalar(this: Matrix2x2<Self>, other: Self) -> Matrix2x2<Self>;
    fn m2x2_div_scalar(this: Matrix2x2<Self>, other: Self) -> Matrix2x2<Self>;
    fn m2x2_mul_add(this: Matrix2x2<Self>, k: Self, other: Matrix2x2<Self>) -> Matrix2x2<Self>;
    fn m2x2_mul_vector(this: Matrix2x2<Self>, other: Vector2<Self>) -> Vector2<Self>;
    fn m2x2_vector_mul(this: Vector2<Self>, other: Matrix2x2<Self>) -> Vector2<Self>;
    fn m2x2_vector_outer_product(col: Vector2<Self>, row: Vector2<Self>) -> Matrix2x2<Self>;
    fn m2x2_mul(this: Matrix2x2<Self>, other: Matrix2x2<Self>) -> Matrix2x2<Self>;
    fn m2x2_trace(this: Matrix2x2<Self>) -> Self;
    fn m2x2_trace_sum_squares(this: Matrix2x2<Self>) -> Self;
    fn m2x2_sum(this: Matrix2x2<Self>) -> Self;
    fn m2x2_mean(this: Matrix2x2<Self>) -> Self;
    fn m2x2_product(this: Matrix2x2<Self>) -> Self;
    fn m2x2_determinant(this: Matrix2x2<Self>) -> Self;
    fn m2x2_adjugate(this: Matrix2x2<Self>) -> (Matrix2x2<Self>, Self);
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
            for r in &mut a {
                *r = -*r;
            }
            Matrix2x2 { a }
        }
    }

    #[inline(always)]
    fn m2x2_abs(this: Matrix2x2<Self>) -> Matrix2x2<Self> {
        let a = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix2x2 { a }
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
            Matrix2x2 { a }
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
            for r in &mut a {
                *r *= other;
            }
            Matrix2x2 { a }
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
    fn m2x2_mul_vector(this: Matrix2x2<Self>, other: Vector2<Self>) -> Vector2<Self> {
        Vector2 { x: this.a[M11] * other.x + this.a[M12] * other.y, y: this.a[M21] * other.x + this.a[M22] * other.y }
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m2x2_vector_mul(this: Vector2<Self>, other: Matrix2x2<Self>) -> Vector2<Self> {
        Vector2 {
            x: this.x * other.a[M11] + this.y * other.a[M21],
            y: this.x * other.a[M12] + this.y * other.a[M22]
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m2x2_vector_outer_product(col: Vector2<Self>, row: Vector2<Self>) -> Matrix2x2<Self> {
        Matrix2x2 {
            a: [
                col.x * row.x, col.y * row.x,
                col.x * row.y, col.y * row.y,
            ],
        }
    }

    #[inline(always)]
    fn m2x2_mul(this: Matrix2x2<Self>, other: Matrix2x2<Self>) -> Matrix2x2<Self> {
        #[cfg(feature = "simd")]
        {
            let a0_simd = f32x2::from_array([this.a[M11], this.a[M12]]);
            let a1_simd = f32x2::from_array([this.a[M21], this.a[M22]]);
            let b0_simd = f32x2::from_array([other.a[M11], other.a[M21]]);
            let b1_simd = f32x2::from_array([other.a[M12], other.a[M22]]);
            let a = [
                (a0_simd * b0_simd).reduce_sum(),
                (a1_simd * b0_simd).reduce_sum(),
                (a0_simd * b1_simd).reduce_sum(),
                (a1_simd * b1_simd).reduce_sum(),
            ];
            Matrix2x2 { a }
        }
        #[cfg(not(feature = "simd"))]
        {
            let a = [
                this.a[M11] * other.a[M11] + this.a[M12] * other.a[M21],
                this.a[M21] * other.a[M11] + this.a[M22] * other.a[M21],
                this.a[M11] * other.a[M12] + this.a[M12] * other.a[M22],
                this.a[M21] * other.a[M12] + this.a[M22] * other.a[M22],
            ];
            Matrix2x2 { a }
        }
    }

    #[inline(always)]
    fn m2x2_trace(this: Matrix2x2<Self>) -> Self {
        this.a[M11] + this.a[M22]
    }

    #[inline(always)]
    fn m2x2_trace_sum_squares(this: Matrix2x2<Self>) -> Self {
        this.a[M11] * this.a[M11] + this.a[M22] * this.a[M22]
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
    fn m2x2_determinant(this: Matrix2x2<Self>) -> Self {
        this.a[M11] * this.a[M22] - this.a[M12] * this.a[M21]
    }

    #[inline(always)]
    fn m2x2_adjugate(this: Matrix2x2<Self>) -> (Matrix2x2<Self>, Self) {
        (
            Matrix2x2 { a: [this.a[M22], -this.a[M21], -this.a[M12], this.a[M11]] },
            this.a[M11] * this.a[M22] - this.a[M12] * this.a[M21],
        )
    }
}

// **** f64 ****

impl Matrix2x2Math for f64 {
    #[inline(always)]
    fn m2x2_neg(this: Matrix2x2<Self>) -> Matrix2x2<Self> {
        let mut a = this.a;
        for r in &mut a {
            *r = -*r;
        }
        Matrix2x2 { a }
    }

    #[inline(always)]
    fn m2x2_abs(this: Matrix2x2<Self>) -> Matrix2x2<Self> {
        let mut a = this.a;
        for r in &mut a {
            *r = r.abs();
        }
        Matrix2x2 { a }
    }

    #[inline(always)]
    fn m2x2_add(this: Matrix2x2<Self>, other: Matrix2x2<Self>) -> Matrix2x2<Self> {
        let mut a = this.a;
        for (ii, r) in a.iter_mut().enumerate() {
            *r += other.a[ii];
        }
        Matrix2x2 { a }
    }

    #[inline(always)]
    fn m2x2_mul_scalar(this: Matrix2x2<Self>, other: Self) -> Matrix2x2<Self> {
        let mut a = this.a;
        for r in &mut a {
            *r *= other;
        }
        Matrix2x2 { a }
    }

    #[inline(always)]
    fn m2x2_div_scalar(this: Matrix2x2<Self>, other: Self) -> Matrix2x2<Self> {
        Self::m2x2_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m2x2_mul_add(this: Matrix2x2<Self>, k: Self, other: Matrix2x2<Self>) -> Matrix2x2<Self> {
        Self::m2x2_add(Self::m2x2_mul_scalar(this, k), other)
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m2x2_mul_vector(this: Matrix2x2<Self>, other: Vector2<Self>) -> Vector2<Self> {
        Vector2 {
            x: this.a[M11] * other.x + this.a[M21] * other.y,
            y: this.a[M12] * other.x + this.a[M22] * other.y
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m2x2_vector_mul(this: Vector2<Self>, other: Matrix2x2<Self>) -> Vector2<Self> {
        Vector2 {
            x: this.x * other.a[M11] + this.y * other.a[M21],
            y: this.x * other.a[M12] + this.y * other.a[M22]
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m2x2_vector_outer_product(col: Vector2<Self>, row: Vector2<Self>) -> Matrix2x2<Self> {
        Matrix2x2 {
            a: [
                col.x * row.x, col.y * row.x,
                col.x * row.y, col.y * row.y,
            ],
        }
    }

    #[inline(always)]
    fn m2x2_mul(this: Matrix2x2<Self>, other: Matrix2x2<Self>) -> Matrix2x2<Self> {
        let a = [
            this.a[M11] * other.a[M11] + this.a[M12] * other.a[M21],
            this.a[M11] * other.a[M12] + this.a[M12] * other.a[M22],
            this.a[M21] * other.a[M11] + this.a[M22] * other.a[M21],
            this.a[M21] * other.a[M12] + this.a[M22] * other.a[M22],
        ];
        Matrix2x2 { a }
    }

    #[inline(always)]
    fn m2x2_trace(this: Matrix2x2<Self>) -> Self {
        this.a[M11] + this.a[M22]
    }

    #[inline(always)]
    fn m2x2_trace_sum_squares(this: Matrix2x2<Self>) -> Self {
        this.a[M11] * this.a[M11] + this.a[M22] * this.a[M22]
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
    fn m2x2_determinant(this: Matrix2x2<Self>) -> Self {
        this.a[M11] * this.a[M22] - this.a[M12] * this.a[M21]
    }

    #[inline(always)]
    fn m2x2_adjugate(this: Matrix2x2<Self>) -> (Matrix2x2<Self>, Self) {
        (
            Matrix2x2 { a: [this.a[M22], -this.a[M21], -this.a[M12], this.a[M11]] },
            this.a[M11] * this.a[M22] - this.a[M12] * this.a[M21],
        )
    }
}
