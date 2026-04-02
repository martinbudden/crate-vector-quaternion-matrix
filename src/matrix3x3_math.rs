#[cfg(feature = "simd")]
use cfg_if::cfg_if;
cfg_if! {
    if #[cfg(feature = "simd")] {
        //use core::{mem::transmute, simd::f32x4};
    }
}

use crate::{Matrix3x3, Vector3d};

// **** Math ****

/// Math functions for Matrix3x3, using SIMD accelerations for f32.
pub trait Matrix3x3Math: Sized {
    fn m3x3_reciprocal(self) -> Self;
    fn m3x3_neg(this: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_abs(this: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_add(this: Matrix3x3<Self>, this: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_mul_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self>;
    fn m3x3_div_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self>;
    fn m3x3_mul_vector(this: Matrix3x3<Self>, other: Vector3d<Self>) -> Vector3d<Self>;
    fn m3x3_vector_mul(this: Vector3d<Self>, other: Matrix3x3<Self>) -> Vector3d<Self>;
    fn m3x3_mul(this: Matrix3x3<Self>, other: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_determinant(this: Matrix3x3<Self>) -> Self;
    fn m3x3_top_right_determinant(this: Matrix3x3<Self>) -> Self;
    fn m3x3_top_right_sum_squares(this: Matrix3x3<Self>) -> Self;
    fn m3x3_trace(this: Matrix3x3<Self>) -> Self;
    fn m3x3_trace_sum_squares(this: Matrix3x3<Self>) -> Self;
    fn m3x3_sum(this: Matrix3x3<Self>) -> Self;
    fn m3x3_mean(this: Matrix3x3<Self>) -> Self;
    fn m3x3_product(this: Matrix3x3<Self>) -> Self;
    fn m3x3_adjugate(this: Matrix3x3<Self>) -> Matrix3x3<Self>;
}

impl Matrix3x3Math for f32 {
    #[inline(always)]
    fn m3x3_reciprocal(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn m3x3_neg(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let mut a = this.a;
        for r in a.iter_mut() {
            *r = -*r;
        }
        Matrix3x3::from(a)
    }

    #[inline(always)]
    fn m3x3_abs(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let mut a = this.a;
        for r in a.iter_mut() {
            *r = r.abs();
        }
        Matrix3x3::from(a)
    }

    #[inline(always)]
    fn m3x3_add(this: Matrix3x3<Self>, other: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let mut a = this.a;
        for (ii, r) in a.iter_mut().enumerate() {
            *r += other.a[ii];
        }
        Matrix3x3::from(a)
    }

    #[inline(always)]
    fn m3x3_mul_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self> {
        let mut a = this.a;
        for r in a.iter_mut() {
            *r *= other;
        }
        Matrix3x3::from(a)
    }

    #[inline(always)]
    fn m3x3_div_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self> {
        Self::m3x3_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m3x3_mul_vector(this: Matrix3x3<Self>, other: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d {
            x: this.a[0] * other.x + this.a[1] * other.y + this.a[2] * other.z,
            y: this.a[3] * other.x + this.a[4] * other.y + this.a[5] * other.z,
            z: this.a[6] * other.x + this.a[7] * other.y + this.a[8] * other.z,
        }
    }

    #[inline(always)]
    fn m3x3_vector_mul(this: Vector3d<Self>, other: Matrix3x3<Self>) -> Vector3d<Self> {
        Vector3d {
            x: this.x * other.a[0] + this.y * other.a[3] + this.z * other.a[6],
            y: this.x * other.a[1] + this.y * other.a[4] + this.z * other.a[7],
            z: this.x * other.a[2] + this.y * other.a[5] + this.z * other.a[8],
        }
    }

    #[inline(always)]
    fn m3x3_mul(this: Matrix3x3<Self>, other: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let a = [
            this.a[0] * other.a[0] + this.a[1] * other.a[3] + this.a[2] * other.a[6],
            this.a[0] * other.a[1] + this.a[1] * other.a[4] + this.a[2] * other.a[7],
            this.a[0] * other.a[2] + this.a[1] * other.a[5] + this.a[2] * other.a[8],
            this.a[3] * other.a[0] + this.a[4] * other.a[3] + this.a[5] * other.a[6],
            this.a[3] * other.a[1] + this.a[4] * other.a[4] + this.a[5] * other.a[7],
            this.a[3] * other.a[2] + this.a[4] * other.a[5] + this.a[5] * other.a[8],
            this.a[6] * other.a[0] + this.a[7] * other.a[3] + this.a[8] * other.a[6],
            this.a[6] * other.a[1] + this.a[7] * other.a[4] + this.a[8] * other.a[7],
            this.a[6] * other.a[2] + this.a[7] * other.a[5] + this.a[8] * other.a[8],
        ];
        Matrix3x3::from(a)
    }

    #[inline(always)]
    fn m3x3_determinant(this: Matrix3x3<Self>) -> Self {
        this.a[0] * (this.a[4] * this.a[8] - this.a[5] * this.a[7])
            - this.a[1] * (this.a[3] * this.a[8] - this.a[5] * this.a[6])
            + this.a[2] * (this.a[3] * this.a[7] - this.a[4] * this.a[6])
    }

    #[inline(always)]
    fn m3x3_top_right_determinant(this: Matrix3x3<Self>) -> Self {
        //let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
        //             0     4     8     5      5      1      1     8    5     2        2     1    5      4     2
        this.a[0] * (this.a[4] * this.a[8] - this.a[5] * this.a[5])
            - this.a[1] * (this.a[1] * this.a[8] - this.a[5] * this.a[2])
            + this.a[2] * (this.a[1] * this.a[5] - this.a[4] * this.a[2])
    }

    #[inline(always)]
    fn m3x3_top_right_sum_squares(this: Matrix3x3<Self>) -> Self {
        this.a[1] * this.a[1] + this.a[2] * this.a[2] + this.a[5] * this.a[5]
    }

    #[inline(always)]
    fn m3x3_trace(this: Matrix3x3<Self>) -> Self {
        this.a[0] + this.a[4] + this.a[8]
    }

    #[inline(always)]
    fn m3x3_trace_sum_squares(this: Matrix3x3<Self>) -> Self {
        this.a[0] * this.a[0] + this.a[4] * this.a[4] + this.a[8] * this.a[8]
    }

    #[inline(always)]
    fn m3x3_sum(this: Matrix3x3<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m3x3_mean(this: Matrix3x3<Self>) -> Self {
        this.sum() / 9.0
    }

    #[inline(always)]
    fn m3x3_product(this: Matrix3x3<Self>) -> Self {
        this.a.iter().product()
    }

    #[inline(always)]
    fn m3x3_adjugate(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let a = [
            this.a[4] * this.a[8] - this.a[5] * this.a[7],    //  (e*i - f*h)
            -(this.a[1] * this.a[8] - this.a[2] * this.a[7]), // -(b*i - c*h)
            this.a[1] * this.a[5] - this.a[2] * this.a[4],    //  (b*f - c*e)
            -(this.a[3] * this.a[8] - this.a[5] * this.a[6]), // -(d*i - f*g)
            this.a[0] * this.a[8] - this.a[2] * this.a[6],    //  (a*i - c*g)
            -(this.a[0] * this.a[5] - this.a[2] * this.a[3]), // -(a*f - c*d)
            this.a[3] * this.a[7] - this.a[4] * this.a[6],    //  (d*h - e*g)
            -(this.a[0] * this.a[7] - this.a[1] * this.a[6]), // -(a*h - b*g)
            this.a[0] * this.a[4] - this.a[1] * this.a[3],    //  (a*e - b*d)
        ];
        Matrix3x3::from(a)
    }
}

// **** f64 ****

impl Matrix3x3Math for f64 {
    #[inline(always)]
    fn m3x3_reciprocal(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn m3x3_neg(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let mut a = this.a;
        for r in a.iter_mut() {
            *r = -*r;
        }
        Matrix3x3::from(a)
    }

    #[inline(always)]
    fn m3x3_abs(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let mut a = this.a;
        for r in a.iter_mut() {
            *r = r.abs();
        }
        Matrix3x3::from(a)
    }

    #[inline(always)]
    fn m3x3_add(this: Matrix3x3<Self>, other: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let mut a = this.a;
        for (ii, r) in a.iter_mut().enumerate() {
            *r += other.a[ii];
        }
        Matrix3x3::from(a)
    }

    #[inline(always)]
    fn m3x3_mul_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self> {
        let mut a = this.a;
        for r in a.iter_mut() {
            *r *= other;
        }
        Matrix3x3::from(a)
    }

    #[inline(always)]
    fn m3x3_div_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self> {
        Self::m3x3_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m3x3_mul_vector(this: Matrix3x3<Self>, other: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d {
            x: this.a[0] * other.x + this.a[1] * other.y + this.a[2] * other.z,
            y: this.a[3] * other.x + this.a[4] * other.y + this.a[5] * other.z,
            z: this.a[6] * other.x + this.a[7] * other.y + this.a[8] * other.z,
        }
    }

    #[inline(always)]
    fn m3x3_vector_mul(this: Vector3d<Self>, other: Matrix3x3<Self>) -> Vector3d<Self> {
        Vector3d {
            x: this.x * other.a[0] + this.y * other.a[3] + this.z * other.a[6],
            y: this.x * other.a[1] + this.y * other.a[4] + this.z * other.a[7],
            z: this.x * other.a[2] + this.y * other.a[5] + this.z * other.a[8],
        }
    }

    #[inline(always)]
    fn m3x3_mul(this: Matrix3x3<Self>, other: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let a = [
            this.a[0] * other.a[0] + this.a[1] * other.a[3] + this.a[2] * other.a[6],
            this.a[0] * other.a[1] + this.a[1] * other.a[4] + this.a[2] * other.a[7],
            this.a[0] * other.a[2] + this.a[1] * other.a[5] + this.a[2] * other.a[8],
            this.a[3] * other.a[0] + this.a[4] * other.a[3] + this.a[5] * other.a[6],
            this.a[3] * other.a[1] + this.a[4] * other.a[4] + this.a[5] * other.a[7],
            this.a[3] * other.a[2] + this.a[4] * other.a[5] + this.a[5] * other.a[8],
            this.a[6] * other.a[0] + this.a[7] * other.a[3] + this.a[8] * other.a[6],
            this.a[6] * other.a[1] + this.a[7] * other.a[4] + this.a[8] * other.a[7],
            this.a[6] * other.a[2] + this.a[7] * other.a[5] + this.a[8] * other.a[8],
        ];
        Matrix3x3::from(a)
    }

    #[inline(always)]
    fn m3x3_determinant(this: Matrix3x3<Self>) -> Self {
        this.a[0] * (this.a[4] * this.a[8] - this.a[5] * this.a[7])
            - this.a[1] * (this.a[3] * this.a[8] - this.a[5] * this.a[6])
            + this.a[2] * (this.a[3] * this.a[7] - this.a[4] * this.a[6])
    }

    #[inline(always)]
    fn m3x3_top_right_determinant(this: Matrix3x3<Self>) -> Self {
        //let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
        //             0     4     8     5      5      1      1     8    5     2        2     1    5      4     2
        this.a[0] * (this.a[4] * this.a[8] - this.a[5] * this.a[5])
            - this.a[1] * (this.a[1] * this.a[8] - this.a[5] * this.a[2])
            + this.a[2] * (this.a[1] * this.a[5] - this.a[4] * this.a[2])
    }

    #[inline(always)]
    fn m3x3_top_right_sum_squares(this: Matrix3x3<Self>) -> Self {
        this.a[1] * this.a[1] + this.a[2] * this.a[2] + this.a[5] * this.a[5]
    }

    #[inline(always)]
    fn m3x3_trace(this: Matrix3x3<Self>) -> Self {
        this.a[0] + this.a[4] + this.a[8]
    }

    #[inline(always)]
    fn m3x3_trace_sum_squares(this: Matrix3x3<Self>) -> Self {
        this.a[0] * this.a[0] + this.a[4] * this.a[4] + this.a[8] * this.a[8]
    }

    #[inline(always)]
    fn m3x3_sum(this: Matrix3x3<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m3x3_mean(this: Matrix3x3<Self>) -> Self {
        this.sum() / 9.0
    }

    #[inline(always)]
    fn m3x3_product(this: Matrix3x3<Self>) -> Self {
        this.a.iter().product()
    }

    #[inline(always)]
    fn m3x3_adjugate(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let a = [
            this.a[4] * this.a[8] - this.a[5] * this.a[7],    //  (e*i - f*h)
            -(this.a[1] * this.a[8] - this.a[2] * this.a[7]), // -(b*i - c*h)
            this.a[1] * this.a[5] - this.a[2] * this.a[4],    //  (b*f - c*e)
            -(this.a[3] * this.a[8] - this.a[5] * this.a[6]), // -(d*i - f*g)
            this.a[0] * this.a[8] - this.a[2] * this.a[6],    //  (a*i - c*g)
            -(this.a[0] * this.a[5] - this.a[2] * this.a[3]), // -(a*f - c*d)
            this.a[3] * this.a[7] - this.a[4] * this.a[6],    //  (d*h - e*g)
            -(this.a[0] * this.a[7] - this.a[1] * this.a[6]), // -(a*h - b*g)
            this.a[0] * this.a[4] - this.a[1] * this.a[3],    //  (a*e - b*d)
        ];
        Matrix3x3::from(a)
    }
}
