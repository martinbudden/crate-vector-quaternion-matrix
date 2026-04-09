use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "simd")] {
        use core::simd::{f32x4,f32x8,num::SimdFloat};
        // must be aligned if using SIMD
        const _: () = assert!(core::mem::size_of::<Matrix3x3<f32>>() == 64);
        const _: () = assert!(core::mem::align_of::<Matrix3x3<f32>>() == 32);
    } else if #[cfg(feature = "no_align")] {
        const _: () = assert!(core::mem::size_of::<Matrix3x3<f32>>() == 36);
        const _: () = assert!(core::mem::align_of::<Matrix3x3<f32>>() == 4);
    } else {
        const _: () = assert!(core::mem::size_of::<Matrix3x3<f32>>() == 64);
        const _: () = assert!(core::mem::align_of::<Matrix3x3<f32>>() == 32);
    }
}

use crate::{Matrix3x3, Vector3d};

// **** Math ****

/// Math functions for Matrix3x3, using **SIMD** accelerations for `f32`.<br><br>
pub trait Matrix3x3Math: Sized {
    fn m3x3_neg(this: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_abs(this: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_add(this: Matrix3x3<Self>, this: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_mul_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self>;
    fn m3x3_div_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self>;
    fn m3x3_mul_add(this: Matrix3x3<Self>, k: Self, other: Matrix3x3<Self>) -> Matrix3x3<Self>;
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
    fn m3x3_neg(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let ret = core::array::from_fn(|ii| -this.a[ii]);
        Matrix3x3::from(ret)
    }

    #[inline(always)]
    fn m3x3_abs(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix3x3::from(ret)
    }

    #[inline(always)]
    fn m3x3_add(this: Matrix3x3<Self>, other: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] + other.a[ii]);
        Matrix3x3::from(ret)
    }

    #[inline(always)]
    fn m3x3_mul_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] * other);
        Matrix3x3::from(ret)
    }

    #[inline(always)]
    fn m3x3_div_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self> {
        Self::m3x3_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m3x3_mul_add(this: Matrix3x3<Self>, k: Self, other: Matrix3x3<Self>) -> Matrix3x3<Self> {
        Self::m3x3_add(Self::m3x3_mul_scalar(this, k), other)
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
        #[cfg(feature = "simd")]
        {
            let a0_simd = f32x4::from_array([this.a[0], this.a[1], this.a[2], 0.0]);
            let a3_simd = f32x4::from_array([this.a[3], this.a[4], this.a[5], 0.0]);
            let a6_simd = f32x4::from_array([this.a[6], this.a[7], this.a[8], 0.0]);
            let b0_simd = f32x4::from_array([other.a[0], other.a[3], other.a[6], 0.0]);
            let b1_simd = f32x4::from_array([other.a[1], other.a[4], other.a[7], 0.0]);
            let b2_simd = f32x4::from_array([other.a[2], other.a[5], other.a[8], 0.0]);
            let a = [
                (a0_simd * b0_simd).reduce_sum(),
                (a0_simd * b1_simd).reduce_sum(),
                (a0_simd * b2_simd).reduce_sum(),
                (a3_simd * b0_simd).reduce_sum(),
                (a3_simd * b1_simd).reduce_sum(),
                (a3_simd * b2_simd).reduce_sum(),
                (a6_simd * b0_simd).reduce_sum(),
                (a6_simd * b1_simd).reduce_sum(),
                (a6_simd * b2_simd).reduce_sum(),
            ];
            Matrix3x3::from(a)
        }
        #[cfg(not(feature = "simd"))]
        {
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
    }

    #[inline(always)]
    fn m3x3_determinant(this: Matrix3x3<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let a_simd = f32x4::from_array([this.a[0], -this.a[1], this.a[2], 0.0]);

            let d = [
                this.a[4] * this.a[8] - this.a[5] * this.a[7],
                this.a[3] * this.a[8] - this.a[5] * this.a[6],
                this.a[3] * this.a[7] - this.a[4] * this.a[6],
                0.0,
            ];
            let d_simd = f32x4::from_array(d);

            (a_simd * d_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.a[0] * (this.a[4] * this.a[8] - this.a[5] * this.a[7])
                - this.a[1] * (this.a[3] * this.a[8] - this.a[5] * this.a[6])
                + this.a[2] * (this.a[3] * this.a[7] - this.a[4] * this.a[6])
        }
    }

    #[inline(always)]
    fn m3x3_top_right_determinant(this: Matrix3x3<Self>) -> Self {
        //let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
        //            a0     a4    a8    a5    a5     a1     a1    a8    a5    a2     a2     a1    a5    a4    a2
        #[cfg(feature = "simd")]
        {
            let a_simd = f32x4::from_array([this.a[0], -this.a[1], this.a[2], 0.0]);

            let d = [
                this.a[4] * this.a[8] - this.a[5] * this.a[5],
                this.a[1] * this.a[8] - this.a[5] * this.a[2],
                this.a[1] * this.a[5] - this.a[4] * this.a[2],
                0.0,
            ];
            let d_simd = f32x4::from_array(d);

            (a_simd * d_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.a[0] * (this.a[4] * this.a[8] - this.a[5] * this.a[5])
                - this.a[1] * (this.a[1] * this.a[8] - this.a[5] * this.a[2])
                + this.a[2] * (this.a[1] * this.a[5] - this.a[4] * this.a[2])
        }
    }

    #[inline(always)]
    fn m3x3_top_right_sum_squares(this: Matrix3x3<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let top_right_simd = f32x4::from_array([this.a[1], this.a[2], this.a[5], 0.0]);
            (top_right_simd * top_right_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.a[1] * this.a[1] + this.a[2] * this.a[2] + this.a[5] * this.a[5]
        }
    }

    #[inline(always)]
    fn m3x3_trace(this: Matrix3x3<Self>) -> Self {
        this.a[0] + this.a[4] + this.a[8]
    }

    #[inline(always)]
    fn m3x3_trace_sum_squares(this: Matrix3x3<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let trace_simd = f32x4::from_array([this.a[0], this.a[4], this.a[8], 0.0]);
            (trace_simd * trace_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.a[0] * this.a[0] + this.a[4] * this.a[4] + this.a[8] * this.a[8]
        }
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
        #[cfg(feature = "simd")]
        {
            let a = this.a;

            // use SIMD to calculate the first 8 elements of the array, and then manually calculate the 9th element.
            // TODO: change the 4 from_arrays into 2 from_arrays and use swizzles.
            let r0a = [a[4], -a[1], a[1], -a[3], a[0], -a[0], a[3], -a[0]];
            let r0b = [a[8], a[8], a[5], a[8], a[8], a[5], a[7], a[7]];

            let r1a = [-a[5], a[2], -a[2], a[5], -a[2], a[2], -a[4], a[1]];
            let r1b = [a[7], a[7], a[4], a[6], a[6], a[3], a[6], a[6]];

            let r0a_simd = f32x8::from_array(r0a);
            let r0b_simd = f32x8::from_array(r0b);

            let r1a_simd = f32x8::from_array(r1a);
            let r1b_simd = f32x8::from_array(r1b);

            let r0_simd = r0a_simd * r0b_simd;
            let r1_simd = r1a_simd * r1b_simd;

            let r: [f32; 8] = (r0_simd + r1_simd).into();

            Matrix3x3::from([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], a[0] * a[4] - a[1] * a[3]])
        }
        #[cfg(not(feature = "simd"))]
        {
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
}

// **** f64 ****

impl Matrix3x3Math for f64 {
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
    fn m3x3_mul_add(this: Matrix3x3<Self>, k: Self, other: Matrix3x3<Self>) -> Matrix3x3<Self> {
        Self::m3x3_add(Self::m3x3_mul_scalar(this, k), other)
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
