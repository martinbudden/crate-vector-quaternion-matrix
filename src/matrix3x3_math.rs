#![allow(clippy::inline_always)]
use cfg_if::cfg_if;
use core::mem::{align_of, size_of};

cfg_if! {
    if #[cfg(feature = "simd")] {
        use core::simd::{f32x4,num::SimdFloat};
        // must be aligned if using SIMD
        const _: () = assert!(size_of::<Matrix3x3<f32>>() == 64);
        const _: () = assert!(align_of::<Matrix3x3<f32>>() == 64);
    } else if #[cfg(feature = "align")] {
        const _: () = assert!(size_of::<Matrix3x3<f32>>() == 64);
        const _: () = assert!(align_of::<Matrix3x3<f32>>() == 64);
    } else {
        const _: () = assert!(size_of::<Matrix3x3<f32>>() == 36);
        const _: () = assert!(align_of::<Matrix3x3<f32>>() == 4);
    }
}

use crate::{Matrix3x3, Vector3};

// Column 1
const M11: usize = 0;
const M21: usize = 1;
const M31: usize = 2;
// Column 2
const M12: usize = 3;
const M22: usize = 4;
const M32: usize = 5;
// Column 3
const M13: usize = 6;
const M23: usize = 7;
const M33: usize = 8;

// **** Math ****

/// Math functions for Matrix3x3.<br>
pub trait Matrix3x3Math: Sized {
    fn m3x3_neg(this: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_abs(this: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_add(this: Matrix3x3<Self>, this: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_mul_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self>;
    fn m3x3_div_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self>;
    fn m3x3_mul_add(this: Matrix3x3<Self>, k: Self, other: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_mul_vector(this: Matrix3x3<Self>, other: Vector3<Self>) -> Vector3<Self>;
    fn m3x3_vector_mul(this: Vector3<Self>, other: Matrix3x3<Self>) -> Vector3<Self>;
    fn m3x3_vector_outer_product(col: Vector3<Self>, row: Vector3<Self>) -> Matrix3x3<Self>;
    fn m3x3_mul(this: Matrix3x3<Self>, other: Matrix3x3<Self>) -> Matrix3x3<Self>;
    fn m3x3_trace(this: Matrix3x3<Self>) -> Self;
    fn m3x3_trace_sum_squares(this: Matrix3x3<Self>) -> Self;
    fn m3x3_sum(this: Matrix3x3<Self>) -> Self;
    fn m3x3_mean(this: Matrix3x3<Self>) -> Self;
    fn m3x3_product(this: Matrix3x3<Self>) -> Self;
    fn m3x3_determinant(this: Matrix3x3<Self>) -> Self;
    fn m3x3_adjugate(this: Matrix3x3<Self>) -> (Matrix3x3<Self>, Self);
    fn m3x3_adjugate_symmetric(this: Matrix3x3<Self>) -> (Matrix3x3<Self>, Self);
}

impl Matrix3x3Math for f32 {
    #[inline(always)]
    fn m3x3_neg(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let a = core::array::from_fn(|ii| -this.a[ii]);
        Matrix3x3 { a }
    }

    #[inline(always)]
    fn m3x3_abs(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let a = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix3x3 { a }
    }

    #[inline(always)]
    fn m3x3_add(this: Matrix3x3<Self>, other: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let a = core::array::from_fn(|ii| this.a[ii] + other.a[ii]);
        Matrix3x3 { a }
    }

    #[inline(always)]
    fn m3x3_mul_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self> {
        let a = core::array::from_fn(|ii| this.a[ii] * other);
        Matrix3x3 { a }
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
    fn m3x3_mul_vector(this: Matrix3x3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 {
            x: this.a[M11] * other.x + this.a[M12] * other.y + this.a[M13] * other.z,
            y: this.a[M21] * other.x + this.a[M22] * other.y + this.a[M23] * other.z,
            z: this.a[M31] * other.x + this.a[M32] * other.y + this.a[M33] * other.z,
        }
    }

    /*#[inline(always)]
    fn m3x3_mul_vector(this: Matrix3x3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        // Map the 16-byte aligned vector into a uniform 4-element array.
        // The 4th element is zeroed out so it contributes nothing to the dot products.
        let v = [other.x, other.y, other.z, 0.0];

        // Unpack the flat matrix into 4-element padded rows.
        let r1 = [this.a[M11], this.a[M12], this.a[M13], 0.0];
        let r2 = [this.a[M21], this.a[M22], this.a[M23], 0.0];
        let r3 = [this.a[M31], this.a[M32], this.a[M33], 0.0];

        // Calculates row dot products using unrolled, 4-wide loops.
        // LLVM easily vectorizes a simple element-wise multiply-and-accumulate loop
        // spanning exactly 4 items, mapping it directly to hardware registers.
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;

        for ii in 0..4 {
            x += r1[ii] * v[ii];
        }
        for ii in 0..4 {
            y += r2[ii] * v[ii];
        }
        for ii in 0..4 {
            z += r3[ii] * v[ii];
        }

        Vector3 { x, y, z }
    }*/

    #[rustfmt::skip]
    #[inline(always)]
    fn m3x3_vector_mul(this: Vector3<Self>, other: Matrix3x3<Self>) -> Vector3<Self> {
        Vector3 {
            x: this.x * other.a[M11] + this.y * other.a[M21] + this.z * other.a[M31],
            y: this.x * other.a[M12] + this.y * other.a[M22] + this.z * other.a[M32],
            z: this.x * other.a[M13] + this.y * other.a[M23] + this.z * other.a[M33],
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m3x3_vector_outer_product(col: Vector3<Self>, row: Vector3<Self>) -> Matrix3x3<Self> {
        #[cfg(feature = "simd")]
        {
            // By taking ownership of the value, Rust guarantees no other pointer
            // can modify these values during our calculation loop.
            // let row_simd = unsafe { *(&row as *const Vector3f32 as *const f32x4) };
            let col_simd = f32x4::from_array([col.x, col.y, col.z, 0.0]);

            let row_x = f32x4::splat(row.x);
            let row_y = f32x4::splat(row.y);
            let row_z = f32x4::splat(row.z);

            let c1 = row_x * col_simd;
            let c2 = row_y * col_simd;
            let c3 = row_z * col_simd;

            Matrix3x3::from_padded_2d_row_array([c1.to_array(), c2.to_array(), c3.to_array()])
        }
        #[cfg(not(feature = "simd"))]
        {
        Matrix3x3 {
            a: [
                col.x * row.x, col.y * row.x, col.z * row.x,
                col.x * row.y, col.y * row.y, col.z * row.y,
                col.x * row.z, col.y * row.z, col.z * row.z,
            ],
        }
        }
    }

    #[inline(always)]
    fn m3x3_mul(this: Matrix3x3<Self>, other: Matrix3x3<Self>) -> Matrix3x3<Self> {
        #[cfg(feature = "simd")]
        {
            let a0_simd = f32x4::from_array([this.a[M11], this.a[M12], this.a[M13], 0.0]);
            let a3_simd = f32x4::from_array([this.a[M21], this.a[M22], this.a[M23], 0.0]);
            let a6_simd = f32x4::from_array([this.a[M31], this.a[M32], this.a[M33], 0.0]);
            let b0_simd = f32x4::from_array([other.a[M11], other.a[M21], other.a[M31], 0.0]);
            let b1_simd = f32x4::from_array([other.a[M12], other.a[M22], other.a[M32], 0.0]);
            let b2_simd = f32x4::from_array([other.a[M13], other.a[M23], other.a[M33], 0.0]);
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
            Matrix3x3 { a }
        }
        #[cfg(not(feature = "simd"))]
        {
            let a = [
                this.a[M11] * other.a[M11] + this.a[M12] * other.a[M21] + this.a[M13] * other.a[M31],
                this.a[M21] * other.a[M11] + this.a[M22] * other.a[M21] + this.a[M23] * other.a[M31],
                this.a[M31] * other.a[M11] + this.a[M32] * other.a[M21] + this.a[M33] * other.a[M31],
                this.a[M11] * other.a[M12] + this.a[M12] * other.a[M22] + this.a[M13] * other.a[M32],
                this.a[M21] * other.a[M12] + this.a[M22] * other.a[M22] + this.a[M23] * other.a[M32],
                this.a[M31] * other.a[M12] + this.a[M32] * other.a[M22] + this.a[M33] * other.a[M32],
                this.a[M11] * other.a[M13] + this.a[M12] * other.a[M23] + this.a[M13] * other.a[M33],
                this.a[M21] * other.a[M13] + this.a[M22] * other.a[M23] + this.a[M23] * other.a[M33],
                this.a[M31] * other.a[M13] + this.a[M32] * other.a[M23] + this.a[M33] * other.a[M33],
            ];
            Matrix3x3 { a }
        }
    }

    #[inline(always)]
    fn m3x3_trace(this: Matrix3x3<Self>) -> Self {
        this.a[M11] + this.a[M22] + this.a[M33]
    }

    #[inline(always)]
    fn m3x3_trace_sum_squares(this: Matrix3x3<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let trace_simd = f32x4::from_array([this.a[M11], this.a[M22], this.a[M33], 0.0]);
            (trace_simd * trace_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            this.a[M11] * this.a[M11] + this.a[M22] * this.a[M22] + this.a[M33] * this.a[M33]
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

    #[rustfmt::skip]
    #[inline(always)]
    fn m3x3_determinant(this: Matrix3x3<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            let a_simd = f32x4::from_array([this.a[M11], -this.a[M12], this.a[M13], 0.0]);

            let d = [
                this.a[M22] * this.a[M33] - this.a[M23] * this.a[M32],
                this.a[M21] * this.a[M33] - this.a[M23] * this.a[M31],
                this.a[M21] * this.a[M32] - this.a[M22] * this.a[M31],
                0.0,
            ];
            let d_simd = f32x4::from_array(d);

            (a_simd * d_simd).reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
             this.a[M11] * (this.a[M22] * this.a[M33] - this.a[M23] * this.a[M32])
            -this.a[M12] * (this.a[M21] * this.a[M33] - this.a[M23] * this.a[M31])
            +this.a[M13] * (this.a[M21] * this.a[M32] - this.a[M22] * this.a[M31])
        }
    }

    /// Returns the adjugate and determinant of a matrix.
    #[rustfmt::skip]
    #[inline(always)]
    fn m3x3_adjugate(this: Matrix3x3<Self>) -> (Matrix3x3<Self>, Self) {
        /*#[cfg(feature = "simd")]
        {
            let a = this.a;

            // use SIMD to calculate the first 8 elements of the array, and then manually calculate the 9th element.
            // TODO: change the 4 from_arrays into 2 from_arrays and use swizzles.
            let r0a = [a[M22], -a[M12], a[M12], -a[M21], a[M11], -a[M11], a[M21], -a[M11]];
            let r0b = [a[M33], a[M33], a[M23], a[M33], a[M33], a[M23], a[M32], a[M32]];

            let r1a = [-a[M23], a[M13], -a[M13], a[M23], -a[M13], a[M13], -a[M22], a[M12]];
            let r1b = [a[M32], a[M32], a[M22], a[M31], a[M31], a[M21], a[M31], a[M31]];

            let r0a_simd = f32x8::from_array(r0a);
            let r0b_simd = f32x8::from_array(r0b);

            let r1a_simd = f32x8::from_array(r1a);
            let r1b_simd = f32x8::from_array(r1b);

            let r0_simd = r0a_simd * r0b_simd;
            let r1_simd = r1a_simd * r1b_simd;

            let r: [f32; 8] = (r0_simd + r1_simd).into();

            Matrix3x3 { a: [r[M11], r[M12], r[M13], r[M21], r[M22], r[M23], r[M31], r[M32], a[M11] * a[M22] - a[M12] * a[M21]] }
        }
        #[cfg(not(feature = "simd"))]*/
        let ei_fh = this.a[M22] * this.a[M33] - this.a[M23] * this.a[M32];
        let di_fg = this.a[M21] * this.a[M33] - this.a[M23] * this.a[M31];
        let dh_eg = this.a[M21] * this.a[M32] - this.a[M22] * this.a[M31];
        let determinant = this.a[M11] * ei_fh - this.a[M12]*di_fg + this.a[M13]* dh_eg;

        let a = [
             ei_fh,                                          //  (e*i - f*h)
            -di_fg,                                          // -(d*i - f*g)
             dh_eg,                                          //  (d*h - e*g)

            -(this.a[M12] * this.a[M33] - this.a[M13] * this.a[M32]), // -(b*i - c*h)
              this.a[M11] * this.a[M33] - this.a[M13] * this.a[M31],  //  (a*i - c*g)
            -(this.a[M11] * this.a[M32] - this.a[M12] * this.a[M31]), // -(a*h - b*g)

              this.a[M12] * this.a[M23] - this.a[M13] * this.a[M22],  //  (b*f - c*e)
            -(this.a[M11] * this.a[M23] - this.a[M13] * this.a[M21]), // -(a*f - c*d)
              this.a[M11] * this.a[M22] - this.a[M12] * this.a[M21],  //  (a*e - b*d)
        ];
        (Matrix3x3 { a }, determinant)
    }

    /// Returns the adjugate and determinant of a matrix, assuming it is symmetric.
    #[rustfmt::skip]
    #[inline(always)]
    fn m3x3_adjugate_symmetric(this: Matrix3x3<Self>) -> (Matrix3x3<Self>, Self) {
        // Cache repetitive terms to minimize multiplications
        let b_sq = this.a[M12] * this.a[M12];
        let c_sq = this.a[M13] * this.a[M13];
        let f_sq = this.a[M23] * this.a[M23];

        // Calculate the unique elements of the adjugate matrix
        let adj_0 = this.a[M22] * this.a[M33] - f_sq;                          // Row 0, Col 0
        let adj_1 = this.a[M13] * this.a[M23] - this.a[M12] * this.a[M33];         // Row 0, Col 1 (and Row 1, Col 0)
        let adj_2 = this.a[M12] * this.a[M23] - this.a[M13] * this.a[M22];         // Row 0, Col 2 (and Row 2, Col 0)
        let adj_4 = this.a[M11] * this.a[M33] - c_sq;                          // Row 1, Col 1
        let adj_5 = this.a[M12] * this.a[M13] - this.a[M11] * this.a[M23];         // Row 1, Col 2 (and Row 2, Col 1)
        let adj_8 = this.a[M11] * this.a[M22] - b_sq;                          // Row 2, Col 2

        // Determinant computed via dot product of Row 0 and Adjugate Row 0
        let determinant = this.a[M11] * adj_0 + this.a[M12] * adj_1 + this.a[M13] * adj_2;

        let a = [
            adj_0, adj_1, adj_2,
            adj_1, adj_4, adj_5,
            adj_2, adj_5, adj_8,
        ];

        (Matrix3x3 { a }, determinant)
    }
}

// **** f64 ****

impl Matrix3x3Math for f64 {
    #[inline(always)]
    fn m3x3_neg(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let mut a = this.a;
        for r in &mut a {
            *r = -*r;
        }
        Matrix3x3 { a }
    }

    #[inline(always)]
    fn m3x3_abs(this: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let mut a = this.a;
        for r in &mut a {
            *r = r.abs();
        }
        Matrix3x3 { a }
    }

    #[inline(always)]
    fn m3x3_add(this: Matrix3x3<Self>, other: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let mut a = this.a;
        for (ii, r) in a.iter_mut().enumerate() {
            *r += other.a[ii];
        }
        Matrix3x3 { a }
    }

    #[inline(always)]
    fn m3x3_mul_scalar(this: Matrix3x3<Self>, other: Self) -> Matrix3x3<Self> {
        let mut a = this.a;
        for r in &mut a {
            *r *= other;
        }
        Matrix3x3 { a }
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
    fn m3x3_mul_vector(this: Matrix3x3<Self>, other: Vector3<Self>) -> Vector3<Self> {
        Vector3 {
            x: this.a[M11] * other.x + this.a[M21] * other.y + this.a[M31] * other.z,
            y: this.a[M12] * other.x + this.a[M22] * other.y + this.a[M32] * other.z,
            z: this.a[M13] * other.x + this.a[M23] * other.y + this.a[M33] * other.z,
        }
    }

    #[inline(always)]
    fn m3x3_vector_mul(this: Vector3<Self>, other: Matrix3x3<Self>) -> Vector3<Self> {
        Vector3 {
            x: this.x * other.a[M11] + this.y * other.a[M21] + this.z * other.a[M31],
            y: this.x * other.a[M12] + this.y * other.a[M22] + this.z * other.a[M32],
            z: this.x * other.a[M13] + this.y * other.a[M23] + this.z * other.a[M33],
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m3x3_vector_outer_product(col: Vector3<Self>, row: Vector3<Self>) -> Matrix3x3<Self> {
        Matrix3x3 {
            a: [
                col.x * row.x, col.y * row.x, col.z * row.x,
                col.x * row.y, col.y * row.y, col.z * row.y,
                col.x * row.z, col.y * row.z, col.z * row.z,
            ],
        }
    }

    #[inline(always)]
    fn m3x3_mul(this: Matrix3x3<Self>, other: Matrix3x3<Self>) -> Matrix3x3<Self> {
        let a = [
            this.a[M11] * other.a[M11] + this.a[M12] * other.a[M21] + this.a[M13] * other.a[M31],
            this.a[M11] * other.a[M12] + this.a[M12] * other.a[M22] + this.a[M13] * other.a[M32],
            this.a[M11] * other.a[M13] + this.a[M12] * other.a[M23] + this.a[M13] * other.a[M33],
            this.a[M21] * other.a[M11] + this.a[M22] * other.a[M21] + this.a[M23] * other.a[M31],
            this.a[M21] * other.a[M12] + this.a[M22] * other.a[M22] + this.a[M23] * other.a[M32],
            this.a[M21] * other.a[M13] + this.a[M22] * other.a[M23] + this.a[M23] * other.a[M33],
            this.a[M31] * other.a[M11] + this.a[M32] * other.a[M21] + this.a[M33] * other.a[M31],
            this.a[M31] * other.a[M12] + this.a[M32] * other.a[M22] + this.a[M33] * other.a[M32],
            this.a[M31] * other.a[M13] + this.a[M32] * other.a[M23] + this.a[M33] * other.a[M33],
        ];
        Matrix3x3 { a }
    }

    #[inline(always)]
    fn m3x3_trace(this: Matrix3x3<Self>) -> Self {
        this.a[M11] + this.a[M22] + this.a[M33]
    }

    #[inline(always)]
    fn m3x3_trace_sum_squares(this: Matrix3x3<Self>) -> Self {
        this.a[M11] * this.a[M11] + this.a[M22] * this.a[M22] + this.a[M33] * this.a[M33]
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

    #[rustfmt::skip]
    #[inline(always)]
    fn m3x3_determinant(this: Matrix3x3<Self>) -> Self {
         this.a[M11] * (this.a[M22] * this.a[M33] - this.a[M23] * this.a[M32])
        -this.a[M12] * (this.a[M21] * this.a[M33] - this.a[M23] * this.a[M31])
        +this.a[M13] * (this.a[M21] * this.a[M32] - this.a[M22] * this.a[M31])
    }

    /// Returns the adjugate and determinant of a matrix.
    #[rustfmt::skip]
    #[inline(always)]
    fn m3x3_adjugate(this: Matrix3x3<Self>) -> (Matrix3x3<Self>, Self) {
        let ei_fh = this.a[M22] * this.a[M33] - this.a[M23] * this.a[M32];
        let di_fg = this.a[M21] * this.a[M33] - this.a[M23] * this.a[M31];
        let dh_eg = this.a[M21] * this.a[M32] - this.a[M22] * this.a[M31];
        let determinant = this.a[M11] * ei_fh - this.a[M12]*di_fg + this.a[M13]* dh_eg;

        let a = [
             ei_fh,                                          //  (e*i - f*h)
            -di_fg,                                          // -(d*i - f*g)
             dh_eg,                                          //  (d*h - e*g)

            -(this.a[M12] * this.a[M33] - this.a[M13] * this.a[M32]), // -(b*i - c*h)
              this.a[M11] * this.a[M33] - this.a[M13] * this.a[M31],  //  (a*i - c*g)
            -(this.a[M11] * this.a[M32] - this.a[M12] * this.a[M31]), // -(a*h - b*g)

              this.a[M12] * this.a[M23] - this.a[M13] * this.a[M22],  //  (b*f - c*e)
            -(this.a[M11] * this.a[M23] - this.a[M13] * this.a[M21]), // -(a*f - c*d)
              this.a[M11] * this.a[M22] - this.a[M12] * this.a[M21],  //  (a*e - b*d)
        ];
        (Matrix3x3 { a }, determinant)
    }

    /// Returns the adjugate and determinant of a matrix, assuming it is symmetric.
    #[rustfmt::skip]
    #[inline(always)]
    fn m3x3_adjugate_symmetric(this: Matrix3x3<Self>) -> (Matrix3x3<Self>, Self) {
        // Cache repetitive terms to minimize multiplications
        let b_sq = this.a[M12] * this.a[M12];
        let c_sq = this.a[M13] * this.a[M13];
        let f_sq = this.a[M23] * this.a[M23];

        // Calculate the unique elements of the adjugate matrix
        let adj_0 = this.a[M22] * this.a[M33] - f_sq;                          // Row 0, Col 0
        let adj_1 = this.a[M13] * this.a[M23] - this.a[M12] * this.a[M33];         // Row 0, Col 1 (and Row 1, Col 0)
        let adj_2 = this.a[M12] * this.a[M23] - this.a[M13] * this.a[M22];         // Row 0, Col 2 (and Row 2, Col 0)
        let adj_4 = this.a[M11] * this.a[M33] - c_sq;                          // Row 1, Col 1
        let adj_5 = this.a[M12] * this.a[M13] - this.a[M11] * this.a[M23];         // Row 1, Col 2 (and Row 2, Col 1)
        let adj_8 = this.a[M11] * this.a[M22] - b_sq;                          // Row 2, Col 2

        // Determinant computed via dot product of Row 0 and Adjugate Row 0
        let determinant = this.a[M11] * adj_0 + this.a[M12] * adj_1 + this.a[M13] * adj_2;

        let a = [
            adj_0, adj_1, adj_2,
            adj_1, adj_4, adj_5,
            adj_2, adj_5, adj_8,
        ];

        (Matrix3x3 { a }, determinant)
    }
}
