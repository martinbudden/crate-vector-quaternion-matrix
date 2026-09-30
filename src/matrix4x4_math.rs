#![allow(clippy::inline_always)]

use crate::{Matrix4x4, Quaternion, Vector4};

// Column 1
const M11: usize = 0;
const M21: usize = 1;
const M31: usize = 2;
const M41: usize = 3;
// Column 2
const M12: usize = 4;
const M22: usize = 5;
const M32: usize = 6;
const M42: usize = 7;
// Column 3
const M13: usize = 8;
const M23: usize = 9;
const M33: usize = 10;
const M43: usize = 11;
// Column 4
const M14: usize = 12;
const M24: usize = 13;
const M34: usize = 14;
const M44: usize = 15;

// **** Math ****

/// Math functions for Matrix4x4.<br>
pub trait Matrix4x4Math: Sized {
    fn m4x4_neg(this: Matrix4x4<Self>) -> Matrix4x4<Self>;
    fn m4x4_abs(this: Matrix4x4<Self>) -> Matrix4x4<Self>;
    fn m4x4_add(this: Matrix4x4<Self>, this: Matrix4x4<Self>) -> Matrix4x4<Self>;
    fn m4x4_mul_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self>;
    fn m4x4_div_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self>;
    fn m4x4_mul_add(this: Matrix4x4<Self>, k: Self, other: Matrix4x4<Self>) -> Matrix4x4<Self>;
    fn m4x4_mul_vector(this: Matrix4x4<Self>, other: Vector4<Self>) -> Vector4<Self>;
    fn m4x4_vector_mul(this: Vector4<Self>, other: Matrix4x4<Self>) -> Vector4<Self>;
    fn m4x4_vector_outer_product(col: Vector4<Self>, row: Vector4<Self>) -> Matrix4x4<Self>;
    fn m4x4_quaternion_outer_product(this: Quaternion<Self>) -> Matrix4x4<Self>;
    fn m4x4_mul(this: Matrix4x4<Self>, other: Matrix4x4<Self>) -> Matrix4x4<Self>;
    fn m4x4_determinant(this: Matrix4x4<Self>) -> Self;
    fn m4x4_trace(this: Matrix4x4<Self>) -> Self;
    fn m4x4_trace_sum_squares(this: Matrix4x4<Self>) -> Self;
    fn m4x4_sum(this: Matrix4x4<Self>) -> Self;
    fn m4x4_mean(this: Matrix4x4<Self>) -> Self;
    fn m4x4_product(this: Matrix4x4<Self>) -> Self;
    fn m4x4_adjugate(this: Matrix4x4<Self>) -> (Matrix4x4<Self>, Self);
}

impl Matrix4x4Math for f32 {
    #[inline(always)]
    fn m4x4_neg(this: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let a = core::array::from_fn(|ii| -this.a[ii]);
        Matrix4x4 { a }
    }

    #[inline(always)]
    fn m4x4_abs(this: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let a = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix4x4 { a }
    }

    #[inline(always)]
    fn m4x4_add(this: Matrix4x4<Self>, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let a = core::array::from_fn(|ii| this.a[ii] + other.a[ii]);
        Matrix4x4 { a }
    }

    #[inline(always)]
    fn m4x4_mul_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self> {
        let a = core::array::from_fn(|ii| this.a[ii] * other);
        Matrix4x4 { a }
    }

    #[inline(always)]
    fn m4x4_div_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self> {
        Self::m4x4_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m4x4_mul_add(this: Matrix4x4<Self>, k: Self, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        Self::m4x4_add(Self::m4x4_mul_scalar(this, k), other)
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m4x4_mul_vector(this: Matrix4x4<Self>, other: Vector4<Self>) -> Vector4<Self> {
        Vector4 {
            x: this.a[M11] * other.x + this.a[M12] * other.y + this.a[M13] * other.z + this.a[M14] * other.t,
            y: this.a[M21] * other.x + this.a[M22] * other.y + this.a[M23] * other.z + this.a[M24] * other.t,
            z: this.a[M31] * other.x + this.a[M32] * other.y + this.a[M33] * other.z + this.a[M34] * other.t,
            t: this.a[M41] * other.x + this.a[M42] * other.y + this.a[M43] * other.z + this.a[M44] * other.t,
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m4x4_vector_mul(this: Vector4<Self>, other: Matrix4x4<Self>) -> Vector4<Self> {
        Vector4 {
            x: this.x * other.a[M11] + this.y * other.a[M21] + this.z * other.a[M31] + this.t * other.a[M41],
            y: this.x * other.a[M12] + this.y * other.a[M22] + this.z * other.a[M32] + this.t * other.a[M42],
            z: this.x * other.a[M13] + this.y * other.a[M23] + this.z * other.a[M33] + this.t * other.a[M43],
            t: this.x * other.a[M14] + this.y * other.a[M24] + this.z * other.a[M34] + this.t * other.a[M44],
        }
    }

    #[inline(always)]
    fn m4x4_vector_outer_product(col: Vector4<Self>, row: Vector4<Self>) -> Matrix4x4<Self> {
        // Structure data into local fixed-size arrays of 4 elements.
        // Since row is align(16), we manually map the implicit 4th buffer element.
        let c = [col.x, col.y, col.z, col.t];

        let mut m0 = [0.0; 4];
        let mut m1 = [0.0; 4];
        let mut m2 = [0.0; 4];
        let mut m3 = [0.0; 4];

        // Write uniform loops spanning exactly 4 elements.
        // LLVM's auto-vectorizer recognizes 4-wide float operations
        // and combines these into parallel execution blocks, if the processor supports it.
        for ii in 0..4 {
            m0[ii] = row.x * c[ii];
        }
        for ii in 0..4 {
            m1[ii] = row.y * c[ii];
        }
        for ii in 0..4 {
            m2[ii] = row.z * c[ii];
        }
        for ii in 0..4 {
            m3[ii] = row.t * c[ii];
        }

        Matrix4x4 {
            a: [
                m0[0], m0[1], m0[2], m0[3], //
                m1[0], m1[1], m1[2], m1[3], //
                m2[0], m2[1], m2[2], m2[3], //
                m3[0], m3[1], m3[2], m3[3], //
            ],
        }
    }

    #[inline(always)]
    fn m4x4_quaternion_outer_product(this: Quaternion<Self>) -> Matrix4x4<Self> {
        // Structure data into local fixed-size arrays of 4 elements.
        // Since row is align(16), we manually map the implicit 4th buffer element.
        let r = [this.w, this.x, this.y, this.z];

        let mut m0 = [0.0; 4];
        let mut m1 = [0.0; 4];
        let mut m2 = [0.0; 4];
        let mut m3 = [0.0; 4];

        // Write uniform loops spanning exactly 4 elements.
        // LLVM's auto-vectorizer recognizes 4-wide float operations
        // and combines these into parallel execution blocks, if the processor supports it.
        for ii in 0..4 {
            m0[ii] = this.w * r[ii];
        }
        for ii in 0..4 {
            m1[ii] = this.x * r[ii];
        }
        for ii in 0..4 {
            m2[ii] = this.y * r[ii];
        }
        for ii in 0..4 {
            m3[ii] = this.z * r[ii];
        }

        Matrix4x4 {
            a: [
                m0[0], m0[1], m0[2], m0[3], //
                m1[0], m1[1], m1[2], m1[3], //
                m2[0], m2[1], m2[2], m2[3], //
                m3[0], m3[1], m3[2], m3[3], //
            ],
        }
    }

    /*
    #[inline]
    fn m4x4_mul(this: Matrix4x4<Self>, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let mut ret = [0.0; 16];

        // Explicitly tell the compiler we are working with fixed 4-element chunks
        for i in 0..4 {
            let col = &this.a[i * 4..(i * 4) + 4];
            for j in 0..4 {
                // By using a local sum and fixed indices,
                // LLVM can more easily unroll this into SIMD 'Multiply-Add' instructions.
                ret[i + j*4] = col[0] * other.a[i]
                    + col[1] * other.a[i + 4]
                    + col[2] * other.a[i + 8]
                    + col[3] * other.a[i + 12];
            }
        }
        Matrix4x4 { a: ret }
    }*/

    // **** Mul ****

    #[inline(always)]
    fn m4x4_mul(this: Matrix4x4<Self>, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let mut ret = [0.0; 16];

        // i iterates through the COLUMNS of the result (and columns of 'other')
        for i in 0..4 {
            // Extract column i from 'other' (since it is column-major)
            let other_col = &other.a[i * 4..(i * 4) + 4];

            // j iterates through the ROWS of the result
            for j in 0..4 {
                // Compute dot product of row j from 'this' and column i from 'other'
                // Row j elements in column-major: M1j (j), M2j (j+4), M3j (j+8), M4j (j+12)
                ret[i * 4 + j] = this.a[j] * other_col[0]
                    + this.a[j + 4] * other_col[1]
                    + this.a[j + 8] * other_col[2]
                    + this.a[j + 12] * other_col[3];
            }
        }
        Matrix4x4 { a: ret }
    }

    #[inline(always)]
    fn m4x4_trace(this: Matrix4x4<Self>) -> Self {
        this.a[M11] + this.a[M22] + this.a[M33] + this.a[M44]
    }

    #[inline(always)]
    fn m4x4_trace_sum_squares(this: Matrix4x4<Self>) -> Self {
        this.a[M11] * this.a[M11] + this.a[M22] * this.a[M22] + this.a[M33] * this.a[M33] + this.a[M44] * this.a[M44]
    }

    #[inline(always)]
    fn m4x4_sum(this: Matrix4x4<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m4x4_mean(this: Matrix4x4<Self>) -> Self {
        Self::m4x4_sum(this) / 16.0
    }

    #[inline(always)]
    fn m4x4_product(this: Matrix4x4<Self>) -> Self {
        this.a.iter().product()
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m4x4_determinant(s: Matrix4x4<Self>) -> Self {
        // Unpack rows according to your column-major index constants
        // Row 1
        let s0  = s.a[M11]; let s1  = s.a[M12]; let s2  = s.a[M13]; let s3  = s.a[M14];
        // Row 2
        let s4  = s.a[M21]; let s5  = s.a[M22]; let s6  = s.a[M23]; let s7  = s.a[M24];
        // Row 3
        let s8  = s.a[M31]; let s9  = s.a[M32]; let s10 = s.a[M33]; let s11 = s.a[M34];
        // Row 4
        let s12 = s.a[M41]; let s13 = s.a[M42]; let s14 = s.a[M43]; let s15 = s.a[M44];

        // Pre-calculate the six 2x2 determinants for the bottom two rows (Rows 3 and 4)
        let b0 = s8 * s13 - s9 * s12;
        let b1 = s8 * s14 - s10 * s12;
        let b2 = s8 * s15 - s11 * s12;
        let b3 = s9 * s14 - s10 * s13;
        let b4 = s9 * s15 - s11 * s13;
        let b5 = s10 * s15 - s11 * s14;

        // Calculate the four required 3x3 determinants for the top row elements
        let c00 =  s5 * b5 - s6 * b4 + s7 * b3;
        let c01 = -s4 * b5 + s6 * b2 - s7 * b1;
        let c02 =  s4 * b4 - s5 * b2 + s7 * b0;
        let c03 = -s4 * b3 + s5 * b1 - s6 * b0;

        // Final dot product to resolve the 4x4 determinant
        s0 * c00 + s1 * c01 + s2 * c02 + s3 * c03
    }

    /// Returns the adjugate and determinant of a matrix.
    #[rustfmt::skip]
    #[inline(always)]
    fn m4x4_adjugate(s: Matrix4x4<Self>) -> (Matrix4x4<Self>, Self) {
        // Unpack rows using column-major index constants
        // Row 1
        let s0  = s.a[M11]; let s1  = s.a[M12]; let s2  = s.a[M13]; let s3  = s.a[M14];
        // Row 2
        let s4  = s.a[M21]; let s5  = s.a[M22]; let s6  = s.a[M23]; let s7  = s.a[M24];
        // Row 3
        let s8  = s.a[M31]; let s9  = s.a[M32]; let s10 = s.a[M33]; let s11 = s.a[M34];
        // Row 4
        let s12 = s.a[M41]; let s13 = s.a[M42]; let s14 = s.a[M43]; let s15 = s.a[M44];

        // Pre-calculate 2x2 determinants for the bottom two rows
        let b0 = s8 * s13 - s9 * s12;
        let b1 = s8 * s14 - s10 * s12;
        let b2 = s8 * s15 - s11 * s12;
        let b3 = s9 * s14 - s10 * s13;
        let b4 = s9 * s15 - s11 * s13;
        let b5 = s10 * s15 - s11 * s14;

        // Pre-calculate 2x2 determinants for the top two rows
        let t0 = s0 * s5 - s1 * s4;
        let t1 = s0 * s6 - s2 * s4;
        let t2 = s0 * s7 - s3 * s4;
        let t3 = s1 * s6 - s2 * s5;
        let t4 = s1 * s7 - s3 * s5;
        let t5 = s2 * s7 - s3 * s6;

        // Calculate cofactors (These variables form the algebraic Rows of the Adjugate)
        let c00 =  s5 * b5 - s6 * b4 + s7 * b3;
        let c01 = -s1 * b5 + s2 * b4 - s3 * b3;
        let c02 =  s13 * t5 - s14 * t4 + s15 * t3;
        let c03 = -s9 * t5 + s10 * t4 - s11 * t3;

        let c10 = -s4 * b5 + s6 * b2 - s7 * b1;
        let c11 =  s0 * b5 - s2 * b2 + s3 * b1;
        let c12 = -s12 * t5 + s14 * t2 - s15 * t1;
        let c13 =  s8 * t5 - s10 * t2 + s11 * t1;

        let c20 =  s4 * b4 - s5 * b2 + s7 * b0;
        let c21 = -s0 * b4 + s1 * b2 - s3 * b0;
        let c22 =  s12 * t4 - s13 * t2 + s15 * t0;
        let c23 = -s8 * t4 + s9 * t2 - s11 * t0;

        let c30 = -s4 * b3 + s5 * b1 - s6 * b0;
        let c31 =  s0 * b3 - s1 * b1 + s2 * b0;
        let c32 = -s12 * t3 + s13 * t1 - s14 * t0;
        let c33 =  s8 * t3 - s9 * t1 + s10 * t0;

        let determinant = s0 * c00 + s1 * c10 + s2 * c20 + s3 * c30;

        let a = [
            c00, c10, c20, c30,
            c01, c11, c21, c31,
            c02, c12, c22, c32,
            c03, c13, c23, c33,
        ];

        (Matrix4x4 { a }, determinant)
    }
}

// **** f64 ****

impl Matrix4x4Math for f64 {
    #[inline(always)]
    fn m4x4_neg(this: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let a = core::array::from_fn(|ii| -this.a[ii]);
        Matrix4x4 { a }
    }

    #[inline(always)]
    fn m4x4_abs(this: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let a = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix4x4 { a }
    }

    #[inline(always)]
    fn m4x4_add(this: Matrix4x4<Self>, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let a = core::array::from_fn(|ii| this.a[ii] + other.a[ii]);
        Matrix4x4 { a }
    }

    #[inline(always)]
    fn m4x4_mul_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self> {
        let a = core::array::from_fn(|ii| this.a[ii] * other);
        Matrix4x4 { a }
    }

    #[inline(always)]
    fn m4x4_div_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self> {
        Self::m4x4_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m4x4_mul_add(this: Matrix4x4<Self>, k: Self, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        Self::m4x4_add(Self::m4x4_mul_scalar(this, k), other)
    }

    #[inline(always)]
    fn m4x4_mul_vector(this: Matrix4x4<Self>, other: Vector4<Self>) -> Vector4<Self> {
        Vector4 {
            x: this.a[M11] * other.x + this.a[M12] * other.y + this.a[M13] * other.z + this.a[M14] * other.t,
            y: this.a[M21] * other.x + this.a[M22] * other.y + this.a[M23] * other.z + this.a[M24] * other.t,
            z: this.a[M31] * other.x + this.a[M32] * other.y + this.a[M33] * other.z + this.a[M34] * other.t,
            t: this.a[M41] * other.x + this.a[M42] * other.y + this.a[M43] * other.z + this.a[M44] * other.t,
        }
    }

    #[inline(always)]
    fn m4x4_vector_mul(this: Vector4<Self>, other: Matrix4x4<Self>) -> Vector4<Self> {
        Vector4 {
            x: this.x * other.a[M11] + this.y * other.a[M21] + this.z * other.a[M31] + this.t * other.a[M41],
            y: this.x * other.a[M12] + this.y * other.a[M22] + this.z * other.a[M32] + this.t * other.a[M42],
            z: this.x * other.a[M13] + this.y * other.a[M23] + this.z * other.a[M33] + this.t * other.a[M43],
            t: this.x * other.a[M14] + this.y * other.a[M24] + this.z * other.a[M34] + this.t * other.a[M44],
        }
    }

    // **** Outer product ****

    #[inline(always)]
    fn m4x4_vector_outer_product(col: Vector4<Self>, row: Vector4<Self>) -> Matrix4x4<Self> {
        // Structure data into local fixed-size arrays of 4 elements.
        // Since row is align(16), we manually map the implicit 4th buffer element.
        let r = [row.x, row.y, row.z, row.t];

        // Write uniform loops spanning exactly 4 elements.
        // LLVM's auto-vectorizer recognizes 4-wide float operations
        // and combines these into parallel execution blocks, if the processor supports it.
        let mut m0 = [0.0; 4];
        for ii in 0..4 {
            m0[ii] = col.x * r[ii];
        }
        let mut m1 = [0.0; 4];
        for ii in 0..4 {
            m1[ii] = col.y * r[ii];
        }
        let mut m2 = [0.0; 4];
        for ii in 0..4 {
            m2[ii] = col.z * r[ii];
        }
        let mut m3 = [0.0; 4];
        for ii in 0..4 {
            m3[ii] = col.t * r[ii];
        }

        Matrix4x4 {
            a: [
                m0[0], m0[1], m0[2], m0[3], //
                m1[0], m1[1], m1[2], m1[3], //
                m2[0], m2[1], m2[2], m2[3], //
                m3[0], m3[1], m3[2], m3[3], //
            ],
        }
    }

    #[inline(always)]
    fn m4x4_quaternion_outer_product(this: Quaternion<Self>) -> Matrix4x4<Self> {
        // Structure data into local fixed-size arrays of 4 elements.
        // Since row is align(16), we manually map the implicit 4th buffer element.
        let r = [this.w, this.x, this.y, this.z];

        // Write uniform loops spanning exactly 4 elements.
        // LLVM's auto-vectorizer recognizes 4-wide float operations
        // and combines these into parallel execution blocks, if the processor supports it.
        let mut m0 = [0.0; 4];
        for ii in 0..4 {
            m0[ii] = this.w * r[ii];
        }
        let mut m1 = [0.0; 4];
        for ii in 0..4 {
            m1[ii] = this.x * r[ii];
        }
        let mut m2 = [0.0; 4];
        for ii in 0..4 {
            m2[ii] = this.y * r[ii];
        }
        let mut m3 = [0.0; 4];
        for ii in 0..4 {
            m3[ii] = this.z * r[ii];
        }

        Matrix4x4 {
            a: [
                m0[0], m0[1], m0[2], m0[3], //
                m1[0], m1[1], m1[2], m1[3], //
                m2[0], m2[1], m2[2], m2[3], //
                m3[0], m3[1], m3[2], m3[3], //
            ],
        }
    }

    // **** Mul ****

    #[inline(always)]
    fn m4x4_mul(this: Matrix4x4<Self>, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let mut ret = [0.0; 16];

        // Explicitly tell the compiler we are working with fixed 4-element chunks
        for i in 0..4 {
            let row = &this.a[i * 4..(i * 4) + 4];
            for j in 0..4 {
                // By using a local sum and fixed indices,
                // LLVM can more easily unroll this into SIMD 'Multiply-Add' instructions.
                ret[i * 4 + j] = row[M11] * other.a[j]
                    + row[M12] * other.a[j + 4]
                    + row[M13] * other.a[j + 8]
                    + row[M14] * other.a[j + 12];
            }
        }
        Matrix4x4 { a: ret }
    }

    #[inline(always)]
    fn m4x4_trace(this: Matrix4x4<Self>) -> Self {
        this.a[M11] + this.a[M22] + this.a[M33] + this.a[M44]
    }

    #[inline(always)]
    fn m4x4_trace_sum_squares(this: Matrix4x4<Self>) -> Self {
        {
            this.a[M11] * this.a[M11]
                + this.a[M22] * this.a[M22]
                + this.a[M33] * this.a[M33]
                + this.a[M44] * this.a[M44]
        }
    }

    #[inline(always)]
    fn m4x4_sum(this: Matrix4x4<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m4x4_mean(this: Matrix4x4<Self>) -> Self {
        Self::m4x4_sum(this) / 16.0
    }

    #[inline(always)]
    fn m4x4_product(this: Matrix4x4<Self>) -> Self {
        this.a.iter().product()
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m4x4_determinant(s: Matrix4x4<Self>) -> Self {
         /*this.a[M11] * Self::m3x3_determinant(Matrix3x3 { a: [this.a[M22], this.a[M23], this.a[M24], this.a[M32], this.a[M33], this.a[M34], this.a[M42], this.a[M43], this.a[M44]]})
        -this.a[M12] * Self::m3x3_determinant(Matrix3x3 { a: [this.a[M21], this.a[M23], this.a[M24], this.a[M31], this.a[M33], this.a[M34], this.a[M41], this.a[M43], this.a[M44]]})
        +this.a[M13] * Self::m3x3_determinant(Matrix3x3 { a: [this.a[M21], this.a[M22], this.a[M24], this.a[M31], this.a[M32], this.a[M34], this.a[M41], this.a[M42], this.a[M44]]})
        -this.a[M14] * Self::m3x3_determinant(Matrix3x3 { a: [this.a[M21], this.a[M22], this.a[M23], this.a[M31], this.a[M32], this.a[M33], this.a[M41], this.a[M42], this.a[M43]]})*/
        let s0  = s.a[M11]; let s1  = s.a[M12]; let s2  = s.a[M13]; let s3  = s.a[M14];
        let s4  = s.a[M21]; let s5  = s.a[M22]; let s6  = s.a[M23]; let s7  = s.a[M24];
        let s8  = s.a[M31]; let s9  = s.a[M32]; let s10 = s.a[M33]; let s11 = s.a[M34];
        let s12 = s.a[M41]; let s13 = s.a[M42]; let s14 = s.a[M43]; let s15 = s.a[M44];

        let b0 = s8 * s13 - s9 * s12;
        let b1 = s8 * s14 - s10 * s12;
        let b2 = s8 * s15 - s11 * s12;
        let b3 = s9 * s14 - s10 * s13;
        let b4 = s9 * s15 - s11 * s13;
        let b5 = s10 * s15 - s11 * s14;

        let c00 =  s5 * b5 - s6 * b4 + s7 * b3;
        let c10 = -s4 * b5 + s6 * b2 - s7 * b1;
        let c20 =  s4 * b4 - s5 * b2 + s7 * b0;
        let c30 = -s4 * b3 + s5 * b1 - s6 * b0;

        s0 * c00 + s1 * c10 + s2 * c20 + s3 * c30
    }

    /// Returns the adjugate and determinant of a matrix.
    #[rustfmt::skip]
    #[inline(always)]
    fn m4x4_adjugate(s: Matrix4x4<Self>) -> (Matrix4x4<Self>, Self) {
        let s0  = s.a[M11]; let s1  = s.a[M12]; let s2  = s.a[M13]; let s3  = s.a[M14];
        let s4  = s.a[M21]; let s5  = s.a[M22]; let s6  = s.a[M23]; let s7  = s.a[M24];
        let s8  = s.a[M31]; let s9  = s.a[M32]; let s10 = s.a[M33]; let s11 = s.a[M34];
        let s12 = s.a[M41]; let s13 = s.a[M42]; let s14 = s.a[M43]; let s15 = s.a[M44];

        // Pre-calculate 2x2 determinants for the bottom two rows
        let b0 = s8 * s13 - s9 * s12;
        let b1 = s8 * s14 - s10 * s12;
        let b2 = s8 * s15 - s11 * s12;
        let b3 = s9 * s14 - s10 * s13;
        let b4 = s9 * s15 - s11 * s13;
        let b5 = s10 * s15 - s11 * s14;

        // Pre-calculate 2x2 determinants for the top two rows
        let t0 = s0 * s5 - s1 * s4;
        let t1 = s0 * s6 - s2 * s4;
        let t2 = s0 * s7 - s3 * s4;
        let t3 = s1 * s6 - s2 * s5;
        let t4 = s1 * s7 - s3 * s5;
        let t5 = s2 * s7 - s3 * s6;

        // Calculate cofactors (already transposed)
        let c00 =  s5 * b5 - s6 * b4 + s7 * b3;
        let c01 = -s1 * b5 + s2 * b4 - s3 * b3;
        let c02 =  s13 * t5 - s14 * t4 + s15 * t3;
        let c03 = -s9 * t5 + s10 * t4 - s11 * t3;

        let c10 = -s4 * b5 + s6 * b2 - s7 * b1;
        let c11 =  s0 * b5 - s2 * b2 + s3 * b1;
        let c12 = -s12 * t5 + s14 * t2 - s15 * t1;
        let c13 =  s8 * t5 - s10 * t2 + s11 * t1;

        let c20 =  s4 * b4 - s5 * b2 + s7 * b0;
        let c21 = -s0 * b4 + s1 * b2 - s3 * b0;
        let c22 =  s12 * t4 - s13 * t2 + s15 * t0;
        let c23 = -s8 * t4 + s9 * t2 - s11 * t0;

        let c30 = -s4 * b3 + s5 * b1 - s6 * b0;
        let c31 =  s0 * b3 - s1 * b1 + s2 * b0;
        let c32 = -s12 * t3 + s13 * t1 - s14 * t0;
        let c33 =  s8 * t3 - s9 * t1 + s10 * t0;

        let determinant = s0 * c00 + s1 * c10 + s2 * c20 + s3 * c30;

        let a = [
            c00, c01, c02, c03,
            c10, c11, c12, c13,
            c20, c21, c22, c23,
            c30, c31, c32, c33,
        ];

        (Matrix4x4 { a }, determinant)
    }
}
