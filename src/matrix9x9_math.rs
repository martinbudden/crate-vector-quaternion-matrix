#![allow(clippy::inline_always)]
#![allow(unused)]

use crate::Matrix9x9;

// **** Math ****

/// Math functions for Matrix9x9.<br><br>
pub trait Matrix9x9Math: Sized {
    fn m9x9_neg(this: Matrix9x9<Self>) -> Matrix9x9<Self>;
    fn m9x9_abs(this: Matrix9x9<Self>) -> Matrix9x9<Self>;
    fn m9x9_add(this: Matrix9x9<Self>, this: Matrix9x9<Self>) -> Matrix9x9<Self>;
    fn m9x9_mul_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self>;
    fn m9x9_div_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self>;
    fn m9x9_mul_add(this: Matrix9x9<Self>, k: Self, other: Matrix9x9<Self>) -> Matrix9x9<Self>;
    fn m9x9_mul(this: Matrix9x9<Self>, other: Matrix9x9<Self>) -> Matrix9x9<Self>;
    fn m9x9_trace(this: Matrix9x9<Self>) -> Self;
    fn m9x9_trace_sum_squares(this: Matrix9x9<Self>) -> Self;
    fn m9x9_sum(this: Matrix9x9<Self>) -> Self;
    fn m9x9_mean(this: Matrix9x9<Self>) -> Self;
    fn m9x9_product(this: Matrix9x9<Self>) -> Self;
}

impl Matrix9x9Math for f32 {
    #[inline(always)]
    fn m9x9_neg(this: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| -this.a[ii]);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_abs(this: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_add(this: Matrix9x9<Self>, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] + other.a[ii]);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_mul_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] * other);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_div_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self> {
        Self::m9x9_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m9x9_mul_add(this: Matrix9x9<Self>, k: Self, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        Self::m9x9_add(Self::m9x9_mul_scalar(this, k), other)
    }

    #[inline]
    fn m9x9_mul(this: Matrix9x9<Self>, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let mut ret = [0.0; 81];

        // Loop through the output matrix column by column (9 elements at a time).
        // In column-major layout, chunks_exact_mut(9) cleanly yields full columns,
        // which eliminates internal array bounds checks.
        #[allow(clippy::chunks_exact_to_as_chunks)]
        for (col_idx, out_column) in ret.chunks_exact_mut(9).enumerate() {
            // Cache the current column of the OTHER matrix in local memory/registers.
            let other_col_offset = col_idx * 9;
            let other_col = &other.a[other_col_offset..other_col_offset + 9];

            // Compute each row element for this specific output column
            for (row_idx, out_element) in out_column.iter_mut().enumerate() {
                // Striding through THIS matrix by 9 to read across a logical Row.
                // row_idx + 0 is Col 1, row_idx + 9 is Col 2, row_idx + 18 is Col 3, etc.
                let mut sum = this.a[row_idx] * other_col[0];
                // Unroll the remaining 8 elements of the dot product.
                // Using fixed index offsets (row_idx + 9, row_idx + 18, etc.) allows the compiler
                // to optimize this into pipelined, branchless fused multiply-add instructions.
                sum += this.a[row_idx + 9] * other_col[1];
                sum += this.a[row_idx + 18] * other_col[2];
                sum += this.a[row_idx + 27] * other_col[3];
                sum += this.a[row_idx + 36] * other_col[4];
                sum += this.a[row_idx + 45] * other_col[5];
                sum += this.a[row_idx + 54] * other_col[6];
                sum += this.a[row_idx + 63] * other_col[7];
                sum += this.a[row_idx + 72] * other_col[8];

                // Store the final computed dot product in the output column slice
                *out_element = sum;
            }
        }
        Matrix9x9::from_column_array(ret)
    }

    #[inline(always)]
    fn m9x9_trace(this: Matrix9x9<Self>) -> Self {
        let mut sum = 0.0;
        for &val in this.a.iter().step_by(10) {
            sum += val;
        }
        sum
    }

    #[inline(always)]
    fn m9x9_trace_sum_squares(this: Matrix9x9<Self>) -> Self {
        let mut sum = 0.0;
        for &val in this.a.iter().step_by(10) {
            sum += val * val;
        }
        sum
    }

    #[inline(always)]
    fn m9x9_sum(this: Matrix9x9<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m9x9_mean(this: Matrix9x9<Self>) -> Self {
        Self::m9x9_sum(this) / 16.0
    }

    #[inline(always)]
    fn m9x9_product(this: Matrix9x9<Self>) -> Self {
        this.a.iter().product()
    }
}

// **** f64 ****

impl Matrix9x9Math for f64 {
    #[inline(always)]
    fn m9x9_neg(this: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| -this.a[ii]);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_abs(this: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_add(this: Matrix9x9<Self>, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] + other.a[ii]);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_mul_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] * other);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_div_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self> {
        Self::m9x9_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m9x9_mul_add(this: Matrix9x9<Self>, k: Self, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        Self::m9x9_add(Self::m9x9_mul_scalar(this, k), other)
    }

    #[inline]
    fn m9x9_mul(this: Matrix9x9<Self>, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let mut ret = [0.0; 81];

        // Loop through the output matrix column by column (9 elements at a time).
        // In column-major layout, chunks_exact_mut(9) cleanly yields full columns,
        // which eliminates internal array bounds checks.
        #[allow(clippy::chunks_exact_to_as_chunks)]
        for (col_idx, out_column) in ret.chunks_exact_mut(9).enumerate() {
            // Cache the current column of the OTHER matrix in local memory/registers.
            let other_col_offset = col_idx * 9;
            let other_col = &other.a[other_col_offset..other_col_offset + 9];

            // Compute each row element for this specific output column
            for (row_idx, out_element) in out_column.iter_mut().enumerate() {
                // Striding through THIS matrix by 9 to read across a logical Row.
                // row_idx + 0 is Col 1, row_idx + 9 is Col 2, row_idx + 18 is Col 3, etc.
                let mut sum = this.a[row_idx] * other_col[0];
                // Unroll the remaining 8 elements of the dot product.
                // Using fixed index offsets (row_idx + 9, row_idx + 18, etc.) allows the compiler
                // to optimize this into pipelined, branchless fused multiply-add instructions.
                sum += this.a[row_idx + 9] * other_col[1];
                sum += this.a[row_idx + 18] * other_col[2];
                sum += this.a[row_idx + 27] * other_col[3];
                sum += this.a[row_idx + 36] * other_col[4];
                sum += this.a[row_idx + 45] * other_col[5];
                sum += this.a[row_idx + 54] * other_col[6];
                sum += this.a[row_idx + 63] * other_col[7];
                sum += this.a[row_idx + 72] * other_col[8];

                // Store the final computed dot product in the output column slice
                *out_element = sum;
            }
        }
        Matrix9x9::from_column_array(ret)
    }

    #[inline(always)]
    fn m9x9_trace(this: Matrix9x9<Self>) -> Self {
        let mut sum = 0.0;
        for &val in this.a.iter().step_by(10) {
            sum += val;
        }
        sum
    }

    #[inline(always)]
    fn m9x9_trace_sum_squares(this: Matrix9x9<Self>) -> Self {
        let mut sum = 0.0;
        for &val in this.a.iter().step_by(10) {
            sum += val * val;
        }
        sum
    }

    #[inline(always)]
    fn m9x9_sum(this: Matrix9x9<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m9x9_mean(this: Matrix9x9<Self>) -> Self {
        Self::m9x9_sum(this) / 16.0
    }

    #[inline(always)]
    fn m9x9_product(this: Matrix9x9<Self>) -> Self {
        this.a.iter().product()
    }
}
