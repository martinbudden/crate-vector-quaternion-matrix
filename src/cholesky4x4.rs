use num_traits::{ConstOne, ConstZero, float::FloatCore};

use crate::{MathMethods, Matrix4x4, Vector4};

impl<T> Matrix4x4<T>
where
    T: Copy + FloatCore + MathMethods + ConstZero,
{
    /// Returns a wrapped Cholesky lower decomposition of a matrix if the matrix is positive definite, otherwise `None`.
    /// The input matrix is assumed to be symmetric and only the lower-triangular part is read.
    /// ```
    /// # use vqm::{Cholesky4x4f32, Matrix4x4f32};
    /// let A = Matrix4x4f32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///      3.0,  5.0, 11.0, 19.0,
    ///      7.0, 11.0, 29.0, 41.0,
    ///     13.0, 19.0, 41.0, 97.0,
    /// ]);
    /// let option = A.cholesky();
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     const R: f32 = core::f32::consts::FRAC_1_SQRT_2;
    ///     assert!(cholesky.l().is_near(Matrix4x4f32::new([
    ///                     2.0*R, 0.0,  0.0, 0.0,
    ///                     3.0*R,   R,  0.0, 0.0,
    ///                     7.0*R,   R,  2.0, 0.0,
    ///                    13.0*R,  -R, -2.0, 4.0*R]), 1e-5));
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub fn cholesky(self) -> Option<Cholesky4x4<T>> {
        Cholesky4x4::new(self)
    }
}

/// Cholesky4x4 matrix of `f32` values<br>
pub type Cholesky4x4f32 = Cholesky4x4<f32>;
/// Cholesky 4x4 matrix of `f64` values<br><br>
pub type Cholesky4x4f64 = Cholesky4x4<f64>;

/// Array representing the lower triangular Cholesky solver of a matrix.
/// Note: the diagonals are stored as reciprocals.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cholesky4x4<T> {
    a: [T; CHOLESKY_4X4_SIZE],
}

const CHOLESKY_4X4_SIZE: usize = 10;

impl<T> Cholesky4x4<T>
where
    T: Copy + FloatCore + MathMethods + ConstZero,
{
    /// Size of lower triangular array representing Cholesky solver.
    pub const SIZE: usize = CHOLESKY_4X4_SIZE;

    // Column 1
    const M11: usize = 0;
    const M21: usize = 1;
    const M31: usize = 2;
    const M41: usize = 3;
    // Column 2
    const M22: usize = 4;
    const M32: usize = 5;
    const M42: usize = 6;
    // Column 3
    const M33: usize = 7;
    const M43: usize = 8;
    // Column 4
    const M44: usize = 9;

    /// Returns a wrapped Cholesky lower decomposition of a matrix if the matrix is positive definite, otherwise `None`.
    /// The matrix is assumed to be symmetric and only the lower-triangular part is read.
    /// Note: internally the diagonal values are stored as reciprocals, but this is transparent to the user.
    /// ```
    /// # use vqm::{Cholesky4x4f32, Matrix4x4f32};
    /// let A = Matrix4x4f32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///      3.0,  5.0, 11.0, 19.0,
    ///      7.0, 11.0, 29.0, 41.0,
    ///     13.0, 19.0, 41.0, 97.0,
    /// ]);
    /// let option = Cholesky4x4f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     const R: f32 = core::f32::consts::FRAC_1_SQRT_2;
    ///     assert!(cholesky.l().is_near(Matrix4x4f32::new([
    ///                     2.0*R, 0.0,  0.0, 0.0,
    ///                     3.0*R,   R,  0.0, 0.0,
    ///                     7.0*R,   R,  2.0, 0.0,
    ///                    13.0*R,  -R, -2.0, 4.0*R]), 1e-5));
    /// }
    /// ```
    #[allow(clippy::many_single_char_names)]
    #[must_use]
    pub fn new(m: Matrix4x4<T>) -> Option<Self> {
        // 26 arithmetic operations: 4 sqrt_reciprocal, 12 multiplications, 10 subtractions
        let mut ret = [T::ZERO; CHOLESKY_4X4_SIZE];

        let a2 = m[Matrix4x4::<T>::M11];
        if a2 <= T::ZERO {
            return None;
        }
        let a_reciprocal = a2.sqrt_reciprocal();
        ret[Self::M11] = a_reciprocal;

        let b = m[Matrix4x4::<T>::M21] * a_reciprocal;
        ret[Self::M21] = b;

        let c2 = m[Matrix4x4::<T>::M22] - b * b;
        if c2 <= T::ZERO {
            return None;
        }
        let c_reciprocal = c2.sqrt_reciprocal();
        ret[Self::M22] = c_reciprocal;

        let d = m[Matrix4x4::<T>::M31] * a_reciprocal;
        ret[Self::M31] = d;

        let e = (m[Matrix4x4::<T>::M32] - b * d) * c_reciprocal;
        ret[Self::M32] = e;

        let f2 = m[Matrix4x4::<T>::M33] - d * d - e * e;
        if f2 <= T::ZERO {
            return None;
        }
        let f_reciprocal = f2.sqrt_reciprocal();
        ret[Self::M33] = f_reciprocal;

        let g = m[Matrix4x4::<T>::M41] * a_reciprocal;
        ret[Self::M41] = g;

        let h = (m[Matrix4x4::<T>::M42] - b * g) * c_reciprocal;
        ret[Self::M42] = h;

        let i = (m[Matrix4x4::<T>::M43] - d * g - e * h) * f_reciprocal;
        ret[Self::M43] = i;

        let j2 = m[Matrix4x4::<T>::M44] - g * g - h * h - i * i;
        if j2 <= T::ZERO {
            return None;
        }
        let j_reciprocal = j2.sqrt_reciprocal();
        ret[Self::M44] = j_reciprocal;

        Some(Self { a: ret })
    }

    /// Returns a wrapped Cholesky lower decomposition of a matrix, from an array representing the lower triangular array of said matrix.
    /// Returns `None` if the  matrix is not positive definite.
    #[allow(clippy::many_single_char_names)]
    #[must_use]
    pub fn from_lower_triangle_array(a: [T; 10]) -> Option<Self> {
        // 26 arithmetic operations: 4 sqrt_reciprocal, 12 multiplications, 10 subtractions
        let mut ret = [T::ZERO; CHOLESKY_4X4_SIZE];

        let a2 = a[Self::M11];
        if a2 <= T::ZERO {
            return None;
        }
        let a_reciprocal = a2.sqrt_reciprocal();
        ret[Self::M11] = a_reciprocal;

        let b = a[Self::M21] * a_reciprocal;
        ret[Self::M21] = b;

        let c2 = a[Self::M22] - b * b;
        if c2 <= T::ZERO {
            return None;
        }
        let c_reciprocal = c2.sqrt_reciprocal();
        ret[Self::M22] = c_reciprocal;

        let d = a[Self::M31] * a_reciprocal;
        ret[Self::M31] = d;

        let e = (a[Self::M32] - b * d) * c_reciprocal;
        ret[Self::M32] = e;

        let f2 = a[Self::M33] - d * d - e * e;
        if f2 <= T::ZERO {
            return None;
        }
        let f_reciprocal = f2.sqrt_reciprocal();
        ret[Self::M33] = f_reciprocal;

        let g = a[Self::M41] * a_reciprocal;
        ret[Self::M41] = g;

        let h = (a[Self::M42] - b * g) * c_reciprocal;
        ret[Self::M42] = h;

        let i = (a[Self::M43] - d * g - e * h) * f_reciprocal;
        ret[Self::M43] = i;

        let j2 = a[Self::M44] - g * g - h * h - i * i;
        if j2 <= T::ZERO {
            return None;
        }
        let j_reciprocal = j2.sqrt_reciprocal();
        ret[Self::M44] = j_reciprocal;

        Some(Self { a: ret })
    }
}

impl<T> Cholesky4x4<T>
where
    T: Copy + ConstOne + ConstZero + FloatCore + MathMethods,
{
    /// Returns the lower triangular matrix represented by the Cholesky.
    /// Note: the diagonals are converted from their internal reciprocal form.
    #[must_use]
    #[inline]
    #[rustfmt::skip]
    pub fn l(self) -> Matrix4x4<T> {
        Matrix4x4::from_column_array([
            T::ONE / self.a[Self::M11], self.a[Self::M21],            self.a[Self::M31],            self.a[Self::M41],
            T::ZERO,                    T::ONE / self.a[Self::M22], self.a[Self::M32],            self.a[Self::M42],
            T::ZERO,                    T::ZERO,                    T::ONE / self.a[Self::M33], self.a[Self::M43],
            T::ZERO,                    T::ZERO,                    T::ZERO,                    T::ONE / self.a[Self::M44],
        ])
    }

    /// Solves A * x = b, for array x and positive-definite matrix A, by using the Cholesky lower-triangular form.
    /// ```
    /// # use vqm::{Cholesky4x4f32, Matrix4x4f32, Vector4f32};
    /// let A = Matrix4x4f32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///      3.0,  5.0, 11.0, 19.0,
    ///      7.0, 11.0, 29.0, 41.0,
    ///     13.0, 19.0, 41.0, 97.0,
    /// ]);
    /// let b = [ 17.0, 23.0, 31.0, 47.0 ];
    ///
    /// let option = Cholesky4x4f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     // Solve Ax = b, for x
    ///     let x = cholesky.solve_array(b);
    ///     assert!( ((x[0] - 144.0).abs() < 5e-4) && ((x[1] - 1.5).abs() < 5e-4) && ((x[2] + 18.0).abs() < 5e-4) && ((x[3] + 11.5).abs() < 5e-4) );
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub fn solve_array(self, a: [T; 4]) -> [T; 4] {
        // Note: since the diagonals are stored as reciprocals we multiply by them in the forward and back substitutions.
        // 26 arithmetic operations: 12 multiplies, 14 subtractions.

        // Forward substitution: L * y = a.
        let y0 = a[0] * self.a[Self::M11];
        let y1 = (a[1] - self.a[Self::M21] * y0) * self.a[Self::M22];
        let y2 = (a[2] - self.a[Self::M31] * y0 - self.a[Self::M32] * y1) * self.a[Self::M33];
        let y3 = (a[3] - self.a[Self::M41] * y0 - self.a[Self::M42] * y1 - self.a[Self::M43] * y2) * self.a[Self::M44];

        // Solve L^T * x = y (back substitution).
        let x3 = y3 * self.a[Self::M44];
        let x2 = (y2 - self.a[Self::M43] * x3) * self.a[Self::M33];
        let x1 = (y1 - self.a[Self::M32] * x2 - self.a[Self::M42] * x3) * self.a[Self::M22];
        let x0 = (y0 - self.a[Self::M21] * x1 - self.a[Self::M31] * x2 - self.a[Self::M41] * x3) * self.a[Self::M11];

        [x0, x1, x2, x3]
    }

    /// Solves A * x = b, for vector x and positive-definite matrix A, by using the Cholesky lower-triangular form.
    /// ```
    /// # use vqm::{Cholesky4x4f32, Matrix4x4f32, Vector4f32};
    /// let A = Matrix4x4f32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///      3.0,  5.0, 11.0, 19.0,
    ///      7.0, 11.0, 29.0, 41.0,
    ///     13.0, 19.0, 41.0, 97.0,
    /// ]);
    /// let b = Vector4f32 { x: 17.0, y: 23.0, z: 31.0, t: 47.0 };
    ///
    /// let option = Cholesky4x4f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     // Solve Ax = b, for x
    ///     let x = cholesky.solve_vector(b);
    ///     assert!(x.is_near(Vector4f32 { x: 144.0, y: 1.5, z: -18.0, t: -11.5 }, 5e-4));
    ///     assert!((A * x).is_near(b, 5e-4));
    /// }
    /// ```
    #[must_use]
    pub fn solve_vector(self, v: Vector4<T>) -> Vector4<T> {
        let a = self.solve_array([v.x, v.y, v.z, v.t]);
        Vector4 { x: a[0], y: a[1], z: a[2], t: a[3] }
    }

    /// Solves A * X = b, for matrix X and positive-definite matrix A, by using the Cholesky lower-triangular form.
    /// ```
    /// # use vqm::{Cholesky4x4f32, Matrix4x4f32};
    /// let A = Matrix4x4f32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///      3.0,  5.0, 11.0, 19.0,
    ///      7.0, 11.0, 29.0, 41.0,
    ///     13.0, 19.0, 41.0, 97.0,
    /// ]);
    /// let option = Cholesky4x4f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     let X = cholesky.solve_matrix(A);
    ///     assert!(X.is_near_identity(1e-1));
    ///     //assert!((A * X).is_near(B, 1e-4));
    /// }
    /// ```
    #[must_use]
    pub fn solve_matrix(self, other: Matrix4x4<T>) -> Matrix4x4<T> {
        // 104 arithmetic operations: 48 multiplies, 56 subtractions.
        let mut a = [T::ZERO; 16];

        for offset in (0..16).step_by(4) {
            let rhs = [other[offset], other[offset + 1], other[offset + 2], other[offset + 3]];

            let solution = self.solve_array(rhs);

            a[offset] = solution[0];
            a[offset + 1] = solution[1];
            a[offset + 2] = solution[2];
            a[offset + 3] = solution[3];
        }

        Matrix4x4 { a }
    }

    /// Equivalent to `solve_matrix(B.transpose()).transpose()`.
    #[must_use]
    pub fn solve_matrix_transposed(self, other: Matrix4x4<T>) -> Matrix4x4<T> {
        let mut a = [T::ZERO; 16];

        // Solve A * B = C, where B = L * L^T.
        // Each row of A satisfies B * a_row^T = c_row^T.

        for row in 0..3 {
            // Extract a row of C.
            let rhs = [other.a[row], other.a[row + 4], other.a[row + 8], other.a[row + 12]];

            // Solve B * a_row^T = c_row^T.
            let solution = self.solve_array(rhs);

            // Store the solution as a row of X.
            a[row] = solution[0];
            a[row + 4] = solution[1];
            a[row + 8] = solution[2];
            a[row + 12] = solution[3];
        }

        Matrix4x4 { a }
    }

    /// Inverts Cholesky lower triangular matrix.
    /// ```
    /// # use vqm::{Cholesky4x4f32, Matrix4x4f32};
    /// let A = Matrix4x4f32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///      3.0,  5.0, 11.0, 19.0,
    ///      7.0, 11.0, 29.0, 41.0,
    ///     13.0, 19.0, 41.0, 97.0,
    /// ]);
    /// let option = Cholesky4x4f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     let X = cholesky.invert();
    ///     assert!(X.is_near(Matrix4x4f32::new([
    ///                 18.5,  -2.5,  -1.75, -1.25,
    ///                 -2.5,   2.25, -0.25,  0.0,
    ///                 -1.75, -0.25,  0.375, 0.125,
    ///                 -1.25,  0.0,   0.125, 0.125 ]), 5e-5));
    ///     assert!((A * X).is_near_identity(1e-5));
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn invert(self) -> Matrix4x4<T> {
        self.solve_matrix(Matrix4x4::identity())
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<Cholesky4x4<f32>>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix4x4f32;

    #[cfg(feature = "libm")]
    #[allow(non_snake_case)]
    #[test]
    fn cholesky_tests() {
        use crate::Vector4f32;
        let A = Matrix4x4f32::new([
            2.0, 3.0, 7.0, 13.0, //
            3.0, 5.0, 11.0, 19.0, //
            7.0, 11.0, 29.0, 41.0, //
            13.0, 19.0, 41.0, 97.0, //
        ]);
        let option = Cholesky4x4f32::new(A);
        assert!(option.is_some());
        if let Some(cholesky) = option {
            const R: f32 = core::f32::consts::FRAC_1_SQRT_2;
            let L = cholesky.l();
            #[cfg(feature = "libm")]
            #[rustfmt::skip]
            assert!(L.is_near(Matrix4x4f32::new([
                            2.0*R, 0.0,  0.0, 0.0,
                            3.0*R,   R,  0.0, 0.0,
                            7.0*R,   R,  2.0, 0.0,
                           13.0*R,  -R, -2.0, 4.0*R]), 1e-5));
            // Check L*L^T = A
            let B = L * L.transpose();
            #[cfg(feature = "libm")]
            assert!(A.is_near(B, 1e-6));

            // Solve Ax = b, for x
            let b = Vector4f32 { x: 17.0, y: 23.0, z: 31.0, t: 47.0 };
            let x = cholesky.solve_vector(b);
            //assert_eq!(x, Vector4f32 { x: 136.0, y: 1.0, z: -17.0, t: -11.0 });
            assert!(x.is_near(Vector4f32 { x: 144.0, y: 1.5, z: -18.0, t: -11.5 }, 5e-4));
            assert!((A * x).is_near(b, 5e-4));

            // Solving A * X = A gives identity
            let X = cholesky.solve_matrix(A);
            assert!(X.is_near_identity(5e-6));

            // Solving A * X = I for X gives A inverse
            let X = cholesky.solve_matrix(Matrix4x4f32::identity());
            #[rustfmt::skip]
            assert!(X.is_near(Matrix4x4f32::new([
                           18.5,  -2.5,  -1.75, -1.25,
                           -2.5,   2.25, -0.25,  0.0,
                           -1.75, -0.25,  0.375, 0.125,
                           -1.25,  0.0,   0.125, 0.125 ]), 5e-5));
            assert!((A * X).is_near_identity(1e-5));
        }
    }

    #[rustfmt::skip]
    #[test]
    fn cholesky_from_lta() {
        let m = Matrix4x4f32::new([
             2.0,  3.0,  7.0, 13.0,
             3.0,  5.0, 11.0, 19.0, 
             7.0, 11.0, 29.0, 41.0,
            13.0, 19.0, 41.0, 97.0,
        ]);
        let a :[f32;Cholesky4x4f32::SIZE] =[
            2.0, 3.0,  7.0, 13.0, 
                 5.0, 11.0, 19.0, 
                      29.0, 41.0,
                            97.0,
        ];
        let option_m = Cholesky4x4f32::new(m);
        assert!(option_m.is_some());
        let option_a = Cholesky4x4f32::from_lower_triangle_array(a);
        assert!(option_a.is_some());

        if let Some(cholesky_a) = option_a
            && let Some(cholesky_m) = option_m {
                assert_eq!(cholesky_a, cholesky_m);
            }
    }
}
