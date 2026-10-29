use num_traits::{ConstOne, ConstZero, float::FloatCore};

use crate::{MathMethods, Matrix3x3, Vector3};

impl<T> Matrix3x3<T>
where
    T: Copy + FloatCore + MathMethods + ConstZero,
{
    /// Returns a wrapped Cholesky lower decomposition of a matrix if the matrix is positive definite, otherwise `None`.
    /// The input matrix is assumed to be symmetric and only the lower-triangular part is read.
    #[must_use]
    #[inline]
    pub fn cholesky(self) -> Option<Cholesky3x3<T>> {
        Cholesky3x3::new(self)
    }
}

/// Cholesky3x3 matrix of `f32` values<br>
pub type Cholesky3x3f32 = Cholesky3x3<f32>;
/// Cholesky 3x3 matrix of `f64` values<br><br>
pub type Cholesky3x3f64 = Cholesky3x3<f64>;

/// Array representing the lower triangular Cholesky solver of a matrix.
/// Note: the diagonals are stored as reciprocals.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cholesky3x3<T> {
    a: [T; CHOLESKY_3X3_SIZE],
}

const CHOLESKY_3X3_SIZE: usize = 6;

impl<T> Cholesky3x3<T>
where
    T: Copy + FloatCore + MathMethods + ConstZero,
{
    /// Size of lower triangular array representing Cholesky solver.
    pub const SIZE: usize = CHOLESKY_3X3_SIZE;

    // Column 1
    const M11: usize = 0;
    const M21: usize = 1;
    const M31: usize = 2;
    // Column 2
    const M22: usize = 3;
    const M32: usize = 4;
    // Column 3
    const M33: usize = 5;

    /// Returns a wrapped Cholesky lower decomposition of a matrix if the matrix is positive definite, otherwise `None`.
    /// The matrix is assumed to be symmetric and only the lower-triangular part is read.
    /// Note: internally the diagonal values are stored as reciprocals, but this is transparent to the user.
    /// ```
    /// # use vqm::{Cholesky3x3f32, Matrix3x3f32};
    /// let A = Matrix3x3f32::new([
    ///     2.0, 3.0, 7.0,
    ///     3.0, 5.0, 11.0,
    ///     7.0, 11.0, 29.0
    /// ]);
    /// let option = Cholesky3x3f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     const R: f32 = core::f32::consts::FRAC_1_SQRT_2;
    ///     let L = cholesky.l();
    ///     assert!(L.is_near(Matrix3x3f32::new([
    ///             2.0*R, 0.0, 0.0,
    ///             3.0*R, R,   0.0,
    ///             7.0*R, R,   2.0 ]),1e-6));
    ///     //Check L*L^T = A
    ///     let B = L * L.transpose();
    ///     assert!(A.is_near(B, 1e-6));
    /// }
    /// ```
    #[allow(clippy::many_single_char_names)]
    #[must_use]
    pub fn new(m: Matrix3x3<T>) -> Option<Self> {
        // 14 arithmetic operations: 3 sqrt_reciprocal, 7 multiplications, 4 subtractions
        let mut ret = [T::ZERO; CHOLESKY_3X3_SIZE];

        let a2 = m[Matrix3x3::<T>::M11];
        if a2 <= T::ZERO {
            return None;
        }
        let a_reciprocal = a2.sqrt_reciprocal();
        ret[Self::M11] = a_reciprocal;

        let b = m[Matrix3x3::<T>::M21] * a_reciprocal;
        ret[Self::M21] = b;

        let c2 = m[Matrix3x3::<T>::M22] - b * b;
        if c2 <= T::ZERO {
            return None;
        }
        let c_reciprocal = c2.sqrt_reciprocal();
        ret[Self::M22] = c_reciprocal;

        let d = m[Matrix3x3::<T>::M31] * a_reciprocal;
        ret[Self::M31] = d;

        let e = (m[Matrix3x3::<T>::M32] - b * d) * c_reciprocal;
        ret[Self::M32] = e;

        let f2 = m[Matrix3x3::<T>::M33] - d * d - e * e;
        if f2 <= T::ZERO {
            return None;
        }
        let f_reciprocal = f2.sqrt_reciprocal();
        ret[Self::M33] = f_reciprocal;

        Some(Self { a: ret })
    }

    /// Returns a wrapped Cholesky lower decomposition of a matrix, from an array representing the lower triangular array of said matrix.
    /// Returns `None` if the  matrix is not positive definite.
    #[allow(clippy::many_single_char_names)]
    #[must_use]
    pub fn from_lower_triangle_array(a: [T; 6]) -> Option<Self> {
        // 14 arithmetic operations: 3 sqrt_reciprocal, 7 multiplications, 4 subtractions
        let mut ret = [T::ZERO; CHOLESKY_3X3_SIZE];

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

        Some(Self { a: ret })
    }
}

impl<T> Cholesky3x3<T>
where
    T: Copy + ConstOne + ConstZero + FloatCore + MathMethods,
{
    /// Returns the lower triangular matrix represented by the Cholesky.
    /// Note: the diagonals are converted from their internal reciprocal form.
    #[must_use]
    #[inline]
    #[rustfmt::skip]
    pub fn l(self) -> Matrix3x3<T> {
        Matrix3x3::from_column_array([
            T::ONE / self.a[Self::M11], self.a[Self::M21],          self.a[Self::M31],
            T::ZERO,                    T::ONE / self.a[Self::M22], self.a[Self::M32],
            T::ZERO,                    T::ZERO,                    T::ONE / self.a[Self::M33],
        ])
    }

    /// Solves A * x = b, for array x and positive-definite matrix A, by using the Cholesky lower-triangular form.
    #[must_use]
    #[inline]
    pub fn solve_array(self, a: [T; 3]) -> [T; 3] {
        // Note: since the diagonals are stored as reciprocals we multiply by them in the forward and back substitutions.
        // 20 arithmetic operations: 12 multiplies, 8 subtractions.

        // Forward substitution: L * y = a.
        let y0 = a[0] * self.a[Self::M11];
        let y1 = (a[1] - self.a[Self::M21] * y0) * self.a[Self::M22];
        let y2 = (a[2] - self.a[Self::M31] * y0 - self.a[Self::M32] * y1) * self.a[Self::M33];

        // Back substitution: L^T * x = y.
        let x2 = y2 * self.a[Self::M33];
        let x1 = (y1 - self.a[Self::M32] * x2) * self.a[Self::M22];
        let x0 = (y0 - self.a[Self::M21] * x1 - self.a[Self::M31] * x2) * self.a[Self::M11];

        [x0, x1, x2]
    }

    /// Solves A * x = b, for vector x and positive-definite matrix A, by using the Cholesky lower-triangular form.
    /// ```
    /// # use vqm::{Cholesky3x3f32, Matrix3x3f32, Vector3f32};
    /// let A = Matrix3x3f32::new([
    ///     2.0, 3.0, 7.0,
    ///     3.0, 5.0, 11.0,
    ///     7.0, 11.0, 29.0,
    /// ]);
    ///     let b = Vector3f32 { x: 17.0, y: 23.0, z: 31.0 };
    ///
    /// let option = Cholesky3x3f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     // Solve Ax = b, for x
    ///     let x = cholesky.solve_vector(b);
    ///     assert!(x.is_near(Vector3f32 { x: 29.0, y: 1.5, z: -6.5 }, 5e-5));
    ///     assert!((A * x).is_near(b, 5e-4));
    /// }
    /// ```
    #[must_use]
    pub fn solve_vector(self, v: Vector3<T>) -> Vector3<T> {
        let a = self.solve_array([v.x, v.y, v.z]);
        Vector3 { x: a[0], y: a[1], z: a[2] }
    }

    /// Solves A * X = b, for matrix X and positive-definite matrix A, by using the Cholesky lower-triangular form.
    /// ```
    /// # use vqm::{Cholesky3x3f32, Matrix3x3f32};
    /// let A = Matrix3x3f32::new([
    ///     2.0, 3.0, 7.0,
    ///     3.0, 5.0, 11.0,
    ///     7.0, 11.0, 29.0
    /// ]);
    /// let B = Matrix3x3f32::new([
    ///     13.0, 23.0, 41.0,
    ///     37.0, 31.0, 47.0,
    ///     19.0, 37.0, 59.0
    /// ]);
    /// let option = Cholesky3x3f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     let X = cholesky.solve_matrix(B);
    ///     assert!((A * X).is_near(B, 1e-4));
    /// }
    /// ```
    #[must_use]
    pub fn solve_matrix(self, other: Matrix3x3<T>) -> Matrix3x3<T> {
        // 60 arithmetic operations: 36 multiplies, 24 subtractions.
        let mut a = [T::ZERO; 9];

        for offset in (0..9).step_by(3) {
            let rhs = [other[offset], other[offset + 1], other[offset + 2]];

            let solution = self.solve_array(rhs);

            a[offset] = solution[0];
            a[offset + 1] = solution[1];
            a[offset + 2] = solution[2];
        }

        Matrix3x3 { a }
    }

    /// Equivalent to `solve_matrix(B.transpose()).transpose()`.
    /// ```
    /// # use vqm::{Cholesky3x3f32, Matrix3x3f32};
    /// let A = Matrix3x3f32::new([
    ///     2.0, 3.0, 7.0,
    ///     3.0, 5.0, 11.0,
    ///     7.0, 11.0, 29.0
    /// ]);
    /// let B = Matrix3x3f32::new([
    ///     13.0, 23.0, 41.0,
    ///     37.0, 31.0, 47.0,
    ///     19.0, 37.0, 59.0
    /// ]);
    /// let option = Cholesky3x3f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     let X = cholesky.solve_matrix(B);
    ///     assert!((A * X).is_near(B, 1e-4));
    ///     let Y = cholesky.solve_matrix_transposed(B);
    ///     let Z = cholesky.solve_matrix(B.transpose()).transpose();
    ///     assert!(Z.is_near(Y, 1e-7));
    /// }
    /// ```
    #[must_use]
    pub fn solve_matrix_transposed(self, other: Matrix3x3<T>) -> Matrix3x3<T> {
        let mut a = [T::ZERO; 9];

        // Solve A * B = C, where B = L * L^T.
        // Each row of A satisfies B * a_row^T = c_row^T.

        for row in 0..3 {
            // Extract a row of C.
            let rhs = [other.a[row], other.a[row + 3], other.a[row + 6]];

            // Solve B * a_row^T = c_row^T.
            let solution = self.solve_array(rhs);

            // Store the solution as a row of X.
            a[row] = solution[0];
            a[row + 3] = solution[1];
            a[row + 6] = solution[2];
        }

        Matrix3x3 { a }
    }

    /// Inverts Cholesky lower triangular matrix.
    /// ```
    /// # use vqm::{Cholesky3x3f32, Matrix3x3f32};
    /// let A = Matrix3x3f32::new([
    ///     2.0, 3.0, 7.0,
    ///     3.0, 5.0, 11.0,
    ///     7.0, 11.0, 29.0
    /// ]);
    /// let option = Cholesky3x3f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     let X = cholesky.invert();
    ///     assert!(X.is_near(Matrix3x3f32::new([
    ///                  6.0,  -2.5,  -0.5,
    ///                 -2.5,   2.25, -0.25,
    ///                 -0.5,  -0.25,  0.25 ]),1e-5));
    ///     assert!((A * X).is_near_identity(1e-5));
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn invert(self) -> Matrix3x3<T> {
        self.solve_matrix(Matrix3x3::identity())
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<Cholesky3x3<f32>>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Matrix3x3f32, Vector3f32};

    #[allow(non_snake_case)]
    #[test]
    fn cholesky_tests() {
        let A = Matrix3x3f32::new([
            2.0, 3.0, 7.0, //
            3.0, 5.0, 11.0, //
            7.0, 11.0, 29.0, //
        ]);
        let B = Matrix3x3f32::new([
            13.0, 23.0, 41.0, //
            17.0, 31.0, 47.0, //
            19.0, 37.0, 59.0, //
        ]);

        let option = Cholesky3x3f32::new(A);
        assert!(option.is_some());

        if let Some(cholesky) = option {
            const R: f32 = core::f32::consts::FRAC_1_SQRT_2;
            let L = cholesky.l();
            #[rustfmt::skip]
            assert!(L.is_near(Matrix3x3f32::new([
                            2.0*R, 0.0, 0.0,
                            3.0*R, R,   0.0,
                            7.0*R, R,   2.0 ]), 1e-6));
            // Check L*L^T = A
            assert!(A.is_near(L * L.transpose(), 1e-6));

            // Solve Ax = b, for x
            let b = Vector3f32 { x: 13.0, y: 17.0, z: 19.0 };
            let x = cholesky.solve_vector(b);
            assert!(x.is_near(Vector3f32 { x: 26.0, y: 1.0, z: -6.0 }, 2e-3));
            assert!((A * x).is_near(b, 2e-5));

            // Solving A * X = A gives identity
            let X = cholesky.solve_matrix(A);
            assert!(X.is_near_identity(2e-6));

            // Solving A * X = I for X gives A inverse
            let X = cholesky.solve_matrix(Matrix3x3f32::identity());
            #[rustfmt::skip]
            assert!(X.is_near(Matrix3x3f32::new([
                            6.0,  -2.5,  -0.5,
                           -2.5,   2.25, -0.25,
                           -0.5,  -0.25,  0.25 ]), 1e-5));
            assert!((A * X).is_near_identity(1e-5));

            let X = cholesky.solve_matrix(B);
            assert!((A * X).is_near(B, 1e-4));

            let Y = cholesky.solve_matrix_transposed(B);
            let Z = cholesky.solve_matrix(B.transpose()).transpose();
            assert!(Z.is_near(Y, 1e-7));
            //assert!((A * X.transpose()).is_near(B.transpose(), 1e-2));
        }
    }

    #[rustfmt::skip]
    #[test]
    fn cholesky_from_lta() {
        let m = Matrix3x3f32::new([
            2.0,  3.0,  7.0,
            3.0,  5.0, 11.0,
            7.0, 11.0, 29.0,
        ]);
        let a :[f32;Cholesky3x3f32::SIZE] =[
            2.0, 3.0,  7.0,
                 5.0, 11.0,
                      29.0,
        ];
        let option_m = Cholesky3x3f32::new(m);
        assert!(option_m.is_some());
        let option_a = Cholesky3x3f32::from_lower_triangle_array(a);
        assert!(option_a.is_some());

        if let Some(cholesky_a) = option_a
            && let Some(cholesky_m) = option_m {
                assert_eq!(cholesky_a, cholesky_m);
            }
    }
}
