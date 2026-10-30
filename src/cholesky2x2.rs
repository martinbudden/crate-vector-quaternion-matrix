use num_traits::{ConstOne, ConstZero, float::FloatCore};

use crate::{MathMethods, Matrix2x2, Vector2};

impl<T> Matrix2x2<T>
where
    T: Copy + FloatCore + MathMethods + ConstZero,
{
    /// Returns a wrapped Cholesky lower decomposition of a matrix if the matrix is positive definite, otherwise `None`.
    /// The input matrix is assumed to be symmetric and only the lower-triangular part is read.
    /// ```
    /// # use vqm::{Cholesky2x2f32, Matrix2x2f32};
    /// let A = Matrix2x2f32::new([
    ///     2.0, 3.0,
    ///     3.0, 5.0
    /// ]);
    /// let option = A.cholesky();
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     const R: f32 = core::f32::consts::FRAC_1_SQRT_2;
    ///     let L = cholesky.l();
    ///     assert!(L.is_near(Matrix2x2f32::new([
    ///             2.0*R, 0.0,
    ///             3.0*R, R]),1e-6));
    ///     // Check L*L^T = A
    ///     let B = L * L.transpose();
    ///     assert!(A.is_near(B, 1e-6));
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub fn cholesky(self) -> Option<Cholesky2x2<T>> {
        Cholesky2x2::new(self)
    }
}

/// Cholesky2x2 matrix of `f32` values<br>
pub type Cholesky2x2f32 = Cholesky2x2<f32>;
/// Cholesky 2x2 matrix of `f64` values<br><br>
pub type Cholesky2x2f64 = Cholesky2x2<f64>;

/// Array representing the lower triangular Cholesky solver of a matrix.
/// Note: the diagonals are stored as reciprocals.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cholesky2x2<T> {
    a: [T; CHOLESKY_2X2_SIZE],
}

const CHOLESKY_2X2_SIZE: usize = 3;

impl<T> Cholesky2x2<T>
where
    T: Copy + FloatCore + MathMethods + ConstZero,
{
    /// Size of lower triangular array representing Cholesky solver.
    pub const SIZE: usize = CHOLESKY_2X2_SIZE;

    // Column 1
    const M11: usize = 0;
    const M21: usize = 1;
    // Column 2
    const M22: usize = 2;

    /// Returns a wrapped Cholesky lower decomposition of a matrix if the matrix is positive definite, otherwise `None`.
    /// The matrix is assumed to be symmetric and only the lower-triangular part is read.
    /// Note: internally the diagonal values are stored as reciprocals, but this is transparent to the user.
    /// ```
    /// # use vqm::{Cholesky2x2f32, Matrix2x2f32};
    /// let A = Matrix2x2f32::new([
    ///     2.0, 3.0,
    ///     3.0, 5.0
    /// ]);
    /// let option = Cholesky2x2f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     const R: f32 = core::f32::consts::FRAC_1_SQRT_2;
    ///     let L = cholesky.l();
    ///     assert!(L.is_near(Matrix2x2f32::new([
    ///             2.0*R, 0.0,
    ///             3.0*R, R]),1e-6));
    ///     // Check L*L^T = A
    ///     let B = L * L.transpose();
    ///     assert!(A.is_near(B, 1e-6));
    /// }
    /// ```
    #[allow(clippy::many_single_char_names)]
    #[must_use]
    pub fn new(m: Matrix2x2<T>) -> Option<Self> {
        // 5 arithmetic operations: 2 sqrt_reciprocal, 2 multiplications, 1 subtractions
        let mut ret = [T::ZERO; CHOLESKY_2X2_SIZE];

        let a2 = m[Matrix2x2::<T>::M11];
        if a2 <= T::ZERO {
            return None;
        }
        let a_reciprocal = a2.sqrt_reciprocal();
        ret[Self::M11] = a_reciprocal;

        let b = m[Matrix2x2::<T>::M21] * a_reciprocal;
        ret[Self::M21] = b;

        let c2 = m[Matrix2x2::<T>::M22] - b * b;
        if c2 <= T::ZERO {
            return None;
        }
        let c_reciprocal = c2.sqrt_reciprocal();
        ret[Self::M22] = c_reciprocal;

        Some(Self { a: ret })
    }

    /// Returns a wrapped Cholesky lower decomposition of a matrix, from an array representing the lower triangular array of said matrix.
    /// Returns `None` if the  matrix is not positive definite.
    #[allow(clippy::many_single_char_names)]
    #[must_use]
    pub fn from_lower_triangle_array(a: [T; 3]) -> Option<Self> {
        // 5 arithmetic operations: 2 sqrt_reciprocal, 2 multiplications, 1 subtractions
        let mut ret = [T::ZERO; CHOLESKY_2X2_SIZE];

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

        Some(Self { a: ret })
    }
}

impl<T> Cholesky2x2<T>
where
    T: Copy + ConstOne + ConstZero + FloatCore + MathMethods,
{
    /// Returns the lower triangular matrix represented by the Cholesky.
    /// Note: the diagonals are converted from their internal reciprocal form.
    #[must_use]
    #[inline]
    #[rustfmt::skip]
    pub fn l(self) -> Matrix2x2<T> {
        Matrix2x2::from_column_array([
            T::ONE / self.a[Self::M11], self.a[Self::M21],
            T::ZERO,                    T::ONE / self.a[Self::M22],
        ])
    }

    /// Solves A * x = b, for array x and positive-definite matrix A, by using the Cholesky lower-triangular form.
    /// ```
    /// # use vqm::{Cholesky2x2f32, Matrix2x2f32, Vector2f32};
    /// let A = Matrix2x2f32::new([
    ///     2.0, 3.0,
    ///     3.0, 5.0
    /// ]);
    /// let b = [ 13.0, 17.0 ];
    ///
    /// let option = Cholesky2x2f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     // Solve Ax = b, for x
    ///     let x = cholesky.solve_array(b);
    ///     assert!( ((x[0] - 14.0).abs() < 5e-5) && ((x[1] + 5.0).abs() < 5e-5) );
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub fn solve_array(self, a: [T; 2]) -> [T; 2] {
        // Note: since the diagonals are stored as reciprocals we multiply by them in the forward and back substitutions.
        // 8 arithmetic operations: 6 multiplies, 2 subtractions.

        // Forward substitution: L * y = a.
        let y0 = a[0] * self.a[Self::M11];
        let y1 = (a[1] - self.a[Self::M21] * y0) * self.a[Self::M22];

        // Back substitution: L^T * x = y.
        let x1 = y1 * self.a[Self::M22];
        let x0 = (y0 - self.a[Self::M21] * x1) * self.a[Self::M11];

        [x0, x1]
    }

    /// Solves A * x = b, for positive-definite A by using the Cholesky lower-triangular form.
    /// ```
    /// # use vqm::{Cholesky2x2f32, Matrix2x2f32, Vector2f32};
    /// let A = Matrix2x2f32::new([
    ///     2.0, 3.0,
    ///     3.0, 5.0
    /// ]);
    /// let b = Vector2f32 { x: 13.0, y: 17.0 };
    ///
    /// let option = Cholesky2x2f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     // Solve Ax = b, for x
    ///     let x = cholesky.solve_vector(b);
    ///     assert!(x.is_near(Vector2f32 { x: 14.0, y: -5.0 }, 5e-5));
    /// }
    /// ```
    #[must_use]
    pub fn solve_vector(self, v: Vector2<T>) -> Vector2<T> {
        let a = self.solve_array([v.x, v.y]);
        Vector2 { x: a[0], y: a[1] }
    }

    /// Solves A * X = b, for matrix X and positive-definite matrix A, by using the Cholesky lower-triangular form.
    /// ```
    /// # use vqm::{Cholesky2x2f32, Matrix2x2f32};
    /// let A = Matrix2x2f32::new([
    ///     2.0, 3.0,
    ///     3.0, 5.0
    /// ]);
    /// let B = Matrix2x2f32::new([
    ///     13.0, 23.0,
    ///     17.0, 31.0
    /// ]);
    /// let option = Cholesky2x2f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     let X = cholesky.solve_matrix(B);
    ///     assert!((A * X).is_near(B, 1e-4));
    /// }
    /// ```
    #[must_use]
    pub fn solve_matrix(self, other: Matrix2x2<T>) -> Matrix2x2<T> {
        // 16 arithmetic operations: 12 multiplies, 4 subtractions.
        let mut a = [T::ZERO; 4];

        for offset in (0..4).step_by(2) {
            let rhs = [other[offset], other[offset + 1]];

            let solution = self.solve_array(rhs);

            a[offset] = solution[0];
            a[offset + 1] = solution[1];
        }

        Matrix2x2 { a }
    }

    /// Equivalent to `solve_matrix(B.transpose()).transpose()`.
    /// ```
    /// # use vqm::{Cholesky2x2f32, Matrix2x2f32};
    /// let A = Matrix2x2f32::new([
    ///     2.0, 3.0,
    ///     3.0, 5.0
    /// ]);
    /// let B = Matrix2x2f32::new([
    ///     13.0, 23.0,
    ///     17.0, 31.0
    /// ]);
    /// let option = Cholesky2x2f32::new(A);
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
    pub fn solve_matrix_transposed(self, other: Matrix2x2<T>) -> Matrix2x2<T> {
        let mut a = [T::ZERO; 4];

        // Solve A * B = C, where B = L * L^T.
        // Each row of A satisfies B * a_row^T = c_row^T.

        for row in 0..2 {
            // Extract a row of C.
            let rhs = [other.a[row], other.a[row + 2]];

            // Solve B * a_row^T = c_row^T.
            let solution = self.solve_array(rhs);

            // Store the solution as a row of X.
            a[row] = solution[0];
            a[row + 2] = solution[1];
        }

        Matrix2x2 { a }
    }

    /// Inverts Cholesky lower triangular matrix.
    /// ```
    /// # use vqm::{Cholesky2x2f32, Matrix2x2f32};
    /// let A = Matrix2x2f32::new([
    ///     2.0, 3.0,
    ///     3.0, 5.0
    /// ]);
    /// let option = Cholesky2x2f32::new(A);
    /// assert!(option.is_some());
    /// if let Some(cholesky) = option {
    ///     let X = cholesky.invert();
    ///     assert!(X.is_near(Matrix2x2f32::new([
    ///                  5.0,  -3.0,
    ///                 -3.0,   2.0 ]),1e-5));
    ///     assert!((A * X).is_near_identity(1e-5));
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn invert(self) -> Matrix2x2<T> {
        self.solve_matrix(Matrix2x2::identity())
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<Cholesky2x2<f32>>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Matrix2x2f32, Vector2f32};

    #[allow(non_snake_case)]
    #[test]
    fn cholesky_tests() {
        let A = Matrix2x2f32::new([
            2.0, 3.0, //
            3.0, 5.0, //
        ]);
        let B = Matrix2x2f32::new([
            13.0, 23.0, //
            17.0, 31.0, //
        ]);
        let option = Cholesky2x2f32::new(A);
        assert!(option.is_some());
        if let Some(cholesky) = option {
            const R: f32 = core::f32::consts::FRAC_1_SQRT_2;
            let L = cholesky.l();
            #[rustfmt::skip]
            assert!(L.is_near(Matrix2x2f32::new([
                            2.0*R, 0.0,
                            3.0*R, R ]), 1e-6));
            // Check L*L^T = A
            assert!(A.is_near(L * L.transpose(), 1e-6));

            // Solve Ax = b, for x
            let b = Vector2f32 { x: 13.0, y: 17.0 };
            let x = cholesky.solve_vector(b);
            assert!(x.is_near(Vector2f32 { x: 14.0, y: -5.0 }, 5e-5));
            assert!((A * x).is_near(b, 5e-6));

            // Solving A * X = A gives identity
            let X = cholesky.solve_matrix(A);
            assert!(X.is_near_identity(2e-6));

            // Solving A * X = I for X gives A inverse
            let X = cholesky.solve_matrix(Matrix2x2f32::identity());
            #[rustfmt::skip]
            assert!(X.is_near(Matrix2x2f32::new([
                            5.0,  -3.0,
                           -3.0,   2.0 ]), 1e-5));
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
        let m = Matrix2x2f32::new([
            2.0,  3.0,
            3.0,  5.0,
        ]);
        let a :[f32;Cholesky2x2f32::SIZE] =[
            2.0, 3.0,
                 5.0,
        ];
        let option_m = Cholesky2x2f32::new(m);
        assert!(option_m.is_some());
        let option_a = Cholesky2x2f32::from_lower_triangle_array(a);
        assert!(option_a.is_some());

        if let Some(cholesky_a) = option_a
            && let Some(cholesky_m) = option_m {
                assert_eq!(cholesky_a, cholesky_m);
            }
    }
}
