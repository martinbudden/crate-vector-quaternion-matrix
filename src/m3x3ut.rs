use core::ops::{Add, Index, Mul};
use num_traits::{Signed, float::FloatCore};

use super::{MathMethods, Matrix3x3, Vector3};

/// Upper triangular array representing a 3x3 matrix of `f32` values<br>
pub type M3x3UpperTrianglef32 = M3x3UpperTriangle<f32>;
/// Upper triangular array representing a 3x3 matrix of `f64` values<br><br>
pub type M3x3UpperTrianglef64 = M3x3UpperTriangle<f64>;

/// A 3x3 symmetric matrix stored as a flat 6-element array (Upper-Triangular layout).
/// ```text
/// [m11, m12, m13,
///       m22, m23,
///            m33]
/// ```
/// Also known as Voigt Notation or Mandel Notation.
// The overwhelming historical convention is to use upper-triangular rather than lower-triangular layout.
// This convention is used by libraries such as BLAS, LAPACK, and Intel MKL.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct M3x3UpperTriangle<T> {
    pub(crate) a: [T; UT_3X3_SIZE],
}
const UT_3X3_SIZE: usize = 6;

impl<T> M3x3UpperTriangle<T> {
    /// Size of upper triangular array.
    pub const SIZE: usize = UT_3X3_SIZE;

    // Row 1
    const M11: usize = 0;
    const M12: usize = 1;
    const M13: usize = 2;
    // Row 2
    const M22: usize = 3;
    const M23: usize = 4;
    // Row 3
    const M33: usize = 5;
}

// **** New ****

impl<T> M3x3UpperTriangle<T>
where
    T: Copy,
{
    /// Create an upper-triangle.
    /// ```
    /// # use vqm::M3x3UpperTrianglef32;
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// ```
    #[inline]
    pub const fn new(a: [T; UT_3X3_SIZE]) -> Self {
        Self { a }
    }
}

// **** Other constructors ****

impl<T> M3x3UpperTriangle<T> {
    /// Create an `M3x3UpperTriangle` from a Matrix3x3.
    /// Does not check if the matrix is symmetric.
    /// ```
    /// # use vqm::{M3x3UpperTrianglef32, Matrix3x3f32};
    /// let m = Matrix3x3f32::new([
    ///     2.0,  3.0,  7.0,
    ///     4.0,  5.0, 11.0,
    ///     6.0,  8.0, 29.0
    /// ]);
    /// let ut = M3x3UpperTrianglef32::from_matrix(m);
    /// assert_eq!(ut, M3x3UpperTrianglef32::new([2.0, 3.0,  7.0,
    ///                                                5.0, 11.0,
    ///                                                     29.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_matrix(m: Matrix3x3<T>) -> Self
    where
        T: Copy,
    {
        Self { a: [
            m[Matrix3x3::<T >::M11], m[Matrix3x3::<T >::M12], m[Matrix3x3::<T >::M13],
                                     m[Matrix3x3::<T >::M22], m[Matrix3x3::<T >::M23],
                                                              m[Matrix3x3::<T >::M33],
        ]}
    }

    /// Tries to create an `M3x3UpperTriangle` from a symmetric Matrix3x3.
    /// Returns `None` if the input matrix is not symmetric within floating-point tolerance `epsilon`.
    /// ```
    /// # use vqm::{M3x3UpperTrianglef32, Matrix3x3f32};
    /// let m = Matrix3x3f32::new([
    ///     2.0,  3.0,  7.0,
    ///     3.0,  5.0, 11.0,
    ///     7.0, 11.0, 29.0
    /// ]);
    /// let option = M3x3UpperTrianglef32::try_from_matrix(m, 1e-6);
    /// assert!(option.is_some());
    /// if let Some(ut) = option {
    ///     assert_eq!(ut, M3x3UpperTrianglef32::new([2.0, 3.0,  7.0,
    ///                                                    5.0, 11.0,
    ///                                                         29.0]));
    /// }
    /// ```
    #[inline]
    pub fn try_from_matrix(m: Matrix3x3<T>, epsilon: T) -> Option<Self>
    where
        T: Copy + Signed + PartialOrd,
    {
        if !m.is_symmetric(epsilon) {
            return None;
        }
        Some(Self::from_matrix(m))
    }

    /// Tries to create an `M3x3UpperTriangle` from the inverse of a symmetric Matrix3x3.
    /// Returns `None` if the input matrix is not symmetric, or if it is ill-conditioned,
    /// ie its reciprocal 1-norm condition number is less than `min_reciprocal_condition`.
    /// Assumes the matrix is symmetric.
    /// ```
    /// # use vqm::{M3x3UpperTrianglef32, Matrix3x3f32};
    /// let m = Matrix3x3f32::new([
    ///     2.0,  3.0,  7.0,
    ///     3.0,  5.0, 11.0,
    ///     7.0, 11.0, 29.0
    /// ]);
    /// let option = M3x3UpperTrianglef32::try_inverse_from_matrix(m, 1e-6);
    /// assert!(option.is_some());
    /// if let Some(ut) = option {
    ///     assert_eq!(ut, M3x3UpperTrianglef32::new([6.0, -2.5,  -0.5,
    ///                                                     2.25, -0.25,
    ///                                                            0.25]));
    ///     let n = Matrix3x3f32::from(ut);
    ///     let i = m * n;
    ///     assert_eq!(i, Matrix3x3f32::identity());
    /// }
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn try_inverse_from_matrix(m: Matrix3x3<T>, min_reciprocal_condition: T) -> Option<Self>
    where
        T: Copy + FloatCore + Signed + PartialOrd,
    {
        Self { a: [
            m[Matrix3x3::<T >::M11], m[Matrix3x3::<T >::M12], m[Matrix3x3::<T >::M13],
                                     m[Matrix3x3::<T >::M22], m[Matrix3x3::<T >::M23],
                                                              m[Matrix3x3::<T >::M33],
        ] }
        .try_inverse(min_reciprocal_condition)
    }

    /// Return the adjugate, determinant, and reciprocal of the 1-norm conditioning number of this matrix.
    /// The adjugate is the transpose of the cofactor matrix, ie the inverse but without dividing by the determinant of the matrix.
    ///
    /// Reciprocal  Interpretation
    /// Condition
    ///   1.0       Perfectly conditioned
    ///   0.1       Well-conditioned
    ///   0.001     Increasingly ill-conditioned
    ///   0.000001  Very poorly conditioned
    ///   0.0       Singular
    /// ```
    /// # use vqm::M3x3UpperTrianglef32;
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let (adjugate, determinant, reciprocal_condition) = ut.adjugate();
    /// assert_eq!(determinant, 4.0);
    /// assert!((reciprocal_condition - 0.0023640662).abs() < 1e-6);
    /// assert_eq!(adjugate, M3x3UpperTrianglef32::new([24.0, -10.0, -2.0,
    ///                                                         9.0, -1.0,
    ///                                                               1.0]));
    /// ```
    #[inline]
    pub fn adjugate(self) -> (Self, T, T)
    where
        T: Copy + FloatCore,
    {
        // Don't need to check if any of the matrix elements are infinity or NaN,
        // since this is picked up by the determinant calculation.

        // Calculate the cofactors needed for the determinant.
        let c11 = self.a[Self::M22] * self.a[Self::M33] - self.a[Self::M23] * self.a[Self::M23];
        let c12 = self.a[Self::M13] * self.a[Self::M23] - self.a[Self::M12] * self.a[Self::M33];
        let c13 = self.a[Self::M12] * self.a[Self::M23] - self.a[Self::M13] * self.a[Self::M22];

        // Calculate determinant using the first row cofactors
        let determinant = self.a[Self::M11] * c11 + self.a[Self::M12] * c12 + self.a[Self::M13] * c13;

        // Calculate remaining cofactors
        let c22 = self.a[Self::M11] * self.a[Self::M33] - self.a[Self::M13] * self.a[Self::M13];
        let c23 = self.a[Self::M13] * self.a[Self::M12] - self.a[Self::M11] * self.a[Self::M23];
        let c33 = self.a[Self::M11] * self.a[Self::M22] - self.a[Self::M12] * self.a[Self::M12];
        let a = [c11, c12, c13, c22, c23, c33];

        // 1-norm of the original matrix.
        let norm = (self.a[Self::M11].abs() + self.a[Self::M12].abs() + self.a[Self::M13].abs())
            .max(self.a[Self::M12].abs() + self.a[Self::M22].abs() + self.a[Self::M23].abs())
            .max(self.a[Self::M13].abs() + self.a[Self::M23].abs() + self.a[Self::M33].abs());
        // 1-norm of the adjugate.
        let norm_adjugate = (a[Self::M11].abs() + a[Self::M12].abs() + a[Self::M13].abs())
            .max(a[Self::M12].abs() + a[Self::M22].abs() + a[Self::M23].abs())
            .max(a[Self::M13].abs() + a[Self::M23].abs() + a[Self::M33].abs());

        // Reciprocal 1-norm condition number.
        let reciprocal_condition = {
            let r = determinant.abs() / norm / norm_adjugate;
            // r.is_finite also handles any infinities, NaNs or zeros in the value of norm or norm_adjugate.
            if r.is_finite() { r.min(T::one()) } else { T::zero() }
        };

        (Self { a }, determinant, reciprocal_condition)
    }

    /// Try and invert Upper Triangular Array.
    /// Returns `None` if the underlying matrix is ill-conditioned,
    /// ie its reciprocal 1-norm condition number is less than `min_reciprocal_condition`.
    /// ```
    /// # use vqm::M3x3UpperTrianglef32;
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let option = ut.try_inverse(1e-4);
    /// assert!(option.is_some());
    /// if let Some(ut) = option {
    ///     assert_eq!(ut, M3x3UpperTrianglef32::new([6.0, -2.5,  -0.5,
    ///                                                     2.25, -0.25,
    ///                                                            0.25]));
    /// }
    /// ```
    #[inline]
    pub fn try_inverse(self, min_reciprocal_condition: T) -> Option<Self>
    where
        T: Copy + FloatCore,
    {
        let (adjugate, determinant, reciprocal_condition) = self.adjugate();
        // Guard against ill-conditioned or singular matrix.
        if reciprocal_condition <= min_reciprocal_condition {
            return None;
        }
        let determinant_reciprocal = T::one() / determinant;
        Some(adjugate * determinant_reciprocal)
    }

    /// Return the sum of all the elements on the diagonal.
    /// ```
    /// # use vqm::M3x3UpperTrianglef32;
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let n = ut.trace();
    ///
    /// assert_eq!(n, 36.0);
    /// ```
    #[inline]
    pub fn trace(self) -> T
    where
        T: Copy + Add<Output = T>,
    {
        self.a[0] + self.a[3] + self.a[5]
    }

    /// Return the sum of the squares of all the elements on the diagonal.
    /// ```
    /// # use vqm::M3x3UpperTrianglef32;
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let n = ut.trace_sum_squares();
    ///
    /// assert_eq!(n, 870.0);
    /// ```
    #[inline]
    pub fn trace_sum_squares(self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        self.a[0] * self.a[0] + self.a[3] * self.a[3] + self.a[5] * self.a[5]
    }

    /// Return norm squared of the upper triangle, that is the sum of the squares of all the elements.
    /// ```
    /// # use vqm::M3x3UpperTrianglef32;
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let n = ut.norm_squared();
    ///
    /// assert_eq!(n, 1049.0);
    /// ```
    #[inline]
    pub fn norm_squared(self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        self.a[0] * self.a[0]
            + self.a[1] * self.a[1]
            + self.a[2] * self.a[2]
            + self.a[3] * self.a[3]
            + self.a[4] * self.a[4]
            + self.a[5] * self.a[5]
    }

    /// Return norm of the upper triangle, that is the square root of the sum of the squares of all the elements.
    /// ```
    /// # use vqm::M3x3UpperTrianglef32;
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let n = ut.norm();
    ///
    /// assert!((n - 32.38827).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn norm(self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T> + MathMethods,
    {
        self.norm_squared().sqrt()
    }

    /// Return norm squared of the matrix represented by this upper triangle.
    /// ```
    /// # use vqm::{M3x3UpperTrianglef32, Matrix3x3f32};
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let n = ut.matrix_norm_squared();
    ///
    /// assert_eq!(n, 1228.0);
    /// let m = Matrix3x3f32::from(ut);
    /// assert_eq!(n, m.norm_squared());
    /// ```
    #[inline]
    pub fn matrix_norm_squared(self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        self.norm_squared() + self.a[1] * self.a[1] + self.a[2] * self.a[2] + self.a[4] * self.a[4]
    }

    /// Return norm of the matrix represented by this upper triangle.
    /// ```
    /// # use vqm::M3x3UpperTrianglef32;
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let n = ut.matrix_norm();
    ///
    /// assert!((n - 35.042830936).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn matrix_norm(self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T> + MathMethods,
    {
        self.matrix_norm_squared().sqrt()
    }
}

// **** Scalar ****

impl<T> Mul<T> for M3x3UpperTriangle<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    /// Post-multiply upper triangle by a scalar.
    /// ```
    /// # use vqm::M3x3UpperTrianglef32;
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let r = ut * 2.0;
    /// assert_eq!(r, M3x3UpperTrianglef32::new([4.0, 6.0, 14.0,
    ///                                              10.0, 22.0,
    ///                                                    58.0]));
    /// ```
    #[inline]
    fn mul(self, k: T) -> Self {
        Self { a: [self.a[0] * k, self.a[1] * k, self.a[2] * k, self.a[3] * k, self.a[4] * k, self.a[5] * k] }
    }
}

// **** VECTOR ****

// **** Pre-multiply ****

impl<T> Mul<M3x3UpperTriangle<T>> for Vector3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector3<T>;
    /// Pre-multiplies an upper triangle by a 3D vector.
    /// Calculates: Result = v * ut.
    /// ```
    /// # use vqm::{M3x3UpperTrianglef32, Vector3f32};
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let v = Vector3f32::new(13.0, 17.0, 19.0);
    /// let r = v * ut;
    /// assert_eq!(r, Vector3f32::new(210.0, 333.0, 829.0));
    /// ```
    #[inline]
    fn mul(self, ut: M3x3UpperTriangle<T>) -> Vector3<T> {
        type Ut<T> = M3x3UpperTriangle<T>;
        Vector3 {
            x: self.x * ut.a[Ut::<T>::M11] + self.y * ut.a[Ut::<T>::M12] + self.z * ut.a[Ut::<T>::M13],
            y: self.x * ut.a[Ut::<T>::M12] + self.y * ut.a[Ut::<T>::M22] + self.z * ut.a[Ut::<T>::M23],
            z: self.x * ut.a[Ut::<T>::M13] + self.y * ut.a[Ut::<T>::M23] + self.z * ut.a[Ut::<T>::M33],
        }
    }
}

// **** Post-multiply ****

impl<T> Mul<Vector3<T>> for M3x3UpperTriangle<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector3<T>;
    /// Post-multiplies this upper triangle by a 3D vector.
    ///
    /// Calculates: Result = ut * v.
    /// ```
    /// # use vqm::{M3x3UpperTrianglef32, Vector3f32, Matrix3x3f32};
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let v = Vector3f32::new(13.0, 17.0, 19.0);
    /// let r = ut * v;
    /// assert_eq!(r, Vector3f32::new(210.0, 333.0, 829.0));
    /// let m = Matrix3x3f32::from(ut);
    /// let r2 = m * v;
    /// assert_eq!(r, r2);
    /// ```
    #[inline]
    fn mul(self, v: Vector3<T>) -> Vector3<T> {
        Vector3 {
            x: self.a[Self::M11] * v.x + self.a[Self::M12] * v.y + self.a[Self::M13] * v.z,
            y: self.a[Self::M12] * v.x + self.a[Self::M22] * v.y + self.a[Self::M23] * v.z,
            z: self.a[Self::M13] * v.x + self.a[Self::M23] * v.y + self.a[Self::M33] * v.z,
        }
    }
}

// **** MATRIX ****

// **** Pre-multiply ****

impl<T> Mul<M3x3UpperTriangle<T>> for Matrix3x3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix3x3<T>;

    /// Pre-multiplies this upper triangle by a 3x3 matrix.
    /// Calculates: Result = m * ut.
    /// ```
    /// # use vqm::{M3x3UpperTrianglef32, Matrix3x3f32};
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let m = Matrix3x3f32::new([ 13.0, 17.0, 59.0,
    ///                             19.0, 37.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let r = m * ut;
    /// assert_eq!(r, Matrix3x3f32::new([  490.0,  773.0, 1989.0,
    ///                                    478.0,  759.0, 1903.0,
    ///                                    426.0,  675.0, 1691.0]));
    /// let n = Matrix3x3f32::from(ut);
    /// let r2 = m * n;
    /// assert_eq!(r, r2);
    /// ```
    #[inline]
    fn mul(self, ut: M3x3UpperTriangle<T>) -> Matrix3x3<T> {
        let a11 = ut.a[0];
        let a12 = ut.a[1];
        let a13 = ut.a[2];
        let a22 = ut.a[3];
        let a23 = ut.a[4];
        let a33 = ut.a[5];

        // Cache rows of M into registers to prevent strided latency stalls
        // Row 1
        let m11 = self[Matrix3x3::<T>::M11];
        let m12 = self[Matrix3x3::<T>::M12];
        let m13 = self[Matrix3x3::<T>::M13];
        // Row 2
        let m21 = self[Matrix3x3::<T>::M21];
        let m22 = self[Matrix3x3::<T>::M22];
        let m23 = self[Matrix3x3::<T>::M23];
        // Row 3
        let m31 = self[Matrix3x3::<T>::M31];
        let m32 = self[Matrix3x3::<T>::M32];
        let m33 = self[Matrix3x3::<T>::M33];

        // Column 1 of Result
        let r11 = m11 * a11 + m12 * a12 + m13 * a13;
        let r21 = m21 * a11 + m22 * a12 + m23 * a13;
        let r31 = m31 * a11 + m32 * a12 + m33 * a13;

        // Column 2 of Result
        let r12 = m11 * a12 + m12 * a22 + m13 * a23;
        let r22 = m21 * a12 + m22 * a22 + m23 * a23;
        let r32 = m31 * a12 + m32 * a22 + m33 * a23;

        // Column 3 of Result
        let r13 = m11 * a13 + m12 * a23 + m13 * a33;
        let r23 = m21 * a13 + m22 * a23 + m23 * a33;
        let r33 = m31 * a13 + m32 * a23 + m33 * a33;

        Matrix3x3::from([r11, r21, r31, r12, r22, r32, r13, r23, r33])
    }
}

// **** Post-multiply ****

impl<T> Mul<Matrix3x3<T>> for M3x3UpperTriangle<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix3x3<T>;

    /// Post-multiplies this upper triangle by a 3x3 matrix.
    ///
    /// Calculates: Result = ut * m.
    /// ```
    /// # use vqm::{M3x3UpperTrianglef32, Matrix3x3f32};
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let m = Matrix3x3f32::new([ 13.0, 17.0, 59.0,
    ///                             19.0, 37.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let r = ut * m;
    /// assert_eq!(r, Matrix3x3f32::new([  244.0,  362.0,  546.0,
    ///                                    387.0,  577.0,  863.0,
    ///                                    967.0, 1425.0, 2119.0]));
    /// let n = Matrix3x3f32::from(ut);
    /// let r2 = n * m;
    /// assert_eq!(r, r2);
    /// ```
    #[inline]
    fn mul(self, m: Matrix3x3<T>) -> Matrix3x3<T> {
        let a11 = self.a[Self::M11];
        let a12 = self.a[Self::M12];
        let a13 = self.a[Self::M13];
        let a22 = self.a[Self::M22];
        let a23 = self.a[Self::M23];
        let a33 = self.a[Self::M33];

        // Column 1
        let m11 = m[Matrix3x3::<T>::M11];
        let m21 = m[Matrix3x3::<T>::M21];
        let m31 = m[Matrix3x3::<T>::M31];
        let r11 = a11 * m11 + a12 * m21 + a13 * m31;
        let r21 = a12 * m11 + a22 * m21 + a23 * m31;
        let r31 = a13 * m11 + a23 * m21 + a33 * m31;

        // Column 2
        let m12 = m[Matrix3x3::<T>::M12];
        let m22 = m[Matrix3x3::<T>::M22];
        let m32 = m[Matrix3x3::<T>::M32];
        let r12 = a11 * m12 + a12 * m22 + a13 * m32;
        let r22 = a12 * m12 + a22 * m22 + a23 * m32;
        let r32 = a13 * m12 + a23 * m22 + a33 * m32;

        // Column 3
        let m13 = m[Matrix3x3::<T>::M13];
        let m23 = m[Matrix3x3::<T>::M23];
        let m33 = m[Matrix3x3::<T>::M33];
        let r13 = a11 * m13 + a12 * m23 + a13 * m33;
        let r23 = a12 * m13 + a22 * m23 + a23 * m33;
        let r33 = a13 * m13 + a23 * m23 + a33 * m33;

        Matrix3x3::from([r11, r21, r31, r12, r22, r32, r13, r23, r33])
    }
}

// **** Index ****

impl<T> Index<usize> for M3x3UpperTriangle<T> {
    type Output = T;

    /// Access upper triangle element by index.
    /// ```
    /// # use vqm::M3x3UpperTrianglef32;
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    ///
    /// assert_eq!(ut[0], 2.0);
    /// assert_eq!(ut[1], 3.0);
    /// assert_eq!(ut[2], 7.0);
    /// assert_eq!(ut[3], 5.0);
    /// assert_eq!(ut[4], 11.0);
    /// assert_eq!(ut[5], 29.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

// **** MATRIX ****

impl<T> From<M3x3UpperTriangle<T>> for Matrix3x3<T>
where
    T: Copy,
{
    /// Matrix3x3 from `M3x3UpperTriangle`.
    /// ```
    /// # use vqm::{M3x3UpperTrianglef32, Matrix3x3f32};
    /// let ut = M3x3UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let m = Matrix3x3f32::from(ut);
    /// assert_eq!(m, Matrix3x3f32::new([
    ///     2.0,  3.0,  7.0,
    ///     3.0,  5.0, 11.0,
    ///     7.0, 11.0, 29.0
    /// ]));
    #[rustfmt::skip]
    #[inline]
    fn from(ut: M3x3UpperTriangle<T>) -> Self {
        Self { a: [
            ut[0], ut[1], ut[2],
            ut[1], ut[3], ut[4],
            ut[2], ut[4], ut[5],
        ] }
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy,
{
    /// Returns the upper triangle (including diagonal) of a matrix as a `M3x3UpperTriangle`.
    /// The lower triangle of the matrix is ignored.
    /// ```
    /// # use vqm::{M3x3UpperTrianglef32, Matrix3x3f32};
    /// let M = Matrix3x3f32::new([
    ///     2.0,  3.0,  7.0,
    ///     4.0,  5.0, 11.0,
    ///     6.0,  8.0, 29.0
    /// ]);
    /// let ut = M.upper_triangle();
    /// assert_eq!(ut, M3x3UpperTrianglef32::new([2.0, 3.0,  7.0,
    ///                                                 5.0, 11.0,
    ///                                                      29.0]));
    /// ```
    #[rustfmt::skip]
    #[must_use]
    #[inline]
    pub fn upper_triangle(self) -> M3x3UpperTriangle<T> {
        M3x3UpperTriangle::<T> { a: [
            self.a[Self::M11], self.a[Self::M12], self.a[Self::M13],
                               self.a[Self::M22], self.a[Self::M23],
                                                  self.a[Self::M33],
        ]}
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<M3x3UpperTriangle<f32>>();
    }
}
