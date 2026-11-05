use core::ops::{Add, Div, Index, Mul, Sub};
use num_traits::{ConstOne, Signed};

use crate::{MathConstants, Matrix4x4, Vector4};

/// Upper-Triangular Array representing a 4x4 matrix of `f32` values<br>
pub type M4x4UpperTrianglef32 = M4x4UpperTriangle<f32>;
/// Upper-Triangular Array representing a 4x4 matrix of `f64` values<br><br>
pub type M4x4UpperTrianglef64 = M4x4UpperTriangle<f64>;

/// A 4x4 symmetric matrix stored as a flat 6-element array (Upper-Triangular layout):
/// `[m11, m12, m13, m14,
///        m22, m23, m24,
///             m33, m34,
///                  m44]`.
/// Also known as Voigt Notation or Mandel Notation.
/// The overwhelming historical convention is to use upper-triangular rather than lower-triangular layout.
/// This convention is used by libraries such as BLAS, LAPACK, and Intel MKL.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct M4x4UpperTriangle<T> {
    pub(crate) a: [T; UT_4X4_SIZE],
}
const UT_4X4_SIZE: usize = 10;

impl<T> M4x4UpperTriangle<T> {
    /// Size of upper triangular array.
    pub const SIZE: usize = UT_4X4_SIZE;

    // Row 1
    const M11: usize = 0;
    const M12: usize = 1;
    const M13: usize = 2;
    const M14: usize = 3;
    // Row 2
    const M22: usize = 4;
    const M23: usize = 5;
    const M24: usize = 6;
    // Row 3
    const M33: usize = 7;
    const M34: usize = 8;
    // Row 4
    const M44: usize = 9;
}

// **** New ****

impl<T> M4x4UpperTriangle<T>
where
    T: Copy,
{
    /// Create an upper-triangle.
    /// ```
    /// # use vqm::M4x4UpperTrianglef32;
    /// let ut = M4x4UpperTrianglef32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///            5.0, 11.0, 19.0,
    ///                 29.0, 41.0,
    ///                       97.0,
    /// ]);
    /// ```
    #[inline]
    pub const fn new(a: [T; UT_4X4_SIZE]) -> Self {
        Self { a }
    }
}

// **** Other constructors ****

impl<T> M4x4UpperTriangle<T> {
    /// Create an `M4x4UpperTriangle` from a Matrix4x4.
    /// Does not check if the matrix is symmetric.
    /// ```
    /// # use vqm::{M4x4UpperTrianglef32, Matrix4x4f32};
    /// let m = Matrix4x4f32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///      3.0,  5.0, 11.0, 19.0,
    ///      7.0, 11.0, 29.0, 41.0,
    ///     13.0, 19.0, 41.0, 97.0,
    /// ]);
    /// let ut = M4x4UpperTrianglef32::from_matrix(m);
    /// assert_eq!(ut, M4x4UpperTrianglef32::new([2.0, 3.0,  7.0, 13.0,
    ///                                                 5.0, 11.0, 19.0,
    ///                                                      29.0, 41.0,
    ///                                                            97.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_matrix(m: Matrix4x4<T>) -> Self
    where
        T: Copy,
    {
        Self { a: [
            m[Matrix4x4::<T >::M11], m[Matrix4x4::<T >::M12], m[Matrix4x4::<T >::M13], m[Matrix4x4::<T >::M14],
                                     m[Matrix4x4::<T >::M22], m[Matrix4x4::<T >::M23], m[Matrix4x4::<T >::M24],
                                                              m[Matrix4x4::<T >::M33], m[Matrix4x4::<T >::M34],
                                                                                       m[Matrix4x4::<T >::M44]
        ] }
    }

    /// Tries to create an `M4x4UpperTriangle` from a symmetric Matrix4x4.
    /// Returns `None` if the input matrix is not symmetric within floating-point tolerance `epsilon`.
    /// ```
    /// # use vqm::{M4x4UpperTrianglef32, Matrix4x4f32};
    /// let m = Matrix4x4f32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///      3.0,  5.0, 11.0, 19.0,
    ///      7.0, 11.0, 29.0, 41.0,
    ///     13.0, 19.0, 41.0, 97.0,
    /// ]);
    /// let option = M4x4UpperTrianglef32::try_from_matrix(m, 1e-6);
    /// assert!(option.is_some());
    /// if let Some(ut) = option {
    ///     assert_eq!(ut, M4x4UpperTrianglef32::new([2.0, 3.0,  7.0, 13.0,
    ///                                                    5.0, 11.0, 19.0,
    ///                                                         29.0, 41.0,
    ///                                                               97.0]));
    /// }
    /// ```
    #[inline]
    pub fn try_from_matrix(m: Matrix4x4<T>, epsilon: T) -> Option<Self>
    where
        T: Copy + Signed + PartialOrd,
    {
        if !m.is_symmetric(epsilon) {
            return None;
        }
        Some(Self::from_matrix(m))
    }

    /// Tries to create an `M4x4UpperTriangle` from the inverse of a symmetric Matrix4x4.
    /// Returns `None` if the input matrix is not symmetric, or not invertible because the determinant is zero within floating-point tolerance `epsilon`.
    /// ```
    /// # use vqm::{M4x4UpperTrianglef32, Matrix4x4f32};
    /// let m = Matrix4x4f32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///      3.0,  5.0, 11.0, 19.0,
    ///      7.0, 11.0, 29.0, 41.0,
    ///     13.0, 19.0, 41.0, 97.0,
    /// ]);
    /// let option = M4x4UpperTrianglef32::try_inverse_from_matrix(m, 1e-6);
    /// assert!(option.is_some());
    /// if let Some(ut) = option {
    ///     assert_eq!(ut, M4x4UpperTrianglef32::new([18.5, -2.5,  -1.75, -1.25,
    ///                                                      2.25, -0.25,  0.0,
    ///                                                             0.375, 0.125,
    ///                                                                    0.125]));
    ///     let n = Matrix4x4f32::from(ut);
    ///     let m_inv = m.inverse();
    ///     assert_eq!(n, m_inv);
    ///     let i = m * n;
    ///     assert_eq!(i, Matrix4x4f32::identity());
    /// }
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn try_inverse_from_matrix(m: Matrix4x4<T>, epsilon: T) -> Option<Self>
    where
        T: Copy + ConstOne + Signed + Sub<Output = T> + PartialOrd + MathConstants,
    {
        if !m.is_symmetric(epsilon) {
            return None;
        }
        Self { a: [
            m[Matrix4x4::<T >::M11], m[Matrix4x4::<T >::M12], m[Matrix4x4::<T >::M13], m[Matrix4x4::<T >::M14],
                                     m[Matrix4x4::<T >::M22], m[Matrix4x4::<T >::M23], m[Matrix4x4::<T >::M24],
                                                              m[Matrix4x4::<T >::M33], m[Matrix4x4::<T >::M34],
                                                                                       m[Matrix4x4::<T >::M44]
        ] }
        .try_inverse(epsilon)
    }

    /// Try and invert Upper Triangular Array.
    /// Returns `None` if the matrix determinant is zero within a tolerance of `epsilon`.
    /// ```
    /// # use vqm::M4x4UpperTrianglef32;
    /// let ut = M4x4UpperTrianglef32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///            5.0, 11.0, 19.0,
    ///                 29.0, 41.0,
    ///                       97.0,
    /// ]);
    /// let option = ut.try_inverse(1e-6);
    /// if let Some(ut_inv) = option {
    ///    let expected = M4x4UpperTrianglef32::new([18.5, -2.5,  -1.75, -1.25,
    ///                                                     2.25, -0.25,  0.0,
    ///                                                            0.375, 0.125,
    ///                                                                   0.125]);
    ///    assert_eq!(ut_inv, expected);
    /// }
    /// ```
    #[inline]
    pub fn try_inverse(&self, epsilon: T) -> Option<Self>
    where
        T: Copy
            + ConstOne
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Signed
            + PartialOrd
            + MathConstants,
    {
        let m11 = self.a[0];
        let m12 = self.a[1];
        let m13 = self.a[2];
        let m14 = self.a[3];
        let m22 = self.a[4];
        let m23 = self.a[5];
        let m24 = self.a[6];
        let m33 = self.a[7];
        let m34 = self.a[8];
        let m44 = self.a[9];

        // Cache recurring products to save multiplication cycles and optimize register re-use
        let m12_sq = m12 * m12;
        let m13_sq = m13 * m13;
        let m14_sq = m14 * m14;
        let m23_sq = m23 * m23;
        let m24_sq = m24 * m24;
        let m34_sq = m34 * m34;

        // Calculate the 3x3 cofactors
        let c11 = m22 * m33 * m44 - m22 * m34_sq - m23_sq * m44 + T::TWO * m23 * m24 * m34 - m24_sq * m33;
        let c12 =
            -m12 * m33 * m44 + m12 * m34_sq + m13 * m23 * m44 - m13 * m24 * m34 - m14 * m23 * m34 + m14 * m24 * m33;
        let c13 =
            m12 * m23 * m44 - m12 * m24 * m34 - m13 * m22 * m44 + m13 * m24_sq + m14 * m22 * m34 - m14 * m23 * m24;
        let c14 =
            -m12 * m23 * m34 + m12 * m24 * m33 + m13 * m22 * m34 - m13 * m23 * m24 - m14 * m22 * m33 + m14 * m23_sq;

        let c22 = m11 * m33 * m44 - m11 * m34_sq - m13_sq * m44 + T::TWO * m13 * m14 * m34 - m14_sq * m33;
        let c23 =
            -m11 * m23 * m44 + m11 * m24 * m34 + m12 * m13 * m44 - m12 * m14 * m34 - m13 * m14 * m24 + m14_sq * m23;
        let c24 =
            m11 * m23 * m34 - m11 * m24 * m33 - m12 * m13 * m34 + m12 * m14 * m33 + m13_sq * m24 - m13 * m14 * m23;

        let c33 = m11 * m22 * m44 - m11 * m24_sq - m12_sq * m44 + T::TWO * m12 * m14 * m24 - m14_sq * m22;
        let c34 =
            -m11 * m22 * m34 + m11 * m23 * m24 + m12_sq * m34 - m12 * m13 * m24 - m12 * m14 * m23 + m13 * m14 * m22;
        let c44 = m11 * m22 * m33 - m11 * m23_sq - m12_sq * m33 + T::TWO * m12 * m13 * m23 - m13_sq * m22;

        // Calculate determinant via Laplace expansion of the first row
        let determinant = m11 * c11 + m12 * c12 + m13 * c13 + m14 * c14;

        // Numerical Stability Check
        let mut largest = m11.abs();
        for i in 1..Self::SIZE {
            if self.a[i].abs() > largest {
                largest = self.a[i].abs();
            }
        }
        let safety_threshold = largest * T::EPSILON * T::EIGHT + epsilon;

        if determinant.abs() < safety_threshold {
            return None;
        }

        let determinant_reciprocal = T::ONE / determinant;

        Some(Self {
            a: [
                c11 * determinant_reciprocal,
                c12 * determinant_reciprocal,
                c13 * determinant_reciprocal,
                c14 * determinant_reciprocal,
                c22 * determinant_reciprocal,
                c23 * determinant_reciprocal,
                c24 * determinant_reciprocal,
                c33 * determinant_reciprocal,
                c34 * determinant_reciprocal,
                c44 * determinant_reciprocal,
            ],
        })
    }
}

// **** VECTOR ****

// **** Pre-multiply ****

impl<T> Mul<M4x4UpperTriangle<T>> for Vector4<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector4<T>;
    /// Pre-multiplies this upper triangle by a 4D vector.
    /// Calculates: Result = v * ut.
    /// ```
    /// # use vqm::{M4x4UpperTrianglef32, Vector4f32, Matrix4x4f32};
    /// let ut = M4x4UpperTrianglef32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///            5.0, 11.0, 19.0,
    ///                 29.0, 41.0,
    ///                       97.0,
    /// ]);
    /// let v = Vector4f32::new(13.0, 17.0, 19.0, 23.0);
    /// let r = v * ut;
    /// assert_eq!(r, Vector4f32::new(509.0, 770.0, 1772.0, 3502.0));
    /// ```
    #[rustfmt::skip]
    #[inline]
    fn mul(self, ut: M4x4UpperTriangle<T>) -> Vector4<T> {
        type Ut<T> = M4x4UpperTriangle<T>;
        Vector4 {
            x: self.x * ut.a[Ut::<T>::M11] + self.y * ut.a[Ut::<T>::M12] + self.z * ut.a[Ut::<T>::M13] + self.t * ut.a[Ut::<T>::M14],
            y: self.x * ut.a[Ut::<T>::M12] + self.y * ut.a[Ut::<T>::M22] + self.z * ut.a[Ut::<T>::M23] + self.t * ut.a[Ut::<T>::M24],
            z: self.x * ut.a[Ut::<T>::M13] + self.y * ut.a[Ut::<T>::M23] + self.z * ut.a[Ut::<T>::M33] + self.t * ut.a[Ut::<T>::M34],
            t: self.x * ut.a[Ut::<T>::M14] + self.y * ut.a[Ut::<T>::M24] + self.z * ut.a[Ut::<T>::M34] + self.t * ut.a[Ut::<T>::M44],
        }
    }
}

// **** Post-multiply ****

impl<T> Mul<Vector4<T>> for M4x4UpperTriangle<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector4<T>;
    /// Post-multiplies this upper triangle by a 4D vector.
    ///
    /// Calculates: Result = ut * v.
    /// ```
    /// # use vqm::{M4x4UpperTrianglef32, Vector4f32, Matrix4x4f32};
    /// let ut = M4x4UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0, 13.0,
    ///          5.0, 11.0, 19.0,
    ///               29.0, 41.0,
    ///                     97.0
    /// ]);
    /// let v = Vector4f32::new(13.0, 17.0, 19.0, 23.0);
    /// let r = ut * v;
    /// assert_eq!(r, Vector4f32::new(509.0, 770.0, 1772.0, 3502.0));
    /// let m = Matrix4x4f32::from(ut);
    /// let r2 = m * v;
    /// assert_eq!(r, r2);
    /// ```
    #[inline]
    fn mul(self, v: Vector4<T>) -> Vector4<T> {
        Vector4 {
            x: self.a[Self::M11] * v.x + self.a[Self::M12] * v.y + self.a[Self::M13] * v.z + self.a[Self::M14] * v.t,
            y: self.a[Self::M12] * v.x + self.a[Self::M22] * v.y + self.a[Self::M23] * v.z + self.a[Self::M24] * v.t,
            z: self.a[Self::M13] * v.x + self.a[Self::M23] * v.y + self.a[Self::M33] * v.z + self.a[Self::M34] * v.t,
            t: self.a[Self::M14] * v.x + self.a[Self::M24] * v.y + self.a[Self::M34] * v.z + self.a[Self::M44] * v.t,
        }
    }
}

// **** MATRIX ****

// **** Pre-multiply ****

impl<T> Mul<M4x4UpperTriangle<T>> for Matrix4x4<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix4x4<T>;

    /// Pre-multiplies this upper triangle by a 4x4 matrix.
    /// Calculates: Result = m * ut.
    /// ```
    /// # use vqm::{M4x4UpperTrianglef32, Matrix4x4f32};
    /// let ut = M4x4UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0, 13.0,
    ///          5.0, 11.0, 19.0,
    ///               29.0, 41.0,
    ///                     97.0
    /// ]);
    /// let m = Matrix4x4f32::new([ 17.0, 37.0, 61.0, 101.0,
    ///                             23.0, 31.0, 59.0, 89.0,
    ///                             43.0, 47.0, 53.0, 83.0,
    ///                             67.0, 71.0, 73.0, 79.0]);
    /// let r = m * ut;
    /// assert_eq!(r, Matrix4x4f32::new([ 1885.0, 2826.0, 6436.0, 13222.0,
    ///                                   1709.0, 2564.0, 5862.0, 11940.0,
    ///                                   1677.0, 2524.0, 5758.0, 11676.0,
    ///                                   1885.0, 2860.0, 6606.0, 12876.0]));
    /// let n = Matrix4x4f32::from(ut);
    /// let r2 = m * n;
    /// assert_eq!(r, r2);
    /// ```
    #[inline]
    fn mul(self, ut: M4x4UpperTriangle<T>) -> Matrix4x4<T> {
        let a11 = ut.a[0];
        let a12 = ut.a[1];
        let a13 = ut.a[2];
        let a14 = ut.a[3];
        let a22 = ut.a[4];
        let a23 = ut.a[5];
        let a24 = ut.a[6];
        let a33 = ut.a[7];
        let a34 = ut.a[8];
        let a44 = ut.a[9];

        // Cache rows of M into registers to prevent strided latency stalls
        // Row 1
        let m11 = self[Matrix4x4::<T>::M11];
        let m12 = self[Matrix4x4::<T>::M12];
        let m13 = self[Matrix4x4::<T>::M13];
        let m14 = self[Matrix4x4::<T>::M14];
        // Row 2
        let m21 = self[Matrix4x4::<T>::M21];
        let m22 = self[Matrix4x4::<T>::M22];
        let m23 = self[Matrix4x4::<T>::M23];
        let m24 = self[Matrix4x4::<T>::M24];
        // Row 3
        let m31 = self[Matrix4x4::<T>::M31];
        let m32 = self[Matrix4x4::<T>::M32];
        let m33 = self[Matrix4x4::<T>::M33];
        let m34 = self[Matrix4x4::<T>::M34];
        // Row 4
        let m41 = self[Matrix4x4::<T>::M41];
        let m42 = self[Matrix4x4::<T>::M42];
        let m43 = self[Matrix4x4::<T>::M43];
        let m44 = self[Matrix4x4::<T>::M44];

        // Column 1 of Result
        let r11 = m11 * a11 + m12 * a12 + m13 * a13 + m14 * a14;
        let r21 = m21 * a11 + m22 * a12 + m23 * a13 + m24 * a14;
        let r31 = m31 * a11 + m32 * a12 + m33 * a13 + m34 * a14;
        let r41 = m41 * a11 + m42 * a12 + m43 * a13 + m44 * a14;

        // Column 2 of Result
        let r12 = m11 * a12 + m12 * a22 + m13 * a23 + m14 * a24;
        let r22 = m21 * a12 + m22 * a22 + m23 * a23 + m24 * a24;
        let r32 = m31 * a12 + m32 * a22 + m33 * a23 + m34 * a24;
        let r42 = m41 * a12 + m42 * a22 + m43 * a23 + m44 * a24;

        // Column 3 of Result
        let r13 = m11 * a13 + m12 * a23 + m13 * a33 + m14 * a34;
        let r23 = m21 * a13 + m22 * a23 + m23 * a33 + m24 * a34;
        let r33 = m31 * a13 + m32 * a23 + m33 * a33 + m34 * a34;
        let r43 = m41 * a13 + m42 * a23 + m43 * a33 + m44 * a34;

        // Column 4 of Result
        let r14 = m11 * a14 + m12 * a24 + m13 * a34 + m14 * a44;
        let r24 = m21 * a14 + m22 * a24 + m23 * a34 + m24 * a44;
        let r34 = m31 * a14 + m32 * a24 + m33 * a34 + m34 * a44;
        let r44 = m41 * a14 + m42 * a24 + m43 * a34 + m44 * a44;

        Matrix4x4::from([r11, r21, r31, r41, r12, r22, r32, r42, r13, r23, r33, r43, r14, r24, r34, r44])
    }
}

// **** Post-multiply ****

impl<T> Mul<Matrix4x4<T>> for M4x4UpperTriangle<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix4x4<T>;

    /// Post-multiplies this upper triangle by a 4x4 matrix.
    ///
    /// Calculates: Result = ut * m.
    /// ```
    /// # use vqm::{M4x4UpperTrianglef32, Matrix4x4f32};
    /// let ut = M4x4UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0, 13.0,
    ///          5.0, 11.0, 19.0,
    ///               29.0, 41.0,
    ///                     97.0
    /// ]);
    /// let m = Matrix4x4f32::new([ 17.0, 37.0, 61.0, 101.0,
    ///                             23.0, 31.0, 59.0, 89.0,
    ///                             43.0, 47.0, 53.0, 83.0,
    ///                             67.0, 71.0, 73.0, 79.0]);
    /// let r = ut * m;
    /// assert_eq!(r, Matrix4x4f32::new([ 1275.0, 1419.0,  1619.0,  2077.0,
    ///                                   1912.0, 2132.0,  2448.0,  3162.0,
    ///                                   4366.0, 4874.0,  5606.0,  7332.0,
    ///                                   8920.0, 9884.0, 11168.0, 14070.0]));
    /// let n = Matrix4x4f32::from(ut);
    /// let r2 = n * m;
    /// assert_eq!(r, r2);
    /// ```
    #[allow(unused)]
    #[inline]
    fn mul(self, m: Matrix4x4<T>) -> Matrix4x4<T> {
        let a11 = self.a[Self::M11];
        let a12 = self.a[Self::M12];
        let a13 = self.a[Self::M13];
        let a14 = self.a[Self::M14];
        let a22 = self.a[Self::M22];
        let a23 = self.a[Self::M23];
        let a24 = self.a[Self::M24];
        let a33 = self.a[Self::M33];
        let a34 = self.a[Self::M34];
        let a44 = self.a[Self::M44];

        // Column 1
        let m11 = m[Matrix4x4::<T>::M11];
        let m21 = m[Matrix4x4::<T>::M21];
        let m31 = m[Matrix4x4::<T>::M31];
        let m41 = m[Matrix4x4::<T>::M41];
        let r11 = a11 * m11 + a12 * m21 + a13 * m31 + a14 * m41;
        let r21 = a12 * m11 + a22 * m21 + a23 * m31 + a24 * m41;
        let r31 = a13 * m11 + a23 * m21 + a33 * m31 + a34 * m41;
        let r41 = a14 * m11 + a24 * m21 + a34 * m31 + a44 * m41;

        // Column 2
        let m12 = m[Matrix4x4::<T>::M12];
        let m22 = m[Matrix4x4::<T>::M22];
        let m32 = m[Matrix4x4::<T>::M32];
        let m42 = m[Matrix4x4::<T>::M42];
        let r12 = a11 * m12 + a12 * m22 + a13 * m32 + a14 * m42;
        let r22 = a12 * m12 + a22 * m22 + a23 * m32 + a24 * m42;
        let r32 = a13 * m12 + a23 * m22 + a33 * m32 + a34 * m42;
        let r42 = a14 * m12 + a24 * m22 + a34 * m32 + a44 * m42;

        // Column 3
        let m13 = m[Matrix4x4::<T>::M13];
        let m23 = m[Matrix4x4::<T>::M23];
        let m33 = m[Matrix4x4::<T>::M33];
        let m43 = m[Matrix4x4::<T>::M43];
        let r13 = a11 * m13 + a12 * m23 + a13 * m33 + a14 * m43;
        let r23 = a12 * m13 + a22 * m23 + a23 * m33 + a24 * m43;
        let r33 = a13 * m13 + a23 * m23 + a33 * m33 + a34 * m43;
        let r43 = a14 * m13 + a24 * m23 + a34 * m33 + a44 * m43;

        // Column 4
        let m14 = m[Matrix4x4::<T>::M14];
        let m24 = m[Matrix4x4::<T>::M24];
        let m34 = m[Matrix4x4::<T>::M34];
        let m44 = m[Matrix4x4::<T>::M44];
        let r14 = a11 * m14 + a12 * m24 + a13 * m34 + a14 * m44;
        let r24 = a12 * m14 + a22 * m24 + a23 * m34 + a24 * m44;
        let r34 = a13 * m14 + a23 * m24 + a33 * m34 + a34 * m44;
        let r44 = a14 * m14 + a24 * m24 + a34 * m34 + a44 * m44;

        Matrix4x4::from([r11, r21, r31, r41, r12, r22, r32, r42, r13, r23, r33, r43, r14, r24, r34, r44])
    }
}

// **** Index ****

impl<T> Index<usize> for M4x4UpperTriangle<T> {
    type Output = T;

    /// Access upper triangle element by index.
    /// ```
    /// # use vqm::M4x4UpperTrianglef32;
    /// let ut = M4x4UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0, 13.0,
    ///          5.0, 11.0, 19.0,
    ///               29.0, 41.0,
    ///                     97.0
    /// ]);
    ///
    /// assert_eq!(ut[0], 2.0);
    /// assert_eq!(ut[1], 3.0);
    /// assert_eq!(ut[2], 7.0);
    /// assert_eq!(ut[3], 13.0);
    /// assert_eq!(ut[4], 5.0);
    /// assert_eq!(ut[5], 11.0);
    /// assert_eq!(ut[6], 19.0);
    /// assert_eq!(ut[7], 29.0);
    /// assert_eq!(ut[8], 41.0);
    /// assert_eq!(ut[9], 97.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

impl<T> From<M4x4UpperTriangle<T>> for Matrix4x4<T>
where
    T: Copy,
{
    /// Matrix4x4 from `M4x4UpperTriangle`.
    /// ```
    /// # use vqm::{M4x4UpperTrianglef32, Matrix4x4f32};
    /// let ut = M4x4UpperTrianglef32::new([
    ///     2.0, 3.0,  7.0, 13.0,
    ///          5.0, 11.0, 19.0,
    ///               29.0, 41.0,
    ///                     97.0
    /// ]);
    /// let m = Matrix4x4f32::from(ut);
    /// assert_eq!(m, Matrix4x4f32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///      3.0,  5.0, 11.0, 19.0,
    ///      7.0, 11.0, 29.0, 41.0,
    ///     13.0, 19.0, 41.0, 97.0
    /// ]));
    #[rustfmt::skip]
    #[inline]
    fn from(ut: M4x4UpperTriangle<T>) -> Self {
        Self { a: [
            ut[0], ut[1], ut[2], ut[3],
            ut[1], ut[4], ut[5], ut[6],
            ut[2], ut[5], ut[7], ut[8],
            ut[3], ut[6], ut[8], ut[9]
        ] }
    }
}

// **** MATRIX ****

impl<T> Matrix4x4<T>
where
    T: Copy,
{
    /// Returns the upper triangle (including diagonal) of a matrix as a `M4x4UpperTriangle`.
    /// The lower triangle of the matrix is ignored.
    /// ```
    /// # use vqm::{M4x4UpperTrianglef32, Matrix4x4f32};
    /// let M = Matrix4x4f32::new([
    ///      2.0,  3.0,  7.0, 13.0,
    ///      3.0,  5.0, 11.0, 19.0,
    ///      7.0, 11.0, 29.0, 41.0,
    ///     13.0, 19.0, 41.0, 97.0,
    /// ]);
    /// let ut = M.upper_triangle();
    ///     assert_eq!(ut, M4x4UpperTrianglef32::new([2.0, 3.0,  7.0, 13.0,
    ///                                                    5.0, 11.0, 19.0,
    ///                                                         29.0, 41.0,
    ///                                                               97.0]));
    /// ```
    #[must_use]
    #[inline]
    pub fn upper_triangle(self) -> M4x4UpperTriangle<T> {
        M4x4UpperTriangle::<T> {
            a: [
                self.a[Self::M11],
                self.a[Self::M12],
                self.a[Self::M13],
                self.a[Self::M14],
                self.a[Self::M22],
                self.a[Self::M23],
                self.a[Self::M24],
                self.a[Self::M33],
                self.a[Self::M34],
                self.a[Self::M44],
            ],
        }
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<M4x4UpperTriangle<f32>>();
    }
}
