use core::ops::{Add, Div, Index, Mul, Sub};
use num_traits::Signed;

use crate::{MathConstants, Matrix3x3, Vector3};

/// Upper-Triangular Array representing a 3x3 matrix of `f32` values<br>
pub type M3x3UpperTriangularArrayf32 = M3x3UpperTriangularArray<f32>;
/// Upper-Triangular Array representing a 3x3 matrix of `f64` values<br><br>
pub type M3x3UpperTriangularArrayf64 = M3x3UpperTriangularArray<f64>;

/// A 3x3 symmetric matrix stored as a flat 6-element array (Upper-Triangular layout):
/// `[m11, m12, m13,
///        m22, m23,
///             m33]`.
/// Also known as Voigt Notation or Mandel Notation.
/// The overwhelming historical convention is to use upper-triangular rather than lower-triangular layout.
/// This convention is used by libraries such as BLAS, LAPACK, and Intel MKL.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct M3x3UpperTriangularArray<T> {
    pub(crate) a: [T; UTA_3X3_SIZE],
}
const UTA_3X3_SIZE: usize = 6;

impl<T> M3x3UpperTriangularArray<T> {
    /// Size of upper triangular array.
    pub const SIZE: usize = UTA_3X3_SIZE;

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

impl<T> M3x3UpperTriangularArray<T>
where
    T: Copy,
{
    /// Create an upper-triangular array.
    /// ```
    /// # use vqm::M3x3UpperTriangularArrayf32;
    /// let uta = M3x3UpperTriangularArrayf32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// ```
    #[inline]
    pub const fn new(a: [T; UTA_3X3_SIZE]) -> Self {
        Self { a }
    }
}

// **** Other constructors ****

impl<T> M3x3UpperTriangularArray<T>
where
    T: Copy + Signed + Sub<Output = T> + PartialOrd + MathConstants,
{
    /// Create an `M3x3UpperTriangularArray` from a symmetric Matrix3x3.
    /// Does not check if the matrix is symmetric.
    /// ```
    /// # use vqm::{M3x3UpperTriangularArrayf32, Matrix3x3f32};
    /// let m = Matrix3x3f32::new([
    ///     2.0,  3.0,  7.0,
    ///     3.0,  5.0, 11.0,
    ///     7.0, 11.0, 29.0
    /// ]);
    /// let uta = M3x3UpperTriangularArrayf32::from_symmetric_matrix(m);
    /// assert_eq!(uta, M3x3UpperTriangularArrayf32::new([2.0, 3.0, 7.0, 5.0, 11.0, 29.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_symmetric_matrix(m: Matrix3x3<T>) -> Self {
        Self { a: [
            m[Matrix3x3::<T >::M11], m[Matrix3x3::<T >::M12], m[Matrix3x3::<T >::M13],
                                     m[Matrix3x3::<T >::M22], m[Matrix3x3::<T >::M23],
                                                              m[Matrix3x3::<T >::M33],
        ]}
    }

    /// Tries to create an `M3x3UpperTriangularArray` from a symmetric Matrix3x3.
    /// Returns `None` if the input matrix is not symmetric within floating-point tolerance `epsilon`.
    /// ```
    /// # use vqm::{M3x3UpperTriangularArrayf32, Matrix3x3f32};
    /// let m = Matrix3x3f32::new([
    ///     2.0,  3.0,  7.0,
    ///     3.0,  5.0, 11.0,
    ///     7.0, 11.0, 29.0
    /// ]);
    /// let option = M3x3UpperTriangularArrayf32::try_from_symmetric_matrix(m, 1e-6);
    /// assert!(option.is_some());
    /// if let Some(uta) = option {
    ///     assert_eq!(uta, M3x3UpperTriangularArrayf32::new([2.0, 3.0, 7.0, 5.0, 11.0, 29.0]));
    /// }
    /// ```
    #[inline]
    pub fn try_from_symmetric_matrix(m: Matrix3x3<T>, epsilon: T) -> Option<Self> {
        if !m.is_symmetric(epsilon) {
            return None;
        }
        Some(Self::from_symmetric_matrix(m))
    }

    /// Tries to create an `M3x3UpperTriangularArray` from the inverse of a symmetric Matrix3x3.
    /// Returns `None` if the input matrix is not symmetric, or not invertible because the determinant is zero within floating-point tolerance `epsilon`.
    /// ```
    /// # use vqm::{M3x3UpperTriangularArrayf32, Matrix3x3f32};
    /// let m = Matrix3x3f32::new([
    ///     2.0, 3.0, 7.0,
    ///     3.0, 5.0, 11.0,
    ///     7.0, 11.0, 29.0
    /// ]);
    /// let option = M3x3UpperTriangularArrayf32::try_inverse_from_symmetric_matrix(m, 1e-6);
    /// assert!(option.is_some());
    /// if let Some(uta) = option {
    ///     assert_eq!(uta, M3x3UpperTriangularArrayf32::new([6.0, -2.5, -0.5, 2.25, -0.25, 0.25]));
    ///     let n = Matrix3x3f32::from(uta);
    ///     let i = m * n;
    ///     assert_eq!(i, Matrix3x3f32::identity());
    /// }
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn try_inverse_from_symmetric_matrix(m: Matrix3x3<T>, epsilon: T) -> Option<Self> {
        if !m.is_symmetric(epsilon) {
            return None;
        }
        Self { a: [
            m[Matrix3x3::<T >::M11], m[Matrix3x3::<T >::M12], m[Matrix3x3::<T >::M13],
                                     m[Matrix3x3::<T >::M22], m[Matrix3x3::<T >::M23],
                                                              m[Matrix3x3::<T >::M33],
        ] }
        .try_inverse(epsilon)
    }
}

impl<T> M3x3UpperTriangularArray<T>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Signed
        + PartialOrd
        + MathConstants,
{
    /// Try and invert Upper Triangular Array.
    /// Returns `None` if the matrix determinant is zero within a tolerance of `epsilon`.
    /// ```
    /// # use vqm::M3x3UpperTriangularArrayf32;
    /// let uta = M3x3UpperTriangularArrayf32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let option = uta.try_inverse(1e-6);
    /// assert!(option.is_some());
    /// if let Some(uta) = option {
    ///     assert_eq!(uta, M3x3UpperTriangularArrayf32::new([6.0, -2.5, -0.5, 2.25, -0.25, 0.25]));
    /// }
    /// ```
    #[inline]
    pub fn try_inverse(self, epsilon: T) -> Option<Self> {
        // Calculate the cofactors needed for the determinant.
        let c11 = self.a[Self::M22] * self.a[Self::M33] - self.a[Self::M23] * self.a[Self::M23];
        let c12 = self.a[Self::M13] * self.a[Self::M23] - self.a[Self::M12] * self.a[Self::M33];
        let c13 = self.a[Self::M12] * self.a[Self::M23] - self.a[Self::M13] * self.a[Self::M22];

        // Calculate determinant using the first row cofactors
        let determinant = self.a[Self::M11] * c11 + self.a[Self::M12] * c12 + self.a[Self::M13] * c13;

        // Guard against singular matrix.
        if determinant.abs() < epsilon {
            return None;
        }

        let determinant_reciprocal = T::one() / determinant;

        // Calculate remaining cofactors
        let c22 = self.a[Self::M11] * self.a[Self::M33] - self.a[Self::M13] * self.a[Self::M13];
        let c23 = self.a[Self::M13] * self.a[Self::M12] - self.a[Self::M11] * self.a[Self::M23];
        let c33 = self.a[Self::M11] * self.a[Self::M22] - self.a[Self::M12] * self.a[Self::M12];

        Some(Self {
            a: [
                c11 * determinant_reciprocal,
                c12 * determinant_reciprocal,
                c13 * determinant_reciprocal,
                c22 * determinant_reciprocal,
                c23 * determinant_reciprocal,
                c33 * determinant_reciprocal,
            ],
        })
    }
}

// **** VECTOR ****

impl<T> Mul<M3x3UpperTriangularArray<T>> for Vector3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector3<T>;
    /// Pre-multiplies this UTA by a 3D vector.
    /// Calculates: Result = v * uta.
    /// ```
    /// # use vqm::{M3x3UpperTriangularArrayf32, Vector3f32};
    /// let uta = M3x3UpperTriangularArrayf32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let v = Vector3f32::new(13.0, 17.0, 19.0);
    /// let r = v * uta;
    /// assert_eq!(r, Vector3f32::new(210.0, 333.0, 829.0));
    /// ```
    #[inline]
    fn mul(self, uta: M3x3UpperTriangularArray<T>) -> Vector3<T> {
        type Uta<T> = M3x3UpperTriangularArray<T>;
        Vector3 {
            x: self.x * uta.a[Uta::<T>::M11] + self.y * uta.a[Uta::<T>::M12] + self.z * uta.a[Uta::<T>::M13],
            y: self.x * uta.a[Uta::<T>::M12] + self.y * uta.a[Uta::<T>::M22] + self.z * uta.a[Uta::<T>::M23],
            z: self.x * uta.a[Uta::<T>::M13] + self.y * uta.a[Uta::<T>::M23] + self.z * uta.a[Uta::<T>::M33],
        }
    }
}

// **** Post-multiply ****

impl<T> Mul<Vector3<T>> for M3x3UpperTriangularArray<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector3<T>;
    /// Post-multiplies this UTA by a 3D vector.
    ///
    /// Calculates: Result = uta * v.
    /// ```
    /// # use vqm::{M3x3UpperTriangularArrayf32, Vector3f32, Matrix3x3f32};
    /// let uta = M3x3UpperTriangularArrayf32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let v = Vector3f32::new(13.0, 17.0, 19.0);
    /// let r = uta * v;
    /// assert_eq!(r, Vector3f32::new(210.0, 333.0, 829.0));
    /// let m = Matrix3x3f32::from(uta);
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

impl<T> Mul<M3x3UpperTriangularArray<T>> for Matrix3x3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix3x3<T>;

    /// Pre-multiplies this UTA by a 3x3 matrix.
    /// Calculates: Result = m * uta.
    /// ```
    /// # use vqm::{M3x3UpperTriangularArrayf32, Matrix3x3f32};
    /// let uta = M3x3UpperTriangularArrayf32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let m = Matrix3x3f32::new([ 13.0, 17.0, 59.0,
    ///                             19.0, 37.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let r = m * uta;
    /// assert_eq!(r, Matrix3x3f32::new([  490.0,  773.0, 1989.0,
    ///                                    478.0,  759.0, 1903.0,
    ///                                    426.0,  675.0, 1691.0]));
    /// let n = Matrix3x3f32::from(uta);
    /// let r2 = m * n;
    /// assert_eq!(r, r2);
    /// ```
    #[inline]
    fn mul(self, uta: M3x3UpperTriangularArray<T>) -> Matrix3x3<T> {
        let a11 = uta.a[0];
        let a12 = uta.a[1];
        let a13 = uta.a[2];
        let a22 = uta.a[3];
        let a23 = uta.a[4];
        let a33 = uta.a[5];

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

impl<T> Mul<Matrix3x3<T>> for M3x3UpperTriangularArray<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix3x3<T>;

    /// Post-multiplies this UTA by a 3x3 matrix.
    ///
    /// Calculates: Result = uta * m.
    /// ```
    /// # use vqm::{M3x3UpperTriangularArrayf32, Matrix3x3f32};
    /// let uta = M3x3UpperTriangularArrayf32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let m = Matrix3x3f32::new([ 13.0, 17.0, 59.0,
    ///                             19.0, 37.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let r = uta * m;
    /// assert_eq!(r, Matrix3x3f32::new([  244.0,  362.0,  546.0,
    ///                                    387.0,  577.0,  863.0,
    ///                                    967.0, 1425.0, 2119.0]));
    /// let n = Matrix3x3f32::from(uta);
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

impl<T> Index<usize> for M3x3UpperTriangularArray<T> {
    type Output = T;

    /// Access uta element by index.
    /// ```
    /// # use vqm::M3x3UpperTriangularArrayf32;
    /// let uta = M3x3UpperTriangularArrayf32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    ///
    /// assert_eq!(uta[0], 2.0);
    /// assert_eq!(uta[1], 3.0);
    /// assert_eq!(uta[2], 7.0);
    /// assert_eq!(uta[3], 5.0);
    /// assert_eq!(uta[4], 11.0);
    /// assert_eq!(uta[5], 29.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

impl<T> From<M3x3UpperTriangularArray<T>> for Matrix3x3<T>
where
    T: Copy,
{
    /// Matrix3x3 from `M3x3UpperTriangularArray`.
    /// ```
    /// # use vqm::{M3x3UpperTriangularArrayf32, Matrix3x3f32};
    /// let uta = M3x3UpperTriangularArrayf32::new([
    ///     2.0, 3.0,  7.0,
    ///          5.0, 11.0,
    ///               29.0
    /// ]);
    /// let m = Matrix3x3f32::from(uta);
    /// assert_eq!(m, Matrix3x3f32::new([
    ///     2.0,  3.0,  7.0,
    ///     3.0,  5.0, 11.0,
    ///     7.0, 11.0, 29.0
    /// ]));
    #[rustfmt::skip]
    #[inline]
    fn from(uta: M3x3UpperTriangularArray<T>) -> Self {
        Self { a: [
            uta[0], uta[1], uta[2],
            uta[1], uta[3], uta[4],
            uta[2], uta[4], uta[5],
        ] }
    }
}
