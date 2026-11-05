use core::ops::{Add, Div, Index, Mul, Sub};
use num_traits::Signed;

use crate::{MathConstants, Matrix2x2, Vector2};

/// Upper-Triangular Array representing a 2x2 matrix of `f32` values<br>
pub type M2x2UpperTrianglef32 = M2x2UpperTriangle<f32>;
/// Upper-Triangular Array representing a 2x2 matrix of `f64` values<br><br>
pub type M2x2UpperTrianglef64 = M2x2UpperTriangle<f64>;

/// A 2x2 symmetric matrix stored as a flat 3-element array (Upper-Triangular layout):
/// `[m11, m12,
///        m22]`.
/// Also known as Voigt Notation or Mandel Notation.
/// The overwhelming historical convention is to use upper-triangular rather than lower-triangular layout.
/// This convention is used by libraries such as BLAS, LAPACK, and Intel MKL.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct M2x2UpperTriangle<T> {
    pub(crate) a: [T; UTA_2X2_SIZE],
}
const UTA_2X2_SIZE: usize = 3;

impl<T> M2x2UpperTriangle<T> {
    /// Size of upper triangular array.
    pub const SIZE: usize = UTA_2X2_SIZE;

    // Row 1
    const M11: usize = 0;
    const M12: usize = 1;
    // Row 2
    const M22: usize = 2;
}

// **** New ****

impl<T> M2x2UpperTriangle<T>
where
    T: Copy,
{
    /// Create an upper-triangular array.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let uta = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// ```
    #[inline]
    pub const fn new(a: [T; UTA_2X2_SIZE]) -> Self {
        Self { a }
    }
}

// **** Other constructors ****

impl<T> M2x2UpperTriangle<T>
where
    T: Copy + Signed + Sub<Output = T> + PartialOrd + MathConstants,
{
    /// Create an `M2x2UpperTriangle` from a symmetric Matrix2x2.
    /// Does not check if the matrix is symmetric.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Matrix2x2f32};
    /// let m = Matrix2x2f32::new([
    ///     2.0,  3.0,
    ///     3.0,  5.0,
    /// ]);
    /// let uta = M2x2UpperTrianglef32::from_symmetric_matrix(m);
    /// assert_eq!(uta, M2x2UpperTrianglef32::new([2.0, 3.0, 5.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_symmetric_matrix(m: Matrix2x2<T>) -> Self {
        Self { a: [
            m[Matrix2x2::<T >::M11], m[Matrix2x2::<T >::M12],
                                     m[Matrix2x2::<T >::M22],
        ]}
    }

    /// Tries to create an `M2x2UpperTriangle` from a symmetric Matrix2x2.
    /// Returns `None` if the input matrix is not symmetric within floating-point tolerance `epsilon`.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Matrix2x2f32};
    /// let m = Matrix2x2f32::new([
    ///     2.0,  3.0,
    ///     3.0,  5.0,
    /// ]);
    /// let option = M2x2UpperTrianglef32::try_from_symmetric_matrix(m, 1e-6);
    /// assert!(option.is_some());
    /// if let Some(uta) = option {
    ///     assert_eq!(uta, M2x2UpperTrianglef32::new([2.0, 3.0, 5.0]));
    /// }
    /// ```
    #[inline]
    pub fn try_from_symmetric_matrix(m: Matrix2x2<T>, epsilon: T) -> Option<Self> {
        if !m.is_symmetric(epsilon) {
            return None;
        }
        Some(Self::from_symmetric_matrix(m))
    }

    /// Tries to create an `M2x2UpperTriangle` from the inverse of a symmetric Matrix2x2.
    /// Returns `None` if the input matrix is not symmetric, or not invertible because the determinant is zero within floating-point tolerance `epsilon`.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Matrix2x2f32};
    /// let m = Matrix2x2f32::new([
    ///     2.0, 3.0,
    ///     3.0, 5.0,
    /// ]);
    /// let option = M2x2UpperTrianglef32::try_inverse_from_symmetric_matrix(m, 1e-6);
    /// assert!(option.is_some());
    /// if let Some(uta) = option {
    ///     assert_eq!(uta, M2x2UpperTrianglef32::new([5.0, -3.0, 2.0]));
    ///     let n = Matrix2x2f32::from(uta);
    ///     let i = m * n;
    ///     assert_eq!(i, Matrix2x2f32::identity());
    /// }
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn try_inverse_from_symmetric_matrix(m: Matrix2x2<T>, epsilon: T) -> Option<Self> {
        if !m.is_symmetric(epsilon) {
            return None;
        }
        Self { a: [
            m[Matrix2x2::<T >::M11], m[Matrix2x2::<T >::M12],
                                     m[Matrix2x2::<T >::M22]
        ] }
        .try_inverse(epsilon)
    }
}

impl<T> M2x2UpperTriangle<T>
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
    /// # use vqm::M2x2UpperTrianglef32;
    /// let uta = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let option = uta.try_inverse(1e-6);
    /// assert!(option.is_some());
    /// if let Some(uta) = option {
    ///     assert_eq!(uta, M2x2UpperTrianglef32::new([5.0, -3.0, 2.0]));
    /// }
    /// ```
    #[inline]
    pub fn try_inverse(self, epsilon: T) -> Option<Self> {
        // Calculate the cofactors needed for the determinant.
        // Calculate determinant using the first row cofactors
        let determinant = self.a[Self::M11] * self.a[Self::M22] - self.a[Self::M12] * self.a[Self::M12];

        // Guard against singular matrix.
        if determinant.abs() < epsilon {
            return None;
        }

        let determinant_reciprocal = T::one() / determinant;

        Some(Self {
            a: [
                self.a[Self::M22] * determinant_reciprocal,
                -self.a[Self::M12] * determinant_reciprocal,
                self.a[Self::M11] * determinant_reciprocal,
            ],
        })
    }
}

// **** VECTOR ****

impl<T> Mul<M2x2UpperTriangle<T>> for Vector2<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector2<T>;
    /// Post-multiplies this UTA by a 2D vector.
    /// Calculates: Result = v * uta.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32,Vector2f32};
    /// let uta = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let v = Vector2f32::new(13.0, 17.0);
    /// let r = v * uta;
    /// assert_eq!(r, Vector2f32::new(77.0, 124.0));
    /// ```
    #[inline]
    fn mul(self, uta: M2x2UpperTriangle<T>) -> Vector2<T> {
        type Uta<T> = M2x2UpperTriangle<T>;
        Vector2 {
            x: self.x * uta.a[Uta::<T>::M11] + self.y * uta.a[Uta::<T>::M12],
            y: self.x * uta.a[Uta::<T>::M12] + self.y * uta.a[Uta::<T>::M22],
        }
    }
}

// **** Post-multiply ****

impl<T> Mul<Vector2<T>> for M2x2UpperTriangle<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector2<T>;
    /// Post-multiplies this UTA by a 2D vector.
    ///
    /// Calculates: Result = uta * v.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Vector2f32, Matrix2x2f32};
    /// let uta = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let v = Vector2f32::new(13.0, 17.0);
    /// let r = uta * v;
    /// assert_eq!(r, Vector2f32::new(77.0, 124.0));
    /// let m = Matrix2x2f32::from(uta);
    /// let r2 = m * v;
    /// assert_eq!(r, r2);
    /// ```
    #[inline]
    fn mul(self, v: Vector2<T>) -> Vector2<T> {
        Vector2 {
            x: self.a[Self::M11] * v.x + self.a[Self::M12] * v.y,
            y: self.a[Self::M12] * v.x + self.a[Self::M22] * v.y,
        }
    }
}

// **** MATRIX ****

// **** Pre-multiply ****

impl<T> Mul<M2x2UpperTriangle<T>> for Matrix2x2<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix2x2<T>;

    /// Pre-multiplies this UTA by a 2x2 matrix.
    /// Calculates: Result = m * uta.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Matrix2x2f32};
    /// let uta = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let m = Matrix2x2f32::new([ 13.0, 17.0,
    ///                             19.0, 37.0]);
    /// let r = m * uta;
    /// assert_eq!(r, Matrix2x2f32::new([   77.0,  124.0,
    ///                                    149.0,  242.0]));
    /// let n = Matrix2x2f32::from(uta);
    /// let r2 = m * n;
    /// assert_eq!(r, r2);
    /// ```
    #[inline]
    fn mul(self, uta: M2x2UpperTriangle<T>) -> Matrix2x2<T> {
        let a11 = uta.a[0];
        let a12 = uta.a[1];
        let a13 = uta.a[2];

        // Cache rows of M into registers to prevent strided latency stalls
        // Row 1
        let m11 = self[Matrix2x2::<T>::M11];
        let m12 = self[Matrix2x2::<T>::M12];
        // Row 2
        let m21 = self[Matrix2x2::<T>::M21];
        let m22 = self[Matrix2x2::<T>::M22];
        // Column 1 of Result
        let r11 = m11 * a11 + m12 * a12;
        let r21 = m21 * a11 + m22 * a12;

        // Column 2 of Result
        let r12 = m11 * a12 + m12 * a13;
        let r22 = m21 * a12 + m22 * a13;

        Matrix2x2::from([r11, r21, r12, r22])
    }
}

// **** Post-multiply ****

impl<T> Mul<Matrix2x2<T>> for M2x2UpperTriangle<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix2x2<T>;

    /// Post-multiplies this UTA by a 2x2 matrix.
    ///
    /// Calculates: Result = uta * m.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Matrix2x2f32};
    /// let uta = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let m = Matrix2x2f32::new([ 13.0, 17.0,
    ///                             19.0, 37.0]);
    /// let r = uta * m;
    /// assert_eq!(r, Matrix2x2f32::new([  83.0,  145.0,
    ///                                   134.0,  236.0]));
    /// let n = Matrix2x2f32::from(uta);
    /// let r2 = n * m;
    /// assert_eq!(r, r2);
    /// ```
    #[inline]
    fn mul(self, m: Matrix2x2<T>) -> Matrix2x2<T> {
        let a11 = self.a[Self::M11];
        let a12 = self.a[Self::M12];
        let a22 = self.a[Self::M22];

        // Column 1
        let m11 = m[Matrix2x2::<T>::M11];
        let m21 = m[Matrix2x2::<T>::M21];
        let r11 = a11 * m11 + a12 * m21;
        let r21 = a12 * m11 + a22 * m21;

        // Column 2
        let m12 = m[Matrix2x2::<T>::M12];
        let m22 = m[Matrix2x2::<T>::M22];
        let r12 = a11 * m12 + a12 * m22;
        let r22 = a12 * m12 + a22 * m22;

        Matrix2x2::from([r11, r21, r12, r22])
    }
}

// **** Index ****

impl<T> Index<usize> for M2x2UpperTriangle<T> {
    type Output = T;

    /// Access uta element by index.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let uta = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    ///
    /// assert_eq!(uta[0], 2.0);
    /// assert_eq!(uta[1], 3.0);
    /// assert_eq!(uta[2], 5.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

impl<T> From<M2x2UpperTriangle<T>> for Matrix2x2<T>
where
    T: Copy,
{
    /// Matrix2x2 from `M2x2UpperTriangle`.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Matrix2x2f32};
    /// let uta = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let m = Matrix2x2f32::from(uta);
    /// assert_eq!(m, Matrix2x2f32::new([
    ///     2.0,  3.0,
    ///     3.0,  5.0,
    /// ]));
    #[rustfmt::skip]
    #[inline]
    fn from(uta: M2x2UpperTriangle<T>) -> Self {
        Self { a: [
            uta[0], uta[1],
            uta[1], uta[2],
        ] }
    }
}
