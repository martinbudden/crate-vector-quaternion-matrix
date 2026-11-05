use core::ops::{Add, Index, Mul, Neg, Sub};
use num_traits::{ConstOne, Signed};

use crate::{MathConstants, MathMethods, Matrix2x2, Vector2};

/// Upper triangular array representing a 2x2 matrix of `f32` values<br>
pub type M2x2UpperTrianglef32 = M2x2UpperTriangle<f32>;
/// Upper triangular array representing a 2x2 matrix of `f64` values<br><br>
pub type M2x2UpperTrianglef64 = M2x2UpperTriangle<f64>;

/// A 2x2 symmetric matrix stored as a flat 3-element array (Upper-Triangular layout).
/// ```
/// [m11, m12,
///        m22]
/// ```
/// Also known as Voigt Notation or Mandel Notation.
// The overwhelming historical convention is to use upper-triangular rather than lower-triangular layout.
// This convention is used by libraries such as BLAS, LAPACK, and Intel MKL.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct M2x2UpperTriangle<T> {
    pub(crate) a: [T; UT_2X2_SIZE],
}
const UT_2X2_SIZE: usize = 3;

impl<T> M2x2UpperTriangle<T> {
    /// Size of upper triangular array.
    pub const SIZE: usize = UT_2X2_SIZE;

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
    /// Create an upper-triangle.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// ```
    #[inline]
    pub const fn new(a: [T; UT_2X2_SIZE]) -> Self {
        Self { a }
    }
}

// **** Other constructors ****

impl<T> M2x2UpperTriangle<T> {
    /// Create an `M2x2UpperTriangle` from a Matrix2x2.
    /// Does not check if the matrix is symmetric.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Matrix2x2f32};
    /// let m = Matrix2x2f32::new([
    ///     2.0,  3.0,
    ///     7.0,  5.0,
    /// ]);
    /// let ut = M2x2UpperTrianglef32::from_matrix(m);
    /// assert_eq!(ut, M2x2UpperTrianglef32::new([2.0, 3.0,
    ///                                                5.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_matrix(m: Matrix2x2<T>) -> Self
    where
        T: Copy,
    {
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
    /// let option = M2x2UpperTrianglef32::try_from_matrix(m, 1e-6);
    /// assert!(option.is_some());
    /// if let Some(ut) = option {
    ///     assert_eq!(ut, M2x2UpperTrianglef32::new([2.0, 3.0,
    ///                                                     5.0]));
    /// }
    /// ```
    #[inline]
    pub fn try_from_matrix(m: Matrix2x2<T>, epsilon: T) -> Option<Self>
    where
        T: Copy + Signed + PartialOrd,
    {
        if !m.is_symmetric(epsilon) {
            return None;
        }
        Some(Self::from_matrix(m))
    }

    /// Tries to create an `M2x2UpperTriangle` from the inverse of a symmetric Matrix2x2.
    /// Returns `None` if the input matrix is not symmetric, or not invertible because the determinant is zero within floating-point tolerance `epsilon`.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Matrix2x2f32};
    /// let m = Matrix2x2f32::new([
    ///     2.0, 3.0,
    ///     3.0, 5.0,
    /// ]);
    /// let option = M2x2UpperTrianglef32::try_inverse_from_matrix(m, 1e-6);
    /// assert!(option.is_some());
    /// if let Some(ut) = option {
    ///     assert_eq!(ut, M2x2UpperTrianglef32::new([5.0, -3.0,
    ///                                                      2.0]));
    ///     let n = Matrix2x2f32::from(ut);
    ///     let i = m * n;
    ///     assert_eq!(i, Matrix2x2f32::identity());
    /// }
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn try_inverse_from_matrix(m: Matrix2x2<T>, epsilon: T) -> Option<Self>
    where
        T: Copy + ConstOne + Signed + Sub<Output = T> + PartialOrd + MathConstants,
    {
        if !m.is_symmetric(epsilon) {
            return None;
        }
        Self { a: [
            m[Matrix2x2::<T >::M11], m[Matrix2x2::<T >::M12],
                                     m[Matrix2x2::<T >::M22]
        ] }
        .try_inverse(epsilon)
    }

    /// Try and invert Upper Triangle.
    /// Returns `None` if the matrix determinant is zero within a tolerance of `epsilon`.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let (adjugate, determinant) = ut.adjugate();
    /// assert_eq!(determinant, 1.0);
    /// assert_eq!(adjugate, M2x2UpperTrianglef32::new([5.0, -3.0,
    ///                                                       2.0]));
    /// ```
    #[inline]
    pub fn adjugate(self) -> (Self, T)
    where
        T: Copy + Neg<Output = T> + Sub<Output = T> + Mul<Output = T>,
        // Calculate the cofactors needed for the determinant.
        // Calculate determinant using the first row cofactors
    {
        let determinant = self.a[Self::M11] * self.a[Self::M22] - self.a[Self::M12] * self.a[Self::M12];

        // Guard against singular matrix.

        (Self { a: [self.a[Self::M22], -self.a[Self::M12], self.a[Self::M11]] }, determinant)
    }

    /// Try and invert Upper Triangle.
    /// Returns `None` if the matrix determinant is zero within a tolerance of `epsilon`.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let option = ut.try_inverse(1e-6);
    /// assert!(option.is_some());
    /// if let Some(ut) = option {
    ///     assert_eq!(ut, M2x2UpperTrianglef32::new([5.0, -3.0,
    ///                                                     2.0]));
    /// }
    /// ```
    #[inline]
    pub fn try_inverse(self, epsilon: T) -> Option<Self>
    where
        T: Copy + ConstOne + Signed + PartialOrd,
    {
        let (adjugate, determinant) = self.adjugate();
        // Guard against singular matrix.
        if determinant.abs() < epsilon {
            return None;
        }
        let determinant_reciprocal = T::ONE / determinant;
        Some(adjugate * determinant_reciprocal)
    }

    /// Return the sum of all the elements on the diagonal.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let n = ut.trace();
    ///
    /// assert_eq!(n, 7.0);
    /// ```
    #[inline]
    pub fn trace(self) -> T
    where
        T: Copy + Add<Output = T>,
    {
        self.a[0] + self.a[2]
    }

    /// Return the sum of the squares of all the elements on the diagonal.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let n = ut.trace_sum_squares();
    ///
    /// assert_eq!(n, 29.0);
    /// ```
    #[inline]
    pub fn trace_sum_squares(self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        self.a[0] * self.a[0] + self.a[2] * self.a[2]
    }

    /// Return norm squared of the upper triangle, that is the sum of the squares of all the elements.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let n = ut.norm_squared();
    ///
    /// assert_eq!(n, 38.0);
    /// ```
    #[inline]
    pub fn norm_squared(self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        self.a[0] * self.a[0] + self.a[1] * self.a[1] + self.a[2] * self.a[2]
    }

    /// Return norm of the upper triangle, that is the square root of the sum of the squares of all the elements.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let n = ut.norm();
    ///
    /// assert!((n - 6.164414003).abs() < 1e-6);
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
    /// # use vqm::M2x2UpperTrianglef32;
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let n = ut.matrix_norm_squared();
    ///
    /// assert_eq!(n, 47.0);
    /// ```
    #[inline]
    pub fn matrix_norm_squared(self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        self.norm_squared() + self.a[1] * self.a[1]
    }

    /// Return norm of the matrix represented by this upper triangle.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let n = ut.matrix_norm();
    ///
    /// assert!((n - 6.8556546).abs() < 1e-6);
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

impl<T> Mul<T> for M2x2UpperTriangle<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    /// Post-multiply upper triangle by a scalar.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let r = ut * 2.0;
    /// assert_eq!(r, M2x2UpperTrianglef32::new([4.0, 6.0,
    ///                                              10.0]));
    /// ```
    #[inline]
    fn mul(self, k: T) -> Self {
        Self { a: [self.a[0] * k, self.a[1] * k, self.a[2] * k] }
    }
}

// **** VECTOR ****

// **** Pre-multiply ****

impl<T> Mul<M2x2UpperTriangle<T>> for Vector2<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector2<T>;
    /// Pre-multiplies an upper triangle by a 2D vector.
    /// Calculates: Result = v * ut.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32,Vector2f32};
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let v = Vector2f32::new(13.0, 17.0);
    /// let r = v * ut;
    /// assert_eq!(r, Vector2f32::new(77.0, 124.0));
    /// ```
    #[inline]
    fn mul(self, ut: M2x2UpperTriangle<T>) -> Vector2<T> {
        type Ut<T> = M2x2UpperTriangle<T>;
        Vector2 {
            x: self.x * ut.a[Ut::<T>::M11] + self.y * ut.a[Ut::<T>::M12],
            y: self.x * ut.a[Ut::<T>::M12] + self.y * ut.a[Ut::<T>::M22],
        }
    }
}

// **** Post-multiply ****

impl<T> Mul<Vector2<T>> for M2x2UpperTriangle<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector2<T>;
    /// Post-multiplies this upper triangle by a 2D vector.
    ///
    /// Calculates: Result = ut * v.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Vector2f32, Matrix2x2f32};
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let v = Vector2f32::new(13.0, 17.0);
    /// let r = ut * v;
    /// assert_eq!(r, Vector2f32::new(77.0, 124.0));
    /// let m = Matrix2x2f32::from(ut);
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

    /// Pre-multiplies this upper triangle by a 2x2 matrix.
    /// Calculates: Result = m * ut.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Matrix2x2f32};
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let m = Matrix2x2f32::new([ 13.0, 17.0,
    ///                             19.0, 37.0]);
    /// let r = m * ut;
    /// assert_eq!(r, Matrix2x2f32::new([   77.0,  124.0,
    ///                                    149.0,  242.0]));
    /// let n = Matrix2x2f32::from(ut);
    /// let r2 = m * n;
    /// assert_eq!(r, r2);
    /// ```
    #[inline]
    fn mul(self, ut: M2x2UpperTriangle<T>) -> Matrix2x2<T> {
        let a11 = ut.a[0];
        let a12 = ut.a[1];
        let a13 = ut.a[2];

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

    /// Post-multiplies this upper triangle by a 2x2 matrix.
    ///
    /// Calculates: Result = ut * m.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Matrix2x2f32};
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let m = Matrix2x2f32::new([ 13.0, 17.0,
    ///                             19.0, 37.0]);
    /// let r = ut * m;
    /// assert_eq!(r, Matrix2x2f32::new([  83.0,  145.0,
    ///                                   134.0,  236.0]));
    /// let n = Matrix2x2f32::from(ut);
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

    /// Access upper triangle element by index.
    /// ```
    /// # use vqm::M2x2UpperTrianglef32;
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    ///
    /// assert_eq!(ut[0], 2.0);
    /// assert_eq!(ut[1], 3.0);
    /// assert_eq!(ut[2], 5.0);
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
    /// let ut = M2x2UpperTrianglef32::new([
    ///     2.0, 3.0,
    ///          5.0,
    /// ]);
    /// let m = Matrix2x2f32::from(ut);
    /// assert_eq!(m, Matrix2x2f32::new([
    ///     2.0,  3.0,
    ///     3.0,  5.0,
    /// ]));
    #[rustfmt::skip]
    #[inline]
    fn from(ut: M2x2UpperTriangle<T>) -> Self {
        Self { a: [
            ut[0], ut[1],
            ut[1], ut[2],
        ] }
    }
}

// **** MATRIX ****

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Returns the upper triangle (including diagonal) of a matrix as a `M2x2UpperTrianglef32`.
    /// The lower triangle of the matrix is ignored.
    /// ```
    /// # use vqm::{M2x2UpperTrianglef32, Matrix2x2f32};
    /// let M = Matrix2x2f32::new([
    ///     2.0, 3.0,
    ///     7.0, 5.0
    /// ]);
    /// let ut = M.upper_triangle();
    /// assert_eq!(ut, M2x2UpperTrianglef32::new([2.0, 3.0,
    ///                                                5.0]));
    /// ```
    #[rustfmt::skip]
    #[must_use]
    #[inline]
    pub fn upper_triangle(self) -> M2x2UpperTriangle<T> {
        M2x2UpperTriangle::<T> { a: [
            self.a[Self::M11], self.a[Self::M12],
                               self.a[Self::M22],
        ]}
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<M2x2UpperTriangle<f32>>();
    }
}
