#![allow(unused)]
use core::ops::{Div, Mul, Neg, Sub};
use num_traits::{One, Zero, float::FloatCore};

use crate::{MathConstants, MathMethods, Matrix3x3, Matrix3x3f32, Vector3d, Vector3df32};

pub struct EigenResult3x3<T> {
    pub eigenvalue: T,
    pub eigenvector: Vector3d<T>,
}

pub type EigenResult3x3f32 = EigenResult3x3<f32>;
pub type EigenResult3x3f64 = EigenResult3x3<f64>;

pub struct EigenResult {
    pub eigenvalues: [f32; 3],
    pub eigenvectors: [f32; 9], // Each column is an eigenvector
}

/*fn sort3_in_place<T: PartialOrd>(arr: &mut [T; 3]) {
    if arr[0] > arr[1] {
        arr.swap(0, 1);
    }
    if arr[1] > arr[2] {
        arr.swap(1, 2);
    }
    if arr[0] > arr[1] {
        arr.swap(0, 1);
    }
}*/

#[allow(unused)]
fn sort3<T: PartialOrd + Copy>(arr: [T; 3]) -> [T; 3] {
    let mut sorted = arr;
    if sorted[0] > sorted[1] {
        sorted.swap(0, 1);
    }
    if sorted[1] > sorted[2] {
        sorted.swap(1, 2);
    }
    if sorted[0] > sorted[1] {
        sorted.swap(0, 1);
    }
    sorted
}

impl Matrix3x3f32 {
    pub fn eigen_symmetric1(&self) -> [f32; 3] {
        let m00 = self[0];
        let m01 = self[1];
        let m02 = self[2];
        let m11 = self[4];
        let m12 = self[5];
        let m22 = self[8];

        // Compute trace and scaled matrix
        let trace = m00 + m11 + m22;
        let q = trace / 3.0;

        // Compute symmetric invariants
        let b00 = m00 - q;
        let b11 = m11 - q;
        let b22 = m22 - q;

        let p = ((b00 * b00
            + m01 * m01
            + m02 * m02
            + m01 * m01
            + b11 * b11
            + m12 * m12
            + m02 * m02
            + m12 * m12
            + b22 * b22)
            / 2.0)
            .sqrt();

        // Avoid division by zero
        if p == 0.0 {
            return [q, q, q];
        }

        // Compute discriminant
        let r = (b00 * (b11 * b22 - m12 * m12) - m01 * (m01 * b22 - m12 * m02) + m02 * (m01 * m12 - b11 * m02))
            / (p * p * p);

        // Clamp for numerical safety
        let r = r.clamp(-1.0, 1.0);

        // Compute angles
        let phi = (r).acos() / 3.0;

        // Eigenvalues
        let eig1 = q + 2.0 * p * phi.cos();
        let eig2 = q + 2.0 * p * (phi + 2.0 * core::f32::consts::PI / 3.0).cos();
        let eig3 = q + 2.0 * p * (phi + 4.0 * core::f32::consts::PI / 3.0).cos();

        // Sort eigenvalues and compute eigenvectors (simplified; full version requires solving (A - λI)v = 0)
        // For brevity, eigenvectors are omitted here but can be computed via null space methods.
        [eig1, eig2, eig3]
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy
        + Zero
        + One
        + MathConstants
        + MathMethods
        + PartialOrd
        + FloatCore
        + Neg<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    // Return the normalized matrix N = (A - qI) / p
    #[allow(clippy::assign_op_pattern)] // so we don't need the MulAssign trait
    pub fn normalize_with_intermediates(&self) -> (Self, T, T) {
        let trace = self.trace();
        let q = trace / T::THREE;

        // Compute p1: sum of squares of off-diagonal elements
        let p1 = self[1] * self[1] + self[2] * self[2] + self[5] * self[5];
        if p1 == T::zero() {
            // matrix is diagonal
            return (
                Matrix3x3::<T>::from([
                    self[0] - q,
                    self[1],
                    self[2],
                    self[3],
                    self[4] - q,
                    self[5],
                    self[6],
                    self[7],
                    self[8] - q,
                ]),
                q,
                T::zero(),
            );
        }

        let mut n = Matrix3x3::<T>::from([
            self[0] - q,
            self[1],
            self[2],
            T::zero(),
            self[4] - q,
            self[5],
            T::zero(),
            T::zero(),
            self[8] - q,
        ]);
        let p2 = n.trace_sum_squares() + T::TWO * p1;
        let p = (p2 / T::SIX).sqrt();

        // Build normalized matrix B = (A - qI) / p
        let p_reciprocal = T::one() / p;
        n = n * p_reciprocal;
        (n, p, q)
    }

    pub fn eigen_symmetric_mb_t(&self) -> [T; 3] {
        let (n, p, q) = self.normalize_with_intermediates();

        // Clamp r to [-1, 1] for numerical stability
        let r = (n.top_right_determinant() / T::TWO).clamp(-T::one(), T::one());

        // Compute eigenvalues
        let phi = r.acos() / T::THREE;
        let eig1 = q + T::TWO * p * phi.cos();
        let eig2 = q + T::TWO * p * (phi + T::TWO * T::PI / T::THREE).cos();
        let eig3 = T::THREE * q - eig1 - eig2; // Trace conservation

        // TODO: Sort eigenvalues and compute eigenvectors via (A - λI) × v = 0
        [eig1, eig2, eig3]
    }
}

impl Matrix3x3f32 {
    pub fn eigen_symmetric_mb(&self) -> [f32; 3] {
        let (n, p, q) = self.normalize_with_intermediates();
        let r = n.top_right_determinant() / 2.0;
        // Clamp r to [-1, 1] for numerical stability
        let r = r.clamp(-1.0, 1.0);
        let phi = r.acos() / 3.0;

        // Compute eigenvalues
        let eig1 = q + 2.0 * p * phi.cos();
        let eig3 = q + 2.0 * p * (phi + 2.0 * core::f32::consts::PI / 3.0).cos();
        let eig2 = 3.0 * q - eig1 - eig3; // Trace conservation

        // Sort eigenvalues and compute eigenvectors via (A - λI) × v = 0
        [eig1, eig2, eig3]
    }
    pub fn eigen_symmetric(&self) -> [f32; 3] {
        let m00 = self[0];
        let m01 = self[1];
        let m02 = self[2];
        let m11 = self[4];
        let m12 = self[5];
        let m22 = self[8];

        // Compute p1: sum of squares of off-diagonal elements
        //let p1 = m01 * m01 + m02 * m02 + m12 * m12;
        // Compute p1: sum of squares of off-diagonal elements
        let p1 = self[1] * self[1] + self[2] * self[2] + self[5] * self[5];

        // If matrix is diagonal
        if p1 == 0.0 {
            return [m00, m11, m22];
        }

        // Compute trace and q
        //let q = (m00 + m11 + m22) / 3.0;
        let q = self.trace() / 3.0;

        let mut b = *self;
        b.subtract_from_diagonal_in_place(Vector3df32 { x: q, y: q, z: q });
        let p2 = b.trace_sum_squares() + 2.0 * p1;

        // Compute p from variance of diagonal elements
        let b00 = m00 - q;
        let b11 = m11 - q;
        let b22 = m22 - q;
        let _p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();

        // Build normalized matrix B = (A - qI) / p
        let p_reciprocal = 1.0 / p;
        let b00 = b00 * p_reciprocal;
        let b11 = b11 * p_reciprocal;
        let b22 = b22 * p_reciprocal;
        let b01 = m01 * p_reciprocal;
        let b02 = m02 * p_reciprocal;
        let b12 = m12 * p_reciprocal;

        b *= p_reciprocal;
        let det_b = b.top_right_determinant();

        // Compute det(B) / 2
        let _det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
        let r = det_b / 2.0;

        // Clamp r to [-1, 1] for numerical stability
        let r = r.clamp(-1.0, 1.0);
        let phi = (r).acos() / 3.0;

        // Compute eigenvalues
        let eig1 = q + 2.0 * p * phi.cos();
        let eig3 = q + 2.0 * p * (phi + 2.0 * core::f32::consts::PI / 3.0).cos();
        let eig2 = 3.0 * q - eig1 - eig3; // Trace conservation

        // Sort eigenvalues and compute eigenvectors via (A - λI) × v = 0
        //let eigenvalues = [eig1, eig2, eig3];
        //let mut eigenvectors = [0.0f32; 9];

        // Simple solver for (A - λI) v = 0 using cross product of rows
        /*for i in 0..3 {
            let a = m00 - eigenvalues[i];
            let b = m01;
            let c = m02;
            let d = m11 - eigenvalues[i];
            let e = m12;
            let f = m22 - eigenvalues[i];

            // Use cross product of first two rows of (A - λI)
            let v0 = b * f - c * e;
            let v1 = c * d - a * f;
            let v2 = a * e - b * d;

            let norm = (v0 * v0 + v1 * v1 + v2 * v2).sqrt();
            if norm > 1e-6 {
                eigenvectors[i * 3 + 0] = v0 / norm;
                eigenvectors[i * 3 + 1] = v1 / norm;
                eigenvectors[i * 3 + 2] = v2 / norm;
            } else {
                // Fallback: identity vector
                eigenvectors[i * 3 + 0] = 1.0;
                eigenvectors[i * 3 + 1] = 0.0;
                eigenvectors[i * 3 + 2] = 0.0;
            }
        }*/

        // Optional: sort by eigenvalue descending
        // (omitted for brevity)

        [eig1, eig2, eig3]
    }
}

impl Matrix3x3f32 {
    pub fn jacobi_eigen(&self) -> [f32; 3] {
        const MAX_ITERATION_COUNT: usize = 50;
        const TOL: f32 = 1e-6;

        // Initialize eigenvectors to identity
        let mut v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut a = *self; // Working matrix

        for _ in 0..MAX_ITERATION_COUNT {
            // Find largest off-diagonal element
            let mut max_off = 0.0;
            let mut p = 0;
            let mut q = 1;
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let val = a[i * 3 + j].abs();
                    if val > max_off {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_off < TOL {
                break;
            }

            // Compute rotation angle
            let g = (a[q * 3 + q] - a[p * 3 + p]) / (2.0 * a[p * 3 + q]);
            let t = if g.abs() < 1e-8 { 1.0 } else { 1.0 / (g + g.signum() * (g * g + 1.0).sqrt()) };

            let c = (1.0 / (1.0 + t * t)).sqrt();
            let s = t * c;

            // Apply rotation: V = V * R
            for i in 0..3 {
                let vp = v[i * 3 + p];
                let vq = v[i * 3 + q];
                v[i * 3 + p] = vp * c - vq * s;
                v[i * 3 + q] = vp * s + vq * c;
            }

            // Apply similarity transform: A = R^T * A * R
            let ap = a[p * 3 + p];
            let aq = a[q * 3 + q];
            let apq = a[p * 3 + q];

            a[p * 3 + p] = ap * c * c - 2.0 * apq * c * s + aq * s * s;
            a[q * 3 + q] = ap * s * s + 2.0 * apq * c * s + aq * c * c;
            a[p * 3 + q] = 0.0;
            a[q * 3 + p] = 0.0;

            for i in 0..3 {
                if i != p && i != q {
                    let aip = a[i * 3 + p];
                    let aiq = a[i * 3 + q];
                    a[i * 3 + p] = aip * c - aiq * s;
                    a[p * 3 + i] = a[i * 3 + p];
                    a[i * 3 + q] = aip * s + aiq * c;
                    a[q * 3 + i] = a[i * 3 + q];
                }
            }
        }

        //EigenResult { eigenvalues: [a[0], a[4], a[8]], eigenvectors: v }
        [a[0], a[4], a[8]]
    }
}

/*Example:
Matrix A:

4  0  1
-1 -6 -2
5  0  0

Step 1: Characteristic equation
det(λI − A) = λ³ + 2λ² − 29λ − 30 = 0

Step 2: Eigenvalues
Solve cubic: λ = -1, -6, 5

Step 3: Eigenvectors
For each λ, solve (A − λI)v = 0:

λ = -1 → v ≈ [-0.2, -0.36, 1]
λ = -6 → v ≈ [0, 1, 0]
λ = 5 → v ≈ [1, -0.27, 1]
 */
/*
let matrix = Matrix3x3 {
    a: [2.0, -1.0, 0.0,
         -1.0, 2.0, -1.0,
          0.0, -1.0, 2.0],
};

Eigenvalues:

2.0
2 + sqrt(2) ≈ 3.414
2 - sqrt(2) ≈ 0.586
Eigenvectors (approximate, normalized):

For 2.0: [0.707, 0.0, -0.707]
For 3.414: [0.5, -0.707, 0.5]
For 0.586: [0.5, 0.707, 0.5]
*/

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<Vector3df32>();
    }
    #[test]
    fn eigen_symmetric1_a() {
        let a = Matrix3x3f32::from([4.0, 0.0, 1.0, -1.0, -6.0, -2.0, 5.0, 0.0, 0.0]);
        let result = sort3(a.eigen_symmetric1());
        assert_eq! {-10.519317, result[0]};
        assert_eq! {-0.005782962, result[1]};
        assert_eq! {8.525097, result[2]};
    }
    #[test]
    fn eigen_symmetric_a() {
        let a = Matrix3x3f32::from([4.0, 0.0, 1.0, -1.0, -6.0, -2.0, 5.0, 0.0, 0.0]);
        let result = sort3(a.eigen_symmetric());
        assert_eq! {-6.613559, result[0]};
        assert_eq! {0.35506535, result[1]};
        assert_eq! {4.2584934, result[2]};
    }
    #[test]
    fn eigen_symmetric_mb_a() {
        let a = Matrix3x3f32::from([4.0, 0.0, 1.0, -1.0, -6.0, -2.0, 5.0, 0.0, 0.0]);
        let result = sort3(a.eigen_symmetric_mb());
        assert_eq! {-6.613559, result[0]};
        assert_eq! {0.35506535, result[1]};
        assert_eq! {4.2584934, result[2]};
    }
    #[test]
    fn jacobi_eigen_a() {
        let a = Matrix3x3f32::from([4.0, 0.0, 1.0, -1.0, -6.0, -2.0, 5.0, 0.0, 0.0]);
        let result = sort3(a.jacobi_eigen());
        assert_eq! {-6.613558, result[0]};
        assert_eq! {0.35506582, result[1]};
        assert_eq! {4.258493, result[2]};
    }
    #[test]
    fn eigen_symmetric1_b() {
        let a = Matrix3x3f32::from([2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0]);
        let result = sort3(a.eigen_symmetric1());
        assert_eq! {-0.4494896, result[0]};
        assert_eq! {2.0, result[1]};
        assert_eq! {4.4494896, result[2]};
    }
    #[test]
    fn eigen_symmetric_b() {
        let a = Matrix3x3f32::from([2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0]);
        let result = sort3(a.eigen_symmetric());
        assert_eq! {0.58578646, result[0]};
        assert_eq! {1.9999999, result[1]};
        assert_eq! {3.4142137, result[2]};
    }
    #[test]
    fn jacobi_eigen_b() {
        let a = Matrix3x3f32::from([2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0]);
        let result = sort3(a.jacobi_eigen());
        assert_eq! {0.5857865, result[0]};
        assert_eq! {1.9999998, result[1]};
        assert_eq! {3.4142141, result[2]};
    }
}
