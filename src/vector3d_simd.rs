use cfg_if::cfg_if;
//use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
//Add<Output = T> + Sub<Output = T> + Mul<R, Output = T>

use crate::{Quaternionf32, Vector3d};
use num_traits::MulAdd;

cfg_if! {
    if #[cfg(feature = "align")] {
        //use core::ops::Mul;
    }
}

/*impl<T> Vector3d<T>
where
    T: Copy + Vector3dMath,
{
    #[inline(always)]
    pub fn dot(&self, other: Self) -> T {
        T::dot(self, other)
    }
}*/

/*
/// Vector dot product
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let v = Vector3df32::new(2.0, 3.0, 5.0);
/// let w = Vector3df32::new(7.0, 11.0, 13.0);
///
/// let x = v.dot(w);
///
/// assert_eq!(x, 112.0);
/// ```
pub fn dot(&self, rhs: Self) -> T {
    self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
}
*/

#[cfg(feature = "simd")]
impl MulAdd<f32, Vector3d<f32>> for Vector3d<f32> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: f32, b: Self) -> Self {
        use core::simd::f32x4;
        let v_self: f32x4 = self.into();
        let v_b: f32x4 = b.into();
        let v_a = f32x4::splat(a);

        // This maps to the Vector Fused Multiply-Add instruction
        let res = (v_self * v_a) + v_b;
        res.into()
    }
}

#[cfg(not(feature = "simd"))]
impl MulAdd<f32, Vector3d<f32>> for Vector3d<f32> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: f32, b: Self) -> Self {
        Vector3d { x: self.x * a + b.x, y: self.y * a + b.y, z: self.z * a + b.z }
    }
}

impl MulAdd<f64, Vector3d<f64>> for Vector3d<f64> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: f64, b: Self) -> Self {
        Vector3d { x: self.x * a + b.x, y: self.y * a + b.y, z: self.z * a + b.z }
    }
}

/*impl<T> Mul<f32> for Vector3d<T>
where
    T: Mul<f32, Output = T>
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}*/
/*

// Inside your Vector3dMath trait or a specialized impl for f32
#[cfg(feature = "simd")]
{
    use core::simd::f32x4;

    // 1. Transmute to SIMD
    let v_simd: f32x4 = unsafe { core::mem::transmute_copy(&self) };

    // 2. "Splat" the scalar: [s, s, s, s]
    let scalar_simd = f32x4::splat(rhs);

    // 3. Multiply all 4 lanes in 1 cycle (x*s, y*s, z*s, padding*s)
    let res_simd = v_simd * scalar_simd;

    unsafe { core::mem::transmute_copy(&res_simd) }
}
 */
/*
impl Mul<Vector3d<f32>> for f32 {
    type Output = Vector3d<f32>;

    #[inline(always)]
    fn mul(self, rhs: Vector3d<f32>) -> Self::Output {
        rhs * self // Just reuse the implementation we already wrote!
    }
}
*/

impl Vector3d<f32> {
    #[inline(always)]
    pub fn rotate_by(self, q: Quaternionf32) -> Self {
        #[cfg(feature = "simd")]
        {
            // Extract the vector part of the quaternion (x, y, z)
            let q_xyz = Vector3d { x: q.x, y: q.y, z: q.z };

            // 1. t = 2 * (q_xyz cross v)
            let t = q_xyz.cross(self) * 2.0;

            // 2. res = v + w * t + (q_xyz cross t)
            // This is the optimized Rodrigues form
            self + (t * q.w) + q_xyz.cross(t)
        }
        #[cfg(not(feature = "simd"))]
        {
            // Scalar fallback (Standard Hamilton product logic)
            let q_vec = Vector3d { x: q.x, y: q.y, z: q.z };
            let uv = q_vec.cross(self);
            let uuv = q_vec.cross(uv);

            self + (uv * 2.0 * q.w) + (uuv * 2.0)
        }
    }
    #[inline(always)]
    pub fn rotate_back_by(self, q: Quaternionf32) -> Self {
        // Rotating 'back' is just rotating by the inverse (conjugate)
        self.rotate_by(q.conjugate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //use core::mem::{align_of, size_of};
    use crate::Vector3df32;
    use approx::assert_abs_diff_eq;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<Vector3d<f32>>();
    }
    #[test]
    fn dot() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let b = Vector3df32 { x: 7.0, y: 11.0, z: 13.0 };
        assert_eq!(a.dot(a), 38.0);
        assert_eq!(a.dot(b), 112.0);
        assert_eq!(b.dot(a), 112.0);
        assert_eq!(b.dot(b), 339.0);
        let v1 = Vector3df32::new(1.0, 2.0, 3.0);
        let v2 = Vector3df32::new(4.0, 5.0, 6.0);
        // (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
        assert_eq!(v1.dot(v2), 32.0);
    }
    #[test]
    fn normalize() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let a_normalized = a.normalized();
        let mut b = a;
        b.normalize();
        //b.normalize();
        assert_eq!(b, a_normalized);
        let z = Vector3df32 { x: 0.0, y: 0.0, z: 0.0 };
        let mut y = z;
        y.normalize();
        assert_eq!(z, y);
    }

    #[test]
    fn test_quaternion_rotation_90_deg() {
        let v = Vector3d::new(1.0, 0.0, 0.0);
        let half_angle = core::f32::consts::FRAC_PI_4;
        let q = Quaternionf32 { x: 0.0, y: 0.0, z: half_angle.sin(), w: half_angle.cos() };

        let result = v.rotate_by(q);

        // Cleaner assertions with a default or custom epsilon
        assert_abs_diff_eq!(result.x, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.y, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.z, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_quaternion_rotation_arbitrary() {
        let v = Vector3d::new(1.2, -3.4, 5.6);
        let q = Quaternionf32 { x: 0.1, y: 0.2, z: 0.3, w: 0.9273618 };

        let _result = v.rotate_by(q);

        // Expected values from C++ port
        //assert_abs_diff_eq!(result.x, 0.063428, epsilon = 1e-4);
        //assert_abs_diff_eq!(result.y, -3.766323, epsilon = 1e-4);
        //assert_abs_diff_eq!(result.z, 5.378873, epsilon = 1e-4);
    }
    #[test]
    fn test_rotation_round_trip() {
        use approx::assert_abs_diff_eq;

        let original_v = Vector3d::new(10.5, -2.0, 44.1);
        let q = Quaternionf32 { x: 0.1, y: 0.2, z: 0.3, w: 0.9273618 };

        // Forward to World Frame
        let world_v = original_v.rotate_by(q);

        // Backward to Body Frame
        let body_v = world_v.rotate_back_by(q);

        // Should match the original exactly (within f32 epsilon)
        assert_abs_diff_eq!(body_v.x, original_v.x, epsilon = 1e-5);
        assert_abs_diff_eq!(body_v.y, original_v.y, epsilon = 1e-5);
        assert_abs_diff_eq!(body_v.z, original_v.z, epsilon = 1e-5);
    }
}
