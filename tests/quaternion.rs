use vector_quaternion_matrix::{Quaternion, Quaternionf32};

const _: () = assert!(core::mem::size_of::<Quaternion<f32>>() == 16);
const _: () = assert!(core::mem::align_of::<Quaternion<f32>>() == 16);

const _: () = assert!(core::mem::size_of::<Quaternion<f64>>() == 32);
const _: () = assert!(core::mem::align_of::<Quaternion<f64>>() == 16);

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<Quaternionf32>();
    }
    #[test]
    fn default() {
        use num_traits::{One, Zero};
        let a = Quaternionf32::default();
        assert_eq!(a, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
        assert!(a.is_one());
        let z = Quaternionf32::zero();
        assert!(z.is_zero());
        let i = Quaternionf32::one();
        assert!(i.is_one());
    }
    #[test]
    fn from() {
        let a = Quaternionf32::from((0.0, 0.0, 0.0));
        let b = Quaternionf32::from_roll_pitch_yaw_angles_radians(0.0, 0.0, 0.0);
        assert_eq!(a, b);
        let c = Quaternionf32::from((0.0, 0.0));
        let d = Quaternionf32::from_roll_pitch_angles_radians(0.0, 0.0);
        assert_eq!(c, d);
    }
    #[test]
    fn neg() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(-a, Quaternion { w: -2.0, x: -3.0, y: -5.0, z: -7.0 });

        let b = -a;
        assert_eq!(b, Quaternion { w: -2.0, x: -3.0, y: -5.0, z: -7.0 });
    }
    #[test]
    fn test_quaternion_negation() {
        use approx::assert_abs_diff_eq;

        let q = Quaternion { x: 0.1, y: -0.2, z: 0.3, w: 0.9 };
        let neg_q = -q;

        assert_abs_diff_eq!(neg_q.x, -0.1, epsilon = 1e-6);
        assert_abs_diff_eq!(neg_q.y, 0.2, epsilon = 1e-6);
        assert_abs_diff_eq!(neg_q.z, -0.3, epsilon = 1e-6);
        assert_abs_diff_eq!(neg_q.w, -0.9, epsilon = 1e-6);
    }
    #[test]
    fn add() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternion { w: 11.0, x: 13.0, y: 17.0, z: 19.0 };
        assert_eq!(a + b, Quaternion { w: 13.0, x: 16.0, y: 22.0, z: 26.0 });
    }
    #[test]
    fn add_assign() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternion { w: 11.0, x: 13.0, y: 17.0, z: 19.0 };
        let mut c = a;
        c += b;
        assert_eq!(c, Quaternion { w: 13.0, x: 16.0, y: 22.0, z: 26.0 });
    }
    #[test]
    fn sub() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternion { w: 11.0, x: 13.0, y: 17.0, z: 23.0 };
        let c = a - b;
        assert_eq!(c, Quaternion { w: -9.0, x: -10.0, y: -12.0, z: -16.0 });
    }
    #[test]
    fn sub_assign() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternion { w: 11.0, x: 13.0, y: 17.0, z: 23.0 };
        let mut c = a;
        c -= b;
        assert_eq!(c, Quaternion { w: -9.0, x: -10.0, y: -12.0, z: -16.0 });
    }
    #[test]
    fn mul() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(a * 2.0, Quaternion { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
        assert_eq!(2.0 * a, Quaternion { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
    }
    #[test]
    fn mul_assign() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let mut b = a;
        b *= 2.0;
        assert_eq!(b, Quaternion { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
    }
    #[test]
    fn div() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(a / 2.0, Quaternion { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
    }
    #[test]
    fn div_assign() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let mut b = a;
        b /= 2.0;
        assert_eq!(b, Quaternion { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
    }
    #[test]
    fn new() {
        let a = Quaternion::new(2.0, 3.0, 5.0, 7.0);
        assert_eq!(a, Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 });
    }
    #[test]
    fn norm_squared() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(a.norm_squared(), 87.0);
    }
    #[test]
    fn norm() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(a.norm(), 87.0_f32.sqrt());
        let z = Quaternion { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        assert_eq!(z.norm(), 0.0);
    }
    #[test]
    fn normalized() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = a / 87.0_f32.sqrt();
        assert_eq!(a.normalized(), b);
        let z = Quaternion { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        assert_eq!(z.normalized_checked(), z);
    }
    #[test]
    fn normalize() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let a_normalized = a.normalized();
        let mut b = a;
        b.normalize();
        assert_eq!(b, a_normalized);
        let z = Quaternion { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        let mut y = z;
        y.normalize_checked();
        assert_eq!(z, y);
    }
    #[test]
    fn abs() {
        let a = Quaternion { w: -2.0, x: 3.0, y: -5.0, z: -7.0 };
        assert_eq!(a.abs(), Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 });
    }
    #[test]
    fn abs_in_place() {
        let a = Quaternion { w: -2.0, x: -3.0, y: 5.0, z: 7.0 };
        let mut b = a;
        b.abs_in_place();
        assert_eq!(b, a.abs());
    }
    #[test]
    fn clamp() {
        let a = Quaternion { w: -5.0, x: -2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.clamp(-1.0, 4.0), Quaternion { w: -1.0, x: -1.0, y: 3.0, z: 4.0 });
    }
    #[test]
    fn clamp_in_place() {
        let a = Quaternion { w: -5.0, x: -2.0, y: 3.0, z: 5.0 };
        let mut b = a;
        b.clamp_in_place(-1.0, 4.0);
        assert_eq!(b, a.clamp(-1.0, 4.0));
    }
    #[test]
    fn test_integration_step() {
        use approx::assert_abs_diff_eq;

        let mut q = Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
        let q_dot = Quaternion { w: 0.1, x: 0.2, y: 0.3, z: 0.4 };
        let dt = 0.01;

        // This line uses SIMD Add, Mul, and AddAssign
        q += q_dot * dt;

        assert_abs_diff_eq!(q.w, 1.001, epsilon = 1e-6);
        assert_abs_diff_eq!(q.x, 0.002, epsilon = 1e-6);
    }
}
