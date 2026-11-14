use vqm::{Quaternion, Quaternionf32, UnitQuaternionf32, Vector3f32};

const _: () = assert!(size_of::<Quaternion<f32>>() == 16);
const _: () = assert!(align_of::<Quaternion<f32>>() == 16);

const _: () = assert!(size_of::<Quaternion<f64>>() == 32);
const _: () = assert!(align_of::<Quaternion<f64>>() == 16);

#[cfg(test)]
mod test_traits {
    use super::*;
    #[cfg(feature = "storage")]
    use sequential_storage::map::PostcardValue;
    #[cfg(feature = "serde")]
    use {
        postcard::experimental::max_size::MaxSize,
        serde::{Deserialize, Serialize},
    };

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}
    #[cfg(feature = "serde")]
    fn is_serde<T: Serialize + MaxSize + for<'a> Deserialize<'a>>() {}
    #[cfg(feature = "storage")]
    fn is_storage<T: for<'a> PostcardValue<'a>>() {}

    #[test]
    fn normal_types() {
        is_full::<Quaternion<f32>>();
        #[cfg(feature = "serde")]
        is_serde::<Quaternion<f32>>();
        #[cfg(feature = "storage")]
        is_storage::<Quaternion<f32>>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    const EPSILON: f32 = 1e-5;

    #[test]
    fn default() {
        use num_traits::{One, Zero};
        let a = Quaternionf32::default();
        assert_eq!(a, Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
        assert!(a.is_one());
        let z = Quaternionf32::zero();
        assert!(z.is_zero());
        let i = Quaternionf32::one();
        assert!(i.is_one());
    }
    #[test]
    fn inv() {
        let q = Quaternionf32::new(1.0, 2.0, 3.0, 4.0);

        let option = q.try_inverse();
        assert!(option.is_some());
        if let Some(q_inv) = option {
            let r = q * q_inv;
            assert_eq!(1.0, r.w);
            assert!(r.x.abs() < 1e-7);
            assert!(r.y.abs() < 1e-7);
            assert!(r.z.abs() < 1e-7);
        }

        let q_inv = q.try_inverse().unwrap_or_default();
        let r = q * q_inv;
        assert_eq!(1.0, r.w);
        assert!(r.x.abs() < 1e-7);
        assert!(r.y.abs() < 1e-7);
        assert!(r.z.abs() < 1e-7);
    }
    #[test]
    fn neg() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(-a, Quaternionf32 { w: -2.0, x: -3.0, y: -5.0, z: -7.0 });

        let b = -a;
        assert_eq!(b, Quaternionf32 { w: -2.0, x: -3.0, y: -5.0, z: -7.0 });
    }
    #[test]
    fn quaternion_negation() {
        let q = Quaternionf32 { x: 0.1, y: -0.2, z: 0.3, w: 0.9 };
        let neg_q = -q;

        assert_abs_diff_eq!(neg_q.x, -0.1, epsilon = 1e-6);
        assert_abs_diff_eq!(neg_q.y, 0.2, epsilon = 1e-6);
        assert_abs_diff_eq!(neg_q.z, -0.3, epsilon = 1e-6);
        assert_abs_diff_eq!(neg_q.w, -0.9, epsilon = 1e-6);
    }
    #[test]
    fn add() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternionf32 { w: 11.0, x: 13.0, y: 17.0, z: 19.0 };
        assert_eq!(a + b, Quaternionf32 { w: 13.0, x: 16.0, y: 22.0, z: 26.0 });
    }
    #[test]
    fn add_assign() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternionf32 { w: 11.0, x: 13.0, y: 17.0, z: 19.0 };
        let mut c = a;
        c += b;
        assert_eq!(c, Quaternionf32 { w: 13.0, x: 16.0, y: 22.0, z: 26.0 });
    }
    #[test]
    fn sub() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternionf32 { w: 11.0, x: 13.0, y: 17.0, z: 23.0 };
        let c = a - b;
        assert_eq!(c, Quaternionf32 { w: -9.0, x: -10.0, y: -12.0, z: -16.0 });
    }
    #[test]
    fn sub_assign() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = Quaternionf32 { w: 11.0, x: 13.0, y: 17.0, z: 23.0 };
        let mut c = a;
        c -= b;
        assert_eq!(c, Quaternionf32 { w: -9.0, x: -10.0, y: -12.0, z: -16.0 });
    }
    #[test]
    fn mul() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(a * 2.0, Quaternionf32 { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
        assert_eq!(2.0 * a, Quaternionf32 { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
    }
    #[test]
    fn mul_assign() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let mut b = a;
        b *= 2.0;
        assert_eq!(b, Quaternionf32 { w: 4.0, x: 6.0, y: 10.0, z: 14.0 });
    }
    #[test]
    fn div() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(a / 2.0, Quaternionf32 { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
    }
    #[test]
    fn div_assign() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let mut b = a;
        b /= 2.0;
        assert_eq!(b, Quaternionf32 { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
    }
    #[test]
    fn new() {
        let a = Quaternion::new(2.0, 3.0, 5.0, 7.0);
        assert_eq!(a, Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 });
    }
    #[test]
    fn norm_squared() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_eq!(a.norm_squared(), 87.0);
    }
    #[test]
    fn norm() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        assert_abs_diff_eq!(a.norm(), 87.0_f32.sqrt(), epsilon = 1e-2);
        let z = Quaternionf32 { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        assert_abs_diff_eq!(z.norm(), 0.0, epsilon = 6e-20);
    }
    #[test]
    fn normalized_unchecked() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let b = a / 87.0_f32.sqrt();
        let n = a.normalized_unchecked();
        assert_abs_diff_eq!(n.w, b.w, epsilon = 1e-4);
        assert_abs_diff_eq!(n.x, b.x, epsilon = 2e-4);
        assert_abs_diff_eq!(n.y, b.y, epsilon = 3e-4);
        assert_abs_diff_eq!(n.z, b.z, epsilon = 3e-4);
        let z = Quaternionf32 { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        assert_eq!(z.normalize(), z);
    }
    #[test]
    fn normalize_unchecked() {
        let a = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let a_normalized = a.normalized_unchecked();
        let mut b = a;
        b.normalize_unchecked_in_place();
        assert_eq!(b, a_normalized);
        let z = Quaternionf32 { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        let mut y = z;
        y.normalize_in_place();
        assert_eq!(z, y);
    }
    #[test]
    fn abs() {
        let a = Quaternionf32 { w: -2.0, x: 3.0, y: -5.0, z: -7.0 };
        assert_eq!(a.abs(), Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 });
    }
    #[test]
    fn abs_in_place() {
        let a = Quaternionf32 { w: -2.0, x: -3.0, y: 5.0, z: 7.0 };
        let mut b = a;
        b.abs_in_place();
        assert_eq!(b, a.abs());
    }
    #[test]
    fn clamped() {
        let a = Quaternionf32 { w: -5.0, x: -2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.clamp(-1.0, 4.0), Quaternionf32 { w: -1.0, x: -1.0, y: 3.0, z: 4.0 });
    }
    #[test]
    fn clamp() {
        let a = Quaternionf32 { w: -5.0, x: -2.0, y: 3.0, z: 5.0 };
        let mut b = a;
        b.clamp_in_place(-1.0, 4.0);
        assert_eq!(b, a.clamp(-1.0, 4.0));
    }
    #[test]
    fn integration_step() {
        use approx::assert_abs_diff_eq;

        let mut q = Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
        let q_dot = Quaternionf32 { w: 0.1, x: 0.2, y: 0.3, z: 0.4 };
        let dt = 0.01;

        // This line uses SIMD Add, Mul, and AddAssign
        q += q_dot * dt;

        assert_abs_diff_eq!(q.w, 1.001, epsilon = 1e-6);
        assert_abs_diff_eq!(q.x, 0.002, epsilon = 1e-6);
    }

    #[test]
    fn rotate_x_90_degrees_from_identity() {
        let mut q = Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

        let s = core::f32::consts::FRAC_1_SQRT_2;

        q.rotate_x(core::f32::consts::FRAC_PI_2);

        assert!((Vector3f32 { x: q.x, y: q.y, z: q.z }).is_approx(Vector3f32 { x: s, y: 0.0, z: 0.0 }, EPSILON));
        assert!((q.w - s).abs() < EPSILON);
    }

    #[test]
    fn rotate_y_90_degrees_from_identity() {
        let mut q = Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

        let s = core::f32::consts::FRAC_1_SQRT_2;

        q.rotate_y(core::f32::consts::FRAC_PI_2);

        assert!((Vector3f32 { x: q.x, y: q.y, z: q.z }).is_approx(Vector3f32 { x: 0.0, y: s, z: 0.0 }, EPSILON));
        assert!((q.w - s).abs() < EPSILON);
    }

    #[test]
    fn rotate_z_90_degrees_from_identity() {
        let mut q = Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

        let s = core::f32::consts::FRAC_1_SQRT_2;

        q.rotate_z(core::f32::consts::FRAC_PI_2);

        assert!((Vector3f32 { x: q.x, y: q.y, z: q.z }).is_approx(Vector3f32 { x: 0.0, y: 0.0, z: s }, EPSILON));
        assert!((q.w - s).abs() < EPSILON);
    }
    #[test]
    fn rotate_x_matches_quaternion_multiplication() {
        let q = Quaternionf32 { w: 0.5, x: 0.5, y: 0.5, z: 0.5 };

        let theta = 0.7_f32;
        let (s, c) = (theta / 2.0).sin_cos();

        let rhs = Quaternionf32 { w: c, x: s, y: 0.0, z: 0.0 };

        let expected = q * rhs;
        let mut actual = q;
        actual.rotate_x(theta);

        assert!(actual.is_approx(expected, EPSILON));
    }

    #[test]
    fn rotate_y_matches_quaternion_multiplication() {
        let q = Quaternionf32 { w: 0.5, x: 0.5, y: 0.5, z: 0.5 };

        let theta = 0.7_f32;
        let (s, c) = (theta / 2.0).sin_cos();

        let rhs = Quaternionf32 { w: c, x: 0.0, y: s, z: 0.0 };

        let expected = q * rhs;
        let mut actual = q;
        actual.rotate_y(theta);

        assert!(actual.is_approx(expected, EPSILON));
    }

    #[test]
    fn rotate_z_matches_quaternion_multiplication() {
        let q = Quaternionf32 { w: 0.5, x: 0.5, y: 0.5, z: 0.5 };

        let theta = 0.7_f32;
        let (s, c) = (theta / 2.0).sin_cos();

        let rhs = Quaternionf32 { w: c, x: 0.0, y: 0.0, z: s };

        let expected = q * rhs;
        let mut actual = q;
        actual.rotate_z(theta);

        assert!(actual.is_approx(expected, EPSILON));
    }
    #[test]
    fn test_non_zero_slerp_midpoint() {
        // Raw values [2.0, 1.0, -1.0, 3.0] divided by magnitude sqrt(15)
        let q_start = UnitQuaternionf32::new(0.5163978, 0.2581989, -0.2581989, 0.7745967);

        // Raw values [-1.0, 4.0, 2.0, 1.0] divided by magnitude sqrt(22)
        let q_end = UnitQuaternionf32::new(-0.2132007, 0.8528029, 0.4264014, 0.2132007);

        // Expected target midpoint (t = 0.5)
        let expected_mid = Quaternionf32::new(0.1986187, 0.7277963, 0.1101863, 0.6470874);

        let actual_mid = q_start.slerp(q_end, 0.5).q();

        // Check every individual element to verify perfect structural matching
        // A delta of 1e-4 is highly rigorous for single-precision float math
        let epsilon = 1e-4;

        assert!(
            (actual_mid.w - expected_mid.w).abs() < epsilon,
            "W element variance! Expected {}, Got {}",
            expected_mid.w,
            actual_mid.w
        );
        assert!(
            (actual_mid.x - expected_mid.x).abs() < epsilon,
            "X element variance! Expected {}, Got {}",
            expected_mid.x,
            actual_mid.x
        );
        assert!(
            (actual_mid.y - expected_mid.y).abs() < epsilon,
            "Y element variance! Expected {}, Got {}",
            expected_mid.y,
            actual_mid.y
        );
        assert!(
            (actual_mid.z - expected_mid.z).abs() < epsilon,
            "Z element variance! Expected {}, Got {}",
            expected_mid.z,
            actual_mid.z
        );

        // Verify it preserved perfect normalization bounds
        let magnitude = actual_mid.norm();
        assert!((magnitude - 1.0).abs() < 1e-5, "Resulting quaternion is not unit length!");
    }
    #[test]
    fn inverse() {
        let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
        let identity = q * q.inverse();
        assert!(identity.is_approx(Quaternionf32::identity(), 3e-8));
    }
    #[test]
    fn conjugate() {
        let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
        assert_eq!(q * q.conjugate(), Quaternionf32::new(87.0, 0.0, 0.0, 0.0));
    }
    #[test]
    fn derivative() {
        let q = Quaternionf32::identity();
        let omega = Vector3f32::new(1.0, 2.0, 3.0);
        assert!(q.derivative(omega).is_approx(q.derivative_world(omega), 1e-10));
    }
    #[test]
    fn axis_rotation_equivalence() {
        let theta = 0.7;

        let mut qx = Quaternionf32::identity();
        qx.rotate_x(theta);
        let qx_expected = Quaternionf32::from_axis_angle_radians(Vector3f32::new(1.0, 0.0, 0.0), theta);
        assert!(qx.is_approx(qx_expected, 3e-8));

        let mut qy = Quaternionf32::identity();
        qy.rotate_y(theta);
        let qx_expected = Quaternionf32::from_axis_angle_radians(Vector3f32::new(0.0, 1.0, 0.0), theta);
        assert!(qy.is_approx(qx_expected, 3e-8));

        let mut qz = Quaternionf32::identity();
        qz.rotate_z(theta);
        let qx_expected = Quaternionf32::from_axis_angle_radians(Vector3f32::new(0.0, 0.0, 1.0), theta);
        assert!(qz.is_approx(qx_expected, 3e-8));
    }
}
