use vqm::{Quaternion, Quaternionf32, Vector3f32};

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
    fn from() {
        let a = Quaternionf32::from((0.0, 0.0, 0.0));
        let b = Quaternionf32::from_roll_pitch_yaw_radians(0.0, 0.0, 0.0);
        assert_eq!(a, b);
        let c = Quaternionf32::from((0.0, 0.0));
        let d = Quaternionf32::from_roll_pitch_radians(0.0, 0.0);
        assert_eq!(c, d);
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
    fn cos_tilt() {
        use approx::assert_abs_diff_eq;

        let q = Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
        assert_abs_diff_eq!(q.cos_tilt(), 1.0, epsilon = 1e-6);

        // Quaternion for 30 deg rotation around Y axis: [cos(15), 0, sin(15), 0]
        let angle = 30.0_f32.to_radians();
        let q = Quaternionf32 { w: (angle / 2.0).cos(), x: 0.0, y: (angle / 2.0).sin(), z: 0.0 };
        assert_abs_diff_eq!(q.cos_tilt(), angle.cos(), epsilon = 1e-6);

        // Define asymmetric test orientation angles (in radians)
        let roll = -18.3_f32.to_radians(); // Leaning left
        let pitch = 12.5_f32.to_radians(); // Tilted forward
        let yaw = 45.0_f32.to_radians(); // Swiveled diagonally

        // Calculate the expected true mathematical Z-axis projection.
        // This represents the geometric baseline our code must match.
        let expected_cos_tilt = pitch.cos() * roll.cos();
        let q = Quaternionf32::from_roll_pitch_yaw_radians(roll, pitch, yaw);
        assert_abs_diff_eq!(expected_cos_tilt, q.cos_tilt(), epsilon = 1e-6);
        assert_abs_diff_eq!(roll.cos(), q.cos_roll(), epsilon = 3e-4);
        assert_abs_diff_eq!(pitch.cos(), q.cos_pitch(), epsilon = 3e-4);

        // check independent of yaw
        let q = Quaternionf32::from_roll_pitch_yaw_radians(roll, pitch, 0.0);
        assert_abs_diff_eq!(expected_cos_tilt, q.cos_tilt(), epsilon = 1e-6);

        let q = Quaternionf32::from_roll_pitch_yaw_angles_degrees(120.0, 0.0, 0.0);
        assert_abs_diff_eq!(-0.5, q.cos_tilt(), epsilon = 1e-6);
    }
    #[test]
    fn conversions() {
        use approx::assert_abs_diff_eq;
        let angle_degrees = 65.0f32;

        let q = Quaternionf32::from_roll_degrees(angle_degrees);
        let q1 = Quaternionf32::from_roll_pitch_yaw_degrees(angle_degrees, 0.0, 0.0);
        assert_abs_diff_eq!(q1.w, q.w, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.x, q.x, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.y, q.y, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.z, q.z, epsilon = 4e-4);
        assert_eq!(0.0, q.y);
        assert_eq!(0.0, q.z);
        let a = q.calculate_roll_degrees();
        assert_abs_diff_eq!(angle_degrees, a, epsilon = 5e-3);

        let q = Quaternionf32::from_pitch_degrees(angle_degrees);
        let q1 = Quaternionf32::from_roll_pitch_yaw_degrees(0.0, angle_degrees, 0.0);
        assert_abs_diff_eq!(q1.w, q.w, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.x, q.x, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.y, q.y, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.z, q.z, epsilon = 4e-4);
        assert_eq!(0.0, q.x);
        assert_eq!(0.0, q.z);
        let a = q.calculate_pitch_degrees();
        assert_abs_diff_eq!(angle_degrees, a, epsilon = 5e-3);

        let q = Quaternionf32::from_yaw_degrees(angle_degrees);
        let q1 = Quaternionf32::from_roll_pitch_yaw_degrees(0.0, 0.0, angle_degrees);
        assert_abs_diff_eq!(q1.w, q.w, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.x, q.x, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.y, q.y, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.z, q.z, epsilon = 4e-4);
        assert_eq!(0.0, q.x);
        assert_eq!(0.0, q.y);
        let y = q.calculate_yaw_degrees();
        assert_abs_diff_eq!(angle_degrees, y, epsilon = 5e-3);
    }

    #[test]
    fn rotate_identity_leaves_vector_unchanged() {
        let q = Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

        let v = Vector3f32 { x: 1.0, y: 2.0, z: 3.0 };

        let rotated = q.rotate(v);
        assert!(rotated.is_near(v, EPSILON));
    }
    #[test]
    fn rotate_90_degrees_about_x() {
        let s = core::f32::consts::FRAC_1_SQRT_2;

        let q = Quaternionf32 { w: s, x: s, y: 0.0, z: 0.0 };

        let v = Vector3f32 { x: 0.0, y: 1.0, z: 0.0 };

        let expected = Vector3f32 { x: 0.0, y: 0.0, z: 1.0 };
        let rotated = q.rotate(v);
        assert!(rotated.is_near(expected, EPSILON));
    }

    #[test]
    fn rotate_90_degrees_about_y() {
        let s = core::f32::consts::FRAC_1_SQRT_2;

        let q = Quaternionf32 { w: s, x: 0.0, y: s, z: 0.0 };

        let v = Vector3f32 { x: 0.0, y: 0.0, z: 1.0 };

        let expected = Vector3f32 { x: 1.0, y: 0.0, z: 0.0 };

        let rotated = q.rotate(v);
        assert!(rotated.is_near(expected, EPSILON));
    }

    #[test]
    fn rotate_90_degrees_about_z() {
        let s = core::f32::consts::FRAC_1_SQRT_2;

        let q = Quaternionf32 { w: s, x: 0.0, y: 0.0, z: s };

        let v = Vector3f32 { x: 1.0, y: 0.0, z: 0.0 };

        let expected = Vector3f32 { x: 0.0, y: 1.0, z: 0.0 };

        let rotated = q.rotate(v);
        assert!(rotated.is_near(expected, EPSILON));
    }
    #[test]
    fn rotate_arbitrary_vector_about_diagonal_axis() {
        // 120-degree rotation about the (1, 1, 1) axis.
        // This cyclically maps (x, y, z) to (z, x, y).
        let q = Quaternionf32 { w: 0.5, x: 0.5, y: 0.5, z: 0.5 };

        let v = Vector3f32 { x: 1.0, y: 2.0, z: 3.0 };

        let expected = Vector3f32 { x: 3.0, y: 1.0, z: 2.0 };

        let rotated = q.rotate(v);
        assert_eq!(rotated, expected);
        assert!(rotated.is_near(expected, EPSILON));
    }

    #[test]
    fn rotate_180_degrees_about_arbitrary_axis() {
        // 180-degree rotation about the normalized (1, 1, 0) axis.
        let s = core::f32::consts::FRAC_1_SQRT_2;

        let q = Quaternionf32 { w: 0.0, x: s, y: s, z: 0.0 };

        let v = Vector3f32 { x: 1.0, y: 2.0, z: 3.0 };

        let expected = Vector3f32 { x: 2.0, y: 1.0, z: -3.0 };

        let rotated = q.rotate(v);
        assert!(rotated.is_near(expected, EPSILON));
    }

    #[test]
    fn rotate_preserves_perpendicularity() {
        let s = core::f32::consts::FRAC_1_SQRT_2;

        let q = Quaternionf32 { w: s, x: 0.0, y: 0.0, z: s };

        let a = Vector3f32 { x: 1.0, y: 0.0, z: 0.0 };

        let b = Vector3f32 { x: 0.0, y: 0.0, z: 1.0 };

        let a_rotated = q.rotate(a);
        let b_rotated = q.rotate(b);

        let dot = a_rotated.x * b_rotated.x + a_rotated.y * b_rotated.y + a_rotated.z * b_rotated.z;

        assert!(dot.abs() < EPSILON, "Rotated vectors should remain perpendicular, dot product = {}", dot);
    }

    #[test]
    fn rotate_same_result_for_negated_quaternion() {
        // q and -q represent the same rotation.
        let q = Quaternionf32 { w: 0.5, x: 0.5, y: 0.5, z: 0.5 };

        let neg_q = Quaternionf32 { w: -q.w, x: -q.x, y: -q.y, z: -q.z };

        let v = Vector3f32 { x: 1.7, y: -3.2, z: 0.8 };

        let expected = neg_q.rotate(v);
        let rotated = q.rotate(v);
        assert!(rotated.is_near(expected, EPSILON));
    }

    #[test]
    fn rotate_zero_vector_remains_zero() {
        let q = Quaternionf32 { w: 0.5, x: 0.5, y: 0.5, z: 0.5 };

        let zero = Vector3f32 { x: 0.0, y: 0.0, z: 0.0 };

        assert!(q.rotate(zero).is_near(zero, EPSILON));
    }
    #[test]
    fn rotate_preserves_vector_magnitude() {
        let q = Quaternionf32 { w: 0.5, x: 0.5, y: 0.5, z: 0.5 };
        assert!(q.is_normalized(EPSILON));

        let v = Vector3f32 { x: 1.3, y: -2.7, z: 4.2 };
        let original_norm_squared = v.norm_squared();

        let rotated = q.rotate(v);
        let rotated_norm_squared = rotated.norm_squared();

        assert!(
            (original_norm_squared - rotated_norm_squared).abs() < EPSILON,
            "Magnitude changed: original norm_squared = {}, rotated norm_squared = {}",
            original_norm_squared,
            rotated_norm_squared
        );
    }

    #[test]
    fn rotate_preserves_magnitude_for_many_inputs() {
        // Deterministic pseudo-random generator.
        // The same sequence is produced on every test run.
        let mut seed: u32 = 0x1234_5678;

        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);

            // Map to approximately [-1, 1].
            (seed as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
        };

        for _ in 0..1_000 {
            // Generate a quaternion and normalize it.
            let w = next_f32();
            let x = next_f32();
            let y = next_f32();
            let z = next_f32();

            let q = Quaternionf32 { w, x, y, z };

            // Avoid the extremely unlikely zero quaternion.
            if q.norm() < 1e-6 {
                continue;
            }
            let q = q.normalize();

            let v = Vector3f32 { x: next_f32() * 100.0, y: next_f32() * 100.0, z: next_f32() * 100.0 };
            let original_norm_squared = v.norm_squared();

            let rotated = q.rotate(v);
            let rotated_norm_squared = rotated.norm_squared();

            let error = (original_norm_squared - rotated_norm_squared).abs();
            let tolerance = 1e-4 * original_norm_squared.max(1.0);

            assert!(
                error < tolerance,
                "Magnitude changed: original norm squared = {}, rotated norm squared = {}, error = {}",
                original_norm_squared,
                rotated_norm_squared,
                error
            );
        }
    }

    #[test]
    fn rotate_x_90_degrees_from_identity() {
        let mut q = Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

        let s = core::f32::consts::FRAC_1_SQRT_2;

        q.rotate_x(core::f32::consts::FRAC_PI_2);

        assert!((Vector3f32 { x: q.x, y: q.y, z: q.z }).is_near(Vector3f32 { x: s, y: 0.0, z: 0.0 }, EPSILON));
        assert!((q.w - s).abs() < EPSILON);
    }

    #[test]
    fn rotate_y_90_degrees_from_identity() {
        let mut q = Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

        let s = core::f32::consts::FRAC_1_SQRT_2;

        q.rotate_y(core::f32::consts::FRAC_PI_2);

        assert!((Vector3f32 { x: q.x, y: q.y, z: q.z }).is_near(Vector3f32 { x: 0.0, y: s, z: 0.0 }, EPSILON));
        assert!((q.w - s).abs() < EPSILON);
    }

    #[test]
    fn rotate_z_90_degrees_from_identity() {
        let mut q = Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

        let s = core::f32::consts::FRAC_1_SQRT_2;

        q.rotate_z(core::f32::consts::FRAC_PI_2);

        assert!((Vector3f32 { x: q.x, y: q.y, z: q.z }).is_near(Vector3f32 { x: 0.0, y: 0.0, z: s }, EPSILON));
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

        assert!(actual.is_near(expected, EPSILON));
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

        assert!(actual.is_near(expected, EPSILON));
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

        assert!(actual.is_near(expected, EPSILON));
    }
}
