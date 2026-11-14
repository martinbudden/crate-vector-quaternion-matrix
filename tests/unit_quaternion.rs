use vqm::{Matrix3x3f32, Quaternionf32, UnitQuaternionf32, Vector3f32};

#[cfg(test)]
mod tests {
    use super::*;
    //use approx::assert_abs_diff_eq;
    const EPSILON: f32 = 1e-5;

    #[test]
    fn from() {
        let a = UnitQuaternionf32::from((0.0, 0.0, 0.0));
        let b = UnitQuaternionf32::from_roll_pitch_yaw_radians(0.0, 0.0, 0.0);
        assert_eq!(a, b);
        let c = UnitQuaternionf32::from((0.0, 0.0));
        let d = UnitQuaternionf32::from_roll_pitch_radians(0.0, 0.0);
        assert_eq!(c, d);
    }

    #[test]
    fn conversions() {
        use approx::assert_abs_diff_eq;
        let angle_degrees = 65.0f32;

        let q = UnitQuaternionf32::from_roll_degrees(angle_degrees);
        let q1 = UnitQuaternionf32::from_roll_pitch_yaw_degrees(angle_degrees, 0.0, 0.0).q();
        assert_abs_diff_eq!(q1.w, q.q().w, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.x, q.q().x, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.y, q.q().y, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.z, q.q().z, epsilon = 4e-4);
        assert_eq!(0.0, q.q().y);
        assert_eq!(0.0, q.q().z);
        let a = q.calculate_roll_degrees();
        assert_abs_diff_eq!(angle_degrees, a, epsilon = 5e-3);

        let q = UnitQuaternionf32::from_pitch_degrees(angle_degrees);
        let q1 = UnitQuaternionf32::from_roll_pitch_yaw_degrees(0.0, angle_degrees, 0.0).q();
        assert_abs_diff_eq!(q1.w, q.q().w, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.x, q.q().x, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.y, q.q().y, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.z, q.q().z, epsilon = 4e-4);
        assert_eq!(0.0, q.q().x);
        assert_eq!(0.0, q.q().z);
        let a = q.calculate_pitch_degrees();
        assert_abs_diff_eq!(angle_degrees, a, epsilon = 5e-3);

        let q = UnitQuaternionf32::from_yaw_degrees(angle_degrees);
        let q1 = UnitQuaternionf32::from_roll_pitch_yaw_degrees(0.0, 0.0, angle_degrees).q();
        assert_abs_diff_eq!(q1.w, q.q().w, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.x, q.q().x, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.y, q.q().y, epsilon = 4e-4);
        assert_abs_diff_eq!(q1.z, q.q().z, epsilon = 4e-4);
        assert_eq!(0.0, q.q().x);
        assert_eq!(0.0, q.q().y);
        let y = q.calculate_yaw_degrees();
        assert_abs_diff_eq!(angle_degrees, y, epsilon = 5e-3);
    }
    #[test]
    fn cos_tilt() {
        use approx::assert_abs_diff_eq;

        let q = UnitQuaternionf32::from_quaternion(Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
        assert_abs_diff_eq!(q.cos_tilt(), 1.0, epsilon = 1e-6);

        // Quaternion for 30 deg rotation around Y axis: [cos(15), 0, sin(15), 0]
        let angle = 30.0_f32.to_radians();
        let q = UnitQuaternionf32::from_quaternion(Quaternionf32 {
            w: (angle / 2.0).cos(),
            x: 0.0,
            y: (angle / 2.0).sin(),
            z: 0.0,
        });
        assert_abs_diff_eq!(q.cos_tilt(), angle.cos(), epsilon = 1e-6);

        // Define asymmetric test orientation angles (in radians)
        let roll = -18.3_f32.to_radians(); // Leaning left
        let pitch = 12.5_f32.to_radians(); // Tilted forward
        let yaw = 45.0_f32.to_radians(); // Swiveled diagonally

        // Calculate the expected true mathematical Z-axis projection.
        // This represents the geometric baseline our code must match.
        let expected_cos_tilt = pitch.cos() * roll.cos();
        let q = UnitQuaternionf32::from_roll_pitch_yaw_radians(roll, pitch, yaw);
        assert_abs_diff_eq!(expected_cos_tilt, q.cos_tilt(), epsilon = 1e-6);
        assert_abs_diff_eq!(roll.cos(), q.cos_roll(), epsilon = 3e-4);
        assert_abs_diff_eq!(pitch.cos(), q.cos_pitch(), epsilon = 3e-4);

        // check independent of yaw
        let q = UnitQuaternionf32::from_roll_pitch_yaw_radians(roll, pitch, 0.0);
        assert_abs_diff_eq!(expected_cos_tilt, q.cos_tilt(), epsilon = 1e-6);

        let q = UnitQuaternionf32::from_roll_pitch_yaw_angles_degrees(120.0, 0.0, 0.0);
        assert_abs_diff_eq!(-0.5, q.cos_tilt(), epsilon = 1e-6);
    }

    #[test]
    fn rotations() {
        let q = UnitQuaternionf32::from_roll_pitch_yaw_radians(0.3, 0.5, 0.7);

        let v = Vector3f32::new(1.0, 2.0, 3.0);

        let rotated1 = q.rotate(v);

        let matrix: Matrix3x3f32 = q.into();
        let rotated2 = matrix * v;

        assert_eq!(rotated1, rotated2);
        assert!(rotated1.is_approx(rotated2, 1e-10));
    }
    #[test]
    fn matrix1() {
        let q1 = UnitQuaternionf32::from_roll_pitch_yaw_radians(0.3, 0.5, 0.7);
        let m: Matrix3x3f32 = q1.into();
        let q2: Quaternionf32 = m.into();
        let q2 = UnitQuaternionf32::from_quaternion(q2);
        assert!(q1.angle_to(q2) < 1e-10);
    }
    #[test]
    fn matrix2() {
        let q1 = UnitQuaternionf32::from_roll_pitch_yaw_radians(0.3, 0.5, 0.7);
        let m1: Matrix3x3f32 = q1.into();
        let q2: Quaternionf32 = m1.into();
        let m2: Matrix3x3f32 = UnitQuaternionf32::from_quaternion(q2).into();
        m1.is_approx(m2, 1e-10);
    }
    #[test]
    fn sign_equivalence() {
        let q = UnitQuaternionf32::from_axis_angle_radians(Vector3f32::new(0.0, 0.0, 1.0), 0.7);
        assert!(q.angle_to(-q) < 1e-12);
    }
    #[test]
    fn rotate_identity_leaves_vector_unchanged() {
        let q = UnitQuaternionf32::new(1.0, 0.0, 0.0, 0.0);

        let v = Vector3f32 { x: 1.0, y: 2.0, z: 3.0 };

        let rotated = q.rotate(v);
        assert!(rotated.is_approx(v, EPSILON));
    }
    #[test]
    fn rotate_90_degrees_about_x() {
        let s = core::f32::consts::FRAC_1_SQRT_2;

        let q = UnitQuaternionf32::new(s, s, 0.0, 0.0);

        let v = Vector3f32 { x: 0.0, y: 1.0, z: 0.0 };

        let expected = Vector3f32 { x: 0.0, y: 0.0, z: 1.0 };
        let rotated = q.rotate(v);
        assert!(rotated.is_approx(expected, EPSILON));
    }

    #[test]
    fn rotate_90_degrees_about_y() {
        let s = core::f32::consts::FRAC_1_SQRT_2;

        let q = UnitQuaternionf32::new(s, 0.0, s, 0.0);

        let v = Vector3f32 { x: 0.0, y: 0.0, z: 1.0 };

        let expected = Vector3f32 { x: 1.0, y: 0.0, z: 0.0 };

        let rotated = q.rotate(v);
        assert!(rotated.is_approx(expected, EPSILON));
    }

    #[test]
    fn rotate_90_degrees_about_z() {
        let s = core::f32::consts::FRAC_1_SQRT_2;

        let q = UnitQuaternionf32::new(s, 0.0, 0.0, s);

        let v = Vector3f32 { x: 1.0, y: 0.0, z: 0.0 };

        let expected = Vector3f32 { x: 0.0, y: 1.0, z: 0.0 };

        let rotated = q.rotate(v);
        assert!(rotated.is_approx(expected, EPSILON));
    }
    #[test]
    fn rotate_arbitrary_vector_about_diagonal_axis() {
        // 120-degree rotation about the (1, 1, 1) axis.
        // This cyclically maps (x, y, z) to (z, x, y).
        let q = UnitQuaternionf32::new(0.5, 0.5, 0.5, 0.5);

        let v = Vector3f32 { x: 1.0, y: 2.0, z: 3.0 };

        let expected = Vector3f32 { x: 3.0, y: 1.0, z: 2.0 };

        let rotated = q.rotate(v);
        assert_eq!(rotated, expected);
        assert!(rotated.is_approx(expected, EPSILON));
    }

    #[test]
    fn rotate_180_degrees_about_arbitrary_axis() {
        // 180-degree rotation about the normalized (1, 1, 0) axis.
        let s = core::f32::consts::FRAC_1_SQRT_2;

        let q = UnitQuaternionf32::new(0.0, s, s, 0.0);

        let v = Vector3f32 { x: 1.0, y: 2.0, z: 3.0 };

        let expected = Vector3f32 { x: 2.0, y: 1.0, z: -3.0 };

        let rotated = q.rotate(v);
        assert!(rotated.is_approx(expected, EPSILON));
    }

    #[test]
    fn rotate_preserves_perpendicularity() {
        let s = core::f32::consts::FRAC_1_SQRT_2;

        let q = UnitQuaternionf32::new(s, 0.0, 0.0, s);

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
        let q = UnitQuaternionf32::new(0.5, 0.5, 0.5, 0.5);

        let neg_q = UnitQuaternionf32::new(-q.q().w, -q.q().x, -q.q().y, -q.q().z);

        let v = Vector3f32 { x: 1.7, y: -3.2, z: 0.8 };

        let expected = neg_q.rotate(v);
        let rotated = q.rotate(v);
        assert!(rotated.is_approx(expected, EPSILON));
    }

    #[test]
    fn rotate_zero_vector_remains_zero() {
        let q = UnitQuaternionf32::new(0.5, 0.5, 0.5, 0.5);

        let zero = Vector3f32 { x: 0.0, y: 0.0, z: 0.0 };

        assert!(q.rotate(zero).is_approx(zero, EPSILON));
    }
    #[test]
    fn rotate_preserves_vector_magnitude() {
        let q = UnitQuaternionf32::new(0.5, 0.5, 0.5, 0.5);

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
            let q = UnitQuaternionf32::from_quaternion(q);

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
}
