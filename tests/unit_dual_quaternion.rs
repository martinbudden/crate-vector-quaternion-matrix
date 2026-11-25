use vqm::{UnitDualQuaternionf32, UnitDualQuaternionf64, UnitQuaternionf32, UnitQuaternionf64, Vector3f32, Vector3f64};

#[cfg(test)]
mod test {
    use super::*;
    const EPSILON: f32 = 0.000_001;

    #[test]
    fn dual_quaternion_normalization() {
        let rotation = UnitQuaternionf64::from_roll_pitch_yaw_radians(0.5, 0.7, 1.2);
        let translation = Vector3f64::new(2.0, 3.0, 5.0);
        let dq = UnitDualQuaternionf64::from_rotation_translation(rotation, translation);
        let dq = dq.renormalize();

        assert!((dq.real().norm_squared() - 1.0).abs() < 1e-12);

        assert!(dq.real().q().dot(dq.dual()).abs() < 1e-12);

        assert!(dq.rotation().is_approx(rotation, 1e-12));
        assert!(dq.translation().is_approx(translation, 1e-12));
    }
    #[test]
    fn dual_quaternion_inverse() {
        let rotation = UnitQuaternionf64::from_roll_pitch_yaw_radians(0.5, 0.7, 1.2);
        let translation = Vector3f64::new(2.0, 3.0, 5.0);
        let dq = UnitDualQuaternionf64::from_rotation_translation(rotation, translation);

        let identity = dq * dq.inverse();
        assert!(identity.is_approx_one(1e-12));
    }
    #[test]
    fn dual_quaternion_composition() {
        let dq1 = UnitDualQuaternionf64::from_rotation_translation(
            UnitQuaternionf64::from_roll_radians(0.5),
            Vector3f64::new(2.0, 3.0, 1.0),
        );

        let dq2 = UnitDualQuaternionf64::from_rotation_translation(
            UnitQuaternionf64::from_pitch_radians(0.7),
            Vector3f64::new(1.0, 2.0, 3.0),
        );

        let combined = dq1 * dq2;

        let v = Vector3f64::new(1.0, 2.0, 3.0);
        let sequential = dq1.rotation().rotate(dq2.rotation().rotate(v) + dq2.translation()) + dq1.translation();
        let result = combined.rotation().rotate(v) + combined.translation();

        assert!(result.is_approx(sequential, 1e-12));
    }
    #[test]
    fn from_rotation_translation_round_trip() {
        let r = UnitQuaternionf32::from_roll_pitch_yaw_degrees(45.0, 60.0, 120.0);
        let t = Vector3f32::new(2.0, 3.0, 5.0);

        let dq = UnitDualQuaternionf32::from_rotation_translation(r, t);

        let r_prime = dq.rotation();
        let t_prime = dq.translation();

        assert!(r_prime.is_approx(r, EPSILON));
        assert!(t_prime.is_approx(t, EPSILON));
    }

    #[test]
    fn rotations() {
        // rotate 90 degrees about X
        let rx = UnitQuaternionf32::from_roll_pitch_yaw_degrees(90.0, 0.0, 0.0);

        // rotate 90 degrees about Y
        let ry = UnitQuaternionf32::from_roll_pitch_yaw_degrees(0.0, 90.0, 0.0);

        // rotate 90 degrees about Z
        let rz = UnitQuaternionf32::from_roll_pitch_yaw_degrees(0.0, 0.0, 90.0);

        let v_zero = Vector3f32 { x: 0.0, y: 0.0, z: 0.0 };

        let dqx = UnitDualQuaternionf32::from_rotation_translation(rx, v_zero);
        let dqy = UnitDualQuaternionf32::from_rotation_translation(ry, v_zero);
        let dqz = UnitDualQuaternionf32::from_rotation_translation(rz, v_zero);

        let dq = dqy * dqz * dqx;

        let r = dq.rotation();
        let t = dq.translation();

        assert_eq!(t, v_zero);

        let rotated = r.rotate(Vector3f32::new(1.0, 0.0, 0.0));
        assert!(rotated.is_approx(Vector3f32::new(0.0, 1.0, 0.0), EPSILON));
    }

    #[test]
    fn translations() {
        let t1 = Vector3f32::new(1.0, 2.0, 3.0);
        let t2 = Vector3f32::new(1.0, -2.0, 0.0);

        let dq1 = UnitDualQuaternionf32::from_rotation_translation(UnitQuaternionf32::identity(), t1);
        let dq2 = UnitDualQuaternionf32::from_rotation_translation(UnitQuaternionf32::identity(), t2);
        let dq3 = dq1 * dq2;

        let r_prime = dq3.rotation();
        let t_prime = dq3.translation();

        let r_expected = UnitQuaternionf32::identity();
        let t_expected = Vector3f32::new(2.0, 0.0, 3.0);

        assert!(t_prime.is_approx(t_expected, EPSILON));

        let rotate1 = r_prime.rotate(Vector3f32::new(1.0, 1.0, 1.0));
        let rotate2 = r_expected.rotate(Vector3f32::new(1.0, 1.0, 1.0));

        assert!(rotate1.is_approx(rotate2, EPSILON));
    }

    #[test]
    fn conjugate() {
        let r = UnitQuaternionf32::from_roll_pitch_yaw_degrees(45.0, 60.0, 120.0);
        let t = Vector3f32::new(2.0, 3.0, 5.0);

        let dq = UnitDualQuaternionf32::from_rotation_translation(r, t);
        let dq_conjugate = dq.conjugate();

        let dq_prime = dq * dq_conjugate;
        let r_prime = dq_prime.rotation();
        let t_prime = dq_prime.translation();

        assert!(t_prime.is_approx_zero(EPSILON));

        assert!(r_prime.is_approx_one(EPSILON));
    }
}
