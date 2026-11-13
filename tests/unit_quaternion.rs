use vqm::{Quaternionf32, UnitQuaternionf32};

#[cfg(test)]
mod tests {
    use super::*;
    //use approx::assert_abs_diff_eq;
    //const EPSILON: f32 = 1e-5;

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
}
