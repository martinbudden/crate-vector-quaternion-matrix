use vqm::{Quaternion, Quaternionf32};

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
    fn inv() {
        let q = Quaternionf32::new(1.0, 2.0, 3.0, 4.0);

        let Some(q_inv) = q.try_inverse() else {
            panic!("zero quaternion");
        };
        let r = q * q_inv;
        assert_eq!(1.0, r.w);
        assert!(r.x.abs() < 1e-7);
        assert!(r.y.abs() < 1e-7);
        assert!(r.z.abs() < 1e-7);

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
        assert_eq!(-a, Quaternion { w: -2.0, x: -3.0, y: -5.0, z: -7.0 });

        let b = -a;
        assert_eq!(b, Quaternion { w: -2.0, x: -3.0, y: -5.0, z: -7.0 });
    }
    #[test]
    fn quaternion_negation() {
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
        assert_abs_diff_eq!(a.norm(), 87.0_f32.sqrt(), epsilon = 1e-2);
        let z = Quaternion { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
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
        let z = Quaternion { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        assert_eq!(z.normalize(), z);
    }
    #[test]
    fn normalize_unchecked() {
        let a = Quaternion { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
        let a_normalized = a.normalized_unchecked();
        let mut b = a;
        b.normalize_unchecked_in_place();
        assert_eq!(b, a_normalized);
        let z = Quaternion { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
        let mut y = z;
        y.normalize_in_place();
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
    fn clamped() {
        let a = Quaternion { w: -5.0, x: -2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.clamp(-1.0, 4.0), Quaternion { w: -1.0, x: -1.0, y: 3.0, z: 4.0 });
    }
    #[test]
    fn clamp() {
        let a = Quaternion { w: -5.0, x: -2.0, y: 3.0, z: 5.0 };
        let mut b = a;
        b.clamp_in_place(-1.0, 4.0);
        assert_eq!(b, a.clamp(-1.0, 4.0));
    }
    #[test]
    fn integration_step() {
        use approx::assert_abs_diff_eq;

        let mut q = Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
        let q_dot = Quaternion { w: 0.1, x: 0.2, y: 0.3, z: 0.4 };
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
}
