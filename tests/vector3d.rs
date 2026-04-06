use cfg_if::cfg_if;
use vector_quaternion_matrix::{Vector3d, Vector3df32, Vector3df64, Vector3di16};

// **** Align ****
cfg_if! {
    if #[cfg(feature = "no_align")] {
        const _: () = assert!(core::mem::size_of::<Vector3di16>() == 8); // would be 6 bytes if aligned on 2 instead of 4
        const _: () = assert!(core::mem::align_of::<Vector3di16>() == 4);
        const _: () = assert!(core::mem::size_of::<Vector3df32>() == 12);
        const _: () = assert!(core::mem::align_of::<Vector3df32>() == 4);
        const _: () = assert!(core::mem::size_of::<Vector3df64>() == 24);
        const _: () = assert!(core::mem::align_of::<Vector3df64>() == 8);
    } else {
        const _: () = assert!(core::mem::size_of::<Vector3di16>() == 16);
        const _: () = assert!(core::mem::align_of::<Vector3di16>() == 16);
        const _: () = assert!(core::mem::size_of::<Vector3df32>() == 16);
        const _: () = assert!(core::mem::align_of::<Vector3df32>() == 16);
        const _: () = assert!(core::mem::size_of::<Vector3df64>() == 32);
        const _: () = assert!(core::mem::align_of::<Vector3df64>() == 16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector3df32;
    use approx::assert_abs_diff_eq;
    use core::mem::{align_of, size_of};
    use vector_quaternion_matrix::Quaternionf32;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<Vector3d<f32>>();
    }
    #[test]
    fn default() {
        let a: Vector3df32 = Vector3d::default();
        assert_eq!(a, Vector3d { x: 0.0, y: 0.0, z: 0.0 });
    }
    #[test]
    fn zero() {
        use num_traits::zero;
        let _z: Vector3df32 = zero();
        //assert_eq!(a, z);
        //assert!(z.is_zero());
    }
    #[test]
    fn test_vector_memory_layout() {
        // A 3-axis f32 vector is 12 bytes.
        // With align(16), the compiler pads it to 16 bytes.
        #[cfg(feature = "no_align")]
        assert_eq!(size_of::<Vector3df32>(), 12);
        #[cfg(not(feature = "no_align"))]
        assert_eq!(size_of::<Vector3df32>(), 16);

        // This ensures the start of every vector is on a 16-byte boundary
        #[cfg(feature = "no_align")]
        assert_eq!(align_of::<Vector3df32>(), 4);
        #[cfg(not(feature = "no_align"))]
        assert_eq!(align_of::<Vector3df32>(), 16);
    }
    #[test]
    fn test_neg_owned() {
        let v = Vector3d { x: 1.0, y: -2.0, z: 3.0 };
        let neg_v = -v;
        assert_eq!(neg_v.x, -1.0);
        assert_eq!(neg_v.y, 2.0);
        assert_eq!(neg_v.z, -3.0);
    }
    #[test]
    fn neg() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(-a, Vector3d { x: -2.0, y: -3.0, z: -5.0 });

        let b = -a;
        assert_eq!(b, Vector3d { x: -2.0, y: -3.0, z: -5.0 });
    }
    #[test]
    fn add() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let b = Vector3d { x: 7.0, y: 11.0, z: 13.0 };
        assert_eq!(a + b, Vector3d { x: 9.0, y: 14.0, z: 18.0 });
    }
    #[test]
    fn add_assign() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let b = Vector3d { x: 7.0, y: 11.0, z: 13.0 };
        let mut c = a;
        c += b;
        assert_eq!(c, Vector3d { x: 9.0, y: 14.0, z: 18.0 });
    }
    #[test]
    fn sub() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let b = Vector3d { x: 7.0, y: 11.0, z: 17.0 };
        let c = a - b;
        assert_eq!(c, Vector3d { x: -5.0, y: -8.0, z: -12.0 });
    }
    #[test]
    fn sub_assign() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let b = Vector3d { x: 7.0, y: 11.0, z: 17.0 };
        let mut c = a;
        c -= b;
        assert_eq!(c, Vector3d { x: -5.0, y: -8.0, z: -12.0 });
    }
    #[test]
    fn mul() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a * 2.0, Vector3d { x: 4.0, y: 6.0, z: 10.0 });
        assert_eq!(2.0 * a, Vector3d { x: 4.0, y: 6.0, z: 10.0 });
    }
    #[test]
    fn mul_assign() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let mut b = a;
        b *= 2.0;
        assert_eq!(b, Vector3d { x: 4.0, y: 6.0, z: 10.0 });
    }
    #[test]
    fn div() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a / 2.0, Vector3d { x: 1.0, y: 1.5, z: 2.5 });
    }
    #[test]
    fn div_assign() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let mut b = a;
        b /= 2.0;
        assert_eq!(b, Vector3d { x: 1.0, y: 1.5, z: 2.5 });
    }
    #[test]
    fn new() {
        let a = Vector3d::new(2.0, 3.0, 5.0);
        assert_eq!(a, Vector3d { x: 2.0, y: 3.0, z: 5.0 });
        let b = Vector3d::from((2.0, 3.0, 5.0));
        assert_eq!(a, b);

        use num_traits::zero;
        let _z: Vector3df32 = zero();
        //assert!(z.is_zero());

        let c: Vector3df32 = (2.0, 3.0, 5.0).into();
        assert_eq!(a, c);
        let d = Vector3d::from((2.0, 3.0, 5.0));
        assert_eq!(a, d);
        let e: Vector3df32 = [2.0, 3.0, 5.0].into();
        assert_eq!(a, e);
        let f = Vector3df32::from([2.0, 3.0, 5.0]);
        assert_eq!(a, f);

        let h = <[f32; 3]>::from(a);
        assert_eq!([2.0, 3.0, 5.0], h);
        let i: [f32; 3] = a.into();
        assert_eq!([2.0, 3.0, 5.0], i);
    }
    #[test]
    fn norm_squared() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.norm_squared(), 38.0);
    }
    #[test]
    fn norm() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.norm(), 38.0_f32.sqrt());
        let z = Vector3d { x: 0.0, y: 0.0, z: 0.0 };
        assert_eq!(z.norm(), 0.0);
    }
    #[test]
    fn normalized() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        let b = a / 38.0_f32.sqrt();
        assert_eq!(a.normalized(), b);
        let z = Vector3d { x: 0.0, y: 0.0, z: 0.0 };
        assert_eq!(z.normalized(), z);
    }
    #[test]
    fn abs() {
        let a = Vector3df32 { x: -2.0, y: -3.0, z: -5.0 };
        assert_eq!(a.abs(), Vector3d { x: 2.0, y: 3.0, z: 5.0 });
    }
    #[test]
    fn abs_in_place() {
        let a = Vector3df32 { x: -2.0, y: -3.0, z: -5.0 };
        let mut b = a;
        b.abs_in_place();
        assert_eq!(b, a.abs());
    }
    #[test]
    fn clamp() {
        let a = Vector3d { x: -2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.clamp(-1.0, 4.0), Vector3d { x: -1.0, y: 3.0, z: 4.0 });
    }
    #[test]
    fn clamp_in_place() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        let mut b = a;
        b.clamp_in_place(-1.0, 4.0);
        assert_eq!(b, a.clamp(-1.0, 4.0));
    }
    #[test]
    fn sum() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.sum(), 10.0);
    }
    #[test]
    fn mean() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.mean(), 10.0 / 3.0);
    }
    #[test]
    fn product() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.product(), 30.0);
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
