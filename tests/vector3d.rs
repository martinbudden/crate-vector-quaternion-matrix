use cfg_if::cfg_if;
use vector_quaternion_matrix::{Vector3d, Vector3df32, Vector3df64, Vector3di16};

// **** Align ****
cfg_if! {
    if #[cfg(feature = "align")] {
        const _: () = assert!(core::mem::size_of::<Vector3di16>() == 16);
        const _: () = assert!(core::mem::align_of::<Vector3di16>() == 16);
        const _: () = assert!(core::mem::size_of::<Vector3df32>() == 16);
        const _: () = assert!(core::mem::align_of::<Vector3df32>() == 16);
        const _: () = assert!(core::mem::size_of::<Vector3df64>() == 32);
        const _: () = assert!(core::mem::align_of::<Vector3df64>() == 16);
    } else {
        const _: () = assert!(core::mem::size_of::<Vector3di16>() == 8); // would be 6 bytes if aligned on 2 instead of 4
        const _: () = assert!(core::mem::align_of::<Vector3di16>() == 4);
        const _: () = assert!(core::mem::size_of::<Vector3df32>() == 12);
        const _: () = assert!(core::mem::align_of::<Vector3df32>() == 4);
        const _: () = assert!(core::mem::size_of::<Vector3df64>() == 24);
        const _: () = assert!(core::mem::align_of::<Vector3df64>() == 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

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
        #[cfg(feature = "align")]
        assert_eq!(size_of::<Vector3df32>(), 16);
        #[cfg(not(feature = "align"))]
        assert_eq!(size_of::<Vector3df32>(), 12);

        // This ensures the start of every vector is on a 16-byte boundary
        #[cfg(feature = "align")]
        assert_eq!(align_of::<Vector3df32>(), 16);
        #[cfg(not(feature = "align"))]
        assert_eq!(align_of::<Vector3df32>(), 4);
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
    fn test_neg_borrowed() {
        let v = Vector3d { x: 1.0, y: -2.0, z: 3.0 };
        let neg_v = -&v; // Uses &Vector3d<T> impl
        assert_eq!(neg_v.x, -1.0);
        assert_eq!(neg_v.y, 2.0);
        assert_eq!(neg_v.z, -3.0);
        // v is still valid
        assert_eq!(v.x, 1.0);
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
}
