use vector_quaternion_matrix::{Vector2d, Vector2df32, Vector2df64};

const _: () = assert!(core::mem::size_of::<Vector2df32>() == 8);
const _: () = assert!(core::mem::align_of::<Vector2df32>() == 8);

const _: () = assert!(core::mem::size_of::<Vector2df64>() == 16);
const _: () = assert!(core::mem::align_of::<Vector2df64>() == 8);

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<Vector2d<f32>>();
    }
    #[test]
    fn default() {
        let a: Vector2df32 = Vector2df32::default();
        assert_eq!(a, Vector2d { x: 0.0, y: 0.0 });
    }
    #[test]
    fn zero() {
        use num_traits::zero;
        //let z: Vector2df32 = Vector2df32::zero();
        let _z: Vector2df32 = zero();
        //assert!(z.is_zero());
    }
    #[test]
    fn test_neg() {
        let v = Vector2d { x: 1.0, y: -2.0 };
        let neg_v = -v;
        assert_eq!(neg_v.x, -1.0);
        assert_eq!(neg_v.y, 2.0);
    }
    #[test]
    fn neg() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        assert_eq!(-a, Vector2d { x: -2.0, y: -3.0 });

        let b = -a;
        assert_eq!(b, Vector2d { x: -2.0, y: -3.0 });
    }
    #[test]
    fn add() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let b = Vector2d { x: 7.0, y: 11.0 };
        assert_eq!(a + b, Vector2d { x: 9.0, y: 14.0 });
    }
    #[test]
    fn add_assign() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let b = Vector2d { x: 7.0, y: 11.0 };
        let mut c = a;
        c += b;
        assert_eq!(c, Vector2d { x: 9.0, y: 14.0 });
    }
    #[test]
    fn sub() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let b = Vector2d { x: 7.0, y: 11.0 };
        let c = a - b;
        assert_eq!(c, Vector2d { x: -5.0, y: -8.0 });
    }
    #[test]
    fn sub_assign() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let b = Vector2d { x: 7.0, y: 11.0 };
        let mut c = a;
        c -= b;
        assert_eq!(c, Vector2d { x: -5.0, y: -8.0 });
    }
    #[test]
    fn mul() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        assert_eq!(a * 2.0, Vector2d { x: 4.0, y: 6.0 });
        assert_eq!(2.0 * a, Vector2d { x: 4.0, y: 6.0 });
    }
    #[test]
    fn mul_assign() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let mut b = a;
        b *= 2.0;
        assert_eq!(b, Vector2d { x: 4.0, y: 6.0 });
    }
    #[test]
    fn div() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        assert_eq!(a / 2.0, Vector2d { x: 1.0, y: 1.5 });
    }
    #[test]
    fn div_assign() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let mut b = a;
        b /= 2.0;
        assert_eq!(b, Vector2d { x: 1.0, y: 1.5 });
    }
    #[test]
    fn new() {
        let a = Vector2d::new(2.0, 3.0);
        assert_eq!(a, Vector2d { x: 2.0, y: 3.0 });
        let b = Vector2d::from((2.0, 3.0));
        assert_eq!(a, b);

        use num_traits::zero;
        let _z: Vector2df32 = zero();
        //assert!(z.is_zero());

        let c: Vector2df32 = (2.0, 3.0).into();
        assert_eq!(a, c);
        let d = Vector2d::from((2.0, 3.0));
        assert_eq!(a, d);
        let e: Vector2df32 = [2.0, 3.0].into();
        assert_eq!(a, e);
        let f = Vector2df32::from([2.0, 3.0]);
        assert_eq!(a, f);

        let h = <[f32; 2]>::from(a);
        assert_eq!([2.0, 3.0], h);
        let i: [f32; 2] = a.into();
        assert_eq!([2.0, 3.0], i);
    }
    #[test]
    fn dot() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let b = Vector2d { x: 7.0, y: 11.0 };
        assert_eq!(a.dot(a), 13.0);
        assert_eq!(a.dot(b), 47.0);
        assert_eq!(b.dot(a), 47.0);
        assert_eq!(b.dot(b), 170.0);
    }
    #[test]
    fn norm_squared() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        assert_eq!(a.norm_squared(), 13.0);
    }
    #[test]
    fn norm() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        assert_eq!(a.norm(), 13.0_f32.sqrt());
        let z = Vector2d { x: 0.0, y: 0.0 };
        assert_eq!(z.norm(), 0.0);
    }
    #[test]
    fn normalized() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        let b = a / 13.0_f32.sqrt();
        assert_eq!(a.normalized(), b);
        let z = Vector2d { x: 0.0, y: 0.0 };
        assert_eq!(z.normalized_checked(), z);
    }
    #[test]
    fn normalize() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let a_normalized = a.normalized();
        let mut b = a;
        b.normalize();
        assert_eq!(b, a_normalized);
        let z = Vector2df32 { x: 0.0, y: 0.0 };
        let mut y = z;
        y.normalize_checked();
        assert_eq!(z, y);
    }
    #[test]
    fn absolute() {
        let a = Vector2df32 { x: -2.0, y: -3.0 };
        assert_eq!(a.abs(), Vector2d { x: 2.0, y: 3.0 });
    }
    #[test]
    fn abs() {
        let a = Vector2df32 { x: -2.0, y: -3.0 };
        let mut b = a;
        b.abs_mut();
        assert_eq!(b, a.abs());
    }
    #[test]
    fn clamped() {
        let a = Vector2d { x: -2.0, y: 3.0 };
        assert_eq!(a.clamped(-1.0, 4.0), Vector2d { x: -1.0, y: 3.0 });
    }
    #[test]
    fn clamp() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        let mut b = a;
        b.clamp(-1.0, 4.0);
        assert_eq!(b, a.clamped(-1.0, 4.0));
    }
    #[test]
    fn sum() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        assert_eq!(a.sum(), 5.0);
    }
    #[test]
    fn mean() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        assert_eq!(a.mean(), 5.0 / 2.0);
    }
    #[test]
    fn product() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        assert_eq!(a.product(), 6.0);
    }
}
