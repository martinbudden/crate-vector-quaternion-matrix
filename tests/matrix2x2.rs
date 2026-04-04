use num_traits::identities::{One, Zero};
use vector_quaternion_matrix::{Matrix2x2, Matrix2x2f32, Vector2d};

// **** Align

const _: () = assert!(core::mem::size_of::<Matrix2x2<f32>>() == 16);
const _: () = assert!(core::mem::align_of::<Matrix2x2<f32>>() == 16);

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<Matrix2x2<f32>>();
    }
    #[test]
    fn default() {
        let a: Matrix2x2<f32> = Matrix2x2f32::default();
        assert_eq!(a, Matrix2x2f32::from([0.0, 0.0, 0.0, 0.0]));
        let z = Matrix2x2f32::zero();
        //let z: Matrix2x2 = zero();
        assert_eq!(a, z);
        assert!(z.is_zero());
        assert!(!z.is_one());
        assert!(z.is_near_zero());

        let i = Matrix2x2f32::one();
        //let i: Matrix2x2 = one();
        assert!(i.is_one());
        assert!(!i.is_zero());
        assert!(i.is_near_identity());
    }
    #[test]
    fn neg() {
        let a = Matrix2x2f32::from([2.0, 3.0, 5.0, 7.0]);
        assert_eq!(-a, Matrix2x2f32::from([-2.0, -3.0, -5.0, -7.0]));

        let b = -a;
        assert_eq!(b, Matrix2x2f32::from([-2.0, -3.0, -5.0, -7.0]));
    }
    #[test]
    fn add() {
        let a = Matrix2x2f32::from([2.0, 3.0, 5.0, 7.0]);
        let b = Matrix2x2f32::from([29.0, 31.0, 37.0, 41.0]);
        let a_plus_b = Matrix2x2f32::from([2.0 + 29.0, 3.0 + 31.0, 5.0 + 37.0, 7.0 + 41.0]);
        assert_eq!(a + b, a_plus_b);
    }
    #[test]
    fn sub() {
        let a = Matrix2x2f32::from([2.0, 3.0, 5.0, 7.0]);
        let b = Matrix2x2f32::from([29.0, 31.0, 37.0, 41.0]);
        let a_minus_b = Matrix2x2::from([2.0 - 29.0, 3.0 - 31.0, 5.0 - 37.0, 7.0 - 41.0]);
        assert_eq!(a - b, a_minus_b);
    }
    #[test]
    fn mul() {
        let a = Matrix2x2f32::from([2.0, 3.0, 5.0, 7.0]);
        let b = Matrix2x2f32::from([29.0, 31.0, 37.0, 41.0]);
        let a_times_b = Matrix2x2::from([
            2.0 * 29.0 + 3.0 * 37.0,
            2.0 * 31.0 + 3.0 * 41.0,
            5.0 * 29.0 + 7.0 * 37.0,
            5.0 * 31.0 + 7.0 * 41.0,
        ]);

        assert_eq!(a * b, a_times_b);
    }
    #[test]
    fn new() {
        let a = Matrix2x2::from([2.0, 3.0, 5.0, 7.0]);
        assert_eq!(a, a);
        let b = Matrix2x2::from([Vector2d { x: 2.0, y: 3.0 }, Vector2d { x: 5.0, y: 7.0 }]);
        assert_eq!(a, b);
        let c = Matrix2x2::from((Vector2d { x: 2.0, y: 3.0 }, Vector2d { x: 5.0, y: 7.0 }));
        assert_eq!(a, c);
        let d: Matrix2x2<f32> = Matrix2x2::new([2.0, 3.0, 5.0, 7.0]);
        assert_eq!(a, d);
    }
    #[test]
    fn from_array() {
        let a = Matrix2x2::from([2.0, 3.0, 5.0, 7.0]);
        assert_eq!(2.0, a[0]);
        assert_eq!(3.0, a[1]);
        assert_eq!(5.0, a[2]);
        assert_eq!(7.0, a[3]);
    }
    #[test]
    fn adjugate() {
        let a: Matrix2x2<f32> = Matrix2x2::from([2.0, 3.0, 5.0, 7.0]);
        let b = a.adjugate();
        let c = a * b;
        let determinant = a.determinant();
        assert!((c / determinant).is_near_identity());
    }
    #[test]
    fn inverse() {
        let a: Matrix2x2<f32> = Matrix2x2::from([2.0, 3.0, 5.0, 7.0]);
        let b = a.inverse();
        let c = a * b;
        assert_eq!(1.0, c[0]);
        assert_eq!(0.0, c[1]);
        assert_eq!(0.0, c[2]);
        assert_eq!(1.0, c[3]);
        //assert!((c[0] - 1.0).abs() < f32::EPSILON*3.0);
        //assert!((c[3] - 1.0).abs() < f32::EPSILON * 3.0);
        //assert!(c[1].abs() < f32::EPSILON);
        //assert!(c[2].abs() < f32::EPSILON);

        //assert!(((c - Matrix2x2::one()) / 5.0).is_near_zero());
    }
}
