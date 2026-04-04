use cfg_if::cfg_if;
use num_traits::identities::{One, Zero};
use vector_quaternion_matrix::{Matrix3x3, Matrix3x3f32, Vector3d};

// **** Align ****
cfg_if! {
    if #[cfg(feature = "align")] {
        const _: () = assert!(core::mem::size_of::<Matrix3x3<f32>>() == 64);
        const _: () = assert!(core::mem::align_of::<Matrix3x3<f32>>() == 32);
        const _: () = assert!(core::mem::size_of::<Matrix3x3<f64>>() == 96);
        const _: () = assert!(core::mem::align_of::<Matrix3x3<f64>>() == 32);
    } else {
        const _: () = assert!(core::mem::size_of::<Matrix3x3<f32>>() == 12);
        const _: () = assert!(core::mem::align_of::<Matrix3x3<f32>>() == 4);
        const _: () = assert!(core::mem::size_of::<Matrix3x3<f64>>() == 24);
        const _: () = assert!(core::mem::align_of::<Matrix3x3<f64>>() == 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<Matrix3x3<f32>>();
    }
    #[test]
    fn default() {
        let a: Matrix3x3<f32> = Matrix3x3f32::default();
        assert_eq!(a, Matrix3x3f32::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        let z = Matrix3x3f32::zero();
        //let z: Matrix3x3 = zero();
        assert_eq!(a, z);
        assert!(z.is_zero());
        assert!(!z.is_one());
        assert!(z.is_near_zero());

        let i = Matrix3x3f32::one();
        //let i: Matrix3x3 = one();
        assert!(i.is_one());
        assert!(!i.is_zero());
        assert!(i.is_near_identity());
    }
    #[test]
    fn neg() {
        let a = Matrix3x3f32::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);

        assert_eq!(-a, Matrix3x3f32::from([-2.0, -3.0, -5.0, -7.0, -11.0, -13.0, -17.0, -19.0, -23.0]));

        let b = -a;
        assert_eq!(b, Matrix3x3f32::from([-2.0, -3.0, -5.0, -7.0, -11.0, -13.0, -17.0, -19.0, -23.0]));
    }
    #[test]
    fn add() {
        let a = Matrix3x3f32::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = Matrix3x3f32::from([29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 61.0]);
        let a_plus_b = Matrix3x3f32::from([
            2.0 + 29.0,
            3.0 + 31.0,
            5.0 + 37.0,
            7.0 + 41.0,
            11.0 + 43.0,
            13.0 + 47.0,
            17.0 + 53.0,
            19.0 + 59.0,
            23.0 + 61.0,
        ]);
        assert_eq!(a + b, a_plus_b);
    }
    #[test]
    fn sub() {
        let a = Matrix3x3f32::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = Matrix3x3f32::from([29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 61.0]);
        let a_minus_b = Matrix3x3::from([
            2.0 - 29.0,
            3.0 - 31.0,
            5.0 - 37.0,
            7.0 - 41.0,
            11.0 - 43.0,
            13.0 - 47.0,
            17.0 - 53.0,
            19.0 - 59.0,
            23.0 - 61.0,
        ]);
        assert_eq!(a - b, a_minus_b);
    }
    #[test]
    fn mul() {
        let a = Matrix3x3f32::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = Matrix3x3f32::from([29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 61.0]);
        let a_times_b = Matrix3x3::from([
            2.0 * 29.0 + 3.0 * 41.0 + 5.0 * 53.0,
            2.0 * 31.0 + 3.0 * 43.0 + 5.0 * 59.0,
            2.0 * 37.0 + 3.0 * 47.0 + 5.0 * 61.0,
            7.0 * 29.0 + 11.0 * 41.0 + 13.0 * 53.0,
            7.0 * 31.0 + 11.0 * 43.0 + 13.0 * 59.0,
            7.0 * 37.0 + 11.0 * 47.0 + 13.0 * 61.0,
            17.0 * 29.0 + 19.0 * 41.0 + 23.0 * 53.0,
            17.0 * 31.0 + 19.0 * 43.0 + 23.0 * 59.0,
            17.0 * 37.0 + 19.0 * 47.0 + 23.0 * 61.0,
        ]);

        assert_eq!(a * b, a_times_b);
    }
    #[test]
    fn new() {
        let a = Matrix3x3::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = Matrix3x3::from([
            Vector3d { x: 2.0, y: 3.0, z: 5.0 },
            Vector3d { x: 7.0, y: 11.0, z: 13.0 },
            Vector3d { x: 17.0, y: 19.0, z: 23.0 },
        ]);
        assert_eq!(a, b);
        let c = Matrix3x3::from((
            Vector3d { x: 2.0, y: 3.0, z: 5.0 },
            Vector3d { x: 7.0, y: 11.0, z: 13.0 },
            Vector3d { x: 17.0, y: 19.0, z: 23.0 },
        ));
        assert_eq!(a, c);
        let d: Matrix3x3<f32> = Matrix3x3::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        assert_eq!(a, d);
    }
    #[test]
    fn from_array() {
        let a = Matrix3x3::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        assert_eq!(2.0, a[0]);
        assert_eq!(3.0, a[1]);
        assert_eq!(5.0, a[2]);
        assert_eq!(7.0, a[3]);
        assert_eq!(11.0, a[4]);
        assert_eq!(13.0, a[5]);
        assert_eq!(17.0, a[6]);
        assert_eq!(19.0, a[7]);
        assert_eq!(23.0, a[8]);
    }
    #[test]
    fn determinant() {
        let a: Matrix3x3<f32> = Matrix3x3::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let determinant = a.determinant();
        assert_eq!(-78.0, determinant);
    }
    #[test]
    fn adjugate() {
        let a: Matrix3x3<f32> = Matrix3x3::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = a.adjugate();
        assert_eq!(b, Matrix3x3f32::from([6.0, 26.0, -16.0, 60.0, -39.0, 9.0, -54.0, 13.0, 1.0]));
        let c = a * b;
        let determinant = a.determinant();
        assert!((c / determinant).is_near_identity());
    }
    #[test]
    fn inverse() {
        let a: Matrix3x3<f32> = Matrix3x3::from([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = a.inverse();
        let c = a * b;
        assert!((c[0] - 1.0).abs() < f32::EPSILON);
        assert!((c[4] - 1.0).abs() < f32::EPSILON * 3.0);
        assert!((c[8] - 1.0).abs() < f32::EPSILON);
        assert!(c[1].abs() < f32::EPSILON);
        assert!(c[2].abs() < f32::EPSILON);
        assert!(c[3].abs() < f32::EPSILON);
        assert!(c[5].abs() < f32::EPSILON);
        assert!(c[6].abs() < f32::EPSILON);
        assert!(c[7].abs() < f32::EPSILON * 5.0);

        assert!(((c - Matrix3x3::one()) / 5.0).is_near_zero());
    }
}
