use cfg_if::cfg_if;
use num_traits::identities::{One, Zero};
use vqm::{Matrix3x3, Matrix3x3f32, Vector3};

// **** Align ****
cfg_if! {
    if #[cfg(feature = "align")] {
        const _: () = assert!(size_of::<Matrix3x3<f32>>() == 64);
        const _: () = assert!(align_of::<Matrix3x3<f32>>() == 64);
        const _: () = assert!(size_of::<Matrix3x3<f64>>() == 128);
        const _: () = assert!(align_of::<Matrix3x3<f64>>() == 64);
    } else {
        const _: () = assert!(size_of::<Matrix3x3<f32>>() == 36);
        const _: () = assert!(align_of::<Matrix3x3<f32>>() == 4);
        const _: () = assert!(size_of::<Matrix3x3<f64>>() == 72);
        const _: () = assert!(align_of::<Matrix3x3<f64>>() == 8);
    }
}

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
        is_full::<Matrix3x3<f32>>();
        #[cfg(feature = "serde")]
        is_serde::<Matrix3x3<f32>>();
        #[cfg(feature = "storage")]
        is_storage::<Matrix3x3<f32>>();
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default() {
        let a: Matrix3x3<f32> = Matrix3x3f32::default();
        assert_eq!(a, Matrix3x3f32::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        let z = Matrix3x3f32::zero();
        //let z: Matrix3x3 = zero();
        assert_eq!(a, z);
        assert!(z.is_zero());
        assert!(!z.is_one());
        assert!(z.is_near_zero(1e-5));

        let i = Matrix3x3f32::one();
        //let i: Matrix3x3 = one();
        assert!(i.is_one());
        assert!(!i.is_zero());
        assert!(i.is_near_identity(1e-5));
    }
    #[test]
    fn m3x3_neg() {
        let a = Matrix3x3f32::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);

        assert_eq!(-a, Matrix3x3f32::new([-2.0, -3.0, -5.0, -7.0, -11.0, -13.0, -17.0, -19.0, -23.0]));

        let b = -a;
        assert_eq!(b, Matrix3x3f32::new([-2.0, -3.0, -5.0, -7.0, -11.0, -13.0, -17.0, -19.0, -23.0]));
    }
    #[test]
    fn m3x3_add() {
        let a = Matrix3x3f32::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = Matrix3x3f32::new([29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 61.0]);
        let a_plus_b = Matrix3x3f32::new([
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
    fn m3x3_sub() {
        let a = Matrix3x3f32::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = Matrix3x3f32::new([29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 61.0]);
        let a_minus_b = Matrix3x3::new([
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
    fn m3x3_mul() {
        let a = Matrix3x3f32::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = Matrix3x3f32::new([29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 61.0]);
        let a_times_b = Matrix3x3::new([
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
    fn m3x3_new() {
        let a = Matrix3x3::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = Matrix3x3::from_rows([
            Vector3 { x: 2.0, y: 3.0, z: 5.0 },
            Vector3 { x: 7.0, y: 11.0, z: 13.0 },
            Vector3 { x: 17.0, y: 19.0, z: 23.0 },
        ]);
        assert_eq!(a, b);
        let c: Matrix3x3<f32> = Matrix3x3::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        assert_eq!(a, c);
    }
    #[test]
    fn m3x3_from_array() {
        let a = Matrix3x3f32::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        assert_eq!(2.0, a[Matrix3x3f32::M11]);
        assert_eq!(3.0, a[Matrix3x3f32::M12]);
        assert_eq!(5.0, a[Matrix3x3f32::M13]);
        assert_eq!(7.0, a[Matrix3x3f32::M21]);
        assert_eq!(11.0, a[Matrix3x3f32::M22]);
        assert_eq!(13.0, a[Matrix3x3f32::M23]);
        assert_eq!(17.0, a[Matrix3x3f32::M31]);
        assert_eq!(19.0, a[Matrix3x3f32::M32]);
        assert_eq!(23.0, a[Matrix3x3f32::M33]);
    }
    #[test]
    fn m3x3_determinant() {
        let a: Matrix3x3<f32> = Matrix3x3::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let determinant = a.determinant();
        assert_eq!(-78.0, determinant);
    }
    #[test]
    fn m3x3_adjugate() {
        let a: Matrix3x3<f32> = Matrix3x3::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let (b, determinant) = a.adjugate();
        assert_eq!(b, Matrix3x3f32::new([6.0, 26.0, -16.0, 60.0, -39.0, 9.0, -54.0, 13.0, 1.0]));
        let c = a * b;
        assert!((c / determinant).is_near_identity(1e-5));
    }
    #[test]
    fn m3x3_inverse() {
        let a: Matrix3x3<f32> = Matrix3x3::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]);
        let b = a.inverse();
        let c = a * b;
        assert!((c[0] - 1.0).abs() < f32::EPSILON);
        assert!((c[4] - 1.0).abs() < f32::EPSILON * 3.0);
        assert!((c[8] - 1.0).abs() < f32::EPSILON);
        assert!(c[1].abs() < f32::EPSILON);
        assert!(c[2].abs() < f32::EPSILON);
        assert!(c[3].abs() < f32::EPSILON);
        assert!(c[5].abs() < f32::EPSILON * 5.0);
        assert!(c[6].abs() < f32::EPSILON);
        assert!(c[7].abs() < f32::EPSILON);

        assert!(((c - Matrix3x3::one()) / 5.0).is_near_zero(1e-5));
    }
}
