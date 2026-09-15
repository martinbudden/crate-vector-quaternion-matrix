use num_traits::identities::{One, Zero};
use vqm::{Matrix2x2, Matrix2x2f32, Vector2};

// **** Align

const _: () = assert!(size_of::<Matrix2x2<f32>>() == 16);
const _: () = assert!(align_of::<Matrix2x2<f32>>() == 16);

const _: () = assert!(size_of::<Matrix2x2<f64>>() == 32);
const _: () = assert!(align_of::<Matrix2x2<f64>>() == 16);

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
        is_full::<Matrix2x2<f32>>();
        #[cfg(feature = "serde")]
        is_serde::<Matrix2x2<f32>>();
        #[cfg(feature = "storage")]
        is_storage::<Matrix2x2<f32>>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m2x2_default() {
        let a: Matrix2x2<f32> = Matrix2x2f32::default();
        assert_eq!(a, Matrix2x2f32::new([0.0, 0.0, 0.0, 0.0]));
        let z = Matrix2x2f32::zero();
        //let z: Matrix2x2 = zero();
        assert_eq!(a, z);
        assert!(z.is_zero());
        assert!(!z.is_one());
        assert!(z.is_near_zero(1e-5));

        let i = Matrix2x2f32::one();
        //let i: Matrix2x2 = one();
        assert!(i.is_one());
        assert!(!i.is_zero());
        assert!(i.is_near_identity(1e-5));
    }
    #[test]
    fn m2x2_neg() {
        let a = Matrix2x2f32::new([2.0, 3.0, 5.0, 7.0]);
        assert_eq!(-a, Matrix2x2f32::new([-2.0, -3.0, -5.0, -7.0]));

        let b = -a;
        assert_eq!(b, Matrix2x2f32::new([-2.0, -3.0, -5.0, -7.0]));
    }
    #[test]
    fn m2x2_add() {
        let a = Matrix2x2f32::new([2.0, 3.0, 5.0, 7.0]);
        let b = Matrix2x2f32::new([29.0, 31.0, 37.0, 41.0]);
        let a_plus_b = Matrix2x2f32::new([2.0 + 29.0, 3.0 + 31.0, 5.0 + 37.0, 7.0 + 41.0]);
        assert_eq!(a + b, a_plus_b);
    }
    #[test]
    fn m2x2_sub() {
        let a = Matrix2x2f32::new([2.0, 3.0, 5.0, 7.0]);
        let b = Matrix2x2f32::new([29.0, 31.0, 37.0, 41.0]);
        let a_minus_b = Matrix2x2::new([2.0 - 29.0, 3.0 - 31.0, 5.0 - 37.0, 7.0 - 41.0]);
        assert_eq!(a - b, a_minus_b);
    }
    #[test]
    fn m2x2_mul() {
        let a = Matrix2x2f32::new([2.0, 3.0, 5.0, 7.0]);
        let b = Matrix2x2f32::new([29.0, 31.0, 37.0, 41.0]);
        let a_times_b = Matrix2x2::new([
            2.0 * 29.0 + 3.0 * 37.0,
            2.0 * 31.0 + 3.0 * 41.0,
            5.0 * 29.0 + 7.0 * 37.0,
            5.0 * 31.0 + 7.0 * 41.0,
        ]);

        assert_eq!(a * b, a_times_b);
    }
    #[test]
    fn m2x2_new() {
        let a = Matrix2x2::new([2.0, 3.0, 5.0, 7.0]);
        assert_eq!(a, a);
        let b = Matrix2x2::from_rows([Vector2 { x: 2.0, y: 3.0 }, Vector2 { x: 5.0, y: 7.0 }]);
        assert_eq!(a, b);
        let d: Matrix2x2<f32> = Matrix2x2::new([2.0, 3.0, 5.0, 7.0]);
        assert_eq!(a, d);
    }
    #[test]
    fn m2x2_from_array() {
        let a = Matrix2x2f32::new([2.0, 3.0, 5.0, 7.0]);
        assert_eq!(2.0, a[Matrix2x2f32::M11]);
        assert_eq!(3.0, a[Matrix2x2f32::M12]);
        assert_eq!(5.0, a[Matrix2x2f32::M21]);
        assert_eq!(7.0, a[Matrix2x2f32::M22]);
    }
    #[test]
    fn m2x2_transpose() {
        let mut m: Matrix2x2<f32> = Matrix2x2::new([2.0, 3.0, 5.0, 7.0]);
        m.transpose_in_place();
        assert_eq!(Matrix2x2::new([2.0, 5.0, 3.0, 7.0]), m);
        m.transpose_in_place();
        assert_eq!(Matrix2x2::new([2.0, 3.0, 5.0, 7.0]), m);
        m.transpose_in_place().transpose_in_place();
        assert_eq!(Matrix2x2::new([2.0, 3.0, 5.0, 7.0]), m);
        let n = m.transpose();
        assert_eq!(Matrix2x2::new([2.0, 5.0, 3.0, 7.0]), n);
        let p = n.transpose();
        assert_eq!(m, p);
    }
    #[test]
    fn m2x2_adjugate() {
        let a: Matrix2x2<f32> = Matrix2x2::new([2.0, 3.0, 5.0, 7.0]);
        let (b, determinant) = a.adjugate();
        let c = a * b;
        assert!((c / determinant).is_near_identity(1e-5));
    }
    #[test]
    fn m2x2_inverse() {
        let a: Matrix2x2<f32> = Matrix2x2::new([2.0, 3.0, 5.0, 7.0]);
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
