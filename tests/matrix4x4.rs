use num_traits::identities::{One, Zero};
use vqm::{Matrix4x4, Matrix4x4f32};

// **** Align

#[cfg(any(feature = "align", feature = "simd"))]
const _: () = assert!(size_of::<Matrix4x4<f32>>() == 64 && align_of::<Matrix4x4<f32>>() == 64);

#[cfg(not(any(feature = "align", feature = "simd")))]
const _: () = assert!(size_of::<Matrix4x4<f32>>() == 64 && align_of::<Matrix4x4<f32>>() == 16);

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
        is_full::<Matrix4x4<f32>>();
        #[cfg(feature = "serde")]
        is_serde::<Matrix4x4<f32>>();
        #[cfg(feature = "storage")]
        is_storage::<Matrix4x4<f32>>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default() {
        let a: Matrix4x4<f32> = Matrix4x4f32::default();
        assert_eq!(
            a,
            Matrix4x4f32::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        let z = Matrix4x4f32::zero();
        assert_eq!(a, z);
        assert!(z.is_zero());
        assert!(!z.is_one());
        assert!(z.is_near_zero(1e-5));

        let i = Matrix4x4f32::one();
        assert!(i.is_one());
        assert!(!i.is_zero());
        assert!(i.is_near_identity(1e-5));
    }
}
