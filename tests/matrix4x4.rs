use vqm::Matrix4x4;

// **** Align

const _: () = assert!(size_of::<Matrix4x4<f32>>() == 64);
const _: () = assert!(align_of::<Matrix4x4<f32>>() == 64);

const _: () = assert!(size_of::<Matrix4x4<f64>>() == 128);
const _: () = assert!(align_of::<Matrix4x4<f64>>() == 64);

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
