use vqm::{Vector4f32, Vector4f64};

const _: () = assert!(size_of::<Vector4f32>() == 16);
const _: () = assert!(align_of::<Vector4f32>() == 16);

const _: () = assert!(size_of::<Vector4f64>() == 32);
const _: () = assert!(align_of::<Vector4f64>() == 16);

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;

    #[test]
    fn default() {
        let a: Vector4f32 = Vector4f32::default();
        assert_eq!(Vector4f32::zero(), a);
    }
    #[test]
    fn zero() {
        use num_traits::{Zero, zero};
        let z: Vector4f32 = zero();
        assert!(z.is_zero());
    }
}
