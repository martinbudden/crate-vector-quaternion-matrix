use num_traits::identities::{One, Zero};
use vqm::{Matrix4x4, Matrix4x4f32};

// **** Align

#[cfg(any(feature = "align", feature = "simd"))]
const _: () = assert!(size_of::<Matrix4x4<f32>>() == 64 && align_of::<Matrix4x4<f32>>() == 16);

#[cfg(not(any(feature = "align", feature = "simd")))]
const _: () = assert!(size_of::<Matrix4x4<f32>>() == 64 && align_of::<Matrix4x4<f32>>() == 4);

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
        assert!(z.is_approx_zero(1e-5));

        let i = Matrix4x4f32::one();
        assert!(i.is_one());
        assert!(!i.is_zero());
        assert!(i.is_approx_identity(1e-5));
    }
}
