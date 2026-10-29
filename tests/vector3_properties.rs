#[cfg(test)]
mod property_tests {
    use proptest::{prelude::*, test_runner::FileFailurePersistence};
    use vqm::Vector3f32;

    // Define strategy to generate valid, sane floating-point bounds.
    // Prevent values which, when squared, overflow or underflow.
    fn sane_float() -> impl Strategy<Value = f32> {
        const UNDERFLOW: f32 = 3.7e-10;
        const OVERFLOW: f32 = 1.8e10;
        prop_oneof![UNDERFLOW..OVERFLOW, -OVERFLOW..-UNDERFLOW,]
    }
    fn sane_vector3() -> impl Strategy<Value = Vector3f32> {
        (sane_float(), sane_float(), sane_float()).prop_map(|(x, y, z)| Vector3f32::new(x, y, z))
    }

    proptest! {
        // Run 500 distinct generated inputs per test instead of the default 256
        //#![proptest_config(ProptestConfig::with_cases(1))]
        #![proptest_config(ProptestConfig {
            cases: 500,
            // Fixed: Saves regressions directly to a local project file
            // without searching for src/lib.rs
            failure_persistence: Some(Box::new(FileFailurePersistence::WithSource("proptest-regressions"))),
            .. ProptestConfig::default()
        })]
        #[test]
        fn test_vector_normalization_invariant(v in sane_vector3()) {
            // Discard the test case if the vector is close to zero length
            prop_assume!(v.x.abs() > 1e-10 &&  v.y.abs() > 1e-10 && v.z.abs() > 1e-10);

            let normalized = v.normalize();
            let magnitude_squared = normalized.norm_squared();

            // Property: A normalized vector's magnitude must ALWAYS be approximately 1.0
            prop_assert!((magnitude_squared - 1.0).abs() < 1e-4);
        }
    }
}
