#![allow(clippy::excessive_precision)]

/// Math constants for use in generic code, eg T:PI, T:SQRT_2 etc.
pub trait MathConstants {
    const EPSILON: Self;

    const PI: Self; // Archimedes’ constant (π)
    const TAU: Self; // The full circle constant (τ = 2π)
    const E: Self; // Euler’s number (e)
    // Natural logarithms of 2 and 10
    const LN_2: Self;
    const LN_10: Self;
    // Logarithms of e
    const LOG2_E: Self;
    const LOG10_E: Self;
    // Logarithms of 10
    const LOG2_10: Self;
    const LOG10_2: Self;
    // Reciprocals of π
    const FRAC_1_PI: Self;
    const FRAC_2_PI: Self;
    const FRAC_2_SQRT_PI: Self;
    // Fractions of π
    const FRAC_PI_2: Self;
    const FRAC_PI_3: Self;
    const FRAC_PI_4: Self;
    const FRAC_PI_6: Self;
    const FRAC_PI_8: Self;
    // Square roots
    const SQRT_2: Self;
    // const SQRT_3: Self;
    const FRAC_1_SQRT_2: Self;
    // const PHI: Self; // The golden ratio (φ)
    // const EGAMMA: Self; // Euler-Mascheroni constant (γ)

    const FILTER_PT2_CUTOFF_CORRECTION: Self;
    const FILTER_PT3_CUTOFF_CORRECTION: Self;

    const HALF: Self;
    const TWO: Self;
    const THREE: Self;
    const FOUR: Self;
    const FIVE: Self;
    const SIX: Self;
    const SEVEN: Self;
    const EIGHT: Self;
    const NINE: Self;
    const TEN: Self;
    const ELEVEN: Self;
    const TWELVE: Self;
}

impl MathConstants for f32 {
    const EPSILON: Self = f32::EPSILON;

    const PI: Self = core::f32::consts::PI;
    const TAU: Self = core::f32::consts::TAU;
    const E: Self = core::f32::consts::E;
    const LN_2: Self = core::f32::consts::LN_2;
    const LN_10: Self = core::f32::consts::LN_10;
    const LOG2_E: Self = core::f32::consts::LOG2_E;
    const LOG10_E: Self = core::f32::consts::LOG10_E;
    const LOG2_10: Self = core::f32::consts::LOG2_10;
    const LOG10_2: Self = core::f32::consts::LOG10_2;
    const FRAC_1_PI: Self = core::f32::consts::FRAC_1_PI;
    const FRAC_2_PI: Self = core::f32::consts::FRAC_2_PI;
    const FRAC_2_SQRT_PI: Self = core::f32::consts::FRAC_2_SQRT_PI;
    const FRAC_PI_2: Self = core::f32::consts::FRAC_PI_2;
    const FRAC_PI_3: Self = core::f32::consts::FRAC_PI_3;
    const FRAC_PI_4: Self = core::f32::consts::FRAC_PI_4;
    const FRAC_PI_6: Self = core::f32::consts::FRAC_PI_6;
    const FRAC_PI_8: Self = core::f32::consts::FRAC_PI_8;
    const SQRT_2: Self = core::f32::consts::SQRT_2;
    // const SQRT_3: Self = core::f32::consts::SQRT_3;
    const FRAC_1_SQRT_2: Self = core::f32::consts::FRAC_1_SQRT_2;
    // const PHI: Self = core::f32::consts::PHI;
    // const EGAMMA: Self = core::f32::consts::EGAMMA;

    // FilterPt<n> cutoff correction = 1/sqrt(2^(1/n) - 1)
    const FILTER_PT2_CUTOFF_CORRECTION: Self = 1.553_773_974;
    const FILTER_PT3_CUTOFF_CORRECTION: Self = 1.961_459_177;

    const HALF: Self = 0.5;
    const TWO: Self = 2.0;
    const THREE: Self = 3.0;
    const FOUR: Self = 4.0;
    const FIVE: Self = 5.0;
    const SIX: Self = 6.0;
    const SEVEN: Self = 7.0;
    const EIGHT: Self = 8.0;
    const NINE: Self = 9.0;
    const TEN: Self = 10.0;
    const ELEVEN: Self = 11.0;
    const TWELVE: Self = 12.0;
}

impl MathConstants for f64 {
    const EPSILON: Self = f64::EPSILON;

    const PI: Self = core::f64::consts::PI;
    const TAU: Self = core::f64::consts::TAU;
    const E: Self = core::f64::consts::E;
    const LN_2: Self = core::f64::consts::LN_2;
    const LN_10: Self = core::f64::consts::LN_10;
    const LOG2_E: Self = core::f64::consts::LOG2_E;
    const LOG10_E: Self = core::f64::consts::LOG10_E;
    const LOG2_10: Self = core::f64::consts::LOG2_10;
    const LOG10_2: Self = core::f64::consts::LOG10_2;
    const FRAC_1_PI: Self = core::f64::consts::FRAC_1_PI;
    const FRAC_2_PI: Self = core::f64::consts::FRAC_2_PI;
    const FRAC_2_SQRT_PI: Self = core::f64::consts::FRAC_2_SQRT_PI;
    const FRAC_PI_2: Self = core::f64::consts::FRAC_PI_2;
    const FRAC_PI_3: Self = core::f64::consts::FRAC_PI_3;
    const FRAC_PI_4: Self = core::f64::consts::FRAC_PI_4;
    const FRAC_PI_6: Self = core::f64::consts::FRAC_PI_6;
    const FRAC_PI_8: Self = core::f64::consts::FRAC_PI_8;
    const SQRT_2: Self = core::f64::consts::SQRT_2;
    // const SQRT_3: Self = core::f64::consts::SQRT_3;
    const FRAC_1_SQRT_2: Self = core::f64::consts::FRAC_1_SQRT_2;
    // const PHI: Self = core::f64::consts::PHI;
    // const EGAMMA: Self = core::f64::consts::EGAMMA;

    // FilterPt<n> cutoff correction = 1/sqrt(2^(1/n) - 1)
    const FILTER_PT2_CUTOFF_CORRECTION: Self = 1.553_773_974;
    const FILTER_PT3_CUTOFF_CORRECTION: Self = 1.961_459_177;

    const HALF: Self = 0.5;
    const TWO: Self = 2.0;
    const THREE: Self = 3.0;
    const FOUR: Self = 4.0;
    const FIVE: Self = 5.0;
    const SIX: Self = 6.0;
    const SEVEN: Self = 7.0;
    const EIGHT: Self = 8.0;
    const NINE: Self = 9.0;
    const TEN: Self = 10.0;
    const ELEVEN: Self = 11.0;
    const TWELVE: Self = 12.0;
}

#[cfg(test)]
mod tests {
    #![allow(unused)]
    use super::*;

    struct Test<F> {
        t: F,
    }
    impl<F> Test<F>
    where
        F: Copy + MathConstants,
    {
        fn pi() -> F {
            F::PI
        }
        fn half() -> F {
            F::HALF
        }
        fn two() -> F {
            F::TWO
        }
    }
    type Testf32 = Test<f32>;
    type Testf64 = Test<f64>;

    #[test]
    fn f32() {
        assert_eq!(core::f32::consts::PI, Testf32::pi());
        assert_eq!(0.5, Testf32::half());
        assert_eq!(2.0, Testf32::two());
    }
    fn f64() {
        assert_eq!(core::f64::consts::PI, Testf64::pi());
        assert_eq!(0.5, Testf64::half());
        assert_eq!(2.0, Testf64::two());
    }
}
