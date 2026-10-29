#![allow(clippy::excessive_precision)]

/// Math constants for use in generic code.<br><br>
pub trait MathConstants {
    /// Constant epsilon, for use in generic code.
    const EPSILON: Self;

    /// Constant radians to degrees conversion factor, for use in generic code.
    const RADIANS_TO_DEGREES: Self;
    /// Constant degrees to radians conversion factor, for use in generic code.
    const DEGREES_TO_RADIANS: Self;

    /// Standard acceleration of earth gravity.
    const G0: Self;
    /// Reciprocal of standard acceleration of earth gravity.
    const G0_RECIPROCAL: Self;

    /// Constant 0.5 for use in generic code.
    const HALF: Self;
    /// Constant 2.0 for use in generic code.
    const TWO: Self;
    /// Constant 3.0 for use in generic code.
    const THREE: Self;
    /// Constant 4.0 for use in generic code.
    const FOUR: Self;
    /// Constant 5.0 for use in generic code.
    const FIVE: Self;
    /// Constant 6.0 for use in generic code.
    const SIX: Self;
    /// Constant 7.0 for use in generic code.
    const SEVEN: Self;
    /// Constant 8.0 for use in generic code.
    const EIGHT: Self;
    /// Constant 9.0 for use in generic code.
    const NINE: Self;
    /// Constant 2.0 for use in generic code.
    const TEN: Self;
    /// Constant 10.0 for use in generic code.
    const ELEVEN: Self;
    /// Constant 11.0 for use in generic code.
    const TWELVE: Self;
    /// Constant 12.0 for use in generic code.
    const SIXTEEN: Self;
    /// Constant 16.0 for use in generic code.
    const TWENTY_FIVE: Self;
    /// Constant 25.0 for use in generic code.
    const SIXTY_FOUR: Self;
    /// Constant 64.0 for use in generic code.
    const EIGHTY_ONE: Self;
    /// Constant 81.0 for use in generic code.
    const TWO_FIFTY_SIX: Self;
    /// Constant 256.0 for use in generic code.
    const SIX_TWENTY_FIVE: Self;
}

impl MathConstants for f32 {
    const EPSILON: Self = f32::EPSILON;

    const RADIANS_TO_DEGREES: f32 = 180.0 / core::f32::consts::PI;
    const DEGREES_TO_RADIANS: f32 = core::f32::consts::PI / 180.0;

    const G0: Self = 9.806_65;
    const G0_RECIPROCAL: Self = 1.0 / 9.806_65;

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
    const SIXTEEN: Self = 16.0;
    const TWENTY_FIVE: Self = 25.0;
    const SIXTY_FOUR: Self = 64.0;
    const EIGHTY_ONE: Self = 81.0;
    const TWO_FIFTY_SIX: Self = 256.0;
    const SIX_TWENTY_FIVE: Self = 625.0;
}

impl MathConstants for f64 {
    const EPSILON: Self = f64::EPSILON;

    const RADIANS_TO_DEGREES: f64 = 180.0 / core::f64::consts::PI;
    const DEGREES_TO_RADIANS: f64 = core::f64::consts::PI / 180.0;

    const G0: Self = 9.806_65;
    const G0_RECIPROCAL: Self = 1.0 / 9.806_65;

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
    const SIXTEEN: Self = 16.0;
    const TWENTY_FIVE: Self = 25.0;
    const SIXTY_FOUR: Self = 64.0;
    const EIGHTY_ONE: Self = 81.0;
    const TWO_FIFTY_SIX: Self = 256.0;
    const SIX_TWENTY_FIVE: Self = 625.0;
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    #[allow(unused)]
    struct Test<F> {
        t: F,
    }
    impl<F> Test<F>
    where
        F: Copy + MathConstants,
    {
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
        assert_eq!(0.5, Testf32::half());
        assert_eq!(2.0, Testf32::two());
    }
    #[test]
    fn f64() {
        assert_eq!(0.5, Testf64::half());
        assert_eq!(2.0, Testf64::two());
    }
}
