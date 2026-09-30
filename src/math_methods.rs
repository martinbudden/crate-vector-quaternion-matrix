#![allow(clippy::inline_always)]

// The form x.fn() is called method call syntax.
// The form fn(x) is called function call syntax.

// List of mathematical methods available in std
// x.sqrt()
// x.sin_cos()
// x.sin(), x.cos(), x.tan()
// x.asin(), x.acos(), x.atan(), x.atan2()
// x.ceil(), x.floor(), x.round(), x.trunc(), x.fract()
// x.exp(), x.exp2(), x.exp_m1()
// x.ln(), x.log2(), x.log10(), x.log()
// x.powf(), x.powi()
// x.ln_1p()
// x.hypot()

// Of those, the following are provided by num_traits:
// x.ceil(), x.floor(), x.round(), x.trunc(), x.fract()
//
// This crate provides:
// x.sqrt()
// x.sin_cos()
// x.sin(), x.cos(), x.tan()
// x.asin(), x.acos(), x.atan(), x.atan2()
// x.exp(), x.exp2()
// x.ln(), x.log2(), x.log10(), x.log()
// x.powf()
// and additionally x.sqrt_reciprocal()
//
// So the following are unprovided
// x.exp_m1()
// x.ln_1p()
// x.hypot()

/// `no_std` implementations of `std` math functions in method call syntax<br>
/// eg `x.sin()`, `x.cos()` etc.<br><br>
pub trait MathMethods: Sized {
    #[must_use]
    fn sin_cos(self) -> (Self, Self);
    #[must_use]
    fn sin(self) -> Self;
    #[must_use]
    fn cos(self) -> Self;
    #[must_use]
    fn tan(self) -> Self;
    #[must_use]
    fn asin(self) -> Self;
    #[must_use]
    fn acos(self) -> Self;
    #[must_use]
    fn atan2(self, y: Self) -> Self;
    #[must_use]
    fn atan(self) -> Self;
    #[must_use]
    fn sqrt(self) -> Self;
    #[must_use]
    fn sqrt_reciprocal(self) -> Self;
    #[must_use]
    fn exp(self) -> Self;
    #[must_use]
    fn exp2(self) -> Self;
    #[must_use]
    fn ln(self) -> Self;
    #[must_use]
    fn log2(self) -> Self;
    #[must_use]
    fn log10(self) -> Self;
    #[must_use]
    fn log(self, base: Self) -> Self;
    #[must_use]
    fn powf(self, e: Self) -> Self;
}

// Use the hardware-linked math methods in Standard Library
#[cfg(feature = "std")]
impl MathMethods for f32 {
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        self.sin_cos()
    }
    #[inline(always)]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline(always)]
    fn cos(self) -> Self {
        self.cos()
    }
    #[inline(always)]
    fn tan(self) -> Self {
        self.tan()
    }
    #[inline(always)]
    fn asin(self) -> Self {
        self.asin()
    }
    #[inline(always)]
    fn acos(self) -> Self {
        self.acos()
    }
    #[inline(always)]
    fn atan2(self, y: Self) -> Self {
        self.atan2(y)
    }
    #[inline(always)]
    fn atan(self) -> Self {
        self.atan()
    }
    #[inline(always)]
    fn sqrt(self) -> f32 {
        self.sqrt()
    }
    #[inline(always)]
    fn sqrt_reciprocal(self) -> f32 {
        1.0 / self.sqrt()
    }
    #[inline(always)]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        self.exp2()
    }
    #[inline(always)]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline(always)]
    fn log2(self) -> Self {
        self.log2()
    }
    #[inline(always)]
    fn log10(self) -> Self {
        self.log10()
    }
    #[inline(always)]
    fn log(self, base: Self) -> Self {
        self.log(base)
    }
    #[inline(always)]
    fn powf(self, e: Self) -> Self {
        self.powf(e)
    }
}

#[cfg(feature = "std")]
impl MathMethods for f64 {
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        self.sin_cos()
    }
    #[inline(always)]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline(always)]
    fn cos(self) -> Self {
        self.cos()
    }
    #[inline(always)]
    fn tan(self) -> Self {
        self.tan()
    }
    #[inline(always)]
    fn asin(self) -> Self {
        self.asin()
    }
    #[inline(always)]
    fn acos(self) -> Self {
        self.acos()
    }
    #[inline(always)]
    fn atan2(self, y: Self) -> Self {
        self.atan2(y)
    }
    #[inline(always)]
    fn atan(self) -> Self {
        self.atan()
    }
    #[inline(always)]
    fn sqrt(self) -> f64 {
        self.sqrt()
    }
    #[inline(always)]
    fn sqrt_reciprocal(self) -> f64 {
        1.0 / self.sqrt()
    }
    #[inline(always)]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        self.exp2()
    }
    #[inline(always)]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline(always)]
    fn log2(self) -> Self {
        self.log2()
    }
    #[inline(always)]
    fn log10(self) -> Self {
        self.log10()
    }
    #[inline(always)]
    fn log(self, base: Self) -> Self {
        self.log(base)
    }
    #[inline(always)]
    fn powf(self, e: Self) -> Self {
        self.powf(e)
    }
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
impl MathMethods for f32 {
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        libm::sincosf(self)
    }
    #[inline(always)]
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    #[inline(always)]
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    #[inline(always)]
    fn tan(self) -> Self {
        libm::tanf(self)
    }
    #[inline(always)]
    fn asin(self) -> Self {
        libm::asinf(self)
    }
    #[inline(always)]
    fn acos(self) -> Self {
        libm::acosf(self)
    }
    #[inline(always)]
    // note: atan2(x, y) = x.atan2(y)
    fn atan2(self, y: Self) -> Self {
        libm::atan2f(self, y)
    }
    #[inline(always)]
    fn atan(self) -> Self {
        libm::atanf(self)
    }
    #[inline(always)]
    fn sqrt(self) -> f32 {
        // Use hardware ASM for ARM chips with an FPU
        #[cfg(all(target_arch = "arm", target_feature = "vfp2"))]
        {
            sqrt_reciprocal_f32(self)
        }
        // Fallback for non-ARM, non-FPU target
        #[cfg(not(all(target_arch = "arm", target_feature = "vfp2")))]
        {
            libm::sqrtf(self)
        }
    }
    #[inline(always)]
    fn sqrt_reciprocal(self) -> f32 {
        // Use hardware ASM for ARM chips with an FPU
        #[cfg(all(target_arch = "arm", target_feature = "vfp2"))]
        {
            sqrt_reciprocal_f32(self)
        }

        // Fallback for non-ARM, non-FPU target
        #[cfg(not(all(target_arch = "arm", target_feature = "vfp2")))]
        {
            1.0 / libm::sqrtf(self)
        }
    }
    #[inline(always)]
    fn exp(self) -> Self {
        libm::expf(self)
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        libm::exp2f(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        libm::logf(self)
    }
    #[inline(always)]
    fn log2(self) -> Self {
        libm::log2f(self)
    }
    #[inline(always)]
    fn log10(self) -> Self {
        libm::log10f(self)
    }
    #[inline(always)]
    fn log(self, base: Self) -> Self {
        libm::logf(self) / libm::logf(base)
    }
    #[inline(always)]
    fn powf(self, e: Self) -> Self {
        libm::powf(self, e)
    }
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
impl MathMethods for f64 {
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        libm::sincos(self)
    }
    #[inline(always)]
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline(always)]
    fn cos(self) -> Self {
        libm::cos(self)
    }
    #[inline(always)]
    fn tan(self) -> Self {
        libm::tan(self)
    }
    #[inline(always)]
    fn asin(self) -> Self {
        libm::asin(self)
    }
    #[inline(always)]
    fn acos(self) -> Self {
        libm::acos(self)
    }
    #[inline(always)]
    // note: atan2(x, y) = x.atan2(y)
    fn atan2(self, y: Self) -> Self {
        libm::atan2(self, y)
    }
    #[inline(always)]
    fn atan(self) -> Self {
        libm::atan(self)
    }
    #[inline(always)]
    fn sqrt(self) -> f64 {
        libm::sqrt(self)
    }
    #[inline(always)]
    fn sqrt_reciprocal(self) -> f64 {
        1.0 / libm::sqrt(self)
    }
    #[inline(always)]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        libm::exp2(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline(always)]
    fn log2(self) -> Self {
        libm::log2(self)
    }
    #[inline(always)]
    fn log10(self) -> Self {
        libm::log10(self)
    }
    #[inline(always)]
    fn log(self, base: Self) -> Self {
        libm::log(self) / libm::log(base)
    }
    #[inline(always)]
    fn powf(self, e: Self) -> Self {
        libm::pow(self, e)
    }
}

#[cfg(all(not(feature = "std"), not(feature = "libm")))]
impl MathMethods for f32 {
    fn sin_cos(self) -> (Self, Self) {
        super::math_approximations::sin_cos_approx_f32(self)
    }
    fn sin(self) -> Self {
        super::math_approximations::sin_approx_f32(self)
    }
    fn cos(self) -> Self {
        super::math_approximations::cos_approx_f32(self)
    }
    fn tan(self) -> Self {
        super::math_approximations::tan_approx_f32(self)
    }
    fn asin(self) -> Self {
        super::math_approximations::asin_approx_f32(self)
    }
    fn acos(self) -> Self {
        super::math_approximations::acos_approx_f32(self)
    }
    // note: atan2(x, y) = x.atan2(y)
    fn atan2(self, y: Self) -> Self {
        super::math_approximations::atan2_approx_f32(self, y)
    }
    fn atan(self) -> Self {
        super::math_approximations::atan2_approx_f32(self, 1.0)
    }
    fn sqrt(self) -> f32 {
        super::sqrt_approximations::sqrt_f32(self)
    }
    fn sqrt_reciprocal(self) -> f32 {
        super::sqrt_approximations::sqrt_reciprocal_f32(self)
    }
    #[inline(always)]
    fn exp(self) -> Self {
        super::math_approximations::exp_approx_f32(self)
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        super::math_approximations::exp2_approx_f32(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        super::math_approximations::ln_approx_f32(self)
    }
    #[inline(always)]
    fn log2(self) -> Self {
        super::math_approximations::log2_approx_f32(self)
    }
    #[inline(always)]
    fn log10(self) -> Self {
        super::math_approximations::log10_approx_f32(self)
    }
    #[inline(always)]
    fn log(self, base: Self) -> Self {
        super::math_approximations::log_approx_f32(self, base)
    }
    #[inline(always)]
    fn powf(self, e: Self) -> Self {
        super::math_approximations::powf_approx_f32(self, e)
    }
}

#[cfg(all(not(feature = "std"), not(feature = "libm")))]
impl MathMethods for f64 {
    fn sin_cos(self) -> (Self, Self) {
        super::math_approximations::sin_cos_approx_f64(self)
    }
    fn sin(self) -> Self {
        super::math_approximations::sin_approx_f64(self)
    }
    fn cos(self) -> Self {
        super::math_approximations::cos_approx_f64(self)
    }
    fn tan(self) -> Self {
        super::math_approximations::tan_approx_f64(self)
    }
    fn asin(self) -> Self {
        super::math_approximations::asin_approx_f64(self)
    }
    fn acos(self) -> Self {
        super::math_approximations::acos_approx_f64(self)
    }
    // note: atan2(x, y) = x.atan2(y)
    fn atan2(self, y: Self) -> Self {
        super::math_approximations::atan2_approx_f64(self, y)
    }
    fn atan(self) -> Self {
        super::math_approximations::atan2_approx_f64(self, 1.0)
    }
    fn sqrt(self) -> f64 {
        super::sqrt_approximations::sqrt_f64(self)
    }
    fn sqrt_reciprocal(self) -> f64 {
        super::sqrt_approximations::sqrt_reciprocal_f64(self)
    }
    #[inline(always)]
    fn exp(self) -> Self {
        super::math_approximations::exp_approx_f64(self)
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        super::math_approximations::exp2_approx_f64(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        super::math_approximations::ln_approx_f64(self)
    }
    #[inline(always)]
    fn log2(self) -> Self {
        super::math_approximations::log2_approx_f64(self)
    }
    #[inline(always)]
    fn log10(self) -> Self {
        super::math_approximations::log10_approx_f64(self)
    }
    #[inline(always)]
    fn log(self, base: Self) -> Self {
        super::math_approximations::log_approx_f64(self, base)
    }
    #[inline(always)]
    fn powf(self, e: Self) -> Self {
        super::math_approximations::powf_approx_f64(self, e)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn sqrt_reciprocal() {
        assert!(4.0_f32.sqrt_reciprocal().abs() - 0.5 < 6e-5);
        assert!(4.0_f64.sqrt_reciprocal().abs() - 0.5 < 6e-5);
    }
    #[cfg(feature = "libm")]
    #[test]
    fn sqrt() {
        assert_eq!(0.0_f32.sqrt(), libm::sqrtf(0.0));
        assert_eq!(4.0_f32.sqrt(), libm::sqrtf(4.0));
    }
}
