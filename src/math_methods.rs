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
/// Angles are expressed in radians.
pub trait MathMethods: Sized {
    /// Calculates the sine and cosine of an angle simultaneously.
    /// Returns a tuple containing `(sin(self), cos(self))`.
    #[must_use]
    fn sin_cos(self) -> (Self, Self);
    /// Calculates the sine of an angle, measured in radians.
    #[must_use]
    fn sin(self) -> Self;
    /// Calculates the cosine of an angle, measured in radians.
    #[must_use]
    fn cos(self) -> Self;
    /// Calculates the tangent of an angle, measured in radians.
    #[must_use]
    fn tan(self) -> Self;
    /// Calculates the arcsine of a value.
    #[must_use]
    fn asin(self) -> Self;
    /// Calculates the arccosine of a value.
    #[must_use]
    fn acos(self) -> Self;
    /// Calculates the four-quadrant arctangent of `self / y`.
    /// Returns the angle in radians corresponding to the Cartesian coordinates `(y, self)`.
    #[must_use]
    fn atan2(self, y: Self) -> Self;
    /// Calculates the arctangent of a value.
    #[must_use]
    fn atan(self) -> Self;
    /// Calculates the square root of a value.
    #[must_use]
    fn sqrt(self) -> Self;
    /// Calculates the reciprocal of the square root of a value.
    #[must_use]
    fn sqrt_reciprocal(self) -> Self;
    /// Calculates the exponential function, raising Euler's number to a power.
    #[must_use]
    fn exp(self) -> Self;

    /// Calculates two raised to the power of `self`.
    #[must_use]
    fn exp2(self) -> Self;
    /// Calculates the natural logarithm of a value.
    #[must_use]
    fn ln(self) -> Self;
    /// Calculates the base-2 logarithm of a value.
    #[must_use]
    fn log2(self) -> Self;
    /// Calculates the base-10 logarithm of a value.
    #[must_use]
    fn log10(self) -> Self;
    /// Calculates the logarithm of a value using the specified base.
    #[must_use]
    fn log(self, base: Self) -> Self;
    /// Raises a value to a floating-point power.
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
