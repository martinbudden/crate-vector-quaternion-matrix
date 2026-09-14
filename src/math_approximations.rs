#![allow(unused)]
#![allow(clippy::inline_always)]
#![allow(clippy::excessive_precision)]

use core::ops::Neg;
use num_traits::{Num, float::FloatCore};
// see [Optimized Trigonometric Functions on TI Arm Cores](https://www.ti.com/lit/an/sprad27a/sprad27a.pdf)
// for explanation of range mapping and coefficients
// r (remainder) is in range [-0.5, 0.5] and pre-scaled by 2/PI
trait Sin5Coefficients {
    const SIN_C1: Self;
    const SIN_C3: Self;
    const SIN_C5: Self;
}

trait Cos6Coefficients {
    const COS_C0: Self;
    const COS_C2: Self;
    const COS_C4: Self;
    const COS_C6: Self;
}

trait ATan7Coefficients {
    const ATAN_C1: Self;
    const ATAN_C3: Self;
    const ATAN_C5: Self;
    const ATAN_C7: Self;
}

trait ExpCoefficients {
    const EXP_C0: Self;
    const EXP_C1: Self;
    const EXP_C2: Self;
    const EXP_C3: Self;
    const EXP_C4: Self;
    const EXP_C5: Self;
    const EXP_C6: Self;
}

trait LnCoefficients {
    const LN_C1: Self;
    const LN_C3: Self;
    const LN_C5: Self;
    const LN_C7: Self;
}

// sin4 (5.60E-07): x * (0.9999949932098388671875 + x2*(-0.166601598262786865234375 + x2*8.12153331935405731201171875e-3))
// sin5 (1.80E-09): x * (1 + x2*(-0.166666507720947265625 +x2*(8.331983350217342376708984375e-3 + x2*(-1.94961365195922553539276123046875e-4))))
// cos4 (6.70E-06): 0.999990046024322509765625 + x2*(-0.4997082054615020751953125 + x2*4.03986163437366485595703125e-2))
// cos5 (6.00E-08): 0.999999940395355224609375 + x2*(-0.499998986721038818359375 + x2*(4.1663490235805511474609375e-2 + x2*(-1.385320327244699001312255859375e-3 + x2*2.31450176215730607509613037109375e-5)))
// tan4 (8.00E-05): x * (0.99921381473541259765625 + x2 * (-0.321175038814544677734375 + x2 * (0.146264731884002685546875 + x2 * (-3.8986742496490478515625e-2))))
// tan4 (2.30E-05): x * (0.999970018863677978515625 + x2 * (-0.3317006528377532958984375 + x2 * (0.1852150261402130126953125 + x2 * (-9.1925732791423797607421875e-2 + x2 * 2.386303804814815521240234375e-2))))

impl Sin5Coefficients for f32 {
    const SIN_C1: Self = core::f32::consts::FRAC_PI_2;
    const SIN_C3: Self = -0.645_685_195_92;
    const SIN_C5: Self = 0.077_562_883_496;
}

impl Sin5Coefficients for f64 {
    const SIN_C1: Self = core::f64::consts::FRAC_PI_2;
    const SIN_C3: Self = -0.645_685_195_92;
    const SIN_C5: Self = 0.077_562_883_496;
}

impl Cos6Coefficients for f32 {
    const COS_C0: Self = 1.0;
    const COS_C2: Self = -1.233_697_652_82;
    const COS_C4: Self = 0.253_601_074_22;
    const COS_C6: Self = -0.020_408_373_326;
}

impl Cos6Coefficients for f64 {
    const COS_C0: Self = 1.0;
    const COS_C2: Self = -1.233_697_652_82;
    const COS_C4: Self = 0.253_601_074_22;
    const COS_C6: Self = -0.020_408_373_326;
}

impl ATan7Coefficients for f32 {
    const ATAN_C1: Self = 0.999_213_814_735_412_597_656_25;
    const ATAN_C3: Self = -0.321_175_038_814_544_677_734_375;
    const ATAN_C5: Self = 0.146_264_731_884_002_685_546_875;
    const ATAN_C7: Self = -3.898_674_249_649_047_851_562_5e-2;
}

impl ATan7Coefficients for f64 {
    const ATAN_C1: Self = 0.999_213_814_735_412_597_656_25;
    const ATAN_C3: Self = -0.321_175_038_814_544_677_734_375;
    const ATAN_C5: Self = 0.146_264_731_884_002_685_546_875;
    const ATAN_C7: Self = -3.898_674_249_649_047_851_562_5e-2;
}

impl ExpCoefficients for f32 {
    const EXP_C0: Self = 1.0;
    const EXP_C1: Self = 1.0;
    const EXP_C2: Self = 0.5;
    const EXP_C3: Self = 0.166_666_666_66; // 1/6
    const EXP_C4: Self = 0.041_666_666_66; // 1/24
    const EXP_C5: Self = 0.008_333_333_33; // 1/120
    const EXP_C6: Self = 0.001_388_888_88; // 1/720
}

impl ExpCoefficients for f64 {
    const EXP_C0: Self = 1.0;
    const EXP_C1: Self = 1.0;
    const EXP_C2: Self = 0.5;
    const EXP_C3: Self = 0.166_666_666_66; // 1/6
    const EXP_C4: Self = 0.041_666_666_66; // 1/24
    const EXP_C5: Self = 0.008_333_333_33; // 1/120
    const EXP_C6: Self = 0.001_388_888_88; // 1/720
}

impl LnCoefficients for f32 {
    const LN_C1: Self = 2.0;
    const LN_C3: Self = 2.0 / 3.0;
    const LN_C5: Self = 2.0 / 5.0;
    const LN_C7: Self = 2.0 / 7.0;
}

impl LnCoefficients for f64 {
    const LN_C1: Self = 2.0;
    const LN_C3: Self = 2.0 / 3.0;
    const LN_C5: Self = 2.0 / 5.0;
    const LN_C7: Self = 2.0 / 7.0;
}

#[inline(always)]
fn sin_poly5<T>(r: T) -> T
where
    T: Copy + Num + Sin5Coefficients,
{
    let r2 = r * r;
    r * (T::SIN_C1 + r2 * (T::SIN_C3 + r2 * T::SIN_C5))
}

#[inline(always)]
fn cos_poly6<T>(r: T) -> T
where
    T: Copy + Num + Cos6Coefficients,
{
    let r2 = r * r;
    T::COS_C0 + r2 * (T::COS_C2 + r2 * (T::COS_C4 + r2 * T::COS_C6))
}

#[allow(unused)]
#[inline(always)]
fn atan_poly7<T>(r: T) -> T
where
    T: Copy + Num + ATan7Coefficients,
{
    let r2 = r * r;
    r * (T::ATAN_C1 + r2 * (T::ATAN_C3 + r2 * (T::ATAN_C5 + r2 * T::ATAN_C7)))
}

#[inline(always)]
fn exp_poly7<T>(r: T) -> T
where
    T: Copy + Num + ExpCoefficients,
{
    // e^r = 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5! + r^6/6!
    T::EXP_C0 + r * (T::EXP_C1 + r * (T::EXP_C2 + r * (T::EXP_C3 + r * (T::EXP_C4 + r * (T::EXP_C5 + r * T::EXP_C6)))))
}

#[inline(always)]
fn ln_poly5<T>(r: T) -> T
where
    T: Copy + Num + LnCoefficients,
{
    let r2 = r * r;
    r * (T::LN_C1 + r2 * (T::LN_C3 + r2 * T::LN_C5))
}

#[inline(always)]
fn ln_poly7<T>(r: T) -> T
where
    T: Copy + Num + LnCoefficients,
{
    let r2 = r * r;
    r * (T::LN_C1 + r2 * (T::LN_C3 + r2 * (T::LN_C5 + r2 * T::LN_C7)))
}

// For sin/cos quadrant helper functions:
// 2 least significant bits of q are quadrant index, ie [0, 1, 2, 3].
#[inline]
fn sin_quadrant<T>(r: T, q: i32) -> T
where
    T: Copy + Num + Neg<Output = T> + Sin5Coefficients + Cos6Coefficients,
{
    if q & 1 == 0 {
        // even quadrant: use sin
        let sin = sin_poly5::<T>(r);
        return if q & 2 == 0 { sin } else { -sin };
    }
    // odd quadrant: use cos
    let cos = cos_poly6::<T>(r);
    if q & 2 == 0 { cos } else { -cos }
}

#[inline]
fn cos_quadrant<T>(r: T, q: i32) -> T
where
    T: Copy + Num + Neg<Output = T> + Sin5Coefficients + Cos6Coefficients,
{
    if q & 1 == 0 {
        // even quadrant: use cos
        let cos = cos_poly6::<T>(r);
        return if q & 2 == 0 { cos } else { -cos };
    }
    // odd quadrant: use sin
    let sin = sin_poly5::<T>(r);
    if q & 2 == 0 { -sin } else { sin }
}

#[inline]
fn sin_cos_quadrant<T>(r: T, q: i32) -> (T, T)
where
    T: Copy + Num + Neg<Output = T> + Sin5Coefficients + Cos6Coefficients,
{
    let sin = sin_poly5::<T>(r);
    let cos = cos_poly6::<T>(r);

    // map values according to quadrant
    let sin_cos = if q & 1 == 0 { (sin, cos) } else { (cos, -sin) };

    if q & 2 == 0 { sin_cos } else { (-sin_cos.0, -sin_cos.1) }
}

#[inline(always)]
#[must_use]
pub fn sin_approx_f32(x: f32) -> f32 {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q;
    #[allow(clippy::cast_possible_truncation)]
    sin_quadrant(r, q as i32)
}

#[inline(always)]
#[must_use]
pub fn sin_approx_f64(x: f64) -> f64 {
    let t = x * core::f64::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q;
    #[allow(clippy::cast_possible_truncation)]
    sin_quadrant(r, q as i32)
}

#[inline(always)]
#[must_use]
pub fn cos_approx_f32(x: f32) -> f32 {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    #[allow(clippy::cast_possible_truncation)]
    cos_quadrant(r, q as i32)
}

#[inline(always)]
#[must_use]
pub fn cos_approx_f64(x: f64) -> f64 {
    let t = x * core::f64::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    #[allow(clippy::cast_possible_truncation)]
    cos_quadrant(r, q as i32)
}

#[inline(always)]
#[must_use]
pub fn sin_cos_approx_f32(x: f32) -> (f32, f32) {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    #[allow(clippy::cast_possible_truncation)]
    sin_cos_quadrant(r, q as i32)
}

#[inline(always)]
#[must_use]
pub fn sin_cos_approx_f64(x: f64) -> (f64, f64) {
    let t = x * core::f64::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    #[allow(clippy::cast_possible_truncation)]
    sin_cos_quadrant(r, q as i32)
}

#[inline(always)]
#[must_use]
pub fn tan_approx_f32(x: f32) -> f32 {
    let (sin, cos) = sin_cos_approx_f32(x);
    sin / cos
}

#[inline(always)]
#[must_use]
pub fn tan_approx_f64(x: f64) -> f64 {
    let (sin, cos) = sin_cos_approx_f64(x);
    sin / cos
}

/// Arctangent of y/x (f32).
#[inline(always)]
#[must_use]
pub fn atan2_approx_f32(y: f32, x: f32) -> f32 {
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }

    let abs_y = y.abs();
    let abs_x = x.abs();

    // Octant reduction
    let (ratio, offset, sign) = if abs_y > abs_x {
        // Octant 2: FRAC_PI_2 - atan(x/y)
        (abs_x / abs_y, core::f32::consts::FRAC_PI_2, -1.0)
    } else {
        // Octant 1: 0.0 + atan(y/x)
        (abs_y / abs_x, 0.0, 1.0)
    };

    // Calculate core first-quadrant angle cleanly
    let mut angle = offset + (sign * atan_poly7(ratio));

    // Map back to the correct quadrant based on original signs
    if x < 0.0 {
        angle = core::f32::consts::PI - angle;
    }
    if y < 0.0 {
        angle = -angle;
    }

    angle
}

#[inline(always)]
#[must_use]
pub fn atan2_approx_f64(y: f64, x: f64) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    f64::from(atan2_approx_f32(y as f32, x as f32))
}

#[inline(always)]
#[must_use]
pub fn asin_approx_f32(x: f32) -> f32 {
    atan2_approx_f32(x, super::sqrt_approximations::sqrt_f32(1.0 - x * x))
}

#[inline(always)]
#[must_use]
pub fn asin_approx_f64(x: f64) -> f64 {
    atan2_approx_f64(x, super::sqrt_approximations::sqrt_f64(1.0 - x * x))
}

#[inline(always)]
#[must_use]
pub fn acos_approx_f32(x: f32) -> f32 {
    atan2_approx_f32(super::sqrt_approximations::sqrt_f32(1.0 - x * x), x)
}

#[inline(always)]
#[must_use]
pub fn acos_approx_f64(x: f64) -> f64 {
    atan2_approx_f64(super::sqrt_approximations::sqrt_f64(1.0 - x * x), x)
}

/// Approximates e^x for f32 using range reduction and a Taylor series.
#[inline(always)]
#[must_use]
pub fn exp_approx_f32(x: f32) -> f32 {
    const LN_2: f32 = core::f32::consts::LN_2;
    const FRAC_1_LN_2: f32 = 1.0 / LN_2;

    if x.is_nan() {
        return f32::NAN;
    }
    if x > 88.722_839 {
        return f32::INFINITY;
    }
    if x < -103.27893 {
        return 0.0;
    }

    // Argument reduction: x = k * ln(2) + r
    // We round to the nearest integer to keep the remainder 'r' as small as possible (-0.35 <= r <= 0.35)
    let k = (x * FRAC_1_LN_2).round();
    let r = x - k * LN_2;

    let exp = exp_poly7(r);

    // Correctly scale the calculated sum by 2^k
    #[allow(clippy::cast_possible_truncation)]
    scale_binary_f32(exp, k as i32)
}

#[inline(always)]
#[must_use]
pub fn exp_approx_f64(x: f64) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    f64::from(exp_approx_f32(x as f32))
}

#[inline(always)]
#[must_use]
pub fn exp2_approx_f32(x: f32) -> f32 {
    exp_approx_f32(x * core::f32::consts::LN_2)
}

#[inline(always)]
#[must_use]
pub fn exp2_approx_f64(x: f64) -> f64 {
    exp_approx_f64(x * core::f64::consts::LN_2)
}

/// Approximates ln(x) for f32 using range reduction and a Padé approximation.
#[inline]
#[must_use]
pub fn ln_approx_f32(x: f32) -> f32 {
    if x <= 0.0 || x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }

    // Range reduction: ln(x) = ln(m * 2^k) = ln(m) + k * ln(2)
    let (mut m, mut k) = mantissa_exponent_f32(x);

    // Narrow the range to [0.707, 1.414] for faster convergence
    if m > core::f32::consts::SQRT_2 {
        m *= 0.5;
        k += 1;
    } else if m < core::f32::consts::FRAC_1_SQRT_2 {
        m *= 2.0;
        k -= 1;
    }

    // Padé approximation around 1.0 using z = (m - 1) / (m + 1)
    let z = (m - 1.0) / (m + 1.0);

    let ln_m = ln_poly7(z);

    // ln(m*2^k) = ln(m) + k * ln(2)
    #[allow(clippy::cast_precision_loss)]
    {
        ln_m + (k as f32) * core::f32::consts::LN_2
    }
}

#[inline(always)]
#[must_use]
pub fn ln_approx_f64(x: f64) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    f64::from(ln_approx_f32(x as f32))
}

#[inline(always)]
#[must_use]
pub fn log2_approx_f32(x: f32) -> f32 {
    ln_approx_f32(x) * core::f32::consts::LOG2_E
}

#[inline(always)]
#[must_use]
pub fn log2_approx_f64(x: f64) -> f64 {
    ln_approx_f64(x) * core::f64::consts::LOG2_E
}

#[inline(always)]
#[must_use]
pub fn log10_approx_f32(x: f32) -> f32 {
    ln_approx_f32(x) * core::f32::consts::LOG10_E
}

#[inline(always)]
#[must_use]
pub fn log10_approx_f64(x: f64) -> f64 {
    ln_approx_f64(x) * core::f64::consts::LOG10_E
}

#[inline(always)]
#[must_use]
pub fn log_approx_f32(x: f32, base: f32) -> f32 {
    ln_approx_f32(x) / ln_approx_f32(base)
}

#[inline(always)]
#[must_use]
pub fn log_approx_f64(x: f64, base: f64) -> f64 {
    ln_approx_f64(x) / ln_approx_f64(base)
}

#[inline]
#[must_use]
pub fn powf_approx_f32(base: f32, exponent: f32) -> f32 {
    if exponent == 0.0 {
        return 1.0;
    }
    if base == 0.0 {
        return 0.0;
    }
    if base < 0.0 {
        // If exponent is an integer, the math is valid
        #[allow(clippy::float_cmp)]
        if exponent == exponent.trunc() {
            let result = exp_approx_f32(exponent * ln_approx_f32(base.abs()));
            return if exponent % 2.0 == 0.0 { result } else { -result };
        }
        return f32::NAN;
    }
    exp_approx_f32(exponent * ln_approx_f32(base))
}

#[inline(always)]
#[must_use]
pub fn powf_approx_f64(base: f64, exponent: f64) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    f64::from(powf_approx_f32(base as f32, exponent as f32))
}

/// Takes x and returns (mantissa, exponent) with mantissa in the range [0.5, 1.0) or [-1.0, -0.5) for negative numbers.
#[inline]
fn mantissa_exponent_f32(x: f32) -> (f32, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let bits = x.to_bits();
    #[allow(clippy::cast_possible_wrap)]
    let exp_bits = ((bits >> 23) & 0xFF) as i32;
    if exp_bits == 0 {
        // Subnormal handling
        return mantissa_exponent_f32(x * 8_388_608.0); // scale up by 2^23
    }
    let k = exp_bits - 126;
    let mantissa = (bits & 0x007F_FFFF) | 0x3F00_0000;
    (f32::from_bits(mantissa), k)
}

/// Returns x * 2^k.
#[inline]
fn scale_binary_f32(x: f32, k: i32) -> f32 {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return x;
    }

    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    #[allow(clippy::cast_possible_wrap)]
    let mut exp_bits = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    // Handle subnormal inputs
    if exp_bits == 0 {
        // Scale up to normalize, then adjust k
        return scale_binary_f32(x * 8_388_608.0, k - 23);
    }

    // Add the new exponent scale factor
    exp_bits += k;

    // Handle overflow to Infinity
    if exp_bits >= 255 {
        return f32::from_bits(sign | 0x7F80_0000);
    }

    // Handle underflow to Subnormal / Zero
    if exp_bits <= 0 {
        if exp_bits < -23 {
            return f32::from_bits(sign); // Returns signed zero
        }
        // Shift mantissa to construct a subnormal number
        let implicit_bit = 0x0080_0000;
        let full_mantissa = mantissa | implicit_bit;
        let shift = 1 - exp_bits;
        return f32::from_bits(sign | (full_mantissa >> shift));
    }

    // Reconstruct the normal float
    #[allow(clippy::cast_sign_loss)]
    f32::from_bits(sign | ((exp_bits as u32) << 23) | mantissa)
}

#[cfg(test)]
mod test_logs {
    #![allow(clippy::float_cmp)]
    use super::*; // Brings your custom functions into scope

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a.is_infinite() && b.is_infinite() {
            return a.is_sign_positive() == b.is_sign_positive();
        }
        (a - b).abs() <= epsilon
    }

    #[cfg(feature = "libm")]
    #[test]
    fn test_ln_approx_f32() {
        let epsilon = 1e-20;
        assert!(approx_equal(libm::logf(1.0e-04), ln_approx_f32(1.0e-04), epsilon));
        assert!(approx_equal(libm::logf(1.0), ln_approx_f32(1.0), epsilon));
        assert!(approx_equal(libm::logf(1.0e04), ln_approx_f32(1.0e04), epsilon));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn test_log2_approx_f32() {
        let epsilon = 1e-20;
        assert!(approx_equal(libm::log2f(1.0e-04), log2_approx_f32(1.0e-04), epsilon));
        assert!(approx_equal(libm::log2f(1.0), log2_approx_f32(1.0), epsilon));
        assert!(approx_equal(libm::log2f(1.0e04), log2_approx_f32(1.0e04), epsilon));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn test_log10_approx_f32() {
        let epsilon = 1e-20;
        assert!(approx_equal(libm::log10f(1.0e-04), log10_approx_f32(1.0e-04), epsilon));
        assert!(approx_equal(libm::log10f(1.0), log10_approx_f32(1.0), epsilon));
        assert!(approx_equal(libm::log10f(1.0e04), log10_approx_f32(1.0e04), epsilon));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn test_log_approx_f32() {
        let epsilon = 1e-20;
        let base = 3.0;
        assert!(approx_equal(libm::logf(1.0e-04) / libm::logf(base), log_approx_f32(1.0e-04, base), epsilon));
        assert!(approx_equal(libm::logf(1.0) / libm::logf(base), log_approx_f32(1.0, base), epsilon));
        assert!(approx_equal(libm::logf(1.0e04) / libm::logf(base), log_approx_f32(1.0e04, base), epsilon));

        let base = 5.0;
        assert!(approx_equal(libm::logf(1.0e-04) / libm::logf(base), log_approx_f32(1.0e-04, base), epsilon));
        assert!(approx_equal(libm::logf(1.0) / libm::logf(base), log_approx_f32(1.0, base), epsilon));
        assert!(approx_equal(libm::logf(1.0e04) / libm::logf(base), log_approx_f32(1.0e04, base), epsilon));

        let base = 127.0;
        assert!(approx_equal(libm::logf(1.0e-04) / libm::logf(base), log_approx_f32(1.0e-04, base), epsilon));
        assert!(approx_equal(libm::logf(1.0) / libm::logf(base), log_approx_f32(1.0, base), epsilon));
        assert!(approx_equal(libm::logf(1.0e04) / libm::logf(base), log_approx_f32(1.0e04, base), epsilon));
    }
}
#[cfg(test)]
mod test_exp {
    #![allow(clippy::float_cmp)]
    use super::*; // Brings your custom functions into scope

    /// Helper to check if two floats are approximately equal within a tolerance.
    fn approx_equal(a: f32, b: f32, max_error: f32) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a.is_infinite() && b.is_infinite() {
            return a.is_sign_positive() == b.is_sign_positive();
        }
        (a - b).abs() <= max_error
    }

    #[test]
    fn test_scale_binary_f32() {
        // Test standard powers of 2 multiplication
        assert_eq!(scale_binary_f32(1.0, 3), 8.0); // 1 * 2^3 = 8
        assert_eq!(scale_binary_f32(1.5, 2), 6.0); // 1.5 * 2^2 = 6
        assert_eq!(scale_binary_f32(16.0, -2), 4.0); // 16 * 2^-2 = 4

        // Test sign preservation
        assert_eq!(scale_binary_f32(-2.0, 2), -8.0);

        // Test identity property
        assert_eq!(scale_binary_f32(5.23, 0), 5.23);

        // Test extreme scaling and limits
        assert_eq!(scale_binary_f32(1.0, 150), f32::INFINITY); // Overflows smoothly
        assert_eq!(scale_binary_f32(1.0, -150), 0.0); // Underflows smoothly to 0
    }

    #[test]
    fn test_exp_approx_f32() {
        // Test standard values against hardware implementation
        assert!(approx_equal(1.0, exp_approx_f32(0.0), 1e-8));
        #[cfg(feature = "libm")]
        assert!(approx_equal(libm::expf(-1.0), exp_approx_f32(-1.0), 1e-10));

        // Test slightly larger inputs where scaling matters
        #[cfg(feature = "libm")]
        assert!(approx_equal(libm::expf(4.5), exp_approx_f32(4.5), 5e-5));

        // Test extreme limits
        assert_eq!(exp_approx_f32(90.0), f32::INFINITY);
        assert_eq!(exp_approx_f32(-110.0), 0.0);
    }

    #[cfg(feature = "libm")]
    #[test]
    fn test_ln_approx_f32() {
        // Test standard values
        assert!(approx_equal(0.0, ln_approx_f32(1.0), 1e-20));
        assert!(approx_equal(libm::logf(2.0), ln_approx_f32(2.0), 1e-20));
        assert!(approx_equal(libm::logf(10.0), ln_approx_f32(10.0), 1e-20));
        assert!(approx_equal(libm::logf(0.5), ln_approx_f32(0.5), 1e-20));

        // Test domain safety limits
        assert!(ln_approx_f32(0.0).is_nan());
        assert!(ln_approx_f32(-1.0).is_nan());
    }

    #[test]
    fn test_powf_approx_f32() {
        // Test basic integer powers using your full pipeline
        assert!(approx_equal(powf_approx_f32(2.0, 3.0), 8.0, 0.005)); // 2^3
        assert!(approx_equal(powf_approx_f32(5.0, -1.0), 0.2, 0.005)); // 5^-1

        // Test fractional powers (square roots)
        assert!(approx_equal(powf_approx_f32(9.0, 0.5), 3.0, 0.005)); // √9

        // Test negative bases with integer exponents
        assert!(approx_equal(powf_approx_f32(-2.0, 3.0), -8.0, 0.005)); // (-2)^3
        assert!(approx_equal(powf_approx_f32(-2.0, 4.0), 16.0, 0.01)); // (-2)^4
        //assert_eq!(custom_no_std_powf_f32(-2.0, 4.0), 16.0);    // (-2)^4

        // Test negative base with fractional exponent (should yield NaN)
        assert!(powf_approx_f32(-4.0, 0.5).is_nan());
    }
}
#[cfg(test)]
mod tests {
    #![allow(unused)]
    use super::*;
    use approx::assert_abs_diff_eq;
    macro_rules! assert_near {
        ($left:expr, $right:expr) => {
            approx::assert_abs_diff_eq!($left, $right, epsilon = 4e-6);
        };
    }

    #[cfg(feature = "libm")]
    #[test]
    fn asin() {
        assert_abs_diff_eq!(0.0_f32.asin(), libm::asinf(0.0));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn sin() {
        assert_near!(sin_approx_f32(10.0_f32.to_radians()), libm::sinf(10.0_f32.to_radians()));
        assert_near!(sin_approx_f32(20.0_f32.to_radians()), libm::sinf(20.0_f32.to_radians()));
        assert_near!(sin_approx_f32(30.0_f32.to_radians()), libm::sinf(30.0_f32.to_radians()));
        assert_near!(sin_approx_f32(40.0_f32.to_radians()), libm::sinf(40.0_f32.to_radians()));
        assert_near!(sin_approx_f32(50.0_f32.to_radians()), libm::sinf(50.0_f32.to_radians()));
        assert_near!(sin_approx_f32(60.0_f32.to_radians()), libm::sinf(60.0_f32.to_radians()));
        assert_near!(sin_approx_f32(70.0_f32.to_radians()), libm::sinf(70.0_f32.to_radians()));
        assert_near!(sin_approx_f32(80.0_f32.to_radians()), libm::sinf(80.0_f32.to_radians()));
        assert_near!(sin_approx_f32(90.0_f32.to_radians()), libm::sinf(90.0_f32.to_radians()));
        assert_near!(sin_approx_f32(100.0_f32.to_radians()), libm::sinf(100.0_f32.to_radians()));
        assert_near!(sin_approx_f32(110.0_f32.to_radians()), libm::sinf(110.0_f32.to_radians()));
        assert_near!(sin_approx_f32(120.0_f32.to_radians()), libm::sinf(120.0_f32.to_radians()));
        assert_near!(sin_approx_f32(130.0_f32.to_radians()), libm::sinf(130.0_f32.to_radians()));
        assert_near!(sin_approx_f32(140.0_f32.to_radians()), libm::sinf(140.0_f32.to_radians()));
        assert_near!(sin_approx_f32(150.0_f32.to_radians()), libm::sinf(150.0_f32.to_radians()));
        assert_near!(sin_approx_f32(160.0_f32.to_radians()), libm::sinf(160.0_f32.to_radians()));
        assert_near!(sin_approx_f32(170.0_f32.to_radians()), libm::sinf(170.0_f32.to_radians()));
        assert_near!(sin_approx_f32(180.0_f32.to_radians()), libm::sinf(180.0_f32.to_radians()));
        assert_near!(sin_approx_f32(190.0_f32.to_radians()), libm::sinf(190.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-10.0_f32.to_radians()), libm::sinf(-10.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-20.0_f32.to_radians()), libm::sinf(-20.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-30.0_f32.to_radians()), libm::sinf(-30.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-40.0_f32.to_radians()), libm::sinf(-40.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-50.0_f32.to_radians()), libm::sinf(-50.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-60.0_f32.to_radians()), libm::sinf(-60.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-70.0_f32.to_radians()), libm::sinf(-70.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-80.0_f32.to_radians()), libm::sinf(-80.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-90.0_f32.to_radians()), libm::sinf(-90.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-100.0_f32.to_radians()), libm::sinf(-100.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-110.0_f32.to_radians()), libm::sinf(-110.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-120.0_f32.to_radians()), libm::sinf(-120.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-130.0_f32.to_radians()), libm::sinf(-130.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-140.0_f32.to_radians()), libm::sinf(-140.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-150.0_f32.to_radians()), libm::sinf(-150.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-160.0_f32.to_radians()), libm::sinf(-160.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-170.0_f32.to_radians()), libm::sinf(-170.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-180.0_f32.to_radians()), libm::sinf(-180.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-190.0_f32.to_radians()), libm::sinf(-190.0_f32.to_radians()));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn cos() {
        assert_near!(cos_approx_f32(10.0_f32.to_radians()), libm::cosf(10.0_f32.to_radians()));
        assert_near!(cos_approx_f32(20.0_f32.to_radians()), libm::cosf(20.0_f32.to_radians()));
        assert_near!(cos_approx_f32(30.0_f32.to_radians()), libm::cosf(30.0_f32.to_radians()));
        assert_near!(cos_approx_f32(40.0_f32.to_radians()), libm::cosf(40.0_f32.to_radians()));
        assert_near!(cos_approx_f32(50.0_f32.to_radians()), libm::cosf(50.0_f32.to_radians()));
        assert_near!(cos_approx_f32(60.0_f32.to_radians()), libm::cosf(60.0_f32.to_radians()));
        assert_near!(cos_approx_f32(70.0_f32.to_radians()), libm::cosf(70.0_f32.to_radians()));
        assert_near!(cos_approx_f32(80.0_f32.to_radians()), libm::cosf(80.0_f32.to_radians()));
        assert_near!(cos_approx_f32(90.0_f32.to_radians()), libm::cosf(90.0_f32.to_radians()));
        assert_near!(cos_approx_f32(100.0_f32.to_radians()), libm::cosf(100.0_f32.to_radians()));
        assert_near!(cos_approx_f32(110.0_f32.to_radians()), libm::cosf(110.0_f32.to_radians()));
        assert_near!(cos_approx_f32(120.0_f32.to_radians()), libm::cosf(120.0_f32.to_radians()));
        assert_near!(cos_approx_f32(130.0_f32.to_radians()), libm::cosf(130.0_f32.to_radians()));
        assert_near!(cos_approx_f32(140.0_f32.to_radians()), libm::cosf(140.0_f32.to_radians()));
        assert_near!(cos_approx_f32(150.0_f32.to_radians()), libm::cosf(150.0_f32.to_radians()));
        assert_near!(cos_approx_f32(160.0_f32.to_radians()), libm::cosf(160.0_f32.to_radians()));
        assert_near!(cos_approx_f32(170.0_f32.to_radians()), libm::cosf(170.0_f32.to_radians()));
        assert_near!(cos_approx_f32(180.0_f32.to_radians()), libm::cosf(180.0_f32.to_radians()));
        assert_near!(cos_approx_f32(190.0_f32.to_radians()), libm::cosf(190.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-10.0_f32.to_radians()), libm::cosf(-10.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-20.0_f32.to_radians()), libm::cosf(-20.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-30.0_f32.to_radians()), libm::cosf(-30.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-40.0_f32.to_radians()), libm::cosf(-40.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-50.0_f32.to_radians()), libm::cosf(-50.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-60.0_f32.to_radians()), libm::cosf(-60.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-70.0_f32.to_radians()), libm::cosf(-70.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-80.0_f32.to_radians()), libm::cosf(-80.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-90.0_f32.to_radians()), libm::cosf(-90.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-100.0_f32.to_radians()), libm::cosf(-100.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-110.0_f32.to_radians()), libm::cosf(-110.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-120.0_f32.to_radians()), libm::cosf(-120.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-130.0_f32.to_radians()), libm::cosf(-130.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-140.0_f32.to_radians()), libm::cosf(-140.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-150.0_f32.to_radians()), libm::cosf(-150.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-160.0_f32.to_radians()), libm::cosf(-160.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-170.0_f32.to_radians()), libm::cosf(-170.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-180.0_f32.to_radians()), libm::cosf(-180.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-190.0_f32.to_radians()), libm::cosf(-190.0_f32.to_radians()));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn sin_cos() {
        let (sin, cos) = 0.0_f32.sin_cos();
        assert_near!(sin, libm::sinf(0.0));
        assert_near!(cos, libm::cosf(0.0));

        let (sin, cos) = 10.0_f32.to_radians().sin_cos();
        assert_near!(sin, libm::sinf(10.0_f32.to_radians()));
        assert_near!(cos, libm::cosf(10.0_f32.to_radians()));

        let (sin, cos) = (-10.0_f32).to_radians().sin_cos();
        assert_near!(sin, libm::sinf(-10.0_f32.to_radians()));
        assert_near!(cos, libm::cosf(-10.0_f32.to_radians()));

        let (sin, cos) = 110.0_f32.to_radians().sin_cos();
        assert_near!(sin, libm::sinf(110.0_f32.to_radians()));
        assert_near!(cos, libm::cosf(110.0_f32.to_radians()));

        let (sin, cos) = (-110.0_f32).to_radians().sin_cos();
        assert_near!(sin, libm::sinf(-110.0_f32.to_radians()));
        assert_near!(cos, libm::cosf(-110.0_f32.to_radians()));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn atan2() {
        assert_abs_diff_eq!(0.0_f32.atan2(1.0), 0.0);
        assert_abs_diff_eq!(libm::atan2f(0.0, 1.0), 0.0);

        assert_abs_diff_eq!(1.0_f32.atan2(0.0), libm::atan2f(1.0, 0.0));
        assert_abs_diff_eq!(-1.0_f32.atan2(0.0), libm::atan2f(-1.0, 0.0));

        assert_abs_diff_eq!(0.0_f32.atan2(1.0), libm::atan2f(0.0, 1.0));
        assert_abs_diff_eq!(0.1_f32.atan2(1.0), libm::atan2f(0.1, 1.0));
        assert_abs_diff_eq!(0.5_f32.atan2(1.0), libm::atan2f(0.5, 1.0));
        assert_abs_diff_eq!(1.0_f32.atan2(1.0), libm::atan2f(1.0, 1.0));
        assert_abs_diff_eq!(2.0_f32.atan2(1.0), libm::atan2f(2.0, 1.0));
        assert_abs_diff_eq!(8.0_f32.atan2(1.0), libm::atan2f(8.0, 1.0));
        assert_abs_diff_eq!(1000.0_f32.atan2(0.0), libm::atan2f(1000.0, 0.0));

        assert_abs_diff_eq!(-0.1_f32.atan2(1.0), libm::atan2f(-0.1, 1.0));
        assert_abs_diff_eq!(-0.5_f32.atan2(1.0), libm::atan2f(-0.5, 1.0));
        assert_abs_diff_eq!(-1.0_f32.atan2(1.0), libm::atan2f(-1.0, 1.0));
        assert_abs_diff_eq!(-2.0_f32.atan2(1.0), libm::atan2f(-2.0, 1.0));
        assert_abs_diff_eq!(-8.0_f32.atan2(1.0), libm::atan2f(-8.0, 1.0));
        assert_abs_diff_eq!(-1000.0_f32.atan2(0.0), libm::atan2f(-1000.0, 0.0));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn atan2_approx() {
        assert_abs_diff_eq!(atan2_approx_f32(0.0, 0.0), libm::atan2f(0.0, 0.0), epsilon = 7.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(0.0, 1.0), libm::atan2f(0.0, 1.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(0.1, 1.0), libm::atan2f(0.1, 1.0), epsilon = 7.0e-5); // 0.09966865
        assert_abs_diff_eq!(atan2_approx_f32(0.5, 1.0), libm::atan2f(0.5, 1.0), epsilon = 8.0e-5); // 0.4636476
        assert_abs_diff_eq!(atan2_approx_f32(1.0, 1.0), libm::atan2f(1.0, 1.0), epsilon = 8.2e-5); // 0.7853982, PI/4
        assert_abs_diff_eq!(atan2_approx_f32(2.0, 1.0), libm::atan2f(2.0, 1.0), epsilon = 8.0e-5); // 1.1071488
        assert_abs_diff_eq!(atan2_approx_f32(8.0, 1.0), libm::atan2f(8.0, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(100.0, 1.0), libm::atan2f(100.00, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(1000.0, 1.0), libm::atan2f(1000.0, 1.0), epsilon = 8.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(-0.1, 1.0), libm::atan2f(-0.1, 1.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-0.5, 1.0), libm::atan2f(-0.5, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1.0, 1.0), libm::atan2f(-1.0, 1.0), epsilon = 8.2e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-2.0, 1.0), libm::atan2f(-2.0, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-8.0, 1.0), libm::atan2f(-8.0, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-100.0, 1.0), libm::atan2f(-100.00, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1000.0, 1.0), libm::atan2f(-1000.0, 1.0), epsilon = 8.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(0.0, -1.0), libm::atan2f(0.0, -1.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(0.1, -1.0), libm::atan2f(0.1, -1.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(0.5, -1.0), libm::atan2f(0.5, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(1.0, -1.0), libm::atan2f(1.0, -1.0), epsilon = 8.2e-5);
        assert_abs_diff_eq!(atan2_approx_f32(2.0, -1.0), libm::atan2f(2.0, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(8.0, -1.0), libm::atan2f(8.0, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(100.0, -1.0), libm::atan2f(100.00, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(1000.0, -1.0), libm::atan2f(1000.0, -1.0), epsilon = 8.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(-0.1, -1.0), libm::atan2f(-0.1, -1.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-0.5, -1.0), libm::atan2f(-0.5, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1.0, -1.0), libm::atan2f(-1.0, -1.0), epsilon = 8.2e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-2.0, -1.0), libm::atan2f(-2.0, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-8.0, -1.0), libm::atan2f(-8.0, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-100.0, -1.0), libm::atan2f(-100.00, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1000.0, -1.0), libm::atan2f(-1000.0, -1.0), epsilon = 8.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(0.1, 0.0), libm::atan2f(0.1, 0.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(0.1, 0.0), libm::atan2f(0.1, 0.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(0.5, 0.0), libm::atan2f(0.5, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(1.0, 0.0), libm::atan2f(1.0, 0.0), epsilon = 8.2e-5);
        assert_abs_diff_eq!(atan2_approx_f32(2.0, 0.0), libm::atan2f(2.0, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(8.0, 0.0), libm::atan2f(8.0, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(100.0, 0.0), libm::atan2f(100.00, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(1000.0, 0.0), libm::atan2f(1000.0, 0.0), epsilon = 8.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(-0.1, 0.0), libm::atan2f(-0.1, 0.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-0.1, 0.0), libm::atan2f(-0.1, 0.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-0.5, 0.0), libm::atan2f(-0.5, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1.0, 0.0), libm::atan2f(-1.0, 0.0), epsilon = 8.2e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-2.0, 0.0), libm::atan2f(-2.0, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-8.0, 0.0), libm::atan2f(-8.0, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-100.0, 0.0), libm::atan2f(-100.00, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1000.0, 0.0), libm::atan2f(-1000.0, 0.0), epsilon = 8.0e-5);
    }
}
