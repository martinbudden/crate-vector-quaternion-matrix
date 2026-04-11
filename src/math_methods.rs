#![allow(clippy::excessive_precision)]

use cfg_if::cfg_if;
use num_traits::Float;

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

/// `no_std` implementations of trigonometric functions in method call syntax<br>
/// eg `x.sin()`, `x.cos()` etc.<br><br>
pub trait TrigonometricMethods: Sized {
    fn sin_cos(self) -> (Self, Self);
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan2(self, y: Self) -> Self;
}

cfg_if! {
    if #[cfg(feature = "std")] {
        // Use the hardware-linked math methods in Standard Library
        impl TrigonometricMethods for f32 {
            #[inline(always)]
            fn sin_cos(self) -> (Self, Self) {
                sin_cos(self)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                sin(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                cos(self)
            }
            #[inline(always)]
            fn tan(self) -> Self {
                tan(self)
            }
            #[inline(always)]
            fn asin(self) -> Self {
                asinf(self)
            }
            #[inline(always)]
            fn acos(self) -> Self {
                acosf(self)
            }
            #[inline(always)]
            fn atan2(self, y: Self) -> Self {
                atan2f(y, self)
            }
        }
        impl TrigonometricMethods for f64 {
            #[inline(always)]
            fn sin_cos(self) -> (Self, Self) {
                sin_cos(self)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                sin(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                cos(self)
            }
            #[inline(always)]
            fn tan(self) -> Self {
                tan(self)
            }
            #[inline(always)]
            fn asin(self) -> Self {
                asinf(self)
            }
            #[inline(always)]
            fn acos(self) -> Self {
                acosf(self)
            }
            #[inline(always)]
            fn atan2(self, y: Self) -> Self {
                atan2f(y, self)
            }
        }
    } else if #[cfg(all(not(feature = "std"), feature = "libm"))] {
        impl TrigonometricMethods for f32 {
            #[inline(always)]
            fn sin_cos(self) -> (Self, Self) {
                sin_cos_approx(self)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                sin_approx(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                cos_approx(self)
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
            fn atan2(self, y: Self) -> Self {
                libm::atan2f(y, self)
            }
        }
        impl TrigonometricMethods for f64 {
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
            fn atan2(self, y: Self) -> Self {
                libm::atan2(y, self)
            }
        }
    } else if #[cfg(all(not(feature = "std"), not(feature = "libm")))] {
        impl TrigonometricMethods for f32 {
            fn sin_cos(self) -> (Self, Self) {
                sin_cos_approx(self)
            }
            fn sin(self) -> Self {
                sin_approx(self)
            }
            fn cos(self) -> Self {
                cos_approx(self)
            }
            fn tan(self) -> Self {
                (sin, cos) = sin_cos_approx(self);
                sin / cos
            }
            fn asin(self) -> Self {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
            fn acos(self) -> Self {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
            fn atan2(self, y: Self) -> Self {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
        }
        impl TrigonometricMethods for f64 {
            fn sin_cos(self) -> (Self, Self) {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
            fn sin(self) -> Self {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
            fn cos(self) -> Self {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
            fn tan(self) -> Self {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
            fn asin(self) -> Self {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
            fn acos(self) -> Self {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
            fn atan2(self, y: Self) -> Self {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
        }
    }
}

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

impl Sin5Coefficients for f32 {
    const SIN_C1: Self = core::f32::consts::FRAC_PI_2;
    const SIN_C3: Self = -0.645_685_195_92;
    const SIN_C5: Self = 0.077_562_883_496;
}

impl Cos6Coefficients for f32 {
    const COS_C0: Self = 1.0;
    const COS_C2: Self = -1.233_697_652_82;
    const COS_C4: Self = 0.253_601_074_22;
    const COS_C6: Self = -0.020_408_373_326;
}

impl Sin5Coefficients for f64 {
    const SIN_C1: Self = core::f64::consts::FRAC_PI_2;
    const SIN_C3: Self = -0.645_685_195_92;
    const SIN_C5: Self = 0.077_562_883_496;
}

impl Cos6Coefficients for f64 {
    const COS_C0: Self = 1.0;
    const COS_C2: Self = -1.233_697_652_82;
    const COS_C4: Self = 0.253_601_074_22;
    const COS_C6: Self = -0.020_408_373_326;
}

// sin4 (5.60E-07): x * (0.9999949932098388671875 + x2*(-0.166601598262786865234375 + x2*8.12153331935405731201171875e-3))
// sin5 (1.80E-09): x * (1 + x2*(-0.166666507720947265625 +x2*(8.331983350217342376708984375e-3 + x2*(-1.94961365195922553539276123046875e-4))))
// cos4 (6.70E-06): 0.999990046024322509765625 + x2*(-0.4997082054615020751953125 + x2*4.03986163437366485595703125e-2)
// cos5 (6.00E-08): 0.999999940395355224609375 + x2*(-0.499998986721038818359375 + x2*(4.1663490235805511474609375e-2 + x2*(-1.385320327244699001312255859375e-3 + x2*2.31450176215730607509613037109375e-5)))

fn sin_poly5<T>(r: T) -> T
where
    T: Float + Sin5Coefficients,
{
    let r2 = r * r;
    r * (T::SIN_C1 + r2 * (T::SIN_C3 + r2 * T::SIN_C5))
}

fn cos_poly6<T>(r: T) -> T
where
    T: Float + Cos6Coefficients,
{
    let r2 = r * r;
    T::COS_C0 + r2 * (T::COS_C2 + r2 * (T::COS_C4 + r2 * T::COS_C6))
}

// For sin/cos quadrant helper functions:
// 2 least significant bits of q are quadrant index, ie [0, 1, 2, 3].
fn sin_quadrant<T>(r: T, q: i32) -> T
where
    T: Float + Sin5Coefficients + Cos6Coefficients,
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

fn cos_quadrant<T>(r: T, q: i32) -> T
where
    T: Float + Sin5Coefficients + Cos6Coefficients,
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

fn sin_cos_quadrant<T>(r: T, q: i32) -> (T, T)
where
    T: Float + Sin5Coefficients + Cos6Coefficients,
{
    let sin = sin_poly5::<T>(r);
    let cos = cos_poly6::<T>(r);

    // map values according to quadrant
    let sin_cos = if q & 1 == 0 { (sin, cos) } else { (cos, -sin) };

    if q & 2 == 0 { sin_cos } else { (-sin_cos.0, -sin_cos.1) }
}

pub fn sin_approx(x: f32) -> f32 {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = libm::roundf(t); // nearest quadrant
    let r = t - q;
    #[allow(clippy::cast_possible_truncation)]
    sin_quadrant(r, q as i32)
}

pub fn cos_approx(x: f32) -> f32 {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = libm::roundf(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    #[allow(clippy::cast_possible_truncation)]
    cos_quadrant(r, q as i32)
}

pub fn sin_cos_approx(x: f32) -> (f32, f32) {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = libm::roundf(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    #[allow(clippy::cast_possible_truncation)]
    sin_cos_quadrant(r, q as i32)
}

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use super::*;
    use approx::assert_abs_diff_eq;
    macro_rules! assert_near {
        ($left:expr, $right:expr) => {
            approx::assert_abs_diff_eq!($left, $right, epsilon = 4e-6);
        };
    }

    #[test]
    fn asin() {
        assert_abs_diff_eq!(0.0_f32.asin(), libm::asinf(0.0));
    }
    #[test]
    fn sin() {
        assert_near!(sin_approx(10.0_f32.to_radians()), libm::sinf(10.0_f32.to_radians()));
        assert_near!(sin_approx(20.0_f32.to_radians()), libm::sinf(20.0_f32.to_radians()));
        assert_near!(sin_approx(30.0_f32.to_radians()), libm::sinf(30.0_f32.to_radians()));
        assert_near!(sin_approx(40.0_f32.to_radians()), libm::sinf(40.0_f32.to_radians()));
        assert_near!(sin_approx(50.0_f32.to_radians()), libm::sinf(50.0_f32.to_radians()));
        assert_near!(sin_approx(60.0_f32.to_radians()), libm::sinf(60.0_f32.to_radians()));
        assert_near!(sin_approx(70.0_f32.to_radians()), libm::sinf(70.0_f32.to_radians()));
        assert_near!(sin_approx(80.0_f32.to_radians()), libm::sinf(80.0_f32.to_radians()));
        assert_near!(sin_approx(90.0_f32.to_radians()), libm::sinf(90.0_f32.to_radians()));
        assert_near!(sin_approx(100.0_f32.to_radians()), libm::sinf(100.0_f32.to_radians()));
        assert_near!(sin_approx(110.0_f32.to_radians()), libm::sinf(110.0_f32.to_radians()));
        assert_near!(sin_approx(120.0_f32.to_radians()), libm::sinf(120.0_f32.to_radians()));
        assert_near!(sin_approx(130.0_f32.to_radians()), libm::sinf(130.0_f32.to_radians()));
        assert_near!(sin_approx(140.0_f32.to_radians()), libm::sinf(140.0_f32.to_radians()));
        assert_near!(sin_approx(150.0_f32.to_radians()), libm::sinf(150.0_f32.to_radians()));
        assert_near!(sin_approx(160.0_f32.to_radians()), libm::sinf(160.0_f32.to_radians()));
        assert_near!(sin_approx(170.0_f32.to_radians()), libm::sinf(170.0_f32.to_radians()));
        assert_near!(sin_approx(180.0_f32.to_radians()), libm::sinf(180.0_f32.to_radians()));
        assert_near!(sin_approx(190.0_f32.to_radians()), libm::sinf(190.0_f32.to_radians()));
        assert_near!(sin_approx(-10.0_f32.to_radians()), libm::sinf(-10.0_f32.to_radians()));
        assert_near!(sin_approx(-20.0_f32.to_radians()), libm::sinf(-20.0_f32.to_radians()));
        assert_near!(sin_approx(-30.0_f32.to_radians()), libm::sinf(-30.0_f32.to_radians()));
        assert_near!(sin_approx(-40.0_f32.to_radians()), libm::sinf(-40.0_f32.to_radians()));
        assert_near!(sin_approx(-50.0_f32.to_radians()), libm::sinf(-50.0_f32.to_radians()));
        assert_near!(sin_approx(-60.0_f32.to_radians()), libm::sinf(-60.0_f32.to_radians()));
        assert_near!(sin_approx(-70.0_f32.to_radians()), libm::sinf(-70.0_f32.to_radians()));
        assert_near!(sin_approx(-80.0_f32.to_radians()), libm::sinf(-80.0_f32.to_radians()));
        assert_near!(sin_approx(-90.0_f32.to_radians()), libm::sinf(-90.0_f32.to_radians()));
        assert_near!(sin_approx(-100.0_f32.to_radians()), libm::sinf(-100.0_f32.to_radians()));
        assert_near!(sin_approx(-110.0_f32.to_radians()), libm::sinf(-110.0_f32.to_radians()));
        assert_near!(sin_approx(-120.0_f32.to_radians()), libm::sinf(-120.0_f32.to_radians()));
        assert_near!(sin_approx(-130.0_f32.to_radians()), libm::sinf(-130.0_f32.to_radians()));
        assert_near!(sin_approx(-140.0_f32.to_radians()), libm::sinf(-140.0_f32.to_radians()));
        assert_near!(sin_approx(-150.0_f32.to_radians()), libm::sinf(-150.0_f32.to_radians()));
        assert_near!(sin_approx(-160.0_f32.to_radians()), libm::sinf(-160.0_f32.to_radians()));
        assert_near!(sin_approx(-170.0_f32.to_radians()), libm::sinf(-170.0_f32.to_radians()));
        assert_near!(sin_approx(-180.0_f32.to_radians()), libm::sinf(-180.0_f32.to_radians()));
        assert_near!(sin_approx(-190.0_f32.to_radians()), libm::sinf(-190.0_f32.to_radians()));
    }
    #[test]
    fn cos() {
        assert_near!(cos_approx(10.0_f32.to_radians()), libm::cosf(10.0_f32.to_radians()));
        assert_near!(cos_approx(20.0_f32.to_radians()), libm::cosf(20.0_f32.to_radians()));
        assert_near!(cos_approx(30.0_f32.to_radians()), libm::cosf(30.0_f32.to_radians()));
        assert_near!(cos_approx(40.0_f32.to_radians()), libm::cosf(40.0_f32.to_radians()));
        assert_near!(cos_approx(50.0_f32.to_radians()), libm::cosf(50.0_f32.to_radians()));
        assert_near!(cos_approx(60.0_f32.to_radians()), libm::cosf(60.0_f32.to_radians()));
        assert_near!(cos_approx(70.0_f32.to_radians()), libm::cosf(70.0_f32.to_radians()));
        assert_near!(cos_approx(80.0_f32.to_radians()), libm::cosf(80.0_f32.to_radians()));
        assert_near!(cos_approx(90.0_f32.to_radians()), libm::cosf(90.0_f32.to_radians()));
        assert_near!(cos_approx(100.0_f32.to_radians()), libm::cosf(100.0_f32.to_radians()));
        assert_near!(cos_approx(110.0_f32.to_radians()), libm::cosf(110.0_f32.to_radians()));
        assert_near!(cos_approx(120.0_f32.to_radians()), libm::cosf(120.0_f32.to_radians()));
        assert_near!(cos_approx(130.0_f32.to_radians()), libm::cosf(130.0_f32.to_radians()));
        assert_near!(cos_approx(140.0_f32.to_radians()), libm::cosf(140.0_f32.to_radians()));
        assert_near!(cos_approx(150.0_f32.to_radians()), libm::cosf(150.0_f32.to_radians()));
        assert_near!(cos_approx(160.0_f32.to_radians()), libm::cosf(160.0_f32.to_radians()));
        assert_near!(cos_approx(170.0_f32.to_radians()), libm::cosf(170.0_f32.to_radians()));
        assert_near!(cos_approx(180.0_f32.to_radians()), libm::cosf(180.0_f32.to_radians()));
        assert_near!(cos_approx(190.0_f32.to_radians()), libm::cosf(190.0_f32.to_radians()));
        assert_near!(cos_approx(-10.0_f32.to_radians()), libm::cosf(-10.0_f32.to_radians()));
        assert_near!(cos_approx(-20.0_f32.to_radians()), libm::cosf(-20.0_f32.to_radians()));
        assert_near!(cos_approx(-30.0_f32.to_radians()), libm::cosf(-30.0_f32.to_radians()));
        assert_near!(cos_approx(-40.0_f32.to_radians()), libm::cosf(-40.0_f32.to_radians()));
        assert_near!(cos_approx(-50.0_f32.to_radians()), libm::cosf(-50.0_f32.to_radians()));
        assert_near!(cos_approx(-60.0_f32.to_radians()), libm::cosf(-60.0_f32.to_radians()));
        assert_near!(cos_approx(-70.0_f32.to_radians()), libm::cosf(-70.0_f32.to_radians()));
        assert_near!(cos_approx(-80.0_f32.to_radians()), libm::cosf(-80.0_f32.to_radians()));
        assert_near!(cos_approx(-90.0_f32.to_radians()), libm::cosf(-90.0_f32.to_radians()));
        assert_near!(cos_approx(-100.0_f32.to_radians()), libm::cosf(-100.0_f32.to_radians()));
        assert_near!(cos_approx(-110.0_f32.to_radians()), libm::cosf(-110.0_f32.to_radians()));
        assert_near!(cos_approx(-120.0_f32.to_radians()), libm::cosf(-120.0_f32.to_radians()));
        assert_near!(cos_approx(-130.0_f32.to_radians()), libm::cosf(-130.0_f32.to_radians()));
        assert_near!(cos_approx(-140.0_f32.to_radians()), libm::cosf(-140.0_f32.to_radians()));
        assert_near!(cos_approx(-150.0_f32.to_radians()), libm::cosf(-150.0_f32.to_radians()));
        assert_near!(cos_approx(-160.0_f32.to_radians()), libm::cosf(-160.0_f32.to_radians()));
        assert_near!(cos_approx(-170.0_f32.to_radians()), libm::cosf(-170.0_f32.to_radians()));
        assert_near!(cos_approx(-180.0_f32.to_radians()), libm::cosf(-180.0_f32.to_radians()));
        assert_near!(cos_approx(-190.0_f32.to_radians()), libm::cosf(-190.0_f32.to_radians()));
    }
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
    #[test]
    fn atan2() {
        assert_abs_diff_eq!(0.0_f32.atan2(1.0), libm::atan2f(0.0, 1.0));
        assert_abs_diff_eq!(1.0_f32.atan2(0.0), libm::atan2f(1.0, 0.0));
        assert_abs_diff_eq!(0.0_f32.atan2(1.0), 0.0);
        assert_abs_diff_eq!(libm::atan2f(0.0, 1.0), 0.0);

        assert_abs_diff_eq!(0.0_f64.atan2(1.0), libm::atan2(0.0, 1.0));
        assert_abs_diff_eq!(1.0_f64.atan2(0.0), libm::atan2(1.0, 0.0));
    }
}
