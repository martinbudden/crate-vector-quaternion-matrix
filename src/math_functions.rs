#![allow(clippy::excessive_precision)]

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

pub trait MathFunctions: Sized {
    fn abs(self) -> Self;

    fn sqrt(self) -> Self;
    fn reciprocal_sqrt(self) -> Self;
    fn half_reciprocal_sqrt(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan2(self, y: Self) -> Self;
}

impl MathFunctions for f32 {
    fn abs(self) -> Self {
        (self).abs()
    }

    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    fn reciprocal_sqrt(self) -> Self {
        1.0 / libm::sqrtf(self)
    }
    fn half_reciprocal_sqrt(self) -> Self {
        0.5 / libm::sqrtf(self)
    }
    fn sin_cos(self) -> (Self, Self) {
        //(libm::sinf(self), libm::cosf(self))
        sin_cos(self)
    }
    fn sin(self) -> Self {
        sin(self)
    }
    fn cos(self) -> Self {
        cos(self)
    }
    fn tan(self) -> Self {
        libm::tanf(self)
    }
    fn asin(self) -> Self {
        libm::asinf(self)
    }
    fn acos(self) -> Self {
        libm::acosf(self)
    }
    fn atan2(self, y: Self) -> Self {
        libm::atan2f(y, self)
    }
}

impl MathFunctions for f64 {
    fn abs(self) -> Self {
        (self).abs()
    }

    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    fn reciprocal_sqrt(self) -> Self {
        1.0 / libm::sqrt(self)
    }
    fn half_reciprocal_sqrt(self) -> Self {
        0.5 / libm::sqrt(self)
    }
    fn sin_cos(self) -> (Self, Self) {
        (libm::sin(self), libm::cos(self))
        //sin_cos(self)
    }
    fn sin(self) -> Self {
        //sin(self)
        libm::sin(self)
    }
    fn cos(self) -> Self {
        //cos(self)
        libm::cos(self)
    }
    fn tan(self) -> Self {
        libm::tan(self)
    }
    fn asin(self) -> Self {
        libm::asin(self)
    }
    fn acos(self) -> Self {
        libm::acos(self)
    }
    fn atan2(self, y: Self) -> Self {
        libm::atan2(y, self)
    }
}

#[cfg(test)]
fn reciprocal_sqrtf(x: f32) -> f32 {
    let mut y: f32 = x;
    let mut i: i32 = y.to_bits().cast_signed();
    i = 0x5F375A86 - (i >> 1);
    y = f32::from_bits(i.cast_unsigned());
    y * (1.69000231 - 0.714158168 * x * y * y) // First iteration
}

#[cfg(test)]
fn quake_reciprocal_sqrt(number: f32) -> f32 {
    let mut y: f32 = number;
    let mut i: i32 = y.to_bits().cast_signed();
    i = 0x5F375A86 - (i >> 1);
    y = f32::from_bits(i.cast_unsigned());
    y * (1.5 - (number * 0.5 * y * y))
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
    const SIN_C3: Self = -0.64568519592;
    const SIN_C5: Self = 0.077562883496;
}

impl Cos6Coefficients for f32 {
    const COS_C0: Self = 1.0;
    const COS_C2: Self = -1.23369765282;
    const COS_C4: Self = 0.25360107422;
    const COS_C6: Self = -0.020408373326;
}

impl Sin5Coefficients for f64 {
    const SIN_C1: Self = core::f64::consts::FRAC_PI_2;
    const SIN_C3: Self = -0.64568519592;
    const SIN_C5: Self = 0.077562883496;
}

impl Cos6Coefficients for f64 {
    const COS_C0: Self = 1.0;
    const COS_C2: Self = -1.23369765282;
    const COS_C4: Self = 0.25360107422;
    const COS_C6: Self = -0.020408373326;
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

fn sin(x: f32) -> f32 {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = libm::roundf(t); // nearest quadrant
    let r = t - q;
    sin_quadrant(r, q as i32)
}

fn cos(x: f32) -> f32 {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = libm::roundf(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    cos_quadrant(r, q as i32)
}

fn sin_cos(x: f32) -> (f32, f32) {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = libm::roundf(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    sin_cos_quadrant(r, q as i32)
}

#[cfg(test)]
mod tests {
    #![allow(unused)]
    use super::*;

    #[test]
    fn reciprocal_sqrt() {
        assert_eq!(quake_reciprocal_sqrt(4.0), 0.49915406);
        assert_eq!(reciprocal_sqrtf(4.0), 0.49435496);
        assert_eq!(4.0.reciprocal_sqrt(), 0.5);
    }
    #[test]
    fn sqrt() {
        assert_eq!(0.0_f32.sqrt(), libm::sqrtf(0.0));
    }
    #[test]
    fn asin() {
        assert_eq!(0.0_f32.asin(), libm::asinf(0.0));
    }
    #[test]
    fn sin() {
        assert_eq!(0.0_f32.sin(), 0.0);
        assert_eq!(10.0_f32.to_radians().sin(), libm::sinf(10.0_f32.to_radians()));
        assert_eq!(20.0_f32.to_radians().sin(), libm::sinf(20.0_f32.to_radians()));
        assert_eq!(30.0_f32.to_radians().sin(), libm::sinf(30.0_f32.to_radians()));
        assert_eq!(40.0_f32.to_radians().sin(), libm::sinf(40.0_f32.to_radians()));
        assert_eq!(50.0_f32.to_radians().sin(), libm::sinf(50.0_f32.to_radians()));
        assert_eq!(60.0_f32.to_radians().sin(), libm::sinf(60.0_f32.to_radians()));
        assert_eq!(70.0_f32.to_radians().sin(), libm::sinf(70.0_f32.to_radians()));
        assert_eq!(80.0_f32.to_radians().sin(), libm::sinf(80.0_f32.to_radians()));
        assert_eq!(90.0_f32.to_radians().sin(), libm::sinf(90.0_f32.to_radians()));
        assert_eq!(100.0_f32.to_radians().sin(), libm::sinf(100.0_f32.to_radians()));
        assert_eq!(110.0_f32.to_radians().sin(), libm::sinf(110.0_f32.to_radians()));
        assert_eq!(120.0_f32.to_radians().sin(), libm::sinf(120.0_f32.to_radians()));
        assert_eq!(130.0_f32.to_radians().sin(), libm::sinf(130.0_f32.to_radians()));
        assert_eq!(140.0_f32.to_radians().sin(), libm::sinf(140.0_f32.to_radians()));
        assert_eq!(150.0_f32.to_radians().sin(), libm::sinf(150.0_f32.to_radians()));
        assert_eq!(160.0_f32.to_radians().sin(), libm::sinf(160.0_f32.to_radians()));
        assert_eq!(170.0_f32.to_radians().sin(), libm::sinf(170.0_f32.to_radians()));
        assert_eq!(180.0_f32.to_radians().sin(), libm::sinf(180.0_f32.to_radians()));
        assert_eq!(190.0_f32.to_radians().sin(), libm::sinf(190.0_f32.to_radians()));
        assert_eq!((-10.0_f32).to_radians().sin(), libm::sinf(-10.0_f32.to_radians()));
        assert_eq!((-20.0_f32).to_radians().sin(), libm::sinf(-20.0_f32.to_radians()));
        assert_eq!((-30.0_f32).to_radians().sin(), libm::sinf(-30.0_f32.to_radians()));
        assert_eq!((-40.0_f32).to_radians().sin(), libm::sinf(-40.0_f32.to_radians()));
        assert_eq!((-50.0_f32).to_radians().sin(), libm::sinf(-50.0_f32.to_radians()));
        assert_eq!((-60.0_f32).to_radians().sin(), libm::sinf(-60.0_f32.to_radians()));
        assert_eq!((-70.0_f32).to_radians().sin(), libm::sinf(-70.0_f32.to_radians()));
        assert_eq!((-80.0_f32).to_radians().sin(), libm::sinf(-80.0_f32.to_radians()));
        assert_eq!((-90.0_f32).to_radians().sin(), libm::sinf(-90.0_f32.to_radians()));
        assert_eq!((-100.0_f32).to_radians().sin(), libm::sinf(-100.0_f32.to_radians()));
        assert_eq!((-110.0_f32).to_radians().sin(), libm::sinf(-110.0_f32.to_radians()));
        assert_eq!((-120.0_f32).to_radians().sin(), libm::sinf(-120.0_f32.to_radians()));
        assert_eq!((-130.0_f32).to_radians().sin(), libm::sinf(-130.0_f32.to_radians()));
        assert_eq!((-140.0_f32).to_radians().sin(), libm::sinf(-140.0_f32.to_radians()));
        assert_eq!((-150.0_f32).to_radians().sin(), libm::sinf(-150.0_f32.to_radians()));
        assert_eq!((-160.0_f32).to_radians().sin(), libm::sinf(-160.0_f32.to_radians()));
        assert_eq!((-170.0_f32).to_radians().sin(), libm::sinf(-170.0_f32.to_radians()));
        assert_eq!((-180.0_f32).to_radians().sin(), libm::sinf(-180.0_f32.to_radians()));
        assert_eq!((-190.0_f32).to_radians().sin(), libm::sinf(-190.0_f32.to_radians()));
    }
    #[test]
    fn cos() {
        assert_eq!(0.0_f32.cos(), 1.0);
        assert_eq!(10.0_f32.to_radians().cos(), libm::cosf(10.0_f32.to_radians()));
        assert_eq!(20.0_f32.to_radians().cos(), libm::cosf(20.0_f32.to_radians()));
        assert_eq!(30.0_f32.to_radians().cos(), libm::cosf(30.0_f32.to_radians()));
        assert_eq!(40.0_f32.to_radians().cos(), libm::cosf(40.0_f32.to_radians()));
        assert_eq!(50.0_f32.to_radians().cos(), libm::cosf(50.0_f32.to_radians()));
        assert_eq!(60.0_f32.to_radians().cos(), libm::cosf(60.0_f32.to_radians()));
        assert_eq!(70.0_f32.to_radians().cos(), libm::cosf(70.0_f32.to_radians()));
        assert_eq!(80.0_f32.to_radians().cos(), libm::cosf(80.0_f32.to_radians()));
        assert_eq!(90.0_f32.to_radians().cos(), libm::cosf(90.0_f32.to_radians()));
        assert_eq!(100.0_f32.to_radians().cos(), libm::cosf(100.0_f32.to_radians()));
        assert_eq!(110.0_f32.to_radians().cos(), libm::cosf(110.0_f32.to_radians()));
        assert_eq!(120.0_f32.to_radians().cos(), libm::cosf(120.0_f32.to_radians()));
        assert_eq!(130.0_f32.to_radians().cos(), libm::cosf(130.0_f32.to_radians()));
        assert_eq!(140.0_f32.to_radians().cos(), libm::cosf(140.0_f32.to_radians()));
        assert_eq!(150.0_f32.to_radians().cos(), libm::cosf(150.0_f32.to_radians()));
        assert_eq!(160.0_f32.to_radians().cos(), libm::cosf(160.0_f32.to_radians()));
        assert_eq!(170.0_f32.to_radians().cos(), libm::cosf(170.0_f32.to_radians()));
        assert_eq!(180.0_f32.to_radians().cos(), libm::cosf(180.0_f32.to_radians()));
        assert_eq!(190.0_f32.to_radians().cos(), libm::cosf(190.0_f32.to_radians()));
        assert_eq!((-10.0_f32).to_radians().cos(), libm::cosf(-10.0_f32.to_radians()));
        assert_eq!((-20.0_f32).to_radians().cos(), libm::cosf(-20.0_f32.to_radians()));
        assert_eq!((-30.0_f32).to_radians().cos(), libm::cosf(-30.0_f32.to_radians()));
        assert_eq!((-40.0_f32).to_radians().cos(), libm::cosf(-40.0_f32.to_radians()));
        assert_eq!((-50.0_f32).to_radians().cos(), libm::cosf(-50.0_f32.to_radians()));
        assert_eq!((-60.0_f32).to_radians().cos(), libm::cosf(-60.0_f32.to_radians()));
        assert_eq!((-70.0_f32).to_radians().cos(), libm::cosf(-70.0_f32.to_radians()));
        assert_eq!((-80.0_f32).to_radians().cos(), libm::cosf(-80.0_f32.to_radians()));
        assert_eq!((-90.0_f32).to_radians().cos(), libm::cosf(-90.0_f32.to_radians()));
        assert_eq!((-100.0_f32).to_radians().cos(), libm::cosf(-100.0_f32.to_radians()));
        assert_eq!((-110.0_f32).to_radians().cos(), libm::cosf(-110.0_f32.to_radians()));
        assert_eq!((-120.0_f32).to_radians().cos(), libm::cosf(-120.0_f32.to_radians()));
        assert_eq!((-130.0_f32).to_radians().cos(), libm::cosf(-130.0_f32.to_radians()));
        assert_eq!((-140.0_f32).to_radians().cos(), libm::cosf(-140.0_f32.to_radians()));
        assert_eq!((-150.0_f32).to_radians().cos(), libm::cosf(-150.0_f32.to_radians()));
        assert_eq!((-160.0_f32).to_radians().cos(), libm::cosf(-160.0_f32.to_radians()));
        assert_eq!((-170.0_f32).to_radians().cos(), libm::cosf(-170.0_f32.to_radians()));
        assert_eq!((-180.0_f32).to_radians().cos(), libm::cosf(-180.0_f32.to_radians()));
        assert_eq!((-190.0_f32).to_radians().cos(), libm::cosf(-190.0_f32.to_radians()));
    }
    #[test]
    fn sin_cos() {
        assert_eq!(0.0_f32.sin_cos(), (0.0, 1.0));
        assert_eq!(
            10.0_f32.to_radians().sin_cos(),
            (libm::sinf(10.0_f32.to_radians()), libm::cosf(10.0_f32.to_radians()))
        );
        assert_eq!(
            (-10.0_f32).to_radians().sin_cos(),
            (libm::sinf(-10.0_f32.to_radians()), libm::cosf(-10.0_f32.to_radians()))
        );
        assert_eq!(
            110.0_f32.to_radians().sin_cos(),
            (libm::sinf(110.0_f32.to_radians()), libm::cosf(110.0_f32.to_radians()))
        );
        assert_eq!(
            (-110.0_f32).to_radians().sin_cos(),
            (libm::sinf(-110.0_f32.to_radians()), libm::cosf(-110.0_f32.to_radians()))
        );
    }
    #[test]
    fn atan2() {
        assert_eq!(0.0_f32.atan2(1.0), libm::atan2f(0.0, 1.0));
        assert_eq!(1.0_f32.atan2(0.0), libm::atan2f(1.0, 0.0));
        assert_eq!(0.0_f32.atan2(1.0), 0.0);
        assert_eq!(libm::atan2f(0.0, 1.0), 0.0);

        assert_eq!(0.0_f64.atan2(1.0), libm::atan2(0.0, 1.0));
        assert_eq!(1.0_f64.atan2(0.0), libm::atan2(1.0, 0.0));
    }
}
