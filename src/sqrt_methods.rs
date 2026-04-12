#![allow(clippy::excessive_precision)]
use cfg_if::cfg_if;

/// `no_std` implementations of `sqrt` and `sqrt_reciprocal` in  method call syntax<br>
/// ie `x.sqrt()`, `x.sqrt_reciprocal()`.
pub trait SqrtMethods: Sized {
    fn sqrt(self) -> Self;
    fn sqrt_reciprocal(self) -> Self;
}

cfg_if! {
    if #[cfg(feature = "std")] {
        // Use the hardware-linked math methods in Standard Library
        impl SqrtMethods for f32 {
            #[inline(always)]
            fn sqrt(self) -> f32 {
                self.sqrt()
            }
            #[inline(always)]
            fn sqrt_reciprocal(self) -> f32 {
                1.0 / self.sqrt()
            }
        }
        impl SqrtMethods for f64 {
            #[inline(always)]
            fn sqrt(self) -> f64 {
                self.sqrt()
            }
            #[inline(always)]
            fn sqrt_reciprocal(self) -> f64 {
                1.0 / self.sqrt()
            }
        }
    } else if #[cfg(all(not(feature = "std"), feature = "libm"))] {
        impl SqrtMethods for f32 {
            #[inline(always)]
            fn sqrt(self) -> f32 {
                // Use hardware ASM for ARM chips with an FPU
                #[cfg(all(target_arch = "arm", target_feature = "vfp2"))]
                {
                    let mut result: f32;
                    unsafe {
                        core::arch::asm!(
                            "vsqrt.f32 {res}, {val}",
                            res = out(vreg) result,
                            val = in(vreg) self,
                            options(nomem, nostack, preserves_flags)
                        );
                    }
                    result
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
                    let mut result: f32;
                    unsafe {
                        core::arch::asm!(
                            "vrsqrt.f32 {res}, {val}",
                            res = out(vreg) result,
                            val = in(vreg) self,
                            options(nomem, nostack)
                        );
                    }
                    result
                }
                // Fallback for non-ARM, non-FPU target
                #[cfg(not(all(target_arch = "arm", target_feature = "vfp2")))]
                {
                    1.0 / libm::sqrtf(self)
                }
            }
        }
        impl SqrtMethods for f64 {
            #[inline(always)]
            fn sqrt(self) -> f64 {
                libm::sqrt(self)
            }
            #[inline(always)]
            fn sqrt_reciprocal(self) -> f64 {
                1.0 / libm::sqrt(self)
            }
        }
    } else if #[cfg(all(not(feature = "std"), not(feature = "libm")))] {
        impl SqrtMethods for f32 {
            fn sqrt(self) -> f32 {
                1.0 / _sqrt_reciprocalf(self)
            }
            fn sqrt_reciprocal(self) -> f32 {
                _sqrt_reciprocalf(self)
            }
        }
        impl SqrtMethods for f64 {
            fn sqrt(self) -> f64 {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
            fn sqrt_reciprocal(self) -> f64 {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
        }
    }
}

#[inline(always)]
fn _sqrt_reciprocalf(x: f32) -> f32 {
    let mut y: f32 = x;
    let mut i: i32 = y.to_bits().cast_signed();
    i = 0x5F37_5A86 - (i >> 1);
    y = f32::from_bits(i.cast_unsigned());
    y * (1.690_002_31 - 0.714_158_168 * x * y * y) // First iteration
}

#[inline(always)]
fn _quake_sqrt_reciprocal(number: f32) -> f32 {
    let mut y: f32 = number;
    let mut i: i32 = y.to_bits().cast_signed();
    i = 0x5F37_5A86 - (i >> 1);
    y = f32::from_bits(i.cast_unsigned());
    y * (1.5 - (number * 0.5 * y * y))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    #![allow(clippy::used_underscore_items)]
    #[allow(unused)]
    use super::*;

    #[test]
    fn sqrt_reciprocal() {
        assert_eq!(_quake_sqrt_reciprocal(4.0), 0.499_154_06);
        assert_eq!(_sqrt_reciprocalf(4.0), 0.494_354_96);
        assert_eq!(4.0.sqrt_reciprocal(), 0.5);
    }
    #[test]
    fn sqrt() {
        assert_eq!(0.0_f32.sqrt(), libm::sqrtf(0.0));
        assert_eq!(4.0_f32.sqrt(), libm::sqrtf(4.0));
    }
}
