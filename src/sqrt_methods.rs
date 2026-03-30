#![allow(clippy::excessive_precision)]
use cfg_if::cfg_if;

/// `no_std` implementations of `sqrt`, `reciprocal_sqrt`, and `half_reciprocal_sqrt` in  method call syntax<br>
pub trait SqrtMethods: Sized {
    fn sqrt(self) -> Self;
    fn reciprocal_sqrt(self) -> Self;
    fn half_reciprocal_sqrt(self) -> Self;
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
            fn reciprocal_sqrt(self) -> f32 {
                1.0 / self.sqrt()
            }
            #[inline(always)]
            fn half_reciprocal_sqrt(self) -> f32 {
                0.5 / self.sqrt()
            }
        }
        impl SqrtMethods for f64 {
            #[inline(always)]
            fn sqrt(self) -> f64 {
                self.sqrt()
            }
            #[inline(always)]
            fn reciprocal_sqrt(self) -> f64 {
                1.0 / self.sqrt()
            }
            #[inline(always)]
            fn half_reciprocal_sqrt(self) -> f64 {
                0.5 / self.sqrt()
            }
        }
    } else if #[cfg(all(not(feature = "std"), feature = "libm"))] {
        impl SqrtMethods for f32 {
            #[inline(always)]
            fn sqrt(self) -> f32 {
                /*#[cfg(feature = "rp2350")]
                {
                    let mut result: f32;
                    unsafe {
                        // vsqrt.f32 {destination}, {source}
                        core::arch::asm!(
                            "vsqrt.f32 {0:s}, {1:s}",
                            out(vreg) result,
                            in(vreg) self,
                            options(nomem, nostack, preserves_flags)
                        );
                    }
                    result
                }
                #[cfg(not(feature = "rp2350"))]*/
                    {libm::sqrtf(self)
                }
            }
            #[inline(always)]
            fn reciprocal_sqrt(self) -> f32 {
                /*#[cfg(feature = "rp2350")]
                {
                    let mut result: f32;
                    unsafe {
                        core::arch::asm!(
                            "vrsqrt.f32 {0:s}, {1:s}",
                            out(vreg) result,
                            in(vreg) self,
                        );
                    }
                    result
                }
                #[cfg(not(feature = "rp2350"))]*/
                {
                    1.0 / libm::sqrtf(self)
                }
            }
            #[inline(always)]
            fn half_reciprocal_sqrt(self) -> f32 {
                0.5 / _sqrtf(self)
            }
        }
        impl SqrtMethods for f64 {
            #[inline(always)]
            fn sqrt(self) -> f64 {
                libm::sqrt(self)
            }
            #[inline(always)]
            fn reciprocal_sqrt(self) -> f64 {
                1.0 / libm::sqrt(self)
            }
            #[inline(always)]
            fn half_reciprocal_sqrt(self) -> f64 {
                0.5 / libm::sqrt(self)
            }
        }
    } else if #[cfg(all(not(feature = "std"), not(feature = "libm")))] {
        impl SqrtMethods for f32 {
            fn sqrt(self) -> f32 {
                1.0 / _reciprocal_sqrtf(self)
            }
            fn reciprocal_sqrt(self) -> f32 {
                _reciprocal_sqrtf(self)
            }
            fn half_reciprocal_sqrt(self) -> f32 {
                0.5 * _reciprocal_sqrtf(self)
            }
        }
        impl SqrtMethods for f64 {
            fn sqrt(self) -> f64 {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
            fn reciprocal_sqrt(self) -> f64 {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
            fn half_reciprocal_sqrt(self) -> f64 {
                compile_error!("Please enable the 'libm' or 'std' feature for math support.")
            }
        }
    }
}
#[inline(always)]
fn _sqrtf(x: f32) -> f32 {
    /*#[cfg(feature = "rp2350")]
    {
        let mut result: f32;
        unsafe {
            // vsqrt.f32 {destination}, {source}
            core::arch::asm!(
                "vsqrt.f32 {0:s}, {1:s}",
                out(vreg) result,
                in(vreg) self,
                options(nomem, nostack, preserves_flags)
            );
        }
        result
    }
    #[cfg(not(feature = "rp2350"))]*/
    {
        #[cfg(feature = "libm")]
        {
            let result: f32 = libm::sqrtf(x);
            result
        }
        #[cfg(not(feature = "libm"))]
        {
            let result: f32 = 1.0 / _reciprocal_sqrtf(x);
            result
        }
    }
}

/*pub trait MathExt {
    fn inv_sqrt(self) -> f32;
}

impl MathExt for f32 {
    #[inline(always)]
    fn inv_sqrt(self) -> f32 {
        #[cfg(feature = "rp2350")]
        {
            let mut result: f32;
            unsafe {
                core::arch::asm!(
                    "vrsqrt.f32 {0:s}, {1:s}",
                    out(vreg) result,
                    in(vreg) self,
                );
            }
            result
        }
        #[cfg(not(feature = "rp2350"))]
        {
            1.0 / self.sqrt() // Fallback for RP2040
        }
    }
}*/

#[inline(always)]
fn _reciprocal_sqrtf(x: f32) -> f32 {
    let mut y: f32 = x;
    let mut i: i32 = y.to_bits().cast_signed();
    i = 0x5F375A86 - (i >> 1);
    y = f32::from_bits(i.cast_unsigned());
    y * (1.69000231 - 0.714158168 * x * y * y) // First iteration
}

#[inline(always)]
fn _quake_reciprocal_sqrt(number: f32) -> f32 {
    let mut y: f32 = number;
    let mut i: i32 = y.to_bits().cast_signed();
    i = 0x5F375A86 - (i >> 1);
    y = f32::from_bits(i.cast_unsigned());
    y * (1.5 - (number * 0.5 * y * y))
}

#[cfg(test)]
mod tests {
    #![allow(unused)]
    use super::*;

    #[test]
    fn reciprocal_sqrt() {
        assert_eq!(_quake_reciprocal_sqrt(4.0), 0.49915406);
        assert_eq!(_reciprocal_sqrtf(4.0), 0.49435496);
        assert_eq!(4.0.reciprocal_sqrt(), 0.5);
    }
    #[test]
    fn sqrt() {
        assert_eq!(0.0_f32.sqrt(), libm::sqrtf(0.0));
    }
}
