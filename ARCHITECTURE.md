# Architecture

This document describes the overall design goals of `vqm` (`vector-quaternion-matrix`).

## Why another vector/linear algebra/math related crate?

There are currently a number of Rust crates that support vector math, quaternions, and matrices. The most notable being
[nalgebra](https://crates.io/crates/nalgebra), [glam](https://crates.io/crates/glam), [vek](https://crates.io/crates/vek),
and [ultraviolet](https://crates.io/crates/ultraviolet).

`nalgebra` is a general purpose linear algebra crate. The others are more focused on graphics and game maths.

In graphics and gaming the requirement is generally to be able to do a relatively small number of operations on a
relatively large number of vectors in a given time slice. The graphics/game focused crates optimize for this
(`ultraviolet` in particular uses  `SoA` (Structure of Arrays) rather than `AoS` (Array of Structs) layout to this end).

In embedded applications the requirement is often to do a relatively large number of operations on a relatively small number
of vectors. This means that `ultraviolet` is not really suited for embedded, and although `glam` or `vek` could be used
they would not be playing to their strengths.

This leaves `nalgebra`. It certainly could be used: even though it is a large library only the bits used would be included
in an application, so it would not cause code bloat.

However I did not really want my code to be dependent on such a large library, so I decided to port my existing C++ vector
library to Rust. ("How hard could it be" - well harder than I thought, but ok).

I decided to take a generic approach from the start (because I wanted to support both `f32` and `f62`) and that decision
has paid unexpected dividends:

1. During the development of version `0.1.13` I realized my generic approach would enable
   Units of Measurement([uom](https://crates.io/crates/uom)) almost "for free" so I added support for it.
1. During the development of version `0.1.15` I realized my generic approach would allow the straightforward
   implementation of a 9x9 matrix as an array of nine 3x3 matrices. I knew this would greatly simplify the
   position Kalman filter I was writing, so I added support for this a the `Matrix9` type.

## Design goals

`vqm` was primarily designed to support the implementation a self-balancing robot and a flight controller.

This means being fast enough to run a 8kHz "gyro-PID" loop on a single core microcontroller with enough spare
bandwidth to do other tasks (such as receive commands from a radio, update and OSD, blackbox recording, etc).

Although the design choices were driven by this specific need, the results are generally applicable.

The gyro-PID loop involves the following:

1. Obtain gyro and accelerometer readings from the IMU.
2. Filter those readings.
3. Update notch filter coefficients according to motor RPM.
4. Perform sensor fusion, to calculate the vehicle orientation quaternion from the filtered gyro and accelerometer readings.
5. Use a PID controller to update the motor output vector based on the vehicle orientation and the setpoints obtained from the receiver.
6. Use a "motor mixer" to convert the motor output vector into actual outputs to drive the motors.

That is a significant amount of work to perform in 125 microseconds. So:

1. Performance is key.
2. Support for other libraries (including the filter library and the sensor fusion library) is key.

My previous C++ library had shown this was possible (even without SIMD).

### Pass parameters by value, not by reference

This is more peformant.

Most of the calculations are performed on `Vector3f32` and `Quaternionf32` types.

If these fit comfortably into the CPUs floating point registers.
What's more, if calculations are chained, then the intermediate values can be retained in the CPU registers,
rather than being copied on and off the stack (which might be the case if passed by reference).

Passing by value is also preferable when SIMD is used.

### Use generics

For the main processing loop `Vector3f32` and `Quaternionf32` was required.

Initially I started implementing `Vector3f32` and `Quaternionf32` etc as separate types,
but during implementation and especially when I started adding SIMD support it became
clear that using generics would make for a simpler implementation.

### Use boilerplate rather than code generation

My experience is that libraries that heavily rely on code generation are difficult to use (and debug into).

Using boilerplate makes things harder for the developer of the library, but easier for the user of the library.

There is significant boilerplating between (eg) `Vector2`, `Vector3`, `Vector4`, and `Quaternion`,
but this is manageable and not excessive.

### Follow standards

1. Follow Rust [standard library] conventions and [API guidelines] where possible
2. High quality [rustdoc] generated document

[standard library]: https://doc.rust-lang.org/std/index.html
[API guidelines]: https://rust-lang.github.io/api-guidelines
[rustdoc]: https://doc.rust-lang.org/rustdoc/index.html

## Naming convention for "return" and "in-place" versions of functions

Several functions (eg `normalize`, `clamp`, `transpose`, `adjugate`, and `inverse`) have both "return" and "in-place" forms.
The convention used is that the `_in_place` suffix is used for the "in-place" form.

```rust
# use vqm::Vector3f32;
// return form:
    let v = Vector3f32::new(2.0, 3.0, 5.0);
    let n = v.normalize();
// in-place form:
    let mut v = Vector3f32::new(2.0, 3.0, 5.0);
    v.normalize_in_place();
```

## Invert or Inverse?

As illustrated below, there is no universal convention for the name of the function used to invert a matrix.
Sometimes `invert` is used, sometimes `inverse`.

| Crate                                               | return                                                     | in-place                                   |
| --------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------ |
| [vqm](https://crates.io/crates/vqm)                 | `inverse(self) -> Self`<br>`try_inverse(self) -> Option<>` | `inverse_in_place(&mut self) -> &mut Self` |
| [nalgebra](https://crates.io/crates/nalgebra)       | `try_inverse(&self) -> Option<>`                           | `try_inverse_mut(&mut self) -> bool`       |
| [glam](https://crates.io/crates/glam)               | `inverse(&self) -> Self`                                   | N/A                                        |
| [vek](https://crates.io/crates/vek)                 | `inverted(self) -> Self`                                   | `invert(&mut self)`                        |
| [ultraviolet](https://crates.io/crates/ultraviolet) | `inversed(&self) -> Self`                                  | `inverse(&mut self)`                       |
| [static-math](https://crates.io/crates/static-math) | `inverse(&self) -> Option<>`                               | N/A                                        |
| [cg-math](https://crates.io/crates/cg-math)         | `invert(&self) -> Option<>`                                | N/A                                        |

Note that only `vqm` and `vek` pass parameters by value.

## Future directions

I have no planned major extensions to this crate.

My main focus is on improving documentation and tests, and fixing any bugs that come up in usage.

I may at some point add the ability to calculate eigenvalues and eigenvectors.

I may do some work on improving the performance of the SIMD implementations.
