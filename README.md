# `vqm` Rust Crate<br>![license](https://img.shields.io/badge/license-MIT-green) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![open source](https://badgen.net/badge/open/source/blue?icon=github)

**vqm** is a lightweight, allocation-free Rust math library for **vectors**, **matrices**, and **quaternions**,
designed specifically for `no_std` embedded robotics and real-time systems.
(In particular stabilized vehicles including self-balancing robots and aircraft).

This crate is `no_std`, that it does not link to the standard library and so does not depend on an operating system
and uses no allocation. This means it is suitable for embedded systems.

MSRV: Rust 1.85
The optional `storage` feature requires Rust 1.89 and `simd` requires nightly.

## Overview

Vectors have 2D, 3D, and 4D versions.

Matrices have 2x2, 3x3, 4x4, and 9x9 versions.
The 9x9 matrix is a partial implementation which has been added to support Kalman filters.

Each type has versions for `f32` and `f64`. So we have:

1. 2D vectors: `Vector2f32`, `Vector2f64`
2. 3D vectors: `Vector3f32`, `Vector3f64`
3. 4D vectors: `Vector4f32`, `Vector4f64`
4. [quaternions](https://en.wikipedia.org/wiki/Quaternion): `Quaternionf32`, `Quaternionf64`
5. 2x2 matrices: `Matrix2x2f32`, `Matrix2x2f64`
6. 3x3 matrices: `Matrix3x3f32`, `Matrix3x3f64`
7. 4x4 matrices: `Matrix4x4f32`, `Matrix4x4f64`
8. 9x9 matrices: `Matrix9x9f32`, `Matrix9x9f64`
9. 3x3xM2x2 matrices: `Matrix3x3xM2x2f32`, `Matrix3x3xM2x2f64`
10. 3x3xM3x3 matrices: `Matrix3x3xM3x3f32`, `Matrix3x3xM3x3f64`

Quaternions are implemented using the Hamilton convention.

Matrix elements are stored in a one-dimensional array, stored in column-major order.
`Matrix9x9`, `Matrix3x3xM2x2`, and `Matrix3x3xM3x3` are partial implementations to support Kalman filters.

(Under the hood, types are implemented using generics, so `Vector3f32` is actually `Vector3<f32>`,
but that is transparent to the user.)

`vqm` can be built without `libm`. When built without `libm`, `vqm` provides its own approximations for the following functions:

* `x.sin_cos()`, `x.sin()`, `x.cos()`, `x.tan()`
* `x.asin()` , `x.acos()`, `x.atan()`, `x.atan2()`
* `x.exp()`, `x.exp2()`
* `x.ln()`, `x.log2()`, `x.log10()`, `x.log()`
* `x.powf()`
* `x.sqrt()`, and additionally `x.sqrt_reciprocal()`

The approximations are typically calculated using range mapping and [Padé approximants](https://en.wikipedia.org/wiki/Pad%C3%A9_approximant).
They are generally accurate to 4 or 5 decimal places, depending on the function.

The `sqrt` functions will directly use inline assembly (eg `vsqrt.f32` and `vrsqrt.f32`) if the target architecture supports it,
otherwise they use [Pizer’s optimization](https://pizer.wordpress.com/2008/10/12/fast-inverse-square-root/)
of the famous [Quake inverse square root](https://en.wikipedia.org/wiki/Fast_inverse_square_root).

## Examples

A small selection from what is available:

```rust
use vqm::{Matrix3x3f32, Quaternionf32, Vector3f32};

// vectors
let a = Vector3f32 { x: 1.0, y: 2.0, z: 3.0 };
let b = Vector3f32::new(5.0, 7.0, 11.0);

// vector arithmetic
let c = a + b;
let mut d = (a - b) * 2.0;
d += a;
d = c - d;

// vector dot and cross product
let dot_product = a.dot(b);
let cross_product = a.cross(b);

// matrices
let m = Matrix3x3f32::new([ 2.0,  3.0,  5.0,
                           7.0, 11.0, 13.0,
                           17.0, 19.0, 23.0]);
let n = Matrix3x3f32::new([29.0, 31.0, 37.0,
                           41.0, 43.0, 47.0,
                           53.0, 59.0, 61.0]);

// matrix arithmetic
let mut p = m * n;
p += m;
p *= 2.0;
let h = p + n * m;
let j = h.try_inverse();

// multiplication of a vector by a matrix
let v = m * a;

let q = Quaternionf32 { w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
let r = Quaternionf32::new(11.0, 13.0, 17.0, 23.0);

// quaternion arithmetic
let s = q + r;
let mut t = (s - q) * 2.0;
t += s;
t = s - t;
let q = s * t;
let c = q.conjugate();

// Euler angles
let orientation = Quaternionf32::from_roll_pitch_yaw_degrees(15.0, 60.0, 120.0);
let pitch = orientation.calculate_pitch_degrees();
```

## Units of Measurement (uom) support

Units of measurement support can be enabled with the `uom` feature.
By default the `autoconvert` feature is off, so `uom` will check that incorrect units are not inadvertently used,
but it will not automatically convert between different units.

Note: the `libm` feature must enabled to use the `uom` feature.

```rust
use vqm::Vector3;
use uom::si::f32::{Area, Length, Ratio, Time, Velocity};
use uom::si::{area::square_meter, length::meter, ratio::ratio, time::second, velocity::meter_per_second};

let a = Vector3 { x: Length::new::<meter>(2.0), y: Length::new::<meter>(5.0), z: Length::new::<meter>(11.0) };
let b = Vector3 { x: Length::new::<meter>(3.0), y: Length::new::<meter>(7.0), z: Length::new::<meter>(13.0) };
let c = a + b;
assert_eq!(c, Vector3 { x: Length::new::<meter>(5.0), y: Length::new::<meter>(12.0), z: Length::new::<meter>(24.0) });

let k = Length::new::<meter>(3.0);
let d = a * k;
assert_eq!(d,Vector3 { x: Area::new::<square_meter>(6.0), y: Area::new::<square_meter>(15.0), z: Area::new::<square_meter>(33.0)});

let t = Time::new::<second>(4.0);
let e = a / t;
assert_eq!(e, Vector3 { x: Velocity::new::<meter_per_second>(0.5), y: Velocity::new::<meter_per_second>(1.25), z: Velocity::new::<meter_per_second>(2.75)})
```

## Specializations

`vqm` includes a number of specializations that you might not find in your typical linear algebra/graphics/math library.

A specialization may be considered for inclusion in `vqm` if it provides useful functionality to a library that uses `vqm`.

A specialization generally won't be considered for inclusion to support a single application.

### Bare metal (that is `no_std` and no `libm`)

`vqm` includes implementations for square root and trigonometric functions that allow it to run without `std` and `libm`.

### Robotics support

`vqm` has additional functionality specifically to support robotics applications. This includes:

1. `Vector3f32` functions to load from a `[u8; 6]`.
2. `RollPitch` and `RollPitchYaw` structs.
3. `to_radians` and `to_degrees` convenience functions for vectors and `RollPitch` and `RollPitchYaw` structs.
4. Quaternion utility functions such as `cos_tilt` and `gravity`.

### Kalman filters

`vqm` has additional functionality specifically to support Kalman filters. This includes:

1. `Matrix3x3xM2x2` - a 3x3 matrix matrix of `Matrix2x2`s (so effectively a 6x6 matrix).
2. `Matrix3x3xM3x3` - a 3x3 matrix matrix of `Matrix3x3`s (so effectively a 9x9 matrix).
3. `Matrix3x3::mul_diagonal_vector` - multiplies vector which is treated as a diagonal matrix by a matrix.
4. `Matrix3x3::add_diagonal_vector` - adds a vector which is treated as a diagonal matrix to a matrix.

## SIMD support

**SIMD** support can be enabled with the `simd` feature.

Currently most microcontrollers (eg Arm Cortex M series) don't directly support **SIMD**, so it is of limited use for embedded applications.

However, that may change: the implementation serves as proof of concept and future proofing: it ensures that future implementations are possible.

For that reason many of the implementations are naive "placeholder" implementations.
These placeholder implementations may be slower than the non-SIMD code, so if you used SIMD make sure you benchmark to show
that you are indeed getting a performance improvement.

This uses [portable simd](https://doc.rust-lang.org/core/simd/index.html), which requires the nightly compiler, since it is still
unstable in rust.

**SIMD** does not work with Units of Measurement `uom`.

**SIMD** requires using the `align` feature flag.

This can be invoked using `rustup`, eg:

```sh
rustup run nightly cargo build --features "simd align" --target thumbv8m.main-none-eabi
```

## Usage by other crates

`vqm` is used by a number of other Rust crates, in particular:

1. [signal-filters](https://crates.io/crates/signal-filters)
   `BiquadFilters` and `PT` filters are templated and provide vector implementations for parallel filtering of values on all axes.
2. [sensor-fusion](https://crates.io/crates/sensor-fusion) - extensively uses vectors, quaternions, and matrices for sensor fusion filters
3. [motor-mixers](https://crates.io/crates/motor-mixers) - uses `Vector3f32` and `BiquadFilterVector3f32` in notch filters to filter
   IMU data based on motor RPM.
4. [imu-sensors](https://crates.io/crates/imu-sensors) - uses `Vector3f32` to return scaled gyro and acceleration readings from the IMU.
5. [protoflight](https://crates.io/crates/protoflight) - extensive use of vectors and quaternions.

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

## Architecture

See [ARCHITECTURE.md] for details on `vqm`'s internals.

[ARCHITECTURE.md]: ARCHITECTURE.md

## Original implementation

I originally implemented this crate as a C++ library:
[Library-VectorQuaternionMatrix](https://github.com/martinbudden/Library-VectorQuaternionMatrix).

The capabilities of this crate now exceed those of the original library.

## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
