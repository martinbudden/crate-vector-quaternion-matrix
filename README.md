# `vqm` Rust Crate<br>![License: MIT](https://img.shields.io/badge/license-MIT-green) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![open source](https://badgen.net/badge/open/source/blue?icon=github)

**vqm** is a lightweight, allocation-free, Rust math library for **vectors**, **quaternions**, and **matrices**.
It is designed for embedded systems, robotics, and real-time applications.

This crate is `no_std`, ie it does not link to the standard library, does not depend on an operating system, and uses no allocation.
This means it is suitable for embedded systems.

It also contains floating point approximations for common mathematical and trigonometry functions, enabling it to be used without `libm`.
By default the `libm` functions are used.

Minimum Supported Rust Version (MSRV): `Rust 1.89`.

## Why vqm?

**vqm** is intended for applications where you need practical vector, quaternion, and matrix mathematics without bringing
a large general-purpose framework into a small embedded application.

It is particularly suited to:

* Embedded control systems
* Robotics
* IMU and sensor processing
* Attitude estimation and orientation tracking
* Kalman filters
* Self-balancing vehicles
* Aircraft and other autonomous vehicles
* Real-time applications

The library uses fixed-size types and does not allocate memory.

## Overview

Vectors have 2D, 3D, and 4D versions.

Matrices have 2x2, 3x3, and 4x4 versions.

Additionally there are 3x3xM2x2 (effectively 6x6) and 3x3xM3x3 (effectively 9x9) matrices which have been added to support Kalman filters.
They are used by the [sensor-fusion](https://crates.io/crates/sensor-fusion) crate to implement Kalman filters.

Each type has versions for `f32` and `f64`. So we have:

1. 2D vectors: `Vector2f32`, `Vector2f64`
2. 3D vectors: `Vector3f32`, `Vector3f64`
3. 4D vectors: `Vector4f32`, `Vector4f64`
4. [quaternions](https://en.wikipedia.org/wiki/Quaternion): `Quaternionf32`, `Quaternionf64`
5. 2x2 matrices: `Matrix2x2f32`, `Matrix2x2f64`
6. 3x3 matrices: `Matrix3x3f32`, `Matrix3x3f64`
7. 4x4 matrices: `Matrix4x4f32`, `Matrix4x4f64`
8. 3x3xM2x2 matrices: `Matrix3x3xM2x2f32`, `Matrix3x3xM2x2f64`
9. 3x3xM3x3 matrices: `Matrix3x3xM3x3f32`, `Matrix3x3xM3x3f64`

Quaternions are implemented using the Hamilton convention.

Matrix elements are stored in a one-dimensional array in column-major order.
`Matrix3x3xM2x2`, and `Matrix3x3xM3x3` are partial implementations to support Kalman filters.

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
They are accurate to 4 or more significant figures, depending on the function.

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
let v = Vector3f32 { x: 1.0, y: 2.0, z: 3.0 };
let u = m * v;

// quaternions
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

## Comparison with other crates

There are currently a number of Rust crates that support vector math, quaternions, and matrices. The most notable being
[nalgebra](https://crates.io/crates/nalgebra), [glam](https://crates.io/crates/glam), [vek](https://crates.io/crates/vek),
[ultraviolet](https://crates.io/crates/ultraviolet), and [micromath](https://crates.io/crates/micromath).

Of these `glam`, `vek` and `ultraviolet` are focused primarily on graphics and gaming.

Comparing `vqm` with the remaining two, `micromath` and `nalgebra`, we have:

|                                 | **vqm**                               | **micromath**                         | **nalgebra**                               |
|---------------------------------|---------------------------------------|---------------------------------------|--------------------------------------------|
| Primary focus                   |Embedded<br>linear algebra,<br>robotics|Small, fast<br>embedded<br>mathematics | General-purpose<br>linear algebra          |
| `no_std`                        | Yes                                   | Yes                                   | Yes                                        |
| Heap allocation<br>required     | No                                    | No                                    |No for static types<br>Yes for dynamic types|
| Vectors                         | 2D, 3D, 4D                            | 2D, 3D                                | 1D-6D static<br>any size dynamic           |
| Fixed-size matrices             | 2×2, 3×3, 4×4                         | ——                                    | Extensive                                  |
| Dynamic matrices                | ——                                    | ——                                    | Yes                                        |
| Quaternions                     | Yes                                   | Yes                                   | Yes                                        |
| `f32`                           | Yes                                   | Yes                                   | Yes                                        |
| `f64`                           | Yes                                   | ——                                    | Yes                                        |
| Approximate<br>math functions   | Yes<br>Fast<br>Accuracy: 4+ SF¹       | Core focus<br>Faster<br>Less accurate²| ——                                         |
| Robotics-oriented<br>operations | Yes                                   | Some                                  | Yes                                        |
| Kalman-filter<br>oriented types | Yes                                   | ——                                    | ——                                         |
| Units of measure                | Optional `uom`                        | ——                                    | ——                                         |
| Serialization                   | Optional                              | ——                                    | Optional                                   |
| General<br>linear algebra       | Focused                               | Limited                               | Extensive                                  |
| MSRV                            | 2024 v1.89                            | 2018 v1.47                            | 2024 v1.89                                 |

(¹) - preliminary testing indicates that:

1. `sqrt`, `sqrt_reciprocal`, `asin`, `acos` - accurate to 4 SF.
2. `atan`, `atan2` - accurate to 5 SF.
3. other functions are accurate to 6+ SF.

Although this has not been formally verified.

(²) - `micromath` documentation states:

1. `sin`, `cos` - maximum error 0.002.
2. `tan` - maximum error 0.6.
3. `sqrt`, `invsqrt` - average deviation ~5%,

`vqm` aims to use the same function names as `nalgebra` (eg `try_inverse` rather than `invert` or `inverted`).
This reduces the cognitive load if:

1. You are using `vqm` for the first time and you are familiar with `nalgebra`.
2. Your project outgrows `vqm` and you want to switch to `nalgebra`.
3. You have a hybrid project, using `vqm` for real-time low level operations (eg reading an IMU and sensor fusion)
   and using `nalgebra` for higher level functions (eg navigation).

## Units of Measurement (uom) support

The optional `uom` feature integrates `vqm` with the units-of-measurement crate, [uom](https://crates.io/crates/uom).

This allows vectors and matrices to carry physical units and lets the type system catch incompatible operations.

For example, vectors can contain lengths rather than bare floating-point values.

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

## Features

All features except `libm` are off by default. The full set, including those described above, is:

* `libm` - enabled by default, uses `libm` math functions. When disabled `vqm` math function approximations are used.
* `serde` - implementations of `Serialize` and `Deserialize` `MaxSize` for all `vqm` types.
* `storage` - adds [sequential-storage](https://crates.io/crates/sequential-storage) support for storing data in flash with minimal erase cycles.
* `uom` - [Units Of Measurement](https://crates.io/crates/uom) support.
* `align` - aligns larger `struct`s to 16-byte boundaries.
* `std` - uses `std` math functions.
* `simd` - (experimental) enables SIMD support via the [portable simd](https://doc.rust-lang.org/core/simd/index.html) module.
         This requires the `align` feature and the nightly Rust toolchain.

## Specializations

`vqm` includes a number of specializations that you might not find in your typical linear algebra/graphics/math library.

A specialization may be considered for inclusion in `vqm` if it provides useful functionality to a library that uses `vqm`.

A specialization generally won't be considered for inclusion to support a single application.

### Bare metal (that is `no_std` and no `libm`)

`vqm` includes implementations for math functions, including square root and trigonometric functions, that allow it to run without `std` and `libm`.

### Robotics support

`vqm` has additional functionality specifically to support robotics applications. This includes:

1. `Vector3f32` functions to load sensor data from a `[u8; 6]`.
2. `RollPitch` and `RollPitchYaw` structs.
3. `to_radians` and `to_degrees` convenience functions for vectors and `RollPitch` and `RollPitchYaw` structs.
4. Quaternion utility functions such as `cos_tilt` and `gravity`.
5. Matrices specifically to support Kalman filters.

### Kalman filters

`vqm` has additional functionality specifically to support Kalman filters. This includes:

1. `Matrix3x3xM2x2` - a 3x3 matrix matrix of `Matrix2x2`s (so effectively a 6x6 matrix).
2. `Matrix3x3xM3x3` - a 3x3 matrix matrix of `Matrix3x3`s (so effectively a 9x9 matrix).
3. `Matrix3x3::mul_diagonal_vector` - multiplies a vector which is treated as a diagonal matrix by a matrix.
4. `Matrix3x3::add_diagonal_vector` - adds a vector which is treated as a diagonal matrix to a matrix.

## Usage by other crates

`vqm` is used by a number of other Rust crates, in particular:

1. [signal-filters](https://crates.io/crates/signal-filters)
   `BiquadFilters` and `PT` filters are templated and provide vector implementations for parallel filtering of values on all axes.
2. [sensor-fusion](https://crates.io/crates/sensor-fusion) - extensively uses vectors, quaternions, and matrices for sensor fusion filters
3. [motor-mixers](https://crates.io/crates/motor-mixers) - uses `Vector3f32` and `BiquadFilterVector3f32` in notch filters to filter
   IMU data based on motor RPM.
4. [imu-sensors](https://crates.io/crates/imu-sensors) - uses `Vector3f32` to return scaled gyro and acceleration readings from the IMU.
5. [protoflight](https://crates.io/crates/protoflight) - extensive use of vectors and quaternions.

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
