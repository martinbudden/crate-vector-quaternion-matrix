# vector-quaternion-matrix Rust Crate ![license](https://img.shields.io/badge/license-MIT-green) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![open source](https://badgen.net/badge/open/source/blue?icon=github)

## Vector, Quaternion, and Matrix Crate

This crate implements a variety of vector, quaternion, and matrix types. Each type has a generic version,
eg `Vector3d<T>`, and specific versions for `f32` and `f64`: eg `Vector3df32` and `Vector3df64`. Vectors and matrices
have 2d and 3d versions. So we have:

1. 2D vectors: `Vector2d<T>`, `Vector2df32`, `Vector2df64`
2. 3D vectors: `Vector3d<T>`, `Vector3df32`, `Vector3df64`
3. [quaternions](https://en.wikipedia.org/wiki/Quaternion): `Quaternion<T>`, `Quaternionf32`, `Quaternionf64`
4. 2D matrices: `Matrix2x2<T>`, `Matrix2x2f32`, `Matrix2x2f64`
5. 3D matrices: `Matrix3x3<T>`, `Matrix3x3f32`,`Matrix3x3f64`

This crate is `no_std`, that it does not link to the standard library and so does not depend on an operating system
and uses no allocation. This means it is suitable for embedded system.

## Mathematical methods and constants

This crate provides implementations of the trigonometric methods normally provided by the standard library, namely:
`sin`, `cos`, `sin_cos`, `tan`, `asin`, `acos`, `atan2`. The are provided in method_call syntax, ie `x.sin()`.

The methods `sqrt`, `reciprocal_sqrt` and `half_reciprocal_sqrt` are also provided.

The `MathConstants` trait provides a the standard mathematical constants in a form that can be used in generic code
ie `T:PI`.

## Wider system

This crates is part of a wider system of crates designed to work together to provide support for stabilized vehicles
including self-balancing robots and aircraft (including quadcopters).

Other crates that use or work with this crate include:

1. [imu-sensors](https://github.com/martinbudden/Crate-imu_sensors)
2. [filters](https://github.com/martinbudden/Crate-filters)
3. [sensor fusion](https://github.com/martinbudden/Crate-sensor_fusion)
4. [stabilized-vehicle](https://github.com/martinbudden/Crate-stabilized_vehicle)

So, for example:

1. the reading from an IMU (Inertial Measurement Unit) is typically a block of 12 bytes of raw data.
2. the imu-sensors crate has methods to convert this first to two `Vector3di16` values representing the gyro and acc reading.
3. it further scales these readings to produce two `Vector3df32` values.
4. in the stabilized-vehicle crate these values are filtered using (for example) a `BiquadFilterf32<Vector3df32>`
5. the filtered acc and gyro values are combined using sensor-fusion to provide an orientation quaternion.

## Why did I write another crate?

My requirements are fairly specific. I wanted a crate that could support the functionality required by stabilized vehicles in general,
and self balancing robots and aircraft in particular.

These requirements included:

1. Ability to align vectors to 16-byte boundaries for performance.
2. Support for `Vector3di16` and `Vector3di32` for reading data from Inertial Management Units (IMUs).
   Some IMUs can return 20-bit values, hence the need for `Vector3di32`.
3. Support for conversion between the integer vectors returned by the IMU and the floating point vectors required for mathematical operations.
4. Ability to work with generic filters.
5. Quaternion support for sensor fusion.
6. Matrix support for navigation, in particular "path following" (waypointing), and trilateration.

I looked at a number of alternatives including
[nalgebra](https://crates.io/crates/nalgebra),
[glam](https://crates.io/crates/glam),
and [micromath](https://crates.io/crates/micromath).
But each one of these fell on one count or another.
I had previously written a C++ VectorQuaternionMatrix library, so porting that to Rust seemed like the best option.

## Vector alignment

By default Vector3df32 is aligned to a 16-byte boundary. This is controlled by the `align` feature flag. To turn off
16-bit alignment, build with:

`cargo build --no-default-features --features libm`

## Original implementation

This crate was originally implemented as a C++ library. The
[original implementation can be found here](https://github.com/martinbudden/Library-VectorQuaternionMatrix).

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
