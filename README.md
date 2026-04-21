# `vqm` Rust Crate<br>![license](https://img.shields.io/badge/license-MIT-green) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![open source](https://badgen.net/badge/open/source/blue?icon=github)

A **vector**, **quaternion**, and **matrix** (**VQM**) library targeted at embedded systems and robotics.
(In particular stabilized vehicles including self-balancing robots and aircraft).

This crate is `no_std`, that it does not link to the standard library and so does not depend on an operating system
and uses no allocation. This means it is suitable for embedded system.

## Overview

Vectors have 2D, 3D, and 4D versions.

Matrices have 2x2 and 3x3 versions.

Each type has versions for `f32` and `f64`. So we have:

1. 2D vectors: `Vector2df32`, `Vector2df64`
2. 3D vectors: `Vector3df32`, `Vector3df64`
3. 4D vectors: `Vector4df32`, `Vector4df64`
4. [quaternions](https://en.wikipedia.org/wiki/Quaternion): `Quaternionf32`, `Quaternionf64`
5. 2x2 matrices: `Matrix2x2f32`, `Matrix2x2f64`
6. 3x3 matrices: `Matrix3x3f32`,`Matrix3x3f64`

The 3D vector additionally has `i16` and `i32` versions: `Vector3di16` and `Vector3di32`.

(Under the hood, types are implemented using generics, so `Vector3df32` is actually `Vector3d<f32>`,
but that is transparent to the user.)

## Mathematical methods and constants

This crate also provides implementations of the trigonometric methods normally provided by the standard library, namely:
`sin`, `cos`, `sin_cos`, `tan`, `asin`, `acos`, `atan2`. The are provided in `method_call` syntax, ie `x.sin()`.

The methods `sqrt` and `sqrt_reciprocal` are also provided.

The `MathConstants` trait provides a the standard mathematical constants in a form that can be used in generic code
ie `T:PI`.

## SIMD support

**SIMD** support can be enabled with the `simd` feature.

It is currently experimental and many of the implementations are naive "placeholder" implementations to be optimized at a later date.
These placeholder implementations may be slower than the non-SIMD code, so if you used SIMD make sure you benchmark to show
that you are indeed getting a performance improvement.

This uses [portable simd](https://doc.rust-lang.org/core/simd/index.html), which requires the nightly compiler, since it is still
unstable in rust.

This can be invoked using `rustup`, eg:

```sh
rustup run nightly cargo build --features simd --target thumbv8m.main-none-eabi
```

## Struct alignment

By default `Vector3df32` is aligned to a 16-byte boundary, and `Matrix3x3f32` is aligned to a 32-byte boundary.
This can be turned off using the `no_align` feature flag:

```sh
cargo build -features no_align
```

If `no_align` is used then SIMD support is not available.

## Naming convention for "return" and "in-place" versions of functions

Several functions (eg `normalize`, `clamp`, `transpose`, `adjugate`, and `invert`) have both "return" and "in-place" forms.
The convention used is that the verbal name (eg `normalize`) is used for the in-place form, and the adjectival name (eg `normalized`)
is used for the return form. So we have:

```rust
# use vqm::Vector3df32;
// return form:
    let v = Vector3df32::new(2.0, 3.0, 5.0);
    let n = v.normalized();
// in-place form:
    let mut v = Vector3df32::new(2.0, 3.0, 5.0);
    v.normalize();
```

## Invert or Inverse?

As illustrated below, there is no universal convention for the name of the function used to invert a matrix.
Sometimes `invert` is used, sometimes `inverse`.

| Crate                                               | return                              | in-place                                 |
| --------------------------------------------------- | ----------------------------------- | ---------------------------------------- |
| vqm                                                 | `fn inverted(self) -> Self;`        | `fn invert(&mut self) -> &mut Self;`     |
| [vek](https://crates.io/crates/vek)                 | `fn inverted(self) -> Self;`        | `fn invert(&mut self);`                  |
| [ultraviolet](https://crates.io/crates/ultraviolet) | `fn inversed(&self) -> Self;`       | `fn inverse(&mut self);`                 |
| [glam](https://crates.io/crates/glam)               | `fn inverse(&self) -> Self;`        | N/A                                      |
| [static-math](https://crates.io/crates/static-math) | `fn inverse(&self) -> Option<>;`    | N/A                                      |
| [cg-math](https://crates.io/crates/cg-math)         | `fn invert(&self) -> Option<>;`     | N/A                                      |
| [nalgebra](https://crates.io/crates/nalgebra)       | `fn try_inverse(self) -> Option<>;` | `fn try_inverse_mut(&mut self) -> bool;` |

Note that only `vqm` and `vek` pass parameters by value.

When both return and in-place forms are supported, the choice is to use `inverted` and `invert` or `inversed` and `inverse`.
I prefer `inverse` to `invert`, but I really don't like `inversed`. So VQM uses `inverted` and `invert`.

## Architecture

See [ARCHITECTURE.md] for details on `vqm`'s internals.

[ARCHITECTURE.md]: ARCHITECTURE.md

## Future directions

Apart from implementing Matrix4x4, I have no planned major extensions to this crate.

My main focus will be on fixing any bugs that come up in usage.

I may at some point add the ability to calculate eigenvalues and eigenvectors.

I may do some work on improving the performance of the SIMD implementations.

## Original implementation

I originally implemented this crate as a C++ library:
[Library-VectorQuaternionMatrix](https://github.com/martinbudden/Library-VectorQuaternionMatrix).

The capabilities of this crate now exceed those of the original library.

## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
