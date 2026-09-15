# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Releases of the form `0.1.n` do not adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
that is each release may contain incompatible API changes.

Once the API has stabilized this project will adopt semantic versioning, the first release to do so will be `0.2.0`.

## [Possible future]

### Added

- May consider adding eigenvalues and eigenvectors.

## [Unreleased]

### Deprecated

At some point [0.1.0] to [0.1.14] will be [YANKED]

## [0.1.17] - 2026-09-xx

### Added

- support for Continuous Integration.

### Changed

- split `serde` feature into `serde` and `storage`.
- updated to Rust version 1.89.

## [0.1.16] - 2026-09-05

### Added

- `add_diagonal`, `add_diagonal_scalar`, `add_diagonal_vector` and `add_diagonal_array` functions to matrices.
- `Matrix3x3xM2x2`.

### Changed

- renamed `mul_diag*` functions to `mul_diagonal*`.
- renamed `Matrix9` to `Matrix3x3xM3x3`.

### Removed

- `outer_product` from `Matrix9` and `Matrix9x9`.
- `multiply_9x3_by_3x3` and `extract_9x3_array` from `Matrix9x9`.

### Fixed

- fixed `enforce_symmetry` in `Matrix9`.

## [0.1.15] - 2026-09-01

### Added

- `Matrix9` a 9x9 matrix implemented as a flat array of nine 3x3 matrices.
- support for `postcard` `MaxSize`.
- matrix multiply by a diagonal matrix, and by a vector or array that represents a diagonal matrix.
- `asin`, `acos(`, `atan`,`atan2`, `exp`, `exp2`, `ln`, `log2`,`log10`, `log`, and `powf` approximations.
- `is_near_zero` functions to vectors and quaternion.

### Changed

- updated to `sequential-storage` `8.0.1`.
- improved `sin` and `cos` approximations.
- renamed `TrigonometricMethods` to `MathMethods` and included `SqrtMethods`.

### Removed

- `core` and `num-traits` consts from `MathConstants`
- `SqrtMethods`, now included in `MathMethods`.

## [0.1.14] - 2026-08-04

### Added

- column iterators to matrices.
- `from_diagonal_element` and `from_diagonal` constructors to matrices.
- `try_from_column_slice` constructors to matrices.
- `enforce_symmetry` to matrices.
- `approx_eq` to vectors.
- `epsilon` parameter to `is_near_zero` and `is_near_identity` matrix functions.

### Changed

- renamed `Vector2d` to `Vector2`, `Vector3d` to `Vector3`, and `Vector4d` to `Vector4`.
- changed matrices to use column-major storage.
- changed to use feature `align` rather than `no_align`.
- renamed matrix `fill` constructor to `from_element`.

## [0.1.13] - 2026-07-04

### Added

- `outer_product` functions for vectors.
- `fill` and `try_from_slice` constructors for matrices.
- multiply for `Matrix9x9`.
- Units Of Measurement (`uom`) support for vectors.
- `lerp` (linear interpolation) for vectors and quaternions.
- `outer_product`, `inverse`, and `half` functions for quaternions.

### Changed

- Moved documentation from `impl` blocks to directly before functions.
- Improved matrix row and column functions.
- Changed `from` functions to explicitly state whether from row or column.

### Removed

- `KalmanStateVector9`.
- `Vector3di16` and `Vector3di32`.
- `serde` support for `Matrix9x9`.

## [0.1.12] - 2026-06-08

### Changed

- Changed `Display` trait to be `std` only.

## [0.1.11] - 2026-05-31

### Changed

- Changed `TrigonometricMethods` trait to use `Num` trait rather than `Float` trait.
- Improved sqrt_reciprocal methods.

## [0.1.10] - 2026-05-27

### Added

- dot product to quaternion.
- `sequential-storage` support to `serde`.
- examples to `README.md`.
- `Display` trait to vectors.

### Fixed

- fixed `atan2` parameter order error..

## [0.1.9] - 2026-05-24

### Added

- `Deref` and `DerefMut` traits to matrices.
- `AsRef` and `AsMut` traits to matrices.
- range traits to matrices.
- iterator traits to matrices.
- custom `Debug` trait to `Matrix4x4` and `Matrix9x9`.
- improved `multiply_9x3_by_3x3` for `Matrix9x9`.

### Changed

- Changed Apache license to standard unabridged text.

### Removed

- multiplication from `Matrix9x9`.
- `One` and `ConstOne` traits from `Matrix9x9` (necessitated by removal of multiplication).

## [0.1.8] - 2026-05-23

### Added

- `Mnn` constants to index matrix elements (eg `M11`, `M23` etc).
- `identity` functions for matrices.
- `Matrix9x9` - partial implementation to support Kalman filters.
- `outer_product` function to matrices.
- `KalmanStateVector9` to support Kalman filters.
- `cos_tilt` and `sin_tilt` functions to `Quaternion`.
- conversion functions to `RollPitch` and `RollPitchYaw`.

### Changed

- optimized `m3x3_mul_vector` to be more compiler-friendly for generating SIMD instructions.
- renamed `try_invert` to `try_inverse`.
- tidied `zero` and `one` functions.
- improved documentation.

### Removed

- `katex-header.html`

## [0.1.7] - 2026-05-17

### Added

- `.cargo/config.toml`

### Changed

- default no longer includes `serde`.
- release build no longer dependent on `approx` crate.

### Fixed

- fixes to `Cargo.toml`, especially better handling of features.
- fixed bug in `quaternion` `rotate`.

## [0.1.6] - 2026-05-16

### Changed

- `serde` to use `default-features = false`.

## [0.1.5] - 2026-05-16

### Added

- `ConstZero` traits to vectors, quaternions, and matrices.
- `ConstOne` traits to quaternions and matrices.

### Removed

- `BitSet64` and `BitSet128`.

## [0.1.4] - 2026-05-13

### Added

- `Deserialize` and `Serialize` for `BitSet64` and `BitSet128`.

### Changed

- removed return value from `set` and `reset` in `BitSet`s.

## [0.1.3] - 2026-05-06

### Added

- `BitSet64Iter`.

### Changed

- removed return value from `set` and `reset` in `BitSet`s.

## [0.1.2]

### Added

- Quaternion calculate_euler_angles_radians() and calculate_euler_angles_degrees() functions
- Elementwise multiplication and division of vectors.

### Changed

- changed many functions from `#[inline(always)]` to `#[inline]`

## [0.1.1]

### Added

- `Vector4d`, `Vector3di32`.
- `to_radians` and `to_degrees` for vectors.
- `BitSet64` and `BitSet128`.
- Benchmarks.
- Many `From` traits to convert between vectors of different dimensions, arrays and matrices.
- **CHANGELOG.md**, **ARCHITECTURE.md**, **CONTRIBUTING.md**.

### Changed

- Renamed crate from `vector-quaternion-matrix` to `vqm`.
- Changed back to use verbal form for return functions and _in_place suffix for in place versions,
  ie `transpose` and `transpose_in_place` rather than `transpose` and `transposed`.
- Updated README.md

## [0.1.0] - 2026-03-05

Initial release.
