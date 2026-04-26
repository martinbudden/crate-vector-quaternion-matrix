# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Releases of the form `0.1.n` do not adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
that is each release may contain incompatible API changes.

Once the API has stabilized this project will adopt semantic versioning, the first release to do so will be `0.2.0`.

## [Unreleased]

### Added

- May add `Matrix4x4`.
- May consider adding eigenvalues and eigenvectors.

### Changed

### Removed

### Deprecated

### Fixed

### Security

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

### Deprecated

At some point [0.1.0] will be [YANKED]

## [0.1.0] - 2023-03-05

Initial release.
