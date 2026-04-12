# Architecture

This document describes the overall design goals of `vqm` (`vector-quaternion-matrix`).

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

### Align parameters and pass by value, not by reference

This is more peformant.

Most of the calculations are performed on `Vector3df32` and `Quaternionf32` types.

If these are aligned they are 16-byte values and fit comfortably into the CPUs floating point registers.
What's more, if calculations are chained, then the intermediate values can be retained in the CPU registers,
rather than being copied on and off the stack (which might be the case if passed by reference).

Passing by value is also preferable when SIMD is used.

The largest item this library may contain is an `Matrix4x4f64` which is 128 bytes, which is well below the
256-byte rust-recommended limit for passing by value.

### Use generics

For the main processing loop `Vector3df32` and `Quaternionf32` was required.

To support reading from an Inertial Measurement Unit (IMU), `Vector3di16` was required.

Initially I started implementing `Vector3df32` and `Vector3di16` as separate types,
but during implementation and especially when I started adding SIMD support it became
clear that using generics would make for a simpler implementation.

### Use boilerplate rather than code generation

My experience is that libraries that heavily rely on code generation are difficult to use (and debug into).

Using boilerplate makes things harder for the developer of the library, but easier for the user of the library.

There is significant boilerplating between (eg) `Vector2d`, `Vector3d`, `Vector4d`, and `Quaternion`,
but this is manageable and not excessive.

### Follow standards

1. Follow Rust [standard library] conventions and [API guidelines] where possible
2. High quality [rustdoc] generated document

[standard library]: https://doc.rust-lang.org/std/index.html
[API guidelines]: https://rust-lang.github.io/api-guidelines
[rustdoc]: https://doc.rust-lang.org/rustdoc/index.html

I have not yet started using newtypes to ensure type safety between different measurement units
(eg degrees and radians).
