#![warn(clippy::pedantic)]
#![warn(unused_results)]

use core::f32::consts::PI;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, rng};
//use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::hint::black_box; // Use core::f32 for no_std

use vector_quaternion_matrix::math_methods::{sin_approx,cos_approx,sin_cos_approx};

// See: target/criterion/Matrix%20Math/report/index.html for results

fn bench_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("Math");

    _ = group.throughput(Throughput::Elements(1));

    _ = group.bench_function("sin", |b| {
        b.iter_batched(|| rng().random_range(-PI / 2.0..PI / 2.0), |x| libm::sinf(black_box(x)), BatchSize::SmallInput);
    });

    _ = group.bench_function("sin_approx", |b| {
        b.iter_batched(|| rng().random_range(-PI / 2.0..PI / 2.0), |x| sin_approx(black_box(x)), BatchSize::SmallInput);
    });

    _ = group.bench_function("cos", |b| {
        b.iter_batched(|| rng().random_range(-PI / 2.0..PI / 2.0), |x| libm::cosf(black_box(x)), BatchSize::SmallInput);
    });

    _ = group.bench_function("cos_approx", |b| {
        b.iter_batched(|| rng().random_range(-PI / 2.0..PI / 2.0), |x| cos_approx(black_box(x)), BatchSize::SmallInput);
    });

    _ = group.bench_function("sin_cos", |b| {
        b.iter_batched(|| rng().random_range(-PI / 2.0..PI / 2.0), |x| libm::sincosf(black_box(x)), BatchSize::SmallInput);
    });

    _ = group.bench_function("sin_cos_approx", |b| {
        b.iter_batched(|| rng().random_range(-PI / 2.0..PI / 2.0), |x| sin_cos_approx(black_box(x)), BatchSize::SmallInput);
    });
    group.finish();
}

criterion_group!(benches, bench_math);
criterion_main!(benches);
