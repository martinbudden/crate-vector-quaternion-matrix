#![warn(clippy::pedantic)]
#![warn(unused_results)]

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, rng};
use std::hint::black_box;

use vqm::{Quaternionf32, Vector3df32};

// see target/criterion/Matrix%20Math/report/index.html for results

// # Replace 'v3d_bench' with the name defined in your Cargo.toml [[bench]] section
// RUSTFLAGS="-C target-cpu=native" cargo asm --bench vq_bench "mul_add"

fn bench_vq(c: &mut Criterion) {
    let mut group = c.benchmark_group("VQ");

    _ = group.throughput(Throughput::Elements(1));

    _ = group.bench_function("v3d_add", |b| {
        b.iter_batched(
            || {
                // Setup: Generate two random vectors
                let a1: [f32; 3] = rng().random();
                let a2: [f32; 3] = rng().random();
                let v1 = Vector3df32::from(a1);
                let v2 = Vector3df32::from(a2);
                (v1, v2)
            },
            |(v1, v2)| black_box(v1) + black_box(v2),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("v3d_mul_k", |b| {
        b.iter_batched(
            || {
                // Setup: Generate two random vectors
                let a: [f32; 3] = rng().random();
                let v = Vector3df32::from(a);
                let k: f32 = rng().random();
                (v, k)
            },
            |(v, k)| black_box(v) * black_box(k),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("v3d_mu_add_*_+", |b| {
        b.iter_batched(
            || {
                // Setup: Generate two random vectors
                let a1: [f32; 3] = rng().random();
                let a2: [f32; 3] = rng().random();
                let v1 = Vector3df32::from(a1);
                let v2 = Vector3df32::from(a2);
                let k: f32 = rng().random();
                (v1, v2, k)
            },
            |(v1, v2, k)| black_box(v1) * black_box(k) + black_box(v2),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("v3d_mul_add", |b| {
        b.iter_batched(
            || {
                // Setup: Generate two random vectors
                let a1: [f32; 3] = rng().random();
                let a2: [f32; 3] = rng().random();
                let v1 = Vector3df32::from(a1);
                let v2 = Vector3df32::from(a2);
                let k: f32 = rng().random();
                (v1, v2, k)
            },
            |(v1, v2, k)| {
                use num_traits::MulAdd;
                // NOTE no semicolon so the result is returned to the benchmark harness
                black_box(v1).mul_add(black_box(k), black_box(v2))
            },
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("v3d normalized", |b| {
        b.iter_batched(
            || {
                let a: [f32; 3] = rng().random();
                Vector3df32::from(a)
            },
            |v| black_box(v).normalized(),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("v3d normalized_u", |b| {
        b.iter_batched(
            || {
                let a: [f32; 3] = rng().random();
                Vector3df32::from(a)
            },
            |v| black_box(v).normalized_unchecked(),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("q normalized", |b| {
        b.iter_batched(
            || {
                let a: [f32; 4] = rng().random();
                Quaternionf32::from(a)
            },
            |q| black_box(q).normalized(),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("q normalized_u", |b| {
        b.iter_batched(
            || {
                let a: [f32; 4] = rng().random();
                Quaternionf32::from(a)
            },
            |q| black_box(q).normalized_unchecked(),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_vq);
criterion_main!(benches);
