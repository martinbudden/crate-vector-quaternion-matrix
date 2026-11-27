#![warn(clippy::pedantic)]
#![warn(unused_results)]

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, rng};
use std::hint::black_box;

use vqm::{Quaternionf32, Vector3f32};

// see target/criterion/Matrix%20Math/report/index.html for results

// # Replace 'v3_bench' with the name defined in Cargo.toml [[bench]] section
// RUSTFLAGS="-C target-cpu=native" cargo asm --bench vq_bench "mul_add"

#[allow(clippy::too_many_lines)]
fn bench_vq(c: &mut Criterion) {
    let mut group = c.benchmark_group("VQ");

    _ = group.throughput(Throughput::Elements(1));

    _ = group.bench_function("v3_add", |b| {
        b.iter_batched(
            || {
                // Setup: Generate two random vectors
                let a1: [f32; 3] = rng().random();
                let a2: [f32; 3] = rng().random();
                let v1 = Vector3f32::from(a1);
                let v2 = Vector3f32::from(a2);
                (v1, v2)
            },
            |(v1, v2)| black_box(v1) + black_box(v2),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("v3_mul_k", |b| {
        b.iter_batched(
            || {
                // Setup: Generate two random vectors
                let a: [f32; 3] = rng().random();
                let v = Vector3f32::from(a);
                let k: f32 = rng().random();
                (v, k)
            },
            |(v, k)| black_box(v) * black_box(k),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("v3_mul_add_*_+", |b| {
        b.iter_batched(
            || {
                // Setup: Generate two random vectors
                let a1: [f32; 3] = rng().random();
                let a2: [f32; 3] = rng().random();
                let v1 = Vector3f32::from(a1);
                let v2 = Vector3f32::from(a2);
                let k: f32 = rng().random();
                (v1, v2, k)
            },
            |(v1, v2, k)| black_box(v1) * black_box(k) + black_box(v2),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("v3_mul_add", |b| {
        b.iter_batched(
            || {
                // Setup: Generate two random vectors
                let a1: [f32; 3] = rng().random();
                let a2: [f32; 3] = rng().random();
                let v1 = Vector3f32::from(a1);
                let v2 = Vector3f32::from(a2);
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

    _ = group.bench_function("v3 normalize", |b| {
        b.iter_batched(
            || {
                let a: [f32; 3] = rng().random();
                Vector3f32::from(a)
            },
            |v| black_box(v).normalize(),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("v3 normalize_u", |b| {
        b.iter_batched(
            || {
                loop {
                    let a: [f32; 3] = rng().random();
                    let v = Vector3f32::from(a);
                    // ensure that v is normalizable.
                    if v.norm_squared() > 4.0 * f32::EPSILON {
                        break v;
                    }
                }
            },
            |v| black_box(v).normalize_unchecked(),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("q normalize", |b| {
        b.iter_batched(
            || {
                let a: [f32; 4] = rng().random();
                Quaternionf32::from(a)
            },
            |q| black_box(q).normalize(),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("q normalize_ip", |b| {
        b.iter_batched(
            || {
                let a: [f32; 4] = rng().random();
                Quaternionf32::from(a)
            },
            |q| black_box(*black_box(q).normalize_in_place()),
            BatchSize::SmallInput,
        );
    });
    _ = group.bench_function("q normalize_ip_u", |b| {
        b.iter_batched(
            || {
                loop {
                    let a: [f32; 4] = rng().random();
                    let q = Quaternionf32::from(a);
                    // ensure that q is normalizable.
                    if q.norm_squared() > 4.0 * f32::EPSILON {
                        break q;
                    }
                }
            },
            |q| black_box(q).normalized_unchecked(),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_vq);
criterion_main!(benches);
