use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::{RngExt, rng};
use std::hint::black_box;

use vector_quaternion_matrix::{Quaternionf32, Vector3df32};

// see target/criterion/Matrix%20Math/report/index.html for results
fn bench_vq(c: &mut Criterion) {
    let mut group = c.benchmark_group("VQ");

    group.bench_function("v3d normalized", |b| {
        b.iter_batched(
            || {
                let a: [f32; 3] = rng().random();
                Vector3df32::from(a)
            },
            |v| black_box(v).normalized(),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("v3d normalized_u", |b| {
        b.iter_batched(
            || {
                let a: [f32; 3] = rng().random();
                Vector3df32::from(a)
            },
            |v| black_box(v).normalized_unchecked(),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("q normalized", |b| {
        b.iter_batched(
            || {
                let a: [f32; 4] = rng().random();
                Quaternionf32::from(a)
            },
            |q| black_box(q).normalized(),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("q normalized_u", |b| {
        b.iter_batched(
            || {
                let a: [f32; 4] = rng().random();
                Quaternionf32::from(a)
            },
            |q| black_box(q).normalized_unchecked(),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_vq);
criterion_main!(benches);
