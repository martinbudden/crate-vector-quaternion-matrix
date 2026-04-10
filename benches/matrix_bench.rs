use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::{RngExt, rng};
use std::hint::black_box;

use vector_quaternion_matrix::{Matrix2x2f32, Matrix3x3f32};

// see target/criterion/Matrix%20Math/report/index.html for results
fn bench_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix");

    group.bench_function("m2x2 determinant", |b| {
        b.iter_batched(
            || {
                let data: [f32; 4] = rng().random();
                Matrix2x2f32::new(data)
            },
            |m| black_box(m).determinant(),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("m3x3 determinant", |b| {
        b.iter_batched(
            || {
                let data: [f32; 9] = rng().random();
                Matrix3x3f32::new(data)
            },
            |m| black_box(m).determinant(),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("m3x3 mul", |b| {
        b.iter_batched(
            || {
                // Setup: Generate two random matrices
                let m1 = Matrix3x3f32::new(rand::rng().random());
                let m2 = Matrix3x3f32::new(rand::rng().random());
                (m1, m2)
            },
            |(m1, m2)| {
                // Timed Routine: Using the * operator
                // black_box hides the inputs from the compiler
                black_box(m1) * black_box(m2)
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("m3x3 inverted", |b| {
        b.iter_batched(
            || {
                let mut my_rng = rand::rng();
                loop {
                    let m = Matrix3x3f32::new(my_rng.random());
                    // We check the determinant; if it's near zero, we try again
                    if m.determinant().abs() > f32::EPSILON {
                        break m;
                    }
                }
            },
            |m| black_box(m).inverted(),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_matrix);
criterion_main!(benches);
