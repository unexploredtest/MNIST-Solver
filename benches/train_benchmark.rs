use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neural_networks::train::train;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("train 1", |b| b.iter(|| train(black_box(10), black_box(10), black_box(3.0f32))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);