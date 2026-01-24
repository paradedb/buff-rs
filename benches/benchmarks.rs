//! Benchmarks for BUFF encoding/decoding operations.

use buff_rs::BuffCodec;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn generate_test_data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            let base = (i as f64) / 100.0;
            base + (i as f64 * 0.001).sin() * 10.0
        })
        .collect()
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    for size in [1000, 10000, 100000] {
        let data = generate_test_data(size);
        let codec = BuffCodec::new(1000);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| codec.encode(black_box(data)))
        });
    }

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    for size in [1000, 10000, 100000] {
        let data = generate_test_data(size);
        let codec = BuffCodec::new(1000);
        let encoded = codec.encode(&data).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &encoded, |b, encoded| {
            b.iter(|| codec.decode(black_box(encoded)))
        });
    }

    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for size in [1000, 10000, 100000] {
        let data = generate_test_data(size);
        let codec = BuffCodec::new(1000);
        let encoded = codec.encode(&data).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &encoded, |b, encoded| {
            b.iter(|| codec.sum(black_box(encoded)))
        });
    }

    group.finish();
}

fn bench_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("max");

    for size in [1000, 10000, 100000] {
        let data = generate_test_data(size);
        let codec = BuffCodec::new(1000);
        let encoded = codec.encode(&data).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &encoded, |b, encoded| {
            b.iter(|| codec.max(black_box(encoded)))
        });
    }

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    for size in [1000, 10000, 100000] {
        let data = generate_test_data(size);
        let codec = BuffCodec::new(1000);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                let encoded = codec.encode(black_box(data)).unwrap();
                codec.decode(&encoded)
            })
        });
    }

    group.finish();
}

fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio");

    for scale in [100, 1000, 10000] {
        let data = generate_test_data(10000);
        let codec = BuffCodec::new(scale);
        let encoded = codec.encode(&data).unwrap();

        let original_size = data.len() * std::mem::size_of::<f64>();
        let compressed_size = encoded.len();
        let ratio = compressed_size as f64 / original_size as f64;

        println!(
            "Scale {}: {} bytes -> {} bytes (ratio: {:.3})",
            scale, original_size, compressed_size, ratio
        );

        group.throughput(Throughput::Bytes(original_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("scale_{}", scale)),
            &data,
            |b, data| b.iter(|| codec.encode(black_box(data))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_sum,
    bench_max,
    bench_roundtrip,
    bench_compression_ratio
);
criterion_main!(benches);
