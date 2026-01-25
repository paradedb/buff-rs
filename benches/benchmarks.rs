//! Benchmarks for BUFF encoding/decoding operations.
//!
//! Run with: `cargo bench`
//!
//! Criterion automatically saves baselines, so you can compare against
//! previous commits by running benchmarks before and after changes.
//! Use `cargo bench -- --save-baseline <name>` and `cargo bench -- --baseline <name>`
//! for explicit baseline management.

use buff_rs::BuffCodec;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

fn generate_test_data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            let base = (i as f64) / 100.0;
            base + (i as f64 * 0.001).sin() * 10.0
        })
        .collect()
}

fn generate_financial_data(size: usize) -> Vec<f64> {
    // Simulates stock prices with 2 decimal places
    (0..size)
        .map(|i| {
            let base = 100.0 + (i as f64 * 0.01).sin() * 50.0;
            (base * 100.0).round() / 100.0
        })
        .collect()
}

fn generate_sensor_data(size: usize) -> Vec<f64> {
    // Simulates temperature readings with 3 decimal places
    (0..size)
        .map(|i| {
            let base = 20.0 + (i as f64 * 0.005).sin() * 10.0;
            (base * 1000.0).round() / 1000.0
        })
        .collect()
}

// ============================================================================
// Core encode/decode benchmarks
// ============================================================================

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

// ============================================================================
// Query benchmarks (aggregations on compressed data)
// ============================================================================

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for size in [1000, 10000, 100000] {
        let data = generate_test_data(size);
        let codec = BuffCodec::new(1000);
        let encoded = codec.encode(&data).unwrap();

        group.throughput(Throughput::Elements(size as u64));

        // BUFF sum on compressed data
        group.bench_with_input(
            BenchmarkId::new("buff_compressed", size),
            &encoded,
            |b, encoded| b.iter(|| codec.sum(black_box(encoded))),
        );

        // Baseline: sum on raw f64 array
        group.bench_with_input(BenchmarkId::new("raw_f64", size), &data, |b, data| {
            b.iter(|| black_box(data).iter().sum::<f64>())
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

        // BUFF max on compressed data
        group.bench_with_input(
            BenchmarkId::new("buff_compressed", size),
            &encoded,
            |b, encoded| b.iter(|| codec.max(black_box(encoded))),
        );

        // Baseline: max on raw f64 array
        group.bench_with_input(BenchmarkId::new("raw_f64", size), &data, |b, data| {
            b.iter(|| {
                black_box(data)
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max)
            })
        });
    }

    group.finish();
}

// ============================================================================
// Special values benchmarks
// ============================================================================

fn bench_special_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_values");

    let codec = BuffCodec::new(1000);

    // Data with special values interspersed
    let data_with_special: Vec<f64> = (0..1000)
        .map(|i| match i % 50 {
            0 => f64::INFINITY,
            25 => f64::NEG_INFINITY,
            49 => f64::NAN,
            _ => (i as f64) / 100.0,
        })
        .collect();

    // Data without special values
    let data_regular: Vec<f64> = (0..1000).map(|i| (i as f64) / 100.0).collect();

    group.throughput(Throughput::Elements(1000));

    group.bench_function("encode_with_special", |b| {
        b.iter(|| codec.encode_with_special(black_box(&data_with_special)))
    });

    group.bench_function("encode_regular", |b| {
        b.iter(|| codec.encode(black_box(&data_regular)))
    });

    let encoded_special = codec.encode_with_special(&data_with_special).unwrap();
    let encoded_regular = codec.encode(&data_regular).unwrap();

    group.bench_function("decode_with_special", |b| {
        b.iter(|| codec.decode(black_box(&encoded_special)))
    });

    group.bench_function("decode_regular", |b| {
        b.iter(|| codec.decode(black_box(&encoded_regular)))
    });

    group.finish();
}

// ============================================================================
// Precision/scale benchmarks
// ============================================================================

fn bench_scales(c: &mut Criterion) {
    let mut group = c.benchmark_group("scales");
    let data = generate_test_data(10000);

    for scale in [10, 100, 1000, 10000, 100000] {
        let codec = BuffCodec::new(scale);

        group.throughput(Throughput::Elements(10000));
        group.bench_with_input(
            BenchmarkId::new("encode", format!("scale_{}", scale)),
            &data,
            |b, data| b.iter(|| codec.encode(black_box(data))),
        );
    }

    group.finish();
}

// ============================================================================
// Data pattern benchmarks
// ============================================================================

fn bench_data_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_patterns");
    let codec = BuffCodec::new(1000);
    let size = 10000;

    // Pattern 1: Financial data (narrow range, 2 decimal places)
    let financial = generate_financial_data(size);

    // Pattern 2: Sensor data (moderate range, 3 decimal places)
    let sensor = generate_sensor_data(size);

    // Pattern 3: Wide range data
    let wide_range: Vec<f64> = (0..size).map(|i| (i as f64) * 1000.0 + 0.123).collect();

    // Pattern 4: Constant values (best case for compression)
    let constant: Vec<f64> = vec![42.123; size];

    // Pattern 5: Random-ish distribution
    let random: Vec<f64> = (0..size)
        .map(|i| ((i * 1234567) % 100000) as f64 / 1000.0)
        .collect();

    group.throughput(Throughput::Elements(size as u64));

    for (name, data) in [
        ("financial", &financial),
        ("sensor", &sensor),
        ("wide_range", &wide_range),
        ("constant", &constant),
        ("random", &random),
    ] {
        group.bench_with_input(BenchmarkId::new("encode", name), data, |b, data| {
            b.iter(|| codec.encode(black_box(data)))
        });
    }

    group.finish();
}

// ============================================================================
// Compression ratio analysis (not a perf benchmark, but useful info)
// ============================================================================

fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_analysis");

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

// ============================================================================
// Comparison with decimal-bytes (when feature enabled)
// ============================================================================

#[cfg(feature = "decimal")]
mod decimal_comparison {
    use super::*;
    use decimal_bytes::{Decimal, Decimal64};
    use std::str::FromStr;

    pub fn bench_vs_decimal_bytes(c: &mut Criterion) {
        let mut group = c.benchmark_group("vs_decimal_bytes");

        let size = 1000;
        let codec = BuffCodec::new(1000); // 3 decimal places

        // Generate test data as f64
        let f64_data: Vec<f64> = (0..size)
            .map(|i| (i as f64) / 100.0 + (i as f64 * 0.01).sin())
            .collect();

        // Convert to Decimal strings for decimal-bytes
        let decimal_strings: Vec<String> = f64_data.iter().map(|f| format!("{:.3}", f)).collect();

        // Parse to Decimal values
        let decimals: Vec<Decimal> = decimal_strings
            .iter()
            .map(|s| Decimal::from_str(s).unwrap())
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        // BUFF: encode array
        group.bench_function("buff_encode_array", |b| {
            b.iter(|| codec.encode(black_box(&f64_data)))
        });

        // decimal-bytes: encode each value individually (more comparable to document store)
        group.bench_function("decimal_bytes_encode_individual", |b| {
            b.iter(|| {
                let _: Vec<&[u8]> = black_box(&decimals).iter().map(|d| d.as_bytes()).collect();
            })
        });

        // BUFF: decode array
        let buff_encoded = codec.encode(&f64_data).unwrap();
        group.bench_function("buff_decode_array", |b| {
            b.iter(|| codec.decode(black_box(&buff_encoded)))
        });

        // decimal-bytes: decode each value individually
        let decimal_bytes_list: Vec<Vec<u8>> =
            decimals.iter().map(|d| d.as_bytes().to_vec()).collect();
        group.bench_function("decimal_bytes_decode_individual", |b| {
            b.iter(|| {
                let _: Vec<Decimal> = black_box(&decimal_bytes_list)
                    .iter()
                    .map(|bytes| Decimal::from_bytes(bytes).unwrap())
                    .collect();
            })
        });

        // Size comparison
        let buff_size = buff_encoded.len();
        let decimal_bytes_total: usize = decimals.iter().map(|d| d.as_bytes().len()).sum();
        println!(
            "Storage for {} values: BUFF={} bytes, decimal-bytes={} bytes",
            size, buff_size, decimal_bytes_total
        );

        // BUFF interop: encode decimals via the decimal feature
        group.bench_function("buff_encode_decimals", |b| {
            b.iter(|| codec.encode_decimals(black_box(&decimals)))
        });

        // BUFF interop: decode to decimals
        group.bench_function("buff_decode_to_decimals", |b| {
            b.iter(|| codec.decode_to_decimals(black_box(&buff_encoded)))
        });

        group.finish();
    }

    pub fn bench_aggregation_comparison(c: &mut Criterion) {
        let mut group = c.benchmark_group("aggregation_comparison");

        let size = 10000;
        let codec = BuffCodec::new(1000);

        // Generate test data
        let f64_data: Vec<f64> = (0..size).map(|i| (i as f64) / 100.0).collect();
        let buff_encoded = codec.encode(&f64_data).unwrap();

        // Decimal versions
        let decimals: Vec<Decimal> = f64_data
            .iter()
            .map(|f| Decimal::from_str(&format!("{:.3}", f)).unwrap())
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        // BUFF sum (on compressed data)
        group.bench_function("buff_sum_compressed", |b| {
            b.iter(|| codec.sum(black_box(&buff_encoded)))
        });

        // Raw f64 sum
        group.bench_function("f64_sum", |b| {
            b.iter(|| black_box(&f64_data).iter().sum::<f64>())
        });

        // decimal-bytes: must decode + convert to do arithmetic
        // (decimal-bytes doesn't support native arithmetic, need to convert)
        group.bench_function("decimal_bytes_sum_via_parse", |b| {
            b.iter(|| {
                black_box(&decimals)
                    .iter()
                    .map(|d| d.to_string().parse::<f64>().unwrap())
                    .sum::<f64>()
            })
        });

        group.finish();
    }

    // ========================================================================
    // Decimal64 benchmarks
    // ========================================================================

    pub fn bench_decimal64(c: &mut Criterion) {
        let mut group = c.benchmark_group("decimal64");

        let size = 1000;
        let codec = BuffCodec::new(1000); // 3 decimal places
        let scale: u8 = 3;

        // Generate test data as f64
        let f64_data: Vec<f64> = (0..size)
            .map(|i| (i as f64) / 100.0 + (i as f64 * 0.01).sin())
            .collect();

        // Convert to Decimal64 values
        let decimal64s: Vec<Decimal64> = f64_data
            .iter()
            .map(|f| Decimal64::new(&format!("{:.3}", f), scale).unwrap())
            .collect();

        // Also create Decimal versions for comparison
        let decimals: Vec<Decimal> = f64_data
            .iter()
            .map(|f| Decimal::from_str(&format!("{:.3}", f)).unwrap())
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        // BUFF: encode Decimal64 array
        group.bench_function("buff_encode_decimal64s", |b| {
            b.iter(|| codec.encode_decimal64s(black_box(&decimal64s)))
        });

        // BUFF: encode Decimal array (for comparison)
        group.bench_function("buff_encode_decimals", |b| {
            b.iter(|| codec.encode_decimals(black_box(&decimals)))
        });

        // BUFF: encode f64 array (baseline)
        group.bench_function("buff_encode_f64", |b| {
            b.iter(|| codec.encode(black_box(&f64_data)))
        });

        // Decode benchmarks
        let buff_encoded = codec.encode(&f64_data).unwrap();

        group.bench_function("buff_decode_to_decimal64s", |b| {
            b.iter(|| codec.decode_to_decimal64s(black_box(&buff_encoded), scale))
        });

        group.bench_function("buff_decode_to_decimals", |b| {
            b.iter(|| codec.decode_to_decimals(black_box(&buff_encoded)))
        });

        group.bench_function("buff_decode_to_f64", |b| {
            b.iter(|| codec.decode(black_box(&buff_encoded)))
        });

        group.finish();
    }

    pub fn bench_decimal64_vs_decimal(c: &mut Criterion) {
        let mut group = c.benchmark_group("decimal64_vs_decimal");

        let size = 1000;
        let scale: u8 = 3;

        // Generate test data
        let values: Vec<String> = (0..size)
            .map(|i| format!("{:.3}", (i as f64) / 100.0))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        // Parse to Decimal64
        group.bench_function("parse_decimal64", |b| {
            b.iter(|| {
                values
                    .iter()
                    .map(|s| Decimal64::new(black_box(s), scale).unwrap())
                    .collect::<Vec<_>>()
            })
        });

        // Parse to Decimal
        group.bench_function("parse_decimal", |b| {
            b.iter(|| {
                values
                    .iter()
                    .map(|s| Decimal::from_str(black_box(s)).unwrap())
                    .collect::<Vec<_>>()
            })
        });

        // Create pre-parsed values
        let decimal64s: Vec<Decimal64> = values
            .iter()
            .map(|s| Decimal64::new(s, scale).unwrap())
            .collect();
        let decimals: Vec<Decimal> = values
            .iter()
            .map(|s| Decimal::from_str(s).unwrap())
            .collect();

        // to_string benchmarks
        group.bench_function("to_string_decimal64", |b| {
            b.iter(|| {
                black_box(&decimal64s)
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
            })
        });

        group.bench_function("to_string_decimal", |b| {
            b.iter(|| {
                black_box(&decimals)
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
            })
        });

        // Memory size comparison
        let decimal64_size = decimal64s.len() * std::mem::size_of::<Decimal64>();
        let decimal_bytes_size: usize = decimals.iter().map(|d| d.as_bytes().len()).sum();
        let decimal_stack_size = decimals.len() * std::mem::size_of::<Decimal>();

        println!("\nMemory for {} values:", size);
        println!("  Decimal64: {} bytes (fixed 8 bytes each)", decimal64_size);
        println!(
            "  Decimal:   {} bytes stack + {} bytes heap",
            decimal_stack_size, decimal_bytes_size
        );

        group.finish();
    }
}

#[cfg(feature = "decimal")]
use decimal_comparison::{
    bench_aggregation_comparison, bench_decimal64, bench_decimal64_vs_decimal,
    bench_vs_decimal_bytes,
};

// ============================================================================
// Criterion group configuration
// ============================================================================

#[cfg(not(feature = "decimal"))]
criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_roundtrip,
    bench_sum,
    bench_max,
    bench_special_values,
    bench_scales,
    bench_data_patterns,
    bench_compression_ratio,
);

#[cfg(feature = "decimal")]
criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_roundtrip,
    bench_sum,
    bench_max,
    bench_special_values,
    bench_scales,
    bench_data_patterns,
    bench_compression_ratio,
    bench_vs_decimal_bytes,
    bench_aggregation_comparison,
    bench_decimal64,
    bench_decimal64_vs_decimal,
);

criterion_main!(benches);
