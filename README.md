# buff-rs

[![CI](https://github.com/paradedb/buff-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/paradedb/buff-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/paradedb/buff-rs/graph/badge.svg)](https://codecov.io/gh/paradedb/buff-rs)
[![Crates.io](https://img.shields.io/crates/v/buff-rs.svg)](https://crates.io/crates/buff-rs)
[![Documentation](https://docs.rs/buff-rs/badge.svg)](https://docs.rs/buff-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Rust implementation of **BUFF: Decomposed Bounded Floats for Fast Compression and Queries**.

Based on the VLDB 2021 paper: [BUFF: Accelerating Queries in Memory through Decomposed Bounded Floats](https://dl.acm.org/doi/abs/10.14778/3476249.3476305).

## Overview

BUFF provides efficient compression and query execution for bounded floating-point data. Unlike general-purpose compression, BUFF is designed specifically for numeric data with known precision bounds (e.g., sensor readings, financial data), enabling:

1. **Precision Bounding** - Determines minimum bits needed to represent values within a given tolerance
2. **Byte Slicing** - Column-oriented storage where each bit position is stored contiguously  
3. **Direct Queries** - Aggregate and filter operations execute directly on compressed data

## Features

- Encode arrays of `f64` values with configurable precision
- Decode back to `f64` with controlled precision loss
- Encode arrays of `decimal_bytes::Decimal` (arbitrary precision) or `Decimal64` (≤16 digits)
- Decode back to `Decimal` or `Decimal64` with controlled precision loss
- Query compressed data directly (sum, max, count)
- **Special value support**: Infinity, -Infinity, NaN
- Columnar storage layout optimized for analytical queries
- Optional `decimal-bytes` interop for PostgreSQL NUMERIC compatibility
- Zero required dependencies (only `thiserror`)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
buff-rs = "0.1"
```

### Optional Features

```toml
[dependencies]
# Enable decimal-bytes interop for PostgreSQL NUMERIC compatibility
buff-rs = { version = "0.1", features = ["decimal"] }
```

## Quick Start

```rust
use buff_rs::BuffCodec;

// Create a codec with 3 decimal places of precision (scale=1000)
let codec = BuffCodec::new(1000);

// Encode an array of f64 values
let data = vec![1.234, 5.678, 9.012, -3.456];
let encoded = codec.encode(&data).unwrap();

// Decode back to f64
let decoded = codec.decode(&encoded).unwrap();

// Query directly on compressed data
let sum = codec.sum(&encoded).unwrap();
let max = codec.max(&encoded).unwrap();
```

## Choosing a Scale

The scale determines the precision of encoded values:

| Scale | Decimal Places | Example Value |
|-------|---------------|---------------|
| 10 | 1 | 3.1 |
| 100 | 2 | 3.14 |
| 1000 | 3 | 3.142 |
| 10000 | 4 | 3.1416 |
| 100000 | 5 | 3.14159 |

Choose a scale that matches your data's required precision. Higher scales provide more precision but may reduce compression ratio.

## Special Values (Infinity, NaN)

BUFF supports special floating-point values:

```rust
use buff_rs::BuffCodec;

let codec = BuffCodec::new(1000);
let data = vec![1.0, f64::INFINITY, 2.0, f64::NAN, f64::NEG_INFINITY];

// Use encode_with_special for arrays containing special values
let encoded = codec.encode_with_special(&data).unwrap();
let decoded = codec.decode(&encoded).unwrap();

assert!(decoded[1].is_infinite());
assert!(decoded[3].is_nan());
```

## Decimal-bytes Interop (PostgreSQL NUMERIC)

Enable the `decimal` feature for `decimal-bytes` compatibility:

```toml
[dependencies]
buff-rs = { version = "0.1", features = ["decimal"] }
```

### With Decimal (Arbitrary Precision)

```rust
use buff_rs::BuffCodec;
use decimal_bytes::Decimal;

let codec = BuffCodec::new(1000);

// Encode Decimal values (with precision loss)
let decimals: Vec<Decimal> = vec![
    "1.234".parse().unwrap(),
    "5.678".parse().unwrap(),
];
let encoded = codec.encode_decimals(&decimals).unwrap();

// Decode back to Decimal
let decoded: Vec<Decimal> = codec.decode_to_decimals(&encoded).unwrap();
```

### With Decimal64 (≤16 digits, Fixed 8 Bytes)

`Decimal64` provides faster conversions for values that fit in 16 significant digits:

```rust
use buff_rs::BuffCodec;
use decimal_bytes::Decimal64;

let codec = BuffCodec::new(1000);

// Encode Decimal64 values
let decimals: Vec<Decimal64> = vec![
    Decimal64::new("1.234", 3).unwrap(),
    Decimal64::new("5.678", 3).unwrap(),
];
let encoded = codec.encode_decimal64s(&decimals).unwrap();

// Decode back to Decimal64 (specify output scale)
let decoded: Vec<Decimal64> = codec.decode_to_decimal64s(&encoded, 3).unwrap();
```

**Note**: Converting between BUFF and Decimal types involves precision loss because BUFF uses bounded floating-point representation while Decimal types use exact fixed-point representation.

## Performance

Key performance characteristics (run `cargo bench --features decimal` locally for your hardware):

### Core Operations

| Operation | Time | Throughput |
|-----------|------|------------|
| Encode (1K values) | 2.4 µs | 410 Melem/s |
| Encode (100K values) | 314 µs | 318 Melem/s |
| Decode (1K values) | 600 ns | 1.66 Gelem/s |
| Decode (100K values) | 138 µs | 725 Melem/s |
| Sum (10K, compressed) | 6.9 µs | 1.44 Gelem/s |

### Compression Ratio

| Scale | Precision | Compressed Size | Ratio |
|-------|-----------|-----------------|-------|
| 100 | 2 decimal places | 20 KB | 25% of original |
| 1000 | 3 decimal places | 30 KB | 37.5% of original |
| 10000 | 4 decimal places | 30 KB | 37.5% of original |

(For 10,000 f64 values = 80 KB uncompressed)

### Decimal64 vs Decimal Encoding

For 1,000 values with BUFF encoding:

| Operation | Decimal64 | Decimal | Speedup |
|-----------|-----------|---------|---------|
| BUFF encode | 4.0 µs | 74 µs | **18x** |
| BUFF decode to type | 213 µs | 189 µs | ~1x |
| BUFF decode to f64 | 600 ns | 600 ns | - |

### Memory Usage (1,000 values)

| Type | Storage |
|------|---------|
| BUFF compressed | **2,020 bytes** |
| Decimal (bytes) | 4,971 bytes |
| Decimal64 | 8,000 bytes (fixed 8 bytes each) |
| Decimal (stack+heap) | 24,000 + 4,807 bytes |

BUFF provides ~2.5x better compression than decimal-bytes and ~100x faster array decoding for columnar workloads.

## When to Use BUFF vs decimal-bytes

| Aspect | Decimal | Decimal64 | buff-rs |
|--------|---------|-----------|---------|
| **Data Type** | Single values | Single values | Arrays of floats |
| **Precision** | Unlimited | ≤16 digits | Bounded (fixed scale) |
| **Storage** | Variable | 8 bytes | Column-oriented |
| **Primary Use** | Document storage | Fixed-size storage | Columnar/time-series |
| **Query Style** | Decode then compare | Decode then compare | Query compressed |
| **Sortable Bytes** | Yes | No | No |

**Use `Decimal`** when:
- You need exact arbitrary-precision decimals (like PostgreSQL NUMERIC)
- Values are stored individually in documents
- You need lexicographically sortable byte representation

**Use `Decimal64`** when:
- You need fixed-size 8-byte storage with ≤16 digits precision
- You want embedded scale for self-describing values
- You need fast comparisons and conversions

**Use `buff-rs`** when:
- You have arrays of floating-point values with known precision
- You're building columnar storage or time-series databases
- You want to query compressed data without full decompression

## How It Works

### Precision Bounding

Given a precision tolerance (e.g., 0.001 for 3 decimal places), BUFF determines the minimum number of bits needed to represent each value. This is done by analyzing the IEEE 754 representation and finding the position where further bits don't affect precision.

### Byte Slicing

Instead of storing values row-by-row, BUFF stores them column-by-column at the byte level:

```
Traditional:  [v1_byte0, v1_byte1] [v2_byte0, v2_byte1] [v3_byte0, v3_byte1]
Byte-sliced:  [v1_byte0, v2_byte0, v3_byte0] [v1_byte1, v2_byte1, v3_byte1]
```

This layout enables SIMD-accelerated comparisons for range queries.

### Sign Flipping

To enable proper ordering of negative and positive values, the sign bit is flipped during encoding. This ensures that comparing encoded bytes yields correct numerical ordering.

## Compression Performance

Compression ratio depends on data characteristics:

- **Narrow value range**: Better compression (fewer bits needed for delta)
- **Wide value range**: More bits needed, lower compression
- **Repetitive values**: Good compression (single base value)

Typical compression ratios range from 0.3x to 0.8x of the original size (8 bytes per f64).

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- Chunwei Liu, et al. "Decomposed Bounded Floats for Fast Compression and Queries." VLDB 2021. [Paper](https://dl.acm.org/doi/abs/10.14778/3476249.3476305)
- Original implementation: [Tranway1/buff](https://github.com/Tranway1/buff)
