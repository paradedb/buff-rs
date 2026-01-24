# buff-rs

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
- Query compressed data directly (sum, max, count)
- Columnar storage layout optimized for analytical queries
- Zero external dependencies (only `thiserror`)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
buff-rs = "0.1"
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

## When to Use BUFF vs decimal-bytes

| Aspect | decimal-bytes | buff-rs |
|--------|--------------|---------|
| **Data Type** | Single decimal values | Arrays of floats |
| **Precision** | Arbitrary (unlimited digits) | Bounded (fixed scale) |
| **Storage Layout** | Row-oriented | Column-oriented (byte-sliced) |
| **Primary Use** | Document storage | Columnar/time-series data |
| **Query Style** | Decode then compare | Query compressed data |
| **Sortable Bytes** | Yes (lexicographic) | No (optimized for compression) |

**Use `decimal-bytes`** when:
- You need exact arbitrary-precision decimals (like PostgreSQL NUMERIC)
- Values are stored individually in documents
- You need lexicographically sortable byte representation

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
