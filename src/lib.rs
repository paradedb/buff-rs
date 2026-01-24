//! # buff-rs
//!
//! A Rust implementation of BUFF: Decomposed Bounded Floats for Fast Compression and Queries.
//!
//! Based on the VLDB 2021 paper: ["BUFF: Accelerating Queries in Memory through
//! Decomposed Bounded Floats"](https://dl.acm.org/doi/abs/10.14778/3476249.3476305).
//!
//! ## Overview
//!
//! BUFF provides efficient compression and query execution for bounded floating-point
//! data (data with a known precision/scale). It achieves this through:
//!
//! 1. **Precision Bounding**: Determining the minimum bits needed to represent values
//!    within a given precision tolerance
//! 2. **Byte Slicing**: Column-oriented storage where each bit position across all values
//!    is stored contiguously
//! 3. **Direct Query Execution**: Aggregate and filter operations can execute directly
//!    on compressed data
//!
//! ## Key Differences from `decimal-bytes`
//!
//! | Feature | decimal-bytes | buff-rs |
//! |---------|--------------|---------|
//! | Data Type | Single decimal values | Arrays of floats |
//! | Precision | Arbitrary (unlimited) | Bounded (fixed scale) |
//! | Storage | Row-oriented | Column-oriented (byte-sliced) |
//! | Primary Use | Document storage | Columnar/time-series data |
//! | Queries | Compare after decode | Query compressed data |
//!
//! ## Quick Start
//!
//! ```rust
//! use buff_rs::BuffCodec;
//!
//! // Create a codec with 3 decimal places of precision (scale=1000)
//! let codec = BuffCodec::new(1000);
//!
//! // Encode an array of f64 values
//! let data = vec![1.234, 5.678, 9.012, -3.456];
//! let encoded = codec.encode(&data).unwrap();
//!
//! // Decode back to f64
//! let decoded = codec.decode(&encoded).unwrap();
//!
//! // Query directly on compressed data
//! let sum = codec.sum(&encoded).unwrap();
//! let max = codec.max(&encoded).unwrap();
//! ```
//!
//! ## Choosing a Scale
//!
//! The scale determines the precision of encoded values:
//!
//! | Scale | Decimal Places | Example |
//! |-------|---------------|---------|
//! | 10 | 1 | 3.1 |
//! | 100 | 2 | 3.14 |
//! | 1000 | 3 | 3.142 |
//! | 10000 | 4 | 3.1416 |
//! | 100000 | 5 | 3.14159 |
//!
//! Choose a scale that matches your data's required precision. Higher scales
//! provide more precision but may reduce compression ratio.
//!
//! ## Compression Performance
//!
//! Compression ratio depends on data characteristics:
//!
//! - **Narrow range**: Better compression (fewer bits needed)
//! - **Wide range**: More bits needed, lower compression
//! - **Repetitive values**: Good compression
//!
//! Typical compression ratios range from 0.3x to 0.8x of the original size.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(dead_code)]

mod bitpack;
mod codec;
mod error;
pub mod precision;

#[cfg(feature = "decimal")]
mod decimal_interop;

pub use codec::{BuffCodec, BuffMetadata, SpecialValue, SpecialValueKind};
pub use error::BuffError;

#[cfg(feature = "decimal")]
pub use decimal_interop::DecimalArrayExt;

/// Convenience type alias for Results with BuffError.
pub type Result<T> = std::result::Result<T, BuffError>;
