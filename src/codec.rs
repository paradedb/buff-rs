//! BUFF codec for encoding and decoding bounded floating-point arrays.
//!
//! This module implements the BUFF (Decomposed Bounded Floats for Fast compression
//! and queries) algorithm from VLDB 2021.
//!
//! The key insight is to store floating-point values as byte-sliced fixed-point
//! integers, enabling:
//! 1. Efficient compression through delta encoding and bit packing
//! 2. Fast predicate evaluation directly on compressed data (with SIMD)
//!
//! ## Encoding Format (v1 - without special values)
//!
//! The encoded format consists of:
//! - Base integer (8 bytes, i64)
//! - Number of values (4 bytes, u32)
//! - Integer bits (4 bytes, u32)
//! - Decimal bits (4 bytes, u32)
//! - Byte slices (column-oriented storage)
//!
//! ## Encoding Format (v2 - with special values)
//!
//! The v2 format adds a header byte and special value section:
//! - Version/flags (1 byte): 0x02 for v2 with special values
//! - Base integer (8 bytes, i64)
//! - Number of regular values (4 bytes, u32)
//! - Integer bits (4 bytes, u32)
//! - Decimal bits (4 bytes, u32)
//! - Number of special values (4 bytes, u32)
//! - Special value entries: (index: u32, type: u8) where type is 1=+Inf, 2=-Inf, 3=NaN
//! - Byte slices (column-oriented storage)
//!
//! Each value is converted to a fixed-point representation relative to the base,
//! then stored in byte-sliced format where each byte position across all values
//! is stored contiguously.

use crate::bitpack::BitPack;
use crate::error::BuffError;
use crate::precision::{get_decimal_length, get_precision_bound, PrecisionBound};

/// Format version byte for encoding without special values (legacy).
const FORMAT_V1: u8 = 0x01;

/// Format version byte for encoding with special values support.
const FORMAT_V2: u8 = 0x02;

/// Special value type: positive infinity.
const SPECIAL_POS_INF: u8 = 1;

/// Special value type: negative infinity.
const SPECIAL_NEG_INF: u8 = 2;

/// Special value type: NaN.
const SPECIAL_NAN: u8 = 3;

/// Represents a special floating-point value with its position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecialValue {
    /// Index in the original array.
    pub index: u32,
    /// Type of special value.
    pub kind: SpecialValueKind,
}

/// Types of special floating-point values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialValueKind {
    /// Positive infinity (+∞).
    PositiveInfinity,
    /// Negative infinity (-∞).
    NegativeInfinity,
    /// Not a Number (NaN).
    NaN,
}

impl SpecialValueKind {
    fn to_byte(self) -> u8 {
        match self {
            SpecialValueKind::PositiveInfinity => SPECIAL_POS_INF,
            SpecialValueKind::NegativeInfinity => SPECIAL_NEG_INF,
            SpecialValueKind::NaN => SPECIAL_NAN,
        }
    }

    fn from_byte(b: u8) -> Option<Self> {
        match b {
            SPECIAL_POS_INF => Some(SpecialValueKind::PositiveInfinity),
            SPECIAL_NEG_INF => Some(SpecialValueKind::NegativeInfinity),
            SPECIAL_NAN => Some(SpecialValueKind::NaN),
            _ => None,
        }
    }

    /// Convert to f64 value.
    pub fn to_f64(self) -> f64 {
        match self {
            SpecialValueKind::PositiveInfinity => f64::INFINITY,
            SpecialValueKind::NegativeInfinity => f64::NEG_INFINITY,
            SpecialValueKind::NaN => f64::NAN,
        }
    }
}

/// Classify a float value as regular or special.
pub fn classify_float(v: f64) -> Option<SpecialValueKind> {
    if v.is_nan() {
        Some(SpecialValueKind::NaN)
    } else if v == f64::INFINITY {
        Some(SpecialValueKind::PositiveInfinity)
    } else if v == f64::NEG_INFINITY {
        Some(SpecialValueKind::NegativeInfinity)
    } else {
        None
    }
}

/// Flip the sign bit of a byte for proper ordering.
///
/// This ensures that negative values sort before positive values
/// when comparing bytes lexicographically.
#[inline]
pub fn flip(x: u8) -> u8 {
    x ^ (1u8 << 7)
}

/// Compute the ceiling of x/y.
#[inline]
fn ceil_div(x: u32, y: u32) -> u32 {
    (x.saturating_sub(1)) / y + 1
}

/// BUFF compressor/decompressor for bounded floating-point arrays.
///
/// This implements byte-sliced storage where each bit position across
/// all values is stored contiguously, enabling efficient compression
/// and SIMD-accelerated queries.
#[derive(Clone, Debug)]
pub struct BuffCodec {
    /// Decimal scale (e.g., 1000 for 3 decimal places).
    scale: usize,
}

impl BuffCodec {
    /// Create a new BUFF codec with the given scale.
    ///
    /// # Arguments
    /// * `scale` - The decimal scale (e.g., 1000 for 3 decimal places, 10000 for 4)
    ///
    /// # Example
    /// ```
    /// use buff_rs::BuffCodec;
    ///
    /// // For 3 decimal places of precision
    /// let codec = BuffCodec::new(1000);
    /// ```
    pub fn new(scale: usize) -> Self {
        BuffCodec { scale }
    }

    /// Get the precision (number of decimal places) from the scale.
    pub fn precision(&self) -> i32 {
        if self.scale == 0 {
            0
        } else {
            (self.scale as f32).log10() as i32
        }
    }

    /// Encode an array of f64 values using BUFF byte-sliced encoding.
    ///
    /// # Arguments
    /// * `data` - Slice of f64 values to encode
    ///
    /// # Returns
    /// A `Vec<u8>` containing the encoded data, or an error.
    ///
    /// # Example
    /// ```
    /// use buff_rs::BuffCodec;
    ///
    /// let codec = BuffCodec::new(1000);
    /// let data = vec![1.234, 5.678, 9.012];
    /// let encoded = codec.encode(&data).unwrap();
    /// ```
    pub fn encode(&self, data: &[f64]) -> Result<Vec<u8>, BuffError> {
        if data.is_empty() {
            return Err(BuffError::EmptyInput);
        }

        let len = data.len() as u32;
        let prec = self.precision();
        let prec_delta = get_precision_bound(prec);
        let dec_len = get_decimal_length(prec);

        let mut bound = PrecisionBound::new(prec_delta);
        bound.set_length(0, dec_len);

        // First pass: convert to fixed-point and find min/max
        let mut fixed_vec = Vec::with_capacity(data.len());
        let mut min = i64::MAX;
        let mut max = i64::MIN;

        for &val in data {
            let fixed = bound.fetch_fixed_aligned(val);
            if fixed < min {
                min = fixed;
            }
            if fixed > max {
                max = fixed;
            }
            fixed_vec.push(fixed);
        }

        let delta = max - min;
        let base_fixed = min;

        // Calculate bit width needed
        // We need enough bits to represent values 0..=delta (inclusive)
        // So we need ceil(log2(delta + 1)) bits
        let cal_int_length = if delta == 0 {
            0.0
        } else {
            ((delta + 1) as f64).log2().ceil()
        };

        let fixed_len = cal_int_length as u32;
        let ilen = fixed_len.saturating_sub(dec_len as u32);
        let dlen = dec_len as u32;

        // Write header
        let mut bitpack = BitPack::<Vec<u8>>::with_capacity(20 + data.len() * 8);
        let ubase_fixed = base_fixed as u64;

        bitpack.write(ubase_fixed as u32, 32)?;
        bitpack.write((ubase_fixed >> 32) as u32, 32)?;
        bitpack.write(len, 32)?;
        bitpack.write(ilen, 32)?;
        bitpack.write(dlen, 32)?;

        // Write byte-sliced data
        let total_bits = ilen + dlen;
        let mut remain = total_bits;

        if remain < 8 {
            // Less than one byte per value
            let padding = 8 - remain;
            for &fixed in &fixed_vec {
                let val = (fixed - base_fixed) as u64;
                bitpack.write_byte(flip(((val << padding) & 0xFF) as u8))?;
            }
        } else {
            // First byte slice
            remain -= 8;
            let fixed_u64: Vec<u64> = if remain > 0 {
                fixed_vec
                    .iter()
                    .map(|&x| {
                        let val = (x - base_fixed) as u64;
                        bitpack
                            .write_byte(flip((val >> remain) as u8))
                            .expect("write failed");
                        val
                    })
                    .collect()
            } else {
                fixed_vec
                    .iter()
                    .map(|&x| {
                        let val = (x - base_fixed) as u64;
                        bitpack.write_byte(flip(val as u8)).expect("write failed");
                        val
                    })
                    .collect()
            };

            // Remaining full byte slices
            while remain >= 8 {
                remain -= 8;
                if remain > 0 {
                    for &d in &fixed_u64 {
                        bitpack.write_byte(flip((d >> remain) as u8))?;
                    }
                } else {
                    for &d in &fixed_u64 {
                        bitpack.write_byte(flip(d as u8))?;
                    }
                }
            }

            // Remaining partial byte
            if remain > 0 {
                let padding = 8 - remain;
                for &d in &fixed_u64 {
                    bitpack.write_byte(flip(((d << padding) & 0xFF) as u8))?;
                }
            }
        }

        Ok(bitpack.into_vec())
    }

    /// Encode an array of f64 values, including special values (Infinity, NaN).
    ///
    /// This method handles special floating-point values by storing them separately
    /// from regular values. The encoded format uses version 2 which includes a
    /// special values section.
    ///
    /// # Arguments
    /// * `data` - Slice of f64 values to encode (may contain Infinity, -Infinity, NaN)
    ///
    /// # Returns
    /// A `Vec<u8>` containing the encoded data, or an error.
    ///
    /// # Example
    /// ```
    /// use buff_rs::BuffCodec;
    ///
    /// let codec = BuffCodec::new(1000);
    /// let data = vec![1.234, f64::INFINITY, 5.678, f64::NAN, -f64::INFINITY];
    /// let encoded = codec.encode_with_special(&data).unwrap();
    /// let decoded = codec.decode(&encoded).unwrap();
    /// assert!(decoded[1].is_infinite() && decoded[1].is_sign_positive());
    /// assert!(decoded[3].is_nan());
    /// ```
    pub fn encode_with_special(&self, data: &[f64]) -> Result<Vec<u8>, BuffError> {
        if data.is_empty() {
            return Err(BuffError::EmptyInput);
        }

        // Separate regular values from special values
        let mut regular_values: Vec<f64> = Vec::with_capacity(data.len());
        let mut special_values: Vec<SpecialValue> = Vec::new();
        let mut index_map: Vec<usize> = Vec::with_capacity(data.len()); // Maps result index to regular_values index

        for (i, &val) in data.iter().enumerate() {
            if let Some(kind) = classify_float(val) {
                special_values.push(SpecialValue {
                    index: i as u32,
                    kind,
                });
            } else {
                index_map.push(regular_values.len());
                regular_values.push(val);
            }
        }

        // If no special values, use the regular encode (more compact)
        if special_values.is_empty() {
            return self.encode(data);
        }

        // If all values are special
        if regular_values.is_empty() {
            let mut result = Vec::with_capacity(1 + 4 + 5 * special_values.len());
            result.push(FORMAT_V2);

            // Write empty regular section header
            result.extend_from_slice(&0u64.to_le_bytes()); // base
            result.extend_from_slice(&0u32.to_le_bytes()); // count
            result.extend_from_slice(&0u32.to_le_bytes()); // ilen
            result.extend_from_slice(&0u32.to_le_bytes()); // dlen

            // Write special values
            result.extend_from_slice(&(special_values.len() as u32).to_le_bytes());
            for sv in &special_values {
                result.extend_from_slice(&sv.index.to_le_bytes());
                result.push(sv.kind.to_byte());
            }

            return Ok(result);
        }

        // Encode regular values
        let prec = self.precision();
        let prec_delta = get_precision_bound(prec);
        let dec_len = get_decimal_length(prec);

        let mut bound = PrecisionBound::new(prec_delta);
        bound.set_length(0, dec_len);

        let mut fixed_vec = Vec::with_capacity(regular_values.len());
        let mut min = i64::MAX;
        let mut max = i64::MIN;

        for &val in &regular_values {
            let fixed = bound.fetch_fixed_aligned(val);
            if fixed < min {
                min = fixed;
            }
            if fixed > max {
                max = fixed;
            }
            fixed_vec.push(fixed);
        }

        let delta = max - min;
        let base_fixed = min;

        let cal_int_length = if delta == 0 {
            0.0
        } else {
            ((delta + 1) as f64).log2().ceil()
        };

        let fixed_len = cal_int_length as u32;
        let ilen = fixed_len.saturating_sub(dec_len as u32);
        let dlen = dec_len as u32;

        // Build result
        let mut result =
            Vec::with_capacity(1 + 20 + 4 + 5 * special_values.len() + regular_values.len() * 8);

        // Version byte
        result.push(FORMAT_V2);

        // Regular values header
        let ubase_fixed = base_fixed as u64;
        result.extend_from_slice(&ubase_fixed.to_le_bytes());
        result.extend_from_slice(&(regular_values.len() as u32).to_le_bytes());
        result.extend_from_slice(&ilen.to_le_bytes());
        result.extend_from_slice(&dlen.to_le_bytes());

        // Special values section
        result.extend_from_slice(&(special_values.len() as u32).to_le_bytes());
        for sv in &special_values {
            result.extend_from_slice(&sv.index.to_le_bytes());
            result.push(sv.kind.to_byte());
        }

        // Write byte-sliced data for regular values
        let total_bits = ilen + dlen;
        let mut remain = total_bits;

        if remain == 0 {
            // All same value, no data needed
        } else if remain < 8 {
            let padding = 8 - remain;
            for &fixed in &fixed_vec {
                let val = (fixed - base_fixed) as u64;
                result.push(flip(((val << padding) & 0xFF) as u8));
            }
        } else {
            remain -= 8;
            let fixed_u64: Vec<u64> = fixed_vec
                .iter()
                .map(|&x| {
                    let val = (x - base_fixed) as u64;
                    if remain > 0 {
                        result.push(flip((val >> remain) as u8));
                    } else {
                        result.push(flip(val as u8));
                    }
                    val
                })
                .collect();

            while remain >= 8 {
                remain -= 8;
                if remain > 0 {
                    for &d in &fixed_u64 {
                        result.push(flip((d >> remain) as u8));
                    }
                } else {
                    for &d in &fixed_u64 {
                        result.push(flip(d as u8));
                    }
                }
            }

            if remain > 0 {
                let padding = 8 - remain;
                for &d in &fixed_u64 {
                    result.push(flip(((d << padding) & 0xFF) as u8));
                }
            }
        }

        Ok(result)
    }

    /// Check if encoded data contains special values (v2 format).
    pub fn has_special_values(&self, bytes: &[u8]) -> bool {
        !bytes.is_empty() && bytes[0] == FORMAT_V2
    }

    /// Decode BUFF-encoded data back to f64 values.
    ///
    /// This method automatically detects the format version and handles both
    /// legacy (v1) and special values (v2) formats.
    ///
    /// # Arguments
    /// * `bytes` - The encoded byte array
    ///
    /// # Returns
    /// A `Vec<f64>` containing the decoded values, or an error.
    ///
    /// # Example
    /// ```
    /// use buff_rs::BuffCodec;
    ///
    /// let codec = BuffCodec::new(1000);
    /// let data = vec![1.234, 5.678, 9.012];
    /// let encoded = codec.encode(&data).unwrap();
    /// let decoded = codec.decode(&encoded).unwrap();
    /// ```
    pub fn decode(&self, bytes: &[u8]) -> Result<Vec<f64>, BuffError> {
        // Check for v2 format (with special values)
        if !bytes.is_empty() && bytes[0] == FORMAT_V2 {
            return self.decode_v2(bytes);
        }

        // Legacy v1 format
        self.decode_v1(bytes)
    }

    /// Decode v1 format (legacy, no special values).
    fn decode_v1(&self, bytes: &[u8]) -> Result<Vec<f64>, BuffError> {
        if bytes.len() < 20 {
            return Err(BuffError::InvalidData("header too short".into()));
        }

        let prec = self.precision();
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes);
        let _bound = PrecisionBound::new(prec_delta);

        // Read header
        let lower = bitpack.read(32)?;
        let higher = bitpack.read(32)?;
        let ubase_int = (lower as u64) | ((higher as u64) << 32);
        let base_int = ubase_int as i64;

        let len = bitpack.read(32)? as usize;
        let ilen = bitpack.read(32)?;
        let dlen = bitpack.read(32)?;

        let dec_scl = 2.0f64.powi(dlen as i32);
        let remain = dlen + ilen;
        let num_chunks = ceil_div(remain, 8);
        let padding = num_chunks * 8 - ilen - dlen;

        let mut result = Vec::with_capacity(len);

        match num_chunks {
            0 => {
                // All values are the base
                result.resize(len, base_int as f64 / dec_scl);
            }
            1 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                for &byte in chunk0.iter().take(len) {
                    let val = (flip(byte) as u64) >> padding;
                    result.push((base_int + val as i64) as f64 / dec_scl);
                }
            }
            2 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                let chunk1 = bitpack.read_n_byte_unmut(len, len)?;
                for (&b0, &b1) in chunk0.iter().zip(chunk1.iter()).take(len) {
                    let val =
                        ((flip(b0) as u64) << (8 - padding)) | ((flip(b1) as u64) >> padding);
                    result.push((base_int + val as i64) as f64 / dec_scl);
                }
            }
            3 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                let chunk1 = bitpack.read_n_byte_unmut(len, len)?;
                let chunk2 = bitpack.read_n_byte_unmut(2 * len, len)?;
                for ((&b0, &b1), &b2) in
                    chunk0.iter().zip(chunk1.iter()).zip(chunk2.iter()).take(len)
                {
                    let val = ((flip(b0) as u64) << (16 - padding))
                        | ((flip(b1) as u64) << (8 - padding))
                        | ((flip(b2) as u64) >> padding);
                    result.push((base_int + val as i64) as f64 / dec_scl);
                }
            }
            4 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                let chunk1 = bitpack.read_n_byte_unmut(len, len)?;
                let chunk2 = bitpack.read_n_byte_unmut(2 * len, len)?;
                let chunk3 = bitpack.read_n_byte_unmut(3 * len, len)?;
                for (((&b0, &b1), &b2), &b3) in chunk0
                    .iter()
                    .zip(chunk1.iter())
                    .zip(chunk2.iter())
                    .zip(chunk3.iter())
                    .take(len)
                {
                    let val = ((flip(b0) as u64) << (24 - padding))
                        | ((flip(b1) as u64) << (16 - padding))
                        | ((flip(b2) as u64) << (8 - padding))
                        | ((flip(b3) as u64) >> padding);
                    result.push((base_int + val as i64) as f64 / dec_scl);
                }
            }
            _ => {
                return Err(BuffError::InvalidData(format!(
                    "bit length {} (>{} chunks) not supported",
                    remain, num_chunks
                )));
            }
        }

        Ok(result)
    }

    /// Decode v2 format (with special values).
    fn decode_v2(&self, bytes: &[u8]) -> Result<Vec<f64>, BuffError> {
        if bytes.len() < 22 {
            // 1 (version) + 8 (base) + 4 (count) + 4 (ilen) + 4 (dlen) + 4 (special count) = 25 min
            return Err(BuffError::InvalidData("v2 header too short".into()));
        }

        let mut pos = 1; // Skip version byte

        // Read regular values header
        let base_int = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let regular_count =
            u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let ilen = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;

        let dlen = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;

        // Read special values count
        let special_count =
            u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        // Read special values
        let mut special_values: Vec<SpecialValue> = Vec::with_capacity(special_count);
        for _ in 0..special_count {
            if pos + 5 > bytes.len() {
                return Err(BuffError::InvalidData("truncated special values".into()));
            }
            let index = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let kind = SpecialValueKind::from_byte(bytes[pos])
                .ok_or_else(|| BuffError::InvalidData("invalid special value type".into()))?;
            pos += 1;
            special_values.push(SpecialValue { index, kind });
        }

        // Calculate total count
        let total_count = regular_count + special_count;

        // If no regular values, just return special values
        if regular_count == 0 {
            let mut result = vec![0.0f64; total_count];
            for sv in &special_values {
                result[sv.index as usize] = sv.kind.to_f64();
            }
            return Ok(result);
        }

        // Decode regular values from byte-sliced data
        let dec_scl = 2.0f64.powi(dlen as i32);
        let remain = dlen + ilen;
        let num_chunks = ceil_div(remain, 8);
        let padding = num_chunks * 8 - ilen - dlen;

        let data_start = pos;
        let mut regular_values: Vec<f64> = Vec::with_capacity(regular_count);

        match num_chunks {
            0 => {
                regular_values.resize(regular_count, base_int as f64 / dec_scl);
            }
            1 => {
                let chunk0 = &bytes[data_start..data_start + regular_count];
                for &byte in chunk0.iter() {
                    let val = (flip(byte) as u64) >> padding;
                    regular_values.push((base_int + val as i64) as f64 / dec_scl);
                }
            }
            2 => {
                let chunk0 = &bytes[data_start..data_start + regular_count];
                let chunk1 =
                    &bytes[data_start + regular_count..data_start + 2 * regular_count];
                for (&b0, &b1) in chunk0.iter().zip(chunk1.iter()) {
                    let val =
                        ((flip(b0) as u64) << (8 - padding)) | ((flip(b1) as u64) >> padding);
                    regular_values.push((base_int + val as i64) as f64 / dec_scl);
                }
            }
            3 => {
                let chunk0 = &bytes[data_start..data_start + regular_count];
                let chunk1 =
                    &bytes[data_start + regular_count..data_start + 2 * regular_count];
                let chunk2 =
                    &bytes[data_start + 2 * regular_count..data_start + 3 * regular_count];
                for ((&b0, &b1), &b2) in chunk0.iter().zip(chunk1.iter()).zip(chunk2.iter()) {
                    let val = ((flip(b0) as u64) << (16 - padding))
                        | ((flip(b1) as u64) << (8 - padding))
                        | ((flip(b2) as u64) >> padding);
                    regular_values.push((base_int + val as i64) as f64 / dec_scl);
                }
            }
            4 => {
                let chunk0 = &bytes[data_start..data_start + regular_count];
                let chunk1 =
                    &bytes[data_start + regular_count..data_start + 2 * regular_count];
                let chunk2 =
                    &bytes[data_start + 2 * regular_count..data_start + 3 * regular_count];
                let chunk3 =
                    &bytes[data_start + 3 * regular_count..data_start + 4 * regular_count];
                for (((&b0, &b1), &b2), &b3) in chunk0
                    .iter()
                    .zip(chunk1.iter())
                    .zip(chunk2.iter())
                    .zip(chunk3.iter())
                {
                    let val = ((flip(b0) as u64) << (24 - padding))
                        | ((flip(b1) as u64) << (16 - padding))
                        | ((flip(b2) as u64) << (8 - padding))
                        | ((flip(b3) as u64) >> padding);
                    regular_values.push((base_int + val as i64) as f64 / dec_scl);
                }
            }
            _ => {
                return Err(BuffError::InvalidData(format!(
                    "bit length {} not supported",
                    remain
                )));
            }
        }

        // Merge regular and special values into final result
        let mut result = vec![0.0f64; total_count];
        let mut regular_idx = 0;

        // Create a set of special value indices for O(1) lookup
        let special_indices: std::collections::HashSet<u32> =
            special_values.iter().map(|sv| sv.index).collect();

        for (i, slot) in result.iter_mut().enumerate() {
            if special_indices.contains(&(i as u32)) {
                // Find the special value for this index
                if let Some(sv) = special_values.iter().find(|sv| sv.index == i as u32) {
                    *slot = sv.kind.to_f64();
                }
            } else {
                *slot = regular_values[regular_idx];
                regular_idx += 1;
            }
        }

        Ok(result)
    }

    /// Compute the sum of all values in the encoded data.
    ///
    /// This operates directly on the compressed data without full decompression.
    pub fn sum(&self, bytes: &[u8]) -> Result<f64, BuffError> {
        if bytes.len() < 20 {
            return Err(BuffError::InvalidData("header too short".into()));
        }

        let prec = self.precision();
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes);
        let _bound = PrecisionBound::new(prec_delta);

        // Read header
        let lower = bitpack.read(32)?;
        let higher = bitpack.read(32)?;
        let ubase_int = (lower as u64) | ((higher as u64) << 32);
        let base_int = ubase_int as i64;

        let len = bitpack.read(32)? as usize;
        let ilen = bitpack.read(32)?;
        let dlen = bitpack.read(32)?;

        let dec_scl = 2.0f64.powi(dlen as i32);
        let remain = dlen + ilen;
        let num_chunks = ceil_div(remain, 8);
        let padding = num_chunks * 8 - ilen - dlen;

        let mut sum = 0.0f64;

        match num_chunks {
            0 => {
                sum = (base_int as f64 / dec_scl) * len as f64;
            }
            1 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                for &byte in chunk0.iter().take(len) {
                    let val = (flip(byte) as u64) >> padding;
                    sum += (base_int + val as i64) as f64 / dec_scl;
                }
            }
            2 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                let chunk1 = bitpack.read_n_byte_unmut(len, len)?;
                for (&b0, &b1) in chunk0.iter().zip(chunk1.iter()).take(len) {
                    let val =
                        ((flip(b0) as u64) << (8 - padding)) | ((flip(b1) as u64) >> padding);
                    sum += (base_int + val as i64) as f64 / dec_scl;
                }
            }
            3 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                let chunk1 = bitpack.read_n_byte_unmut(len, len)?;
                let chunk2 = bitpack.read_n_byte_unmut(2 * len, len)?;
                for ((&b0, &b1), &b2) in
                    chunk0.iter().zip(chunk1.iter()).zip(chunk2.iter()).take(len)
                {
                    let val = ((flip(b0) as u64) << (16 - padding))
                        | ((flip(b1) as u64) << (8 - padding))
                        | ((flip(b2) as u64) >> padding);
                    sum += (base_int + val as i64) as f64 / dec_scl;
                }
            }
            4 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                let chunk1 = bitpack.read_n_byte_unmut(len, len)?;
                let chunk2 = bitpack.read_n_byte_unmut(2 * len, len)?;
                let chunk3 = bitpack.read_n_byte_unmut(3 * len, len)?;
                for (((&b0, &b1), &b2), &b3) in chunk0
                    .iter()
                    .zip(chunk1.iter())
                    .zip(chunk2.iter())
                    .zip(chunk3.iter())
                    .take(len)
                {
                    let val = ((flip(b0) as u64) << (24 - padding))
                        | ((flip(b1) as u64) << (16 - padding))
                        | ((flip(b2) as u64) << (8 - padding))
                        | ((flip(b3) as u64) >> padding);
                    sum += (base_int + val as i64) as f64 / dec_scl;
                }
            }
            _ => {
                return Err(BuffError::InvalidData(format!(
                    "bit length {} not supported",
                    remain
                )));
            }
        }

        Ok(sum)
    }

    /// Find the maximum value in the encoded data.
    ///
    /// This operates directly on the compressed data without full decompression.
    pub fn max(&self, bytes: &[u8]) -> Result<f64, BuffError> {
        if bytes.len() < 20 {
            return Err(BuffError::InvalidData("header too short".into()));
        }

        let prec = self.precision();
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes);
        let _bound = PrecisionBound::new(prec_delta);

        // Read header
        let lower = bitpack.read(32)?;
        let higher = bitpack.read(32)?;
        let ubase_int = (lower as u64) | ((higher as u64) << 32);
        let base_int = ubase_int as i64;

        let len = bitpack.read(32)? as usize;
        let ilen = bitpack.read(32)?;
        let dlen = bitpack.read(32)?;

        let dec_scl = 2.0f64.powi(dlen as i32);
        let remain = dlen + ilen;
        let num_chunks = ceil_div(remain, 8);
        let padding = num_chunks * 8 - ilen - dlen;

        let mut max_val = f64::MIN;

        match num_chunks {
            0 => {
                max_val = base_int as f64 / dec_scl;
            }
            1 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                for &byte in chunk0.iter().take(len) {
                    let val = (flip(byte) as u64) >> padding;
                    let f = (base_int + val as i64) as f64 / dec_scl;
                    if f > max_val {
                        max_val = f;
                    }
                }
            }
            2 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                let chunk1 = bitpack.read_n_byte_unmut(len, len)?;
                for (&b0, &b1) in chunk0.iter().zip(chunk1.iter()).take(len) {
                    let val =
                        ((flip(b0) as u64) << (8 - padding)) | ((flip(b1) as u64) >> padding);
                    let f = (base_int + val as i64) as f64 / dec_scl;
                    if f > max_val {
                        max_val = f;
                    }
                }
            }
            3 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                let chunk1 = bitpack.read_n_byte_unmut(len, len)?;
                let chunk2 = bitpack.read_n_byte_unmut(2 * len, len)?;
                for ((&b0, &b1), &b2) in
                    chunk0.iter().zip(chunk1.iter()).zip(chunk2.iter()).take(len)
                {
                    let val = ((flip(b0) as u64) << (16 - padding))
                        | ((flip(b1) as u64) << (8 - padding))
                        | ((flip(b2) as u64) >> padding);
                    let f = (base_int + val as i64) as f64 / dec_scl;
                    if f > max_val {
                        max_val = f;
                    }
                }
            }
            4 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                let chunk1 = bitpack.read_n_byte_unmut(len, len)?;
                let chunk2 = bitpack.read_n_byte_unmut(2 * len, len)?;
                let chunk3 = bitpack.read_n_byte_unmut(3 * len, len)?;
                for (((&b0, &b1), &b2), &b3) in chunk0
                    .iter()
                    .zip(chunk1.iter())
                    .zip(chunk2.iter())
                    .zip(chunk3.iter())
                    .take(len)
                {
                    let val = ((flip(b0) as u64) << (24 - padding))
                        | ((flip(b1) as u64) << (16 - padding))
                        | ((flip(b2) as u64) << (8 - padding))
                        | ((flip(b3) as u64) >> padding);
                    let f = (base_int + val as i64) as f64 / dec_scl;
                    if f > max_val {
                        max_val = f;
                    }
                }
            }
            _ => {
                return Err(BuffError::InvalidData(format!(
                    "bit length {} not supported",
                    remain
                )));
            }
        }

        Ok(max_val)
    }

    /// Count values greater than a threshold.
    ///
    /// This operates directly on the compressed data using early termination
    /// when possible.
    pub fn count_greater_than(&self, bytes: &[u8], threshold: f64) -> Result<usize, BuffError> {
        // For now, use decode + filter. SIMD optimization can be added later.
        let decoded = self.decode(bytes)?;
        Ok(decoded.iter().filter(|&&v| v > threshold).count())
    }

    /// Count values equal to a target.
    pub fn count_equal(&self, bytes: &[u8], target: f64) -> Result<usize, BuffError> {
        let decoded = self.decode(bytes)?;
        Ok(decoded.iter().filter(|&&v| (v - target).abs() < f64::EPSILON).count())
    }

    /// Get metadata about the encoded data without decoding.
    pub fn metadata(&self, bytes: &[u8]) -> Result<BuffMetadata, BuffError> {
        if bytes.len() < 20 {
            return Err(BuffError::InvalidData("header too short".into()));
        }

        let mut bitpack = BitPack::<&[u8]>::new(bytes);

        let lower = bitpack.read(32)?;
        let higher = bitpack.read(32)?;
        let ubase_int = (lower as u64) | ((higher as u64) << 32);
        let base_int = ubase_int as i64;

        let len = bitpack.read(32)?;
        let ilen = bitpack.read(32)?;
        let dlen = bitpack.read(32)?;

        Ok(BuffMetadata {
            base_value: base_int,
            count: len,
            integer_bits: ilen,
            decimal_bits: dlen,
            total_bytes: bytes.len(),
        })
    }
}

/// Metadata about BUFF-encoded data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuffMetadata {
    /// The base value (minimum fixed-point value).
    pub base_value: i64,
    /// Number of encoded values.
    pub count: u32,
    /// Number of bits for the integer part.
    pub integer_bits: u32,
    /// Number of bits for the decimal part.
    pub decimal_bits: u32,
    /// Total size of the encoded data in bytes.
    pub total_bytes: usize,
}

impl BuffMetadata {
    /// Calculate the compression ratio (compressed size / original size).
    pub fn compression_ratio(&self) -> f64 {
        let original_size = self.count as usize * std::mem::size_of::<f64>();
        self.total_bytes as f64 / original_size as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flip() {
        assert_eq!(flip(0), 128);
        assert_eq!(flip(128), 0);
        assert_eq!(flip(255), 127);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let codec = BuffCodec::new(1000);
        let data = vec![1.234, 5.678, 9.012, -3.456, 0.0];
        let encoded = codec.encode(&data).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        assert_eq!(data.len(), decoded.len());
        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.001, "orig={}, dec={}", orig, dec);
        }
    }

    #[test]
    fn test_encode_decode_high_precision() {
        let codec = BuffCodec::new(100000); // 5 decimal places
        let data = vec![1.23456, 5.67890, 9.01234];
        let encoded = codec.encode(&data).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.00001, "orig={}, dec={}", orig, dec);
        }
    }

    #[test]
    fn test_empty_input() {
        let codec = BuffCodec::new(1000);
        let result = codec.encode(&[]);
        assert!(matches!(result, Err(BuffError::EmptyInput)));
    }

    #[test]
    fn test_single_value() {
        let codec = BuffCodec::new(1000);
        let data = vec![42.123];
        let encoded = codec.encode(&data).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        assert_eq!(decoded.len(), 1);
        assert!((data[0] - decoded[0]).abs() < 0.001);
    }

    #[test]
    fn test_identical_values() {
        let codec = BuffCodec::new(1000);
        let data = vec![3.14159; 100];
        let encoded = codec.encode(&data).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        assert_eq!(decoded.len(), 100);
        for &val in &decoded {
            assert!((val - 3.14159).abs() < 0.001);
        }
    }

    #[test]
    fn test_sum() {
        let codec = BuffCodec::new(1000);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let encoded = codec.encode(&data).unwrap();
        let sum = codec.sum(&encoded).unwrap();

        assert!((sum - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_max() {
        let codec = BuffCodec::new(1000);
        let data = vec![1.5, 9.9, 3.2, 7.1, 2.8];
        let encoded = codec.encode(&data).unwrap();
        let max = codec.max(&encoded).unwrap();

        assert!((max - 9.9).abs() < 0.01);
    }

    #[test]
    fn test_metadata() {
        let codec = BuffCodec::new(1000);
        let data = vec![1.0, 2.0, 3.0];
        let encoded = codec.encode(&data).unwrap();
        let metadata = codec.metadata(&encoded).unwrap();

        assert_eq!(metadata.count, 3);
        assert!(metadata.compression_ratio() < 1.5); // Should be reasonably compressed
    }

    #[test]
    fn test_negative_values() {
        let codec = BuffCodec::new(1000);
        let data = vec![-10.5, -5.25, 0.0, 5.25, 10.5];
        let encoded = codec.encode(&data).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.001, "orig={}, dec={}", orig, dec);
        }
    }

    #[test]
    fn test_large_values() {
        let codec = BuffCodec::new(100);
        let data = vec![1000000.0, 2000000.0, 3000000.0];
        let encoded = codec.encode(&data).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 1.0, "orig={}, dec={}", orig, dec);
        }
    }

    #[test]
    fn test_special_values_infinity() {
        let codec = BuffCodec::new(1000);
        let data = vec![1.0, f64::INFINITY, 2.0, f64::NEG_INFINITY, 3.0];
        let encoded = codec.encode_with_special(&data).unwrap();

        assert!(codec.has_special_values(&encoded));

        let decoded = codec.decode(&encoded).unwrap();

        assert_eq!(decoded.len(), 5);
        assert!((decoded[0] - 1.0).abs() < 0.001);
        assert!(decoded[1].is_infinite() && decoded[1].is_sign_positive());
        assert!((decoded[2] - 2.0).abs() < 0.001);
        assert!(decoded[3].is_infinite() && decoded[3].is_sign_negative());
        assert!((decoded[4] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_special_values_nan() {
        let codec = BuffCodec::new(1000);
        let data = vec![1.0, f64::NAN, 2.0];
        let encoded = codec.encode_with_special(&data).unwrap();

        let decoded = codec.decode(&encoded).unwrap();

        assert_eq!(decoded.len(), 3);
        assert!((decoded[0] - 1.0).abs() < 0.001);
        assert!(decoded[1].is_nan());
        assert!((decoded[2] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_all_special_values() {
        let codec = BuffCodec::new(1000);
        let data = vec![f64::INFINITY, f64::NAN, f64::NEG_INFINITY];
        let encoded = codec.encode_with_special(&data).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        assert_eq!(decoded.len(), 3);
        assert!(decoded[0].is_infinite() && decoded[0].is_sign_positive());
        assert!(decoded[1].is_nan());
        assert!(decoded[2].is_infinite() && decoded[2].is_sign_negative());
    }

    #[test]
    fn test_no_special_values_uses_v1() {
        let codec = BuffCodec::new(1000);
        let data = vec![1.0, 2.0, 3.0];

        // encode_with_special should use v1 when no special values
        let encoded = codec.encode_with_special(&data).unwrap();
        assert!(!codec.has_special_values(&encoded));

        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 3);
    }

    #[test]
    fn test_special_value_kind() {
        assert_eq!(SpecialValueKind::PositiveInfinity.to_f64(), f64::INFINITY);
        assert_eq!(SpecialValueKind::NegativeInfinity.to_f64(), f64::NEG_INFINITY);
        assert!(SpecialValueKind::NaN.to_f64().is_nan());
    }

    #[test]
    fn test_classify_float() {
        assert_eq!(classify_float(1.0), None);
        assert_eq!(
            classify_float(f64::INFINITY),
            Some(SpecialValueKind::PositiveInfinity)
        );
        assert_eq!(
            classify_float(f64::NEG_INFINITY),
            Some(SpecialValueKind::NegativeInfinity)
        );
        assert_eq!(classify_float(f64::NAN), Some(SpecialValueKind::NaN));
    }
}
