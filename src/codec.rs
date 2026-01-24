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
//! ## Encoding Format
//!
//! The encoded format consists of:
//! - Base integer (8 bytes, i64)
//! - Number of values (4 bytes, u32)
//! - Integer bits (4 bytes, u32)
//! - Decimal bits (4 bytes, u32)
//! - Byte slices (column-oriented storage)
//!
//! Each value is converted to a fixed-point representation relative to the base,
//! then stored in byte-sliced format where each byte position across all values
//! is stored contiguously.

use crate::bitpack::BitPack;
use crate::error::BuffError;
use crate::precision::{get_decimal_length, get_precision_bound, PrecisionBound};

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
    /// A Vec<u8> containing the encoded data, or an error.
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

    /// Decode BUFF-encoded data back to f64 values.
    ///
    /// # Arguments
    /// * `bytes` - The encoded byte array
    ///
    /// # Returns
    /// A Vec<f64> containing the decoded values, or an error.
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
                    let val = ((flip(b0) as u64) << (8 - padding)) | ((flip(b1) as u64) >> padding);
                    result.push((base_int + val as i64) as f64 / dec_scl);
                }
            }
            3 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                let chunk1 = bitpack.read_n_byte_unmut(len, len)?;
                let chunk2 = bitpack.read_n_byte_unmut(2 * len, len)?;
                for ((&b0, &b1), &b2) in chunk0
                    .iter()
                    .zip(chunk1.iter())
                    .zip(chunk2.iter())
                    .take(len)
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
                    let val = ((flip(b0) as u64) << (8 - padding)) | ((flip(b1) as u64) >> padding);
                    sum += (base_int + val as i64) as f64 / dec_scl;
                }
            }
            3 => {
                let chunk0 = bitpack.read_n_byte_unmut(0, len)?;
                let chunk1 = bitpack.read_n_byte_unmut(len, len)?;
                let chunk2 = bitpack.read_n_byte_unmut(2 * len, len)?;
                for ((&b0, &b1), &b2) in chunk0
                    .iter()
                    .zip(chunk1.iter())
                    .zip(chunk2.iter())
                    .take(len)
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
                    let val = ((flip(b0) as u64) << (8 - padding)) | ((flip(b1) as u64) >> padding);
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
                for ((&b0, &b1), &b2) in chunk0
                    .iter()
                    .zip(chunk1.iter())
                    .zip(chunk2.iter())
                    .take(len)
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
        Ok(decoded
            .iter()
            .filter(|&&v| (v - target).abs() < f64::EPSILON)
            .count())
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
}
