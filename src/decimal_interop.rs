//! Optional interop with the `decimal-bytes` crate.
//!
//! This module provides conversion traits between `buff_rs` and `decimal_bytes::Decimal`.
//!
//! **Important**: Converting between BUFF and Decimal involves precision loss because:
//! - BUFF uses bounded floating-point representation
//! - Decimal uses exact arbitrary-precision representation
//!
//! ## Usage
//!
//! Enable the `decimal` feature in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! buff-rs = { version = "0.1", features = ["decimal"] }
//! ```
//!
//! Then you can convert between types:
//!
//! ```ignore
//! use buff_rs::BuffCodec;
//! use decimal_bytes::Decimal;
//!
//! let codec = BuffCodec::new(1000);
//!
//! // Convert Decimal array to BUFF-encoded bytes
//! let decimals: Vec<Decimal> = vec![
//!     "1.234".parse().unwrap(),
//!     "5.678".parse().unwrap(),
//! ];
//! let encoded = codec.encode_decimals(&decimals).unwrap();
//!
//! // Decode back to Decimal
//! let decoded: Vec<Decimal> = codec.decode_to_decimals(&encoded).unwrap();
//! ```

use crate::codec::{classify_float, BuffCodec};
use crate::error::BuffError;
use decimal_bytes::Decimal;

impl BuffCodec {
    /// Encode an array of `decimal_bytes::Decimal` values.
    ///
    /// **Warning**: This involves precision loss. Decimal values are converted
    /// to f64 for BUFF encoding. Use this only when the precision loss is acceptable
    /// for your use case (e.g., analytics on time-series data).
    ///
    /// Special values (Infinity, -Infinity, NaN) in the Decimal are handled
    /// and stored separately.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use buff_rs::BuffCodec;
    /// use decimal_bytes::Decimal;
    ///
    /// let codec = BuffCodec::new(1000);
    /// let decimals: Vec<Decimal> = vec![
    ///     "1.234".parse().unwrap(),
    ///     "5.678".parse().unwrap(),
    /// ];
    /// let encoded = codec.encode_decimals(&decimals).unwrap();
    /// ```
    pub fn encode_decimals(&self, data: &[Decimal]) -> Result<Vec<u8>, BuffError> {
        if data.is_empty() {
            return Err(BuffError::EmptyInput);
        }

        // Convert decimals to f64
        let floats: Vec<f64> = data
            .iter()
            .map(|d| {
                if d.is_nan() {
                    f64::NAN
                } else if d.is_infinity() {
                    if d.is_negative() {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    }
                } else {
                    // Parse decimal to f64 (involves precision loss)
                    d.to_string().parse::<f64>().unwrap_or(0.0)
                }
            })
            .collect();

        // Check if any special values exist
        let has_special = floats.iter().any(|v| classify_float(*v).is_some());

        if has_special {
            self.encode_with_special(&floats)
        } else {
            self.encode(&floats)
        }
    }

    /// Decode BUFF-encoded data to `decimal_bytes::Decimal` values.
    ///
    /// **Warning**: The decoded Decimals have the precision determined by the
    /// codec's scale, not the original Decimal precision.
    ///
    /// Special values are converted to `Decimal::nan()`, `Decimal::infinity()`,
    /// or `Decimal::neg_infinity()` as appropriate.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use buff_rs::BuffCodec;
    /// use decimal_bytes::Decimal;
    ///
    /// let codec = BuffCodec::new(1000);
    /// let data = vec![1.234, 5.678];
    /// let encoded = codec.encode(&data).unwrap();
    /// let decimals: Vec<Decimal> = codec.decode_to_decimals(&encoded).unwrap();
    /// ```
    pub fn decode_to_decimals(&self, bytes: &[u8]) -> Result<Vec<Decimal>, BuffError> {
        let floats = self.decode(bytes)?;

        floats
            .into_iter()
            .map(|f| {
                if f.is_nan() {
                    Ok(Decimal::nan())
                } else if f == f64::INFINITY {
                    Ok(Decimal::infinity())
                } else if f == f64::NEG_INFINITY {
                    Ok(Decimal::neg_infinity())
                } else {
                    // Convert f64 to Decimal string representation
                    let precision = self.precision() as u32;
                    let formatted = format!("{:.prec$}", f, prec = precision as usize);
                    formatted.parse::<Decimal>().map_err(|e| {
                        BuffError::InvalidData(format!("failed to parse decimal: {}", e))
                    })
                }
            })
            .collect()
    }

    /// Convert a single f64 to `decimal_bytes::Decimal`.
    ///
    /// This is a convenience method for single-value conversions.
    pub fn f64_to_decimal(&self, value: f64) -> Result<Decimal, BuffError> {
        if value.is_nan() {
            Ok(Decimal::nan())
        } else if value == f64::INFINITY {
            Ok(Decimal::infinity())
        } else if value == f64::NEG_INFINITY {
            Ok(Decimal::neg_infinity())
        } else {
            let precision = self.precision() as usize;
            let formatted = format!("{:.prec$}", value, prec = precision);
            formatted
                .parse::<Decimal>()
                .map_err(|e| BuffError::InvalidData(format!("failed to parse decimal: {}", e)))
        }
    }

    /// Convert a `decimal_bytes::Decimal` to f64.
    ///
    /// This is a convenience method for single-value conversions.
    ///
    /// **Warning**: This involves precision loss for high-precision decimals.
    pub fn decimal_to_f64(&self, decimal: &Decimal) -> f64 {
        if decimal.is_nan() {
            f64::NAN
        } else if decimal.is_infinity() {
            if decimal.is_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        } else {
            decimal.to_string().parse::<f64>().unwrap_or(0.0)
        }
    }
}

/// Extension trait for converting arrays between Decimal and BUFF.
pub trait DecimalArrayExt {
    /// Convert to f64 array for BUFF encoding.
    fn to_f64_array(&self) -> Vec<f64>;
}

impl DecimalArrayExt for [Decimal] {
    fn to_f64_array(&self) -> Vec<f64> {
        self.iter()
            .map(|d| {
                if d.is_nan() {
                    f64::NAN
                } else if d.is_infinity() {
                    if d.is_negative() {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    }
                } else {
                    d.to_string().parse::<f64>().unwrap_or(0.0)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_decimals() {
        let codec = BuffCodec::new(1000);

        let decimals: Vec<Decimal> = vec![
            "1.234".parse().unwrap(),
            "5.678".parse().unwrap(),
            "9.012".parse().unwrap(),
        ];

        let encoded = codec.encode_decimals(&decimals).unwrap();
        let decoded = codec.decode_to_decimals(&encoded).unwrap();

        assert_eq!(decoded.len(), 3);
        // Check approximate equality (precision loss expected)
        for (orig, dec) in decimals.iter().zip(decoded.iter()) {
            let orig_f: f64 = orig.to_string().parse().unwrap();
            let dec_f: f64 = dec.to_string().parse().unwrap();
            assert!((orig_f - dec_f).abs() < 0.001, "orig={}, dec={}", orig, dec);
        }
    }

    #[test]
    fn test_special_values_decimal() {
        let codec = BuffCodec::new(1000);

        let decimals: Vec<Decimal> = vec![
            "1.0".parse().unwrap(),
            Decimal::infinity(),
            "2.0".parse().unwrap(),
            Decimal::nan(),
            Decimal::neg_infinity(),
        ];

        let encoded = codec.encode_decimals(&decimals).unwrap();
        let decoded = codec.decode_to_decimals(&encoded).unwrap();

        assert_eq!(decoded.len(), 5);
        assert!(!decoded[0].is_special());
        assert!(decoded[1].is_infinity() && !decoded[1].is_negative());
        assert!(!decoded[2].is_special());
        assert!(decoded[3].is_nan());
        assert!(decoded[4].is_infinity() && decoded[4].is_negative());
    }

    #[test]
    fn test_f64_to_decimal_conversion() {
        let codec = BuffCodec::new(1000);

        let dec = codec.f64_to_decimal(3.14159).unwrap();
        let f = codec.decimal_to_f64(&dec);

        assert!((f - 3.142).abs() < 0.001); // Rounded to 3 decimal places
    }

    #[test]
    fn test_decimal_array_ext() {
        let decimals: Vec<Decimal> = vec![
            "1.5".parse().unwrap(),
            Decimal::infinity(),
            "2.5".parse().unwrap(),
        ];

        let floats = decimals.to_f64_array();

        assert!((floats[0] - 1.5).abs() < f64::EPSILON);
        assert!(floats[1].is_infinite() && floats[1].is_sign_positive());
        assert!((floats[2] - 2.5).abs() < f64::EPSILON);
    }
}
