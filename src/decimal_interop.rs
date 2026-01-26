//! Optional interop with the `decimal-bytes` crate.
//!
//! This module provides conversion traits between `buff_rs` and `decimal_bytes` types.
//!
//! ## Supported Types
//!
//! - **`Decimal`**: Arbitrary precision decimals (variable-length)
//! - **`Decimal64`**: Fixed 8-byte decimals with embedded scale (≤16 digits)
//!
//! **Important**: Converting between BUFF and Decimal types involves precision loss because:
//! - BUFF uses bounded floating-point representation
//! - Decimal types use exact fixed-point representation
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
//! ### With Decimal (arbitrary precision)
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
//!
//! ### With Decimal64 (fixed-size, ≤16 digits)
//!
//! ```ignore
//! use buff_rs::BuffCodec;
//! use decimal_bytes::Decimal64;
//!
//! let codec = BuffCodec::new(1000);
//!
//! // Convert Decimal64 array to BUFF-encoded bytes
//! let decimals: Vec<Decimal64> = vec![
//!     Decimal64::new("1.234", 3).unwrap(),
//!     Decimal64::new("5.678", 3).unwrap(),
//! ];
//! let encoded = codec.encode_decimal64s(&decimals).unwrap();
//!
//! // Decode back to Decimal64 (specify output scale)
//! let decoded: Vec<Decimal64> = codec.decode_to_decimal64s(&encoded, 3).unwrap();
//! ```

use crate::codec::{classify_float, BuffCodec};
use crate::error::BuffError;
use decimal_bytes::{Decimal, Decimal64};

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

    // ==================== Decimal64 Methods ====================

    /// Encode an array of `decimal_bytes::Decimal64` values.
    ///
    /// `Decimal64` values are converted to f64 for BUFF encoding. Since Decimal64
    /// has a maximum of 16 significant digits, precision loss is minimal for most
    /// use cases.
    ///
    /// Special values (Infinity, -Infinity, NaN) are handled and stored separately.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use buff_rs::BuffCodec;
    /// use decimal_bytes::Decimal64;
    ///
    /// let codec = BuffCodec::new(1000);
    /// let decimals: Vec<Decimal64> = vec![
    ///     Decimal64::new("1.234", 3).unwrap(),
    ///     Decimal64::new("5.678", 3).unwrap(),
    /// ];
    /// let encoded = codec.encode_decimal64s(&decimals).unwrap();
    /// ```
    pub fn encode_decimal64s(&self, data: &[Decimal64]) -> Result<Vec<u8>, BuffError> {
        if data.is_empty() {
            return Err(BuffError::EmptyInput);
        }

        // Convert Decimal64 to f64
        let floats: Vec<f64> = data.iter().map(decimal64_to_f64).collect();

        // Check if any special values exist
        let has_special = floats.iter().any(|v| classify_float(*v).is_some());

        if has_special {
            self.encode_with_special(&floats)
        } else {
            self.encode(&floats)
        }
    }

    /// Decode BUFF-encoded data to `decimal_bytes::Decimal64` values.
    ///
    /// The decoded Decimal64 values will have the specified `scale`.
    ///
    /// Special values are converted to `Decimal64::nan()`, `Decimal64::infinity()`,
    /// or `Decimal64::neg_infinity()` as appropriate.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The BUFF-encoded data
    /// * `scale` - The scale for the output Decimal64 values (0-18)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use buff_rs::BuffCodec;
    /// use decimal_bytes::Decimal64;
    ///
    /// let codec = BuffCodec::new(1000);
    /// let data = vec![1.234, 5.678];
    /// let encoded = codec.encode(&data).unwrap();
    /// let decimals: Vec<Decimal64> = codec.decode_to_decimal64s(&encoded, 3).unwrap();
    /// ```
    pub fn decode_to_decimal64s(
        &self,
        bytes: &[u8],
        scale: u8,
    ) -> Result<Vec<Decimal64>, BuffError> {
        let floats = self.decode(bytes)?;

        floats
            .into_iter()
            .map(|f| {
                if f.is_nan() {
                    Ok(Decimal64::nan())
                } else if f == f64::INFINITY {
                    Ok(Decimal64::infinity())
                } else if f == f64::NEG_INFINITY {
                    Ok(Decimal64::neg_infinity())
                } else {
                    // Convert f64 to Decimal64 with the specified scale
                    f64_to_decimal64(f, scale).map_err(|e| {
                        BuffError::InvalidData(format!("failed to create Decimal64: {}", e))
                    })
                }
            })
            .collect()
    }

    /// Convert a single f64 to `decimal_bytes::Decimal64`.
    ///
    /// # Arguments
    ///
    /// * `value` - The f64 value to convert
    /// * `scale` - The scale for the output Decimal64 (0-18)
    pub fn f64_to_decimal64(&self, value: f64, scale: u8) -> Result<Decimal64, BuffError> {
        if value.is_nan() {
            Ok(Decimal64::nan())
        } else if value == f64::INFINITY {
            Ok(Decimal64::infinity())
        } else if value == f64::NEG_INFINITY {
            Ok(Decimal64::neg_infinity())
        } else {
            f64_to_decimal64(value, scale)
                .map_err(|e| BuffError::InvalidData(format!("failed to create Decimal64: {}", e)))
        }
    }

    /// Convert a `decimal_bytes::Decimal64` to f64.
    ///
    /// This is a convenience method for single-value conversions.
    /// Since Decimal64 has ≤16 significant digits, precision loss is minimal.
    pub fn decimal64_to_f64(&self, decimal: &Decimal64) -> f64 {
        decimal64_to_f64(decimal)
    }
}

// ==================== Helper Functions ====================

/// Convert a Decimal64 to f64.
fn decimal64_to_f64(d: &Decimal64) -> f64 {
    if d.is_nan() {
        f64::NAN
    } else if d.is_pos_infinity() {
        f64::INFINITY
    } else if d.is_neg_infinity() {
        f64::NEG_INFINITY
    } else {
        // Get the raw value and scale
        let value = d.value();
        let scale = d.scale();

        if scale == 0 {
            value as f64
        } else {
            value as f64 / 10f64.powi(scale as i32)
        }
    }
}

/// Convert f64 to Decimal64 with specified scale.
fn f64_to_decimal64(value: f64, scale: u8) -> Result<Decimal64, decimal_bytes::DecimalError> {
    // Format with appropriate precision
    let precision = scale as usize;
    let formatted = format!("{:.prec$}", value, prec = precision);
    Decimal64::new(&formatted, scale)
}

/// Extension trait for converting Decimal arrays to f64 for BUFF encoding.
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

/// Extension trait for converting Decimal64 arrays to f64 for BUFF encoding.
pub trait Decimal64ArrayExt {
    /// Convert Decimal64 array to f64 array for BUFF encoding.
    fn decimal64s_to_f64_array(&self) -> Vec<f64>;
}

impl Decimal64ArrayExt for [Decimal64] {
    fn decimal64s_to_f64_array(&self) -> Vec<f64> {
        self.iter().map(decimal64_to_f64).collect()
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

    // ==================== Decimal64 Tests ====================

    #[test]
    fn test_encode_decode_decimal64s() {
        let codec = BuffCodec::new(1000);

        let decimals: Vec<Decimal64> = vec![
            Decimal64::new("1.234", 3).unwrap(),
            Decimal64::new("5.678", 3).unwrap(),
            Decimal64::new("9.012", 3).unwrap(),
        ];

        let encoded = codec.encode_decimal64s(&decimals).unwrap();
        let decoded = codec.decode_to_decimal64s(&encoded, 3).unwrap();

        assert_eq!(decoded.len(), 3);
        // Check approximate equality (precision loss expected)
        for (orig, dec) in decimals.iter().zip(decoded.iter()) {
            let orig_f = decimal64_to_f64(orig);
            let dec_f = decimal64_to_f64(dec);
            assert!((orig_f - dec_f).abs() < 0.001, "orig={}, dec={}", orig, dec);
        }
    }

    #[test]
    fn test_special_values_decimal64() {
        let codec = BuffCodec::new(1000);

        let decimals: Vec<Decimal64> = vec![
            Decimal64::new("1.0", 1).unwrap(),
            Decimal64::infinity(),
            Decimal64::new("2.0", 1).unwrap(),
            Decimal64::nan(),
            Decimal64::neg_infinity(),
        ];

        let encoded = codec.encode_decimal64s(&decimals).unwrap();
        let decoded = codec.decode_to_decimal64s(&encoded, 1).unwrap();

        assert_eq!(decoded.len(), 5);
        assert!(!decoded[0].is_special());
        assert!(decoded[1].is_pos_infinity());
        assert!(!decoded[2].is_special());
        assert!(decoded[3].is_nan());
        assert!(decoded[4].is_neg_infinity());
    }

    #[test]
    fn test_f64_to_decimal64_conversion() {
        let codec = BuffCodec::new(1000);

        let d64 = codec.f64_to_decimal64(3.14159, 3).unwrap();
        let f = codec.decimal64_to_f64(&d64);

        assert!((f - 3.142).abs() < 0.001); // Rounded to 3 decimal places
    }

    #[test]
    fn test_decimal64_array_ext() {
        let decimals: Vec<Decimal64> = vec![
            Decimal64::new("1.5", 1).unwrap(),
            Decimal64::infinity(),
            Decimal64::new("2.5", 1).unwrap(),
        ];

        let floats = decimals.decimal64s_to_f64_array();

        assert!((floats[0] - 1.5).abs() < f64::EPSILON);
        assert!(floats[1].is_infinite() && floats[1].is_sign_positive());
        assert!((floats[2] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decimal64_direct_conversion() {
        // Test direct value/scale conversion without string parsing
        let d64 = Decimal64::new("123.45", 2).unwrap();
        let f = decimal64_to_f64(&d64);
        assert!((f - 123.45).abs() < f64::EPSILON);

        let d64_neg = Decimal64::new("-99.99", 2).unwrap();
        let f_neg = decimal64_to_f64(&d64_neg);
        assert!((f_neg - (-99.99)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decimal64_roundtrip_various_scales() {
        let codec = BuffCodec::new(10000); // 4 decimal places

        // Test with different scales
        for scale in [0u8, 1, 2, 3, 4] {
            let value = format!("{:.prec$}", 123.456789, prec = scale as usize);
            let d64 = Decimal64::new(&value, scale).unwrap();

            let encoded = codec.encode_decimal64s(&[d64]).unwrap();
            let decoded = codec.decode_to_decimal64s(&encoded, scale).unwrap();

            let orig_f = decimal64_to_f64(&d64);
            let dec_f = decimal64_to_f64(&decoded[0]);

            assert!(
                (orig_f - dec_f).abs() < 0.0001,
                "scale={}, orig={}, decoded={}",
                scale,
                orig_f,
                dec_f
            );
        }
    }

    #[test]
    fn test_encode_decimals_empty() {
        let codec = BuffCodec::new(1000);
        let result = codec.encode_decimals(&[]);
        assert!(matches!(result, Err(BuffError::EmptyInput)));
    }

    #[test]
    fn test_encode_decimal64s_empty() {
        let codec = BuffCodec::new(1000);
        let result = codec.encode_decimal64s(&[]);
        assert!(matches!(result, Err(BuffError::EmptyInput)));
    }

    #[test]
    fn test_decimal_to_f64_special_values() {
        let codec = BuffCodec::new(1000);

        // Test NaN
        let nan = Decimal::nan();
        let f_nan = codec.decimal_to_f64(&nan);
        assert!(f_nan.is_nan());

        // Test +Infinity
        let inf = Decimal::infinity();
        let f_inf = codec.decimal_to_f64(&inf);
        assert!(f_inf.is_infinite() && f_inf.is_sign_positive());

        // Test -Infinity
        let neg_inf = Decimal::neg_infinity();
        let f_neg_inf = codec.decimal_to_f64(&neg_inf);
        assert!(f_neg_inf.is_infinite() && f_neg_inf.is_sign_negative());
    }

    #[test]
    fn test_f64_to_decimal_special_values() {
        let codec = BuffCodec::new(1000);

        // Test NaN
        let dec_nan = codec.f64_to_decimal(f64::NAN).unwrap();
        assert!(dec_nan.is_nan());

        // Test +Infinity
        let dec_inf = codec.f64_to_decimal(f64::INFINITY).unwrap();
        assert!(dec_inf.is_infinity() && !dec_inf.is_negative());

        // Test -Infinity
        let dec_neg_inf = codec.f64_to_decimal(f64::NEG_INFINITY).unwrap();
        assert!(dec_neg_inf.is_infinity() && dec_neg_inf.is_negative());
    }

    #[test]
    fn test_f64_to_decimal64_special_values() {
        let codec = BuffCodec::new(1000);

        // Test NaN
        let d64_nan = codec.f64_to_decimal64(f64::NAN, 3).unwrap();
        assert!(d64_nan.is_nan());

        // Test +Infinity
        let d64_inf = codec.f64_to_decimal64(f64::INFINITY, 3).unwrap();
        assert!(d64_inf.is_pos_infinity());

        // Test -Infinity
        let d64_neg_inf = codec.f64_to_decimal64(f64::NEG_INFINITY, 3).unwrap();
        assert!(d64_neg_inf.is_neg_infinity());
    }

    #[test]
    fn test_decimal64_to_f64_scale_zero() {
        // Test with scale=0 (integer)
        let d64 = Decimal64::new("12345", 0).unwrap();
        let f = decimal64_to_f64(&d64);
        assert!((f - 12345.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decimal64_to_f64_nan() {
        let d64 = Decimal64::nan();
        let f = decimal64_to_f64(&d64);
        assert!(f.is_nan());
    }

    #[test]
    fn test_decimal64_to_f64_pos_infinity() {
        let d64 = Decimal64::infinity();
        let f = decimal64_to_f64(&d64);
        assert!(f.is_infinite() && f.is_sign_positive());
    }

    #[test]
    fn test_decimal64_to_f64_neg_infinity() {
        let d64 = Decimal64::neg_infinity();
        let f = decimal64_to_f64(&d64);
        assert!(f.is_infinite() && f.is_sign_negative());
    }

    #[test]
    fn test_decimal_array_ext_neg_infinity() {
        let decimals: Vec<Decimal> = vec!["1.0".parse().unwrap(), Decimal::neg_infinity()];

        let floats = decimals.to_f64_array();

        assert!((floats[0] - 1.0).abs() < f64::EPSILON);
        assert!(floats[1].is_infinite() && floats[1].is_sign_negative());
    }

    #[test]
    fn test_decimal_array_ext_nan() {
        let decimals: Vec<Decimal> = vec!["1.0".parse().unwrap(), Decimal::nan()];

        let floats = decimals.to_f64_array();

        assert!((floats[0] - 1.0).abs() < f64::EPSILON);
        assert!(floats[1].is_nan());
    }

    #[test]
    fn test_decimal64_array_ext_special() {
        let decimals: Vec<Decimal64> = vec![Decimal64::nan(), Decimal64::neg_infinity()];

        let floats = decimals.decimal64s_to_f64_array();

        assert!(floats[0].is_nan());
        assert!(floats[1].is_infinite() && floats[1].is_sign_negative());
    }

    #[test]
    fn test_negative_decimals() {
        let codec = BuffCodec::new(1000);

        let decimals: Vec<Decimal> = vec![
            "-1.234".parse().unwrap(),
            "-5.678".parse().unwrap(),
            "0.0".parse().unwrap(),
        ];

        let encoded = codec.encode_decimals(&decimals).unwrap();
        let decoded = codec.decode_to_decimals(&encoded).unwrap();

        assert_eq!(decoded.len(), 3);
        for (orig, dec) in decimals.iter().zip(decoded.iter()) {
            let orig_f: f64 = orig.to_string().parse().unwrap();
            let dec_f: f64 = dec.to_string().parse().unwrap();
            assert!((orig_f - dec_f).abs() < 0.001, "orig={}, dec={}", orig, dec);
        }
    }

    #[test]
    fn test_negative_decimal64s() {
        let codec = BuffCodec::new(1000);

        let decimals: Vec<Decimal64> = vec![
            Decimal64::new("-1.234", 3).unwrap(),
            Decimal64::new("-5.678", 3).unwrap(),
            Decimal64::new("0.0", 3).unwrap(),
        ];

        let encoded = codec.encode_decimal64s(&decimals).unwrap();
        let decoded = codec.decode_to_decimal64s(&encoded, 3).unwrap();

        assert_eq!(decoded.len(), 3);
        for (orig, dec) in decimals.iter().zip(decoded.iter()) {
            let orig_f = decimal64_to_f64(orig);
            let dec_f = decimal64_to_f64(dec);
            assert!((orig_f - dec_f).abs() < 0.001, "orig={}, dec={}", orig, dec);
        }
    }
}
