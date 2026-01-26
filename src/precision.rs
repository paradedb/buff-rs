//! Precision bound computation for bounded floating-point values.
//!
//! This module implements the core algorithm for determining the minimum number
//! of bits needed to represent bounded floating-point values within a given
//! precision tolerance.

/// Mask for extracting the exponent bits from an IEEE 754 double.
const EXP_MASK: u64 = 0x7FF0_0000_0000_0000;

/// Mask for the sign bit (most significant bit).
const FIRST_ONE: u64 = 0x8000_0000_0000_0000;

/// All ones (for complement operations).
const NEG_ONE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// Precision bound calculator for floating-point values.
///
/// Given a precision tolerance (e.g., 0.00005 for 4 decimal places),
/// this struct computes the minimum bits needed to represent values
/// within that tolerance, and provides methods to convert floats to
/// their fixed-point representation.
#[derive(Debug, Clone)]
pub struct PrecisionBound {
    /// Current bit position during bound calculation.
    position: u64,
    /// The precision tolerance (e.g., 0.00005).
    precision: f64,
    /// Exponent of the precision (derived from IEEE 754 representation).
    precision_exp: i32,
    /// Number of bits needed for the integer part.
    int_length: u64,
    /// Number of bits needed for the decimal part.
    decimal_length: u64,
}

impl PrecisionBound {
    /// Create a new precision bound calculator.
    ///
    /// # Arguments
    /// * `precision` - The precision tolerance (e.g., 0.00005 for 4 decimal places)
    ///
    /// # Example
    /// ```
    /// use buff_rs::precision::PrecisionBound;
    ///
    /// let bound = PrecisionBound::new(0.00005);
    /// ```
    pub fn new(precision: f64) -> Self {
        let xu = precision.to_bits();
        let precision_exp = ((xu & EXP_MASK) >> 52) as i32 - 1023;

        PrecisionBound {
            position: 0,
            precision,
            precision_exp,
            int_length: 0,
            decimal_length: 0,
        }
    }

    /// Compute the precision-bounded representation of a float.
    ///
    /// This finds the minimum precision representation of `orig` that
    /// is within the precision tolerance.
    pub fn precision_bound(&mut self, orig: f64) -> f64 {
        let mask = !0u64;
        let origu = orig.to_bits();

        let mut curu = origu & (mask << self.position) | (1u64 << self.position);
        let mut cur = f64::from_bits(curu);
        let mut pre = cur;
        let bounded = self.is_bounded(orig, cur);

        if bounded {
            // Find first bit where it's not bounded
            loop {
                if self.position == 52 {
                    return pre;
                }
                self.position += 1;
                curu = origu & (mask << self.position) | (1u64 << self.position);
                cur = f64::from_bits(curu);
                if !self.is_bounded(orig, cur) {
                    break;
                }
                pre = cur;
            }
        } else {
            // Find the first bit where it's bounded
            loop {
                if self.position == 0 {
                    break;
                }
                self.position -= 1;
                curu = origu & (mask << self.position) | (1u64 << self.position);
                cur = f64::from_bits(curu);
                if self.is_bounded(orig, cur) {
                    pre = cur;
                    break;
                }
            }
        }
        pre
    }

    /// Calculate and update the required bit lengths for a precision-bounded value.
    pub fn cal_length(&mut self, x: f64) {
        let xu = x.to_bits();
        let trailing_zeros = xu.trailing_zeros();
        let exp = ((xu & EXP_MASK) >> 52) as i32 - 1023;

        let mut dec_length = 0u64;
        if trailing_zeros >= 52 {
            if exp < 0 {
                dec_length = (-exp) as u64;
                if exp < self.precision_exp {
                    dec_length = 0;
                }
            }
        } else if (52 - trailing_zeros as i32) > exp {
            dec_length = ((52 - trailing_zeros as i32) - exp) as u64;
        }

        if exp >= 0 && (exp + 1) as u64 > self.int_length {
            self.int_length = (exp + 1) as u64;
        }
        if dec_length > self.decimal_length {
            self.decimal_length = dec_length;
        }
    }

    /// Get the computed integer and decimal bit lengths.
    pub fn get_length(&self) -> (u64, u64) {
        (self.int_length, self.decimal_length)
    }

    /// Set the integer and decimal bit lengths manually.
    #[inline]
    pub fn set_length(&mut self, ilen: u64, dlen: u64) {
        self.int_length = ilen;
        self.decimal_length = dlen;
    }

    /// Fetch the integer and decimal components of a precision-bounded float.
    #[inline]
    pub fn fetch_components(&self, bd: f64) -> (i64, u64) {
        let bdu = bd.to_bits();
        let exp = ((bdu & EXP_MASK) >> 52) as i32 - 1023;
        let sign = bdu & FIRST_ONE;
        let mut int_part = 0u64;
        let mut dec_part: u64;

        if exp >= 0 {
            dec_part = bdu << (12 + exp) as u64;
            int_part = ((bdu << 11) | FIRST_ONE) >> (63 - exp) as u64;
            if sign != 0 {
                int_part = !int_part;
                dec_part = (!dec_part).wrapping_add(1);
            }
        } else if exp < self.precision_exp {
            dec_part = 0u64;
            if sign != 0 {
                int_part = NEG_ONE;
                dec_part = !dec_part;
            }
        } else {
            dec_part = ((bdu << 11) | FIRST_ONE) >> ((-exp - 1) as u64);
            if sign != 0 {
                int_part = NEG_ONE;
                dec_part = !dec_part;
            }
        }

        let signed_int = int_part as i64;
        (signed_int, dec_part >> (64u64 - self.decimal_length))
    }

    /// Fetch the byte-aligned fixed-point representation.
    ///
    /// This converts a float to its fixed-point representation suitable
    /// for byte-sliced storage.
    #[inline]
    pub fn fetch_fixed_aligned(&self, bd: f64) -> i64 {
        let bdu = bd.to_bits();
        let exp = ((bdu & EXP_MASK) >> 52) as i32 - 1023;
        let sign = bdu & FIRST_ONE;
        let mut fixed: u64;

        if exp < self.precision_exp {
            fixed = 0u64;
        } else {
            fixed = ((bdu << 11) | FIRST_ONE) >> (63 - exp - self.decimal_length as i32) as u64;
            if sign != 0 {
                fixed = !(fixed.wrapping_sub(1));
            }
        }

        fixed as i64
    }

    /// Check if two values are within the precision tolerance.
    #[inline]
    pub fn is_bounded(&self, a: f64, b: f64) -> bool {
        (a - b).abs() < self.precision
    }
}

/// Get the precision bound value for a given number of decimal places.
///
/// # Arguments
/// * `precision` - Number of decimal places (e.g., 4 for 0.0001 precision)
///
/// # Returns
/// The precision tolerance value (e.g., 0.000049 for 4 decimal places)
pub fn get_precision_bound(precision: i32) -> f64 {
    if precision <= 0 {
        return 0.49;
    }

    let mut s = String::from("0.");
    for _ in 0..precision {
        s.push('0');
    }
    s.push_str("49");
    s.parse().unwrap_or(0.49)
}

/// Precomputed decimal lengths for common precision values.
///
/// Maps precision (decimal places) to the number of bits needed for the decimal part.
pub const PRECISION_MAP: [(i32, u64); 16] = [
    (0, 0),
    (1, 4),
    (2, 7),
    (3, 10),
    (4, 14),
    (5, 17),
    (6, 20),
    (7, 24),
    (8, 27),
    (9, 30),
    (10, 34),
    (11, 37),
    (12, 40),
    (13, 44),
    (14, 47),
    (15, 50),
];

/// Get the decimal length for a given precision.
pub fn get_decimal_length(precision: i32) -> u64 {
    PRECISION_MAP
        .iter()
        .find(|(p, _)| *p == precision)
        .map(|(_, len)| *len)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_bound_creation() {
        let bound = PrecisionBound::new(0.00005);
        assert!(bound.precision > 0.0);
    }

    #[test]
    fn test_get_precision_bound() {
        let pb = get_precision_bound(4);
        assert!(pb > 0.0 && pb < 0.001);
    }

    #[test]
    fn test_is_bounded() {
        let bound = PrecisionBound::new(0.05);
        assert!(bound.is_bounded(1.0, 1.01));
        assert!(!bound.is_bounded(1.0, 1.1));
    }

    #[test]
    fn test_precision_map() {
        assert_eq!(get_decimal_length(0), 0);
        assert_eq!(get_decimal_length(4), 14);
        assert_eq!(get_decimal_length(10), 34);
    }

    #[test]
    fn test_fetch_fixed_aligned() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(4, 14);

        let fixed = bound.fetch_fixed_aligned(3.14159);
        assert!(fixed > 0);
    }

    #[test]
    fn test_fetch_fixed_aligned_negative() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(4, 14);

        let fixed = bound.fetch_fixed_aligned(-3.14159);
        assert!(fixed < 0);
    }

    #[test]
    fn test_fetch_fixed_aligned_zero() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(4, 14);

        let fixed = bound.fetch_fixed_aligned(0.0);
        assert_eq!(fixed, 0);
    }

    #[test]
    fn test_fetch_fixed_aligned_small() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(4, 14);

        // Very small value below precision threshold
        let fixed = bound.fetch_fixed_aligned(0.00001);
        // Should be within precision bounds
        assert!(fixed.abs() <= 1);
    }

    #[test]
    fn test_get_length() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(8, 16);

        let (ilen, dlen) = bound.get_length();
        assert_eq!(ilen, 8);
        assert_eq!(dlen, 16);
    }

    #[test]
    fn test_cal_length() {
        let mut bound = PrecisionBound::new(0.00005);

        // Large value should increase int_length
        bound.cal_length(1000.0);
        let (ilen, _dlen) = bound.get_length();
        assert!(ilen > 0);

        // Small decimal should affect decimal_length
        let mut bound2 = PrecisionBound::new(0.00005);
        bound2.cal_length(0.123456);
        let (_, dlen) = bound2.get_length();
        assert!(dlen > 0);
    }

    #[test]
    fn test_fetch_components() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(4, 14);

        let (int_part, dec_part) = bound.fetch_components(3.5);
        assert_eq!(int_part, 3);
        assert!(dec_part > 0);
    }

    #[test]
    fn test_fetch_components_negative() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(4, 14);

        let (int_part, _dec_part) = bound.fetch_components(-3.5);
        assert!(int_part < 0);
    }

    #[test]
    fn test_fetch_components_small_value() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(0, 14);

        // Value smaller than precision threshold
        let (int_part, dec_part) = bound.fetch_components(0.001);
        assert_eq!(int_part, 0);
        assert!(dec_part >= 0);
    }

    #[test]
    fn test_precision_bound_method() {
        let mut bound = PrecisionBound::new(0.005);

        // Should find bounded representation
        let result = bound.precision_bound(3.14159);
        assert!((result - 3.14159).abs() < 0.01);
    }

    #[test]
    fn test_precision_bound_exact() {
        let mut bound = PrecisionBound::new(0.05);

        // Value that's already at a good precision boundary
        let result = bound.precision_bound(1.0);
        assert!((result - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_get_precision_bound_zero() {
        let pb = get_precision_bound(0);
        assert!((pb - 0.49).abs() < 0.01);
    }

    #[test]
    fn test_get_precision_bound_negative() {
        let pb = get_precision_bound(-5);
        assert!((pb - 0.49).abs() < 0.01);
    }

    #[test]
    fn test_get_precision_bound_high() {
        let pb = get_precision_bound(10);
        assert!(pb < 0.0000000001);
        assert!(pb > 0.0);
    }

    #[test]
    fn test_get_decimal_length_unknown() {
        // Precision value not in the map
        let len = get_decimal_length(99);
        assert_eq!(len, 0);
    }

    #[test]
    fn test_get_decimal_length_all_values() {
        // Test all values in PRECISION_MAP
        for (prec, expected_len) in PRECISION_MAP {
            assert_eq!(get_decimal_length(prec), expected_len);
        }
    }

    #[test]
    fn test_precision_bound_debug() {
        let bound = PrecisionBound::new(0.00005);
        let debug_str = format!("{:?}", bound);
        assert!(debug_str.contains("PrecisionBound"));
    }

    #[test]
    fn test_precision_bound_clone() {
        let bound = PrecisionBound::new(0.00005);
        let cloned = bound.clone();
        assert_eq!(bound.precision, cloned.precision);
    }

    #[test]
    fn test_cal_length_negative_exponent() {
        let mut bound = PrecisionBound::new(0.0005);

        // Value with negative exponent (very small)
        bound.cal_length(0.00123);
        let (_, dlen) = bound.get_length();
        assert!(dlen > 0);
    }

    #[test]
    fn test_cal_length_large_exponent() {
        let mut bound = PrecisionBound::new(0.0005);

        // Value with large exponent
        bound.cal_length(1000000.0);
        let (ilen, _) = bound.get_length();
        assert!(ilen > 10); // 2^20 ≈ 1M, so need at least 20 bits
    }

    #[test]
    fn test_cal_length_power_of_two() {
        let mut bound = PrecisionBound::new(0.0005);

        // Exact power of two
        bound.cal_length(1024.0);
        let (ilen, _) = bound.get_length();
        assert!(ilen > 0);
    }

    #[test]
    fn test_fetch_components_power_of_two() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(16, 14);

        let (int_part, dec_part) = bound.fetch_components(256.0);
        assert_eq!(int_part, 256);
        assert_eq!(dec_part, 0);
    }

    #[test]
    fn test_fetch_components_with_decimal() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(4, 14);

        let (int_part, dec_part) = bound.fetch_components(5.5);
        assert_eq!(int_part, 5);
        assert!(dec_part > 0); // Should have decimal part
    }

    #[test]
    fn test_precision_bound_very_tight() {
        let mut bound = PrecisionBound::new(0.000001); // Very tight precision

        let result = bound.precision_bound(1.23456789);
        assert!((result - 1.23456789).abs() < 0.00001);
    }

    #[test]
    fn test_precision_bound_loose() {
        let mut bound = PrecisionBound::new(0.5); // Loose precision

        let result = bound.precision_bound(1.7);
        assert!((result - 1.7).abs() < 1.0);
    }

    #[test]
    fn test_fetch_fixed_aligned_large_value() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(20, 14);

        let fixed = bound.fetch_fixed_aligned(1000000.123);
        assert!(fixed > 0);
    }

    #[test]
    fn test_fetch_fixed_aligned_very_small() {
        let mut bound = PrecisionBound::new(0.00005);
        bound.set_length(0, 14);

        // Value smaller than precision - should be 0
        let fixed = bound.fetch_fixed_aligned(0.0000001);
        assert_eq!(fixed, 0);
    }

    #[test]
    fn test_is_bounded_same_value() {
        let bound = PrecisionBound::new(0.05);
        assert!(bound.is_bounded(5.0, 5.0));
    }

    #[test]
    fn test_is_bounded_at_boundary() {
        let bound = PrecisionBound::new(0.05);
        // Exactly at the boundary
        assert!(bound.is_bounded(1.0, 1.049));
        assert!(!bound.is_bounded(1.0, 1.06));
    }

    #[test]
    fn test_get_precision_bound_various() {
        for prec in 1..=15 {
            let pb = get_precision_bound(prec);
            assert!(pb > 0.0);
            assert!(pb < 1.0);
        }
    }

    #[test]
    fn test_precision_map_coverage() {
        // Verify all precision map entries are valid
        for (prec, len) in PRECISION_MAP {
            assert!(prec >= 0);
            assert!(len <= 64);
        }
    }

    #[test]
    fn test_fetch_components_negative_exponent_below_precision() {
        let mut bound = PrecisionBound::new(0.005); // precision_exp around -8
        bound.set_length(0, 10);

        // Value with exponent below precision threshold
        let (int_part, dec_part) = bound.fetch_components(0.0000001);
        assert_eq!(int_part, 0);
        // dec_part should be 0 or very small
        assert!(dec_part == 0 || dec_part < 10);
    }

    #[test]
    fn test_fetch_components_negative_value_small() {
        let mut bound = PrecisionBound::new(0.005);
        bound.set_length(0, 10);

        let (int_part, dec_part) = bound.fetch_components(-0.001);
        // For small negative values below precision
        assert!(int_part <= 0 || dec_part > 0);
    }

    #[test]
    fn test_cal_length_trailing_zeros() {
        let mut bound = PrecisionBound::new(0.0005);

        // Value with many trailing zeros in binary representation
        bound.cal_length(8.0); // 8 = 2^3, has trailing zeros
        let (ilen, _) = bound.get_length();
        assert!(ilen >= 4); // 2^4 > 8
    }
}
