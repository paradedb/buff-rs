//! Error types for BUFF encoding/decoding operations.

use thiserror::Error;

/// Errors that can occur during BUFF operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum BuffError {
    /// The input data is empty.
    #[error("input data is empty")]
    EmptyInput,

    /// Invalid precision value (must be >= 0).
    #[error("invalid precision: {0} (must be >= 0)")]
    InvalidPrecision(i32),

    /// The encoded data is corrupted or invalid.
    #[error("invalid encoded data: {0}")]
    InvalidData(String),

    /// Buffer overflow during bit packing operations.
    #[error("buffer overflow: attempted to write {attempted} bits, only {available} available")]
    BufferOverflow {
        /// The number of bits that were attempted to be written.
        attempted: usize,
        /// The number of bits available in the buffer.
        available: usize,
    },

    /// The bit width exceeds the maximum supported (32 bits).
    #[error("bit width {0} exceeds maximum of 32")]
    BitWidthExceeded(usize),

    /// Special float values (Infinity, NaN) cannot be converted to decimal.
    #[error("special float value cannot be converted: {0}")]
    SpecialValueConversion(String),

    /// Precision loss during decimal conversion exceeds acceptable threshold.
    #[error("precision loss too high: original={original}, converted={converted}")]
    PrecisionLoss {
        /// The original value.
        original: String,
        /// The converted value.
        converted: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_empty_input() {
        let err = BuffError::EmptyInput;
        assert_eq!(err.to_string(), "input data is empty");
    }

    #[test]
    fn test_error_display_invalid_precision() {
        let err = BuffError::InvalidPrecision(-5);
        assert!(err.to_string().contains("-5"));
        assert!(err.to_string().contains("must be >= 0"));
    }

    #[test]
    fn test_error_display_invalid_data() {
        let err = BuffError::InvalidData("corrupted header".to_string());
        assert!(err.to_string().contains("corrupted header"));
    }

    #[test]
    fn test_error_display_buffer_overflow() {
        let err = BuffError::BufferOverflow {
            attempted: 32,
            available: 8,
        };
        let msg = err.to_string();
        assert!(msg.contains("32"));
        assert!(msg.contains("8"));
    }

    #[test]
    fn test_error_display_bit_width_exceeded() {
        let err = BuffError::BitWidthExceeded(64);
        assert!(err.to_string().contains("64"));
        assert!(err.to_string().contains("32"));
    }

    #[test]
    fn test_error_display_special_value_conversion() {
        let err = BuffError::SpecialValueConversion("Infinity".to_string());
        assert!(err.to_string().contains("Infinity"));
    }

    #[test]
    fn test_error_display_precision_loss() {
        let err = BuffError::PrecisionLoss {
            original: "3.14159".to_string(),
            converted: "3.14".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("3.14159"));
        assert!(msg.contains("3.14"));
    }

    #[test]
    fn test_error_debug() {
        let err = BuffError::EmptyInput;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("EmptyInput"));
    }

    #[test]
    fn test_error_clone() {
        let err = BuffError::InvalidData("test".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_error_eq() {
        let err1 = BuffError::EmptyInput;
        let err2 = BuffError::EmptyInput;
        assert_eq!(err1, err2);

        let err3 = BuffError::BitWidthExceeded(33);
        let err4 = BuffError::BitWidthExceeded(33);
        assert_eq!(err3, err4);

        let err5 = BuffError::BitWidthExceeded(33);
        let err6 = BuffError::BitWidthExceeded(64);
        assert_ne!(err5, err6);
    }
}
