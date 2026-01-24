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
}
