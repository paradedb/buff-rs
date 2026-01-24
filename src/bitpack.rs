//! Bit packing utilities for efficient storage.
//!
//! This module provides a `BitPack` type for reading and writing
//! variable-width integers to a byte buffer.

use crate::error::BuffError;

/// Maximum number of bits that can be written in a single operation.
pub const MAX_BITS: usize = 32;

/// Number of bits in a byte.
const BYTE_BITS: usize = 8;

/// A bit packer for reading and writing variable-width integers.
///
/// This supports both reading from a byte slice and writing to a growable Vec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitPack<B> {
    buff: B,
    cursor: usize,
    bits: usize,
}

impl<B> BitPack<B> {
    /// Create a new BitPack with the given buffer.
    #[inline]
    pub fn new(buff: B) -> Self {
        BitPack {
            buff,
            cursor: 0,
            bits: 0,
        }
    }

    /// Get the total number of bits processed so far.
    #[inline]
    pub fn sum_bits(&self) -> usize {
        self.cursor * BYTE_BITS + self.bits
    }

    /// Set the cursor position.
    #[inline]
    pub fn with_cursor(&mut self, cursor: usize) -> &mut Self {
        self.cursor = cursor;
        self
    }

    /// Set the bit position within the current byte.
    #[inline]
    pub fn with_bits(&mut self, bits: usize) -> &mut Self {
        self.bits = bits;
        self
    }
}

impl<B: AsRef<[u8]>> BitPack<B> {
    /// Get a reference to the underlying buffer as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.buff.as_ref()
    }
}

// Reading operations for byte slices
impl BitPack<&[u8]> {
    /// Read `bits` bits from the buffer and return as u32.
    ///
    /// # Arguments
    /// * `bits` - Number of bits to read (max 32)
    ///
    /// # Returns
    /// The value read, or an error if there are not enough bits available.
    pub fn read(&mut self, mut bits: usize) -> Result<u32, BuffError> {
        if bits > MAX_BITS {
            return Err(BuffError::BitWidthExceeded(bits));
        }
        if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            return Err(BuffError::BufferOverflow {
                attempted: bits,
                available: self.buff.len() * BYTE_BITS - self.sum_bits(),
            });
        }

        let mut bits_left = 0u32;
        let mut output = 0u32;

        loop {
            let byte_left = BYTE_BITS - self.bits;

            if bits <= byte_left {
                let mut bb = self.buff[self.cursor] as u32;
                bb >>= self.bits as u32;
                bb &= ((1 << bits) - 1) as u32;
                output |= bb << bits_left;
                self.bits += bits;
                break;
            }

            let mut bb = self.buff[self.cursor] as u32;
            bb >>= self.bits as u32;
            bb &= ((1 << byte_left) - 1) as u32;
            output |= bb << bits_left;
            self.bits += byte_left;
            bits_left += byte_left as u32;
            bits -= byte_left;

            if self.bits >= BYTE_BITS {
                self.cursor += 1;
                self.bits -= BYTE_BITS;
            }
        }

        Ok(output)
    }

    /// Read a single byte from the buffer.
    #[inline]
    pub fn read_byte(&mut self) -> Result<u8, BuffError> {
        self.cursor += 1;
        if self.cursor >= self.buff.len() {
            return Err(BuffError::InvalidData("unexpected end of buffer".into()));
        }
        Ok(self.buff[self.cursor])
    }

    /// Read `n` bytes from the buffer.
    #[inline]
    pub fn read_n_byte(&mut self, n: usize) -> Result<&[u8], BuffError> {
        self.cursor += 1;
        let end = self.cursor + n;
        if end > self.buff.len() {
            return Err(BuffError::BufferOverflow {
                attempted: n,
                available: self.buff.len() - self.cursor,
            });
        }
        let output = &self.buff[self.cursor..end];
        self.cursor += n - 1;
        Ok(output)
    }

    /// Read `n` bytes from the buffer at a given offset without advancing cursor.
    #[inline]
    pub fn read_n_byte_unmut(&self, start: usize, n: usize) -> Result<&[u8], BuffError> {
        let s = start + self.cursor + 1;
        let end = s + n;
        if end > self.buff.len() {
            return Err(BuffError::BufferOverflow {
                attempted: n,
                available: self.buff.len().saturating_sub(s),
            });
        }
        Ok(&self.buff[s..end])
    }

    /// Skip `n` bytes.
    #[inline]
    pub fn skip_n_byte(&mut self, n: usize) {
        self.cursor += n;
    }

    /// Skip `bits` bits.
    #[inline]
    pub fn skip(&mut self, bits: usize) -> Result<(), BuffError> {
        if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            return Err(BuffError::BufferOverflow {
                attempted: bits,
                available: self.buff.len() * BYTE_BITS - self.sum_bits(),
            });
        }

        let bytes = bits / BYTE_BITS;
        let left = bits % BYTE_BITS;
        let cur_bits = self.bits + left;
        self.cursor = self.cursor + bytes + cur_bits / BYTE_BITS;
        self.bits = cur_bits % BYTE_BITS;

        Ok(())
    }

    /// Finish reading the current byte and move to the next.
    #[inline]
    pub fn finish_read_byte(&mut self) {
        self.cursor += 1;
        self.bits = 0;
    }
}

// Writing operations for mutable byte slices
impl BitPack<&mut [u8]> {
    /// Write `bits` bits of `value` to the buffer.
    pub fn write(&mut self, mut value: u32, mut bits: usize) -> Result<(), BuffError> {
        if bits > MAX_BITS {
            return Err(BuffError::BitWidthExceeded(bits));
        }
        if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            return Err(BuffError::BufferOverflow {
                attempted: bits,
                available: self.buff.len() * BYTE_BITS - self.sum_bits(),
            });
        }

        if bits < MAX_BITS {
            value &= ((1 << bits) - 1) as u32;
        }

        loop {
            let bits_left = BYTE_BITS - self.bits;

            if bits <= bits_left {
                self.buff[self.cursor] |= (value as u8) << self.bits as u8;
                self.bits += bits;

                if self.bits >= BYTE_BITS {
                    self.cursor += 1;
                    self.bits = 0;
                }
                break;
            }

            let bb = value & ((1 << bits_left) - 1) as u32;
            self.buff[self.cursor] |= (bb as u8) << self.bits as u8;
            self.cursor += 1;
            self.bits = 0;
            value >>= bits_left as u32;
            bits -= bits_left;
        }

        Ok(())
    }
}

impl Default for BitPack<Vec<u8>> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

// Writing operations for growable Vec
impl BitPack<Vec<u8>> {
    /// Create a new BitPack with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(Vec::with_capacity(capacity))
    }

    /// Write `bits` bits of `value` to the buffer.
    ///
    /// The buffer will grow as needed.
    #[inline]
    pub fn write(&mut self, value: u32, bits: usize) -> Result<(), BuffError> {
        if bits > MAX_BITS {
            return Err(BuffError::BitWidthExceeded(bits));
        }

        let len = self.buff.len();
        if let Some(bits_needed) = (self.sum_bits() + bits).checked_sub(len * BYTE_BITS) {
            self.buff.resize(len + bits_needed.div_ceil(BYTE_BITS), 0x0);
        }

        let mut bitpack = BitPack {
            buff: self.buff.as_mut_slice(),
            cursor: self.cursor,
            bits: self.bits,
        };

        bitpack.write(value, bits)?;

        self.bits = bitpack.bits;
        self.cursor = bitpack.cursor;

        Ok(())
    }

    /// Write a single byte to the buffer.
    #[inline]
    pub fn write_byte(&mut self, value: u8) -> Result<(), BuffError> {
        self.buff.push(value);
        Ok(())
    }

    /// Write multiple bytes to the buffer.
    #[inline]
    pub fn write_bytes(&mut self, values: &[u8]) {
        self.buff.extend_from_slice(values);
    }

    /// Finish writing the current byte and prepare for the next.
    #[inline]
    pub fn finish_write_byte(&mut self) {
        let len = self.buff.len();
        self.buff.resize(len + 1, 0x0);
        self.bits = 0;
        self.cursor = len;
    }

    /// Consume the BitPack and return the underlying buffer.
    #[inline]
    pub fn into_vec(self) -> Vec<u8> {
        self.buff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_read_roundtrip() {
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(10, 4).unwrap();
        bitpack_vec.write(1021, 10).unwrap();
        bitpack_vec.write(3, 2).unwrap();

        let mut bitpack = BitPack::<&[u8]>::new(bitpack_vec.as_slice());
        assert_eq!(bitpack.read(4).unwrap(), 10);
        assert_eq!(bitpack.read(10).unwrap(), 1021);
        assert_eq!(bitpack.read(2).unwrap(), 3);
    }

    #[test]
    fn test_single_bits() {
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(1);
        bitpack_vec.write(1, 1).unwrap();
        bitpack_vec.write(0, 1).unwrap();
        bitpack_vec.write(0, 1).unwrap();
        bitpack_vec.write(1, 1).unwrap();

        let mut bitpack = BitPack::<&[u8]>::new(bitpack_vec.as_slice());
        assert_eq!(bitpack.read(1).unwrap(), 1);
        assert_eq!(bitpack.read(1).unwrap(), 0);
        assert_eq!(bitpack.read(1).unwrap(), 0);
        assert_eq!(bitpack.read(1).unwrap(), 1);
    }

    #[test]
    fn test_full_bytes() {
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(255, 8).unwrap();
        bitpack_vec.write(65535, 16).unwrap();
        bitpack_vec.write(255, 8).unwrap();

        let mut bitpack = BitPack::<&[u8]>::new(bitpack_vec.as_slice());
        assert_eq!(bitpack.read(8).unwrap(), 255);
        assert_eq!(bitpack.read(16).unwrap(), 65535);
        assert_eq!(bitpack.read(8).unwrap(), 255);
    }

    #[test]
    fn test_bit_width_exceeded() {
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        let result = bitpack_vec.write(0, 33);
        assert!(matches!(result, Err(BuffError::BitWidthExceeded(33))));
    }
}
