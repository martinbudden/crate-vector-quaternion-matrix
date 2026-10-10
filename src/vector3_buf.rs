use core::{error::Error, fmt};

use crate::Vector3f32;

/// The error type returned when a slice is too short to parse to a `Vector3f32`.<br><br>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceTooShortError;

// Implement Display so it can be printed
impl fmt::Display for SliceTooShortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "source slice must contain at least 6 bytes")
    }
}

/// Implement the standard Error trait.
impl Error for SliceTooShortError {}

impl Vector3f32 {
    /// Creates a Vector3f32 from a 6-byte little-endian byte array.
    /// ```
    /// # use vqm::Vector3f32;
    /// let bytes = [0x01, 0x00, 0x00, 0x01, 0x2A, 0x00];
    /// let v = Vector3f32::from_le_bytes_6(bytes);
    /// assert_eq!(Vector3f32 { x: 1.0, y: 256.0, z: 42.0 }, v);
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_le_bytes_6(buf: [u8; 6]) -> Self {
        let x = i16::from_le_bytes([buf[0], buf[1]]);
        let y = i16::from_le_bytes([buf[2], buf[3]]);
        let z = i16::from_le_bytes([buf[4], buf[5]]);
        Self { x: x as f32, y: y as f32, z: z as f32 }
    }

    /// Creates a Vector3f32 from a little-endian byte slice.
    /// ```
    /// # use vqm::Vector3f32;
    /// let bytes = [0x01, 0x00, 0x00, 0x01, 0x2A, 0x00];
    /// let v = Vector3f32::from_le_slice_6(&bytes);
    /// assert_eq!(Vector3f32 { x: 1.0, y: 256.0, z: 42.0 }, v);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_le_slice_6(slice: &[u8]) -> Self {
        let chunk = &slice[0..6];
        let x = i16::from_le_bytes([chunk[0], chunk[1]]);
        let y = i16::from_le_bytes([chunk[2], chunk[3]]);
        let z = i16::from_le_bytes([chunk[4], chunk[5]]);
        Self { x: f32::from(x), y: f32::from(y), z: f32::from(z) }
    }

    /// Creates a `Vector3f32` from a little-endian byte slice.
    /// ```
    /// # use vqm::Vector3f32;
    /// let bytes = [0x01, 0x00, 0x00, 0x01, 0x2A, 0x00, 0x23, 0x45, 0x56, 0x78, 0x9a, 0xbc];
    /// let v = Vector3f32::try_from_le_slice_6(&bytes);
    /// assert_eq!(Ok(Vector3f32 { x: 1.0, y: 256.0, z: 42.0 }), v);
    /// ```
    /// # Errors
    /// Returns an [`SliceTooShortError`] if the provided `slice` contains fewer than 6 bytes.
    pub fn try_from_le_slice_6(slice: &[u8]) -> Result<Self, SliceTooShortError> {
        if slice.len() < 6 {
            return Err(SliceTooShortError);
        }
        let chunk = &slice[0..6];
        let x = i16::from_le_bytes([chunk[0], chunk[1]]);
        let y = i16::from_le_bytes([chunk[2], chunk[3]]);
        let z = i16::from_le_bytes([chunk[4], chunk[5]]);
        Ok(Self { x: f32::from(x), y: f32::from(y), z: f32::from(z) })
    }

    /// Creates a Vector3f32 from a 6-byte big-endian byte array.
    /// ```
    /// # use vqm::Vector3f32;
    /// let bytes = [0x00, 0x01, 0x01, 0x00, 0x00, 0x2A];
    /// let v = Vector3f32::from_be_bytes_6(bytes);
    /// assert_eq!(Vector3f32 { x: 1.0, y: 256.0, z: 42.0 }, v);
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_be_bytes_6(buf: [u8; 6]) -> Self {
        let x = i16::from_be_bytes([buf[0], buf[1]]);
        let y = i16::from_be_bytes([buf[2], buf[3]]);
        let z = i16::from_be_bytes([buf[4], buf[5]]);
        Self { x: x as f32, y: y as f32, z: z as f32 }
    }

    /// Creates a Vector3f32 from a big-endian byte slice.
    /// ```
    /// # use vqm::Vector3f32;
    /// let bytes = [0x00, 0x01, 0x01, 0x00, 0x00, 0x2A];
    /// let v = Vector3f32::from_be_slice_6(&bytes);
    /// assert_eq!(Vector3f32 { x: 1.0, y: 256.0, z: 42.0 }, v);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_be_slice_6(slice: &[u8]) -> Self {
        let chunk = &slice[0..6];
        let x = i16::from_be_bytes([chunk[0], chunk[1]]);
        let y = i16::from_be_bytes([chunk[2], chunk[3]]);
        let z = i16::from_be_bytes([chunk[4], chunk[5]]);
        Self { x: f32::from(x), y: f32::from(y), z: f32::from(z) }
    }

    /// Creates a `Vector3f32` from a big-endian byte slice.
    /// ```
    /// # use vqm::Vector3f32;
    /// let bytes = [0x00, 0x01, 0x01, 0x00, 0x00, 0x2A];
    /// let v = Vector3f32::try_from_be_slice_6(&bytes);
    /// assert_eq!(Ok(Vector3f32 { x: 1.0, y: 256.0, z: 42.0 }), v);
    /// ```
    /// # Errors
    /// Returns an [`SliceTooShortError`] if the provided `slice` contains fewer than 6 bytes.
    pub fn try_from_be_slice_6(slice: &[u8]) -> Result<Self, SliceTooShortError> {
        if slice.len() < 6 {
            return Err(SliceTooShortError);
        }
        let chunk = &slice[0..6];
        let x = i16::from_be_bytes([chunk[0], chunk[1]]);
        let y = i16::from_be_bytes([chunk[2], chunk[3]]);
        let z = i16::from_be_bytes([chunk[4], chunk[5]]);
        Ok(Self { x: f32::from(x), y: f32::from(y), z: f32::from(z) })
    }
}
