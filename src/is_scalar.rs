/// Marker trait for types that act as scalars.
pub trait IsScalar {}

// Implement IsScalar for float types.
impl IsScalar for f32 {}
impl IsScalar for f64 {}

// Implement IsScalar for uom Quantity types.
#[cfg(feature = "uom")]
use uom::si::{Dimension, Quantity, Units};

#[cfg(feature = "uom")]
impl<D, U, V> IsScalar for Quantity<D, U, V>
where
    D: Dimension + ?Sized,
    U: Units<V> + ?Sized,
    V: num_traits::Num + uom::Conversion<V>,
{
}
