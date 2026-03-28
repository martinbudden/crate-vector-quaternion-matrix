use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{One, Signed, Zero, float::FloatCore};

use crate::MathMethods;

pub type Vector3di8 = Vector3d<i8>;
pub type Vector3di16 = Vector3d<i16>;
pub type Vector3di32 = Vector3d<i32>;
pub type Vector3df32 = Vector3d<f32>;
pub type Vector3df64 = Vector3d<f64>;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector3d<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

// **** Zero ****
/// Zero vector
/// ```
/// # use vector_quaternion_matrix::Vector3d;
/// # use num_traits::zero;
///
/// let z: Vector3d::<f32> = zero();
///
/// assert_eq!(z, Vector3d::<f32> { x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl<T> Zero for Vector3d<T>
where
    T: Zero + PartialEq,
{
    fn zero() -> Self {
        Self { x: T::zero(), y: T::zero(), z: T::zero() }
    }

    fn is_zero(&self) -> bool {
        self.x == T::zero() && self.y == T::zero() && self.z == T::zero()
    }
}

// **** Neg ****
/// Negate vector
/// ```
/// # use vector_quaternion_matrix::Vector3d;
/// let v = Vector3d::<f32> { x: 2.0, y: 3.0, z: 5.0 };
/// let r = -v;
///
/// assert_eq!(r, Vector3d::<f32> { x: -2.0, y: -3.0, z: -5.0 });
/// ```
impl<T> Neg for Vector3d<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

/// Negate vector reference
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::<f32> { x: 2.0, y: -3.0, z: 5.0 };
/// let r = -&v;
///
/// assert_eq!(r, Vector3d::<f32> { x: -2.0, y: 3.0, z: -5.0 });
/// assert_eq!(v, Vector3d::<f32> { x: 2.0, y: -3.0, z: 5.0 });
/// ```
impl<T> Neg for &Vector3d<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Vector3d<T>;
    fn neg(self) -> Self::Output {
        Vector3d { x: -self.x, y: -self.y, z: -self.z }
    }
}

// **** Add ****
/// Add two vectors
/// ```
/// # use vector_quaternion_matrix::Vector3d;
/// let u = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// let v = Vector3d::<f32>::new(7.0, 11.0, 13.0);
/// let r = u + v;
///
/// assert_eq!(r, Vector3d::<f32> { x: 9.0, y: 14.0, z: 18.0 });
/// ```
impl<T> Add for Vector3d<T>
where
    T: Add<Output = T>,
{
    type Output = Vector3d<T>;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

/// Add two vectors references
/// ```
/// # use vector_quaternion_matrix::Vector3d;
/// let u = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// let v = Vector3d::<f32>::new(7.0, 11.0, 13.0);
/// let r = &u + &v;
///
/// assert_eq!(r, Vector3d::<f32> { x: 9.0, y: 14.0, z: 18.0 });
/// ```
impl<T> Add for &Vector3d<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Vector3d<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Vector3d { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

// **** AddAssign ****
/// Add one vector to another
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let mut r = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// let u = Vector3d::<f32>::new(7.0, 11.0, 13.0);
/// r += u;
///
/// assert_eq!(r, Vector3d::<f32> { x: 9.0, y: 14.0, z: 18.0 });
///
/// # use num_traits::zero;
///
/// let z: Vector3d::<f32> = zero();
/// let r = u + z;
/// assert_eq!(r, u);
/// ```
impl<T> AddAssign for Vector3d<T>
where
    T: Copy + Add<Output = T>,
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** Sub ****
/// Subtract two vectors
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let u = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// let v = Vector3d::<f32>::new(7.0, 11.0, 13.0);
/// let r = u - v;
///
/// assert_eq!(r, Vector3d::<f32> { x: -5.0, y: -8.0, z: -8.0 });
/// ```
impl<T> Sub for Vector3d<T>
where
    T: Sub<Output = T>,
{
    type Output = Vector3d<T>;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}
/// Subtract two vectors references
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let u = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// let v = Vector3d::<f32>::new(7.0, 11.0, 13.0);
/// let r = &u - &v;
///
/// assert_eq!(r, Vector3d::<f32> { x: -5.0, y: -8.0, z: -8.0 });
/// ```
impl<T> Sub for &Vector3d<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Vector3d<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Vector3d { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

// **** SubAssign ****
/// Subtract one vector from another
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let mut r = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// let v = Vector3d::<f32>::new(7.0, 11.0, 17.0);
/// r -= v;
///
/// assert_eq!(r, Vector3d { x: -5.0, y: -8.0, z: -12.0 });
/// ```
impl<T> SubAssign for Vector3d<T>
where
    T: Copy + Sub<Output = T>,
{
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Pre-multiply ****
/// Pre-multiply vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// let r = 2.0 * v;
///
/// assert_eq!(r, Vector3d::<f32> { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl Mul<Vector3d<f32>> for f32 {
    type Output = Vector3d<f32>;
    fn mul(self, rhs: Vector3d<f32>) -> Vector3d<f32> {
        Vector3d { x: self * rhs.x, y: self * rhs.y, z: self * rhs.z }
    }
}

impl Mul<Vector3d<f64>> for f64 {
    type Output = Vector3d<f64>;
    fn mul(self, rhs: Vector3d<f64>) -> Vector3d<f64> {
        Vector3d { x: self * rhs.x, y: self * rhs.y, z: self * rhs.z }
    }
}

// **** Mul ****
/// Multiply vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// let r = v * 2.0;
///
/// assert_eq!(r, Vector3d { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl<T> Mul<T> for Vector3d<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, k: T) -> Self::Output {
        Self { x: self.x * k, y: self.y * k, z: self.z * k }
    }
}

impl Mul<f32> for Vector3d<i8> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i8, y: (self.y as f32 * k) as i8, z: (self.z as f32 * k) as i8 }
    }
}

impl Mul<f32> for Vector3d<i16> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i16, y: (self.y as f32 * k) as i16, z: (self.z as f32 * k) as i16 }
    }
}

impl Mul<f32> for Vector3d<i32> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i32, y: (self.y as f32 * k) as i32, z: (self.z as f32 * k) as i32 }
    }
}

// **** MulAssign ****
/// In-place multiply a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let mut v = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// v *= 2.0;
///
/// assert_eq!(v, Vector3d { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl<T> MulAssign<T> for Vector3d<T>
where
    T: Copy + Mul<Output = T>,
{
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

// **** Div ****
/// Divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// let r = v / 2.0;
///
/// assert_eq!(r, Vector3d { x: 1.0, y: 1.5, z: 2.5 });
/// ```
impl<T> Div<T> for Vector3d<T>
where
    T: Copy + One + Div<Output = T>,
{
    type Output = Self;
    fn div(self, k: T) -> Self {
        let r: T = T::one() / k;
        Self { x: self.x * r, y: self.y * r, z: self.z * r }
    }
}

// **** DivAssign ****
/// In-place divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let mut v = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// v /= 2.0;
///
/// assert_eq!(v, Vector3d { x: 1.0, y: 1.5, z: 2.5 });
/// ```
impl<T> DivAssign<T> for Vector3d<T>
where
    T: Copy + One + Div<Output = T>,
{
    fn div_assign(&mut self, k: T) {
        *self = *self / k;
    }
}

// **** Index ****
/// Access vector component by index
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::<f32>::new(2.0, 3.0, 5.0);
///
/// assert_eq!(v[0], 2.0);
/// assert_eq!(v[1], 3.0);
/// assert_eq!(v[2], 5.0);
/// ```
impl<T> Index<usize> for Vector3d<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => &self.z, // default to z component if index out of range
        }
    }
}

// **** IndexMut ****
// Set vector component by index
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let mut v = Vector3d::<f32>::new(2.0, 3.0, 5.0);
/// v[0] = 7.0;
/// v[1] = 11.0;
/// v[2] = 13.0;
///
/// assert_eq!(v, Vector3d { x:7.0, y:11.0, z:13.0 });
/// ```
impl<T> IndexMut<usize> for Vector3d<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => &mut self.z, // default to z component if index out of range
        }
    }
}

// **** impl new ****
impl<T> Vector3d<T>
where
    T: Copy,
{
    /// Create a vector
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

// **** impl abs ****
impl<T> Vector3d<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the vector with all components set to their absolute values
    pub fn abs(&self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() }
    }

    /// Set all components of the vector to their absolute values
    pub fn abs_in_place(&mut self) {
        *self = self.abs();
    }
}

// **** impl clamp ****
impl<T> Vector3d<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the vector with all components clamped to the specified range
    pub fn clamp(&self, min: T, max: T) -> Self {
        Self { x: self.x.clamp(min, max), y: self.y.clamp(min, max), z: self.z.clamp(min, max) }
    }

    /// Clamp all components of the vector to the specified range
    pub fn clamp_in_place(&mut self, min: T, max: T) {
        self.x = self.x.clamp(min, max);
        self.y = self.y.clamp(min, max);
        self.z = self.z.clamp(min, max);
    }
}

// **** impl squared_norm ****
impl<T> Vector3d<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Return square of Euclidean norm
    pub fn squared_norm(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Return distance between two points, squared
    pub fn distance_squared(&self, rhs: Self) -> T {
        (*self - rhs).squared_norm()
    }

    /// Vector dot product
    /// ```
    /// # use vector_quaternion_matrix::Vector3df32;
    ///
    /// let v = Vector3df32::new(2.0, 3.0, 5.0);
    /// let w = Vector3df32::new(7.0, 11.0, 13.0);
    ///
    /// let x = v.dot(w);
    ///
    /// assert_eq!(x, 112.0);
    /// ```
    pub fn dot(&self, rhs: Self) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Vector cross product
    /// ```
    /// # use vector_quaternion_matrix::Vector3df32;
    ///
    /// let v = Vector3df32::new(2.0, 3.0, 5.0);
    /// let w = Vector3df32::new(7.0, 11.0, 13.0);
    ///
    /// let x = v.cross(w);
    ///
    /// assert_eq!(x, Vector3df32 { x: 3.0 * 13.0 - 5.0 * 11.0, y: -2.0 * 13.0 + 5.0 * 7.0, z: 2.0 * 11.0 - 3.0 * 7.0 });
    /// ```
    pub fn cross(&self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    /// Return the sum of all components of the vector
    pub fn sum(&self) -> T {
        self.x + self.y + self.z
    }

    /// Return the product of all components of the vector
    pub fn product(&self) -> T {
        self.x * self.y * self.z
    }
}

// **** impl mean ****
impl<T> Vector3d<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the vector
    pub fn mean(&self) -> T {
        let three = T::one() + T::one() + T::one();
        (self.x + self.y + self.z) / three
    }
}

// **** impl norm ****
impl<T> Vector3d<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + MathMethods,
{
    /// Return Euclidean norm
    pub fn norm(&self) -> T {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl<T> Vector3d<T>
where
    T: Copy + Zero + One + PartialEq + Add<Output = T> + Sub<Output = T> + Div<Output = T> + MathMethods,
{
    /// Return normalized form of the vector
    pub fn normalized(&self) -> Self {
        let norm = self.norm();
        // If norm == 0.0 then the vector is already normalized
        if norm == T::zero() {
            return *self;
        }
        *self / norm
    }

    /// Normalize the vector in place
    pub fn normalize(&mut self) {
        let norm = self.norm();
        #[allow(clippy::assign_op_pattern)]
        // If norm == 0.0 then the vector is already normalized
        if norm != T::zero() {
            *self = *self / norm;
        }
    }
}

impl<T> Vector3d<T>
where
    T: Copy + Zero + One + Sub<Output = T> + MathMethods,
{
    // Return distance between two points
    pub fn distance(&self, rhs: Self) -> T {
        self.distance_squared(rhs).sqrt()
    }
}

// **** From ****
/// Vector from tuple
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::<f32>::from((2.0, 3.0, 5.0));
/// let w: Vector3d::<f32> = (7.0, 11.0, 13.0).into();
///
/// assert_eq!(v, Vector3d::<f32> { x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w, Vector3d::<f32> { x: 7.0, y: 11.0, z: 13.0 });
/// ```
impl<T> From<(T, T, T)> for Vector3d<T> {
    fn from((x, y, z): (T, T, T)) -> Self {
        Self { x, y, z }
    }
}

/// Vector from array
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::<f32>::from([2.0, 3.0, 5.0]);
/// let w: Vector3d::<f32> = [7.0, 11.0, 13.0].into();
///
/// assert_eq!(v, Vector3d::<f32> { x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w, Vector3d::<f32> { x: 7.0, y: 11.0, z: 13.0 });
/// ```
impl<T> From<[T; 3]> for Vector3d<T>
where
    T: Copy,
{
    fn from(v: [T; 3]) -> Self {
        Self { x: v[0], y: v[1], z: v[2] }
    }
}

/// Array from vector
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::<f32> { x: 2.0, y: 3.0, z: 5.0 };
///
/// let a = <[f32; 3]>::from(v);
/// let b: [f32; 3] = v.into();
///
/// assert_eq!(a, [2.0, 3.0, 5.0]);
/// assert_eq!(b, [2.0, 3.0, 5.0]);
/// ```
impl<T> From<Vector3d<T>> for [T; 3] {
    fn from(v: Vector3d<T>) -> Self {
        [v.x, v.y, v.z]
    }
}

/// `Vector3d<f32>` from `Vector3d<i16>`
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v_i16 = Vector3d::<i16>{x: 2, y: 3, z: 5};
/// let v_f32 = Vector3d::<f32>::from(v_i16);
///
/// let w_f32 = Vector3d::<f32>{x: 7.0, y: 11.0, z: 13.0};
/// let w_i16 : Vector3d::<i16> = w_f32.into();
///
/// let u_i32 = Vector3d::<i32>{x: 17, y: 19, z: 23};
/// let u_f32 : Vector3d::<f32> = u_i32.into();
///
/// assert_eq!(v_f32, Vector3d::<f32> { x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w_i16, Vector3d::<i16> { x: 7, y: 11, z: 13 });
/// assert_eq!(u_f32, Vector3d::<f32> { x: 17.0, y: 19.0, z: 23.0 });
/// ```
impl From<Vector3d<i16>> for Vector3d<f32> {
    fn from(v: Vector3d<i16>) -> Self {
        Self { x: v.x as f32, y: v.y as f32, z: v.z as f32 }
    }
}

impl From<Vector3d<f32>> for Vector3d<i16> {
    fn from(v: Vector3d<f32>) -> Self {
        Self { x: v.x as i16, y: v.y as i16, z: v.z as i16 }
    }
}
impl From<Vector3d<i32>> for Vector3d<f32> {
    fn from(v: Vector3d<i32>) -> Self {
        Self { x: v.x as f32, y: v.y as f32, z: v.z as f32 }
    }
}

impl From<Vector3d<f32>> for Vector3d<i32> {
    fn from(v: Vector3d<f32>) -> Self {
        Self { x: v.x as i32, y: v.y as i32, z: v.z as i32 }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<Vector3d<f32>>();
    }
    #[test]
    fn default() {
        let a: Vector3df32 = Vector3d::default();
        assert_eq!(a, Vector3d { x: 0.0, y: 0.0, z: 0.0 });
        let z: Vector3df32 = Vector3d::zero();
        //let z: Vector3d = zero();
        assert_eq!(a, z);
        assert!(z.is_zero());
    }
    #[test]
    fn test_neg_owned() {
        let v = Vector3d { x: 1.0, y: -2.0, z: 3.0 };
        let neg_v = -v;
        assert_eq!(neg_v.x, -1.0);
        assert_eq!(neg_v.y, 2.0);
        assert_eq!(neg_v.z, -3.0);
    }

    #[test]
    fn test_neg_borrowed() {
        let v = Vector3d { x: 1.0, y: -2.0, z: 3.0 };
        let neg_v = -&v; // Uses &Vector3d<T> impl
        assert_eq!(neg_v.x, -1.0);
        assert_eq!(neg_v.y, 2.0);
        assert_eq!(neg_v.z, -3.0);
        // v is still valid
        assert_eq!(v.x, 1.0);
    }
    #[test]
    fn neg() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(-a, Vector3d { x: -2.0, y: -3.0, z: -5.0 });

        let b = -a;
        assert_eq!(b, Vector3d { x: -2.0, y: -3.0, z: -5.0 });
    }
    #[test]
    fn add() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let b = Vector3d { x: 7.0, y: 11.0, z: 13.0 };
        assert_eq!(a + b, Vector3d { x: 9.0, y: 14.0, z: 18.0 });
    }
    #[test]
    fn add_assign() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let b = Vector3d { x: 7.0, y: 11.0, z: 13.0 };
        let mut c = a;
        c += b;
        assert_eq!(c, Vector3d { x: 9.0, y: 14.0, z: 18.0 });
    }
    #[test]
    fn sub() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let b = Vector3d { x: 7.0, y: 11.0, z: 17.0 };
        let c = a - b;
        assert_eq!(c, Vector3d { x: -5.0, y: -8.0, z: -12.0 });
    }
    #[test]
    fn sub_assign() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let b = Vector3d { x: 7.0, y: 11.0, z: 17.0 };
        let mut c = a;
        c -= b;
        assert_eq!(c, Vector3d { x: -5.0, y: -8.0, z: -12.0 });
    }
    #[test]
    fn mul() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a * 2.0, Vector3d { x: 4.0, y: 6.0, z: 10.0 });
        assert_eq!(2.0 * a, Vector3d { x: 4.0, y: 6.0, z: 10.0 });
    }
    #[test]
    fn mul_assign() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let mut b = a;
        b *= 2.0;
        assert_eq!(b, Vector3d { x: 4.0, y: 6.0, z: 10.0 });
    }
    #[test]
    fn div() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a / 2.0, Vector3d { x: 1.0, y: 1.5, z: 2.5 });
    }
    #[test]
    fn div_assign() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let mut b = a;
        b /= 2.0;
        assert_eq!(b, Vector3d { x: 1.0, y: 1.5, z: 2.5 });
    }
    #[test]
    fn new() {
        let a = Vector3d::new(2.0, 3.0, 5.0);
        assert_eq!(a, Vector3d { x: 2.0, y: 3.0, z: 5.0 });
        let b = Vector3d::from((2.0, 3.0, 5.0));
        assert_eq!(a, b);

        use num_traits::zero;
        let z: Vector3df32 = zero();
        assert!(z.is_zero());

        let c: Vector3df32 = (2.0, 3.0, 5.0).into();
        assert_eq!(a, c);
        let d = Vector3d::from((2.0, 3.0, 5.0));
        assert_eq!(a, d);
        let e: Vector3df32 = [2.0, 3.0, 5.0].into();
        assert_eq!(a, e);
        let f = Vector3df32::from([2.0, 3.0, 5.0]);
        assert_eq!(a, f);

        let h = <[f32; 3]>::from(a);
        assert_eq!([2.0, 3.0, 5.0], h);
        let i: [f32; 3] = a.into();
        assert_eq!([2.0, 3.0, 5.0], i);
    }
    #[test]
    fn dot() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let b = Vector3d { x: 7.0, y: 11.0, z: 13.0 };
        assert_eq!(a.dot(a), 38.0);
        assert_eq!(a.dot(b), 112.0);
        assert_eq!(b.dot(a), 112.0);
        assert_eq!(b.dot(b), 339.0);
    }
    #[test]
    fn squared_norm() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.squared_norm(), 38.0);
    }
    #[test]
    fn norm() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.norm(), 38.0_f32.sqrt());
        let z = Vector3d { x: 0.0, y: 0.0, z: 0.0 };
        assert_eq!(z.norm(), 0.0);
    }
    #[test]
    fn normalized() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        let b = a / 38.0_f32.sqrt();
        assert_eq!(a.normalized(), b);
        let z = Vector3d { x: 0.0, y: 0.0, z: 0.0 };
        assert_eq!(z.normalized(), z);
    }
    #[test]
    fn normalize() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let a_normalized = a.normalized();
        let mut b = a;
        b.normalize();
        assert_eq!(b, a_normalized);
        let z = Vector3df32 { x: 0.0, y: 0.0, z: 0.0 };
        let mut y = z;
        y.normalize();
        assert_eq!(z, y);
    }
    #[test]
    fn abs() {
        let a = Vector3df32 { x: -2.0, y: -3.0, z: -5.0 };
        assert_eq!(a.abs(), Vector3d { x: 2.0, y: 3.0, z: 5.0 });
    }
    #[test]
    fn abs_in_place() {
        let a = Vector3df32 { x: -2.0, y: -3.0, z: -5.0 };
        let mut b = a;
        b.abs_in_place();
        assert_eq!(b, a.abs());
    }
    #[test]
    fn clamp() {
        let a = Vector3d { x: -2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.clamp(-1.0, 4.0), Vector3d { x: -1.0, y: 3.0, z: 4.0 });
    }
    #[test]
    fn clamp_in_place() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        let mut b = a;
        b.clamp_in_place(-1.0, 4.0);
        assert_eq!(b, a.clamp(-1.0, 4.0));
    }
    #[test]
    fn sum() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.sum(), 10.0);
    }
    #[test]
    fn mean() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.mean(), 10.0 / 3.0);
    }
    #[test]
    fn product() {
        let a = Vector3d { x: 2.0, y: 3.0, z: 5.0 };
        assert_eq!(a.product(), 30.0);
    }
}
