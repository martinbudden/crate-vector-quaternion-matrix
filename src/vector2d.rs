use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{One, Signed, Zero, float::FloatCore};

use crate::MathMethods;

pub type Vector2di8 = Vector2d<i8>;
pub type Vector2di16 = Vector2d<i16>;
pub type Vector2di32 = Vector2d<i32>;
pub type Vector2df32 = Vector2d<f32>;
pub type Vector2df64 = Vector2d<f64>;

/// `Vector2d<T>`: 2D vector of type `T`.<br>
/// `Vector2d32` and `Vector2df64` and several integer aliases are provided.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector2d<T> {
    pub x: T,
    pub y: T,
}

// **** Zero ****
/// Zero vector
/// ```
/// # use vector_quaternion_matrix::Vector2d;
/// # use num_traits::zero;
///
/// let z: Vector2d::<f32> = zero();
///
/// assert_eq!(z, Vector2d::<f32> { x: 0.0, y: 0.0 });
/// ```
impl<T> Zero for Vector2d<T>
where
    T: Zero + PartialEq,
{
    fn zero() -> Self {
        Self { x: T::zero(), y: T::zero() }
    }

    fn is_zero(&self) -> bool {
        self.x == T::zero() && self.y == T::zero()
    }
}

// **** Neg ****
/// Negate vector
/// ```
/// # use vector_quaternion_matrix::Vector2d;
/// let v = Vector2d::<f32> { x: 2.0, y: 3.0 };
/// let r = -v;
///
/// assert_eq!(r, Vector2d::<f32> { x: -2.0, y: -3.0 });
/// ```
impl<T> Neg for Vector2d<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self { x: -self.x, y: -self.y }
    }
}

/// Negate vector reference
/// ```
/// # use vector_quaternion_matrix::Vector2d;
///
/// let v = Vector2d::<f32> { x: 2.0, y: -3.0 };
/// let r = -&v;
///
/// assert_eq!(r, Vector2d::<f32> { x: -2.0, y: 3.0 });
/// assert_eq!(v, Vector2d::<f32> { x: 2.0, y: -3.0 });
/// ```
impl<T> Neg for &Vector2d<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Vector2d<T>;
    fn neg(self) -> Self::Output {
        Vector2d { x: -self.x, y: -self.y }
    }
}

// **** Add ****
/// Add two vectors
/// ```
/// # use vector_quaternion_matrix::Vector2d;
/// let u = Vector2d::<f32>::new(2.0, 3.0);
/// let v = Vector2d::<f32>::new(7.0, 11.0);
/// let r = u + v;
///
/// assert_eq!(r, Vector2d::<f32> { x: 9.0, y: 14.0 });
/// ```
impl<T> Add for Vector2d<T>
where
    T: Add<Output = T>,
{
    type Output = Vector2d<T>;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}

/// Add two vectors references
/// ```
/// # use vector_quaternion_matrix::Vector2d;
/// let u = Vector2d::<f32>::new(2.0, 3.0);
/// let v = Vector2d::<f32>::new(7.0, 11.0);
/// let r = &u + &v;
///
/// assert_eq!(r, Vector2d::<f32> { x: 9.0, y: 14.0 });
/// ```
impl<T> Add for &Vector2d<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Vector2d<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Vector2d { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}

// **** AddAssign ****
/// Add one vector to another
/// ```
/// # use vector_quaternion_matrix::Vector2d;
///
/// let mut r = Vector2d::<f32>::new(2.0, 3.0);
/// let u = Vector2d::<f32>::new(7.0, 11.0);
/// r += u;
///
/// assert_eq!(r, Vector2d::<f32> { x: 9.0, y: 14.0 });
///
/// # use num_traits::zero;
///
/// let z: Vector2d::<f32> = zero();
/// let r = u + z;
/// assert_eq!(r, u);
/// ```
impl<T> AddAssign for Vector2d<T>
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
/// # use vector_quaternion_matrix::Vector2d;
///
/// let u = Vector2d::<f32>::new(2.0, 3.0);
/// let v = Vector2d::<f32>::new(7.0, 11.0);
/// let r = u - v;
///
/// assert_eq!(r, Vector2d::<f32> { x: -5.0, y: -8.0 });
/// ```
impl<T> Sub for Vector2d<T>
where
    T: Sub<Output = T>,
{
    type Output = Vector2d<T>;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}
/// Subtract two vectors references
/// ```
/// # use vector_quaternion_matrix::Vector2d;
///
/// let u = Vector2d::<f32>::new(2.0, 3.0);
/// let v = Vector2d::<f32>::new(7.0, 11.0);
/// let r = &u - &v;
///
/// assert_eq!(r, Vector2d::<f32> { x: -5.0, y: -8.0 });
/// ```
impl<T> Sub for &Vector2d<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Vector2d<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Vector2d { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

// **** SubAssign ****
/// Subtract one vector from another
/// ```
/// # use vector_quaternion_matrix::Vector2d;
///
/// let mut r = Vector2d::<f32>::new(2.0, 3.0);
/// let v = Vector2d::<f32>::new(7.0, 11.0);
/// r -= v;
///
/// assert_eq!(r, Vector2d { x: -5.0, y: -8.0 });
/// ```
impl<T> SubAssign for Vector2d<T>
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
/// # use vector_quaternion_matrix::Vector2d;
///
/// let v = Vector2d::<f32>::new(2.0, 3.0);
/// let r = 2.0 * v;
///
/// assert_eq!(r, Vector2d::<f32> { x: 4.0, y: 6.0 });
/// ```
impl Mul<Vector2d<f32>> for f32 {
    type Output = Vector2d<f32>;
    fn mul(self, rhs: Vector2d<f32>) -> Vector2d<f32> {
        Vector2d { x: self * rhs.x, y: self * rhs.y }
    }
}

impl Mul<Vector2d<f64>> for f64 {
    type Output = Vector2d<f64>;
    fn mul(self, rhs: Vector2d<f64>) -> Vector2d<f64> {
        Vector2d { x: self * rhs.x, y: self * rhs.y }
    }
}

// **** Mul ****
/// Multiply vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector2d;
///
/// let v = Vector2d::<f32>::new(2.0, 3.0);
/// let r = v * 2.0;
///
/// assert_eq!(r, Vector2d { x: 4.0, y: 6.0 });
/// ```
impl<T> Mul<T> for Vector2d<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, k: T) -> Self::Output {
        Self { x: self.x * k, y: self.y * k }
    }
}

impl Mul<f32> for Vector2d<i8> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i8, y: (self.y as f32 * k) as i8 }
    }
}

impl Mul<f32> for Vector2d<i16> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i16, y: (self.y as f32 * k) as i16 }
    }
}

impl Mul<f32> for Vector2d<i32> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i32, y: (self.y as f32 * k) as i32 }
    }
}

// **** MulAssign ****
/// In-place multiply a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector2d;
///
/// let mut v = Vector2d::<f32>::new(2.0, 3.0);
/// v *= 2.0;
///
/// assert_eq!(v, Vector2d { x: 4.0, y: 6.0 });
/// ```
impl<T> MulAssign<T> for Vector2d<T>
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
/// # use vector_quaternion_matrix::Vector2d;
///
/// let v = Vector2d::<f32>::new(2.0, 3.0);
/// let r = v / 2.0;
///
/// assert_eq!(r, Vector2d { x: 1.0, y: 1.5 });
/// ```
impl<T> Div<T> for Vector2d<T>
where
    T: Copy + One + Div<Output = T>,
{
    type Output = Self;
    fn div(self, k: T) -> Self {
        let r: T = T::one() / k;
        Self { x: self.x * r, y: self.y * r }
    }
}

// **** DivAssign ****
/// In-place divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector2d;
///
/// let mut v = Vector2d::<f32>::new(2.0, 3.0);
/// v /= 2.0;
///
/// assert_eq!(v, Vector2d { x: 1.0, y: 1.5 });
/// ```
impl<T> DivAssign<T> for Vector2d<T>
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
/// # use vector_quaternion_matrix::Vector2d;
///
/// let v = Vector2d::<f32>::new(2.0, 3.0);
///
/// assert_eq!(v[0], 2.0);
/// assert_eq!(v[1], 3.0);
/// ```
impl<T> Index<usize> for Vector2d<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => &self.y, // default to z component if index out of range
        }
    }
}

// **** IndexMut ****
// Set vector component by index
/// ```
/// # use vector_quaternion_matrix::Vector2d;
///
/// let mut v = Vector2d::<f32>::new(2.0, 3.0);
/// v[0] = 7.0;
/// v[1] = 11.0;
///
/// assert_eq!(v, Vector2d { x:7.0, y:11.0 });
/// ```
impl<T> IndexMut<usize> for Vector2d<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.y, // default to z component if index out of range
        }
    }
}

// **** impl new ****
impl<T> Vector2d<T>
where
    T: Copy,
{
    /// Create a vector
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

// **** impl abs ****
impl<T> Vector2d<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the vector with all components set to their absolute values
    pub fn abs(&self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs() }
    }

    /// Set all components of the vector to their absolute values
    pub fn abs_in_place(&mut self) {
        *self = self.abs();
    }
}

// **** impl clamp ****
impl<T> Vector2d<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the vector with all components clamped to the specified range
    pub fn clamp(&self, min: T, max: T) -> Self {
        Self { x: self.x.clamp(min, max), y: self.y.clamp(min, max) }
    }

    /// Clamp all components of the vector to the specified range
    pub fn clamp_in_place(&mut self, min: T, max: T) {
        self.x = self.x.clamp(min, max);
        self.y = self.y.clamp(min, max);
    }
}

// **** impl squared_norm ****
impl<T> Vector2d<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Return square of Euclidean norm
    pub fn squared_norm(&self) -> T {
        self.x * self.x + self.y * self.y
    }

    /// Return distance between two points, squared
    pub fn distance_squared(&self, rhs: Self) -> T {
        (*self - rhs).squared_norm()
    }

    /// Vector dot product
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    ///
    /// let v = Vector2df32::new(2.0, 3.0);
    /// let w = Vector2df32::new(7.0, 11.0);
    ///
    /// let x = v.dot(w);
    ///
    /// assert_eq!(x, 47.0);
    /// ```
    pub fn dot(&self, rhs: Self) -> T {
        self.x * rhs.x + self.y * rhs.y
    }

    /// Z component of vector cross product of self and rhs extended to 3D
    /// ```
    /// # use vector_quaternion_matrix::Vector2df32;
    ///
    /// let v = Vector2df32::new(2.0, 3.0);
    /// let w = Vector2df32::new(7.0, 11.0);
    ///
    /// let x = v.cross(w);
    ///
    /// assert_eq!(x, 1.0);
    /// ```
    pub fn cross(&self, rhs: Self) -> T {
        self.x * rhs.y - self.y * rhs.x
    }

    /// Return the sum of all components of the vector
    pub fn sum(&self) -> T {
        self.x + self.y
    }

    /// Return the product of all components of the vector
    pub fn product(&self) -> T {
        self.x * self.y
    }
}

// **** impl mean ****
impl<T> Vector2d<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the vector
    pub fn mean(&self) -> T {
        (self.x + self.y) / (T::one() + T::one())
    }
}

// **** impl norm ****
impl<T> Vector2d<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + MathMethods,
{
    /// Return Euclidean norm
    pub fn norm(&self) -> T {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

impl<T> Vector2d<T>
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

impl<T> Vector2d<T>
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
/// # use vector_quaternion_matrix::Vector2d;
///
/// let v = Vector2d::<f32>::from((2.0, 3.0));
/// let w: Vector2d::<f32> = (7.0, 11.0).into();
///
/// assert_eq!(v, Vector2d::<f32> { x: 2.0, y: 3.0 });
/// assert_eq!(w, Vector2d::<f32> { x: 7.0, y: 11.0 });
/// ```
impl<T> From<(T, T)> for Vector2d<T> {
    fn from((x, y): (T, T)) -> Self {
        Self { x, y }
    }
}

/// Vector from array
/// ```
/// # use vector_quaternion_matrix::Vector2d;
///
/// let v = Vector2d::<f32>::from([2.0, 3.0]);
/// let w: Vector2d::<f32> = [7.0, 11.0].into();
///
/// assert_eq!(v, Vector2d::<f32> { x: 2.0, y: 3.0 });
/// assert_eq!(w, Vector2d::<f32> { x: 7.0, y: 11.0 });
/// ```
impl<T> From<[T; 2]> for Vector2d<T>
where
    T: Copy,
{
    fn from(v: [T; 2]) -> Self {
        Self { x: v[0], y: v[1] }
    }
}

/// Array from vector
/// ```
/// # use vector_quaternion_matrix::Vector2d;
///
/// let v = Vector2d::<f32> { x: 2.0, y: 3.0 };
///
/// let a = <[f32; 2]>::from(v);
/// let b: [f32; 2] = v.into();
///
/// assert_eq!(a, [2.0, 3.0]);
/// assert_eq!(b, [2.0, 3.0]);
/// ```
impl<T> From<Vector2d<T>> for [T; 2] {
    fn from(v: Vector2d<T>) -> Self {
        [v.x, v.y]
    }
}

/// `Vector2d<f32>` from `Vector2d<i16>`
/// ```
/// # use vector_quaternion_matrix::Vector2d;
///
/// let v_i16 = Vector2d::<i16>{x: 2, y: 3};
/// let v_f32 = Vector2d::<f32>::from(v_i16);
///
/// let w_f32 = Vector2d::<f32>{x: 7.0, y: 11.0};
/// let w_i16 : Vector2d::<i16> = w_f32.into();
///
/// let u_i32 = Vector2d::<i32>{x: 17, y: 19};
/// let u_f32 : Vector2d::<f32> = u_i32.into();
///
/// assert_eq!(v_f32, Vector2d::<f32> { x: 2.0, y: 3.0 });
/// assert_eq!(w_i16, Vector2d::<i16> { x: 7, y: 11 });
/// assert_eq!(u_f32, Vector2d::<f32> { x: 17.0, y: 19.0 });
/// ```
impl From<Vector2d<i16>> for Vector2d<f32> {
    fn from(v: Vector2d<i16>) -> Self {
        Self { x: v.x as f32, y: v.y as f32 }
    }
}

impl From<Vector2d<f32>> for Vector2d<i16> {
    fn from(v: Vector2d<f32>) -> Self {
        Self { x: v.x as i16, y: v.y as i16 }
    }
}
impl From<Vector2d<i32>> for Vector2d<f32> {
    fn from(v: Vector2d<i32>) -> Self {
        Self { x: v.x as f32, y: v.y as f32 }
    }
}

impl From<Vector2d<f32>> for Vector2d<i32> {
    fn from(v: Vector2d<f32>) -> Self {
        Self { x: v.x as i32, y: v.y as i32 }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<Vector2d<f32>>();
    }
    #[test]
    fn default() {
        let a: Vector2df32 = Vector2d::default();
        assert_eq!(a, Vector2d { x: 0.0, y: 0.0 });
        let z: Vector2df32 = Vector2d::zero();
        //let z: Vector2d = zero();
        assert_eq!(a, z);
        assert!(z.is_zero());
    }
    #[test]
    fn test_neg_owned() {
        let v = Vector2d { x: 1.0, y: -2.0 };
        let neg_v = -v;
        assert_eq!(neg_v.x, -1.0);
        assert_eq!(neg_v.y, 2.0);
    }

    #[test]
    fn test_neg_borrowed() {
        let v = Vector2d { x: 1.0, y: -2.0 };
        let neg_v = -&v; // Uses &Vector2d<T> impl
        assert_eq!(neg_v.x, -1.0);
        assert_eq!(neg_v.y, 2.0);
        // v is still valid
        assert_eq!(v.x, 1.0);
    }
    #[test]
    fn neg() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        assert_eq!(-a, Vector2d { x: -2.0, y: -3.0 });

        let b = -a;
        assert_eq!(b, Vector2d { x: -2.0, y: -3.0 });
    }
    #[test]
    fn add() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let b = Vector2d { x: 7.0, y: 11.0 };
        assert_eq!(a + b, Vector2d { x: 9.0, y: 14.0 });
    }
    #[test]
    fn add_assign() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let b = Vector2d { x: 7.0, y: 11.0 };
        let mut c = a;
        c += b;
        assert_eq!(c, Vector2d { x: 9.0, y: 14.0 });
    }
    #[test]
    fn sub() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let b = Vector2d { x: 7.0, y: 11.0 };
        let c = a - b;
        assert_eq!(c, Vector2d { x: -5.0, y: -8.0 });
    }
    #[test]
    fn sub_assign() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let b = Vector2d { x: 7.0, y: 11.0 };
        let mut c = a;
        c -= b;
        assert_eq!(c, Vector2d { x: -5.0, y: -8.0 });
    }
    #[test]
    fn mul() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        assert_eq!(a * 2.0, Vector2d { x: 4.0, y: 6.0 });
        assert_eq!(2.0 * a, Vector2d { x: 4.0, y: 6.0 });
    }
    #[test]
    fn mul_assign() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let mut b = a;
        b *= 2.0;
        assert_eq!(b, Vector2d { x: 4.0, y: 6.0 });
    }
    #[test]
    fn div() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        assert_eq!(a / 2.0, Vector2d { x: 1.0, y: 1.5 });
    }
    #[test]
    fn div_assign() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let mut b = a;
        b /= 2.0;
        assert_eq!(b, Vector2d { x: 1.0, y: 1.5 });
    }
    #[test]
    fn new() {
        let a = Vector2d::new(2.0, 3.0);
        assert_eq!(a, Vector2d { x: 2.0, y: 3.0 });
        let b = Vector2d::from((2.0, 3.0));
        assert_eq!(a, b);

        use num_traits::zero;
        let z: Vector2df32 = zero();
        assert!(z.is_zero());

        let c: Vector2df32 = (2.0, 3.0).into();
        assert_eq!(a, c);
        let d = Vector2d::from((2.0, 3.0));
        assert_eq!(a, d);
        let e: Vector2df32 = [2.0, 3.0].into();
        assert_eq!(a, e);
        let f = Vector2df32::from([2.0, 3.0]);
        assert_eq!(a, f);

        let h = <[f32; 2]>::from(a);
        assert_eq!([2.0, 3.0], h);
        let i: [f32; 2] = a.into();
        assert_eq!([2.0, 3.0], i);
    }
    #[test]
    fn dot() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let b = Vector2d { x: 7.0, y: 11.0 };
        assert_eq!(a.dot(a), 13.0);
        assert_eq!(a.dot(b), 47.0);
        assert_eq!(b.dot(a), 47.0);
        assert_eq!(b.dot(b), 170.0);
    }
    #[test]
    fn squared_norm() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        assert_eq!(a.squared_norm(), 13.0);
    }
    #[test]
    fn norm() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        assert_eq!(a.norm(), 13.0_f32.sqrt());
        let z = Vector2d { x: 0.0, y: 0.0 };
        assert_eq!(z.norm(), 0.0);
    }
    #[test]
    fn normalized() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        let b = a / 13.0_f32.sqrt();
        assert_eq!(a.normalized(), b);
        let z = Vector2d { x: 0.0, y: 0.0 };
        assert_eq!(z.normalized(), z);
    }
    #[test]
    fn normalize() {
        let a = Vector2df32 { x: 2.0, y: 3.0 };
        let a_normalized = a.normalized();
        let mut b = a;
        b.normalize();
        assert_eq!(b, a_normalized);
        let z = Vector2df32 { x: 0.0, y: 0.0 };
        let mut y = z;
        y.normalize();
        assert_eq!(z, y);
    }
    #[test]
    fn abs() {
        let a = Vector2df32 { x: -2.0, y: -3.0 };
        assert_eq!(a.abs(), Vector2d { x: 2.0, y: 3.0 });
    }
    #[test]
    fn abs_in_place() {
        let a = Vector2df32 { x: -2.0, y: -3.0 };
        let mut b = a;
        b.abs_in_place();
        assert_eq!(b, a.abs());
    }
    #[test]
    fn clamp() {
        let a = Vector2d { x: -2.0, y: 3.0 };
        assert_eq!(a.clamp(-1.0, 4.0), Vector2d { x: -1.0, y: 3.0 });
    }
    #[test]
    fn clamp_in_place() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        let mut b = a;
        b.clamp_in_place(-1.0, 4.0);
        assert_eq!(b, a.clamp(-1.0, 4.0));
    }
    #[test]
    fn sum() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        assert_eq!(a.sum(), 5.0);
    }
    #[test]
    fn mean() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        assert_eq!(a.mean(), 5.0 / 2.0);
    }
    #[test]
    fn product() {
        let a = Vector2d { x: 2.0, y: 3.0 };
        assert_eq!(a.product(), 6.0);
    }
}
