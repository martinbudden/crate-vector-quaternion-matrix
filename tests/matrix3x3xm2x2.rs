#[cfg(test)]
mod test_traits {
    use vqm::Matrix3x3xM2x2;
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<Matrix3x3xM2x2<f32>>();
    }
}

#[cfg(test)]
mod tests {
    use vqm::{Matrix2x2f32, Matrix3x3xM2x2f32};

    #[rustfmt::skip]
    #[test]
    fn new() {
        let matrix = Matrix3x3xM2x2f32::new([
             1.0,  2.0,    3.0,  4.0,    5.0,  6.0,
             7.0,  8.0,    9.0, 10.0,   11.0, 12.0,

            13.0, 14.0,   15.0, 16.0,   17.0, 18.0,
            19.0, 20.0,   21.0, 22.0,   23.0, 24.0,

            25.0, 26.0,   27.0, 28.0,   29.0, 30.0,
            31.0, 32.0,   33.0, 34.0,   35.0, 36.0,
        ]);

        assert_eq!(matrix[0], Matrix2x2f32::new([
            1.0, 2.0,
            7.0, 8.0]));
        assert_eq!(matrix[1], Matrix2x2f32::new([
            3.0, 4.0,
            9.0, 10.0]));
        assert_eq!(matrix[2], Matrix2x2f32::new([
            5.0, 6.0,
            11.0, 12.0]));
        assert_eq!(matrix[3], Matrix2x2f32::new([
            13.0, 14.0,
            19.0, 20.0]));
        assert_eq!(matrix[4], Matrix2x2f32::new([
            15.0, 16.0,
            21.0, 22.0]));
        assert_eq!(matrix[5], Matrix2x2f32::new([
            17.0, 18.0,
            23.0, 24.0]));
        assert_eq!(matrix[6], Matrix2x2f32::new([
            25.0, 26.0,
            31.0, 32.0]));
        assert_eq!(matrix[7], Matrix2x2f32::new([
            27.0, 28.0,
            33.0, 34.0]));
        assert_eq!(matrix[8], Matrix2x2f32::new([
            29.0, 30.0,
            35.0, 36.0]));
        }
}
