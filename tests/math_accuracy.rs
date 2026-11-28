use vqm::{atan2_approx_f32, cos_approx_f32, ln_approx_f32, sin_approx_f32, sqrt_approx_f32};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sin_accuracy() {
        const SAMPLES: usize = 10_000;
        const MAX_X: f32 = std::f32::consts::FRAC_PI_2;
        const REL_EPSILON: f32 = 1.0e-6;

        let mut max_abs_error = 0.0_f32;
        let mut max_rel_error = 0.0_f32;
        let mut sum_sq_error = 0.0_f64;

        let mut max_abs_x = 0.0_f32;
        let mut max_rel_x = 0.0_f32;

        for i in 0..=SAMPLES {
            let x = MAX_X * (i as f32 / SAMPLES as f32);
            let reference = x.sin();
            let approx = sin_approx_f32(x);
            let abs_error = (approx - reference).abs();
            if abs_error > max_abs_error {
                max_abs_error = abs_error;
                max_abs_x = x;
            }
            sum_sq_error += (abs_error as f64).powi(2);
            if reference.abs() > REL_EPSILON {
                let rel_error = abs_error / reference.abs();
                if rel_error > max_rel_error {
                    max_rel_error = rel_error;
                    max_rel_x = x;
                }
            }
        }
        let rms_error = (sum_sq_error / (SAMPLES + 1) as f64).sqrt();
        println!("Maximum absolute error: {max_abs_error:e}");
        println!("  At x = {max_abs_x:.9}");
        println!("Maximum relative error: {max_rel_error:e}");
        println!("  At x = {max_rel_x:.9}");
        println!("RMS absolute error:      {rms_error:e}");
    }

    #[test]
    fn test_cos_accuracy() {
        const SAMPLES: usize = 10_000;
        const MAX_X: f32 = std::f32::consts::FRAC_PI_2;
        const REL_EPSILON: f32 = 1.0e-6;

        let mut max_abs_error = 0.0_f32;
        let mut max_rel_error = 0.0_f32;
        let mut sum_sq_error = 0.0_f64;

        let mut max_abs_x = 0.0_f32;
        let mut max_rel_x = 0.0_f32;

        for i in 0..=SAMPLES {
            let x = MAX_X * (i as f32 / SAMPLES as f32);
            let reference = x.cos();
            let approx = cos_approx_f32(x);
            let abs_error = (approx - reference).abs();
            if abs_error > max_abs_error {
                max_abs_error = abs_error;
                max_abs_x = x;
            }
            sum_sq_error += (abs_error as f64).powi(2);
            if reference.abs() > REL_EPSILON {
                let rel_error = abs_error / reference.abs();
                if rel_error > max_rel_error {
                    max_rel_error = rel_error;
                    max_rel_x = x;
                }
            }
        }
        let rms_error = (sum_sq_error / (SAMPLES + 1) as f64).sqrt();
        println!("Maximum absolute error: {max_abs_error:e}");
        println!("  At x = {max_abs_x:.9}");
        println!("Maximum relative error: {max_rel_error:e}");
        println!("  At x = {max_rel_x:.9}");
        println!("RMS absolute error:      {rms_error:e}");
    }

    #[test]
    fn test_atan2_accuracy() {
        const SAMPLES: usize = 500;
        const RANGE: f32 = 1.0;
        const REL_EPSILON: f32 = 1.0e-6;
        const EXCLUSION_SQUARE: f32 = 0.05;

        let mut max_abs_error = 0.0_f32;
        let mut max_rel_error = 0.0_f32;
        let mut sum_sq_error = 0.0_f64;

        let mut max_abs_x = 0.0_f32;
        let mut max_abs_y = 0.0_f32;
        let mut max_rel_x = 0.0_f32;
        let mut max_rel_y = 0.0_f32;

        let mut count = 0usize;

        for i in 0..=SAMPLES {
            let x = -RANGE + 2.0 * RANGE * (i as f32 / SAMPLES as f32);

            for j in 0..=SAMPLES {
                let y = -RANGE + 2.0 * RANGE * (j as f32 / SAMPLES as f32);
                if x.abs() < EXCLUSION_SQUARE && y.abs() < EXCLUSION_SQUARE {
                    continue;
                }

                // atan2 convention: atan2(y, x)
                let reference = y.atan2(x);
                let approx = atan2_approx_f32(y, x);

                let abs_error = (approx - reference).abs();

                if abs_error > max_abs_error {
                    max_abs_error = abs_error;
                    max_abs_x = x;
                    max_abs_y = y;
                }

                sum_sq_error += (abs_error as f64) * (abs_error as f64);
                count += 1;

                if reference.abs() > REL_EPSILON {
                    let rel_error = abs_error / reference.abs();

                    if rel_error > max_rel_error {
                        max_rel_error = rel_error;
                        max_rel_x = x;
                        max_rel_y = y;
                    }
                }
            }
        }

        let rms_error = (sum_sq_error / count as f64).sqrt();

        println!("Maximum absolute error: {max_abs_error:e}");
        let y_over_x = max_abs_y / max_abs_x;
        println!("  At x = {max_abs_x:.6}, y = {max_abs_y:.6}, y/x = {y_over_x:.6}");

        println!("Maximum relative error: {max_rel_error:e}");
        println!("  At x = {max_rel_x:.6}, y = {max_rel_y:.6}");

        println!("RMS absolute error:      {rms_error:e}");
    }

    #[test]
    fn test_ln_accuracy() {
        const SAMPLES: usize = 10_000;
        const MIN_EXP: f32 = -6.0;
        const MAX_EXP: f32 = 6.0;
        const REL_EPSILON: f32 = 1.0e-6;

        let mut max_abs_error = 0.0_f32;
        let mut max_rel_error = 0.0_f32;
        let mut sum_sq_error = 0.0_f64;

        let mut max_abs_x = 0.0_f32;
        let mut max_rel_x = 0.0_f32;

        for i in 0..=SAMPLES {
            // Logarithmically distributed inputs: 1e-6 to 1e6.
            let t = i as f32 / SAMPLES as f32;
            let exponent = MIN_EXP + t * (MAX_EXP - MIN_EXP);
            let x = 10.0_f32.powf(exponent);

            let reference = x.ln();
            let approx = ln_approx_f32(x);

            let abs_error = (approx - reference).abs();

            if abs_error > max_abs_error {
                max_abs_error = abs_error;
                max_abs_x = x;
            }

            sum_sq_error += (abs_error as f64).powi(2);

            if reference.abs() > REL_EPSILON {
                let rel_error = abs_error / reference.abs();

                if rel_error > max_rel_error {
                    max_rel_error = rel_error;
                    max_rel_x = x;
                }
            }
        }

        let rms_error = (sum_sq_error / (SAMPLES + 1) as f64).sqrt();

        println!("Maximum absolute error: {max_abs_error:e}");
        println!("  At x = {max_abs_x:e}");

        println!("Maximum relative error: {max_rel_error:e}");
        println!("  At x = {max_rel_x:e}");

        println!("RMS absolute error:      {rms_error:e}");
    }
    #[test]

    fn test_sqrt_accuracy() {
        const SAMPLES: usize = 10_000;
        const MIN_EXP: f32 = -6.0;
        const MAX_EXP: f32 = 6.0;

        let mut max_abs_error = 0.0_f32;
        let mut max_rel_error = 0.0_f32;
        let mut sum_sq_error = 0.0_f64;

        let mut max_abs_x = 0.0_f32;
        let mut max_rel_x = 0.0_f32;

        for i in 0..=SAMPLES {
            // Logarithmically distributed inputs: 1e-6 to 1e6.
            let t = i as f32 / SAMPLES as f32;
            let exponent = MIN_EXP + t * (MAX_EXP - MIN_EXP);
            let x = 10.0_f32.powf(exponent);

            let reference = x.sqrt();
            let approx = sqrt_approx_f32(x);

            let abs_error = (approx - reference).abs();

            if abs_error > max_abs_error {
                max_abs_error = abs_error;
                max_abs_x = x;
            }

            sum_sq_error += (abs_error as f64).powi(2);

            if reference.abs() > 1.0e-6 {
                let rel_error = abs_error / reference.abs();

                if rel_error > max_rel_error {
                    max_rel_error = rel_error;
                    max_rel_x = x;
                }
            }
        }

        let rms_error = (sum_sq_error / (SAMPLES + 1) as f64).sqrt();

        println!("Maximum absolute error: {max_abs_error:e}");
        println!("  At x = {max_abs_x:e}");

        println!("Maximum relative error: {max_rel_error:e}");
        println!("  At x = {max_rel_x:e}");

        println!("RMS absolute error:      {rms_error:e}");
    }
}
