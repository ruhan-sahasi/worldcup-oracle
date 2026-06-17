//! Minimal Poisson machinery.
//!
//! We compute the PMF in log-space and exponentiate, which is numerically stable
//! for the small goal counts (0..=~10) a football match produces and avoids pulling
//! in a heavyweight stats dependency for what is three lines of arithmetic.

/// `ln(k!)` via a summed log (exact for the small `k` we deal with).
pub fn ln_factorial(k: u32) -> f64 {
    (2..=k).map(|i| f64::from(i).ln()).sum()
}

/// Poisson probability mass `P(X = k)` for rate `lambda ≥ 0`.
pub fn poisson_pmf(k: u32, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    (-lambda + f64::from(k) * lambda.ln() - ln_factorial(k)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pmf_sums_to_one() {
        let lambda = 1.7;
        let total: f64 = (0..40).map(|k| poisson_pmf(k, lambda)).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mean_matches_lambda() {
        let lambda = 2.3;
        let mean: f64 = (0..60).map(|k| f64::from(k) * poisson_pmf(k, lambda)).sum();
        assert!((mean - lambda).abs() < 1e-6);
    }
}
