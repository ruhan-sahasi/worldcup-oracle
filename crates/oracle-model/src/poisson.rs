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

/// Bivariate Poisson PMF `P(X = x, Y = y)` for the shared-component model
/// `X = U1 + U3`, `Y = U2 + U3` with `Ui ~ Poisson(lambda_i)` independent. `lambda3` is the
/// covariance term: `lambda3 = 0` recovers two independent Poissons, and `lambda3 > 0`
/// induces positive correlation (more equal scorelines, i.e. more draws than independence).
///
/// `P(x, y) = e^{-(l1+l2+l3)} Σ_{k=0}^{min(x,y)} l1^{x-k}/(x-k)! · l2^{y-k}/(y-k)! · l3^k/k!`
pub fn bivariate_poisson_pmf(x: u32, y: u32, lambda1: f64, lambda2: f64, lambda3: f64) -> f64 {
    if lambda3 <= 0.0 {
        return poisson_pmf(x, lambda1) * poisson_pmf(y, lambda2);
    }
    let (l1, l2) = (lambda1.max(0.0), lambda2.max(0.0));
    let prefactor = (-(l1 + l2 + lambda3)).exp();
    let mut sum = 0.0;
    for k in 0..=x.min(y) {
        // ln of the k-th term, exponentiated, for numerical stability with small counts.
        let ln_term = f64::from(x - k) * safe_ln(l1) - ln_factorial(x - k)
            + f64::from(y - k) * safe_ln(l2)
            - ln_factorial(y - k)
            + f64::from(k) * lambda3.ln()
            - ln_factorial(k);
        sum += ln_term.exp();
    }
    prefactor * sum
}

/// `ln(x)` with `ln(0)` mapped to a large negative number, so a zero rate contributes a
/// zero term (via `exp`) instead of a `NaN`.
fn safe_ln(x: f64) -> f64 {
    if x <= 0.0 {
        f64::NEG_INFINITY
    } else {
        x.ln()
    }
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
    fn bivariate_reduces_to_independent_at_zero_covariance() {
        for (x, y) in [(0, 0), (1, 2), (3, 1)] {
            let bp = bivariate_poisson_pmf(x, y, 1.4, 1.1, 0.0);
            let indep = poisson_pmf(x, 1.4) * poisson_pmf(y, 1.1);
            assert!((bp - indep).abs() < 1e-12, "({x},{y}): {bp} vs {indep}");
        }
    }

    #[test]
    fn bivariate_pmf_sums_to_one() {
        let total: f64 = (0..40)
            .flat_map(|x| (0..40).map(move |y| (x, y)))
            .map(|(x, y)| bivariate_poisson_pmf(x, y, 1.2, 1.0, 0.4))
            .sum();
        assert!((total - 1.0).abs() < 1e-9, "sum = {total}");
    }

    #[test]
    fn positive_covariance_raises_draw_mass_at_fixed_means() {
        // Hold both marginal means at `mean` by setting l1 = l2 = mean - l3 (the way the
        // goal grid uses it), so only the correlation changes.
        let mean = 1.3;
        let draws = |l3: f64| -> f64 {
            (0..30)
                .map(|k| bivariate_poisson_pmf(k, k, mean - l3, mean - l3, l3))
                .sum::<f64>()
        };
        assert!(
            draws(0.4) > draws(0.0),
            "positive covariance at fixed means should add draw mass"
        );
    }

    #[test]
    fn mean_matches_lambda() {
        let lambda = 2.3;
        let mean: f64 = (0..60).map(|k| f64::from(k) * poisson_pmf(k, lambda)).sum();
        assert!((mean - lambda).abs() < 1e-6);
    }
}
