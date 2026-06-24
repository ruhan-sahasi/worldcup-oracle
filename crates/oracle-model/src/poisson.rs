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

/// Natural log of the gamma function (Lanczos approximation, g = 7), accurate to ~1e-10 over the
/// positive reals we use. `ln_gamma(n + 1) == ln(n!)`, but it accepts the *real* shape parameter
/// the negative binomial needs.
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_1,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula keeps the approximation accurate for small/negative arguments.
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let t = x + G + 0.5;
        let mut a = C[0];
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Negative-binomial probability mass `P(X = k)` for mean `mean ≥ 0` and dispersion `size`
/// (the NB "size"/`r`). The variance is `mean + mean²/size`, so a smaller `size` means more
/// overdispersion and `size → ∞` converges to `Poisson(mean)`; `size ≤ 0` is treated as Poisson.
/// This is the Gamma-Poisson mixture: the rate itself varies match to match, which gives the
/// fatter scoreline tails (more blowouts, more goalless draws) that real football shows.
pub fn neg_binomial_pmf(k: u32, mean: f64, size: f64) -> f64 {
    if size <= 0.0 {
        return poisson_pmf(k, mean);
    }
    if mean <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    let (kf, r) = (f64::from(k), size);
    let ln_coef = ln_gamma(kf + r) - ln_gamma(r) - ln_gamma(kf + 1.0);
    let ln_p = r * (r / (r + mean)).ln() + kf * (mean / (r + mean)).ln();
    (ln_coef + ln_p).exp()
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

    #[test]
    fn ln_gamma_matches_factorial() {
        // ln_gamma(n+1) == ln(n!).
        for n in [0u32, 1, 4, 7, 10] {
            assert!((ln_gamma(f64::from(n) + 1.0) - ln_factorial(n)).abs() < 1e-7);
        }
    }

    #[test]
    fn neg_binomial_sums_to_one_and_preserves_the_mean() {
        let (mean, size) = (1.6, 6.0);
        let total: f64 = (0..80).map(|k| neg_binomial_pmf(k, mean, size)).sum();
        assert!((total - 1.0).abs() < 1e-9, "sum = {total}");
        let m: f64 = (0..80)
            .map(|k| f64::from(k) * neg_binomial_pmf(k, mean, size))
            .sum();
        assert!((m - mean).abs() < 1e-6, "mean = {m}");
    }

    #[test]
    fn neg_binomial_is_overdispersed_and_converges_to_poisson() {
        let mean = 1.6;
        // Variance = mean + mean^2/size > mean for finite size.
        let var = |size: f64| -> f64 {
            (0..80)
                .map(|k| (f64::from(k) - mean).powi(2) * neg_binomial_pmf(k, mean, size))
                .sum()
        };
        assert!(var(4.0) > mean + 0.1, "finite size is overdispersed");
        assert!(var(4.0) > var(40.0), "smaller size = more overdispersion");
        // As size grows it approaches the Poisson PMF (variance -> mean).
        for k in 0..12 {
            assert!((neg_binomial_pmf(k, mean, 5000.0) - poisson_pmf(k, mean)).abs() < 1e-3);
        }
    }
}
