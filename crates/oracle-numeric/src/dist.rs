//! Discrete probability masses and the log-space special functions they are built from.
//!
//! Everything here is computed in log space and exponentiated at the end. That is not premature
//! caution: a Poisson mass written directly as `lambda^k * e^-lambda / k!` overflows `k!` long
//! before the mass itself underflows, and the log form costs the same handful of arithmetic
//! operations while staying stable across the whole range.

/// `ln(k!)`, as a summed log.
///
/// Exact to the last bit or two for the goal counts a football match produces (`0..=~10`). For a
/// *real* argument - the shape parameter of a negative binomial, say - use [`ln_gamma`], which
/// agrees with this at the integers.
pub fn ln_factorial(k: u32) -> f64 {
    (2..=k).map(|i| f64::from(i).ln()).sum()
}

/// Poisson probability mass `P(X = k)` for rate `lambda >= 0`.
///
/// A non-positive rate is treated as a point mass at zero, which is the right limit and keeps
/// callers from having to special-case a team with no expected goals.
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

/// `ln(x)` with `ln(0)` mapped to negative infinity, so a zero rate contributes a zero term (via
/// `exp`) to a log-space sum instead of a `NaN`.
pub fn safe_ln(x: f64) -> f64 {
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
    fn poisson_mass_sums_to_one() {
        let lambda = 1.7;
        let total: f64 = (0..40).map(|k| poisson_pmf(k, lambda)).sum();
        assert!((total - 1.0).abs() < 1e-9, "sum = {total}");
    }

    #[test]
    fn poisson_mean_matches_its_rate() {
        let lambda = 2.3;
        let mean: f64 = (0..60).map(|k| f64::from(k) * poisson_pmf(k, lambda)).sum();
        assert!((mean - lambda).abs() < 1e-6, "mean = {mean}");
    }

    #[test]
    fn a_non_positive_rate_is_a_point_mass_at_zero() {
        for lambda in [0.0, -1.0] {
            assert_eq!(poisson_pmf(0, lambda), 1.0);
            assert_eq!(poisson_pmf(1, lambda), 0.0);
        }
    }

    #[test]
    fn poisson_mass_survives_a_large_count() {
        // The naive lambda^k / k! form overflows here; the log form must not.
        let p = poisson_pmf(170, 1.5);
        assert!(p.is_finite() && p >= 0.0, "p = {p}");
    }

    #[test]
    fn ln_factorial_matches_a_direct_product() {
        for n in [0u32, 1, 2, 5, 9] {
            let direct: f64 = (1..=n).map(f64::from).product();
            assert!((ln_factorial(n) - direct.ln()).abs() < 1e-12, "n = {n}");
        }
    }

    #[test]
    fn ln_gamma_matches_factorial_at_the_integers() {
        for n in [0u32, 1, 4, 7, 10] {
            assert!((ln_gamma(f64::from(n) + 1.0) - ln_factorial(n)).abs() < 1e-7);
        }
    }

    #[test]
    fn ln_gamma_knows_the_half_integer() {
        // Gamma(1/2) = sqrt(pi), which the reflection branch has to get right.
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-10);
    }

    #[test]
    fn safe_ln_maps_zero_to_negative_infinity() {
        assert_eq!(safe_ln(0.0), f64::NEG_INFINITY);
        assert_eq!(safe_ln(-1.0), f64::NEG_INFINITY);
        assert_eq!(safe_ln(0.0).exp(), 0.0, "and exponentiates back to zero");
        assert!((safe_ln(std::f64::consts::E) - 1.0).abs() < 1e-12);
    }
}
