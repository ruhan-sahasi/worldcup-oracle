//! The Gaussian: its density, its distribution function, and the `erf` underneath both.
//!
//! Two very different parts of the oracle need these. The state-space rating filter turns a rating
//! gap and its uncertainty into outcome probabilities, and Bayesian optimisation needs an expected
//! improvement, which is an integral over the normal tail. Neither justifies a stats dependency for
//! what amounts to one polynomial.

/// Standard-normal probability density at `x`.
pub fn normal_pdf(x: f64) -> f64 {
    (-(0.5 * x * x)).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard-normal cumulative distribution `P(Z <= x)`, from [`erf`].
///
/// Inherits `erf`'s ~1.5e-7 absolute accuracy. Both callers feed this into a probability that is
/// then rounded for display or compared against another candidate, so seven digits is several more
/// than the decision needs.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// The error function, by the Abramowitz-Stegun 7.1.26 rational approximation (max absolute error
/// ~1.5e-7).
///
/// The approximation is stated for `x >= 0`; `erf` is odd, so a negative argument is reflected.
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stated accuracy of the Abramowitz-Stegun approximation.
    const TOL: f64 = 1.5e-7;

    #[test]
    fn erf_matches_known_values() {
        // Reference values from the standard tables.
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_8),
            (1.0, 0.842_700_792_9),
            (2.0, 0.995_322_265_0),
            (3.0, 0.999_977_909_5),
        ] {
            assert!((erf(x) - want).abs() < TOL, "erf({x}) = {}", erf(x));
        }
    }

    #[test]
    fn erf_is_odd_and_bounded() {
        for x in [-4.0, -1.5, -0.25, 0.25, 1.5, 4.0] {
            assert!((erf(x) + erf(-x)).abs() < 1e-12, "erf is odd at {x}");
            assert!(erf(x).abs() <= 1.0, "erf({x}) escaped [-1, 1]");
        }
    }

    #[test]
    fn normal_cdf_matches_known_quantiles() {
        for (x, want) in [
            (-1.96, 0.024_997_895_1),
            (-1.0, 0.158_655_253_9),
            (0.0, 0.5),
            (1.0, 0.841_344_746_1),
            (1.96, 0.975_002_104_9),
        ] {
            let got = normal_cdf(x);
            assert!((got - want).abs() < TOL, "normal_cdf({x}) = {got}");
        }
    }

    #[test]
    fn normal_cdf_is_monotone_and_symmetric() {
        let mut prev = 0.0;
        let mut x = -5.0;
        while x <= 5.0 {
            let p = normal_cdf(x);
            assert!(p >= prev, "normal_cdf dipped at {x}");
            assert!((p + normal_cdf(-x) - 1.0).abs() < TOL, "asymmetric at {x}");
            prev = p;
            x += 0.1;
        }
    }

    #[test]
    fn normal_pdf_integrates_to_one() {
        // Trapezoid over [-8, 8]: the tails beyond that hold less than 1e-15 of the mass.
        let (lo, hi, n) = (-8.0, 8.0, 16_000);
        let h = (hi - lo) / n as f64;
        let total: f64 = (0..=n)
            .map(|i| {
                let w = if i == 0 || i == n { 0.5 } else { 1.0 };
                w * normal_pdf(lo + i as f64 * h) * h
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-9, "integral = {total}");
    }

    #[test]
    fn normal_pdf_peaks_at_the_mean() {
        let peak = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((normal_pdf(0.0) - peak).abs() < 1e-12);
        assert!(normal_pdf(0.5) < normal_pdf(0.0));
        assert!((normal_pdf(1.3) - normal_pdf(-1.3)).abs() < 1e-15);
    }
}
