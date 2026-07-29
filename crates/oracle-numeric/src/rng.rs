//! The workspace's single pseudo-random generator.

/// The SplitMix64 increment: the odd 64-bit fraction of the golden ratio, which is what gives the
/// generator its equidistribution.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// A small, fast, fully seeded pseudo-random generator (SplitMix64).
///
/// Every Monte-Carlo path in the oracle - tournament draws, in-play match simulation, Golden Boot
/// races, HMC momentum, bootstrap resampling - is driven from one of these, because reproducibility
/// is a correctness property here: a forecast we cannot regenerate is a forecast we cannot debug.
/// Given a seed, the stream is exactly reproducible across platforms and across runs.
///
/// SplitMix64 is chosen for three reasons: it is a single `u64` of state (cheap to clone per
/// worker), it needs no warm-up or seeding ceremony (any seed, including zero, gives a good
/// stream), and it is short enough to read and verify in place. It is emphatically **not**
/// cryptographic - it is a simulation generator, and the state is recoverable from a few outputs.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator. The same seed always yields the same stream.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform draw in `[0, 1)`, from the top 53 bits (one full `f64` mantissa).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A uniform index in `0..n` (and `0` when `n == 0`), for picking an element out of a slice.
    ///
    /// This is plain modulo reduction, which is very slightly biased towards low indices. The bias
    /// is on the order of `n / 2^64` - for the collection sizes here (bootstrap resamples of a few
    /// thousand matches) it is some twenty orders of magnitude below sampling noise, so paying for
    /// rejection sampling would buy nothing.
    pub fn index_below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// A uniform draw in `[lo, hi)`. An empty or reversed interval yields `lo`.
    ///
    /// Consumes one uniform, so a caller's stream position does not depend on the bounds.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        if hi <= lo {
            return lo;
        }
        lo + self.unit() * (hi - lo)
    }

    /// A uniform integer in `[lo, hi]`, both ends included. `hi <= lo` yields `lo`.
    ///
    /// Inclusive because the ranges this replaces are inclusive by nature - a match minute is
    /// `1..=90`, and an exclusive bound would quietly drop the ninetieth. Carries the same
    /// negligible modulo bias as [`index_below`](Self::index_below); the arithmetic is done in
    /// `i128` so no combination of `i64` bounds can overflow.
    pub fn int_inclusive(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        let span = i128::from(hi) - i128::from(lo) + 1;
        let offset = (u128::from(self.next_u64()) % span as u128) as i128;
        (i128::from(lo) + offset) as i64
    }

    /// A standard-normal draw, by the Box-Muller transform.
    ///
    /// Box-Muller produces two independent normals per pair of uniforms; we return one and discard
    /// the other rather than cache it, so that the number of uniforms consumed per call is fixed
    /// and a stream stays reproducible regardless of how callers interleave their draws.
    pub fn normal(&mut self) -> f64 {
        // The first uniform is nudged off zero to avoid ln(0).
        let u1 = (self.unit() + 1e-12).min(1.0);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// A Poisson-distributed count with mean `lambda` (`0` for a non-positive `lambda`), by Knuth's
    /// multiplication method.
    ///
    /// Knuth's method draws one uniform per event, so its cost grows with `lambda`. Every rate in
    /// this codebase is a football goal rate - per-team totals near 1.5, per-player rates well
    /// below 1 - where that cost is a handful of multiplications and the method is exact.
    pub fn poisson(&mut self, lambda: f64) -> u32 {
        if lambda <= 0.0 {
            return 0;
        }
        let threshold = (-lambda).exp();
        let mut product = 1.0;
        let mut k = 0u32;
        loop {
            product *= self.unit();
            if product <= threshold {
                return k;
            }
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seed_reproduces_its_stream() {
        let (mut a, mut b) = (Rng::new(42), Rng::new(42));
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
        // Zero is a legal seed and must not degenerate into a run of zeros.
        assert_ne!(Rng::new(0).next_u64(), 0);
    }

    #[test]
    fn unit_draws_stay_in_the_half_open_interval() {
        let mut rng = Rng::new(7);
        for _ in 0..10_000 {
            let x = rng.unit();
            assert!((0.0..1.0).contains(&x), "unit() returned {x}");
        }
    }

    #[test]
    fn unit_draws_are_roughly_uniform() {
        // Ten buckets over 100k draws: each should hold ~10% of the mass.
        let mut rng = Rng::new(99);
        let mut buckets = [0usize; 10];
        const N: usize = 100_000;
        for _ in 0..N {
            buckets[(rng.unit() * 10.0) as usize] += 1;
        }
        for (i, count) in buckets.iter().enumerate() {
            let share = *count as f64 / N as f64;
            assert!((share - 0.1).abs() < 0.01, "bucket {i} held {share}");
        }
    }

    #[test]
    fn index_below_stays_in_range() {
        let mut rng = Rng::new(3);
        assert_eq!(rng.index_below(0), 0, "an empty range yields 0");
        assert_eq!(rng.index_below(1), 0);
        for _ in 0..1_000 {
            assert!(rng.index_below(7) < 7);
        }
    }

    #[test]
    fn range_draws_stay_within_their_bounds() {
        let mut rng = Rng::new(21);
        for _ in 0..10_000 {
            let x = rng.range(-2.5, 7.5);
            assert!((-2.5..7.5).contains(&x), "range() returned {x}");
        }
    }

    #[test]
    fn range_centres_on_the_midpoint() {
        let mut rng = Rng::new(22);
        const N: usize = 100_000;
        let mean = (0..N).map(|_| rng.range(-0.3, 0.3)).sum::<f64>() / N as f64;
        // SD of one draw is 0.6/sqrt(12) ~ 0.173, so the standard error is ~0.0005.
        assert!(mean.abs() < 0.005, "mean was {mean}");
    }

    #[test]
    fn a_degenerate_or_reversed_range_yields_its_lower_bound() {
        let mut rng = Rng::new(23);
        assert_eq!(rng.range(1.5, 1.5), 1.5);
        assert_eq!(rng.range(4.0, 2.0), 4.0, "reversed bounds do not wrap");
    }

    #[test]
    fn int_inclusive_covers_both_endpoints() {
        let mut rng = Rng::new(24);
        let mut seen_lo = false;
        let mut seen_hi = false;
        for _ in 0..2_000 {
            let x = rng.int_inclusive(1, 90);
            assert!((1..=90).contains(&x), "int_inclusive returned {x}");
            seen_lo |= x == 1;
            seen_hi |= x == 90;
        }
        assert!(seen_lo && seen_hi, "both ends must be reachable");
    }

    #[test]
    fn int_inclusive_handles_degenerate_and_extreme_bounds() {
        let mut rng = Rng::new(25);
        assert_eq!(rng.int_inclusive(7, 7), 7, "a single-value range");
        assert_eq!(rng.int_inclusive(9, 3), 9, "reversed bounds");
        // The widest possible span is exactly 2^64, so the modulo is a no-op and the result must be
        // the raw draw offset from i64::MIN. Checking that identity exercises the i128 span
        // arithmetic where an i64 or u64 computation would have overflowed.
        let (mut a, mut b) = (Rng::new(99), Rng::new(99));
        let want = (i128::from(i64::MIN) + i128::from(b.next_u64())) as i64;
        assert_eq!(a.int_inclusive(i64::MIN, i64::MAX), want);
    }

    #[test]
    fn int_inclusive_is_roughly_uniform_over_a_small_range() {
        let mut rng = Rng::new(26);
        let mut counts = [0usize; 6];
        const N: usize = 60_000;
        for _ in 0..N {
            counts[rng.int_inclusive(0, 5) as usize] += 1;
        }
        for (i, c) in counts.iter().enumerate() {
            let share = *c as f64 / N as f64;
            assert!((share - 1.0 / 6.0).abs() < 0.01, "face {i} held {share}");
        }
    }

    #[test]
    fn normal_draws_recover_their_moments() {
        let mut rng = Rng::new(11);
        const N: usize = 200_000;
        let xs: Vec<f64> = (0..N).map(|_| rng.normal()).collect();
        let mean = xs.iter().sum::<f64>() / N as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / N as f64;
        assert!(mean.abs() < 0.01, "mean was {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance was {var}");
    }

    #[test]
    fn poisson_draws_recover_their_mean() {
        let mut rng = Rng::new(13);
        const N: usize = 100_000;
        let lambda = 1.4;
        let total: u64 = (0..N).map(|_| u64::from(rng.poisson(lambda))).sum();
        let mean = total as f64 / N as f64;
        // A Poisson's variance equals its mean, so the standard error is sqrt(lambda / N) ~ 0.004.
        assert!((mean - lambda).abs() < 0.02, "mean was {mean}");
    }

    #[test]
    fn a_non_positive_rate_scores_nothing() {
        let mut rng = Rng::new(1);
        assert_eq!(rng.poisson(0.0), 0);
        assert_eq!(rng.poisson(-1.0), 0);
    }
}
