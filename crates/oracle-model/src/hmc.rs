//! Hamiltonian Monte Carlo: sampling the *full* posterior, not just a Gaussian approximation.
//!
//! The Laplace approximation ([`crate::dixon_coles::GoalModel::strength_uncertainty`]) summarizes
//! the posterior by a Gaussian at the mode. HMC instead draws genuine samples from the posterior
//! by simulating Hamiltonian dynamics: it augments the parameters `θ` with a momentum `p`, rolls
//! the pair forward with the **leapfrog** integrator using the log-posterior gradient, and accepts
//! the proposal with a Metropolis step that corrects for the integrator's discretization error.
//! Because it follows the gradient it explores high-dimensional posteriors far more efficiently
//! than a random walk, and the samples capture the true (possibly skewed, correlated) shape.
//!
//! This is a compact, dependency-free implementation. A **diagonal mass matrix** preconditions
//! the dynamics (set it to the Laplace posterior variances and the target becomes roughly
//! isotropic, so a single step size mixes well across parameters of very different scale). Momentum
//! draws come from a seeded SplitMix64 + Box-Muller, so a run is fully reproducible.

/// Tuning for an HMC run.
#[derive(Debug, Clone, Copy)]
pub struct HmcConfig {
    /// Kept samples (after warmup).
    pub n_samples: usize,
    /// Warmup (burn-in) iterations, discarded.
    pub n_warmup: usize,
    /// Leapfrog step size `ε`.
    pub step_size: f64,
    /// Leapfrog steps per proposal `L`.
    pub n_leapfrog: usize,
    pub seed: u64,
}

/// The draws plus the post-warmup acceptance rate (a health check: well-tuned HMC sits ~0.6-0.9).
#[derive(Debug, Clone)]
pub struct HmcResult {
    pub samples: Vec<Vec<f64>>,
    pub accept_rate: f64,
}

/// Sample a target density known up to a constant, given its **log-density and gradient**
/// `grad_log_post(θ) -> (log p(θ), ∇ log p(θ))`. `inv_mass` is the diagonal of the inverse mass
/// matrix `M⁻¹` (a per-coordinate preconditioner; momentum is drawn from `N(0, M)`); set it to the
/// posterior variances for good mixing. Sampling starts from `init` (ideally the MAP).
pub fn sample<F: FnMut(&[f64]) -> (f64, Vec<f64>)>(
    init: Vec<f64>,
    inv_mass: &[f64],
    cfg: HmcConfig,
    mut grad_log_post: F,
) -> HmcResult {
    let dim = init.len();
    let mut rng = SplitMix::new(cfg.seed);
    let mut theta = init;
    let (mut logp, mut grad) = grad_log_post(&theta);

    let total = cfg.n_warmup + cfg.n_samples;
    let mut samples = Vec::with_capacity(cfg.n_samples);
    let mut accepts = 0usize;
    let eps = cfg.step_size;

    let kinetic =
        |p: &[f64]| -> f64 { 0.5 * (0..dim).map(|i| p[i] * p[i] * inv_mass[i]).sum::<f64>() };

    // Jitter the trajectory length each iteration to avoid resonances: at a fixed length a
    // (near-)harmonic posterior can loop back near its start and bias the sampled variance.
    let l_lo = (cfg.n_leapfrog / 2).max(1);
    let l_hi = cfg.n_leapfrog.max(l_lo);

    for it in 0..total {
        // Sample momentum p ~ N(0, M): Var(p_i) = M_i = 1 / inv_mass_i.
        let p0: Vec<f64> = (0..dim)
            .map(|i| rng.normal() / inv_mass[i].sqrt())
            .collect();
        let h0 = -logp + kinetic(&p0);
        let n_steps = l_lo + (rng.next_u64() as usize % (l_hi - l_lo + 1));

        // Leapfrog integration of Hamilton's equations.
        let mut th = theta.clone();
        let mut p = p0;
        let mut g = grad.clone();
        let mut lp_end = logp;
        // Initial half step on momentum.
        for i in 0..dim {
            p[i] += 0.5 * eps * g[i];
        }
        for step in 0..n_steps {
            for i in 0..dim {
                th[i] += eps * inv_mass[i] * p[i];
            }
            let (lp, gg) = grad_log_post(&th);
            lp_end = lp;
            g = gg;
            if step + 1 < n_steps {
                for i in 0..dim {
                    p[i] += eps * g[i];
                }
            }
        }
        // Final half step on momentum (position is fixed, so log p is `lp_end`).
        for i in 0..dim {
            p[i] += 0.5 * eps * g[i];
        }

        let hn = -lp_end + kinetic(&p);
        // Metropolis acceptance with the discretization-error correction.
        let accept = hn.is_finite() && rng.unit() < (h0 - hn).exp();
        if accept {
            theta = th;
            logp = lp_end;
            grad = g;
        }
        if it >= cfg.n_warmup {
            if accept {
                accepts += 1;
            }
            samples.push(theta.clone());
        }
    }

    HmcResult {
        samples,
        accept_rate: accepts as f64 / cfg.n_samples.max(1) as f64,
    }
}

/// A tiny seeded SplitMix64 generator with standard-normal draws (Box-Muller).
struct SplitMix(u64);
impl SplitMix {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        // Box-Muller; the uniforms are nudged off 0 to avoid ln(0).
        let u1 = (self.unit() + 1e-12).min(1.0);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
    fn variance(xs: &[f64]) -> f64 {
        let m = mean(xs);
        xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
    }

    #[test]
    fn recovers_a_known_gaussian() {
        // Target: independent N(mu_i, sigma_i^2). log p = -0.5 Σ (θ-μ)²/σ², grad = -(θ-μ)/σ².
        let mu = [1.0, -2.0];
        let sigma2 = [0.25, 4.0];
        let inv_mass = sigma2; // precondition by the posterior variances -> isotropic, mixes well.
        let res = sample(
            vec![0.0, 0.0],
            &inv_mass,
            HmcConfig {
                // In the preconditioned space the target is standard normal (oscillator period
                // 2π); a trajectory length L·ε ≈ π decorrelates well (≈ 2π would return to start).
                n_samples: 6000,
                n_warmup: 1000,
                step_size: 0.39,
                n_leapfrog: 8,
                seed: 1,
            },
            |theta| {
                let logp = -0.5
                    * (0..2)
                        .map(|i| (theta[i] - mu[i]).powi(2) / sigma2[i])
                        .sum::<f64>();
                let grad: Vec<f64> = (0..2).map(|i| -(theta[i] - mu[i]) / sigma2[i]).collect();
                (logp, grad)
            },
        );
        assert!(
            res.accept_rate > 0.6,
            "acceptance should be healthy, got {}",
            res.accept_rate
        );
        let d0: Vec<f64> = res.samples.iter().map(|s| s[0]).collect();
        let d1: Vec<f64> = res.samples.iter().map(|s| s[1]).collect();
        assert!((mean(&d0) - mu[0]).abs() < 0.1, "mean0 = {}", mean(&d0));
        assert!((mean(&d1) - mu[1]).abs() < 0.2, "mean1 = {}", mean(&d1));
        assert!(
            (variance(&d0) - sigma2[0]).abs() < 0.08,
            "var0 = {}",
            variance(&d0)
        );
        assert!(
            (variance(&d1) - sigma2[1]).abs() < 0.6,
            "var1 = {}",
            variance(&d1)
        );
    }

    #[test]
    fn is_reproducible_for_a_fixed_seed() {
        let target = |theta: &[f64]| (-0.5 * theta[0] * theta[0], vec![-theta[0]]);
        let cfg = HmcConfig {
            n_samples: 200,
            n_warmup: 50,
            step_size: 0.5,
            n_leapfrog: 5,
            seed: 7,
        };
        let a = sample(vec![0.0], &[1.0], cfg, target);
        let b = sample(vec![0.0], &[1.0], cfg, target);
        assert_eq!(a.samples, b.samples);
    }
}
