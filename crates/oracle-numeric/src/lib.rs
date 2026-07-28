//! # oracle-numeric
//!
//! The numerical floor of **worldcup-oracle**: one seeded pseudo-random generator and the handful
//! of statistical special functions the models are built on. It depends on nothing at all - not
//! even `serde` - and performs no I/O.
//!
//! ## Why this crate exists
//!
//! The prediction crates each need the same small pieces of mathematics: a Poisson mass function, a
//! normal CDF, a reproducible stream of random numbers. Those pieces are individually short enough
//! that writing them inline is tempting, and the workspace duly grew five copies of SplitMix64,
//! three of the Poisson mass, and two of the `erf` approximation. Copies of numerics are worse than
//! copies of ordinary code: each one is a place where a tolerance can drift out of step with the
//! test that guards it, and a bug fixed in one is silently still live in the other four.
//!
//! So they live here once, with the tests that pin their accuracy, and every other crate calls in.
//! The deliberate absence of dependencies is what makes that affordable - taking `oracle-numeric`
//! costs a crate nothing.
//!
//! ## Module map
//! - [`rng`] - the seeded SplitMix64 generator ([`Rng`]) with uniform, normal, and Poisson draws
#![forbid(unsafe_code)]

pub mod rng;

pub use rng::Rng;
