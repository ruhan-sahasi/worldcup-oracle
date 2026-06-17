//! A small async token-bucket rate limiter.
//!
//! Free-tier football APIs cap requests (football-data.org allows ~10/min). Rather
//! than pull in a heavyweight dependency we implement the classic token-bucket: the
//! bucket refills at a steady rate up to a capacity, and each request must
//! [`acquire`](RateLimiter::acquire) a token, awaiting (without busy-spinning) when
//! the bucket is empty. This is the back-pressure mechanism on the live data source.

use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::{sleep, Instant};

struct Bucket {
    tokens: f64,
    capacity: f64,
    refill_per_sec: f64,
    last: Instant,
}

/// A token-bucket limiter shared across tasks via `&self`.
pub struct RateLimiter {
    bucket: Mutex<Bucket>,
}

impl RateLimiter {
    /// Allow at most `n` requests per minute (burst capacity = `n`).
    pub fn per_minute(n: u32) -> Self {
        let capacity = n.max(1) as f64;
        Self {
            bucket: Mutex::new(Bucket {
                tokens: capacity,
                capacity,
                refill_per_sec: capacity / 60.0,
                last: Instant::now(),
            }),
        }
    }

    /// Acquire one token, awaiting until one is available.
    pub async fn acquire(&self) {
        loop {
            let wait = {
                let mut b = self.bucket.lock().await;
                let now = Instant::now();
                let elapsed = now.duration_since(b.last).as_secs_f64();
                b.tokens = (b.tokens + elapsed * b.refill_per_sec).min(b.capacity);
                b.last = now;
                if b.tokens >= 1.0 {
                    b.tokens -= 1.0;
                    return;
                }
                // Seconds until the next whole token becomes available.
                Duration::from_secs_f64((1.0 - b.tokens) / b.refill_per_sec)
            };
            sleep(wait).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn burst_then_throttle() {
        // 60/min ⇒ 1/sec; capacity 60 ⇒ first 60 are instant, the 61st waits ~1s.
        let rl = RateLimiter::per_minute(60);
        let start = Instant::now();
        for _ in 0..60 {
            rl.acquire().await;
        }
        // With a paused clock, the burst consumes no virtual time.
        assert!(start.elapsed() < Duration::from_millis(1));
        rl.acquire().await; // must wait for a refill
        assert!(start.elapsed() >= Duration::from_secs(1));
    }
}
