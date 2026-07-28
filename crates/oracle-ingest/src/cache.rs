//! A tiny thread-safe TTL cache.
//!
//! The live provider caches API responses for a short window so repeated polls
//! don't burn the request budget on data that hasn't changed. The lock is only ever
//! held for the map operation itself - never across an `.await` - so it can be a
//! plain `std::sync::Mutex`.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A clone-on-read cache where entries expire after `ttl`.
pub struct TtlCache<K, V> {
    ttl: Duration,
    map: Mutex<HashMap<K, (Instant, V)>>,
}

impl<K: Eq + Hash + Clone, V: Clone> TtlCache<K, V> {
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            map: Mutex::new(HashMap::new()),
        }
    }

    /// Return a live (unexpired) value, or `None` if absent/stale.
    ///
    /// # Panics
    /// If the internal mutex is poisoned, i.e. a previous caller panicked while holding it. The
    /// guarded section is a `HashMap` lookup that cannot panic on its own, so in practice this
    /// only propagates a panic that already happened.
    pub fn get(&self, key: &K) -> Option<V> {
        let map = self.map.lock().expect("cache mutex poisoned");
        map.get(key).and_then(|(stored, v)| {
            if stored.elapsed() < self.ttl {
                Some(v.clone())
            } else {
                None
            }
        })
    }

    /// Insert/replace a value, stamping it with the current time.
    ///
    /// # Panics
    /// If the internal mutex is poisoned; see [`get`](Self::get).
    pub fn put(&self, key: K, value: V) {
        let mut map = self.map.lock().expect("cache mutex poisoned");
        map.insert(key, (Instant::now(), value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_then_expire() {
        let cache: TtlCache<&str, u32> = TtlCache::new(Duration::from_millis(20));
        cache.put("k", 7);
        assert_eq!(cache.get(&"k"), Some(7));
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(cache.get(&"k"), None, "entry should have expired");
    }
}
