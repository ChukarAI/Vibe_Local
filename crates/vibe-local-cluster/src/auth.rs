//! Fleet shared-secret enforcement: constant-time `X-Compost-Key` header
//! check with `?key=` query fallback for browser artifact/download links.
//! Wire-compatible with compost v0.7.0's auth package.

use vibe_local_core::util::constant_time_eq;

pub const HEADER: &str = "X-Compost-Key";
pub const QUERY_PARAM: &str = "key";

#[derive(Clone, Default)]
pub struct Enforcer {
    secret: String,
}

impl Enforcer {
    /// Empty secret = open fleet (auth disabled).
    pub fn new(secret: &str) -> Enforcer {
        Enforcer {
            secret: secret.to_string(),
        }
    }

    pub fn enabled(&self) -> bool {
        !self.secret.is_empty()
    }

    pub fn check(&self, candidate: &str) -> bool {
        if !self.enabled() {
            return true;
        }
        constant_time_eq(candidate.as_bytes(), self.secret.as_bytes())
    }

    /// Validate header, optionally falling back to the query parameter.
    pub fn authorize(
        &self,
        header_value: Option<&str>,
        query_value: Option<&str>,
        allow_query: bool,
    ) -> bool {
        if !self.enabled() {
            return true;
        }
        if let Some(got) = header_value
            && !got.is_empty()
        {
            return self.check(got);
        }
        if allow_query
            && let Some(got) = query_value
            && !got.is_empty()
        {
            return self.check(got);
        }
        false
    }
}

pub const UNAUTHORIZED_BODY: &str = "missing or invalid X-Compost-Key";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_fleet_allows_everything() {
        let e = Enforcer::new("");
        assert!(!e.enabled());
        assert!(e.authorize(None, None, false));
        assert!(e.check("anything"));
    }

    #[test]
    fn constant_time_key_check() {
        let e = Enforcer::new("fleet-secret");
        assert!(e.enabled());
        assert!(e.check("fleet-secret"));
        assert!(!e.check("fleet-secre"));
        assert!(!e.check("fleet-secrets"));
        assert!(!e.check(""));
        assert!(e.authorize(Some("fleet-secret"), None, false));
        assert!(!e.authorize(Some("wrong"), None, false));
        assert!(!e.authorize(None, None, false));
        // Query fallback only when allowed.
        assert!(!e.authorize(None, Some("fleet-secret"), false));
        assert!(e.authorize(None, Some("fleet-secret"), true));
        // Header wins over query.
        assert!(!e.authorize(Some("wrong"), Some("fleet-secret"), true));
    }
}
