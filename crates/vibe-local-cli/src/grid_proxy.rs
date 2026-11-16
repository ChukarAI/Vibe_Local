//! Same-origin proxy from the Vibe-Link server (:20129) to a Vibe Grid
//! coordinator (:7900), so the web dashboard and Android app can reach the
//! fleet without cross-origin requests (strict CSP `connect-src 'self'`).
//!
//! Unlike the HF download guard, this proxy intentionally allows loopback
//! and RFC1918 targets — the coordinator lives on the user's LAN — but still
//! rejects link-local, multicast and unspecified addresses, and only
//! http/https schemes.

use std::net::ToSocketAddrs;
use std::time::Duration;

use serde_json::json;
use url::Url;

use vibe_local_cluster::moe_grid::{self, GridNode};

pub const DEFAULT_COORDINATOR: &str = "http://127.0.0.1:7900";

pub fn coordinator_base(explicit: Option<&str>) -> String {
    explicit
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .or_else(|| std::env::var("VIBE_GRID_COORDINATOR").ok())
        .unwrap_or_else(|| DEFAULT_COORDINATOR.to_string())
        .trim_end_matches('/')
        .to_string()
}

/// LAN-proxy URL validation: http(s) only; block link-local/multicast/
/// unspecified; loopback and private ranges are allowed (that's the point).
pub fn validate_lan_url(raw: &str) -> anyhow::Result<Url> {
    let url = Url::parse(raw).map_err(|e| anyhow::anyhow!("invalid coordinator URL: {e}"))?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https"),
        "coordinator URL must be http(s)"
    );
    let host = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("coordinator URL has no host"))?;
    let port = url.port_or_known_default().unwrap_or(80);
    let addrs: Vec<_> = (host, port)
        .to_socket_addrs()
        .map_err(|e| anyhow::anyhow!("cannot resolve coordinator host {host}: {e}"))?
        .collect();
    anyhow::ensure!(!addrs.is_empty(), "no addresses for {host}");
    for addr in &addrs {
        let ip = addr.ip();
        let blocked = match ip {
            std::net::IpAddr::V4(v4) => {
                v4.is_link_local() || v4.is_multicast() || v4.is_unspecified() || v4.is_broadcast()
            }
            std::net::IpAddr::V6(v6) => {
                let seg = v6.segments();
                v6.is_multicast() || v6.is_unspecified() || (seg[0] & 0xffc0) == 0xfe80
            }
        };
        anyhow::ensure!(!blocked, "coordinator address {ip} is not allowed");
    }
    Ok(url)
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(15))
        .build()
}

/// GET JSON from the coordinator, forwarding the fleet key.
pub fn fetch_json(base: &str, path: &str, key: Option<&str>) -> anyhow::Result<serde_json::Value> {
    let url = format!("{base}{path}");
    validate_lan_url(&url)?;
    let mut req = agent().get(&url);
    if let Some(k) = key
        && !k.is_empty()
    {
        req = req.set("X-Compost-Key", k);
    }
    let resp = req
        .call()
        .map_err(|e| anyhow::anyhow!("coordinator request failed: {e}"))?;
    Ok(resp.into_json()?)
}

/// POST JSON to the coordinator, forwarding the fleet key.
pub fn post_json(
    base: &str,
    path: &str,
    key: Option<&str>,
    body: serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    let url = format!("{base}{path}");
    validate_lan_url(&url)?;
    let mut req = agent().post(&url).set("Content-Type", "application/json");
    if let Some(k) = key
        && !k.is_empty()
    {
        req = req.set("X-Compost-Key", k);
    }
    let resp = req
        .send_json(body)
        .map_err(|e| anyhow::anyhow!("coordinator request failed: {e}"))?;
    Ok(resp.into_json()?)
}

/// GET raw bytes (artifact passthrough).
pub fn fetch_bytes(base: &str, path: &str, key: Option<&str>) -> anyhow::Result<(Vec<u8>, String)> {
    let url = format!("{base}{path}");
    validate_lan_url(&url)?;
    let mut req = agent().get(&url);
    if let Some(k) = key
        && !k.is_empty()
    {
        req = req.set("X-Compost-Key", k);
    }
    let resp = req
        .call()
        .map_err(|e| anyhow::anyhow!("coordinator request failed: {e}"))?;
    let ctype = resp
        .header("Content-Type")
        .unwrap_or("application/octet-stream")
        .to_string();
    let mut buf = Vec::new();
    resp.into_reader()
        .take(64 * 1024 * 1024)
        .read_to_end(&mut buf)?;
    Ok((buf, ctype))
}

use std::io::Read;

/// Compute a MoE partition plan from the live fleet (coordinator nodes +
/// this machine as the coordinator/primary node).
pub fn plan_moe_live(
    model_id: &str,
    context: u64,
    base: &str,
    key: Option<&str>,
) -> anyhow::Result<serde_json::Value> {
    let spec = moe_grid::find_model(model_id).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown MoE model {model_id:?}; known: {:?}",
            moe_grid::moe_registry()
                .iter()
                .map(|m| m.model_id.clone())
                .collect::<Vec<_>>()
        )
    })?;
    let mut nodes: Vec<GridNode> = vec![moe_grid::local_grid_node()];
    // Fleet state is best-effort: a solo plan is valid when no coordinator
    // is running yet.
    if let Ok(state) = fetch_json(base, "/api/v1/state", key)
        && let Ok(st) = serde_json::from_value::<vibe_local_cluster::proto::StateResponse>(state)
    {
        let fleet = moe_grid::grid_nodes_from_state(&st, "");
        // Avoid duplicating the local machine if it already joined.
        let local_uuid = nodes[0].node_uuid.clone();
        for n in fleet {
            if n.node_uuid != local_uuid {
                nodes.push(n);
            }
        }
    }
    let plan = moe_grid::plan_moe(&spec, &nodes, "Q4_K_M", context, 5.0)?;
    Ok(serde_json::to_value(plan)?)
}

/// Combined health + state for dashboards.
pub fn status_combined(base: &str, key: Option<&str>) -> anyhow::Result<serde_json::Value> {
    let health = fetch_json(base, "/healthz", key).unwrap_or(json!({"status": "unreachable"}));
    let state = fetch_json(base, "/api/v1/state", key)?;
    Ok(json!({"health": health, "state": state}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lan_url_validation() {
        assert!(validate_lan_url("http://127.0.0.1:7900").is_ok());
        assert!(validate_lan_url("http://192.168.1.50:7900").is_ok());
        assert!(validate_lan_url("https://example.com").is_ok());
        assert!(validate_lan_url("ftp://127.0.0.1").is_err());
        assert!(validate_lan_url("http://169.254.169.254").is_err());
        assert!(validate_lan_url("http://224.0.0.1").is_err());
        assert!(validate_lan_url("http://0.0.0.0").is_err());
        assert!(validate_lan_url("not a url").is_err());
    }

    #[test]
    fn coordinator_base_resolution() {
        assert_eq!(
            coordinator_base(Some("http://10.0.0.5:7900/")),
            "http://10.0.0.5:7900"
        );
        assert_eq!(coordinator_base(None), DEFAULT_COORDINATOR);
        assert_eq!(coordinator_base(Some("")), DEFAULT_COORDINATOR);
    }

    #[test]
    fn solo_plan_works_without_coordinator() {
        // Nothing listens on this port: plan must still succeed solo.
        let plan = plan_moe_live("OLMoE", 4096, "http://127.0.0.1:1", None).unwrap();
        assert_eq!(plan["routed_experts"], 64);
        assert!(!plan["shards"].as_array().unwrap().is_empty());
        assert!(plan_moe_live("not-a-model", 4096, "http://127.0.0.1:1", None).is_err());
    }
}
