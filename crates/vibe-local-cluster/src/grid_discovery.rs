//! Vibe Grid LAN discovery: coordinators answer `VIBE_GRID_DISCOVER_V1` UDP
//! probes on port 20132 with a JSON descriptor, so Vibe Router and agents can
//! find the fleet without manual URLs. Best-effort: a bind failure logs a
//! warning and never aborts the coordinator.

use std::net::UdpSocket;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const DISCOVER_MAGIC: &str = "VIBE_GRID_DISCOVER_V1";
pub const DISCOVER_PORT: u16 = 20132;

/// JSON reply body for a valid probe. `host` is the local IP the probe
/// sender would route to (derived via a connected UDP socket — no packets).
pub fn discovery_reply(coordinator_port: u16, host: &str) -> serde_json::Value {
    serde_json::json!({
        "service": "vibe-grid",
        "coordinator_port": coordinator_port,
        "host": host,
        "version": crate::proto::VERSION,
    })
}

/// Spawn the UDP responder thread; returns None (with a stderr warning)
/// when the port cannot be bound. The thread exits when `stop` is set.
pub fn spawn_responder(
    coordinator_port: u16,
    stop: Arc<AtomicBool>,
) -> Option<std::thread::JoinHandle<()>> {
    let sock = match UdpSocket::bind(("0.0.0.0", DISCOVER_PORT)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "[grid-discovery] UDP responder unavailable on port {DISCOVER_PORT} ({e}); \
                 coordinator continues without LAN discovery"
            );
            return None;
        }
    };
    if let Err(e) = sock.set_read_timeout(Some(Duration::from_millis(500))) {
        eprintln!("[grid-discovery] cannot set read timeout ({e}); responder disabled");
        return None;
    }
    std::thread::Builder::new()
        .name("grid-discovery".into())
        .spawn(move || {
            let mut buf = [0u8; 512];
            while !stop.load(Ordering::SeqCst) {
                let Ok((n, from)) = sock.recv_from(&mut buf) else {
                    continue;
                };
                if String::from_utf8_lossy(&buf[..n]).trim() != DISCOVER_MAGIC {
                    continue;
                }
                let host = source_ip_for(&from.to_string())
                    .map(|ip| ip.to_string())
                    .unwrap_or_default();
                let reply = discovery_reply(coordinator_port, &host);
                let _ = sock.send_to(reply.to_string().as_bytes(), from);
            }
        })
        .ok()
}

/// Local source IP the OS would use to reach `target` (routing-table probe;
/// UDP connect sends no packets).
fn source_ip_for(target: &str) -> Option<std::net::IpAddr> {
    let sock = UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect(target).ok()?;
    let ip = sock.local_addr().ok()?.ip();
    if ip.is_unspecified() { None } else { Some(ip) }
}

/// Broadcast a probe and collect coordinator replies until `timeout`.
/// Results are deduplicated by (host, coordinator_port).
pub fn discover(timeout: Duration) -> Vec<serde_json::Value> {
    let Ok(sock) = UdpSocket::bind("0.0.0.0:0") else {
        return Vec::new();
    };
    let _ = sock.set_broadcast(true);
    let _ = sock.set_read_timeout(Some(Duration::from_millis(200)));
    for target in [
        format!("255.255.255.255:{DISCOVER_PORT}"),
        format!("127.0.0.1:{DISCOVER_PORT}"),
    ] {
        let _ = sock.send_to(DISCOVER_MAGIC.as_bytes(), &target);
    }
    let deadline = Instant::now() + timeout;
    let mut out: Vec<serde_json::Value> = Vec::new();
    let mut buf = [0u8; 2048];
    while Instant::now() < deadline {
        let Ok((n, _)) = sock.recv_from(&mut buf) else {
            continue;
        };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&buf[..n]) else {
            continue;
        };
        if v["service"].as_str() != Some("vibe-grid") {
            continue;
        }
        let key = (
            v["host"].as_str().unwrap_or("").to_string(),
            v["coordinator_port"].as_u64().unwrap_or(0),
        );
        if out.iter().any(|prev| {
            (
                prev["host"].as_str().unwrap_or("").to_string(),
                prev["coordinator_port"].as_u64().unwrap_or(0),
            ) == key
        }) {
            continue;
        }
        out.push(v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both tests bind the fixed discovery port — run them one at a time.
    static PORT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn reply_shape() {
        let v = discovery_reply(7900, "192.168.1.50");
        assert_eq!(v["service"], "vibe-grid");
        assert_eq!(v["coordinator_port"], 7900);
        assert_eq!(v["host"], "192.168.1.50");
        assert_eq!(v["version"], crate::proto::VERSION);
    }

    #[test]
    fn responder_answers_probe_on_loopback() {
        let _guard = PORT_LOCK.lock().unwrap();
        // Bind the real discovery port; skip gracefully when occupied
        // (another coordinator already owns it).
        let stop = Arc::new(AtomicBool::new(false));
        let Some(handle) = spawn_responder(7911, Arc::clone(&stop)) else {
            return;
        };
        let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
        sock.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        sock.send_to(DISCOVER_MAGIC.as_bytes(), "127.0.0.1:20132")
            .unwrap();
        let mut buf = [0u8; 2048];
        let (n, _) = sock.recv_from(&mut buf).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&buf[..n]).unwrap();
        assert_eq!(v["service"], "vibe-grid");
        assert_eq!(v["coordinator_port"], 7911);
        assert!(!v["host"].as_str().unwrap().is_empty());

        // Garbage probes get no reply.
        sock.send_to(b"NOT_A_PROBE", "127.0.0.1:20132").unwrap();
        sock.set_read_timeout(Some(Duration::from_millis(300)))
            .unwrap();
        assert!(sock.recv_from(&mut buf).is_err());

        stop.store(true, Ordering::SeqCst);
        handle.join().unwrap();
    }

    #[test]
    fn discover_finds_local_responder() {
        let _guard = PORT_LOCK.lock().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let Some(handle) = spawn_responder(7922, Arc::clone(&stop)) else {
            return;
        };
        let found = discover(Duration::from_millis(1500));
        assert!(
            found
                .iter()
                .any(|v| v["coordinator_port"].as_u64() == Some(7922)),
            "expected local responder in {found:?}"
        );
        stop.store(true, Ordering::SeqCst);
        handle.join().unwrap();
    }
}
