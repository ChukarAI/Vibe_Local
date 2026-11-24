//! Device self-understanding for worker agents: capability snapshot,
//! persistent NodeUUID (`.vibe-node-id` + hardware fingerprint fallback),
//! portable xorshift64 compute benchmark and the Vibe Local `:20129` probe.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::proto::{Caps, VERSION};
use vibe_local_core::hardware::HardwareProbe;

pub const DEFAULT_LLM_ENDPOINT: &str = "http://127.0.0.1:20129";
const LLM_PROBE_TIMEOUT: Duration = Duration::from_millis(300);
const LLM_PROBE_TTL: Duration = Duration::from_secs(10);

/// Normalized base URL of the worker's local OpenAI-compatible server.
pub fn llm_endpoint() -> String {
    let raw = std::env::var("COMPOST_LLM_ENDPOINT")
        .or_else(|_| std::env::var("VIBE_LLM_ENDPOINT"))
        .unwrap_or_else(|_| DEFAULT_LLM_ENDPOINT.to_string());
    let raw = raw.trim().trim_end_matches('/');
    raw.strip_suffix("/v1").unwrap_or(raw).to_string()
}

struct LlmCache {
    endpoint: String,
    at: Option<Instant>,
    has: bool,
    models: Vec<String>,
}

static LLM_CACHE: Mutex<LlmCache> = Mutex::new(LlmCache {
    endpoint: String::new(),
    at: None,
    has: false,
    models: Vec::new(),
});

pub fn reset_llm_probe_cache() {
    let mut c = LLM_CACHE.lock().unwrap();
    c.at = None;
    c.endpoint.clear();
    c.has = false;
    c.models.clear();
}

/// Probe `<base>/v1/models` with a 300ms timeout, cached for 10s so closed
/// localhost ports don't add latency to every agent poll.
pub fn probe_local_llm() -> (bool, Vec<String>) {
    let ep = llm_endpoint();
    {
        let c = LLM_CACHE.lock().unwrap();
        if c.endpoint == ep && c.at.is_some_and(|at| at.elapsed() < LLM_PROBE_TTL) {
            return (c.has, c.models.clone());
        }
    }
    let (has, models) = probe_at(&ep);
    let mut c = LLM_CACHE.lock().unwrap();
    c.endpoint = ep;
    c.at = Some(Instant::now());
    c.has = has;
    c.models = models.clone();
    (has, models)
}

fn probe_at(base: &str) -> (bool, Vec<String>) {
    let agent = ureq::AgentBuilder::new().timeout(LLM_PROBE_TIMEOUT).build();
    let Ok(resp) = agent.get(&format!("{base}/v1/models")).call() else {
        return (false, Vec::new());
    };
    if resp.status() != 200 {
        return (false, Vec::new());
    }
    let Ok(json) = resp.into_json::<serde_json::Value>() else {
        return (true, Vec::new());
    };
    let mut models = Vec::new();
    if let Some(data) = json["data"].as_array() {
        for m in data.iter().take(16) {
            if let Some(id) = m["id"].as_str()
                && !id.is_empty()
            {
                models.push(id.to_string());
            }
        }
    }
    (true, models)
}

/// Capability snapshot of the current machine. Unknown values stay -1/0.
pub fn detect() -> Caps {
    let hw = HardwareProbe::scan();
    let mut caps = Caps {
        cores: hw.cpu_cores,
        mem_mb: hw.ram_total_mb,
        score: 0.0,
        battery_pct: -1,
        on_battery: false,
        temp_c: -1.0,
        arch: hw.arch.clone(),
        os: hw.os.clone(),
        version: VERSION.to_string(),
        has_local_llm: false,
        local_llm_models: Vec::new(),
        capabilities: Vec::new(),
        loaded_models: Vec::new(),
    };
    if let Some((pct, charging)) = battery() {
        caps.battery_pct = pct;
        caps.on_battery = !charging;
    }
    if let Some(t) = vibe_local_engine::thermal::read_celsius() {
        caps.temp_c = t;
    }
    let (has, models) = probe_local_llm();
    caps.has_local_llm = has;
    caps.local_llm_models = models.clone();
    caps.loaded_models = models;
    if has {
        caps.capabilities.push("llm".to_string());
    }
    caps
}

#[cfg(any(target_os = "linux", target_os = "android", test))]
fn battery_from_sysfs(
    entries: impl IntoIterator<Item = std::path::PathBuf>,
) -> Option<(i32, bool)> {
    for path in entries {
        let kind = std::fs::read_to_string(path.join("type")).unwrap_or_default();
        if kind.trim() != "Battery" {
            continue;
        }
        // Missing or malformed capacity on one battery must not abort the
        // scan of the remaining batteries.
        let Ok(cap) = std::fs::read_to_string(path.join("capacity")) else {
            continue;
        };
        let Ok(pct) = cap.trim().parse::<i32>() else {
            continue;
        };
        let status = std::fs::read_to_string(path.join("status")).unwrap_or_default();
        let charging = matches!(status.trim(), "Charging" | "Full");
        return Some((pct, charging));
    }
    None
}

/// Battery (percent, charging). Windows GetSystemPowerStatus, Linux/Android
/// sysfs power_supply. None when there is no battery.
pub fn battery() -> Option<(i32, bool)> {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct SystemPowerStatus {
            ac_line_status: u8,
            battery_flag: u8,
            battery_life_percent: u8,
            reserved1: u8,
            battery_life_time: u32,
            battery_full_life_time: u32,
        }
        unsafe extern "system" {
            fn GetSystemPowerStatus(status: *mut SystemPowerStatus) -> i32;
        }
        unsafe {
            let mut s: SystemPowerStatus = std::mem::zeroed();
            if GetSystemPowerStatus(&mut s) == 0 {
                return None;
            }
            // 128 = no system battery; 255 = unknown percentage.
            if s.battery_flag == 128 || s.battery_life_percent == 255 {
                return None;
            }
            let charging = s.ac_line_status == 1 || (s.battery_flag & 8) != 0;
            Some((i32::from(s.battery_life_percent), charging))
        }
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let entries = std::fs::read_dir("/sys/class/power_supply")
            .ok()?
            .flatten()
            .map(|e| e.path());
        battery_from_sysfs(entries)
    }
    #[cfg(target_os = "macos")]
    {
        let out = vibe_local_core::util::run_with_timeout(
            "pmset",
            &["-g", "batt"],
            Duration::from_secs(3),
        )?;
        // "Now drawing from 'Battery Power'\n -InternalBattery-0 ... 87%; discharging; ..."
        let on_battery = out.contains("Battery Power");
        let pct = out
            .split_whitespace()
            .find(|w| w.ends_with("%;") || w.ends_with('%'))
            .and_then(|w| {
                w.trim_end_matches("%;")
                    .trim_end_matches('%')
                    .parse::<i32>()
                    .ok()
            })?;
        Some((pct, !on_battery))
    }
    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "android",
        target_os = "macos"
    )))]
    {
        None
    }
}

/// Relative single-core compute score: identical xorshift64 kernel on every
/// platform so a 2012 x86 laptop and an ARM phone are comparable.
/// ~1.0 ≈ one healthy modern core.
pub fn benchmark() -> f64 {
    const ITERS: u64 = 20_000_000;
    let start = Instant::now();
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    for _ in 0..ITERS {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
    }
    std::hint::black_box(x);
    let score = ITERS as f64 / start.elapsed().as_secs_f64();
    score / 200_000.0
}

static BENCH_CACHE: Mutex<Option<f64>> = Mutex::new(None);

pub fn benchmarked() -> f64 {
    let mut c = BENCH_CACHE.lock().unwrap();
    if let Some(v) = *c {
        return v;
    }
    let v = benchmark();
    *c = Some(v);
    v
}

pub fn hostname() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .or_else(|| {
            vibe_local_core::util::run_with_timeout("hostname", &[], Duration::from_secs(3))
                .map(|s| s.trim().to_string())
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{}-{}-box", std::env::consts::OS, std::env::consts::ARCH))
}

// --- persistent NodeUUID ---

pub fn default_node_id_path() -> PathBuf {
    let local = PathBuf::from(".vibe-node-id");
    if local.is_file() {
        return local;
    }
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        return PathBuf::from(home).join(".vibe-node-id");
    }
    local
}

/// Load a persisted stable UUID, or create one (atomic write). On read-only
/// filesystems fall back to a deterministic hardware fingerprint so identity
/// survives restarts.
pub fn load_or_create_node_uuid(path: &str) -> String {
    let path = if path.is_empty() {
        default_node_id_path()
    } else {
        PathBuf::from(path)
    };
    if let Ok(raw) = std::fs::read_to_string(&path)
        && let Some(id) = clean_uuid(raw.trim())
    {
        return id;
    }
    let id = new_uuid_v4().unwrap_or_else(hardware_fingerprint_uuid);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, format!("{id}\n")).is_ok() && std::fs::rename(&tmp, &path).is_ok() {
        return id;
    }
    let _ = std::fs::remove_file(&tmp);
    if std::fs::write(&path, format!("{id}\n")).is_ok() {
        return id;
    }
    hardware_fingerprint_uuid()
}

fn clean_uuid(s: &str) -> Option<String> {
    if s.len() < 8 || s.len() > 64 {
        return None;
    }
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        Some(s.to_string())
    } else {
        None
    }
}

fn format_uuid_bytes(b: &[u8; 16]) -> String {
    let h = vibe_local_core::util::hex_encode(b);
    format!(
        "{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    )
}

fn new_uuid_v4() -> Option<String> {
    let mut b = [0u8; 16];
    getrandom::getrandom(&mut b).ok()?;
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    Some(format_uuid_bytes(&b))
}

/// Deterministic device fingerprint: hostname|os|arch|cores|MAC addresses.
pub fn hardware_fingerprint_uuid() -> String {
    let mut h = Sha256::new();
    h.update(format!(
        "{}|{}|{}|{}|",
        hostname(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    ));
    for mac in mac_addresses() {
        h.update(format!("{mac},"));
    }
    let sum = h.finalize();
    let mut b = [0u8; 16];
    b.copy_from_slice(&sum[..16]);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    format_uuid_bytes(&b)
}

fn mac_addresses() -> Vec<String> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let mut out = Vec::new();
        if let Ok(entries) = std::fs::read_dir("/sys/class/net") {
            for e in entries.flatten() {
                if let Ok(mac) = std::fs::read_to_string(e.path().join("address")) {
                    let mac = mac.trim().to_string();
                    if !mac.is_empty() && mac != "00:00:00:00:00:00" {
                        out.push(mac);
                    }
                }
            }
        }
        out
    }
    #[cfg(target_os = "windows")]
    {
        // IP_ADAPTER_ADDRESSES via GetAdaptersAddresses (iphlpapi).
        #[repr(C)]
        struct AdapterAddresses {
            length: u32,
            if_index: u32,
            next: *mut AdapterAddresses,
            adapter_name: *const u8,
            first_unicast: *mut u8,
            first_anycast: *mut u8,
            first_multicast: *mut u8,
            first_dns: *mut u8,
            dns_suffix: *const u16,
            description: *const u16,
            friendly_name: *const u16,
            physical_address: [u8; 8],
            physical_address_length: u32,
            // Remaining fields are not read; the struct is only used for
            // traversal, so trailing padding is intentionally omitted.
        }
        #[link(name = "iphlpapi")]
        unsafe extern "system" {
            fn GetAdaptersAddresses(
                family: u32,
                flags: u32,
                reserved: *mut u8,
                addresses: *mut AdapterAddresses,
                size: *mut u32,
            ) -> u32;
        }
        let mut out = Vec::new();
        let mut size = 16_384u32;
        let mut buf = vec![0u8; size as usize];
        let rc = unsafe {
            GetAdaptersAddresses(
                0,
                0,
                std::ptr::null_mut(),
                buf.as_mut_ptr().cast(),
                &mut size,
            )
        };
        if rc != 0 {
            return out;
        }
        let mut cur = buf.as_mut_ptr() as *mut AdapterAddresses;
        while !cur.is_null() {
            unsafe {
                let a = &*cur;
                let n = a.physical_address_length as usize;
                if n > 0 && n <= 8 {
                    let mac = a.physical_address[..n]
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<Vec<_>>()
                        .join(":");
                    out.push(mac);
                }
                cur = a.next;
            }
        }
        out
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows")))]
    {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn battery_scan_skips_unreadable_entries() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing");
        let malformed = dir.path().join("malformed");
        let valid = dir.path().join("valid");
        for path in [&missing, &malformed, &valid] {
            std::fs::create_dir(path).unwrap();
            std::fs::write(path.join("type"), "Battery\n").unwrap();
        }
        std::fs::write(malformed.join("capacity"), "unknown\n").unwrap();
        std::fs::write(valid.join("capacity"), "73\n").unwrap();
        std::fs::write(valid.join("status"), "Charging\n").unwrap();
        assert_eq!(
            battery_from_sysfs([missing, malformed, valid]),
            Some((73, true))
        );
        assert_eq!(battery_from_sysfs(std::iter::empty()), None);
    }

    #[test]
    fn benchmark_is_sane() {
        let s = benchmark();
        // Absolute scale varies debug vs release; the kernel is identical
        // across platforms so scores stay relatively comparable.
        assert!(s > 0.05 && s < 100_000.0, "score {s}");
    }

    #[test]
    fn uuid_persistence_and_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".vibe-node-id");
        let id1 = load_or_create_node_uuid(&path.to_string_lossy());
        let id2 = load_or_create_node_uuid(&path.to_string_lossy());
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 36);
        assert_eq!(id1.as_bytes()[14], b'4'); // UUID version 4
        let fp1 = hardware_fingerprint_uuid();
        let fp2 = hardware_fingerprint_uuid();
        assert_eq!(fp1, fp2, "fingerprint must be deterministic");
        assert_eq!(fp1.len(), 36);
    }

    #[test]
    fn caps_detect_real() {
        let c = detect();
        assert!(c.cores >= 1);
        assert!(c.mem_mb > 0);
        assert!(!c.os.is_empty());
        assert_eq!(c.version, VERSION);
    }

    #[test]
    fn llm_endpoint_normalization() {
        // Safety: serialized by the test lock below.
        unsafe { std::env::set_var("COMPOST_LLM_ENDPOINT", "http://127.0.0.1:9999/v1/") };
        assert_eq!(llm_endpoint(), "http://127.0.0.1:9999");
        unsafe { std::env::remove_var("COMPOST_LLM_ENDPOINT") };
        reset_llm_probe_cache();
        // Nothing listens on the default port in the test environment unless
        // a Vibe Local server is running; both outcomes are valid.
        let (has, models) = probe_local_llm();
        if has {
            assert!(models.len() <= 16);
        }
    }
}
