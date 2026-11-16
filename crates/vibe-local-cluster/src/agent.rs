//! Worker agent: join → poll → execute → heartbeat → async gzip upload.
//! All connections are initiated outward (NAT/phone friendly). Battery and
//! thermal guards pause the node; the coordinator never dials in.

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine;

use crate::node;
use crate::proto::{
    Caps, HeartbeatRequest, JoinRequest, JoinResponse, PollRequest, PollResponse, ResultMsg, Task,
    TaskLeaseRef,
};
use crate::workload::Registry;

pub const KEY_HEADER: &str = "X-Compost-Key";
const GZIP_THRESHOLD: usize = 4 << 10;
const MAX_PENDING: usize = 256;

#[derive(Clone)]
pub struct AgentConfig {
    pub coordinator: String,
    pub name: String,
    pub key: String,
    pub node_uuid: String,
    pub node_id_file: String,
    pub battery_guard_off: bool,
    pub workers: usize,
    pub heartbeat_every: Duration,
    /// Optional injected caps (Android supplies battery/thermal from Kotlin).
    pub caps_override: Option<Caps>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            coordinator: String::new(),
            name: String::new(),
            key: String::new(),
            node_uuid: String::new(),
            node_id_file: String::new(),
            battery_guard_off: false,
            workers: 2,
            heartbeat_every: Duration::from_secs(5),
            caps_override: None,
        }
    }
}

struct AgentState {
    id: String,
    pending: Vec<ResultMsg>,
    executing: usize,
    inflight: BTreeMap<String, u64>,
    upload_err: u32,
}

pub struct Agent {
    cfg: AgentConfig,
    state: Mutex<AgentState>,
    stop: Arc<AtomicBool>,
    caps_score: f64,
    registry: Registry,
}

pub struct AgentHandle {
    agent: Arc<Agent>,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl AgentHandle {
    pub fn stop(&self) {
        self.agent.stop.store(true, Ordering::SeqCst);
        for h in self.threads.lock().unwrap().drain(..) {
            let _ = h.join();
        }
        self.agent.flush_all();
    }

    pub fn node_id(&self) -> String {
        self.agent.state.lock().unwrap().id.clone()
    }
}

impl Drop for AgentHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

fn backoff_full_jitter(attempt: u32) -> Duration {
    let base = Duration::from_secs(1);
    let max_wait = Duration::from_secs(32);
    let ceil = (base * 2u32.pow(attempt.min(5))).min(max_wait);
    let mut buf = [0u8; 8];
    let _ = getrandom::getrandom(&mut buf);
    let r = u64::from_le_bytes(buf) % (ceil.as_millis() as u64 + 1);
    Duration::from_millis(r)
}

/// Start the agent loop on background threads; returns a stoppable handle.
pub fn start_agent(mut cfg: AgentConfig) -> anyhow::Result<Arc<AgentHandle>> {
    anyhow::ensure!(!cfg.coordinator.is_empty(), "--coordinator URL is required");
    cfg.workers = cfg.workers.clamp(1, 16);
    if cfg.heartbeat_every.is_zero() {
        cfg.heartbeat_every = Duration::from_secs(5);
    }
    if cfg.name.is_empty() {
        cfg.name = node::hostname();
    }
    if cfg.node_uuid.is_empty() {
        cfg.node_uuid = node::load_or_create_node_uuid(&cfg.node_id_file);
    }
    let score = node::benchmarked();
    let agent = Arc::new(Agent {
        cfg,
        state: Mutex::new(AgentState {
            id: String::new(),
            pending: Vec::new(),
            executing: 0,
            inflight: BTreeMap::new(),
            upload_err: 0,
        }),
        stop: Arc::new(AtomicBool::new(false)),
        caps_score: score,
        registry: Registry::with_defaults(),
    });

    // Join (blocking with backoff until the coordinator answers or stop).
    agent.join()?;
    eprintln!(
        "[agent] {} joined as {} (uuid={})",
        agent.cfg.name,
        agent.state.lock().unwrap().id,
        agent.cfg.node_uuid
    );

    let mut threads = Vec::new();
    // Upload loop.
    {
        let a = Arc::clone(&agent);
        threads.push(
            std::thread::Builder::new()
                .name("grid-agent-upload".into())
                .spawn(move || a.upload_loop())?,
        );
    }
    // Heartbeat loop.
    {
        let a = Arc::clone(&agent);
        threads.push(
            std::thread::Builder::new()
                .name("grid-agent-heartbeat".into())
                .spawn(move || a.heartbeat_loop())?,
        );
    }
    // Worker pool + poll loop.
    let (task_tx, task_rx) = std::sync::mpsc::sync_channel::<Task>(64);
    let task_rx = Arc::new(Mutex::new(task_rx));
    for i in 0..agent.cfg.workers {
        let a = Arc::clone(&agent);
        let rx = Arc::clone(&task_rx);
        threads.push(
            std::thread::Builder::new()
                .name(format!("grid-agent-worker-{i}"))
                .spawn(move || {
                    loop {
                        let t = {
                            let guard = rx.lock().unwrap();
                            guard.recv_timeout(Duration::from_millis(500))
                        };
                        match t {
                            Ok(task) => a.execute(task),
                            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                                if a.stop.load(Ordering::SeqCst) {
                                    return;
                                }
                            }
                            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
                        }
                    }
                })?,
        );
    }
    {
        let a = Arc::clone(&agent);
        threads.push(
            std::thread::Builder::new()
                .name("grid-agent-poll".into())
                .spawn(move || {
                    while !a.stop.load(Ordering::SeqCst) {
                        let poll_after = if a.congested() {
                            5000
                        } else {
                            a.poll_and_work(&task_tx)
                        };
                        let deadline = std::time::Instant::now()
                            + Duration::from_millis(poll_after.max(200) as u64);
                        while std::time::Instant::now() < deadline && !a.stop.load(Ordering::SeqCst)
                        {
                            std::thread::sleep(Duration::from_millis(100));
                        }
                    }
                })?,
        );
    }

    Ok(Arc::new(AgentHandle {
        agent,
        threads: Mutex::new(threads),
    }))
}

impl Agent {
    fn client(&self) -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(120))
            .build()
    }

    fn authorize(&self, req: ureq::Request) -> ureq::Request {
        if self.cfg.key.is_empty() {
            req
        } else {
            req.set(KEY_HEADER, &self.cfg.key)
        }
    }

    fn detect_caps(&self) -> Caps {
        let mut c = match &self.cfg.caps_override {
            Some(override_caps) => {
                let mut base = node::detect();
                base.battery_pct = override_caps.battery_pct;
                base.on_battery = override_caps.on_battery;
                base.temp_c = override_caps.temp_c;
                if !override_caps.os.is_empty() {
                    base.os = override_caps.os.clone();
                }
                base
            }
            None => node::detect(),
        };
        c.score = self.caps_score;
        c
    }

    fn pause_reason(&self, c: &Caps) -> String {
        if c.battery_pct >= 0 && c.on_battery && c.battery_pct < 30 && !self.cfg.battery_guard_off {
            return format!("battery {}% (<30%)", c.battery_pct);
        }
        if c.temp_c > 80.0 {
            return format!("thermal {:.0}°C (>80°C)", c.temp_c);
        }
        if c.os == "android" && c.temp_c > 45.0 {
            return format!("battery thermal {:.0}°C (>45°C Android)", c.temp_c);
        }
        String::new()
    }

    fn join(&self) -> anyhow::Result<()> {
        let mut attempt = 0u32;
        loop {
            if self.stop.load(Ordering::SeqCst) {
                anyhow::bail!("agent stopped before joining");
            }
            let body = JoinRequest {
                name: self.cfg.name.clone(),
                node_uuid: self.cfg.node_uuid.clone(),
                caps: self.detect_caps(),
            };
            let result = self
                .authorize(
                    self.client()
                        .post(&format!("{}/api/v1/join", self.cfg.coordinator)),
                )
                .send_json(serde_json::to_value(&body)?);
            match result {
                Ok(resp) if resp.status() == 200 => {
                    let jr: JoinResponse = resp.into_json()?;
                    if !jr.node_id.is_empty() {
                        self.state.lock().unwrap().id = jr.node_id;
                        return Ok(());
                    }
                }
                Ok(resp) if resp.status() == 401 => {
                    anyhow::bail!("fleet rejected our key — check --key");
                }
                Ok(resp) => {
                    eprintln!("[agent] join HTTP {}", resp.status());
                }
                Err(e) => {
                    eprintln!("[agent] waiting for coordinator: {e}");
                }
            }
            let wait = backoff_full_jitter(attempt);
            attempt += 1;
            let deadline = std::time::Instant::now() + wait;
            while std::time::Instant::now() < deadline && !self.stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    fn node_id(&self) -> String {
        self.state.lock().unwrap().id.clone()
    }

    fn executing_count(&self) -> usize {
        let s = self.state.lock().unwrap();
        s.executing + s.pending.len()
    }

    fn congested(&self) -> bool {
        self.state.lock().unwrap().pending.len() >= MAX_PENDING
    }

    fn poll_and_work(&self, tx: &std::sync::mpsc::SyncSender<Task>) -> i64 {
        let caps = self.detect_caps();
        let body = PollRequest {
            node_id: self.node_id(),
            node_uuid: self.cfg.node_uuid.clone(),
            name: self.cfg.name.clone(),
            paused: self.pause_reason(&caps),
            running: self.executing_count(),
            workers: self.cfg.workers,
            caps,
        };
        let resp = match self
            .authorize(
                self.client()
                    .post(&format!("{}/api/v1/poll", self.cfg.coordinator)),
            )
            .send_json(serde_json::to_value(&body).unwrap_or_default())
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[agent] poll failed: {e} — re-joining");
                let _ = self.join();
                return 1000;
            }
        };
        match resp.status() {
            404 => {
                let _ = self.join();
                return 1000;
            }
            401 => {
                eprintln!("[agent] fleet rejected our key — check --key");
                return 5000;
            }
            200 => {}
            _ => return 2000,
        }
        let pr: PollResponse = match resp.into_json() {
            Ok(p) => p,
            Err(_) => return 2000,
        };
        let list = pr.task_list();
        if list.is_empty() {
            return pr.poll_after_ms.max(500);
        }
        for t in list {
            if self.stop.load(Ordering::SeqCst) {
                return 0;
            }
            // Bounded dispatch: never block the poll loop forever.
            let mut task = Some(t);
            while let Some(t) = task.take() {
                match tx.try_send(t) {
                    Ok(()) => {}
                    Err(std::sync::mpsc::TrySendError::Full(t)) => {
                        task = Some(t);
                        std::thread::sleep(Duration::from_millis(100));
                        if self.stop.load(Ordering::SeqCst) {
                            return 0;
                        }
                    }
                    Err(std::sync::mpsc::TrySendError::Disconnected(_)) => return 0,
                }
            }
        }
        0
    }

    fn execute(&self, t: Task) {
        {
            let mut s = self.state.lock().unwrap();
            s.inflight.insert(t.id.clone(), t.lease_epoch);
            s.executing += 1;
        }
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.execute_inner(&t)));
        let res = match result {
            Ok(r) => r,
            Err(_) => ResultMsg {
                task_id: t.id.clone(),
                node_id: self.node_id(),
                lease_epoch: t.lease_epoch,
                error: "panic during task execution".into(),
                millis: 1,
                ..Default::default()
            },
        };
        {
            let mut s = self.state.lock().unwrap();
            s.inflight.remove(&t.id);
            s.executing = s.executing.saturating_sub(1);
            s.pending.push(res);
        }
    }

    fn execute_inner(&self, t: &Task) -> ResultMsg {
        let start = std::time::Instant::now();
        let outcome = match self.registry.get(&t.kind) {
            Some(preset) => preset.execute(t),
            None => Err(anyhow::anyhow!(
                "agent: no preset for kind {:?} (agent older than coordinator?)",
                t.kind
            )),
        };
        let exec_millis = (start.elapsed().as_millis() as i64).max(1);
        let mut res = ResultMsg {
            task_id: t.id.clone(),
            node_id: self.node_id(),
            lease_epoch: t.lease_epoch,
            millis: exec_millis,
            exec_millis,
            ..Default::default()
        };
        match outcome {
            Ok((data, _enc)) => {
                res.data = Some(data);
                eprintln!(
                    "[agent] task {} done in {exec_millis}ms (epoch={})",
                    t.id, t.lease_epoch
                );
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("no preset for kind") {
                    res.unsupported = true;
                }
                res.error = msg;
            }
        }
        res
    }

    fn heartbeat_loop(&self) {
        while !self.stop.load(Ordering::SeqCst) {
            let deadline = std::time::Instant::now() + self.cfg.heartbeat_every;
            while std::time::Instant::now() < deadline && !self.stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(100));
            }
            let refs: Vec<TaskLeaseRef> = {
                let s = self.state.lock().unwrap();
                s.inflight
                    .iter()
                    .map(|(id, ep)| TaskLeaseRef {
                        task_id: id.clone(),
                        lease_epoch: *ep,
                    })
                    .collect()
            };
            if refs.is_empty() {
                continue;
            }
            let body = HeartbeatRequest {
                node_id: self.node_id(),
                tasks: refs,
            };
            let _ = self
                .authorize(
                    self.client()
                        .post(&format!("{}/api/v1/heartbeat", self.cfg.coordinator)),
                )
                .send_json(serde_json::to_value(&body).unwrap_or_default());
        }
    }

    fn upload_loop(&self) {
        let base_tick = Duration::from_millis(150);
        while !self.stop.load(Ordering::SeqCst) {
            let ok = self.upload_once();
            let wait = if ok {
                self.state.lock().unwrap().upload_err = 0;
                base_tick
            } else {
                let mut s = self.state.lock().unwrap();
                s.upload_err += 1;
                backoff_full_jitter(s.upload_err.min(7)).max(base_tick)
            };
            let deadline = std::time::Instant::now() + wait;
            while std::time::Instant::now() < deadline && !self.stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }

    fn upload_once(&self) -> bool {
        let snapshot: Vec<ResultMsg> = self.state.lock().unwrap().pending.clone();
        for res in snapshot {
            let Ok(mut body) = serde_json::to_vec(&res) else {
                return false;
            };
            let mut use_gzip = false;
            if body.len() > GZIP_THRESHOLD {
                let mut enc =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
                if enc.write_all(&body).is_ok()
                    && let Ok(compressed) = enc.finish()
                    && compressed.len() < body.len()
                {
                    body = compressed;
                    use_gzip = true;
                }
            }
            let mut req = self
                .client()
                .post(&format!("{}/api/v1/result", self.cfg.coordinator))
                .set("Content-Type", "application/json");
            if use_gzip {
                req = req.set("Content-Encoding", "gzip");
            }
            req = self.authorize(req);
            let resp = match req.send_bytes(&body) {
                Ok(r) => r,
                Err(_) => return false,
            };
            match resp.status() {
                200 => self.drop_result(&res.task_id),
                401 => {
                    eprintln!("[agent] fleet rejected our key — dropping queued results");
                    self.drop_result(&res.task_id);
                }
                s if s < 500 => self.drop_result(&res.task_id),
                _ => return false,
            }
        }
        true
    }

    fn drop_result(&self, task_id: &str) {
        let mut s = self.state.lock().unwrap();
        s.pending.retain(|r| r.task_id != task_id);
    }

    fn flush_all(&self) {
        for _ in 0..3 {
            self.upload_once();
            if self.state.lock().unwrap().pending.is_empty() {
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

/// Encode a result payload for the wire (base64, Go []byte convention).
pub fn encode_payload(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::{CoordinatorOptions, start_coordinator};

    #[test]
    fn agent_end_to_end_primes() {
        let dir = tempfile::tempdir().unwrap();
        let coord = start_coordinator(CoordinatorOptions {
            addr: "127.0.0.1:0".into(),
            state_path: Some(dir.path().join("state.gz")),
            allow_same_ip_replicate: true,
            ..Default::default()
        })
        .unwrap();
        let base = format!("http://127.0.0.1:{}", coord.port);
        // Submit a primes job first.
        let submit: serde_json::Value = ureq::post(&format!("{base}/api/v1/jobs"))
            .send_json(serde_json::json!({"kind": "primes", "params": {"limit": 200000}}))
            .unwrap()
            .into_json()
            .unwrap();
        let job_id = submit["job"]["id"].as_str().unwrap().to_string();
        // Start a real agent against it.
        let agent = start_agent(AgentConfig {
            coordinator: base.clone(),
            name: "test-agent".into(),
            node_id_file: dir
                .path()
                .join(".vibe-node-id")
                .to_string_lossy()
                .into_owned(),
            workers: 2,
            ..Default::default()
        })
        .unwrap();
        assert!(!agent.node_id().is_empty());
        // Wait for the job to settle.
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut done = false;
        while std::time::Instant::now() < deadline {
            let st: serde_json::Value = ureq::get(&format!("{base}/api/v1/state"))
                .call()
                .unwrap()
                .into_json()
                .unwrap();
            if let Some(j) = st["jobs"]
                .as_array()
                .unwrap()
                .iter()
                .find(|j| j["id"] == job_id)
                && j["status"] == "done"
            {
                done = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        agent.stop();
        assert!(done, "agent must complete the primes job");
        // Artifact verifies against the known π value.
        let mut buf = Vec::new();
        ureq::get(&format!("{base}/api/v1/jobs/{job_id}/artifact"))
            .call()
            .unwrap()
            .into_reader()
            .read_to_end(&mut buf)
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&buf),
            "primes below 200000: 17984\n"
        );
        coord.shutdown();
    }

    #[test]
    fn backoff_bounds() {
        for attempt in 0..10 {
            let d = backoff_full_jitter(attempt);
            assert!(d <= Duration::from_secs(32));
        }
    }

    #[test]
    fn pause_reasons() {
        let agent = Agent {
            cfg: AgentConfig::default(),
            state: Mutex::new(AgentState {
                id: String::new(),
                pending: Vec::new(),
                executing: 0,
                inflight: BTreeMap::new(),
                upload_err: 0,
            }),
            stop: Arc::new(AtomicBool::new(false)),
            caps_score: 1.0,
            registry: Registry::with_defaults(),
        };
        let mut c = Caps {
            battery_pct: 20,
            on_battery: true,
            temp_c: -1.0,
            ..Default::default()
        };
        assert!(agent.pause_reason(&c).contains("battery 20%"));
        c.battery_pct = 80;
        c.temp_c = 85.0;
        assert!(agent.pause_reason(&c).contains("thermal"));
        c.temp_c = 46.0;
        c.os = "android".into();
        assert!(agent.pause_reason(&c).contains("Android"));
        c.temp_c = 30.0;
        assert!(agent.pause_reason(&c).is_empty());
    }
}
