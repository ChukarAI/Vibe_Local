//! Vibe Grid coordinator: pull-based HTTP API (compost v0.7.0 wire
//! compatible), artifact assembly with 8-entry LRU + disk offload, payload
//! reclamation, gzip-bomb protection, Prometheus metrics, ETA/Waiting
//! annotation, crash-safe persistence and an embedded status dashboard.

use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::json;
use tiny_http::{Header, Method, Request, Response, Server};

use crate::auth::{self, Enforcer};
use crate::metrics;
use crate::moe_grid;
use crate::persist;
use crate::proto::{
    HeartbeatRequest, HeartbeatResponse, JOB_CANCELLED, JOB_DONE, JOB_FAILED, JOB_QUEUED,
    JOB_RUNNING, JoinRequest, JoinResponse, PollRequest, PollResponse, ResultMsg, StateResponse,
    SubmitJobRequest, SubmitJobResponse, TASK_EMBED_BATCH, TASK_LLM_BATCH, TASK_MOE_DISTRIBUTED,
    TASK_WORDCOUNT,
};
use crate::scheduler::Queue;
use crate::workload::Registry;

pub const MAX_BODY_BYTES: usize = 9 << 20;
pub const MAX_JOB_INPUT: usize = 8 << 20;
pub const MAX_CONCURRENT_JOBS: usize = 8;
pub const MAX_CACHED_ARTIFACTS: usize = 8;
const PERSIST_EVERY: Duration = Duration::from_secs(10);
const ARTIFACT_WAIT_CLIENT: Duration = Duration::from_secs(60);
const ARTIFACT_WAIT_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const WORKER_THREADS: usize = 4;

const CSP: &str = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'";

#[derive(Clone, Default)]
pub struct CoordinatorOptions {
    pub addr: String,
    pub key: String,
    pub protect_artifacts: bool,
    pub allow_same_ip_replicate: bool,
    pub state_path: Option<PathBuf>,
}

struct ArtifactState {
    ready: Mutex<bool>,
    cv: Condvar,
    building: AtomicBool,
    data: Mutex<Option<Vec<u8>>>,
    ctype: Mutex<String>,
    ext: Mutex<String>,
    evicted: AtomicBool,
}

impl ArtifactState {
    fn new() -> Arc<ArtifactState> {
        Arc::new(ArtifactState {
            ready: Mutex::new(false),
            cv: Condvar::new(),
            building: AtomicBool::new(false),
            data: Mutex::new(None),
            ctype: Mutex::new(String::new()),
            ext: Mutex::new(String::new()),
            evicted: AtomicBool::new(false),
        })
    }
    fn settle(&self) {
        let mut r = self.ready.lock().unwrap();
        *r = true;
        self.cv.notify_all();
    }
    fn wait(&self, timeout: Duration) -> bool {
        let r = self.ready.lock().unwrap();
        if *r {
            return true;
        }
        let (r, _) = self.cv.wait_timeout(r, timeout).unwrap();
        *r
    }
}

/// Shared artifact cache: 8-entry LRU over assembled job artifacts.
#[derive(Default)]
struct ArtifactCache {
    arts: Mutex<BTreeMap<String, Arc<ArtifactState>>>,
    lru: Mutex<Vec<String>>,
}

impl ArtifactCache {
    fn get_or_insert(&self, id: &str) -> Arc<ArtifactState> {
        let mut arts = self.arts.lock().unwrap();
        arts.entry(id.to_string())
            .or_insert_with(ArtifactState::new)
            .clone()
    }
    fn get(&self, id: &str) -> Option<Arc<ArtifactState>> {
        self.arts.lock().unwrap().get(id).cloned()
    }
    fn touch(&self, id: &str) {
        {
            let mut lru = self.lru.lock().unwrap();
            lru.retain(|x| x != id);
            lru.insert(0, id.to_string());
        }
        let victims: Vec<String> = {
            let mut lru = self.lru.lock().unwrap();
            let mut v = Vec::new();
            while lru.len() > MAX_CACHED_ARTIFACTS {
                v.push(lru.pop().unwrap());
            }
            v
        };
        if !victims.is_empty() {
            let mut arts = self.arts.lock().unwrap();
            for v in victims {
                arts.remove(&v);
                eprintln!("[artifact] evicted {v} from cache (budget {MAX_CACHED_ARTIFACTS})");
            }
        }
    }
}

pub struct Coordinator {
    pub queue: Arc<Queue>,
    registry: Arc<Registry>,
    keys: Enforcer,
    opts: CoordinatorOptions,
    cache: Arc<ArtifactCache>,
    settled: Mutex<HashSet<String>>,
    instance_id: String,
    boot: Instant,
}

pub struct CoordinatorHandle {
    pub port: u16,
    pub coordinator: Arc<Coordinator>,
    stop: Arc<AtomicBool>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    background: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl CoordinatorHandle {
    pub fn shutdown(&self) {
        if self.stop.swap(true, Ordering::SeqCst) {
            return;
        }
        for _ in 0..WORKER_THREADS {
            if let Ok(mut stream) = std::net::TcpStream::connect(("127.0.0.1", self.port)) {
                use std::io::Write;
                let _ =
                    stream.write_all(b"GET /__vibe_shutdown HTTP/1.1\r\nHost: localhost\r\n\r\n");
            }
        }
        for h in self.workers.lock().unwrap().drain(..) {
            let _ = h.join();
        }
        self.coordinator.final_save();
        for h in self.background.lock().unwrap().drain(..) {
            let _ = h.join();
        }
    }
}

impl Drop for CoordinatorHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub fn start_coordinator(opts: CoordinatorOptions) -> anyhow::Result<Arc<CoordinatorHandle>> {
    let registry = Arc::new(Registry::with_defaults());
    let queue = Arc::new(Queue::new());
    queue.set_allow_same_ip_replicate(opts.allow_same_ip_replicate);
    {
        let reg = Arc::clone(&registry);
        queue.set_replica_verifier(Box::new(move |kind, a, b| match reg.get(kind) {
            Some(p) => p.verify_replica(a, b),
            None => a == b,
        }));
    }
    let mut instance_bytes = [0u8; 4];
    let _ = getrandom::getrandom(&mut instance_bytes);
    let coordinator = Arc::new(Coordinator {
        queue,
        registry,
        keys: Enforcer::new(&opts.key),
        opts: opts.clone(),
        cache: Arc::new(ArtifactCache::default()),
        settled: Mutex::new(HashSet::new()),
        instance_id: vibe_local_core::util::hex_encode(&instance_bytes),
        boot: Instant::now(),
    });

    // Restore previous state if configured.
    if let Some(path) = &opts.state_path {
        match persist::load(path) {
            Ok(Some(mut snap)) => {
                snap.jobs
                    .retain(|rec| coordinator.registry.get(&rec.job.kind).is_some());
                coordinator.queue.restore(&snap);
                let profiles: Vec<crate::scheduler::SpeedProfile> = snap
                    .learned_speeds
                    .iter()
                    .map(|ls| crate::scheduler::SpeedProfile {
                        node_uuid: ls.node_uuid.clone(),
                        name: ls.name.clone(),
                        factor: ls.factor,
                        n: ls.n,
                        preset_speeds: ls.preset_speeds.clone(),
                    })
                    .collect();
                coordinator.queue.seed_learn_profiles(profiles);
                for rec in &snap.jobs {
                    if let Some(data) = &rec.artifact {
                        if data.is_empty() {
                            continue;
                        }
                        let art = coordinator.cache.get_or_insert(&rec.job.id);
                        *art.data.lock().unwrap() = Some(data.clone());
                        *art.ctype.lock().unwrap() = rec.artifact_type.clone();
                        *art.ext.lock().unwrap() = rec.artifact_ext.clone();
                        art.settle();
                        coordinator.cache.touch(&rec.job.id);
                    }
                }
            }
            Ok(None) => {}
            Err(e) => eprintln!("[persist] continuing WITHOUT previous state: {e}"),
        }
    }

    let bind: SocketAddr = opts
        .addr
        .parse()
        .map_err(|e| anyhow::anyhow!("bad --addr {}: {e}", opts.addr))?;
    let server = Server::http(bind).map_err(|e| anyhow::anyhow!("cannot bind {bind}: {e}"))?;
    let port = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        // The coordinator always binds an IP socket; this arm exists so
        // the match is exhaustive on Unix (tiny_http has a Unix variant).
        _ => 0,
    };
    let server = Arc::new(server);
    let stop = Arc::new(AtomicBool::new(false));

    let mut workers = Vec::new();
    for i in 0..WORKER_THREADS {
        let server = Arc::clone(&server);
        let coord = Arc::clone(&coordinator);
        let stop = Arc::clone(&stop);
        workers.push(
            std::thread::Builder::new()
                .name(format!("grid-http-{i}"))
                .spawn(move || {
                    while !stop.load(Ordering::SeqCst) {
                        match server.recv() {
                            Ok(req) => handle_request(req, &coord),
                            Err(_) => break,
                        }
                    }
                })?,
        );
    }

    // Reaper + metrics + persistence background loops.
    let mut background = Vec::new();
    {
        let coord = Arc::clone(&coordinator);
        let stop = Arc::clone(&stop);
        background.push(
            std::thread::Builder::new()
                .name("grid-reaper".into())
                .spawn(move || {
                    let mut last_persist = Instant::now();
                    while !stop.load(Ordering::SeqCst) {
                        std::thread::sleep(Duration::from_secs(1));
                        coord.queue.reap();
                        let st = coord.queue.snapshot();
                        metrics::publish_gauges(&st);
                        metrics::set_requeued(coord.queue.requeued_total());
                        metrics::set_unsupported(coord.queue.unsupported_total());
                        coord.count_new_outcomes(&st.jobs);
                        if coord.opts.state_path.is_some()
                            && last_persist.elapsed() >= PERSIST_EVERY
                        {
                            coord.save_now();
                            last_persist = Instant::now();
                        }
                    }
                })?,
        );
    }

    // Best-effort LAN discovery responder (UDP :20132).
    if let Some(h) = crate::grid_discovery::spawn_responder(port, Arc::clone(&stop)) {
        background.push(h);
    }

    if coordinator.keys.enabled() {
        eprintln!(
            "[coordinator] fleet secret: ENABLED — mutations + sensitive artifacts need {}",
            auth::HEADER
        );
    } else {
        eprintln!(
            "[coordinator] fleet secret: DISABLED — anyone on the network can join or submit"
        );
    }
    eprintln!(
        "[coordinator] Vibe Grid listening on {} (port {port})",
        opts.addr
    );

    Ok(Arc::new(CoordinatorHandle {
        port,
        coordinator,
        stop,
        workers: Mutex::new(workers),
        background: Mutex::new(background),
    }))
}

impl Coordinator {
    fn count_new_outcomes(&self, jobs: &[crate::proto::Job]) {
        let mut settled = self.settled.lock().unwrap();
        for j in jobs {
            if matches!(j.status.as_str(), JOB_DONE | JOB_FAILED | JOB_CANCELLED)
                && !settled.contains(&j.id)
            {
                settled.insert(j.id.clone());
                metrics::job_settled(&j.status);
            }
        }
    }

    fn artifact_disk_dir(&self) -> Option<PathBuf> {
        self.opts.state_path.as_ref().map(|p| {
            p.parent()
                .unwrap_or(std::path::Path::new("."))
                .join("vibe-grid-artifacts")
        })
    }

    fn save_now(&self) {
        let Some(path) = &self.opts.state_path else {
            return;
        };
        let mut snap = self.queue.save();
        for rec in &mut snap.jobs {
            if let Some(art) = self.cache.get(&rec.job.id)
                && let Some(data) = art.data.lock().unwrap().clone()
                && !data.is_empty()
            {
                rec.artifact = Some(data);
                rec.artifact_type = art.ctype.lock().unwrap().clone();
                rec.artifact_ext = art.ext.lock().unwrap().clone();
            }
        }
        if let Err(e) = persist::save(path, &snap) {
            eprintln!("[persist] save failed: {e}");
        }
    }

    fn final_save(&self) {
        if self.opts.state_path.is_some() {
            self.save_now();
            eprintln!("[persist] final snapshot saved");
        }
    }

    fn artifact(&self, job_id: &str) -> Option<Arc<ArtifactState>> {
        let art = self.cache.get_or_insert(job_id);
        self.cache.touch(job_id);
        // Exactly one builder per artifact (compare-exchange on building).
        if !art.wait(Duration::from_millis(0))
            && art
                .building
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            let queue = Arc::clone(&self.queue);
            let registry = Arc::clone(&self.registry);
            let cache = Arc::clone(&self.cache);
            let art2 = Arc::clone(&art);
            let job_id = job_id.to_string();
            let disk_dir = self.artifact_disk_dir();
            let spawned = std::thread::Builder::new()
                .name(format!("grid-artifact-{job_id}"))
                .spawn(move || {
                    build_artifact(queue, registry, cache, &job_id, &art2, disk_dir);
                });
            if spawned.is_err() {
                // Release the builder flag or every future request for this
                // artifact waits out the full timeout with no builder.
                art.building.store(false, Ordering::SeqCst);
                return None;
            }
        }
        if art.wait(ARTIFACT_WAIT_CLIENT) {
            Some(art)
        } else {
            None
        }
    }

    fn settle_artifact(&self, job_id: &str) {
        if let Some(art) = self.cache.get(job_id)
            && art.data.lock().unwrap().is_none()
        {
            art.settle();
        }
    }
}

fn build_artifact(
    queue: Arc<Queue>,
    registry: Arc<Registry>,
    cache: Arc<ArtifactCache>,
    job_id: &str,
    art: &Arc<ArtifactState>,
    disk_dir: Option<PathBuf>,
) {
    let deadline = Instant::now() + ARTIFACT_WAIT_TIMEOUT;
    let settled = loop {
        if Instant::now() > deadline {
            eprintln!("[artifact] {job_id} gave up waiting for job to settle");
            art.settle();
            return;
        }
        if let Some((j, order, payloads)) = queue.assemble_inputs(job_id) {
            match j.status.as_str() {
                JOB_DONE => break (j, order, payloads),
                JOB_FAILED | JOB_CANCELLED => {
                    art.settle();
                    return;
                }
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(300));
    };
    let (job, order, payloads) = settled;

    let load_disk = |dir: &std::path::Path| -> Option<(Vec<u8>, String, String)> {
        let meta_raw = std::fs::read_to_string(dir.join(format!("{job_id}.meta.json"))).ok()?;
        let meta: serde_json::Value = serde_json::from_str(&meta_raw).ok()?;
        let data = std::fs::read(dir.join(format!("{job_id}.bin"))).ok()?;
        Some((
            data,
            meta["content_type"]
                .as_str()
                .unwrap_or("application/octet-stream")
                .to_string(),
            meta["ext"].as_str().unwrap_or("").to_string(),
        ))
    };

    if queue.is_reclaimed(job_id) || payloads.len() < order.len() {
        if let Some(dir) = &disk_dir
            && let Some((data, ctype, ext)) = load_disk(dir)
        {
            *art.data.lock().unwrap() = Some(data);
            *art.ctype.lock().unwrap() = ctype;
            *art.ext.lock().unwrap() = ext;
            art.settle();
            cache.touch(job_id);
            return;
        }
        if queue.is_reclaimed(job_id) {
            art.evicted.store(true, Ordering::SeqCst);
        }
        art.settle();
        return;
    }

    let Some(preset) = registry.get(&job.kind) else {
        art.settle();
        return;
    };
    match preset.aggregate(&job, &order, &payloads) {
        Ok((data, ctype, ext)) => {
            if let Some(dir) = &disk_dir
                && std::fs::create_dir_all(dir).is_ok()
                && !data.is_empty()
            {
                let meta = json!({"content_type": ctype, "ext": ext});
                let _ = std::fs::write(dir.join(format!("{job_id}.meta.json")), meta.to_string());
                let _ = std::fs::write(dir.join(format!("{job_id}.bin")), &data);
            }
            *art.data.lock().unwrap() = Some(data.clone());
            *art.ctype.lock().unwrap() = ctype.clone();
            *art.ext.lock().unwrap() = ext.clone();
            art.settle();
            cache.touch(job_id);
            queue.reclaim_job_payloads(job_id);
            eprintln!(
                "[artifact] {job_id} assembled ({} bytes {ctype})",
                data.len()
            );
        }
        Err(e) => {
            eprintln!("[artifact] {job_id} assembly failed: {e}");
            art.settle();
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid header")
}

type HttpResp = Response<std::io::Cursor<Vec<u8>>>;

fn json_resp(status: u16, value: serde_json::Value) -> HttpResp {
    Response::from_data(serde_json::to_vec(&value).unwrap_or_default())
        .with_status_code(status)
        .with_header(header("Content-Type", "application/json"))
        .with_header(header("Content-Security-Policy", CSP))
}

fn err_resp(status: u16, msg: &str) -> HttpResp {
    json_resp(status, json!({"error": msg}))
}

fn unauthorized() -> HttpResp {
    json_resp(401, json!({"error": auth::UNAUTHORIZED_BODY}))
        .with_header(header("WWW-Authenticate", "Key realm=\"compost\""))
}

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn query_param(url: &str, key: &str) -> Option<String> {
    let query = url.split_once('?')?.1;
    for pair in query.split('&') {
        // Valueless pairs (?flag&key=v) must not abort the scan.
        let Some((k, v)) = pair.split_once('=') else {
            continue;
        };
        if urldecode(k) == key {
            return Some(urldecode(v));
        }
    }
    None
}

fn req_header(req: &Request, name: &str) -> Option<String> {
    req.headers()
        .iter()
        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|h| h.value.as_str().to_string())
}

fn client_ip(req: &Request) -> String {
    req.remote_addr()
        .map(|a| a.ip().to_string())
        .unwrap_or_default()
}

/// DNS-rebinding defense (same rule as Vibe-Link): Host must be an IP
/// literal, localhost, or this machine's hostname. Protects open fleets
/// (no --key) from remote web pages resolving their domain to the LAN IP.
fn host_ok(req: &Request) -> bool {
    let Some(host) = req_header(req, "Host") else {
        return false;
    };
    let host = host.trim();
    let name = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else if host.matches(':').count() > 1 {
        host
    } else {
        host.split(':').next().unwrap_or(host)
    };
    let name = name.trim_end_matches('.').to_lowercase();
    if name.is_empty() {
        return false;
    }
    if name.parse::<std::net::IpAddr>().is_ok() || name == "localhost" {
        return true;
    }
    let hn = crate::node::hostname().to_lowercase();
    name == hn || name == format!("{hn}.local")
}

/// Read the body, transparently decompressing gzip under a 9MiB cap.
fn read_body(req: &mut Request) -> Result<Vec<u8>, (u16, String)> {
    let mut raw = Vec::new();
    req.as_reader()
        .take(MAX_BODY_BYTES as u64 + 1)
        .read_to_end(&mut raw)
        .map_err(|e| (400u16, format!("read error: {e}")))?;
    if raw.len() > MAX_BODY_BYTES {
        return Err((413, format!("body exceeds {MAX_BODY_BYTES} bytes")));
    }
    let gzip = req_header(req, "Content-Encoding").is_some_and(|v| v.eq_ignore_ascii_case("gzip"));
    if !gzip {
        return Ok(raw);
    }
    let decoder = flate2::read::GzDecoder::new(&raw[..]);
    let mut plain = Vec::new();
    decoder
        .take(MAX_BODY_BYTES as u64 + 1)
        .read_to_end(&mut plain)
        .map_err(|_| (400u16, "bad gzip body".to_string()))?;
    if plain.len() > MAX_BODY_BYTES {
        return Err((
            413,
            format!("decompressed body exceeds {MAX_BODY_BYTES} bytes"),
        ));
    }
    Ok(plain)
}

fn sanitize_name(raw: &str) -> String {
    let trimmed: String = raw.trim().chars().take(64).collect();
    trimmed
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == ' ' || *c == '-' || *c == '_' || *c == '.')
        .collect::<String>()
        .trim()
        .to_string()
}

fn sanitize_uuid(raw: &str) -> String {
    raw.trim()
        .chars()
        .take(64)
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect()
}

fn handle_request(mut req: Request, coord: &Arc<Coordinator>) {
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("/").to_string();
    let method = req.method().clone();

    if !host_ok(&req) {
        let _ = req.respond(err_resp(
            421,
            "Host header must name this machine (IP, localhost or hostname)",
        ));
        return;
    }

    // Body limits + gzip apply to every route.
    let body = if matches!(method, Method::Post) {
        match read_body(&mut req) {
            Ok(b) => b,
            Err((code, msg)) => {
                let _ = req.respond(err_resp(code, &msg));
                return;
            }
        }
    } else {
        Vec::new()
    };

    let key_header = req_header(&req, auth::HEADER);
    let key_query = query_param(&url, auth::QUERY_PARAM);
    let authed = coord
        .keys
        .authorize(key_header.as_deref(), key_query.as_deref(), true);
    let authed_header_only = coord.keys.authorize(key_header.as_deref(), None, false);

    let resp = route(
        coord,
        &method,
        &path,
        &body,
        authed,
        authed_header_only,
        &client_ip(&req),
    );
    let _ = req.respond(resp);
}

#[allow(clippy::too_many_arguments)]
fn route(
    coord: &Arc<Coordinator>,
    method: &Method,
    path: &str,
    body: &[u8],
    authed: bool,
    authed_header_only: bool,
    ip: &str,
) -> HttpResp {
    // Agent API: header-key gated when a fleet secret is configured.
    let agent_route = matches!(
        path,
        "/api/v1/join" | "/api/v1/poll" | "/api/v1/heartbeat" | "/api/v1/result"
    );
    let client_mutation = matches!(path, "/api/v1/jobs") || path.ends_with("/cancel");
    if (agent_route || client_mutation)
        && *method == Method::Post
        && coord.keys.enabled()
        && !authed_header_only
    {
        return unauthorized();
    }

    match (method.clone(), path) {
        (Method::Post, "/api/v1/join") => {
            metrics::count_route("join");
            let Ok(req) = serde_json::from_slice::<JoinRequest>(body) else {
                return err_resp(400, "bad json");
            };
            let name = sanitize_name(&req.name);
            if name.is_empty() {
                return err_resp(400, "name must contain visible characters");
            }
            let uuid = sanitize_uuid(&req.node_uuid);
            let id = coord.queue.join_with_meta(&name, &uuid, ip, req.caps);
            json_resp(
                200,
                serde_json::to_value(JoinResponse { node_id: id }).unwrap(),
            )
        }
        (Method::Post, "/api/v1/poll") => {
            metrics::count_route("poll");
            let Ok(req) = serde_json::from_slice::<PollRequest>(body) else {
                return err_resp(400, "bad json");
            };
            let name = {
                let n = sanitize_name(&req.name);
                if n.is_empty() { "agent".to_string() } else { n }
            };
            let uuid = sanitize_uuid(&req.node_uuid);
            if !coord
                .queue
                .touch_with_meta(&req.node_id, &name, &uuid, ip, req.caps)
            {
                return err_resp(404, "unknown node");
            }
            coord.queue.note_running(&req.node_id, req.running);
            let tasks =
                coord
                    .queue
                    .lease_batch(&req.node_id, &req.paused, req.running, req.workers);
            let mut resp = PollResponse {
                task: None,
                tasks: Vec::new(),
                poll_after_ms: 2000,
            };
            if tasks.len() == 1 {
                resp.task = Some(tasks[0].clone());
            } else if tasks.len() > 1 {
                resp.tasks = tasks;
            }
            json_resp(200, serde_json::to_value(resp).unwrap())
        }
        (Method::Post, "/api/v1/heartbeat") => {
            let Ok(req) = serde_json::from_slice::<HeartbeatRequest>(body) else {
                return err_resp(400, "bad json");
            };
            if !coord.queue.node_exists(&req.node_id) {
                return err_resp(404, "unknown node");
            }
            let extended = coord.queue.renew_leases(&req.node_id, &req.tasks);
            json_resp(
                200,
                serde_json::to_value(HeartbeatResponse { extended }).unwrap(),
            )
        }
        (Method::Post, "/api/v1/result") => {
            metrics::count_route("result");
            metrics::count_result();
            let Ok(mut res) = serde_json::from_slice::<ResultMsg>(body) else {
                return err_resp(400, "bad json");
            };
            if res.error.is_empty()
                && let Some(task) = coord.queue.task_by_id(&res.task_id)
                && let Some(preset) = coord.registry.get(&task.kind)
            {
                let expected = preset.result_size(&task);
                let got = res.payload().len();
                if expected > 0 && got < expected {
                    eprintln!(
                        "[result] {}: payload {got} bytes < expected {expected} — rejecting",
                        res.task_id
                    );
                    res.error = format!("payload too small ({got} < {expected} bytes)");
                    res.data = None;
                    res.pixels = None;
                }
            }
            coord.queue.result_in(res);
            json_resp(200, json!({"status": "ok"}))
        }
        (Method::Post, "/api/v1/jobs") => {
            metrics::count_route("submit");
            let Ok(req) = serde_json::from_slice::<SubmitJobRequest>(body) else {
                return err_resp(400, "bad json");
            };
            let Some(preset) = coord.registry.get(&req.kind) else {
                return err_resp(
                    400,
                    &format!(
                        "unknown kind {}; known: {:?}",
                        req.kind,
                        coord.registry.names()
                    ),
                );
            };
            for (k, v) in &req.params {
                if !v.is_finite() {
                    return err_resp(400, &format!("param {k:?} must be a finite number"));
                }
            }
            let mut job = match preset.validate(&req.params) {
                Ok(j) => j,
                Err(e) => return err_resp(400, &e.to_string()),
            };
            job.replicate = req.replicate;
            job.priority = req.priority.clamp(0, 10);
            if let Some(cap) = &req.required_capability {
                if !crate::proto::CAPABILITIES.contains(&cap.as_str()) {
                    return err_resp(
                        400,
                        &format!(
                            "unknown capability {cap:?}; known: {:?}",
                            crate::proto::CAPABILITIES
                        ),
                    );
                }
                job.required_capability = Some(cap.clone());
            }
            job.meta = req.meta.clone();
            if !req.mode.is_empty() {
                job.meta.entry("mode".into()).or_insert(req.mode.clone());
            }
            if req.kind == TASK_WORDCOUNT
                || req.kind == TASK_LLM_BATCH
                || req.kind == TASK_EMBED_BATCH
            {
                let Some(input) = &req.input else {
                    return err_resp(
                        400,
                        &format!("{} requires non-empty input (use --input FILE)", req.kind),
                    );
                };
                if input.is_empty() {
                    return err_resp(
                        400,
                        &format!("{} requires non-empty input (use --input FILE)", req.kind),
                    );
                }
                if input.len() > MAX_JOB_INPUT {
                    return err_resp(413, &format!("input exceeds {MAX_JOB_INPUT} bytes"));
                }
                job.input = Some(input.clone());
            }
            // Fleet-aware MoE planning: inject the live shard map at submit.
            if req.kind == TASK_MOE_DISTRIBUTED {
                let model_id = job
                    .meta
                    .get("model_id")
                    .cloned()
                    .unwrap_or_else(|| "OLMoE-1B-7B-0924-Instruct".into());
                let Some(spec) = moe_grid::find_model(&model_id) else {
                    return err_resp(400, &format!("moe_distributed: unknown model {model_id:?}"));
                };
                let quant = job
                    .meta
                    .get("quant")
                    .cloned()
                    .unwrap_or_else(|| "Q4_K_M".into());
                let mut nodes = vec![moe_grid::local_grid_node()];
                let state = coord.queue.snapshot();
                nodes.extend(moe_grid::grid_nodes_from_state(&state, ""));
                let ctx = job.width.max(512) as u64;
                match moe_grid::plan_moe(&spec, &nodes, &quant, ctx, 5.0) {
                    Ok(plan) => {
                        job.meta
                            .insert("plan".into(), serde_json::to_string(&plan).unwrap());
                        job.meta.insert("model_id".into(), spec.model_id.clone());
                    }
                    Err(e) => return err_resp(400, &format!("moe_distributed: {e}")),
                }
            }
            if coord.queue.active_jobs() >= MAX_CONCURRENT_JOBS {
                return err_resp(
                    429,
                    &format!(
                        "{MAX_CONCURRENT_JOBS} jobs already active (max {MAX_CONCURRENT_JOBS}) — cancel one first"
                    ),
                );
            }
            job.id = coord.queue.next_job_id();
            let tasks = match preset.split(&job) {
                Ok(t) => t,
                Err(e) => return err_resp(400, &e.to_string()),
            };
            let stored = coord.queue.add_job(job, tasks);
            json_resp(
                200,
                serde_json::to_value(SubmitJobResponse { job: stored }).unwrap(),
            )
        }
        (Method::Post, p) if p.ends_with("/cancel") => {
            metrics::count_route("cancel");
            let job_id = p
                .trim_start_matches("/api/v1/jobs/")
                .trim_end_matches("/cancel");
            if !coord.queue.cancel_job(job_id) {
                return err_resp(404, &format!("unknown job {job_id}"));
            }
            coord.settle_artifact(job_id);
            json_resp(200, json!({"status": "cancelled", "job": job_id}))
        }
        (Method::Get, "/api/v1/capabilities") => json_resp(200, coord.queue.capability_directory()),
        (Method::Get, "/api/v1/state") => {
            metrics::count_route("state");
            let mut st = coord.queue.snapshot();
            st.workloads = coord.registry.info();
            annotate_eta(&mut st);
            if coord.keys.enabled() && !authed {
                for j in &mut st.jobs {
                    j.input = None;
                    if is_sensitive(&j.kind) || coord.opts.protect_artifacts {
                        if !j.error.is_empty() {
                            j.error = "(redacted — authenticate to view)".into();
                        }
                        if !j.meta.is_empty() {
                            let mode = j.meta.get("mode").cloned();
                            j.meta.clear();
                            if let Some(m) = mode {
                                j.meta.insert("mode".into(), m);
                            }
                        }
                    }
                }
            }
            json_resp(200, serde_json::to_value(st).unwrap())
        }
        (Method::Get, p) if p.ends_with("/tiles") => {
            metrics::count_route("tiles");
            let job_id = p
                .trim_start_matches("/api/v1/jobs/")
                .trim_end_matches("/tiles");
            if !artifact_auth_ok(coord, job_id, authed) {
                return unauthorized();
            }
            match coord.queue.tile_states(job_id) {
                Some(states) => json_resp(200, serde_json::to_value(states).unwrap()),
                None => err_resp(404, "unknown job"),
            }
        }
        (Method::Get, p) if p.ends_with("/artifact") => {
            metrics::count_route("artifact");
            let job_id = p
                .trim_start_matches("/api/v1/jobs/")
                .trim_end_matches("/artifact");
            if !artifact_auth_ok(coord, job_id, authed) {
                return unauthorized();
            }
            if !coord.queue.job_exists(job_id) {
                return err_resp(404, "unknown job");
            }
            let Some(art) = coord.artifact(job_id) else {
                return err_resp(102, "job not finished");
            };
            if art.evicted.load(Ordering::SeqCst) {
                return err_resp(
                    410,
                    "artifact evicted from cache (enable --state to keep artifacts on disk)",
                );
            }
            let data = art.data.lock().unwrap().clone();
            let Some(data) = data else {
                return err_resp(409, "no artifact for this job (failed or cancelled)");
            };
            let ctype = art.ctype.lock().unwrap().clone();
            let ext = art.ext.lock().unwrap().clone();
            Response::from_data(data)
                .with_status_code(200)
                .with_header(header("Content-Type", &ctype))
                .with_header(header(
                    "Content-Disposition",
                    &format!("attachment; filename={job_id}{ext}"),
                ))
                .with_header(header("Content-Security-Policy", CSP))
        }
        (Method::Get, "/healthz") => {
            let st = coord.queue.snapshot();
            let online = st.nodes.iter().filter(|n| n.online).count();
            let llm = st
                .nodes
                .iter()
                .filter(|n| n.online && n.caps.has_local_llm)
                .count();
            let mut resp = json!({
                "status": "ok",
                "version": crate::proto::VERSION,
                "instance": coord.instance_id,
                "uptime_s": coord.boot.elapsed().as_secs(),
                "nodes_total": st.nodes.len(),
                "nodes_online": online,
                "nodes_with_llm": llm,
                "jobs_active": coord.queue.active_jobs(),
                "persistence": coord.opts.state_path.is_some(),
            });
            if let Some(p) = &coord.opts.state_path
                && (!coord.keys.enabled() || !coord.opts.protect_artifacts || authed)
            {
                resp["state_path"] = json!(p.to_string_lossy());
            }
            json_resp(200, resp)
        }
        (Method::Get, "/metrics") => Response::from_data(metrics::render().into_bytes())
            .with_status_code(200)
            .with_header(header("Content-Type", "text/plain; version=0.0.4")),
        (Method::Get, "/") | (Method::Get, "/index.html") => {
            Response::from_data(DASHBOARD_HTML.as_bytes().to_vec())
                .with_status_code(200)
                .with_header(header("Content-Type", "text/html; charset=utf-8"))
                .with_header(header("Content-Security-Policy", CSP))
        }
        (Method::Get, "/dash.css") => Response::from_data(DASHBOARD_CSS.as_bytes().to_vec())
            .with_status_code(200)
            .with_header(header("Content-Type", "text/css; charset=utf-8"))
            .with_header(header("Content-Security-Policy", CSP)),
        (Method::Get, "/dash.js") => Response::from_data(DASHBOARD_JS.as_bytes().to_vec())
            .with_status_code(200)
            .with_header(header("Content-Type", "text/javascript; charset=utf-8"))
            .with_header(header("Content-Security-Policy", CSP)),
        _ => err_resp(404, "not found"),
    }
}

fn is_sensitive(kind: &str) -> bool {
    kind == TASK_WORDCOUNT || kind == TASK_LLM_BATCH || kind == TASK_EMBED_BATCH
}

/// Artifact/tile auth: checked BEFORE JobExists or LRU insertion.
fn artifact_auth_ok(coord: &Coordinator, job_id: &str, authed: bool) -> bool {
    if !coord.keys.enabled() {
        return true;
    }
    if coord.opts.protect_artifacts {
        return authed;
    }
    if let Some(kind) = coord.queue.job_kind(job_id)
        && is_sensitive(&kind)
    {
        return authed;
    }
    true
}

/// ETA from live fleet throughput + plain-language Waiting reasons.
pub fn annotate_eta(st: &mut StateResponse) {
    let mut rate = 0.0;
    let mut online = 0;
    let mut working = 0;
    let mut llm_ready = 0;
    let mut embed_ready = 0;
    let mut paused_reasons: Vec<String> = Vec::new();
    for n in &st.nodes {
        if !n.online {
            continue;
        }
        online += 1;
        if n.rate > 0.0 {
            rate += n.rate;
            working += 1;
        }
        if n.caps.has_local_llm && n.paused.is_empty() {
            llm_ready += 1;
        }
        if n.caps.capabilities.iter().any(|c| c == "embed_rag") && n.paused.is_empty() {
            embed_ready += 1;
        }
        if !n.paused.is_empty() && !paused_reasons.contains(&n.paused) {
            paused_reasons.push(n.paused.clone());
        }
    }
    for j in &mut st.jobs {
        if j.status != JOB_RUNNING && j.status != JOB_QUEUED {
            continue;
        }
        // Recomputed fresh per snapshot (Go semantics: Waiting is not stored).
        j.waiting.clear();
        j.eta_seconds = 0;
        let remaining = j.tasks_total.saturating_sub(j.tasks_done);
        if remaining > 0 && rate > 0.0 {
            j.eta_seconds = (remaining as f64 / rate) as i64;
        }
        if remaining == 0 {
            continue;
        }
        let verify = j
            .meta
            .get("mode")
            .is_some_and(|m| m.eq_ignore_ascii_case("verify"))
            || j.params.get("verify").copied().unwrap_or(0.0) > 0.0;
        if j.kind == TASK_LLM_BATCH && !verify && online > 0 && llm_ready == 0 && j.tasks_done == 0
        {
            j.waiting = "no online agents have Vibe Local (:20129) available".into();
            continue;
        }
        if j.kind == TASK_EMBED_BATCH
            && !verify
            && online > 0
            && embed_ready == 0
            && j.tasks_done == 0
        {
            j.waiting = "no online agents advertise the embed_rag capability".into();
            continue;
        }
        if rate > 0.0 {
            continue;
        }
        j.waiting = if online == 0 && st.nodes.is_empty() {
            "no agents have joined yet — start one with `vibe-local cluster agent`".into()
        } else if online == 0 {
            "all agents offline — waiting for the fleet to come back".into()
        } else if working == 0 && !paused_reasons.is_empty() {
            paused_reasons.sort();
            let shown: Vec<String> = paused_reasons.iter().take(2).cloned().collect();
            let extra = if paused_reasons.len() > 2 {
                format!(" (+{} more)", paused_reasons.len() - 2)
            } else {
                String::new()
            };
            format!("all {online} agents paused: {}{extra}", shown.join("; "))
        } else {
            "waiting for agents to pick up work".into()
        };
    }
}

/// Minimal embedded status dashboard (full UI lives in leaderboard-web).
const DASHBOARD_HTML: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vibe Grid Coordinator</title><link rel="stylesheet" href="/dash.css"></head>
<body><main id="app"><h1>Vibe Grid</h1><p id="status">Loading…</p>
<section><h2>Fleet</h2><div id="nodes"></div></section>
<section><h2>Jobs</h2><div id="jobs"></div></section>
<p><a href="/healthz">healthz</a> · <a href="/metrics">metrics</a> · full dashboard: Vibe Local leaderboard → Vibe Grid</p>
</main><script src="/dash.js"></script></body></html>"#;

const DASHBOARD_CSS: &str = r#"body{background:#090A0F;color:#E6E9F2;font-family:system-ui,sans-serif;margin:0;padding:2rem}
main{max-width:960px;margin:0 auto}h1{color:#7C5CFF}h2{color:#22D3EE;font-size:1rem;text-transform:uppercase;letter-spacing:.08em}
.card{background:#131620;border:1px solid #232838;border-radius:8px;padding:.75rem 1rem;margin:.5rem 0}
.badge{display:inline-block;padding:.1rem .5rem;border-radius:4px;font-size:.75rem;margin-right:.4rem}
.online{background:#123524;color:#4ade80}.offline{background:#3a1d1d;color:#f87171}.paused{background:#3a2f1d;color:#fbbf24}
a{color:#22D3EE}#status{color:#9aa3b8}"#;

const DASHBOARD_JS: &str = r#"// Safe DOM only: no innerHTML, no inline handlers (strict CSP).
const key = localStorage.getItem('vibe-grid-key') || '';
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}
async function refresh() {
  const status = document.getElementById('status');
  try {
    const headers = key ? { 'X-Compost-Key': key } : {};
    const res = await fetch('/api/v1/state', { headers, cache: 'no-store' });
    const st = await res.json();
    status.textContent = `${st.nodes.length} nodes · ${st.jobs.length} jobs · ${st.workloads ? st.workloads.length : 0} workloads`;
    const nodes = document.getElementById('nodes');
    nodes.replaceChildren();
    for (const n of st.nodes) {
      const card = el('div', 'card');
      card.appendChild(el('strong', null, n.name));
      card.appendChild(el('span', n.online ? 'badge online' : 'badge offline', n.online ? 'online' : 'offline'));
      if (n.paused) card.appendChild(el('span', 'badge paused', n.paused));
      card.appendChild(el('div', null, `${n.caps.os}/${n.caps.arch} · ${n.caps.cores} cores · ${n.caps.mem_mb} MB · speed ×${(n.speed_factor || 0).toFixed(2)} · done ${n.done}${n.caps.has_local_llm ? ' · LLM :20129' : ''}`));
      nodes.appendChild(card);
    }
    const jobs = document.getElementById('jobs');
    jobs.replaceChildren();
    for (const j of st.jobs) {
      const card = el('div', 'card');
      card.appendChild(el('strong', null, `${j.id} · ${j.kind}`));
      card.appendChild(el('span', 'badge ' + (j.status === 'done' ? 'online' : j.status === 'failed' ? 'offline' : 'paused'), j.status));
      card.appendChild(el('div', null, `${j.tasks_done}/${j.tasks_total} tasks${j.eta_seconds ? ' · ETA ' + j.eta_seconds + 's' : ''}${j.waiting ? ' · ' + j.waiting : ''}${j.error ? ' · ' + j.error : ''}`));
      jobs.appendChild(card);
    }
  } catch (e) {
    status.textContent = 'coordinator unreachable: ' + e.message;
  }
}
refresh();
setInterval(refresh, 2000);
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::Caps;

    fn opts(dir: &std::path::Path) -> CoordinatorOptions {
        CoordinatorOptions {
            addr: "127.0.0.1:0".into(),
            key: String::new(),
            protect_artifacts: false,
            allow_same_ip_replicate: true,
            state_path: Some(dir.join("state.json.gz")),
        }
    }

    fn post(base: &str, path: &str, body: serde_json::Value, key: Option<&str>) -> ureq::Response {
        let mut req = ureq::post(&format!("{base}{path}"));
        if let Some(k) = key {
            req = req.set(auth::HEADER, k);
        }
        req.send_json(body).unwrap()
    }

    #[test]
    fn query_param_percent_decodes_keys_and_values() {
        assert_eq!(
            query_param("/x?flag&key=a%20b+c", "key").as_deref(),
            Some("a b c")
        );
        assert_eq!(
            query_param("/x?%6bey=%2Ftmp%2Fa", "key").as_deref(),
            Some("/tmp/a")
        );
        assert_eq!(query_param("/x?key=100%25", "key").as_deref(), Some("100%"));
        assert_eq!(
            query_param("/x?key=bad%ZZ", "key").as_deref(),
            Some("bad%ZZ")
        );
        assert_eq!(query_param("/x?flag&key=v", "flag"), None);
    }

    #[test]
    fn full_primes_pipeline_with_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let handle = start_coordinator(opts(dir.path())).unwrap();
        let base = format!("http://127.0.0.1:{}", handle.port);
        // Join an agent.
        let join: serde_json::Value = post(&base, "/api/v1/join", json!({
            "name": "test-pc", "node_uuid": "uuid-test", "caps": {"cores": 4, "mem_mb": 8192, "score": 1.0, "battery_pct": -1, "on_battery": false, "temp_c": -1, "arch": "x86_64", "os": "linux", "version": "0.7.0"}
        }), None).into_json().unwrap();
        let node_id = join["node_id"].as_str().unwrap().to_string();
        // Submit a small primes job.
        let submit: serde_json::Value = post(
            &base,
            "/api/v1/jobs",
            json!({"kind": "primes", "params": {"limit": 100000}}),
            None,
        )
        .into_json()
        .unwrap();
        let job_id = submit["job"]["id"].as_str().unwrap().to_string();
        assert_eq!(submit["job"]["tasks_total"], 1);
        // Poll, execute locally, report.
        let poll: serde_json::Value = post(&base, "/api/v1/poll", json!({"node_id": node_id, "name": "test-pc", "caps": {"cores": 4, "mem_mb": 8192, "score": 1.0, "battery_pct": -1, "on_battery": false, "temp_c": -1, "arch": "x86_64", "os": "linux", "version": "0.7.0"}, "running": 0, "workers": 2}), None).into_json().unwrap();
        let task: crate::proto::Task = serde_json::from_value(poll["task"].clone()).unwrap();
        assert_eq!(task.lease_epoch, 1);
        let preset = crate::workload::primes::Primes;
        use crate::workload::Preset;
        let (data, _) = preset.execute(&task).unwrap();
        use base64::Engine;
        post(
            &base,
            "/api/v1/result",
            json!({"task_id": task.id, "node_id": node_id, "lease_epoch": task.lease_epoch, "data": base64::engine::general_purpose::STANDARD.encode(&data), "millis": 50, "exec_millis": 50}),
            None,
        );
        // Wait for settlement + artifact.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut artifact = None;
        while Instant::now() < deadline {
            let resp = ureq::get(&format!("{base}/api/v1/jobs/{job_id}/artifact")).call();
            if let Ok(r) = resp
                && r.status() == 200
            {
                let mut buf = Vec::new();
                r.into_reader().read_to_end(&mut buf).unwrap();
                artifact = Some(buf);
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let artifact = artifact.expect("artifact must assemble");
        assert_eq!(
            String::from_utf8_lossy(&artifact),
            "primes below 100000: 9592\n"
        );
        // Payloads reclaimed after assembly.
        assert!(handle.coordinator.queue.is_reclaimed(&job_id));
        assert_eq!(
            handle.coordinator.queue.payload_size(&job_id, &task.id),
            Some(4)
        );
        // Tiles report done.
        let tiles: serde_json::Value = ureq::get(&format!("{base}/api/v1/jobs/{job_id}/tiles"))
            .call()
            .unwrap()
            .into_json()
            .unwrap();
        assert_eq!(tiles[0], "done");
        // healthz + metrics.
        let health: serde_json::Value = ureq::get(&format!("{base}/healthz"))
            .call()
            .unwrap()
            .into_json()
            .unwrap();
        assert_eq!(health["status"], "ok");
        assert_eq!(health["nodes_with_llm"], 0);
        // Metrics gauges/counters are published by the 1s reaper loop; poll
        // until this job's outcome is counted (global counters are shared
        // across parallel tests, so assert >= 1).
        let metrics_deadline = Instant::now() + Duration::from_secs(10);
        let mut done_count = 0u64;
        while Instant::now() < metrics_deadline {
            let metrics_text = ureq::get(&format!("{base}/metrics"))
                .call()
                .unwrap()
                .into_string()
                .unwrap();
            if let Some(line) = metrics_text
                .lines()
                .find(|l| l.starts_with("compost_jobs_total{status=\"done\"}"))
            {
                done_count = line.rsplit(' ').next().unwrap().parse().unwrap_or(0);
                if done_count >= 1 {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        assert!(done_count >= 1);
        handle.shutdown();
        // Persistence: snapshot exists and restores into a fresh coordinator.
        let snap = persist::load(&dir.path().join("state.json.gz"))
            .unwrap()
            .unwrap();
        assert_eq!(snap.jobs.len(), 1);
        assert!(snap.jobs[0].artifact.is_some());
        let handle2 = start_coordinator(opts(dir.path())).unwrap();
        let st = handle2.coordinator.queue.snapshot();
        assert_eq!(st.jobs[0].status, JOB_DONE);
        handle2.shutdown();
    }

    #[test]
    fn capability_routing_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let handle = start_coordinator(opts(dir.path())).unwrap();
        let base = format!("http://127.0.0.1:{}", handle.port);
        let plain_caps = json!({"cores": 4, "mem_mb": 8192, "score": 1.0, "battery_pct": -1, "on_battery": false, "temp_c": -1, "arch": "x86_64", "os": "linux", "version": "0.7.0"});
        let mut llm_caps = plain_caps.clone();
        llm_caps["capabilities"] = json!(["llm"]);
        llm_caps["loaded_models"] = json!(["qwen2.5-7b"]);
        let plain: serde_json::Value = post(
            &base,
            "/api/v1/join",
            json!({"name": "plain", "caps": plain_caps}),
            None,
        )
        .into_json()
        .unwrap();
        let llm: serde_json::Value = post(
            &base,
            "/api/v1/join",
            json!({"name": "gpu", "caps": llm_caps}),
            None,
        )
        .into_json()
        .unwrap();
        // Unknown capability labels are rejected at submit.
        let err = ureq::post(&format!("{base}/api/v1/jobs"))
            .send_json(json!({"kind": "primes", "params": {"limit": 1000}, "required_capability": "telepathy"}))
            .unwrap_err();
        match err {
            ureq::Error::Status(code, _) => assert_eq!(code, 400),
            other => panic!("expected HTTP 400, got {other:?}"),
        }
        let submit: serde_json::Value = post(
            &base,
            "/api/v1/jobs",
            json!({"kind": "primes", "params": {"limit": 1000}, "required_capability": "llm"}),
            None,
        )
        .into_json()
        .unwrap();
        assert_eq!(submit["job"]["required_capability"], "llm");
        // The capability-less node never receives the targeted task.
        let p: serde_json::Value = post(
            &base,
            "/api/v1/poll",
            json!({"node_id": plain["node_id"], "name": "plain", "caps": plain_caps, "running": 0, "workers": 1}),
            None,
        )
        .into_json()
        .unwrap();
        assert!(p["task"].is_null());
        let p: serde_json::Value = post(
            &base,
            "/api/v1/poll",
            json!({"node_id": llm["node_id"], "name": "gpu", "caps": llm_caps, "running": 0, "workers": 1}),
            None,
        )
        .into_json()
        .unwrap();
        assert_eq!(p["task"]["kind"], "primes");
        // Capability directory for Vibe Router.
        let dirv: serde_json::Value = ureq::get(&format!("{base}/api/v1/capabilities"))
            .call()
            .unwrap()
            .into_json()
            .unwrap();
        let llm_group = dirv["capabilities"]["llm"].as_array().unwrap();
        assert_eq!(llm_group.len(), 1);
        assert_eq!(llm_group[0]["name"], "gpu");
        assert_eq!(llm_group[0]["loaded_models"][0], "qwen2.5-7b");
        assert!(
            llm_group[0]["lan_endpoint"]
                .as_str()
                .unwrap()
                .starts_with("http://127.0.0.1:")
        );
        assert_eq!(dirv["capabilities"]["compute"].as_array().unwrap().len(), 1);
        handle.shutdown();
    }

    #[test]
    fn embed_batch_capability_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        let handle = start_coordinator(opts(dir.path())).unwrap();
        let base = format!("http://127.0.0.1:{}", handle.port);
        let plain_caps = json!({"cores": 4, "mem_mb": 8192, "score": 1.0, "battery_pct": -1, "on_battery": false, "temp_c": -1, "arch": "x86_64", "os": "linux", "version": "0.7.0"});
        let mut embed_caps = plain_caps.clone();
        embed_caps["capabilities"] = json!(["llm", "embed_rag"]);
        embed_caps["loaded_models"] = json!(["bge-large-en-v1.5-Q8_0.gguf"]);
        embed_caps["has_local_llm"] = json!(true);
        let plain: serde_json::Value = post(
            &base,
            "/api/v1/join",
            json!({"name": "plain", "caps": plain_caps}),
            None,
        )
        .into_json()
        .unwrap();
        let embed: serde_json::Value = post(
            &base,
            "/api/v1/join",
            json!({"name": "embedder", "caps": embed_caps}),
            None,
        )
        .into_json()
        .unwrap();

        // Submit an embed_batch job in verify mode (deterministic builtin
        // embeddings; no live runtime needed). No required_capability is
        // sent — the preset itself stamps embed_rag.
        use base64::Engine;
        let input = base64::engine::general_purpose::STANDARD
            .encode(b"alpha text\nbeta text\ngamma text\n");
        let submit: serde_json::Value = post(
            &base,
            "/api/v1/jobs",
            json!({"kind": "embed_batch", "params": {"batch_size": 2, "verify": 1}, "input": input}),
            None,
        )
        .into_json()
        .unwrap();
        let job_id = submit["job"]["id"].as_str().unwrap().to_string();
        assert_eq!(submit["job"]["kind"], "embed_batch");
        assert_eq!(submit["job"]["required_capability"], "embed_rag");
        assert_eq!(submit["job"]["tasks_total"], 2);

        // The plain node is never leased an embed task.
        let p: serde_json::Value = post(
            &base,
            "/api/v1/poll",
            json!({"node_id": plain["node_id"], "name": "plain", "caps": plain_caps, "running": 0, "workers": 1}),
            None,
        )
        .into_json()
        .unwrap();
        assert!(p["task"].is_null());

        // The embed_rag node executes both chunks and reports results.
        let preset = crate::workload::embed_batch::EmbedBatch;
        use crate::workload::Preset;
        for _ in 0..2 {
            let p: serde_json::Value = post(
                &base,
                "/api/v1/poll",
                json!({"node_id": embed["node_id"], "name": "embedder", "caps": embed_caps, "running": 0, "workers": 1}),
                None,
            )
            .into_json()
            .unwrap();
            let task: crate::proto::Task = serde_json::from_value(p["task"].clone()).unwrap();
            assert_eq!(task.kind, "embed_batch");
            let (data, _) = preset.execute(&task).unwrap();
            post(
                &base,
                "/api/v1/result",
                json!({"task_id": task.id, "node_id": embed["node_id"], "lease_epoch": task.lease_epoch, "data": base64::engine::general_purpose::STANDARD.encode(&data), "millis": 5, "exec_millis": 5}),
                None,
            );
        }

        // Artifact assembles: JSON array, 3 rows in order, full vectors.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut artifact = None;
        while Instant::now() < deadline {
            let resp = ureq::get(&format!("{base}/api/v1/jobs/{job_id}/artifact")).call();
            if let Ok(r) = resp
                && r.status() == 200
            {
                let mut buf = Vec::new();
                r.into_reader().read_to_end(&mut buf).unwrap();
                artifact = Some(buf);
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let artifact = artifact.expect("embed artifact must assemble");
        // Artifact is a JSON array of {index, text_hash, embedding,
        // engine} rows in input order.
        let rows: serde_json::Value = serde_json::from_slice(&artifact).unwrap();
        let rows = rows.as_array().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["index"], 0);
        assert_eq!(rows[2]["index"], 2);
        {
            use sha2::Digest;
            let expect = vibe_local_core::util::hex_encode(&sha2::Sha256::digest(b"alpha text"));
            assert_eq!(rows[0]["text_hash"], expect);
        }
        assert_eq!(rows[0]["engine"], "builtin-microeval");
        assert_eq!(
            rows[0]["embedding"].as_array().unwrap().len(),
            crate::workload::embed_batch::EMBED_DIM
        );
        handle.shutdown();
    }

    #[test]
    fn auth_guards_and_redaction() {
        let dir = tempfile::tempdir().unwrap();
        let mut o = opts(dir.path());
        o.key = "fleet-secret".into();
        o.protect_artifacts = true;
        let handle = start_coordinator(o).unwrap();
        let base = format!("http://127.0.0.1:{}", handle.port);
        // Join without key → 401.
        let err = ureq::post(&format!("{base}/api/v1/join"))
            .send_json(json!({"name": "x", "caps": {}}))
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(401, _)));
        // Join with key → 200.
        let resp = post(
            &base,
            "/api/v1/join",
            json!({"name": "pc", "node_uuid": "u", "caps": {"cores": 1, "mem_mb": 1024, "score": 1.0, "battery_pct": -1, "on_battery": false, "temp_c": -1, "arch": "x", "os": "linux", "version": "0.7.0"}}),
            Some("fleet-secret"),
        );
        assert_eq!(resp.status(), 200);
        // Submit wordcount (sensitive) with key.
        use base64::Engine;
        let input = base64::engine::general_purpose::STANDARD.encode(b"hello world hello");
        let submit: serde_json::Value = post(
            &base,
            "/api/v1/jobs",
            json!({"kind": "wordcount", "input": input}),
            Some("fleet-secret"),
        )
        .into_json()
        .unwrap();
        let job_id = submit["job"]["id"].as_str().unwrap().to_string();
        // Artifact without key → 401 (protect-artifacts), even before job exists check.
        let err = ureq::get(&format!("{base}/api/v1/jobs/{job_id}/artifact"))
            .call()
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(401, _)));
        let err = ureq::get(&format!("{base}/api/v1/jobs/nonexistent/artifact"))
            .call()
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(401, _)));
        // Query-param fallback works.
        let resp = ureq::get(&format!(
            "{base}/api/v1/jobs/{job_id}/tiles?key=fleet-secret"
        ))
        .call()
        .unwrap();
        assert_eq!(resp.status(), 200);
        // State redaction for unauthenticated watchers.
        let st: serde_json::Value = ureq::get(&format!("{base}/api/v1/state"))
            .call()
            .unwrap()
            .into_json()
            .unwrap();
        assert!(
            st["jobs"]
                .as_array()
                .unwrap()
                .iter()
                .all(|j| j.get("input").is_none())
        );
        handle.shutdown();
    }

    #[test]
    fn gzip_bomb_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let handle = start_coordinator(opts(dir.path())).unwrap();
        let base = format!("http://127.0.0.1:{}", handle.port);
        // 12MiB of zeros compresses tiny; decompressed must trip the 9MiB cap.
        use flate2::write::GzEncoder;
        use std::io::Write;
        let mut enc = GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(&vec![0u8; 12 << 20]).unwrap();
        let bomb = enc.finish().unwrap();
        let err = ureq::post(&format!("{base}/api/v1/join"))
            .set("Content-Encoding", "gzip")
            .set("Content-Type", "application/json")
            .send_bytes(&bomb)
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(413, _)));
        // Valid small gzip body passes.
        let body = serde_json::to_vec(&json!({"name": "gz", "caps": {"cores": 1, "mem_mb": 1024, "score": 1.0, "battery_pct": -1, "on_battery": false, "temp_c": -1, "arch": "x", "os": "linux", "version": "0.7.0"}})).unwrap();
        let mut enc = GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&body).unwrap();
        let ok = enc.finish().unwrap();
        let resp = ureq::post(&format!("{base}/api/v1/join"))
            .set("Content-Encoding", "gzip")
            .set("Content-Type", "application/json")
            .send_bytes(&ok)
            .unwrap();
        assert_eq!(resp.status(), 200);
        handle.shutdown();
    }

    #[test]
    fn eta_and_waiting_annotation() {
        let mut st = StateResponse {
            nodes: vec![],
            jobs: vec![crate::proto::Job {
                id: "j1".into(),
                kind: "primes".into(),
                status: JOB_QUEUED.into(),
                tasks_total: 10,
                tasks_done: 0,
                ..Default::default()
            }],
            workloads: vec![],
        };
        annotate_eta(&mut st);
        assert!(st.jobs[0].waiting.contains("no agents have joined"));
        st.nodes.push(crate::proto::Node {
            id: "n1".into(),
            node_uuid: String::new(),
            name: "pc".into(),
            caps: Caps::default(),
            joined_at: String::new(),
            last_seen: String::new(),
            online: true,
            rate: 2.0,
            running: 0,
            done: 0,
            failed: 0,
            speed_factor: 0.0,
            preset_speeds: BTreeMap::new(),
            paused: String::new(),
        });
        annotate_eta(&mut st);
        assert_eq!(st.jobs[0].eta_seconds, 5);
        assert!(st.jobs[0].waiting.is_empty());
    }
}
