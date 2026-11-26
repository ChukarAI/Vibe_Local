//! Vibe-Link HTTP server: OpenAI-compatible `/v1/*` routes for Vibe Router,
//! `/api/*` control routes, the web leaderboard and strict security defaults
//! (loopback bind, LAN token auth, CSRF content-type guard, CSP).

use std::io::Read;
use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::json;
use tiny_http::{Header, Method, Request, Response, Server};

use vibe_local_core::catalog::{self, CatalogEntry};
use vibe_local_core::db::SharedDb;
use vibe_local_core::hardware::HardwareProbe;
use vibe_local_core::{hf_crawler, recommender, util};
use vibe_local_engine::{ChatMessage, Engine, LoadRequest};

pub const DEFAULT_PORT: u16 = 20129;
const MAX_BODY_BYTES: u64 = 8 * 1024 * 1024;
// 8 workers so slow routes (vibemark, discover, chat) cannot starve the
// fast /api/status and /api/hardware probes Vibe Router polls.
const WORKER_THREADS: usize = 8;

const CSP: &str = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'";

#[derive(Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub lan: bool,
    pub token: Option<String>,
    pub db: SharedDb,
    pub engine: Arc<Engine>,
    pub web_root: Option<PathBuf>,
    pub catalog_path: Option<PathBuf>,
}

pub struct ServerHandle {
    pub port: u16,
    pub lan: bool,
    stop: Arc<AtomicBool>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl ServerHandle {
    pub fn shutdown(&self) {
        if self.stop.swap(true, Ordering::SeqCst) {
            return;
        }
        // Wake blocked recv() loops with one dummy request per worker.
        for _ in 0..WORKER_THREADS {
            if let Ok(mut stream) = TcpStream::connect(("127.0.0.1", self.port)) {
                let _ =
                    stream.write_all(b"GET /__vibe_shutdown HTTP/1.1\r\nHost: localhost\r\n\r\n");
            }
        }
        for handle in self.workers.lock().unwrap().drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub fn start_server(config: ServerConfig) -> anyhow::Result<Arc<ServerHandle>> {
    let bind: SocketAddr = if config.lan {
        ([0, 0, 0, 0], config.port).into()
    } else {
        ([127, 0, 0, 1], config.port).into()
    };
    let server = Server::http(bind).map_err(|e| anyhow::anyhow!("cannot bind {bind}: {e}"))?;
    let actual_port = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        // The server always binds an IP socket; this arm exists so the
        // match is exhaustive on Unix (tiny_http has a Unix variant).
        _ => 0,
    };
    let server = Arc::new(server);
    let stop = Arc::new(AtomicBool::new(false));
    let state = Arc::new(AppState {
        config: config.clone(),
        stop: Arc::clone(&stop),
        vibemark_running: AtomicBool::new(false),
        prewarm_running: Arc::new(AtomicBool::new(false)),
    });
    let mut workers = Vec::new();
    for i in 0..WORKER_THREADS {
        let server = Arc::clone(&server);
        let state = Arc::clone(&state);
        let stop = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name(format!("vibe-http-{i}"))
            .spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    match server.recv() {
                        Ok(request) => handle_request(request, &state),
                        Err(_) => break,
                    }
                }
            })?;
        workers.push(handle);
    }
    Ok(Arc::new(ServerHandle {
        port: actual_port,
        lan: config.lan,
        stop,
        workers: Mutex::new(workers),
    }))
}

struct AppState {
    config: ServerConfig,
    stop: Arc<AtomicBool>,
    vibemark_running: AtomicBool,
    prewarm_running: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid header")
}

fn json_response(
    status: u16,
    value: serde_json::Value,
    extra: &[(&str, String)],
) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = serde_json::to_vec(&value).unwrap_or_default();
    let mut resp = Response::from_data(body)
        .with_status_code(status)
        .with_header(header("Content-Type", "application/json; charset=utf-8"))
        .with_header(header("Content-Security-Policy", CSP))
        .with_header(header("X-Content-Type-Options", "nosniff"));
    for (k, v) in extra {
        resp = resp.with_header(header(k, v));
    }
    resp
}

fn ok_json(value: serde_json::Value) -> Response<std::io::Cursor<Vec<u8>>> {
    json_response(200, value, &[])
}

fn err_json(status: u16, message: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    json_response(status, json!({"ok": false, "error": message}), &[])
}

#[derive(Debug)]
struct BodyTooLarge;

impl std::fmt::Display for BodyTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "body exceeds {MAX_BODY_BYTES} bytes")
    }
}

impl std::error::Error for BodyTooLarge {}

fn read_body(request: &mut Request) -> anyhow::Result<String> {
    let mut buf = Vec::new();
    request
        .as_reader()
        .take(MAX_BODY_BYTES + 1)
        .read_to_end(&mut buf)?;
    if buf.len() as u64 > MAX_BODY_BYTES {
        return Err(BodyTooLarge.into());
    }
    Ok(String::from_utf8(buf)?)
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

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
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

// ---------------------------------------------------------------------------
// Security middleware
// ---------------------------------------------------------------------------

fn request_header(request: &Request, name: &str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|h| h.value.as_str().to_string())
}

fn authorized(request: &Request, state: &AppState) -> bool {
    if !state.config.lan {
        return true; // loopback-only bind: local trust boundary
    }
    let Some(expected) = state.config.token.as_deref() else {
        return false; // LAN without a token is never allowed
    };
    let provided = request_header(request, "X-Vibe-Local-Token").or_else(|| {
        request_header(request, "Authorization")
            .and_then(|v| v.strip_prefix("Bearer ").map(|t| t.to_string()))
    });
    match provided {
        Some(token) => util::constant_time_eq(token.as_bytes(), expected.as_bytes()),
        None => false,
    }
}

/// State-mutating routes require JSON content-type or the admin header.
fn csrf_ok(request: &Request) -> bool {
    let ct = request_header(request, "Content-Type").unwrap_or_default();
    ct.to_lowercase().starts_with("application/json")
        || request_header(request, "X-Vibe-Admin").as_deref() == Some("1")
}

/// DNS-rebinding defense: the Host header must name this machine — an IP
/// literal, `localhost`, or the local hostname (optionally `.local`).
/// Without this, a remote web page could resolve its own domain to
/// 127.0.0.1 and read loopback API responses same-origin (no token needed
/// in loopback mode).
fn host_ok(request: &Request) -> bool {
    let Some(host) = request_header(request, "Host") else {
        return false;
    };
    let host = host.trim();
    let name = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or("") // [::1]:port
    } else if host.matches(':').count() > 1 {
        host // bare IPv6 literal
    } else {
        host.split(':').next().unwrap_or(host) // name:port
    };
    let name = name.trim_end_matches('.').to_lowercase();
    if name.is_empty() {
        return false;
    }
    if name.parse::<std::net::IpAddr>().is_ok() || name == "localhost" {
        return true;
    }
    let hn = crate::discovery::local_hostname().to_lowercase();
    name == hn || name == format!("{hn}.local")
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

fn handle_request(mut request: Request, state: &AppState) {
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or("/").to_string();
    let method = request.method().clone();

    if state.stop.load(Ordering::SeqCst) {
        let _ = request.respond(err_json(503, "server shutting down"));
        return;
    }

    if !host_ok(&request) {
        let _ = request.respond(err_json(
            421,
            "Host header must name this machine (IP, localhost or hostname) — DNS-rebinding guard",
        ));
        return;
    }

    if matches!(
        method,
        Method::Post | Method::Put | Method::Delete | Method::Patch
    ) && request_header(&request, "Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .is_some_and(|n| n > MAX_BODY_BYTES)
    {
        let _ = request.respond(err_json(
            413,
            &format!("body exceeds {MAX_BODY_BYTES} bytes"),
        ));
        return;
    }

    // Static UI is public even in LAN mode (no secrets in the bundle);
    // every API/v1 route is token-gated when LAN is enabled.
    let is_api = path.starts_with("/api/") || path.starts_with("/v1/");
    if is_api && !authorized(&request, state) {
        let _ = request.respond(err_json(
            401,
            "missing or invalid X-Vibe-Local-Token / Bearer token",
        ));
        return;
    }
    let mutating = matches!(method, Method::Post | Method::Delete);
    if is_api && mutating && !csrf_ok(&request) {
        let _ = request.respond(err_json(
            415,
            "state-mutating requests require Content-Type: application/json or X-Vibe-Admin: 1",
        ));
        return;
    }

    let result = route(&mut request, state, &path, &method, &url);
    let response = match result {
        Ok(resp) => resp,
        Err(e) if e.downcast_ref::<BodyTooLarge>().is_some() => {
            err_json(413, &format!("body exceeds {MAX_BODY_BYTES} bytes"))
        }
        Err(e) => err_json(500, &format!("{e:#}")),
    };
    let _ = request.respond(response);
}

type HttpResult = anyhow::Result<Response<std::io::Cursor<Vec<u8>>>>;

fn route(
    request: &mut Request,
    state: &AppState,
    path: &str,
    method: &Method,
    url: &str,
) -> HttpResult {
    match (method.clone(), path) {
        (Method::Get, "/api/hardware") => {
            let report = HardwareProbe::scan();
            let db = state.config.db.lock().unwrap();
            db.insert_scan(&report)?;
            Ok(ok_json(serde_json::to_value(&report)?))
        }
        (Method::Get, "/api/recommendations") => {
            let report = HardwareProbe::scan();
            let rec = recommender::recommend(&report);
            Ok(ok_json(serde_json::to_value(&rec)?))
        }
        (Method::Get, "/api/status") => {
            let status = state.config.engine.status()?;
            Ok(ok_json(json!({
                "ok": true,
                "loaded_model": status.loaded_model,
                "backend": status.backend,
                "engine": status.engine,
                "idle_timeout_secs": status.idle_timeout_secs,
                "thermal": {"celsius": status.thermal_celsius, "state": status.thermal_state},
                "has_vision": status.has_vision,
                "has_embeddings": status.has_embeddings,
                "mmproj_loaded": status.has_vision,
                "embedding": status.has_embeddings,
                "lan": state.config.lan,
                "port": state.config.port,
            })))
        }
        (Method::Get, "/api/leaderboard") => {
            let profile = query_param(url, "profile").unwrap_or_default();
            Ok(ok_json(leaderboard_payload(state, &profile)?))
        }
        (Method::Post, "/api/leaderboard/submit") => {
            let body = read_body(request)?;
            let value: serde_json::Value = serde_json::from_str(&body)?;
            let entry_value = value.get("entry").cloned().unwrap_or(value);
            let mut entry: CatalogEntry = serde_json::from_value(entry_value)?;
            if entry.provenance == "seed_estimate" {
                return Ok(err_json(
                    400,
                    "seed_estimate entries cannot be submitted via the API",
                ));
            }
            // Keep honest partial-run labels; anything else is a submission.
            if entry.provenance != "measured" && entry.provenance != "throughput_only" {
                entry.provenance = "submitted".into();
            }
            entry.validate()?;
            let db = state.config.db.lock().unwrap();
            db.upsert_catalog_entry(&entry)?;
            for (pillar, score) in &entry.pillars {
                db.insert_benchmark(
                    &entry.model_id,
                    &entry.hf_repo,
                    &entry.filename,
                    &entry.quant,
                    &entry.hardware_profile,
                    &entry.thr_throughput_backend(),
                    pillar,
                    score.score,
                    &json!({"metric": score.metric, "n": score.n, "detail": score.detail}),
                    &entry.provenance,
                )?;
            }
            Ok(ok_json(
                json!({"ok": true, "model_id": entry.model_id, "provenance": entry.provenance}),
            ))
        }
        (Method::Get, "/api/models") => {
            let query = query_param(url, "query").unwrap_or_default();
            let db = state.config.db.lock().unwrap();
            Ok(ok_json(
                json!({"ok": true, "results": db.models_for_query(&query)?}),
            ))
        }
        (Method::Get, "/api/discover") => {
            let query = query_param(url, "query").unwrap_or_else(|| "gguf".into());
            let files = hf_crawler::discover(&query, 5)?;
            let db = state.config.db.lock().unwrap();
            for f in &files {
                db.upsert_model(f)?;
            }
            Ok(ok_json(json!({"ok": true, "results": files})))
        }
        (Method::Post, "/api/load") => {
            let body = read_body(request)?;
            let req: LoadRequest = serde_json::from_str(&body)?;
            state.config.engine.load(req)?;
            Ok(ok_json(json!({"ok": true})))
        }
        (Method::Post, "/api/prewarm") => prewarm_route(request, state),
        (Method::Post, "/api/unload") => {
            state.config.engine.unload();
            Ok(ok_json(json!({"ok": true})))
        }
        (Method::Post, "/api/idle") => {
            let body = read_body(request)?;
            let value: serde_json::Value = serde_json::from_str(&body)?;
            let secs = value["seconds"].as_u64().unwrap_or(300);
            state.config.engine.set_idle_secs(secs);
            Ok(ok_json(json!({"ok": true, "idle_timeout_secs": secs})))
        }
        (Method::Post, "/api/thermal") => {
            let body = read_body(request)?;
            let value: serde_json::Value = serde_json::from_str(&body)?;
            let celsius = value["celsius"]
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("celsius required"))?;
            // Same plausibility window as the JNI bridge: an absurd injected
            // value (e.g. 1e300) would otherwise pause inference forever.
            if !(-40.0..=125.0).contains(&celsius) {
                return Ok(err_json(400, "implausible temperature"));
            }
            state.config.engine.thermal().set_override(celsius);
            Ok(ok_json(json!({"ok": true})))
        }
        (Method::Post, "/api/vibemark") => {
            // Only one benchmark at a time: a second request gets 409
            // immediately instead of occupying another worker thread.
            if state
                .vibemark_running
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                return Ok(err_json(409, "a VibeMark run is already in progress"));
            }
            let result = vibemark_route(request, state);
            state.vibemark_running.store(false, Ordering::SeqCst);
            result
        }
        (Method::Get, "/api/vibemark/latest") => {
            let db = state.config.db.lock().unwrap();
            Ok(ok_json(
                json!({"ok": true, "result": db.latest_vibemark()?}),
            ))
        }
        (Method::Get, "/v1/models") => {
            let status = state.config.engine.status()?;
            let mut data = Vec::new();
            if let Some(model) = &status.loaded_model {
                let id = std::path::Path::new(model)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| model.clone());
                data.push(json!({
                    "id": id, "object": "model", "created": util::unix_now(),
                    "owned_by": "vibe-local", "backend": status.backend, "engine": status.engine,
                    "mmproj": status.has_vision, "embedding": status.has_embeddings,
                }));
            }
            Ok(ok_json(json!({"object": "list", "data": data})))
        }
        (Method::Post, "/v1/chat/completions") => {
            let body = read_body(request)?;
            let value: serde_json::Value = serde_json::from_str(&body)?;
            let strip = value["strip_think"].as_bool() == Some(true)
                || request_header(request, "X-Vibe-Strip-Think").as_deref() == Some("1");
            let max_tokens = value["max_tokens"].as_u64().unwrap_or(512).min(8192) as u32;
            // The full OpenAI message array reaches llama-server unflattened
            // (system role + multi-turn intact); the builtin fallback
            // flattens internally and stays labeled builtin-microeval.
            let reply = if let Some(arr) = value["messages"].as_array() {
                let messages: Vec<ChatMessage> = arr
                    .iter()
                    .filter_map(|m| {
                        Some(ChatMessage {
                            role: m["role"].as_str()?.to_string(),
                            content: m["content"].as_str().unwrap_or("").to_string(),
                        })
                    })
                    .collect();
                if messages.iter().all(|m| m.content.trim().is_empty()) {
                    return Ok(err_json(400, "messages must contain user content"));
                }
                state
                    .config
                    .engine
                    .chat_messages(messages, strip, max_tokens)?
            } else {
                let prompt = value["prompt"].as_str().unwrap_or("");
                if prompt.trim().is_empty() {
                    return Ok(err_json(400, "messages or prompt required"));
                }
                state.config.engine.chat(prompt, strip, max_tokens)?
            };
            // reply.model is captured at inference time — a follow-up
            // status() would queue behind other chats and could time out.
            let model_id = if reply.model.is_empty() {
                "vibe-local".to_string()
            } else {
                reply.model.clone()
            };
            let prompt_text = extract_prompt(&value);
            let resp = json!({
                "id": format!("chatcmpl-vibe-{}", util::unix_now()),
                "object": "chat.completion",
                "created": util::unix_now(),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply.text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": vibe_local_engine::microeval::tokenize(&prompt_text).len(),
                    "completion_tokens": reply.tokens_generated,
                    "total_tokens": vibe_local_engine::microeval::tokenize(&prompt_text).len() as u64 + reply.tokens_generated,
                },
            });
            Ok(json_response(
                200,
                resp,
                &[
                    ("X-Vibe-Router-Cost-Micros", "0".to_string()),
                    (
                        "X-Vibe-Local-Tok-Per-Sec",
                        format!("{:.2}", reply.tok_per_sec),
                    ),
                    ("X-Vibe-Local-Backend", reply.backend.clone()),
                    ("X-Vibe-Local-Engine", reply.engine.clone()),
                ],
            ))
        }
        (Method::Post, "/v1/embeddings") => {
            let body = read_body(request)?;
            let value: serde_json::Value = serde_json::from_str(&body)?;
            // OpenAI shape: `input` is a string or an array of strings.
            let inputs: Vec<String> = match &value["input"] {
                serde_json::Value::String(s) => vec![s.clone()],
                serde_json::Value::Array(arr) => {
                    let mut out = Vec::with_capacity(arr.len());
                    for item in arr {
                        let Some(s) = item.as_str() else {
                            return Ok(err_json(400, "input array must contain only strings"));
                        };
                        out.push(s.to_string());
                    }
                    out
                }
                _ => return Ok(err_json(400, "input (string or array of strings) required")),
            };
            if inputs.is_empty() || inputs.iter().any(|s| s.trim().is_empty()) {
                return Ok(err_json(
                    400,
                    "input must contain at least one non-empty text",
                ));
            }
            if inputs.len() > vibe_local_engine::MAX_EMBED_INPUTS {
                return Ok(err_json(
                    400,
                    &format!(
                        "input is limited to {} texts per request",
                        vibe_local_engine::MAX_EMBED_INPUTS
                    ),
                ));
            }
            let reply = state.config.engine.embeddings(inputs)?;
            let model_id = if reply.model.is_empty() {
                "vibe-local".to_string()
            } else {
                std::path::Path::new(&reply.model)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| reply.model.clone())
            };
            let data: Vec<serde_json::Value> = reply
                .vectors
                .iter()
                .enumerate()
                .map(|(i, v)| json!({"object": "embedding", "index": i, "embedding": v}))
                .collect();
            let resp = json!({
                "object": "list",
                "data": data,
                "model": model_id,
                "engine": reply.engine,
                "usage": {
                    "prompt_tokens": reply.prompt_tokens,
                    "total_tokens": reply.prompt_tokens,
                },
            });
            Ok(json_response(
                200,
                resp,
                &[
                    ("X-Vibe-Router-Cost-Micros", "0".to_string()),
                    ("X-Vibe-Local-Backend", reply.backend.clone()),
                    ("X-Vibe-Local-Engine", reply.engine.clone()),
                ],
            ))
        }
        (Method::Get, "/catalog.json") => Ok(ok_json(catalog_payload(state)?)),
        // --- Vibe Grid proxy (same-origin access to the :7900 coordinator) ---
        (Method::Get, "/api/grid/state") => {
            let base =
                crate::grid_proxy::coordinator_base(query_param(url, "coordinator").as_deref());
            let key = request_header(request, "X-Compost-Key");
            Ok(ok_json(crate::grid_proxy::fetch_json(
                &base,
                "/api/v1/state",
                key.as_deref(),
            )?))
        }
        (Method::Get, "/api/grid/capabilities") => {
            let base =
                crate::grid_proxy::coordinator_base(query_param(url, "coordinator").as_deref());
            let key = request_header(request, "X-Compost-Key");
            Ok(ok_json(crate::grid_proxy::capabilities_directory(
                &base,
                key.as_deref(),
            )?))
        }
        (Method::Get, "/api/grid/plan") => {
            let base =
                crate::grid_proxy::coordinator_base(query_param(url, "coordinator").as_deref());
            let key = request_header(request, "X-Compost-Key");
            let model =
                query_param(url, "model_id").unwrap_or_else(|| "OLMoE-1B-7B-0924-Instruct".into());
            let context: u64 = query_param(url, "context")
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096);
            Ok(ok_json(crate::grid_proxy::plan_moe_live(
                &model,
                context,
                &base,
                key.as_deref(),
            )?))
        }
        (Method::Get, p) if p.starts_with("/api/grid/tiles/") => {
            let base =
                crate::grid_proxy::coordinator_base(query_param(url, "coordinator").as_deref());
            let key = request_header(request, "X-Compost-Key");
            let job_id = p.trim_start_matches("/api/grid/tiles/");
            if !valid_job_id(job_id) {
                return Ok(err_json(400, "invalid job id"));
            }
            Ok(ok_json(crate::grid_proxy::fetch_json(
                &base,
                &format!("/api/v1/jobs/{job_id}/tiles"),
                key.as_deref(),
            )?))
        }
        (Method::Get, p) if p.starts_with("/api/grid/artifact/") => {
            let base =
                crate::grid_proxy::coordinator_base(query_param(url, "coordinator").as_deref());
            let key = request_header(request, "X-Compost-Key");
            let job_id = p.trim_start_matches("/api/grid/artifact/");
            if !valid_job_id(job_id) {
                return Ok(err_json(400, "invalid job id"));
            }
            let (bytes, ctype) = crate::grid_proxy::fetch_bytes(
                &base,
                &format!("/api/v1/jobs/{job_id}/artifact"),
                key.as_deref(),
            )?;
            Ok(Response::from_data(bytes)
                .with_status_code(200)
                .with_header(header("Content-Type", &ctype))
                .with_header(header("Content-Security-Policy", CSP)))
        }
        (Method::Post, "/api/grid/jobs") => {
            let base =
                crate::grid_proxy::coordinator_base(query_param(url, "coordinator").as_deref());
            let key = request_header(request, "X-Compost-Key");
            let body = read_body(request)?;
            let value: serde_json::Value = serde_json::from_str(&body)?;
            Ok(ok_json(crate::grid_proxy::post_json(
                &base,
                "/api/v1/jobs",
                key.as_deref(),
                value,
            )?))
        }
        (Method::Post, p) if p.starts_with("/api/grid/cancel/") => {
            let base =
                crate::grid_proxy::coordinator_base(query_param(url, "coordinator").as_deref());
            let key = request_header(request, "X-Compost-Key");
            let job_id = p.trim_start_matches("/api/grid/cancel/");
            if !valid_job_id(job_id) {
                return Ok(err_json(400, "invalid job id"));
            }
            Ok(ok_json(crate::grid_proxy::post_json(
                &base,
                &format!("/api/v1/jobs/{job_id}/cancel"),
                key.as_deref(),
                json!({}),
            )?))
        }
        (Method::Get, "/") | (Method::Get, "/leaderboard") => serve_static(state, "index.html"),
        (Method::Get, p) if !p.contains("..") => {
            let rel = p.trim_start_matches('/');
            if rel.is_empty() {
                return serve_static(state, "index.html");
            }
            serve_static(state, rel)
        }
        _ => Ok(err_json(404, "not found")),
    }
}

trait ThroughputBackend {
    fn thr_throughput_backend(&self) -> String;
}

/// Body of POST /api/vibemark, split out so the 409 concurrency guard can
/// reset its flag on every exit path (including errors).
fn vibemark_route(request: &mut Request, state: &AppState) -> HttpResult {
    let body = read_body(request)?;
    let value: serde_json::Value = if body.trim().is_empty() {
        json!({})
    } else {
        serde_json::from_str(&body)?
    };
    if let Some(path) = value["path"].as_str() {
        let status = state.config.engine.status()?;
        if status.loaded_model.as_deref() != Some(path) {
            state.config.engine.load(LoadRequest {
                path: path.to_string(),
                backend: value["backend"].as_str().unwrap_or("cpu").to_string(),
                ngl: value["ngl"].as_u64().unwrap_or(0) as u32,
                ctx: value["ctx"].as_u64().unwrap_or(0) as u32,
                draft_model_path: value["draft_model_path"].as_str().map(String::from),
                draft_ngl: value["draft_ngl"].as_i64().map(|v| v as i32),
                draft_max: value["draft_max"].as_u64().map(|v| v as u32),
                mmproj_path: value["mmproj_path"].as_str().map(String::from),
                embedding: value["embedding"].as_bool().unwrap_or(false),
            })?;
        }
    }
    let budget = value["budget_secs"].as_u64().unwrap_or(15);
    let result = state.config.engine.vibemark(budget)?;
    let db = state.config.db.lock().unwrap();
    db.insert_vibemark(
        &result.model,
        &result.backend,
        &result.engine,
        result.prefill_tps,
        result.decode_tps,
        result.ttft_ms,
        result.peak_ram_mb,
        result.duration_s,
    )?;
    Ok(ok_json(serde_json::to_value(&result)?))
}

/// POST /api/prewarm: start loading a model in the background so Vibe Router
/// / Vibe Assembly can hide cold-start latency behind planner reasoning.
/// 202 when a load starts, 200 when the model is already resident, 409 when
/// another prewarm is still in flight.
fn prewarm_route(request: &mut Request, state: &AppState) -> HttpResult {
    let body = read_body(request)?;
    let value: serde_json::Value = serde_json::from_str(&body)?;
    let Some(path) = value["path"].as_str() else {
        return Ok(err_json(400, "path is required"));
    };
    // Claim the slot before touching the engine: status() queues behind any
    // running benchmark, so checking it first would shrink the 409 window.
    if state
        .prewarm_running
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Ok(err_json(409, "a prewarm load is already in progress"));
    }
    let status = match state.config.engine.status() {
        Ok(s) => s,
        Err(e) => {
            state.prewarm_running.store(false, Ordering::SeqCst);
            return Err(e);
        }
    };
    if status.loaded_model.as_deref() == Some(path) {
        state.prewarm_running.store(false, Ordering::SeqCst);
        return Ok(ok_json(json!({"ok": true, "state": "already_loaded"})));
    }
    let req = LoadRequest {
        path: path.to_string(),
        backend: value["backend"].as_str().unwrap_or("cpu").to_string(),
        ngl: value["ngl"].as_u64().unwrap_or(0) as u32,
        ctx: value["ctx"].as_u64().unwrap_or(0) as u32,
        draft_model_path: value["draft_model_path"].as_str().map(String::from),
        draft_ngl: value["draft_ngl"].as_i64().map(|v| v as i32),
        draft_max: value["draft_max"].as_u64().map(|v| v as u32),
        mmproj_path: value["mmproj_path"].as_str().map(String::from),
        embedding: value["embedding"].as_bool().unwrap_or(false),
    };
    let engine = Arc::clone(&state.config.engine);
    let flag = Arc::clone(&state.prewarm_running);
    if let Err(e) = std::thread::Builder::new()
        .name("vibe-prewarm".into())
        .spawn(move || {
            if let Err(e) = engine.load(req) {
                eprintln!("[vibe-prewarm] load failed: {e:#}");
            }
            flag.store(false, Ordering::SeqCst);
        })
    {
        // Never leak the single-flight slot on a spawn failure.
        state.prewarm_running.store(false, Ordering::SeqCst);
        return Err(e.into());
    }
    Ok(json_response(
        202,
        json!({"ok": true, "state": "loading"}),
        &[],
    ))
}

impl ThroughputBackend for CatalogEntry {
    fn thr_throughput_backend(&self) -> String {
        match self.hardware_profile.as_str() {
            "rx580_vulkan" => "vulkan".into(),
            "tesla_t4" => "cuda".into(),
            _ => "cpu".into(),
        }
    }
}

/// Coordinator job ids are `<prefix>-<seq>`; anything else could smuggle
/// path segments or query strings into the proxied coordinator URL.
fn valid_job_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn extract_prompt(value: &serde_json::Value) -> String {
    if let Some(messages) = value["messages"].as_array() {
        let mut parts = Vec::new();
        for m in messages {
            let role = m["role"].as_str().unwrap_or("user");
            let content = m["content"].as_str().unwrap_or("");
            if role == "system" {
                parts.push(format!("System: {content}"));
            } else if role == "user" {
                parts.push(content.to_string());
            }
        }
        return parts.join("\n");
    }
    value["prompt"].as_str().unwrap_or("").to_string()
}

// ---------------------------------------------------------------------------
// Leaderboard / catalog payloads
// ---------------------------------------------------------------------------

fn default_profiles() -> serde_json::Value {
    json!({
        "rx580_vulkan": {"label": "AMD Radeon RX 580 8GB (Vulkan)", "backend": "vulkan", "vram_mb": 8192, "ram_mb": 16384},
        "nothing_phone_2a": {"label": "Nothing Phone 2a 12GB (CPU NEON)", "backend": "cpu", "vram_mb": 0, "ram_mb": 12288},
        "i5_4690_ddr3": {"label": "Intel i5-4690 16GB DDR3 (CPU AVX2)", "backend": "cpu", "vram_mb": 0, "ram_mb": 16384},
        "tesla_t4": {"label": "NVIDIA Tesla T4 16GB (CUDA)", "backend": "cuda", "vram_mb": 16384, "ram_mb": 30720},
    })
}

fn load_catalog_file(state: &AppState) -> Option<catalog::Catalog> {
    let path = state
        .config
        .catalog_path
        .clone()
        .or_else(catalog::find_catalog_file)?;
    catalog::Catalog::load(&path).ok()
}

fn leaderboard_payload(state: &AppState, profile: &str) -> anyhow::Result<serde_json::Value> {
    let db = state.config.db.lock().unwrap();
    let mut entries = db.leaderboard(profile)?;
    drop(db);
    let mask = catalog::pareto_frontier(&entries);
    for (entry, optimal) in entries.iter_mut().zip(mask) {
        entry.pareto_optimal = Some(optimal);
    }
    let profiles = load_catalog_file(state)
        .map(|c| serde_json::to_value(&c.hardware_profiles).unwrap_or_else(|_| default_profiles()))
        .unwrap_or_else(default_profiles);
    let labels: serde_json::Map<String, serde_json::Value> = catalog::PILLARS
        .iter()
        .map(|p| ((*p).to_string(), json!(catalog::pillar_label(p))))
        .collect();
    Ok(json!({
        "ok": true,
        "pillars": catalog::PILLARS,
        "pillar_labels": labels,
        "profiles": profiles,
        "entries": entries,
    }))
}

fn catalog_payload(state: &AppState) -> anyhow::Result<serde_json::Value> {
    if let Some(catalog) = load_catalog_file(state) {
        return Ok(serde_json::to_value(&catalog)?);
    }
    let db = state.config.db.lock().unwrap();
    let entries = db.leaderboard("")?;
    let labels: serde_json::Map<String, serde_json::Value> = catalog::PILLARS
        .iter()
        .map(|p| ((*p).to_string(), json!(catalog::pillar_label(p))))
        .collect();
    Ok(json!({
        "schema_version": 1,
        "generated": util::iso8601_now(),
        "provenance_note": "Entries stored in the local Vibe Local database.",
        "pillars": catalog::PILLARS,
        "pillar_labels": labels,
        "hardware_profiles": default_profiles(),
        "entries": entries,
    }))
}

// ---------------------------------------------------------------------------
// Static files (web leaderboard) with traversal protection + CSP
// ---------------------------------------------------------------------------

fn find_web_root(state: &AppState) -> Option<PathBuf> {
    if let Some(root) = &state.config.web_root
        && root.is_dir()
    {
        return Some(root.clone());
    }
    let mut bases = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        bases.push(dir.to_path_buf());
        bases.push(dir.join(".."));
        bases.push(dir.join("../.."));
    }
    if let Ok(cwd) = std::env::current_dir() {
        bases.push(cwd);
    }
    for base in bases {
        let candidate = base.join("leaderboard-web");
        if candidate.join("index.html").is_file() {
            return Some(candidate);
        }
    }
    None
}

fn content_type_for(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn serve_static(state: &AppState, rel: &str) -> HttpResult {
    if rel.contains("..") || rel.contains('\\') {
        return Ok(err_json(400, "invalid path"));
    }
    let Some(root) = find_web_root(state) else {
        return Ok(err_json(
            503,
            "leaderboard-web assets not found; run from the repository root or set VIBE_WEB_ROOT",
        ));
    };
    let full = root.join(rel);
    let Ok(canonical) = full.canonicalize() else {
        return Ok(err_json(404, "not found"));
    };
    let Ok(root_canonical) = root.canonicalize() else {
        return Ok(err_json(500, "web root unavailable"));
    };
    if !canonical.starts_with(&root_canonical) || !canonical.is_file() {
        return Ok(err_json(404, "not found"));
    }
    let bytes = std::fs::read(&canonical)?;
    let resp = Response::from_data(bytes)
        .with_status_code(200)
        .with_header(header("Content-Type", content_type_for(rel)))
        .with_header(header("Content-Security-Policy", CSP))
        .with_header(header("X-Content-Type-Options", "nosniff"))
        .with_header(header("Cache-Control", "no-cache"));
    Ok(resp)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use vibe_local_core::Db;
    use vibe_local_engine::EngineConfig;

    fn test_state(lan: bool, token: Option<&str>) -> (Arc<ServerHandle>, String) {
        let db = Db::open_in_memory().unwrap().shared();
        let engine = Engine::start(EngineConfig {
            internal_port: 20198,
            ..Default::default()
        });
        let config = ServerConfig {
            port: 0,
            lan,
            token: token.map(|t| t.to_string()),
            db,
            engine,
            web_root: None,
            catalog_path: None,
        };
        let handle = start_server(config).unwrap();
        let base = format!("http://127.0.0.1:{}", handle.port);
        (handle, base)
    }

    fn get(url: &str) -> ureq::Response {
        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(10))
            .build()
            .get(url)
            .call()
            .unwrap()
    }

    #[test]
    fn hardware_and_recommendations() {
        let (_h, base) = test_state(false, None);
        let hw: serde_json::Value = get(&format!("{base}/api/hardware")).into_json().unwrap();
        assert!(hw["ram_total_mb"].as_u64().unwrap() > 0);
        let rec: serde_json::Value = get(&format!("{base}/api/recommendations"))
            .into_json()
            .unwrap();
        assert!(rec["tier"].as_str().is_some());
        assert!(rec["candidates"].as_array().is_some());
    }

    #[test]
    fn lan_auth_enforced() {
        let (_h, base) = test_state(true, Some("secret-token-123"));
        // No token -> 401
        let err = ureq::get(&format!("{base}/api/hardware"))
            .call()
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(401, _)));
        // Wrong token -> 401
        let err = ureq::get(&format!("{base}/api/hardware"))
            .set("X-Vibe-Local-Token", "wrong")
            .call()
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(401, _)));
        // Correct header -> 200
        let resp = ureq::get(&format!("{base}/api/hardware"))
            .set("X-Vibe-Local-Token", "secret-token-123")
            .call()
            .unwrap();
        assert_eq!(resp.status(), 200);
        // Bearer form -> 200
        let resp = ureq::get(&format!("{base}/api/status"))
            .set("Authorization", "Bearer secret-token-123")
            .call()
            .unwrap();
        assert_eq!(resp.status(), 200);
        // Static UI stays public
        let resp = ureq::get(&format!("{base}/")).call();
        assert!(resp.is_ok() || matches!(resp, Err(ureq::Error::Status(503, _))));
    }

    #[test]
    fn lan_without_token_rejects_all_api() {
        let (_h, base) = test_state(true, None);
        let err = ureq::get(&format!("{base}/api/hardware"))
            .call()
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(401, _)));
    }

    #[test]
    fn csrf_content_type_guard() {
        let (_h, base) = test_state(true, Some("t"));
        // POST without JSON content-type -> 415
        let err = ureq::post(&format!("{base}/api/unload"))
            .set("X-Vibe-Local-Token", "t")
            .send_string("")
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(415, _)));
        // With JSON content-type -> 200
        let resp = ureq::post(&format!("{base}/api/unload"))
            .set("X-Vibe-Local-Token", "t")
            .set("Content-Type", "application/json")
            .send_string("{}")
            .unwrap();
        assert_eq!(resp.status(), 200);
        // Admin header alternative
        let resp = ureq::post(&format!("{base}/api/unload"))
            .set("X-Vibe-Local-Token", "t")
            .set("X-Vibe-Admin", "1")
            .send_string("")
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[test]
    fn host_guard_rejects_dns_rebinding() {
        let (_h, base) = test_state(false, None);
        // A remote page resolving its own domain to 127.0.0.1 sends that
        // domain as Host — must be rejected even in loopback (no-token) mode.
        let err = ureq::get(&format!("{base}/api/status"))
            .set("Host", "evil.com")
            .call()
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(421, _)));
        // IP-literal Host (what ureq sends by default) passes.
        let resp = ureq::get(&format!("{base}/api/status")).call().unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[test]
    fn thermal_override_plausibility() {
        let (_h, base) = test_state(false, None);
        let err = ureq::post(&format!("{base}/api/thermal"))
            .set("Content-Type", "application/json")
            .send_json(json!({"celsius": 1e300}))
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(400, _)));
        let resp = ureq::post(&format!("{base}/api/thermal"))
            .set("Content-Type", "application/json")
            .send_json(json!({"celsius": 41.5}))
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[test]
    fn oversized_body_is_rejected() {
        let (_h, base) = test_state(false, None);
        let err = ureq::post(&format!("{base}/api/idle"))
            .set("Content-Type", "application/json")
            .send_bytes(&vec![b' '; MAX_BODY_BYTES as usize + 1])
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(413, _)));
    }

    #[test]
    fn query_param_skips_valueless_pairs() {
        assert_eq!(
            query_param(
                "/x?flag&coordinator=http%3A%2F%2F127.0.0.1%3A7900",
                "coordinator"
            )
            .as_deref(),
            Some("http://127.0.0.1:7900")
        );
        assert_eq!(query_param("/x?a&b", "b"), None);
        assert_eq!(
            query_param("/x?%71uery=hello+world", "query").as_deref(),
            Some("hello world")
        );
        assert_eq!(query_param("/x", "k"), None);
    }

    #[test]
    fn grid_proxy_rejects_unsafe_job_ids() {
        let (_h, base) = test_state(false, None);
        let err = ureq::get(&format!("{base}/api/grid/tiles/bad%20id"))
            .call()
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(400, _)));
        assert!(!valid_job_id("../state"));
        assert!(!valid_job_id("a?b"));
        assert!(valid_job_id("job-12"));
    }

    #[test]
    fn chat_route_multi_turn_and_model_id() {
        let (_h, base) = test_state(false, None);
        if vibe_local_engine::supervisor::find_llama_server(None).is_some() {
            return; // synthetic GGUF only loads in builtin mode
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m-Q4_K_M.gguf");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&vibe_local_core::gguf::GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&8u32.to_le_bytes());
        let val = b"llama";
        bytes.extend_from_slice(&(val.len() as u64).to_le_bytes());
        bytes.extend_from_slice(val);
        std::fs::write(&path, bytes).unwrap();
        let path_str = path.to_string_lossy().into_owned();
        ureq::post(&format!("{base}/api/load"))
            .set("Content-Type", "application/json")
            .send_json(json!({"path": path_str}))
            .unwrap();
        // Multi-turn array reaches the engine intact (builtin flattens and
        // stays labeled); model id comes from the reply, not a status race.
        let resp = ureq::post(&format!("{base}/v1/chat/completions"))
            .set("Content-Type", "application/json")
            .send_json(json!({"messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "hello"}
            ]}))
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.header("X-Vibe-Local-Engine"),
            Some("builtin-microeval")
        );
        let v: serde_json::Value = resp.into_json().unwrap();
        assert_eq!(v["model"].as_str(), Some(path_str.as_str()));
        assert!(
            !v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .is_empty()
        );
        // Single user turn still hits the arithmetic path.
        let resp = ureq::post(&format!("{base}/v1/chat/completions"))
            .set("Content-Type", "application/json")
            .send_json(json!({"messages": [{"role": "user", "content": "What is 21 * 2?"}]}))
            .unwrap();
        let v: serde_json::Value = resp.into_json().unwrap();
        assert!(
            v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("42")
        );
        // Empty messages -> 400.
        let err = ureq::post(&format!("{base}/v1/chat/completions"))
            .set("Content-Type", "application/json")
            .send_json(json!({"messages": []}))
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(400, _)));
    }

    #[test]
    fn prewarm_and_vibemark_concurrency_guards() {
        let (_h, base) = test_state(false, None);
        if vibe_local_engine::supervisor::find_llama_server(None).is_some() {
            return; // synthetic GGUFs only load in builtin mode
        }
        let dir = tempfile::tempdir().unwrap();
        let mut paths = Vec::new();
        for name in ["a-Q4_K_M.gguf", "b-Q4_K_M.gguf"] {
            let path = dir.path().join(name);
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&vibe_local_core::gguf::GGUF_MAGIC.to_le_bytes());
            bytes.extend_from_slice(&3u32.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            let key = b"general.architecture";
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key);
            bytes.extend_from_slice(&8u32.to_le_bytes());
            let val = b"llama";
            bytes.extend_from_slice(&(val.len() as u64).to_le_bytes());
            bytes.extend_from_slice(val);
            std::fs::write(&path, bytes).unwrap();
            paths.push(path.to_string_lossy().into_owned());
        }
        // Reduce ureq results to (status, body) so thread closures never
        // return the large ureq::Error variant.
        fn post_status(
            url: String,
            body: serde_json::Value,
            secs: u64,
        ) -> (u16, serde_json::Value) {
            let result = ureq::post(&url)
                .set("Content-Type", "application/json")
                .timeout(Duration::from_secs(secs))
                .send_json(body);
            let resp = match result {
                Ok(resp) => resp,
                Err(ureq::Error::Status(_, resp)) => resp,
                Err(e) => panic!("request to {url} failed: {e}"),
            };
            let code = resp.status();
            (code, resp.into_json().unwrap_or(json!({})))
        }
        let post_json =
            |path: &str, body: serde_json::Value| post_status(format!("{base}{path}"), body, 30);
        // path is required.
        assert_eq!(post_json("/api/prewarm", json!({})).0, 400);
        // Hold the engine worker with a VibeMark run (~6s builtin budget).
        let base2 = base.clone();
        let pa = paths[0].clone();
        let mark = std::thread::spawn(move || {
            post_status(
                format!("{base2}/api/vibemark"),
                json!({"path": pa, "budget_secs": 6}),
                60,
            )
            .0
        });
        std::thread::sleep(Duration::from_millis(1000));
        // Concurrent VibeMark -> 409.
        assert_eq!(post_json("/api/vibemark", json!({})).0, 409);
        // Prewarm a different model while the engine is busy: both requests
        // race for the prewarm slot; one wins (202 queued), the other gets 409.
        let mut prewarms = Vec::new();
        for _ in 0..2 {
            let base3 = base.clone();
            let pb = paths[1].clone();
            prewarms.push(std::thread::spawn(move || {
                post_status(format!("{base3}/api/prewarm"), json!({"path": pb}), 30)
            }));
        }
        let results: Vec<_> = prewarms.into_iter().map(|h| h.join().unwrap()).collect();
        let mut accepted = 0;
        let mut rejected = 0;
        for (code, v) in results {
            match code {
                202 => {
                    assert_eq!(v["state"], "loading");
                    accepted += 1;
                }
                409 => rejected += 1,
                other => panic!("unexpected prewarm status {other}: {v}"),
            }
        }
        assert_eq!(accepted, 1, "expected exactly one 202");
        assert_eq!(rejected, 1, "expected exactly one 409");
        // First VibeMark completes normally.
        assert_eq!(mark.join().unwrap(), 200);
        // Background prewarm settles; repeating it reports already_loaded.
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            let st: serde_json::Value = get(&format!("{base}/api/status")).into_json().unwrap();
            if st["loaded_model"].as_str() == Some(paths[1].as_str()) {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "prewarm never settled"
            );
            std::thread::sleep(Duration::from_millis(200));
        }
        let (code, v) = post_json("/api/prewarm", json!({"path": paths[1]}));
        assert_eq!(code, 200);
        assert_eq!(v["state"], "already_loaded");
    }

    #[test]
    fn leaderboard_submit_roundtrip() {
        let (_h, base) = test_state(false, None);
        let entry = json!({
            "model_id": "TestModel-7B",
            "hf_repo": "bartowski/TestModel-7B-GGUF",
            "filename": "TestModel-7B-Q4_K_M.gguf",
            "download_url": "https://huggingface.co/bartowski/TestModel-7B-GGUF/resolve/main/TestModel-7B-Q4_K_M.gguf",
            "quant": "Q4_K_M",
            "architecture": "qwen2",
            "family": "dense",
            "total_params_b": 7.6,
            "active_params_b": 7.6,
            "size_bytes": 4680000000u64,
            "context_length": 32768,
            "hardware_profile": "rx580_vulkan",
            "provenance": "measured",
            "pillars": {
                "code_generation": {"score": 0.8, "metric": "pass@1", "n": 164},
                "math_reasoning": {"score": 0.7, "metric": "accuracy", "n": 50}
            },
            "throughput": {"prefill_tps": 400.0, "decode_tps": 30.0, "ttft_ms": 150.0, "peak_ram_mb": 5200.0}
        });
        let resp = ureq::post(&format!("{base}/api/leaderboard/submit"))
            .send_json(json!({"entry": entry}))
            .unwrap();
        assert_eq!(resp.status(), 200);
        let board: serde_json::Value = get(&format!("{base}/api/leaderboard?profile=rx580_vulkan"))
            .into_json()
            .unwrap();
        let entries = board["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["provenance"], "measured");
        assert!(entries[0]["vibe_quality_score"].as_f64().is_some());
        assert!(entries[0]["pareto_optimal"].as_bool().is_some());
        assert_eq!(board["pillars"].as_array().unwrap().len(), 10);
        // seed_estimate submission rejected
        let mut seed = entry.clone();
        seed["provenance"] = json!("seed_estimate");
        let err = ureq::post(&format!("{base}/api/leaderboard/submit"))
            .send_json(seed)
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(400, _)));
        // invalid entry rejected
        let mut bad = entry.clone();
        bad["family"] = json!("alien");
        let err = ureq::post(&format!("{base}/api/leaderboard/submit"))
            .send_json(bad)
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(500, _)));
    }

    #[test]
    fn chat_headers_and_think_strip() {
        let (handle, base) = test_state(false, None);
        if vibe_local_engine::supervisor::find_llama_server(None).is_some() {
            handle.shutdown();
            return; // real llama-server present; builtin path not exercised
        }
        // Load a synthetic GGUF (builtin micro-eval mode).
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("tiny-Q4_K_M.gguf");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&vibe_local_core::gguf::GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&model_path, bytes).unwrap();
        ureq::post(&format!("{base}/api/load"))
            .send_json(json!({"path": model_path.to_string_lossy(), "backend": "cpu"}))
            .unwrap();

        let resp = ureq::post(&format!("{base}/v1/chat/completions"))
            .send_json(json!({"messages": [{"role": "user", "content": "What is 6 * 7?"}], "strip_think": true}))
            .unwrap();
        assert_eq!(resp.header("X-Vibe-Router-Cost-Micros"), Some("0"));
        assert!(resp.header("X-Vibe-Local-Tok-Per-Sec").is_some());
        assert_eq!(resp.header("X-Vibe-Local-Backend"), Some("cpu"));
        let body: serde_json::Value = resp.into_json().unwrap();
        assert!(
            body["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("42")
        );

        let models: serde_json::Value = get(&format!("{base}/v1/models")).into_json().unwrap();
        assert_eq!(models["data"].as_array().unwrap().len(), 1);
        handle.shutdown();
    }

    fn load_tiny_model(base: &str, token: Option<&str>) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("tiny-Q4_K_M.gguf");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&vibe_local_core::gguf::GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&model_path, bytes).unwrap();
        let mut req = ureq::post(&format!("{base}/api/load"));
        if let Some(t) = token {
            req = req.set("X-Vibe-Local-Token", t);
        }
        req.send_json(
            json!({"path": model_path.to_string_lossy(), "backend": "cpu", "embedding": true}),
        )
        .unwrap();
        dir
    }

    #[test]
    fn embeddings_route_single_and_array() {
        let (handle, base) = test_state(false, None);
        if vibe_local_engine::supervisor::find_llama_server(None).is_some() {
            handle.shutdown();
            return; // real llama-server present; builtin path not exercised
        }
        let _dir = load_tiny_model(&base, None);

        // Single string input.
        let resp = ureq::post(&format!("{base}/v1/embeddings"))
            .send_json(json!({"input": "hello world"}))
            .unwrap();
        assert_eq!(resp.header("X-Vibe-Router-Cost-Micros"), Some("0"));
        assert_eq!(
            resp.header("X-Vibe-Local-Engine"),
            Some("builtin-microeval")
        );
        let body: serde_json::Value = resp.into_json().unwrap();
        assert_eq!(body["object"], "list");
        // The producing engine is labeled in the body, not just headers.
        assert_eq!(body["engine"], "builtin-microeval");
        let data = body["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["index"], 0);
        let vec0 = data[0]["embedding"].as_array().unwrap();
        assert_eq!(vec0.len(), vibe_local_engine::microeval::DET_EMBED_DIM);
        assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);

        // Array input: two vectors, indices in order, first one identical
        // to the single-string call (deterministic builtin).
        let resp = ureq::post(&format!("{base}/v1/embeddings"))
            .send_json(json!({"input": ["hello world", "another text"]}))
            .unwrap();
        let body: serde_json::Value = resp.into_json().unwrap();
        let data = body["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["index"], 0);
        assert_eq!(data[1]["index"], 1);
        assert_eq!(data[0]["embedding"], serde_json::json!(vec0));
        assert_ne!(data[0]["embedding"], data[1]["embedding"]);

        // Bad inputs are 400, not 500.
        for bad in [
            json!({}),
            json!({"input": []}),
            json!({"input": "   "}),
            json!({"input": [42]}),
            json!({"input": ["ok", ""]}),
        ] {
            let err = ureq::post(&format!("{base}/v1/embeddings"))
                .send_json(bad)
                .unwrap_err();
            assert!(matches!(err, ureq::Error::Status(400, _)), "{err:?}");
        }

        // /v1/models advertises the embedding flag for capability detection.
        let models: serde_json::Value = get(&format!("{base}/v1/models")).into_json().unwrap();
        assert_eq!(models["data"][0]["embedding"], true);
        assert_eq!(models["data"][0]["mmproj"], false);

        // /api/status reports the spec field names for the same flags.
        let status: serde_json::Value = get(&format!("{base}/api/status")).into_json().unwrap();
        assert_eq!(status["has_embeddings"], true);
        assert_eq!(status["has_vision"], false);
        handle.shutdown();
    }

    #[test]
    fn embeddings_lan_guards() {
        let (handle, base) = test_state(true, Some("secret-token-123"));
        if vibe_local_engine::supervisor::find_llama_server(None).is_some() {
            handle.shutdown();
            return;
        }
        // LAN auth: no token → 401, wrong token → 401.
        let err = ureq::post(&format!("{base}/v1/embeddings"))
            .send_json(json!({"input": "hello"}))
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(401, _)));
        let err = ureq::post(&format!("{base}/v1/embeddings"))
            .set("X-Vibe-Local-Token", "wrong")
            .send_json(json!({"input": "hello"}))
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(401, _)));
        // Host guard: foreign Host → 421 even with a valid token.
        let err = ureq::post(&format!("{base}/v1/embeddings"))
            .set("X-Vibe-Local-Token", "secret-token-123")
            .set("Host", "evil.com")
            .send_json(json!({"input": "hello"}))
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(421, _)));
        // Body limit: oversized Content-Length → 413.
        let err = ureq::post(&format!("{base}/v1/embeddings"))
            .set("X-Vibe-Local-Token", "secret-token-123")
            .set("Content-Type", "application/json")
            .send_bytes(&vec![b' '; MAX_BODY_BYTES as usize + 1])
            .unwrap_err();
        assert!(matches!(err, ureq::Error::Status(413, _)));
        // Valid token + valid Host + small body → 200 with the cost header.
        let _dir = load_tiny_model(&base, Some("secret-token-123"));
        let resp = ureq::post(&format!("{base}/v1/embeddings"))
            .set("X-Vibe-Local-Token", "secret-token-123")
            .send_json(json!({"input": "hello"}))
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.header("X-Vibe-Router-Cost-Micros"), Some("0"));
        handle.shutdown();
    }

    #[test]
    fn csp_on_api_responses() {
        let (_h, base) = test_state(false, None);
        let resp = get(&format!("{base}/api/status"));
        let csp = resp.header("Content-Security-Policy").unwrap();
        assert!(csp.contains("default-src 'self'"));
        assert!(csp.contains("frame-ancestors 'none'"));
    }

    #[test]
    fn url_param_parsing() {
        assert_eq!(
            query_param("/api/leaderboard?profile=rx580_vulkan", "profile").as_deref(),
            Some("rx580_vulkan")
        );
        assert_eq!(
            query_param("/api/discover?query=qwen%20coder", "query").as_deref(),
            Some("qwen coder")
        );
        assert_eq!(query_param("/api/x", "y"), None);
    }
}
