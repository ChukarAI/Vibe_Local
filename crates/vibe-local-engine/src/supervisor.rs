//! Inference engine supervisor: manages a `llama-server` child process with
//! backend selection (Vulkan / CUDA / CPU), falls back to the deterministic
//! built-in micro-eval engine when the binary is absent, enforces the idle
//! auto-unload timer (default 300s) and the thermal governor. All commands
//! flow through a bounded channel (capacity 16).

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{RecvTimeoutError, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use vibe_local_core::util;

use crate::microeval::{MicroEval, strip_think, tokenize};
use crate::thermal::{ThermalGovernor, ThermalState};
use crate::vibemark::{self, ProbeOutcome, VibeMarkResult};

pub const DEFAULT_IDLE_SECS: u64 = 300;
pub const DEFAULT_INTERNAL_PORT: u16 = 20130;
const COMMAND_CHANNEL_CAPACITY: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadRequest {
    pub path: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_ngl")]
    pub ngl: u32,
    /// 0 = auto-size from the GGUF RAM estimates against live VRAM/RAM,
    /// preserving a 512MB safety margin.
    #[serde(default)]
    pub ctx: u32,
    /// Small draft GGUF for native speculative decoding (`--model-draft`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_model_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_ngl: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_max: Option<u32>,
    /// Multimodal projector GGUF for vision models (`--mmproj`). Validated
    /// like the draft model: the file must exist and carry a GGUF header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_path: Option<String>,
    /// Load the model in embedding mode (`--embedding`): llama-server then
    /// serves `POST /v1/embeddings` with real model embeddings. The builtin
    /// fallback serves deterministic hash embeddings either way (labeled
    /// `builtin-microeval`), so this flag is a no-op without llama-server.
    #[serde(default, skip_serializing_if = "is_false")]
    pub embedding: bool,
}

fn is_false(v: &bool) -> bool {
    !*v
}

fn default_backend() -> String {
    "cpu".into()
}
fn default_ngl() -> u32 {
    0
}

pub const CTX_SAFETY_MARGIN_MB: f64 = 512.0;
pub const DEFAULT_DRAFT_MAX: u32 = 8;

/// Pick the largest GGUF-estimated context window that fits live memory
/// (VRAM when offloading, else free RAM) minus a 512MB safety margin,
/// capped at the model's trained context length.
pub fn auto_context(
    info: &vibe_local_core::gguf::GgufInfo,
    req: &LoadRequest,
    hw: &vibe_local_core::HardwareReport,
) -> u32 {
    let vram = if req.ngl > 0 && req.backend != "cpu" {
        hw.gpus.iter().filter_map(|g| g.vram_mb).max()
    } else {
        None
    };
    let budget_mb = vram.unwrap_or(hw.ram_free_mb) as f64 - CTX_SAFETY_MARGIN_MB;
    let mut estimates: Vec<_> = info.ram.iter().collect();
    estimates.sort_by_key(|r| r.ctx);
    let cap = info.context_length.unwrap_or(u64::MAX);
    let mut chosen: Option<u64> = None;
    for r in &estimates {
        if r.ctx <= cap && r.total_mb <= budget_mb {
            chosen = Some(r.ctx);
        }
    }
    // Nothing fits: fall back to the smallest estimated window (or 2048
    // when the GGUF carries no estimates) rather than inventing capacity.
    let pick = chosen
        .or_else(|| estimates.first().map(|r| r.ctx))
        .unwrap_or(2048);
    pick.min(cap) as u32
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatReply {
    pub text: String,
    pub tokens_generated: u64,
    pub elapsed_secs: f64,
    pub tok_per_sec: f64,
    pub backend: String,
    pub engine: String,
    /// Path of the model that produced this reply (captured at inference
    /// time, so callers never need a racy follow-up status query).
    #[serde(default)]
    pub model: String,
}

/// OpenAI-style chat message, passed through to llama-server unflattened.
///
/// `content` may arrive either as a plain string or as a multimodal
/// content-part array (`[{"type":"text","text":...},{"type":"image_url",
/// ...}]`). The array form is preserved verbatim in `content_parts` and
/// re-serialized as the message `content`, so llama-server receives the
/// multimodal payload intact; `content` then holds the concatenated
/// text parts, which is what the builtin fallback (and token accounting)
/// consumes. Images never reach the builtin — it is a text-only engine
/// and stays labeled `builtin-microeval`.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Raw OpenAI content-part array when the message arrived in
    /// multimodal form; `None` for plain-string messages.
    pub content_parts: Option<serde_json::Value>,
}

impl ChatMessage {
    /// Plain-text message constructor.
    pub fn text(role: &str, content: &str) -> Self {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            content_parts: None,
        }
    }

    /// Parse one OpenAI message object (`{"role": ..., "content": ...}`),
    /// accepting string or content-part-array content. Returns `None`
    /// when the role is missing.
    pub fn from_openai(value: &serde_json::Value) -> Option<Self> {
        let role = value["role"].as_str()?.to_string();
        Some(match &value["content"] {
            serde_json::Value::String(s) => ChatMessage {
                role,
                content: s.clone(),
                content_parts: None,
            },
            parts @ serde_json::Value::Array(_) => ChatMessage {
                role,
                content: extract_content_text(parts),
                content_parts: Some(parts.clone()),
            },
            _ => ChatMessage {
                role,
                content: String::new(),
                content_parts: None,
            },
        })
    }
}

/// Concatenate the text of OpenAI content parts: every
/// `{"type":"text","text":...}` part contributes its text (parts without
/// a `type` but with a string `text` field count too); image and other
/// parts contribute nothing. Never panics on malformed parts.
pub fn extract_content_text(parts: &serde_json::Value) -> String {
    let Some(arr) = parts.as_array() else {
        return String::new();
    };
    let mut out = Vec::new();
    for part in arr {
        if let Some(text) = part["text"].as_str()
            && !text.is_empty()
        {
            out.push(text.to_string());
        }
    }
    out.join("\n")
}

impl serde::Serialize for ChatMessage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("ChatMessage", 2)?;
        st.serialize_field("role", &self.role)?;
        match &self.content_parts {
            Some(parts) => st.serialize_field("content", parts)?,
            None => st.serialize_field("content", &self.content)?,
        }
        st.end()
    }
}

impl<'de> serde::Deserialize<'de> for ChatMessage {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Wire {
            role: String,
            #[serde(default)]
            content: Option<serde_json::Value>,
        }
        let wire = Wire::deserialize(deserializer)?;
        Ok(match &wire.content {
            Some(parts @ serde_json::Value::Array(_)) => ChatMessage {
                role: wire.role,
                content: extract_content_text(parts),
                content_parts: Some(parts.clone()),
            },
            Some(serde_json::Value::String(s)) => ChatMessage {
                role: wire.role,
                content: s.clone(),
                content_parts: None,
            },
            _ => ChatMessage {
                role: wire.role,
                content: String::new(),
                content_parts: None,
            },
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    pub loaded_model: Option<String>,
    pub backend: String,
    pub engine: String,
    pub idle_timeout_secs: u64,
    pub thermal_celsius: Option<f64>,
    pub thermal_state: ThermalState,
    pub internal_port: u16,
    /// A multimodal projector (`mmproj_path`) is loaded with the model.
    #[serde(default, alias = "mmproj_loaded")]
    pub has_vision: bool,
    /// The model was loaded with the `embedding` flag (`--embedding`).
    #[serde(default, alias = "embedding")]
    pub has_embeddings: bool,
}

/// Result of an embeddings request: one vector per input, in input order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingReply {
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: usize,
    pub backend: String,
    pub engine: String,
    /// Path of the model that produced these vectors (captured at
    /// inference time, like `ChatReply::model`).
    #[serde(default)]
    pub model: String,
}

/// Maximum inputs accepted in one embeddings request.
pub const MAX_EMBED_INPUTS: usize = 128;

enum EngineCommand {
    Load(LoadRequest, SyncSender<anyhow::Result<()>>),
    Unload(SyncSender<()>),
    Chat {
        messages: Vec<ChatMessage>,
        strip: bool,
        max_tokens: u32,
        resp: SyncSender<anyhow::Result<ChatReply>>,
    },
    VibeMark {
        budget_secs: u64,
        resp: SyncSender<anyhow::Result<VibeMarkResult>>,
    },
    Embeddings {
        inputs: Vec<String>,
        resp: SyncSender<anyhow::Result<EmbeddingReply>>,
    },
    Status(SyncSender<EngineStatus>),
    SetIdle(u64),
    Shutdown,
}

pub struct EngineConfig {
    pub llama_server_path: Option<PathBuf>,
    pub internal_port: u16,
    pub idle_secs: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            llama_server_path: None,
            internal_port: DEFAULT_INTERNAL_PORT,
            idle_secs: DEFAULT_IDLE_SECS,
        }
    }
}

pub struct Engine {
    tx: SyncSender<EngineCommand>,
    thermal: Arc<ThermalGovernor>,
    stopped: Arc<AtomicBool>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

struct LoadedModel {
    req: LoadRequest,
    child: Option<Child>,
    /// "llama-server" or "builtin-microeval"
    engine: String,
    last_activity: Instant,
}

impl Engine {
    pub fn start(config: EngineConfig) -> Arc<Engine> {
        let (tx, rx) = sync_channel::<EngineCommand>(COMMAND_CHANNEL_CAPACITY);
        let thermal = Arc::new(ThermalGovernor::new());
        let stopped = Arc::new(AtomicBool::new(false));
        let worker = {
            let thermal = Arc::clone(&thermal);
            let stopped = Arc::clone(&stopped);
            std::thread::Builder::new()
                .name("vibe-engine".into())
                .spawn(move || {
                    worker_loop(rx, config, thermal, stopped);
                })
                .expect("spawn engine worker")
        };
        Arc::new(Engine {
            tx,
            thermal,
            stopped,
            worker: Mutex::new(Some(worker)),
        })
    }

    pub fn thermal(&self) -> Arc<ThermalGovernor> {
        Arc::clone(&self.thermal)
    }

    fn call<T>(
        &self,
        make: impl FnOnce(SyncSender<anyhow::Result<T>>) -> EngineCommand,
        timeout: Duration,
    ) -> anyhow::Result<T> {
        let (resp_tx, resp_rx) = sync_channel(1);
        self.tx
            .send(make(resp_tx))
            .map_err(|_| anyhow::anyhow!("engine worker is not running"))?;
        resp_rx
            .recv_timeout(timeout)
            .map_err(|_| anyhow::anyhow!("engine request timed out"))?
    }

    pub fn load(&self, req: LoadRequest) -> anyhow::Result<()> {
        self.call(|tx| EngineCommand::Load(req, tx), Duration::from_secs(180))
    }

    pub fn unload(&self) {
        let (tx, rx) = sync_channel(1);
        if self.tx.send(EngineCommand::Unload(tx)).is_ok() {
            let _ = rx.recv_timeout(Duration::from_secs(30));
        }
    }

    pub fn chat(&self, prompt: &str, strip: bool, max_tokens: u32) -> anyhow::Result<ChatReply> {
        self.chat_messages(vec![ChatMessage::text("user", prompt)], strip, max_tokens)
    }

    /// Multi-turn chat: the full OpenAI message array reaches llama-server
    /// intact; the builtin fallback flattens it (labeled builtin-microeval).
    pub fn chat_messages(
        &self,
        messages: Vec<ChatMessage>,
        strip: bool,
        max_tokens: u32,
    ) -> anyhow::Result<ChatReply> {
        anyhow::ensure!(!messages.is_empty(), "messages must not be empty");
        for m in &messages {
            anyhow::ensure!(
                matches!(m.role.as_str(), "system" | "user" | "assistant"),
                "unsupported message role {:?} (system/user/assistant)",
                m.role
            );
        }
        self.call(
            |tx| EngineCommand::Chat {
                messages,
                strip,
                max_tokens,
                resp: tx,
            },
            Duration::from_secs(300),
        )
    }

    /// Embed a batch of texts with the loaded model. llama-server serves
    /// real model embeddings; the builtin fallback serves deterministic
    /// hash embeddings labeled `builtin-microeval`.
    pub fn embeddings(&self, inputs: Vec<String>) -> anyhow::Result<EmbeddingReply> {
        anyhow::ensure!(!inputs.is_empty(), "input must not be empty");
        anyhow::ensure!(
            inputs.len() <= MAX_EMBED_INPUTS,
            "input is limited to {MAX_EMBED_INPUTS} texts per request"
        );
        for input in &inputs {
            anyhow::ensure!(!input.trim().is_empty(), "input texts must not be empty");
        }
        self.call(
            |tx| EngineCommand::Embeddings { inputs, resp: tx },
            Duration::from_secs(300),
        )
    }

    pub fn vibemark(&self, budget_secs: u64) -> anyhow::Result<VibeMarkResult> {
        self.call(
            |tx| EngineCommand::VibeMark {
                budget_secs,
                resp: tx,
            },
            Duration::from_secs(budget_secs + 120),
        )
    }

    pub fn status(&self) -> anyhow::Result<EngineStatus> {
        let (tx, rx) = sync_channel(1);
        self.tx
            .send(EngineCommand::Status(tx))
            .map_err(|_| anyhow::anyhow!("engine worker is not running"))?;
        rx.recv_timeout(Duration::from_secs(10))
            .map_err(|_| anyhow::anyhow!("engine status timed out"))
    }

    pub fn set_idle_secs(&self, secs: u64) {
        let _ = self.tx.send(EngineCommand::SetIdle(secs.clamp(10, 86_400)));
    }

    pub fn shutdown(&self) {
        if self.stopped.swap(true, Ordering::SeqCst) {
            return;
        }
        let _ = self.tx.send(EngineCommand::Shutdown);
        if let Some(handle) = self.worker.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Locate a llama-server binary: explicit path > VIBE_LLAMA_SERVER env >
/// PATH > common local build locations.
pub fn find_llama_server(explicit: Option<&PathBuf>) -> Option<PathBuf> {
    if let Some(p) = explicit
        && p.is_file()
    {
        return Some(p.clone());
    }
    if let Ok(p) = std::env::var("VIBE_LLAMA_SERVER") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    let exe_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    if let Some(path) = std::env::var_os("PATH") {
        let found = std::env::split_paths(&path)
            .map(|dir| dir.join(exe_name))
            .find(|candidate| candidate.is_file());
        if found.is_some() {
            return found;
        }
    }
    [
        PathBuf::from("llama.cpp/build/bin").join(exe_name),
        PathBuf::from("llama.cpp").join(exe_name),
        PathBuf::from(exe_name),
    ]
    .into_iter()
    .find(|candidate| candidate.is_file())
}

fn worker_loop(
    rx: std::sync::mpsc::Receiver<EngineCommand>,
    config: EngineConfig,
    thermal: Arc<ThermalGovernor>,
    stopped: Arc<AtomicBool>,
) {
    let mut loaded: Option<LoadedModel> = None;
    let mut idle_secs = config.idle_secs;
    let mut last_thermal_poll = Instant::now();
    let micro = MicroEval::default();

    loop {
        if stopped.load(Ordering::SeqCst) {
            break;
        }
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(cmd) => match cmd {
                EngineCommand::Shutdown => break,
                EngineCommand::SetIdle(secs) => idle_secs = secs,
                EngineCommand::Status(tx) => {
                    let _ = tx.send(status_of(
                        &loaded,
                        idle_secs,
                        &thermal,
                        config.internal_port,
                    ));
                }
                EngineCommand::Unload(tx) => {
                    unload_model(&mut loaded);
                    let _ = tx.send(());
                }
                EngineCommand::Load(req, tx) => {
                    unload_model(&mut loaded);
                    let result = load_model(&req, &config, &micro);
                    match result {
                        Ok(model) => {
                            loaded = Some(model);
                            let _ = tx.send(Ok(()));
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e));
                        }
                    }
                }
                EngineCommand::Chat {
                    messages,
                    strip,
                    max_tokens,
                    resp,
                } => {
                    thermal.poll();
                    let result = match loaded.as_mut() {
                        None => Err(anyhow::anyhow!("no model loaded; POST /api/load first")),
                        Some(_) if !thermal.allows_inference() => Err(anyhow::anyhow!(
                            "thermal pause: device at >=45°C; inference resumes below 43°C"
                        )),
                        Some(model) => {
                            model.last_activity = Instant::now();
                            let delay = thermal.throttle_delay_ms();
                            if delay > 0 {
                                std::thread::sleep(Duration::from_millis(delay));
                            }
                            let effective_max = if thermal.state() == ThermalState::Throttled {
                                max_tokens / 2
                            } else {
                                max_tokens
                            };
                            chat_with(
                                model,
                                &messages,
                                strip,
                                effective_max.max(16),
                                &micro,
                                config.internal_port,
                            )
                        }
                    };
                    let _ = resp.send(result);
                }
                EngineCommand::VibeMark { budget_secs, resp } => {
                    thermal.poll();
                    let result = match loaded.as_mut() {
                        None => Err(anyhow::anyhow!(
                            "no model loaded; load a GGUF before benchmarking"
                        )),
                        Some(_) if !thermal.allows_inference() => Err(anyhow::anyhow!(
                            "thermal pause: benchmark blocked at >=45°C"
                        )),
                        Some(model) => {
                            model.last_activity = Instant::now();
                            run_vibemark(model, budget_secs, &micro, config.internal_port)
                        }
                    };
                    let _ = resp.send(result);
                }
                EngineCommand::Embeddings { inputs, resp } => {
                    thermal.poll();
                    let result = match loaded.as_mut() {
                        None => Err(anyhow::anyhow!("no model loaded; POST /api/load first")),
                        Some(_) if !thermal.allows_inference() => Err(anyhow::anyhow!(
                            "thermal pause: device at >=45°C; inference resumes below 43°C"
                        )),
                        Some(model) => {
                            model.last_activity = Instant::now();
                            embed_with(model, &inputs, config.internal_port)
                        }
                    };
                    let _ = resp.send(result);
                }
            },
            Err(RecvTimeoutError::Timeout) => {
                if last_thermal_poll.elapsed() >= Duration::from_secs(5) {
                    thermal.poll();
                    last_thermal_poll = Instant::now();
                }
                let idle_expired = loaded.as_ref().is_some_and(|model| {
                    model.last_activity.elapsed() >= Duration::from_secs(idle_secs)
                });
                if idle_expired {
                    eprintln!(
                        "[vibe-engine] idle {}s reached, auto-unloading model",
                        idle_secs
                    );
                    unload_model(&mut loaded);
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    unload_model(&mut loaded);
}

fn status_of(
    loaded: &Option<LoadedModel>,
    idle_secs: u64,
    thermal: &ThermalGovernor,
    port: u16,
) -> EngineStatus {
    EngineStatus {
        loaded_model: loaded.as_ref().map(|m| m.req.path.clone()),
        backend: loaded
            .as_ref()
            .map(|m| m.req.backend.clone())
            .unwrap_or_else(|| "none".into()),
        engine: loaded
            .as_ref()
            .map(|m| m.engine.clone())
            .unwrap_or_else(|| "none".into()),
        idle_timeout_secs: idle_secs,
        thermal_celsius: thermal.current_reading(),
        thermal_state: thermal.state(),
        internal_port: port,
        has_vision: loaded.as_ref().is_some_and(|m| m.req.mmproj_path.is_some()),
        has_embeddings: loaded.as_ref().is_some_and(|m| m.req.embedding),
    }
}

fn unload_model(loaded: &mut Option<LoadedModel>) {
    if let Some(mut model) = loaded.take()
        && let Some(child) = model.child.as_mut()
    {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn load_model(
    req: &LoadRequest,
    config: &EngineConfig,
    micro: &MicroEval,
) -> anyhow::Result<LoadedModel> {
    let path = PathBuf::from(&req.path);
    anyhow::ensure!(path.is_file(), "model file not found: {}", req.path);
    // Validate the GGUF header before committing to a load.
    let info = vibe_local_core::gguf::parse_file(&path)?;
    let mut req = req.clone();
    if req.ctx == 0 {
        req.ctx = auto_context(
            &info,
            &req,
            &vibe_local_core::hardware::HardwareProbe::scan(),
        );
        eprintln!("[vibe-engine] auto context window: {} tokens", req.ctx);
    }
    if let Some(draft) = &req.draft_model_path {
        let draft_path = PathBuf::from(draft);
        anyhow::ensure!(draft_path.is_file(), "draft model file not found: {draft}");
        vibe_local_core::gguf::parse_file(&draft_path)?;
    }
    if let Some(mmproj) = &req.mmproj_path {
        let mmproj_path = PathBuf::from(mmproj);
        anyhow::ensure!(mmproj_path.is_file(), "mmproj file not found: {mmproj}");
        vibe_local_core::gguf::parse_file(&mmproj_path)?;
    }
    let _ = micro.ingest(&info.architecture);

    let Some(binary) = find_llama_server(config.llama_server_path.as_ref()) else {
        return Ok(LoadedModel {
            req,
            child: None,
            engine: "builtin-microeval".into(),
            last_activity: Instant::now(),
        });
    };

    let mut cmd = Command::new(&binary);
    cmd.arg("-m")
        .arg(&req.path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(config.internal_port.to_string())
        .arg("-ngl")
        .arg(req.ngl.to_string())
        .arg("-c")
        .arg(req.ctx.to_string())
        .arg("-np")
        .arg("1");
    if let Some(draft) = &req.draft_model_path {
        cmd.arg("--model-draft")
            .arg(draft)
            .arg("--gpu-layers-draft")
            .arg(req.draft_ngl.unwrap_or(0).to_string())
            .arg("--draft-max")
            .arg(req.draft_max.unwrap_or(DEFAULT_DRAFT_MAX).to_string());
    }
    if let Some(mmproj) = &req.mmproj_path {
        cmd.arg("--mmproj").arg(mmproj);
    }
    if req.embedding {
        cmd.arg("--embedding");
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    match req.backend.as_str() {
        "cuda" => {
            cmd.env("CUDA_VISIBLE_DEVICES", "0");
        }
        "vulkan" => {
            cmd.env("GGML_VK_DEVICE", "0");
        }
        _ => {
            cmd.env("CUDA_VISIBLE_DEVICES", "");
        }
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn llama-server: {e}"))?;

    // Poll /health until the model is resident (large models take a while).
    let health_url = format!("http://127.0.0.1:{}/health", config.internal_port);
    let deadline = Instant::now() + Duration::from_secs(150);
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(3))
        .build();
    let mut healthy = false;
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().ok().flatten() {
            anyhow::bail!("llama-server exited during load (status {status})");
        }
        if let Ok(resp) = agent.get(&health_url).call()
            && resp.status() == 200
        {
            healthy = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    if !healthy {
        let _ = child.kill();
        let _ = child.wait();
        anyhow::bail!("llama-server did not become healthy within 150s");
    }
    Ok(LoadedModel {
        req,
        child: Some(child),
        engine: "llama-server".into(),
        last_activity: Instant::now(),
    })
}

fn chat_with(
    model: &mut LoadedModel,
    messages: &[ChatMessage],
    strip: bool,
    max_tokens: u32,
    micro: &MicroEval,
    port: u16,
) -> anyhow::Result<ChatReply> {
    let start = Instant::now();
    let (text, tokens) = if model.engine == "llama-server" {
        let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
        let body = serde_json::json!({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": false,
        });
        let resp = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(240))
            .build()
            .post(&url)
            .send_json(body)?;
        let json: serde_json::Value = resp.into_json()?;
        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        (text, tokens)
    } else {
        // Builtin fallback has no chat template: flatten like the Vibe-Link
        // server's extract_prompt (System: prefix + user turns).
        let mut parts = Vec::new();
        for m in messages {
            if m.role == "system" {
                parts.push(format!("System: {}", m.content));
            } else if m.role == "user" {
                parts.push(m.content.clone());
            }
        }
        let text = micro.generate_text(&parts.join("\n"));
        let tokens = tokenize(&text).len() as u64;
        (text, tokens)
    };
    let elapsed = start.elapsed().as_secs_f64();
    let text = if strip { strip_think(&text) } else { text };
    let tokens = if strip {
        tokenize(&text).len() as u64
    } else {
        tokens
    };
    Ok(ChatReply {
        tok_per_sec: if elapsed > 0.0 {
            tokens as f64 / elapsed
        } else {
            0.0
        },
        text,
        tokens_generated: tokens,
        elapsed_secs: elapsed,
        backend: model.req.backend.clone(),
        engine: model.engine.clone(),
        model: model.req.path.clone(),
    })
}

/// One inference probe used by VibeMark. `stream` requests TTFT measurement.
fn probe(
    model: &LoadedModel,
    prompt: &str,
    n_predict: u32,
    stream: bool,
    micro: &MicroEval,
    port: u16,
) -> anyhow::Result<ProbeOutcome> {
    if model.engine == "llama-server" {
        let url = format!("http://127.0.0.1:{port}/completion");
        let body = serde_json::json!({
            "prompt": prompt,
            "n_predict": n_predict,
            "stream": stream,
            "cache_prompt": false,
            "temperature": 0.0,
        });
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(240))
            .build();
        let start = Instant::now();
        let resp = agent.post(&url).send_json(body)?;
        if stream {
            // Read the SSE stream until the first content chunk (TTFT),
            // then drain the rest to let the server finish cleanly.
            use std::io::{BufRead, BufReader};
            let mut reader = BufReader::new(resp.into_reader());
            let mut ttft_ms = None;
            let mut predicted: u64 = 0;
            let mut line = String::new();
            loop {
                line.clear();
                if reader.read_line(&mut line)? == 0 {
                    break;
                }
                let trimmed = line.trim();
                let Some(data) = trimmed.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    break;
                }
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    let content = json["content"].as_str().unwrap_or("");
                    if !content.is_empty() {
                        if ttft_ms.is_none() {
                            ttft_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                        }
                        predicted += tokenize(content).len() as u64;
                    }
                    if json["stop"].as_bool() == Some(true) {
                        break;
                    }
                }
            }
            let total_ms = start.elapsed().as_secs_f64() * 1000.0;
            let ttft = ttft_ms.unwrap_or(total_ms);
            Ok(ProbeOutcome {
                prompt_tokens: tokenize(prompt).len() as u64,
                prompt_ms: ttft,
                predicted_tokens: predicted.max(1),
                predicted_ms: (total_ms - ttft).max(0.001),
                ttft_ms: Some(ttft),
                text: String::new(),
            })
        } else {
            let json: serde_json::Value = resp.into_json()?;
            let timings = &json["timings"];
            let prompt_n = timings["prompt_n"]
                .as_u64()
                .unwrap_or_else(|| tokenize(prompt).len() as u64);
            let prompt_ms = timings["prompt_ms"].as_f64().unwrap_or(0.0);
            let predicted_n = timings["predicted_n"].as_u64().unwrap_or(0);
            let predicted_ms = timings["predicted_ms"].as_f64().unwrap_or(0.0);
            Ok(ProbeOutcome {
                prompt_tokens: prompt_n,
                prompt_ms: prompt_ms.max(0.001),
                predicted_tokens: predicted_n,
                predicted_ms: predicted_ms.max(0.001),
                ttft_ms: None,
                text: json["content"].as_str().unwrap_or("").to_string(),
            })
        }
    } else {
        // Builtin micro-eval: real measured work (ingest + generation).
        let t0 = Instant::now();
        let prompt_tokens = micro.ingest(prompt) as u64;
        let prompt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();
        let tokens = micro.generate(prompt, n_predict as usize);
        let predicted_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let text = tokens.join(" ");
        Ok(ProbeOutcome {
            prompt_tokens,
            prompt_ms: prompt_ms.max(0.001),
            predicted_tokens: tokens.len() as u64,
            predicted_ms: predicted_ms.max(0.001),
            ttft_ms: Some(prompt_ms + predicted_ms / tokens.len().max(1) as f64),
            text,
        })
    }
}

fn embed_with(model: &LoadedModel, inputs: &[String], port: u16) -> anyhow::Result<EmbeddingReply> {
    let prompt_tokens: usize = inputs.iter().map(|t| tokenize(t).len()).sum();
    let vectors = if model.engine == "llama-server" {
        let url = format!("http://127.0.0.1:{port}/v1/embeddings");
        let body = serde_json::json!({ "input": inputs });
        let resp = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(240))
            .build()
            .post(&url)
            .send_json(body)?;
        let json: serde_json::Value = resp.into_json()?;
        let data = json["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("embeddings: response has no data array"))?;
        anyhow::ensure!(
            data.len() == inputs.len(),
            "embeddings: got {} vectors for {} inputs",
            data.len(),
            inputs.len()
        );
        let mut ordered: Vec<Option<Vec<f32>>> = vec![None; inputs.len()];
        for item in data {
            let idx = item["index"].as_u64().unwrap_or(0) as usize;
            anyhow::ensure!(idx < inputs.len(), "embeddings: index {idx} out of range");
            let vec: Vec<f32> = item["embedding"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("embeddings: missing embedding array"))?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            anyhow::ensure!(!vec.is_empty(), "embeddings: empty vector at index {idx}");
            ordered[idx] = Some(vec);
        }
        ordered
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow::anyhow!("embeddings: missing vector in response"))?
    } else {
        inputs
            .iter()
            .map(|t| crate::microeval::deterministic_embedding(t, crate::microeval::DET_EMBED_DIM))
            .collect()
    };
    Ok(EmbeddingReply {
        vectors,
        prompt_tokens,
        backend: model.req.backend.clone(),
        engine: model.engine.clone(),
        model: model.req.path.clone(),
    })
}

fn run_vibemark(
    model: &mut LoadedModel,
    budget_secs: u64,
    micro: &MicroEval,
    port: u16,
) -> anyhow::Result<VibeMarkResult> {
    let model_name = std::path::Path::new(&model.req.path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.req.path.clone());
    let pid = model.child.as_ref().map(|c| c.id());
    let engine = model.engine.clone();
    let backend = model.req.backend.clone();
    let sampler = move || -> Option<f64> {
        match pid {
            Some(pid) => util::process_rss_mb(pid),
            None => util::process_rss_mb(std::process::id()),
        }
    };
    let result = vibemark::run(
        budget_secs,
        &model_name,
        &backend,
        &engine,
        |prompt, n, stream| probe(model, prompt, n, stream, micro, port),
        sampler,
    )?;
    model.last_activity = Instant::now();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_lifecycle_builtin() {
        // Create a tiny synthetic GGUF so load validates the header.
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("tiny-Q4_K_M.gguf");
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
        std::fs::write(&model_path, bytes).unwrap();

        let engine = Engine::start(EngineConfig {
            // Force builtin mode even if llama-server exists on PATH.
            llama_server_path: None,
            internal_port: 20199,
            idle_secs: 300,
        });
        // Without a llama-server binary the engine uses builtin micro-eval.
        // If one IS installed this test still passes via the real server path
        // only when the model is valid; tiny synthetic file would fail there,
        // so gate on binary presence.
        if find_llama_server(None).is_some() {
            engine.shutdown();
            return;
        }
        engine
            .load(LoadRequest {
                path: model_path.to_string_lossy().into_owned(),
                backend: "cpu".into(),
                ngl: 0,
                ctx: 2048,
                draft_model_path: None,
                draft_ngl: None,
                draft_max: None,
                mmproj_path: None,
                embedding: false,
            })
            .unwrap();
        let status = engine.status().unwrap();
        assert_eq!(status.engine, "builtin-microeval");
        assert!(status.loaded_model.is_some());

        let reply = engine.chat("What is 21 * 2?", true, 128).unwrap();
        assert!(reply.text.contains("42"), "got {}", reply.text);
        assert!(reply.tok_per_sec > 0.0);
        assert_eq!(reply.engine, "builtin-microeval");

        engine.unload();
        assert!(engine.status().unwrap().loaded_model.is_none());
        assert!(engine.chat("hello", false, 32).is_err());
        engine.shutdown();
    }

    #[test]
    fn chat_messages_multi_turn_builtin() {
        let engine = Engine::start(EngineConfig::default());
        // Boundary validation happens before the worker is touched.
        assert!(engine.chat_messages(vec![], true, 64).is_err());
        assert!(
            engine
                .chat_messages(vec![ChatMessage::text("tool", "x")], true, 64)
                .is_err()
        );
        if find_llama_server(None).is_some() {
            engine.shutdown();
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("tiny-Q4_K_M.gguf");
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
        std::fs::write(&model_path, bytes).unwrap();
        engine
            .load(LoadRequest {
                path: model_path.to_string_lossy().into_owned(),
                backend: "cpu".into(),
                ngl: 0,
                ctx: 0,
                draft_model_path: None,
                draft_ngl: None,
                draft_max: None,
                mmproj_path: None,
                embedding: false,
            })
            .unwrap();
        let reply = engine
            .chat_messages(
                vec![
                    ChatMessage::text("system", "You are a calculator."),
                    ChatMessage::text("user", "What is 21 * 2?"),
                ],
                true,
                128,
            )
            .unwrap();
        // The builtin fallback flattens turns; with a System prefix the
        // pure-arithmetic path no longer applies, but a reply must exist.
        assert!(!reply.text.is_empty());
        assert_eq!(reply.engine, "builtin-microeval");
        // A single user turn behaves exactly like chat().
        let single = engine
            .chat_messages(
                vec![ChatMessage::text("user", "What is 21 * 2?")],
                true,
                128,
            )
            .unwrap();
        assert!(single.text.contains("42"), "got {}", single.text);
        engine.shutdown();
    }

    #[test]
    fn chat_message_multimodal_parsing_and_serialization() {
        // OpenAI multimodal content array: text extracted, parts kept.
        let payload = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
            ]
        });
        let msg = ChatMessage::from_openai(&payload).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "What is in this image?");
        assert!(msg.content_parts.is_some());
        // Serialization restores the array intact for llama-server.
        let wire = serde_json::to_value(&msg).unwrap();
        assert_eq!(wire["content"], payload["content"]);
        // Deserialization (the JNI chatMessages path) behaves identically.
        let back: ChatMessage = serde_json::from_value(payload.clone()).unwrap();
        assert_eq!(back.content, "What is in this image?");
        assert_eq!(back.content_parts, msg.content_parts);
        // Plain strings stay plain strings end to end.
        let plain = ChatMessage::text("user", "hello");
        let wire = serde_json::to_value(&plain).unwrap();
        assert_eq!(wire["content"], serde_json::json!("hello"));
        let back: ChatMessage = serde_json::from_value(wire).unwrap();
        assert!(back.content_parts.is_none());
        // Malformed parts never panic and contribute no text.
        assert_eq!(
            extract_content_text(&serde_json::json!([
                {"type": "image_url"},
                {"text": 42},
                "junk",
                {"type": "text", "text": "kept"}
            ])),
            "kept"
        );
        assert_eq!(extract_content_text(&serde_json::json!("nope")), "");
    }

    #[test]
    fn chat_messages_multimodal_builtin_uses_text_part() {
        if find_llama_server(None).is_some() {
            return; // builtin-only assertion
        }
        let engine = Engine::start(EngineConfig::default());
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("tiny-Q4_K_M.gguf");
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
        std::fs::write(&model_path, bytes).unwrap();
        engine
            .load(LoadRequest {
                path: model_path.to_string_lossy().into_owned(),
                backend: "cpu".into(),
                ngl: 0,
                ctx: 0,
                draft_model_path: None,
                draft_ngl: None,
                draft_max: None,
                mmproj_path: None,
                embedding: false,
            })
            .unwrap();
        let msg = ChatMessage::from_openai(&serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "What is 21 * 2?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
            ]
        }))
        .unwrap();
        let reply = engine.chat_messages(vec![msg], true, 128).unwrap();
        // The builtin answered from the extracted text part alone.
        assert!(reply.text.contains("42"), "got {}", reply.text);
        assert_eq!(reply.engine, "builtin-microeval");
        engine.shutdown();
    }

    #[test]
    fn load_missing_file_fails() {
        let engine = Engine::start(EngineConfig::default());
        let err = engine
            .load(LoadRequest {
                path: "/nonexistent/model.gguf".into(),
                backend: "cpu".into(),
                ngl: 0,
                ctx: 2048,
                draft_model_path: None,
                draft_ngl: None,
                draft_max: None,
                mmproj_path: None,
                embedding: false,
            })
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
        engine.shutdown();
    }

    #[test]
    fn idle_timeout_config() {
        let engine = Engine::start(EngineConfig {
            idle_secs: 42,
            ..Default::default()
        });
        assert_eq!(engine.status().unwrap().idle_timeout_secs, 42);
        engine.set_idle_secs(600);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(engine.status().unwrap().idle_timeout_secs, 600);
        engine.shutdown();
    }

    fn synthetic_info() -> vibe_local_core::gguf::GgufInfo {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&vibe_local_core::gguf::GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let mut info = vibe_local_core::gguf::parse_reader(std::io::Cursor::new(bytes)).unwrap();
        info.ram = vec![
            vibe_local_core::gguf::ContextRam {
                ctx: 2048,
                weights_mb: 4000.0,
                kv_cache_mb: 100.0,
                compute_mb: 200.0,
                total_mb: 4300.0,
            },
            vibe_local_core::gguf::ContextRam {
                ctx: 4096,
                weights_mb: 4000.0,
                kv_cache_mb: 200.0,
                compute_mb: 200.0,
                total_mb: 4400.0,
            },
            vibe_local_core::gguf::ContextRam {
                ctx: 8192,
                weights_mb: 4000.0,
                kv_cache_mb: 400.0,
                compute_mb: 200.0,
                total_mb: 4600.0,
            },
            vibe_local_core::gguf::ContextRam {
                ctx: 16384,
                weights_mb: 4000.0,
                kv_cache_mb: 800.0,
                compute_mb: 200.0,
                total_mb: 5000.0,
            },
        ];
        info
    }

    fn req_for(ctx: u32, ngl: u32, backend: &str) -> LoadRequest {
        LoadRequest {
            path: "m.gguf".into(),
            backend: backend.into(),
            ngl,
            ctx,
            draft_model_path: None,
            draft_ngl: None,
            draft_max: None,
            mmproj_path: None,
            embedding: false,
        }
    }

    #[test]
    fn auto_context_picks_largest_safe_window() {
        let info = synthetic_info();
        let mut hw = vibe_local_core::hardware::HardwareProbe::scan();
        hw.ram_free_mb = 5200; // 5200 - 512 margin = 4688 -> 8192 fits, 16384 not
        hw.gpus.clear();
        assert_eq!(auto_context(&info, &req_for(0, 0, "cpu"), &hw), 8192);
        // VRAM path: 8GB card offloading -> 16384 (5000MB) fits in 8192-512.
        hw.gpus.push(vibe_local_core::hardware::GpuInfo {
            name: "Test GPU".into(),
            vendor: "test".into(),
            vram_mb: Some(8192),
            backend: "vulkan".into(),
        });
        assert_eq!(auto_context(&info, &req_for(0, 99, "vulkan"), &hw), 16384);
        // Tiny budget: nothing fits -> smallest estimate, never a fake big ctx.
        hw.ram_free_mb = 1024;
        hw.gpus.clear();
        assert_eq!(auto_context(&info, &req_for(0, 0, "cpu"), &hw), 2048);
        // Trained context cap is respected.
        let mut capped = synthetic_info();
        capped.context_length = Some(4096);
        hw.ram_free_mb = 32000;
        assert_eq!(auto_context(&capped, &req_for(0, 0, "cpu"), &hw), 4096);
        // No estimates at all -> conservative 2048.
        let mut empty = synthetic_info();
        empty.ram.clear();
        assert_eq!(auto_context(&empty, &req_for(0, 0, "cpu"), &hw), 2048);
    }

    #[test]
    fn load_request_serde_defaults() {
        let req: LoadRequest = serde_json::from_str(r#"{"path":"m.gguf"}"#).unwrap();
        assert_eq!(req.ctx, 0, "omitted ctx must mean auto-size");
        assert_eq!(req.backend, "cpu");
        assert!(req.draft_model_path.is_none());
        let full: LoadRequest = serde_json::from_str(
            r#"{"path":"m.gguf","ctx":8192,"draft_model_path":"d.gguf","draft_ngl":99,"draft_max":16}"#,
        )
        .unwrap();
        assert_eq!(full.draft_max, Some(16));
        assert_eq!(full.draft_ngl, Some(99));
        // mmproj / embedding default to absent/false when omitted.
        assert!(req.mmproj_path.is_none());
        assert!(!req.embedding);
        let vision: LoadRequest = serde_json::from_str(
            r#"{"path":"m.gguf","mmproj_path":"mmproj.gguf","embedding":true}"#,
        )
        .unwrap();
        assert_eq!(vision.mmproj_path.as_deref(), Some("mmproj.gguf"));
        assert!(vision.embedding);
        // And they serialize back only when set.
        let plain = serde_json::to_string(&req).unwrap();
        assert!(!plain.contains("mmproj_path"));
        assert!(!plain.contains("embedding"));
        let vision_json = serde_json::to_string(&vision).unwrap();
        assert!(vision_json.contains("\"mmproj_path\":\"mmproj.gguf\""));
        assert!(vision_json.contains("\"embedding\":true"));
    }

    #[test]
    fn draft_model_validated_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("tiny-Q4_K_M.gguf");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&vibe_local_core::gguf::GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&model_path, &bytes).unwrap();

        let engine = Engine::start(EngineConfig::default());
        // Missing draft file is rejected before any spawn.
        let err = engine
            .load(LoadRequest {
                path: model_path.to_string_lossy().into_owned(),
                draft_model_path: Some(dir.path().join("nope.gguf").to_string_lossy().into_owned()),
                ..req_for(2048, 0, "cpu")
            })
            .unwrap_err();
        assert!(err.to_string().contains("draft model file not found"));
        // Invalid draft header is rejected too.
        let bad_draft = dir.path().join("bad.gguf");
        std::fs::write(&bad_draft, b"not a gguf").unwrap();
        assert!(
            engine
                .load(LoadRequest {
                    path: model_path.to_string_lossy().into_owned(),
                    draft_model_path: Some(bad_draft.to_string_lossy().into_owned()),
                    ..req_for(2048, 0, "cpu")
                })
                .is_err()
        );
        if find_llama_server(None).is_none() {
            // Valid draft loads in builtin mode; ctx 0 auto-resolves.
            let draft = dir.path().join("draft-Q4_K_M.gguf");
            std::fs::write(&draft, &bytes).unwrap();
            engine
                .load(LoadRequest {
                    path: model_path.to_string_lossy().into_owned(),
                    draft_model_path: Some(draft.to_string_lossy().into_owned()),
                    draft_max: Some(16),
                    ..req_for(0, 0, "cpu")
                })
                .unwrap();
            assert!(engine.status().unwrap().loaded_model.is_some());
            engine.unload();
        }
        engine.shutdown();
    }

    fn tiny_gguf(dir: &std::path::Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&vibe_local_core::gguf::GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    #[test]
    fn mmproj_validated_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = tiny_gguf(dir.path(), "tiny-Q4_K_M.gguf");
        let engine = Engine::start(EngineConfig::default());
        // Missing mmproj file is rejected before any spawn.
        let err = engine
            .load(LoadRequest {
                path: model_path.to_string_lossy().into_owned(),
                mmproj_path: Some(
                    dir.path()
                        .join("no-mmproj.gguf")
                        .to_string_lossy()
                        .into_owned(),
                ),
                ..req_for(2048, 0, "cpu")
            })
            .unwrap_err();
        assert!(err.to_string().contains("mmproj file not found"), "{err}");
        // Invalid mmproj header is rejected too.
        let bad = dir.path().join("bad-mmproj.gguf");
        std::fs::write(&bad, b"not a gguf").unwrap();
        assert!(
            engine
                .load(LoadRequest {
                    path: model_path.to_string_lossy().into_owned(),
                    mmproj_path: Some(bad.to_string_lossy().into_owned()),
                    ..req_for(2048, 0, "cpu")
                })
                .is_err()
        );
        if find_llama_server(None).is_none() {
            // Valid mmproj + embedding flag load in builtin mode and are
            // reported in the engine status.
            let mmproj = tiny_gguf(dir.path(), "mmproj-F16.gguf");
            engine
                .load(LoadRequest {
                    path: model_path.to_string_lossy().into_owned(),
                    mmproj_path: Some(mmproj.to_string_lossy().into_owned()),
                    embedding: true,
                    ..req_for(2048, 0, "cpu")
                })
                .unwrap();
            let status = engine.status().unwrap();
            assert!(status.has_vision);
            assert!(status.has_embeddings);
            engine.unload();
            let status = engine.status().unwrap();
            assert!(!status.has_vision);
            assert!(!status.has_embeddings);
        }
        engine.shutdown();
    }

    #[test]
    fn embeddings_builtin_deterministic() {
        let engine = Engine::start(EngineConfig::default());
        // No model loaded yet.
        assert!(engine.embeddings(vec!["hello".into()]).is_err());
        // Boundary validation happens before the worker is touched.
        assert!(engine.embeddings(vec![]).is_err());
        assert!(engine.embeddings(vec!["  ".into()]).is_err());
        assert!(
            engine
                .embeddings(vec!["x".to_string(); MAX_EMBED_INPUTS + 1])
                .is_err()
        );
        if find_llama_server(None).is_some() {
            engine.shutdown();
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let model_path = tiny_gguf(dir.path(), "tiny-Q4_K_M.gguf");
        engine
            .load(LoadRequest {
                path: model_path.to_string_lossy().into_owned(),
                embedding: true,
                ..req_for(2048, 0, "cpu")
            })
            .unwrap();
        let reply = engine
            .embeddings(vec!["hello world".into(), "another text".into()])
            .unwrap();
        assert_eq!(reply.engine, "builtin-microeval");
        assert_eq!(reply.vectors.len(), 2);
        assert_eq!(reply.vectors[0].len(), crate::microeval::DET_EMBED_DIM);
        assert_ne!(reply.vectors[0], reply.vectors[1]);
        assert!(reply.prompt_tokens > 0);
        // Deterministic across calls.
        let again = engine.embeddings(vec!["hello world".into()]).unwrap();
        assert_eq!(again.vectors[0], reply.vectors[0]);
        engine.shutdown();
    }
}
