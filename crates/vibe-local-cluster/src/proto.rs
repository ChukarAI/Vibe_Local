//! Wire types for the Vibe Grid cluster protocol — JSON over HTTP,
//! field-compatible with compost v0.7.0 (`/api/v1/*`). Byte payloads use
//! base64 (Go `[]byte` JSON convention); durations are integer milliseconds.

use std::collections::BTreeMap;

use base64::Engine;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub const VERSION: &str = "0.7.0";

pub const TASK_MANDELBROT: &str = "mandelbrot";
pub const TASK_PRIMES: &str = "primes";
pub const TASK_WORDCOUNT: &str = "wordcount";
pub const TASK_LLM_BATCH: &str = "llm_batch";
pub const TASK_MOE_DISTRIBUTED: &str = "moe_distributed";

pub const JOB_QUEUED: &str = "queued";
pub const JOB_RUNNING: &str = "running";
pub const JOB_DONE: &str = "done";
pub const JOB_FAILED: &str = "failed";
pub const JOB_CANCELLED: &str = "cancelled";

pub mod b64 {
    use super::*;
    pub fn serialize<S: Serializer>(v: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(bytes) if !bytes.is_empty() => {
                s.serialize_some(&base64::engine::general_purpose::STANDARD.encode(bytes))
            }
            _ => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<u8>>, D::Error> {
        let opt: Option<String> = Option::deserialize(d)?;
        match opt {
            Some(s) if !s.is_empty() => base64::engine::general_purpose::STANDARD
                .decode(s.as_bytes())
                .map(Some)
                .map_err(serde::de::Error::custom),
            _ => Ok(None),
        }
    }
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}
fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}
fn is_zero_i64(v: &i64) -> bool {
    *v == 0
}
fn is_zero_i32(v: &i32) -> bool {
    *v == 0
}
fn is_zero_usize(v: &usize) -> bool {
    *v == 0
}
fn is_false(v: &bool) -> bool {
    !*v
}
fn is_empty_str(v: &str) -> bool {
    v.is_empty()
}
fn is_empty_map(v: &BTreeMap<String, f64>) -> bool {
    v.is_empty()
}
fn is_empty_smap(v: &BTreeMap<String, String>) -> bool {
    v.is_empty()
}
fn is_empty_ps(v: &BTreeMap<String, f64>) -> bool {
    v.is_empty()
}

/// Caps describes what a device can do right now; refreshed on every poll.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Caps {
    pub cores: usize,
    pub mem_mb: u64,
    /// Relative compute benchmark (~1.0 = one modern-ish core).
    pub score: f64,
    /// -1 when unknown / no battery.
    pub battery_pct: i32,
    pub on_battery: bool,
    /// -1 when unknown.
    pub temp_c: f64,
    pub arch: String,
    pub os: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub has_local_llm: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub local_llm_models: Vec<String>,
    /// Task specializations this node can serve: "llm", "vision", "stt",
    /// "tts", "image_gen", "video_gen", "embed_rag". Empty = general compute
    /// (compost v0.7.0 agents that predate capability routing).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    /// Models currently resident on this node (advertised, not verified).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loaded_models: Vec<String>,
}

/// Known capability labels for heterogeneous task routing.
pub const CAPABILITIES: [&str; 7] = [
    "llm",
    "vision",
    "stt",
    "tts",
    "image_gen",
    "video_gen",
    "embed_rag",
];

/// Node is a registered device as seen by the coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    #[serde(default, skip_serializing_if = "is_empty_str")]
    pub node_uuid: String,
    pub name: String,
    pub caps: Caps,
    pub joined_at: String,
    pub last_seen: String,
    pub online: bool,
    pub rate: f64,
    pub running: usize,
    pub done: u64,
    pub failed: u64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub speed_factor: f64,
    #[serde(default, skip_serializing_if = "is_empty_ps")]
    pub preset_speeds: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "is_empty_str")]
    pub paused: String,
}

/// Task is a unit of work leased to a specific node. Idempotent: if a node
/// dies the coordinator silently requeues the same task for another.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Task {
    pub id: String,
    pub job_id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub lease_epoch: u64,

    // Mandelbrot parameters.
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub width: i32,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub height: i32,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub x0: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub y0: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub x1: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub y1: f64,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub max_iter: i32,

    // Primes / chunk parameters.
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub start: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub count: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub end: i64,

    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub est_millis: i64,

    #[serde(default, skip_serializing_if = "Option::is_none", with = "b64")]
    pub input: Option<Vec<u8>>,

    #[serde(default, skip_serializing_if = "is_false")]
    pub verify_mode: bool,
}

/// Result carries completed work back to the coordinator.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResultMsg {
    pub task_id: String,
    #[serde(default)]
    pub node_id: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub lease_epoch: u64,
    #[serde(default, skip_serializing_if = "Option::is_none", with = "b64")]
    pub data: Option<Vec<u8>>,
    /// Deprecated alias for `data` (pre-0.5 agents).
    #[serde(default, skip_serializing_if = "Option::is_none", with = "b64")]
    pub pixels: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "is_empty_str")]
    pub error: String,
    pub millis: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub exec_millis: i64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub unsupported: bool,
}

impl ResultMsg {
    pub fn payload(&self) -> Vec<u8> {
        self.data
            .clone()
            .or_else(|| self.pixels.clone())
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinRequest {
    pub name: String,
    #[serde(default)]
    pub node_uuid: String,
    pub caps: Caps,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinResponse {
    pub node_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollRequest {
    pub node_id: String,
    #[serde(default)]
    pub node_uuid: String,
    #[serde(default)]
    pub name: String,
    pub caps: Caps,
    #[serde(default)]
    pub running: usize,
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub workers: usize,
    #[serde(default, skip_serializing_if = "is_empty_str")]
    pub paused: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PollResponse {
    /// Singular form kept for pre-0.5 agent compatibility (null when absent).
    pub task: Option<Task>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tasks: Vec<Task>,
    pub poll_after_ms: i64,
}

impl PollResponse {
    pub fn task_list(&self) -> Vec<Task> {
        if !self.tasks.is_empty() {
            return self.tasks.clone();
        }
        match &self.task {
            Some(t) => vec![t.clone()],
            None => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskLeaseRef {
    pub task_id: String,
    pub lease_epoch: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub node_id: String,
    #[serde(default)]
    pub tasks: Vec<TaskLeaseRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub extended: usize,
}

/// Job is a user-submitted workload composed of tasks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub kind: String,
    pub status: String,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub replicate: bool,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub priority: i32,

    pub width: i32,
    pub height: i32,
    pub max_iter: i32,
    pub cols: i32,
    pub rows: i32,
    pub tile_w: i32,
    pub tile_h: i32,

    pub tasks_total: usize,
    pub tasks_done: usize,
    #[serde(default, skip_serializing_if = "is_empty_str")]
    pub error: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub reclaimed: bool,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub payload_bytes: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub eta_seconds: i64,
    #[serde(default, skip_serializing_if = "is_empty_str")]
    pub waiting: String,
    #[serde(default, skip_serializing_if = "is_empty_map")]
    pub params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "is_empty_smap")]
    pub meta: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none", with = "b64")]
    pub input: Option<Vec<u8>>,
    /// Only lease this job's tasks to nodes advertising this capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_capability: Option<String>,
}

/// SubmitJobRequest accepts numeric params plus string options (mode/model/
/// format) inside `params`, `meta`, or top-level `mode` — like compost.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SubmitJobRequest {
    pub kind: String,
    #[serde(skip_serializing_if = "is_empty_map")]
    pub params: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "is_empty_smap")]
    pub meta: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "is_empty_str")]
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none", with = "b64")]
    pub input: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "is_false")]
    pub replicate: bool,
    #[serde(skip_serializing_if = "is_zero_i32")]
    pub priority: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_capability: Option<String>,
}

impl<'de> Deserialize<'de> for SubmitJobRequest {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            kind: String,
            #[serde(default)]
            params: BTreeMap<String, serde_json::Value>,
            #[serde(default)]
            meta: BTreeMap<String, String>,
            #[serde(default)]
            mode: String,
            #[serde(default, with = "b64")]
            input: Option<Vec<u8>>,
            #[serde(default)]
            replicate: bool,
            #[serde(default)]
            priority: i32,
            #[serde(default)]
            required_capability: Option<String>,
        }
        let raw = Raw::deserialize(d)?;
        let mut out = SubmitJobRequest {
            kind: raw.kind,
            params: BTreeMap::new(),
            meta: raw.meta,
            mode: raw.mode,
            input: raw.input,
            replicate: raw.replicate,
            priority: raw.priority,
            required_capability: raw.required_capability,
        };
        for (k, v) in raw.params {
            match v {
                serde_json::Value::Number(n) => {
                    let f = n.as_f64().ok_or_else(|| {
                        serde::de::Error::custom(format!("param {k:?} not finite"))
                    })?;
                    out.params.insert(k, f);
                }
                serde_json::Value::String(s) => {
                    if k == "mode" && out.mode.is_empty() {
                        out.mode = s.clone();
                    }
                    out.meta.insert(k, s);
                }
                _ => {
                    return Err(serde::de::Error::custom(format!(
                        "param {k:?} must be a number or string"
                    )));
                }
            }
        }
        if !out.mode.is_empty() {
            out.meta
                .entry("mode".to_string())
                .or_insert_with(|| out.mode.clone());
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitJobResponse {
    pub job: Job,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadInfo {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateResponse {
    pub nodes: Vec<Node>,
    pub jobs: Vec<Job>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub workloads: Vec<WorkloadInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_wire_roundtrip() {
        let t = Task {
            id: "job-1-r0c0".into(),
            job_id: "job-1".into(),
            kind: "mandelbrot".into(),
            lease_epoch: 3,
            width: 64,
            height: 48,
            x0: -2.5,
            y0: -1.0,
            x1: 1.0,
            y1: 1.0,
            max_iter: 800,
            est_millis: 120,
            input: Some(vec![1, 2, 3]),
            ..Default::default()
        };
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("\"lease_epoch\":3"));
        assert!(json.contains("\"input\":\"AQID\""));
        let back: Task = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn submit_params_string_routing() {
        let req: SubmitJobRequest = serde_json::from_str(
            r#"{"kind":"llm_batch","params":{"batch_size":4,"mode":"verify","model":"qwen2.5"},"priority":2}"#,
        )
        .unwrap();
        assert_eq!(req.params["batch_size"], 4.0);
        assert_eq!(req.mode, "verify");
        assert_eq!(req.meta["model"], "qwen2.5");
        assert_eq!(req.meta["mode"], "verify");
        let bad: Result<SubmitJobRequest, _> =
            serde_json::from_str(r#"{"kind":"x","params":{"a":true}}"#);
        assert!(bad.is_err());
    }

    #[test]
    fn caps_backward_compat_and_capability_roundtrip() {
        // compost v0.7.0 agents send no capabilities/loaded_models fields.
        let legacy: Caps = serde_json::from_str(
            r#"{"cores":4,"mem_mb":8192,"score":1.0,"battery_pct":-1,"on_battery":false,"temp_c":-1,"arch":"x86_64","os":"linux","version":"0.7.0"}"#,
        )
        .unwrap();
        assert!(legacy.capabilities.is_empty());
        assert!(legacy.loaded_models.is_empty());
        assert!(
            !serde_json::to_string(&legacy)
                .unwrap()
                .contains("capabilities")
        );
        let mut c = legacy.clone();
        c.capabilities = vec!["llm".into(), "vision".into()];
        c.loaded_models = vec!["qwen2.5-7b".into()];
        let back: Caps = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn submit_required_capability_roundtrip() {
        let req: SubmitJobRequest =
            serde_json::from_str(r#"{"kind":"llm_batch","required_capability":"llm"}"#).unwrap();
        assert_eq!(req.required_capability.as_deref(), Some("llm"));
        assert!(
            serde_json::to_string(&req)
                .unwrap()
                .contains("\"required_capability\":\"llm\"")
        );
        let none: SubmitJobRequest = serde_json::from_str(r#"{"kind":"primes"}"#).unwrap();
        assert!(none.required_capability.is_none());
        assert!(
            !serde_json::to_string(&none)
                .unwrap()
                .contains("required_capability")
        );
    }

    #[test]
    fn poll_response_legacy_task() {
        let pr: PollResponse = serde_json::from_str(
            r#"{"task":{"id":"t1","job_id":"j","kind":"primes"},"poll_after_ms":2000}"#,
        )
        .unwrap();
        assert_eq!(pr.task_list().len(), 1);
        let empty: PollResponse =
            serde_json::from_str(r#"{"task":null,"poll_after_ms":500}"#).unwrap();
        assert!(empty.task_list().is_empty());
    }

    #[test]
    fn result_payload_alias() {
        let r: ResultMsg =
            serde_json::from_str(r#"{"task_id":"t","pixels":"AQID","millis":5}"#).unwrap();
        assert_eq!(r.payload(), vec![1, 2, 3]);
        let r2: ResultMsg =
            serde_json::from_str(r#"{"task_id":"t","data":"BAUG","pixels":"AQID","millis":5}"#)
                .unwrap();
        assert_eq!(r2.payload(), vec![4, 5, 6]);
    }
}
