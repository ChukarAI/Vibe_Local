//! Batch image understanding distributed across `vision` nodes running
//! a local multimodal OpenAI-compatible server (Vibe Local on :20129
//! with a model loaded via `mmproj_path`).
//!
//! Jobs are capability-filtered by construction: `validate` stamps
//! `required_capability = "vision"`, so the coordinator only leases
//! tasks to nodes that detected a vision-capable model (see `node.rs`).
//!
//! Input is JSONL — one object per image:
//! `{"prompt": "...", "image_base64": "...", "mime_type"?: "image/png"}`
//! or `{"prompt": "...", "image_url": "data:<mime>;base64,..."}`.
//! Remote image URLs are rejected outright: workers must never fetch
//! arbitrary URLs on a submitter's behalf (SSRF), so only inline
//! `data:` payloads are accepted, and every payload is base64-decoded
//! once at parse time to prove it is well-formed.
//!
//! `mode=verify` computes a deterministic summary hash over
//! `{index, prompt, image_sha256, bytes}` labeled `"engine": "verify"`,
//! so submission, capability-filtered leasing, execution and artifact
//! assembly can be tested end-to-end without a live vision runtime.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::node;
use crate::proto::{JOB_QUEUED, Job, TASK_VISION_BATCH, Task};
use crate::workload::{AggregateOut, ENC_JSONL, Preset, get_param};

pub struct VisionBatch;

const DEFAULT_BATCH_SIZE: i32 = 4;
const MAX_BATCH_SIZE: i32 = 32;
pub const MAX_INPUT: usize = 8 << 20;
/// Decoded per-image cap; job input itself is capped by `MAX_INPUT`.
pub const MAX_IMAGE_BYTES: usize = 4 << 20;
const DEFAULT_MIME: &str = "image/png";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionItem {
    pub prompt: String,
    /// Raw base64 payload (no `data:` prefix) — the wire form workers
    /// use to rebuild the multimodal request.
    pub image_base64: String,
    #[serde(default = "default_mime")]
    pub mime_type: String,
}

fn default_mime() -> String {
    DEFAULT_MIME.to_string()
}

impl VisionItem {
    /// Decode the image payload (validated at parse time).
    pub fn image_bytes(&self) -> anyhow::Result<Vec<u8>> {
        decode_base64(&self.image_base64)
            .map_err(|e| anyhow::anyhow!("vision_batch: stored image is not valid base64: {e}"))
    }

    /// SHA-256 (hex) of the decoded image bytes.
    pub fn image_sha256(&self) -> anyhow::Result<String> {
        Ok(vibe_local_core::util::hex_encode(&Sha256::digest(
            self.image_bytes()?,
        )))
    }

    /// The `data:` URL a multimodal chat request carries.
    pub fn data_url(&self) -> String {
        format!("data:{};base64,{}", self.mime_type, self.image_base64)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VisionTaskEnvelope {
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub model: String,
    pub items: Vec<VisionItem>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VisionResultItem {
    pub index: usize,
    pub prompt: String,
    /// SHA-256 (hex) of the decoded image bytes — the artifact row's
    /// image identity, stable across replicas.
    pub image_sha256: String,
    /// Model response text (live) or the deterministic verify summary.
    pub response: String,
    /// Engine that produced the response: `verify` in verify mode,
    /// otherwise the engine reported by the chat endpoint
    /// (`llama-server` when it does not label itself).
    #[serde(default)]
    pub engine: String,
    /// Backend reported by the chat endpoint (live mode only).
    #[serde(default)]
    pub backend: String,
    pub latency_ms: i64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub mode: String,
}

fn decode_base64(payload: &str) -> Result<Vec<u8>, base64::DecodeError> {
    let cleaned: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD.decode(cleaned.as_bytes())
}

/// Split a `data:<mime>;base64,<payload>` URL into (mime, payload).
/// Anything else — remote `http(s)://` URLs included — is an error.
fn parse_data_url(url: &str) -> anyhow::Result<(String, String)> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| anyhow::anyhow!("only inline data: image URLs are accepted"))?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| anyhow::anyhow!("malformed data: URL (missing ',')"))?;
    anyhow::ensure!(
        meta.ends_with(";base64"),
        "data: URL must be base64-encoded (;base64)"
    );
    let mime = meta.trim_end_matches(";base64");
    anyhow::ensure!(
        mime.starts_with("image/"),
        "data: URL mime must be an image type, got {mime:?}"
    );
    Ok((mime.to_string(), payload.to_string()))
}

/// Parse JSONL vision input. Every non-empty line must be one JSON
/// object with a non-empty `prompt` and exactly one image source
/// (`image_base64` or a `data:` `image_url`). Remote URLs and
/// malformed base64 are rejected here, before any task is created.
pub fn parse_input(raw: &[u8]) -> anyhow::Result<Vec<VisionItem>> {
    let text = String::from_utf8_lossy(raw).replace("\r\n", "\n");
    #[derive(Deserialize)]
    struct RawItem {
        #[serde(default)]
        prompt: String,
        #[serde(default)]
        image_base64: String,
        #[serde(default)]
        image_url: String,
        #[serde(default)]
        mime_type: String,
    }
    let mut items = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let raw: RawItem = serde_json::from_str(line)
            .map_err(|e| anyhow::anyhow!("vision_batch: bad JSONL on line {}: {e}", idx + 1))?;
        anyhow::ensure!(
            !raw.prompt.trim().is_empty(),
            "vision_batch: JSONL line {} needs a non-empty 'prompt'",
            idx + 1
        );
        let (mime, b64) = if !raw.image_base64.trim().is_empty() {
            let mime = if raw.mime_type.trim().is_empty() {
                DEFAULT_MIME.to_string()
            } else {
                raw.mime_type.trim().to_string()
            };
            (mime, raw.image_base64.trim().to_string())
        } else if !raw.image_url.trim().is_empty() {
            let url = raw.image_url.trim();
            if !(url.starts_with("data:")
                || url.starts_with("http://")
                || url.starts_with("https://"))
                && !url.contains(";base64,")
            {
                // A bare payload in image_url is not a URL at all.
                anyhow::bail!(
                    "vision_batch: line {} image_url must be a data: URL",
                    idx + 1
                );
            }
            if url.starts_with("http://") || url.starts_with("https://") {
                anyhow::bail!(
                    "vision_batch: line {} uses a remote image URL — rejected (workers never fetch remote URLs; inline the image as data: base64)",
                    idx + 1
                );
            }
            parse_data_url(url)
                .map_err(|e| anyhow::anyhow!("vision_batch: line {}: {e}", idx + 1))?
        } else {
            anyhow::bail!(
                "vision_batch: JSONL line {} needs 'image_base64' or a data: 'image_url'",
                idx + 1
            );
        };
        let bytes = decode_base64(&b64).map_err(|e| {
            anyhow::anyhow!(
                "vision_batch: line {} image is not valid base64: {e}",
                idx + 1
            )
        })?;
        anyhow::ensure!(
            !bytes.is_empty(),
            "vision_batch: line {} image decodes to zero bytes",
            idx + 1
        );
        anyhow::ensure!(
            bytes.len() <= MAX_IMAGE_BYTES,
            "vision_batch: line {} image exceeds {MAX_IMAGE_BYTES} bytes decoded",
            idx + 1
        );
        items.push(VisionItem {
            prompt: raw.prompt,
            image_base64: b64,
            mime_type: mime,
        });
    }
    Ok(items)
}

fn execute_item_verify(idx: usize, item: &VisionItem) -> anyhow::Result<VisionResultItem> {
    let bytes = item.image_bytes()?;
    let image_sha256 = item.image_sha256()?;
    // Deterministic summary over {index, prompt, image_sha256, bytes}.
    let mut h = Sha256::new();
    h.update(idx.to_string().as_bytes());
    h.update(b"|");
    h.update(item.prompt.as_bytes());
    h.update(b"|");
    h.update(image_sha256.as_bytes());
    h.update(b"|");
    h.update(bytes.len().to_string().as_bytes());
    let summary = vibe_local_core::util::hex_encode(&h.finalize());
    Ok(VisionResultItem {
        index: idx,
        prompt: item.prompt.clone(),
        image_sha256,
        response: format!(
            "verify summary {summary} ({} bytes, {})",
            bytes.len(),
            item.mime_type
        ),
        engine: "verify".into(),
        backend: String::new(),
        latency_ms: 1,
        mode: "verify".into(),
    })
}

/// Live mode: one multimodal `/v1/chat/completions` request per item
/// against the node's local server, with the OpenAI content-part array.
fn execute_live(base: usize, env: &VisionTaskEnvelope) -> anyhow::Result<Vec<VisionResultItem>> {
    let endpoint = format!("{}/v1/chat/completions", node::llm_endpoint());
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(600))
        .build();
    let mut out = Vec::with_capacity(env.items.len());
    for (pos, item) in env.items.iter().enumerate() {
        let body = serde_json::json!({
            "model": env.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": item.prompt},
                    {"type": "image_url", "image_url": {"url": item.data_url()}},
                ],
            }],
            "max_tokens": 512,
        });
        let start = Instant::now();
        let resp = agent
            .post(&endpoint)
            .send_json(body)
            .map_err(|e| anyhow::anyhow!("vision_batch: request to {endpoint} failed: {e}"))?;
        let latency_ms = (start.elapsed().as_millis() as i64).max(1);
        let status = resp.status();
        let engine = resp
            .header("X-Vibe-Local-Engine")
            .unwrap_or("llama-server")
            .to_string();
        let backend = resp
            .header("X-Vibe-Local-Backend")
            .unwrap_or("")
            .to_string();
        let parsed: serde_json::Value = resp.into_json()?;
        anyhow::ensure!(
            status == 200,
            "vision_batch: {endpoint} returned HTTP {status}: {parsed}"
        );
        let response = parsed["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        out.push(VisionResultItem {
            index: base + pos,
            prompt: item.prompt.clone(),
            image_sha256: item.image_sha256()?,
            response,
            engine: parsed["engine"].as_str().unwrap_or(&engine).to_string(),
            backend,
            latency_ms,
            mode: "live".into(),
        });
    }
    Ok(out)
}

impl Preset for VisionBatch {
    fn name(&self) -> &'static str {
        TASK_VISION_BATCH
    }
    fn describe(&self) -> &'static str {
        "batch image understanding via vision nodes (Vibe Local :20129 multimodal) → JSON array"
    }

    fn validate(&self, params: &BTreeMap<String, f64>) -> anyhow::Result<Job> {
        let batch_size =
            (get_param(params, "batch_size", DEFAULT_BATCH_SIZE as f64).round() as i32).max(1);
        anyhow::ensure!(
            batch_size <= MAX_BATCH_SIZE,
            "vision_batch: batch_size must be 1..{MAX_BATCH_SIZE}, got {batch_size}"
        );
        let verify = get_param(params, "verify", 0.0);
        let mut out_params = BTreeMap::from([("batch_size".into(), f64::from(batch_size))]);
        if verify > 0.0 {
            out_params.insert("verify".into(), 1.0);
        }
        Ok(Job {
            id: String::new(),
            kind: self.name().into(),
            status: JOB_QUEUED.into(),
            width: batch_size,
            params: out_params,
            // Vision work only makes sense on a node that detected a
            // multimodal model; the coordinator enforces this at lease
            // time unless the submitter overrides it explicitly.
            required_capability: Some("vision".into()),
            ..Default::default()
        })
    }

    fn split(&self, job: &Job) -> anyhow::Result<Vec<Task>> {
        let input = job
            .input
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("vision_batch: job has no input"))?;
        anyhow::ensure!(
            input.len() <= MAX_INPUT,
            "vision_batch: input exceeds {MAX_INPUT} bytes"
        );
        let items = parse_input(input)?;
        anyhow::ensure!(
            !items.is_empty(),
            "vision_batch: input contained zero items"
        );

        let batch_size = job.width.max(1) as usize;
        let mut mode = job
            .meta
            .get("mode")
            .map(|m| m.trim().to_lowercase())
            .unwrap_or_default();
        let model = job
            .meta
            .get("model")
            .map(|m| m.trim().to_string())
            .unwrap_or_default();
        let verify_mode =
            mode == "verify" || job.params.get("verify").copied().unwrap_or(0.0) > 0.0;
        if verify_mode {
            mode = "verify".into();
        }

        let mut tasks = Vec::new();
        let mut off = 0usize;
        while off < items.len() {
            let end = (off + batch_size).min(items.len());
            let chunk = &items[off..end];
            let env = VisionTaskEnvelope {
                mode: mode.clone(),
                model: model.clone(),
                items: chunk.to_vec(),
            };
            let raw = serde_json::to_vec(&env)?;
            let est = if verify_mode {
                (chunk.len() * 5) as i64
            } else {
                (2000 + chunk.len() as i64 * 1500).max(2500)
            };
            tasks.push(Task {
                id: format!("{}-b{}", job.id, tasks.len()),
                job_id: job.id.clone(),
                kind: self.name().into(),
                input: Some(raw),
                start: off as i64,
                count: chunk.len() as i64,
                est_millis: est,
                verify_mode,
                ..Default::default()
            });
            off = end;
        }
        Ok(tasks)
    }

    fn execute(&self, t: &Task) -> anyhow::Result<(Vec<u8>, &'static str)> {
        let input = t
            .input
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("vision_batch: task has no input chunk"))?;
        let env: VisionTaskEnvelope = serde_json::from_slice(input)
            .map_err(|e| anyhow::anyhow!("vision_batch: bad task envelope: {e}"))?;
        anyhow::ensure!(!env.items.is_empty(), "vision_batch: empty task item list");
        let base = t.start.max(0) as usize;
        let results = if t.verify_mode || env.mode == "verify" {
            env.items
                .iter()
                .enumerate()
                .map(|(i, item)| execute_item_verify(base + i, item))
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            execute_live(base, &env)?
        };
        let mut out = Vec::new();
        for res in &results {
            out.extend_from_slice(&serde_json::to_vec(res)?);
            out.push(b'\n');
        }
        Ok((out, ENC_JSONL))
    }

    fn aggregate(
        &self,
        job: &Job,
        order: &[String],
        results: &BTreeMap<String, Vec<u8>>,
    ) -> anyhow::Result<AggregateOut> {
        let mut all: Vec<VisionResultItem> = Vec::new();
        for task_id in order {
            let buf = results
                .get(task_id)
                .ok_or_else(|| anyhow::anyhow!("vision_batch: missing chunk {task_id}"))?;
            let text = String::from_utf8_lossy(buf);
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let item: VisionResultItem = serde_json::from_str(line)
                    .map_err(|e| anyhow::anyhow!("vision_batch: bad chunk {task_id} JSONL: {e}"))?;
                all.push(item);
            }
        }
        all.sort_by_key(|i| i.index);
        anyhow::ensure!(
            all.iter().enumerate().all(|(i, item)| item.index == i),
            "vision_batch: artifact indices are not contiguous from 0"
        );
        let _ = job;
        // The artifact is a single JSON array of the contract rows:
        // {index, prompt, image_sha256, response, engine} in input order.
        let rows: Vec<serde_json::Value> = all
            .iter()
            .map(|i| {
                serde_json::json!({
                    "index": i.index,
                    "prompt": i.prompt,
                    "image_sha256": i.image_sha256,
                    "response": i.response,
                    "engine": i.engine,
                })
            })
            .collect();
        let out = serde_json::to_vec(&rows)?;
        Ok((out, "application/json".into(), ".json".into()))
    }

    fn result_size(&self, _t: &Task) -> usize {
        0
    }

    fn verify_replica(&self, primary: &[u8], replica: &[u8]) -> bool {
        !primary.is_empty() && primary == replica
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1x1 transparent PNG.
    const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

    fn png_bytes() -> Vec<u8> {
        decode_base64(PNG_B64).unwrap()
    }

    #[test]
    fn parse_accepts_base64_and_data_url_forms() {
        let jsonl = format!(
            "{{\"prompt\":\"describe it\",\"image_base64\":\"{PNG_B64}\"}}\n{{\"prompt\":\"what color?\",\"image_url\":\"data:image/jpeg;base64,{PNG_B64}\"}}\n"
        );
        let items = parse_input(jsonl.as_bytes()).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].mime_type, "image/png");
        assert_eq!(items[0].prompt, "describe it");
        assert_eq!(items[1].mime_type, "image/jpeg");
        assert_eq!(items[0].image_bytes().unwrap(), png_bytes());
        assert_eq!(
            items[0].image_sha256().unwrap(),
            vibe_local_core::util::hex_encode(&Sha256::digest(png_bytes()))
        );
        assert!(items[0].data_url().starts_with("data:image/png;base64,"));
        assert!(parse_input(b"  \n").unwrap().is_empty());
    }

    #[test]
    fn parse_rejects_remote_urls_and_bad_payloads() {
        // Remote URLs are an SSRF vector from workers — always rejected.
        let remote = br#"{"prompt":"x","image_url":"https://example.com/a.png"}"#;
        let err = parse_input(remote).unwrap_err().to_string();
        assert!(err.contains("remote image URL"), "{err}");
        let remote_http = br#"{"prompt":"x","image_url":"http://10.0.0.5/a.png"}"#;
        assert!(parse_input(remote_http).is_err());
        // Malformed base64.
        let bad = br#"{"prompt":"x","image_base64":"!!!not-base64!!!"}"#;
        assert!(parse_input(bad).is_err());
        // Non-base64 data URL and non-image data URL.
        let plain = br#"{"prompt":"x","image_url":"data:image/png,rawbytes"}"#;
        assert!(parse_input(plain).is_err());
        let text = br#"{"prompt":"x","image_url":"data:text/plain;base64,aGk="}"#;
        assert!(parse_input(text).is_err());
        // Missing image / missing prompt / not JSON.
        assert!(parse_input(br#"{"prompt":"x"}"#).is_err());
        assert!(parse_input(br#"{"image_base64":"aGk="}"#).is_err());
        assert!(parse_input(b"not json\n").is_err());
    }

    #[test]
    fn validate_defaults_to_vision_capability() {
        let j = VisionBatch.validate(&BTreeMap::new()).unwrap();
        assert_eq!(j.width, 4);
        assert_eq!(j.required_capability.as_deref(), Some("vision"));
        assert!(
            VisionBatch
                .validate(&BTreeMap::from([("batch_size".into(), 64.0)]))
                .is_err()
        );
    }

    #[test]
    fn verify_mode_pipeline() {
        let mut job = VisionBatch
            .validate(&BTreeMap::from([
                ("batch_size".into(), 2.0),
                ("verify".into(), 1.0),
            ]))
            .unwrap();
        job.id = "job-v".into();
        let line = |p: &str| format!("{{\"prompt\":\"{p}\",\"image_base64\":\"{PNG_B64}\"}}");
        job.input =
            Some(format!("{}\n{}\n{}\n", line("one"), line("two"), line("three")).into_bytes());
        let tasks = VisionBatch.split(&job).unwrap();
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().all(|t| t.verify_mode));
        let mut results = BTreeMap::new();
        let order: Vec<String> = tasks.iter().map(|t| t.id.clone()).collect();
        for t in &tasks {
            let (data, enc) = VisionBatch.execute(t).unwrap();
            assert_eq!(enc, ENC_JSONL);
            results.insert(t.id.clone(), data);
        }
        let (artifact, ct, ext) = VisionBatch.aggregate(&job, &order, &results).unwrap();
        assert_eq!(ext, ".json");
        assert_eq!(ct, "application/json");
        let rows: serde_json::Value = serde_json::from_slice(&artifact).unwrap();
        let rows = rows.as_array().unwrap();
        assert_eq!(rows.len(), 3);
        let expected_sha = vibe_local_core::util::hex_encode(&Sha256::digest(png_bytes()));
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row["index"], i);
            assert_eq!(row["image_sha256"], expected_sha);
            assert_eq!(row["engine"], "verify");
            assert!(
                row["response"]
                    .as_str()
                    .unwrap()
                    .starts_with("verify summary "),
                "{row}"
            );
        }
        assert_eq!(rows[0]["prompt"], "one");
        assert_eq!(rows[2]["prompt"], "three");
        // Same image, different prompts → different verify summaries.
        assert_ne!(rows[0]["response"], rows[1]["response"]);
        // Deterministic: identical re-execution, byte for byte.
        let (a, _) = VisionBatch.execute(&tasks[0]).unwrap();
        let (b, _) = VisionBatch.execute(&tasks[0]).unwrap();
        assert_eq!(a, b);
    }
}
