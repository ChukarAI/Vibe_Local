//! Batch text embedding distributed across `embed_rag` nodes running a
//! local OpenAI-compatible server (Vibe Local on :20129).
//!
//! Jobs are capability-filtered by construction: `validate` stamps
//! `required_capability = "embed_rag"`, so the coordinator only leases
//! tasks to nodes that auto-detected an embedding model (see `node.rs`).
//! `mode=verify` runs the deterministic builtin embedding
//! (`vibe_local_engine::microeval::deterministic_embedding`, honestly
//! labeled `builtin-microeval`) so the full pipeline — submission,
//! capability-filtered leasing, execution and artifact assembly — can be
//! tested end-to-end without a live embedding runtime.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::node;
use crate::proto::{JOB_QUEUED, Job, TASK_EMBED_BATCH, Task};
use crate::workload::{AggregateOut, ENC_JSONL, Preset, get_param};

pub struct EmbedBatch;

const DEFAULT_BATCH_SIZE: i32 = 8;
const MAX_BATCH_SIZE: i32 = 64;
pub const MAX_INPUT: usize = 8 << 20;
/// Dimension of verify-mode vectors (matches the builtin fallback).
pub const EMBED_DIM: usize = vibe_local_engine::microeval::DET_EMBED_DIM;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbedItem {
    #[serde(default)]
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbedTaskEnvelope {
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub model: String,
    pub items: Vec<EmbedItem>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbedResultItem {
    pub index: usize,
    pub id: String,
    pub text: String,
    pub model: String,
    /// The embedding vector, rounded to 6 decimals so artifacts are
    /// byte-stable across replicas and platforms.
    pub embedding: Vec<f32>,
    pub dimensions: usize,
    pub latency_ms: i64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub mode: String,
}

/// Accept JSONL objects (`{"text": ...}` or `{"prompt": ...}`),
/// `---`-separated Markdown blocks, or plain line-delimited texts — the
/// same input conventions as `llm_batch`.
pub fn parse_input(raw: &[u8]) -> anyhow::Result<Vec<EmbedItem>> {
    let text = String::from_utf8_lossy(raw).replace("\r\n", "\n");
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let lines: Vec<&str> = text.split('\n').collect();

    if lines.iter().any(|l| l.trim() == "---") {
        let mut items = Vec::new();
        let mut cur: Vec<&str> = Vec::new();
        let flush = |cur: &mut Vec<&str>, items: &mut Vec<EmbedItem>| {
            let block = cur.join("\n").trim().to_string();
            cur.clear();
            if block.is_empty() {
                return;
            }
            items.push(EmbedItem {
                id: format!("e{}", items.len() + 1),
                text: block,
            });
        };
        for line in &lines {
            if line.trim() == "---" {
                flush(&mut cur, &mut items);
                continue;
            }
            cur.push(line);
        }
        flush(&mut cur, &mut items);
        return Ok(items);
    }

    let non_empty: Vec<&str> = lines
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if !non_empty.is_empty() && non_empty.iter().all(|l| l.starts_with('{')) {
        #[derive(Deserialize)]
        struct RawItem {
            #[serde(default)]
            id: String,
            #[serde(default)]
            text: String,
            #[serde(default)]
            prompt: String,
        }
        let mut items = Vec::new();
        for (idx, line) in non_empty.iter().enumerate() {
            let raw: RawItem = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("embed_batch: bad JSONL on line {}: {e}", idx + 1))?;
            let text = if !raw.text.trim().is_empty() {
                raw.text
            } else {
                raw.prompt
            };
            anyhow::ensure!(
                !text.trim().is_empty(),
                "embed_batch: JSONL line {} needs 'text' or 'prompt'",
                idx + 1
            );
            items.push(EmbedItem {
                id: if raw.id.is_empty() {
                    format!("e{}", idx + 1)
                } else {
                    raw.id
                },
                text,
            });
        }
        return Ok(items);
    }

    Ok(non_empty
        .iter()
        .enumerate()
        .map(|(i, l)| EmbedItem {
            id: format!("e{}", i + 1),
            text: (*l).to_string(),
        })
        .collect())
}

fn round_vec(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| (x * 1e6).round() / 1e6).collect()
}

fn execute_item(
    idx: usize,
    item: &EmbedItem,
    env: &EmbedTaskEnvelope,
    verify: bool,
) -> anyhow::Result<EmbedResultItem> {
    if verify || env.mode == "verify" {
        let model = if env.model.trim().is_empty() {
            "builtin-microeval".to_string()
        } else {
            env.model.trim().to_string()
        };
        let vec = vibe_local_engine::microeval::deterministic_embedding(&item.text, EMBED_DIM);
        return Ok(EmbedResultItem {
            index: idx,
            id: item.id.clone(),
            text: item.text.clone(),
            model,
            dimensions: vec.len(),
            embedding: round_vec(&vec),
            latency_ms: 1,
            mode: "verify".into(),
        });
    }

    anyhow::bail!("embed_batch: execute_item is verify-mode only; live mode uses execute_live")
}

/// Live mode: embed the whole chunk in ONE batched `/v1/embeddings`
/// request against the node's local server.
fn execute_live(base: usize, env: &EmbedTaskEnvelope) -> anyhow::Result<Vec<EmbedResultItem>> {
    let body = serde_json::json!({
        "model": env.model,
        "input": env.items.iter().map(|i| i.text.clone()).collect::<Vec<_>>(),
    });
    let endpoint = format!("{}/v1/embeddings", node::llm_endpoint());
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(600))
        .build();
    let start = Instant::now();
    let resp = agent
        .post(&endpoint)
        .send_json(body)
        .map_err(|e| anyhow::anyhow!("embed_batch: request to {endpoint} failed: {e}"))?;
    let latency_ms = (start.elapsed().as_millis() as i64).max(1);
    let status = resp.status();
    let parsed: serde_json::Value = resp.into_json()?;
    anyhow::ensure!(
        status == 200,
        "embed_batch: {endpoint} returned HTTP {status}: {parsed}"
    );
    let data = parsed["data"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("embed_batch: response has no data array"))?;
    anyhow::ensure!(
        data.len() == env.items.len(),
        "embed_batch: got {} vectors for {} texts",
        data.len(),
        env.items.len()
    );
    let mut by_index: Vec<Option<Vec<f32>>> = vec![None; env.items.len()];
    for entry in data {
        let i = entry["index"].as_u64().unwrap_or(0) as usize;
        anyhow::ensure!(i < env.items.len(), "embed_batch: index {i} out of range");
        let vec: Vec<f32> = entry["embedding"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("embed_batch: missing embedding array"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        by_index[i] = Some(vec);
    }
    let model = parsed["model"].as_str().unwrap_or(&env.model).to_string();
    let mut out = Vec::with_capacity(env.items.len());
    for (pos, item) in env.items.iter().enumerate() {
        let vec = by_index[pos]
            .clone()
            .ok_or_else(|| anyhow::anyhow!("embed_batch: missing vector for item {}", item.id))?;
        out.push(EmbedResultItem {
            index: base + pos,
            id: item.id.clone(),
            text: item.text.clone(),
            model: model.clone(),
            dimensions: vec.len(),
            embedding: round_vec(&vec),
            latency_ms,
            mode: "live".into(),
        });
    }
    Ok(out)
}

impl Preset for EmbedBatch {
    fn name(&self) -> &'static str {
        TASK_EMBED_BATCH
    }
    fn describe(&self) -> &'static str {
        "batch text embedding via embed_rag nodes (Vibe Local :20129) → JSONL"
    }

    fn validate(&self, params: &BTreeMap<String, f64>) -> anyhow::Result<Job> {
        let batch_size =
            (get_param(params, "batch_size", DEFAULT_BATCH_SIZE as f64).round() as i32).max(1);
        anyhow::ensure!(
            batch_size <= MAX_BATCH_SIZE,
            "embed_batch: batch_size must be 1..{MAX_BATCH_SIZE}, got {batch_size}"
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
            // Embedding work only makes sense on a node that detected an
            // embedding-capable model; the coordinator enforces this at
            // lease time unless the submitter overrides it explicitly.
            required_capability: Some("embed_rag".into()),
            ..Default::default()
        })
    }

    fn split(&self, job: &Job) -> anyhow::Result<Vec<Task>> {
        let input = job
            .input
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("embed_batch: job has no input"))?;
        anyhow::ensure!(
            input.len() <= MAX_INPUT,
            "embed_batch: input exceeds {MAX_INPUT} bytes"
        );
        let items = parse_input(input)?;
        anyhow::ensure!(!items.is_empty(), "embed_batch: input contained zero texts");

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
            let env = EmbedTaskEnvelope {
                mode: mode.clone(),
                model: model.clone(),
                items: chunk.to_vec(),
            };
            let raw = serde_json::to_vec(&env)?;
            let chars: usize = chunk.iter().map(|it| it.text.len()).sum();
            let est = if verify_mode {
                (chunk.len() * 5) as i64
            } else {
                (1000 + chars as i64 * 2).max(1500)
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
            .ok_or_else(|| anyhow::anyhow!("embed_batch: task has no input chunk"))?;
        let env: EmbedTaskEnvelope = serde_json::from_slice(input)
            .map_err(|e| anyhow::anyhow!("embed_batch: bad task envelope: {e}"))?;
        anyhow::ensure!(!env.items.is_empty(), "embed_batch: empty task item list");
        let mut out = Vec::new();
        let base = t.start.max(0) as usize;
        let results = if t.verify_mode || env.mode == "verify" {
            env.items
                .iter()
                .enumerate()
                .map(|(i, item)| execute_item(base + i, item, &env, true))
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            execute_live(base, &env)?
        };
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
        let mut all: Vec<EmbedResultItem> = Vec::new();
        for task_id in order {
            let buf = results
                .get(task_id)
                .ok_or_else(|| anyhow::anyhow!("embed_batch: missing chunk {task_id}"))?;
            let text = String::from_utf8_lossy(buf);
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let item: EmbedResultItem = serde_json::from_str(line)
                    .map_err(|e| anyhow::anyhow!("embed_batch: bad chunk {task_id} JSONL: {e}"))?;
                all.push(item);
            }
        }
        all.sort_by_key(|i| i.index);
        anyhow::ensure!(
            all.iter().enumerate().all(|(i, item)| item.index == i),
            "embed_batch: artifact indices are not contiguous from 0"
        );
        if let Some(first) = all.first() {
            anyhow::ensure!(
                all.iter().all(|i| i.dimensions == first.dimensions),
                "embed_batch: inconsistent embedding dimensions across chunks"
            );
        }
        let _ = job;
        let mut out = Vec::new();
        for item in &all {
            out.extend_from_slice(&serde_json::to_vec(item)?);
            out.push(b'\n');
        }
        Ok((out, "text/plain; charset=utf-8".into(), ".jsonl".into()))
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

    #[test]
    fn parse_three_formats() {
        let jsonl = b"{\"text\":\"hello world\",\"id\":\"a\"}\n{\"prompt\":\"second text\"}\n";
        let items = parse_input(jsonl).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, "a");
        assert_eq!(items[1].id, "e2");
        assert_eq!(items[1].text, "second text");

        let md = b"first block\n---\nsecond block\n---\n";
        let items = parse_input(md).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[1].text, "second block");

        let lines = b"one\ntwo\n\nthree\n";
        let items = parse_input(lines).unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[2].text, "three");

        assert!(parse_input(b"   ").unwrap().is_empty());
        assert!(parse_input(b"{\"bad\":true}\n").is_err());
    }

    #[test]
    fn validate_defaults_to_embed_rag_capability() {
        let j = EmbedBatch.validate(&BTreeMap::new()).unwrap();
        assert_eq!(j.width, 8);
        assert_eq!(j.required_capability.as_deref(), Some("embed_rag"));
        assert!(
            EmbedBatch
                .validate(&BTreeMap::from([("batch_size".into(), 128.0)]))
                .is_err()
        );
    }

    #[test]
    fn verify_mode_pipeline() {
        let mut job = EmbedBatch
            .validate(&BTreeMap::from([
                ("batch_size".into(), 2.0),
                ("verify".into(), 1.0),
            ]))
            .unwrap();
        job.id = "job-e".into();
        job.input = Some(b"alpha text\nbeta text\ngamma text\n".to_vec());
        let tasks = EmbedBatch.split(&job).unwrap();
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().all(|t| t.verify_mode));
        let mut results = BTreeMap::new();
        let order: Vec<String> = tasks.iter().map(|t| t.id.clone()).collect();
        for t in &tasks {
            let (data, enc) = EmbedBatch.execute(t).unwrap();
            assert_eq!(enc, ENC_JSONL);
            results.insert(t.id.clone(), data);
        }
        let (artifact, ct, ext) = EmbedBatch.aggregate(&job, &order, &results).unwrap();
        assert_eq!(ext, ".jsonl");
        assert!(ct.starts_with("text/plain"));
        let text = String::from_utf8_lossy(&artifact);
        assert_eq!(text.lines().count(), 3);
        let first: EmbedResultItem = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(first.dimensions, EMBED_DIM);
        assert_eq!(first.embedding.len(), EMBED_DIM);
        assert_eq!(first.mode, "verify");
        let norm: f32 = first.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "norm {norm}");
        // Deterministic: identical re-execution, byte for byte.
        let (a, _) = EmbedBatch.execute(&tasks[0]).unwrap();
        let (b, _) = EmbedBatch.execute(&tasks[0]).unwrap();
        assert_eq!(a, b);
        // Different texts embed differently.
        let third: EmbedResultItem = serde_json::from_str(text.lines().nth(2).unwrap()).unwrap();
        assert_ne!(first.embedding, third.embedding);
    }
}
