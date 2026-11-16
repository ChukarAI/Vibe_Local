//! Batch LLM inference distributed across workers running a local
//! OpenAI-compatible server (Vibe Local on :20129, or COMPOST_LLM_ENDPOINT).
//! `mode=verify` runs a deterministic offline verifier so the full pipeline
//! can be tested without a live LLM runtime.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::node;
use crate::proto::{JOB_QUEUED, Job, TASK_LLM_BATCH, Task};
use crate::workload::{AggregateOut, ENC_JSONL, Preset, get_param};

pub struct LlmBatch;

const DEFAULT_BATCH_SIZE: i32 = 1;
const MAX_BATCH_SIZE: i32 = 32;
const DEFAULT_MAX_TOKENS: i32 = 512;
const DEFAULT_TEMP: f64 = 0.2;
pub const MAX_INPUT: usize = 8 << 20;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptItem {
    #[serde(default)]
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub system: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<LlmMessage>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskEnvelope {
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub model: String,
    pub max_tokens: i32,
    pub temperature: f64,
    pub items: Vec<PromptItem>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResultItem {
    pub index: usize,
    pub id: String,
    pub prompt: String,
    pub model: String,
    pub response: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub latency_ms: i64,
    pub tok_per_sec: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub mode: String,
}

/// Accept JSONL prompt objects, `---`-separated Markdown blocks, or plain
/// line-delimited prompts.
pub fn parse_input(raw: &[u8]) -> anyhow::Result<Vec<PromptItem>> {
    let text = String::from_utf8_lossy(raw).replace("\r\n", "\n");
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let lines: Vec<&str> = text.split('\n').collect();

    // 1. Markdown section separators.
    if lines.iter().any(|l| l.trim() == "---") {
        let mut items = Vec::new();
        let mut cur: Vec<&str> = Vec::new();
        let flush = |cur: &mut Vec<&str>, items: &mut Vec<PromptItem>| {
            let block = cur.join("\n").trim().to_string();
            cur.clear();
            if block.is_empty() {
                return;
            }
            items.push(PromptItem {
                id: format!("p{}", items.len() + 1),
                prompt: block,
                ..Default::default()
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

    // 2. JSONL when every non-empty line looks like a JSON object.
    let non_empty: Vec<&str> = lines
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if !non_empty.is_empty() && non_empty.iter().all(|l| l.starts_with('{')) {
        let mut items = Vec::new();
        for (idx, line) in non_empty.iter().enumerate() {
            let mut item: PromptItem = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("llm_batch: bad JSONL on line {}: {e}", idx + 1))?;
            if item.id.is_empty() {
                item.id = format!("p{}", idx + 1);
            }
            anyhow::ensure!(
                !item.prompt.trim().is_empty() || !item.messages.is_empty(),
                "llm_batch: JSONL line {} needs 'prompt' or 'messages'",
                idx + 1
            );
            items.push(item);
        }
        return Ok(items);
    }

    // 3. Line-delimited prompts.
    Ok(non_empty
        .iter()
        .enumerate()
        .map(|(i, l)| PromptItem {
            id: format!("p{}", i + 1),
            prompt: (*l).to_string(),
            ..Default::default()
        })
        .collect())
}

fn prompt_preview(item: &PromptItem) -> String {
    if !item.prompt.is_empty() {
        return item.prompt.clone();
    }
    for m in item.messages.iter().rev() {
        if m.role == "user" && !m.content.is_empty() {
            return m.content.clone();
        }
    }
    item.messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

fn build_messages(item: &PromptItem) -> Vec<LlmMessage> {
    if !item.messages.is_empty() {
        return item.messages.clone();
    }
    let mut msgs = Vec::new();
    if !item.system.trim().is_empty() {
        msgs.push(LlmMessage {
            role: "system".into(),
            content: item.system.clone(),
        });
    }
    msgs.push(LlmMessage {
        role: "user".into(),
        content: item.prompt.clone(),
    });
    msgs
}

fn execute_item(
    idx: usize,
    item: &PromptItem,
    env: &TaskEnvelope,
    verify: bool,
) -> anyhow::Result<ResultItem> {
    let prompt_text = prompt_preview(item);
    let mut model = item.model.trim().to_string();
    if model.is_empty() {
        model = env.model.trim().to_string();
    }

    if verify || env.mode == "verify" {
        if model.is_empty() {
            model = "verify-mock".into();
        }
        let words = prompt_text.split_whitespace().count();
        let sum = Sha256::digest(format!("{}\n{prompt_text}", item.id));
        let tag = vibe_local_core::util::hex_encode(&sum[..6]);
        let resp = format!("[verify:{model}] processed {words} words (sha256:{tag})");
        let p_tok = (prompt_text.len() / 4).max(1);
        let c_tok = (resp.len() / 4).max(1);
        return Ok(ResultItem {
            index: idx,
            id: item.id.clone(),
            prompt: prompt_text,
            model,
            response: resp,
            prompt_tokens: p_tok,
            completion_tokens: c_tok,
            latency_ms: 1,
            tok_per_sec: c_tok as f64 * 1000.0,
            mode: "verify".into(),
        });
    }

    if model.is_empty() {
        let (has, models) = node::probe_local_llm();
        model = if has && !models.is_empty() {
            models[0].clone()
        } else {
            "local-model".into()
        };
    }
    let body = serde_json::json!({
        "model": model,
        "messages": build_messages(item),
        "temperature": env.temperature,
        "max_tokens": env.max_tokens,
        "stream": false,
    });
    let endpoint = format!("{}/v1/chat/completions", node::llm_endpoint());
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(600))
        .build();
    let start = Instant::now();
    let resp = agent
        .post(&endpoint)
        .send_json(body)
        .map_err(|e| anyhow::anyhow!("llm_batch: request to {endpoint} failed: {e}"))?;
    let latency_ms = (start.elapsed().as_millis() as i64).max(1);
    let status = resp.status();
    let parsed: serde_json::Value = resp.into_json()?;
    anyhow::ensure!(
        status == 200,
        "llm_batch: {endpoint} returned HTTP {status}: {parsed}"
    );
    let choices = parsed["choices"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("llm_batch: empty choices in response"))?;
    anyhow::ensure!(!choices.is_empty(), "llm_batch: empty choices in response");
    let mut answer = choices[0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    if answer.is_empty() {
        answer = choices[0]["text"].as_str().unwrap_or("").to_string();
    }
    if let Some(m) = parsed["model"].as_str()
        && !m.is_empty()
    {
        model = m.to_string();
    }
    let mut p_tok = parsed["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize;
    let mut c_tok = parsed["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize;
    if p_tok == 0 {
        p_tok = (prompt_text.len() / 4).max(1);
    }
    if c_tok == 0 && !answer.is_empty() {
        c_tok = (answer.len() / 4).max(1);
    }
    let tps = c_tok as f64 / (latency_ms as f64 / 1000.0);
    Ok(ResultItem {
        index: idx,
        id: item.id.clone(),
        prompt: prompt_text,
        model,
        response: answer,
        prompt_tokens: p_tok,
        completion_tokens: c_tok,
        latency_ms,
        tok_per_sec: (tps * 100.0).round() / 100.0,
        mode: "live".into(),
    })
}

impl Preset for LlmBatch {
    fn name(&self) -> &'static str {
        TASK_LLM_BATCH
    }
    fn describe(&self) -> &'static str {
        "batch LLM inference via Vibe Local (:20129) → JSONL/Markdown"
    }

    fn validate(&self, params: &BTreeMap<String, f64>) -> anyhow::Result<Job> {
        let batch_size =
            (get_param(params, "batch_size", DEFAULT_BATCH_SIZE as f64).round() as i32).max(1);
        anyhow::ensure!(
            batch_size <= MAX_BATCH_SIZE,
            "llm_batch: batch_size must be 1..{MAX_BATCH_SIZE}, got {batch_size}"
        );
        let max_tokens =
            (get_param(params, "max_tokens", DEFAULT_MAX_TOKENS as f64).round() as i32).max(16);
        anyhow::ensure!(
            max_tokens <= 8192,
            "llm_batch: max_tokens must be ≤8192, got {max_tokens}"
        );
        let temp = get_param(params, "temperature", DEFAULT_TEMP).clamp(0.0, 2.0);
        let verify = get_param(params, "verify", 0.0);
        let mut out_params = BTreeMap::from([
            ("batch_size".into(), f64::from(batch_size)),
            ("max_tokens".into(), f64::from(max_tokens)),
            ("temperature".into(), temp),
        ]);
        if verify > 0.0 {
            out_params.insert("verify".into(), 1.0);
        }
        Ok(Job {
            id: String::new(),
            kind: self.name().into(),
            status: JOB_QUEUED.into(),
            width: batch_size,
            max_iter: max_tokens,
            params: out_params,
            ..Default::default()
        })
    }

    fn split(&self, job: &Job) -> anyhow::Result<Vec<Task>> {
        let input = job
            .input
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("llm_batch: job has no input"))?;
        anyhow::ensure!(
            input.len() <= MAX_INPUT,
            "llm_batch: input exceeds {MAX_INPUT} bytes"
        );
        let items = parse_input(input)?;
        anyhow::ensure!(!items.is_empty(), "llm_batch: input contained zero prompts");

        let batch_size = job.width.max(1) as usize;
        let max_tokens = job.max_iter.max(16);
        let temp = job
            .params
            .get("temperature")
            .copied()
            .unwrap_or(DEFAULT_TEMP);
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
            let env = TaskEnvelope {
                mode: mode.clone(),
                model: model.clone(),
                max_tokens,
                temperature: temp,
                items: chunk.to_vec(),
            };
            let raw = serde_json::to_vec(&env)?;
            let est = if verify_mode {
                (chunk.len() * 5) as i64
            } else {
                let chars: usize = chunk
                    .iter()
                    .map(|it| {
                        it.prompt.len()
                            + it.system.len()
                            + it.messages.iter().map(|m| m.content.len()).sum::<usize>()
                    })
                    .sum();
                (2000 + chars as i64 * 12).max(3000)
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
            .ok_or_else(|| anyhow::anyhow!("llm_batch: task has no input chunk"))?;
        let env: TaskEnvelope = serde_json::from_slice(input)
            .map_err(|e| anyhow::anyhow!("llm_batch: bad task envelope: {e}"))?;
        anyhow::ensure!(!env.items.is_empty(), "llm_batch: empty task item list");
        let mut out = Vec::new();
        let base = t.start.max(0) as usize;
        for (i, item) in env.items.iter().enumerate() {
            let res = execute_item(base + i, item, &env, t.verify_mode)?;
            out.extend_from_slice(&serde_json::to_vec(&res)?);
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
        let mut all: Vec<ResultItem> = Vec::new();
        for task_id in order {
            let buf = results
                .get(task_id)
                .ok_or_else(|| anyhow::anyhow!("llm_batch: missing chunk {task_id}"))?;
            let text = String::from_utf8_lossy(buf);
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let item: ResultItem = serde_json::from_str(line)
                    .map_err(|e| anyhow::anyhow!("llm_batch: bad chunk {task_id} JSONL: {e}"))?;
                all.push(item);
            }
        }
        let format = job
            .meta
            .get("format")
            .map(|f| f.trim().to_lowercase())
            .unwrap_or_default();
        if format == "markdown" || format == "md" {
            return Ok((
                markdown_report(job, &all),
                "text/markdown; charset=utf-8".into(),
                ".md".into(),
            ));
        }
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

fn markdown_report(job: &Job, items: &[ResultItem]) -> Vec<u8> {
    let total_tok: usize = items.iter().map(|i| i.completion_tokens).sum();
    let total_lat: i64 = items.iter().map(|i| i.latency_ms).sum();
    let avg_lat = if items.is_empty() {
        0
    } else {
        total_lat / items.len() as i64
    };
    let avg_tps = if total_lat > 0 {
        total_tok as f64 / (total_lat as f64 / 1000.0)
    } else {
        0.0
    };
    let mut out = format!(
        "# Vibe Grid `llm_batch` Report (`{}`)\n\n- **Prompts completed:** {}\n- **Total completion tokens:** {}\n- **Average latency:** {} ms\n- **Aggregate throughput:** {:.2} tok/s\n\n---\n\n",
        job.id,
        items.len(),
        total_tok,
        avg_lat,
        avg_tps
    );
    for it in items {
        out.push_str(&format!(
            "## {} (model: `{}`, {} tok, {} ms, {:.1} tok/s)\n\n**Prompt:**\n\n{}\n\n**Response:**\n\n{}\n\n---\n\n",
            it.id, it.model, it.completion_tokens, it.latency_ms, it.tok_per_sec, it.prompt, it.response
        ));
    }
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_three_formats() {
        let jsonl = b"{\"prompt\":\"hello\",\"id\":\"a\"}\n{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}\n";
        let items = parse_input(jsonl).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, "a");
        assert_eq!(items[1].id, "p2");

        let md = b"first prompt\n---\nsecond prompt\n---\n";
        let items = parse_input(md).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[1].prompt, "second prompt");

        let lines = b"one\ntwo\n\nthree\n";
        let items = parse_input(lines).unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[2].prompt, "three");

        assert!(parse_input(b"   ").unwrap().is_empty());
        assert!(parse_input(b"{\"bad\":true}\n").is_err());
    }

    #[test]
    fn verify_mode_pipeline() {
        let mut job = LlmBatch
            .validate(&BTreeMap::from([
                ("batch_size".into(), 2.0),
                ("verify".into(), 1.0),
            ]))
            .unwrap();
        job.id = "job-l".into();
        job.input = Some(b"prompt one\nprompt two\nprompt three\n".to_vec());
        let tasks = LlmBatch.split(&job).unwrap();
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().all(|t| t.verify_mode));
        let mut results = BTreeMap::new();
        let order: Vec<String> = tasks.iter().map(|t| t.id.clone()).collect();
        for t in &tasks {
            let (data, enc) = LlmBatch.execute(t).unwrap();
            assert_eq!(enc, ENC_JSONL);
            results.insert(t.id.clone(), data);
        }
        let (artifact, ct, ext) = LlmBatch.aggregate(&job, &order, &results).unwrap();
        assert_eq!(ext, ".jsonl");
        assert!(ct.starts_with("text/plain"));
        let text = String::from_utf8_lossy(&artifact);
        assert_eq!(text.lines().count(), 3);
        assert!(text.contains("[verify:verify-mock]"));
        // Deterministic: identical re-execution.
        let (a, _) = LlmBatch.execute(&tasks[0]).unwrap();
        let (b, _) = LlmBatch.execute(&tasks[0]).unwrap();
        assert_eq!(a, b);
        // Markdown format.
        job.meta.insert("format".into(), "md".into());
        let (md, ct, ext) = LlmBatch.aggregate(&job, &order, &results).unwrap();
        assert_eq!(ext, ".md");
        assert!(ct.starts_with("text/markdown"));
        assert!(String::from_utf8_lossy(&md).contains("# Vibe Grid `llm_batch` Report"));
    }

    #[test]
    fn validation_bounds() {
        assert!(
            LlmBatch
                .validate(&BTreeMap::from([("batch_size".into(), 64.0)]))
                .is_err()
        );
        assert!(
            LlmBatch
                .validate(&BTreeMap::from([("max_tokens".into(), 99999.0)]))
                .is_err()
        );
        let j = LlmBatch.validate(&BTreeMap::new()).unwrap();
        assert_eq!(j.width, 1);
        assert_eq!(j.max_iter, 512);
    }
}
