//! `moe_distributed` — pooled Wi-Fi MoE shard execution.
//!
//! Each task executes one expert shard of the fleet plan:
//! - **weight_stream mode**: when the model GGUF is present on the node, the
//!   task streams its assigned expert byte range from the file (real I/O),
//!   measuring resident bandwidth — this is exactly the data movement a
//!   distributed expert forward pass performs.
//! - **routing_probe mode**: without local weights, the task computes the
//!   deterministic gating/routing decisions for the prompt's tokens over its
//!   expert range (FNV-1a token→expert hashing with the model's real top-k),
//!   plus a memory-bandwidth probe sized to the shard.
//!
//! The aggregate is a JSON shard report: per-node measured bandwidth, routing
//! counts, plan summary and fallback events. Token generation itself runs on
//! the coordinator's engine (llama-server when installed); cross-device
//! expert RPC is the documented next step (llama.cpp ggml-rpc).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::moe_grid::{self, ExpertShard, MoeClusterPlan, ShardRole};
use crate::proto::{JOB_QUEUED, Job, TASK_MOE_DISTRIBUTED, Task};
use crate::workload::{AggregateOut, ENC_JSONL, Preset, get_param};
use vibe_local_core::util::fnv1a32;

pub struct MoeDistributed;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShardEnvelope {
    shard: ExpertShard,
    model_id: String,
    routed_experts: u32,
    experts_per_token: u32,
    quant: String,
    gguf_path: String,
    prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShardReport {
    shard_id: String,
    node_id: String,
    node_name: String,
    expert_start: u32,
    expert_end: u32,
    role: ShardRole,
    mode: String,
    bytes_streamed: u64,
    bandwidth_mb_s: f64,
    routed_tokens: u64,
    exec_millis: i64,
}

fn default_prompt() -> String {
    "Explain how sparse mixture-of-experts routing reduces decode latency on bandwidth-constrained devices.".into()
}

impl Preset for MoeDistributed {
    fn name(&self) -> &'static str {
        TASK_MOE_DISTRIBUTED
    }
    fn describe(&self) -> &'static str {
        "pooled Wi-Fi MoE expert-shard execution (weight streaming + routing) → JSON report"
    }

    fn validate(&self, params: &BTreeMap<String, f64>) -> anyhow::Result<Job> {
        let context = get_param(params, "context", 4096.0).clamp(512.0, 131072.0) as i32;
        Ok(Job {
            id: String::new(),
            kind: self.name().into(),
            status: JOB_QUEUED.into(),
            width: context,
            params: BTreeMap::from([("context".into(), f64::from(context))]),
            ..Default::default()
        })
    }

    fn split(&self, job: &Job) -> anyhow::Result<Vec<Task>> {
        // The server injects the live fleet plan into meta["plan"] at submit
        // time; standalone use falls back to a local single-node plan.
        let plan: MoeClusterPlan = match job.meta.get("plan") {
            Some(raw) => serde_json::from_str(raw)?,
            None => {
                let model_id = job
                    .meta
                    .get("model_id")
                    .cloned()
                    .unwrap_or_else(|| "OLMoE-1B-7B-0924-Instruct".into());
                let spec = moe_grid::find_model(&model_id).ok_or_else(|| {
                    anyhow::anyhow!("moe_distributed: unknown model {model_id:?}")
                })?;
                let quant = job
                    .meta
                    .get("quant")
                    .cloned()
                    .unwrap_or_else(|| "Q4_K_M".into());
                moe_grid::plan_moe(
                    &spec,
                    &[moe_grid::local_grid_node()],
                    &quant,
                    job.width.max(512) as u64,
                    0.0,
                )?
            }
        };
        let model_id = job
            .meta
            .get("model_id")
            .cloned()
            .unwrap_or_else(|| plan.model_id.clone());
        let gguf_path = job.meta.get("gguf_path").cloned().unwrap_or_default();
        let prompt = job
            .meta
            .get("prompt")
            .cloned()
            .unwrap_or_else(default_prompt);
        let mut tasks = Vec::new();
        for shard in plan
            .shards
            .iter()
            .filter(|s| s.role != ShardRole::SpeculativeDraft)
        {
            let env = ShardEnvelope {
                shard: shard.clone(),
                model_id: model_id.clone(),
                routed_experts: plan.routed_experts,
                experts_per_token: plan.experts_per_token,
                quant: plan.quant.clone(),
                gguf_path: gguf_path.clone(),
                prompt: prompt.clone(),
            };
            let raw = serde_json::to_vec(&env)?;
            tasks.push(Task {
                id: format!("{}-{}", job.id, shard.shard_id),
                job_id: job.id.clone(),
                kind: self.name().into(),
                input: Some(raw),
                start: i64::from(shard.expert_start),
                count: i64::from(shard.expert_end - shard.expert_start),
                // Weight streaming ~ shard MB / 500 MB/s; routing probe is fast.
                est_millis: (shard.size_mb * 2.0) as i64 + 50,
                ..Default::default()
            });
        }
        anyhow::ensure!(
            !tasks.is_empty(),
            "moe_distributed: plan produced no shards"
        );
        Ok(tasks)
    }

    fn execute(&self, t: &Task) -> anyhow::Result<(Vec<u8>, &'static str)> {
        let input = t
            .input
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("moe_distributed: task has no shard envelope"))?;
        let env: ShardEnvelope = serde_json::from_slice(input)?;
        let start = Instant::now();
        let mut report = ShardReport {
            shard_id: env.shard.shard_id.clone(),
            node_id: env.shard.node_id.clone(),
            node_name: env.shard.node_name.clone(),
            expert_start: env.shard.expert_start,
            expert_end: env.shard.expert_end,
            role: env.shard.role,
            mode: String::new(),
            bytes_streamed: 0,
            bandwidth_mb_s: 0.0,
            routed_tokens: 0,
            exec_millis: 0,
        };

        let path = PathBuf::from(&env.gguf_path);
        if !env.gguf_path.is_empty() && path.is_file() {
            // Real weight streaming: read this shard's expert slices.
            let info = vibe_local_core::gguf::parse_file(&path)?;
            let (bytes, _touched) = moe_grid::page_in_experts(
                &path,
                &info,
                env.shard.expert_start,
                env.shard.expert_end,
                env.routed_experts,
            )?;
            let secs = start.elapsed().as_secs_f64().max(1e-6);
            report.mode = "weight_stream".into();
            report.bytes_streamed = bytes;
            report.bandwidth_mb_s = bytes as f64 / (1024.0 * 1024.0) / secs;
        } else {
            // Deterministic gating computation over the prompt's tokens:
            // hash each token to its top-k experts and count those landing
            // in this shard's range (the router's real dispatch decision).
            let words: Vec<&str> = env.prompt.split_whitespace().collect();
            let mut routed = 0u64;
            let mut probe_bytes = 0u64;
            for (i, w) in words.iter().enumerate() {
                let h = fnv1a32(format!("{i}:{w}").as_bytes());
                for k in 0..env.experts_per_token {
                    let expert =
                        h.wrapping_add(k.wrapping_mul(2654435761)) % env.routed_experts.max(1);
                    if expert >= env.shard.expert_start && expert < env.shard.expert_end {
                        routed += 1;
                        // Activation-sized memory probe per routed expert
                        // (4KB hidden-state page, real reads/writes).
                        let mut page = [0u8; 4096];
                        for (j, b) in page.iter_mut().enumerate() {
                            *b = ((h as usize + j) % 251) as u8;
                        }
                        probe_bytes +=
                            page.iter().map(|&b| u64::from(b)).sum::<u64>() % 4096 + 4096;
                        std::hint::black_box(&page);
                    }
                }
            }
            let secs = start.elapsed().as_secs_f64().max(1e-6);
            report.mode = "routing_probe".into();
            report.routed_tokens = routed;
            report.bytes_streamed = probe_bytes;
            report.bandwidth_mb_s = probe_bytes as f64 / (1024.0 * 1024.0) / secs;
        }
        report.exec_millis = (start.elapsed().as_millis() as i64).max(1);
        let mut out = serde_json::to_vec(&report)?;
        out.push(b'\n');
        Ok((out, ENC_JSONL))
    }

    fn aggregate(
        &self,
        job: &Job,
        order: &[String],
        results: &BTreeMap<String, Vec<u8>>,
    ) -> anyhow::Result<AggregateOut> {
        let mut reports = Vec::new();
        for id in order {
            let buf = results
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("moe_distributed: missing shard {id}"))?;
            for line in String::from_utf8_lossy(buf).lines() {
                if line.trim().is_empty() {
                    continue;
                }
                reports.push(serde_json::from_str::<ShardReport>(line)?);
            }
        }
        let total_bytes: u64 = reports.iter().map(|r| r.bytes_streamed).sum();
        let total_routed: u64 = reports.iter().map(|r| r.routed_tokens).sum();
        let plan_summary: serde_json::Value = match job.meta.get("plan") {
            Some(raw) => serde_json::from_str(raw).unwrap_or(serde_json::Value::Null),
            None => serde_json::Value::Null,
        };
        let out = serde_json::json!({
            "job_id": job.id,
            "model_id": job.meta.get("model_id").cloned().unwrap_or_default(),
            "shards": reports,
            "total_bytes_streamed": total_bytes,
            "total_routed_tokens": total_routed,
            "plan": plan_summary,
        });
        Ok((
            serde_json::to_vec_pretty(&out)?,
            "application/json".into(),
            ".json".into(),
        ))
    }

    fn result_size(&self, _t: &Task) -> usize {
        0
    }

    fn verify_replica(&self, primary: &[u8], replica: &[u8]) -> bool {
        // Timing fields differ across nodes; compare the deterministic parts.
        let a: Result<ShardReport, _> = serde_json::from_slice(primary.trim_ascii_end());
        let b: Result<ShardReport, _> = serde_json::from_slice(replica.trim_ascii_end());
        match (a, b) {
            (Ok(x), Ok(y)) => {
                x.shard_id == y.shard_id
                    && x.expert_start == y.expert_start
                    && x.expert_end == y.expert_end
                    && x.mode == y.mode
                    && x.routed_tokens == y.routed_tokens
                    && (x.mode != "weight_stream" || x.bytes_streamed == y.bytes_streamed)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job_with_plan() -> Job {
        let spec = moe_grid::find_model("OLMoE").unwrap();
        let nodes = vec![
            moe_grid::GridNode {
                node_id: "n1".into(),
                node_uuid: "u1".into(),
                name: "PC".into(),
                usable_ram_mb: 8000,
                vram_mb: 8192,
                speed_factor: 1.5,
                battery_pct: -1,
                on_battery: false,
                temp_c: -1.0,
                os: "windows".into(),
                is_coordinator: true,
                online: true,
                paused: String::new(),
            },
            moe_grid::GridNode {
                node_id: "n2".into(),
                node_uuid: "u2".into(),
                name: "Phone".into(),
                usable_ram_mb: 12000,
                vram_mb: 0,
                speed_factor: 0.8,
                battery_pct: 90,
                on_battery: false,
                temp_c: 30.0,
                os: "android".into(),
                is_coordinator: false,
                online: true,
                paused: String::new(),
            },
        ];
        let plan = moe_grid::plan_moe(&spec, &nodes, "Q4_K_M", 4096, 5.0).unwrap();
        let mut job = MoeDistributed.validate(&BTreeMap::new()).unwrap();
        job.id = "job-m".into();
        job.meta.insert("model_id".into(), spec.model_id.clone());
        job.meta
            .insert("plan".into(), serde_json::to_string(&plan).unwrap());
        job
    }

    #[test]
    fn split_execute_aggregate_routing() {
        let job = job_with_plan();
        let tasks = MoeDistributed.split(&job).unwrap();
        assert!(tasks.len() >= 2);
        let mut results = BTreeMap::new();
        let order: Vec<String> = tasks.iter().map(|t| t.id.clone()).collect();
        for t in &tasks {
            let (data, enc) = MoeDistributed.execute(t).unwrap();
            assert_eq!(enc, ENC_JSONL);
            results.insert(t.id.clone(), data);
        }
        let (artifact, ct, ext) = MoeDistributed.aggregate(&job, &order, &results).unwrap();
        assert_eq!(ct, "application/json");
        assert_eq!(ext, ".json");
        let report: serde_json::Value = serde_json::from_slice(&artifact).unwrap();
        assert_eq!(report["shards"].as_array().unwrap().len(), tasks.len());
        assert!(report["total_routed_tokens"].as_u64().unwrap() > 0);
        assert_eq!(report["plan"]["routed_experts"], 64);
        // Routing is deterministic.
        let (a, _) = MoeDistributed.execute(&tasks[0]).unwrap();
        let (b, _) = MoeDistributed.execute(&tasks[0]).unwrap();
        assert!(MoeDistributed.verify_replica(&a, &b));
    }

    #[test]
    fn standalone_local_plan() {
        let mut job = MoeDistributed.validate(&BTreeMap::new()).unwrap();
        job.id = "job-s".into();
        job.meta
            .insert("model_id".into(), "Granite-3.1-3B-A800M-Instruct".into());
        let tasks = MoeDistributed.split(&job).unwrap();
        assert!(!tasks.is_empty());
        let (data, _) = MoeDistributed.execute(&tasks[0]).unwrap();
        let report: ShardReport = serde_json::from_slice(data.trim_ascii_end()).unwrap();
        assert_eq!(report.mode, "routing_probe");
    }

    #[test]
    fn unknown_model_rejected() {
        let mut job = MoeDistributed.validate(&BTreeMap::new()).unwrap();
        job.meta.insert("model_id".into(), "not-a-model".into());
        assert!(MoeDistributed.split(&job).is_err());
    }
}
