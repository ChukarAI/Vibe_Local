//! Distributed MoE over Wi-Fi: expert & layer sharding across phones + PC.
//!
//! Sparse MoE models activate only top-k experts per token, so — unlike dense
//! pipeline parallelism, which pays Wi-Fi latency on *every* layer — expert
//! shards can live on different devices and only the routed activations
//! cross the network. This module computes the fleet partition
//! (`MoeClusterPlan`), routes experts to shard holders at runtime, and
//! re-routes or pages shards when a phone drops (battery/thermal/Wi-Fi).
//!
//! Execution model: the coordinator hosts the shared backbone (embedding,
//! attention, gating) plus the hottest experts; phone workers host contiguous
//! expert ranges or act as speculative draft generators. When a shard holder
//! vanishes, its expert range is re-assigned to the node with the most
//! headroom, or paged into coordinator host RAM/disk from the GGUF file —
//! the generation stream continues without crashing.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use vibe_local_core::gguf::{self, GgufInfo};
use vibe_local_core::recommender::bits_per_weight;

/// Public expert topology of well-known open MoE instruct models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeModelSpec {
    pub model_id: String,
    pub hf_repo: String,
    pub architecture: String,
    pub total_params_b: f64,
    pub active_params_b: f64,
    /// Non-expert (backbone) parameters: embedding + attention + gating.
    pub backbone_params_b: f64,
    pub routed_experts: u32,
    pub shared_experts: u32,
    pub experts_per_token: u32,
    pub context_length: u64,
}

pub fn moe_registry() -> Vec<MoeModelSpec> {
    vec![
        MoeModelSpec {
            model_id: "OLMoE-1B-7B-0924-Instruct".into(),
            hf_repo: "bartowski/OLMoE-1B-7B-0924-Instruct-GGUF".into(),
            architecture: "olmoe".into(),
            total_params_b: 6.9,
            active_params_b: 1.3,
            backbone_params_b: 1.0,
            routed_experts: 64,
            shared_experts: 0,
            experts_per_token: 8,
            context_length: 4096,
        },
        MoeModelSpec {
            model_id: "Granite-3.1-3B-A800M-Instruct".into(),
            hf_repo: "bartowski/granite-3.1-3b-a800m-instruct-GGUF".into(),
            architecture: "granitemoe".into(),
            total_params_b: 3.3,
            active_params_b: 0.8,
            backbone_params_b: 0.5,
            routed_experts: 64,
            shared_experts: 0,
            experts_per_token: 8,
            context_length: 131072,
        },
        MoeModelSpec {
            model_id: "Qwen1.5-MoE-A2.7B-Chat".into(),
            hf_repo: "bartowski/Qwen1.5-MoE-A2.7B-Chat-GGUF".into(),
            architecture: "qwen2moe".into(),
            total_params_b: 14.3,
            active_params_b: 2.7,
            backbone_params_b: 2.0,
            routed_experts: 60,
            shared_experts: 4,
            experts_per_token: 4,
            context_length: 8192,
        },
        MoeModelSpec {
            model_id: "DeepSeek-V2-Lite-16B".into(),
            hf_repo: "bartowski/DeepSeek-V2-Lite-Chat-GGUF".into(),
            architecture: "deepseek2".into(),
            total_params_b: 15.7,
            active_params_b: 2.4,
            backbone_params_b: 2.4,
            routed_experts: 64,
            shared_experts: 2,
            experts_per_token: 6,
            context_length: 32768,
        },
    ]
}

pub fn find_model(model_id: &str) -> Option<MoeModelSpec> {
    let needle = model_id.to_lowercase();
    moe_registry().into_iter().find(|m| {
        m.model_id.to_lowercase() == needle
            || m.model_id.to_lowercase().contains(&needle)
            || needle.contains(&m.model_id.to_lowercase())
    })
}

/// Live capabilities of one fleet member, as reported by the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridNode {
    pub node_id: String,
    pub node_uuid: String,
    pub name: String,
    pub usable_ram_mb: u64,
    pub vram_mb: u64,
    /// Learned EWMA speed factor (1.0 = reference device).
    pub speed_factor: f64,
    pub battery_pct: i32,
    pub on_battery: bool,
    pub temp_c: f64,
    pub os: String,
    pub is_coordinator: bool,
    pub online: bool,
    pub paused: String,
}

impl GridNode {
    pub fn usable(&self) -> bool {
        self.online && self.paused.is_empty() && self.usable_ram_mb > 0
    }

    /// Placement weight: memory capacity scaled by measured speed.
    pub fn weight(&self) -> f64 {
        self.usable_ram_mb as f64 * self.speed_factor.max(0.1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShardRole {
    /// Shared embedding/attention/gating + hottest experts (coordinator).
    Backbone,
    /// Contiguous routed-expert range on a worker.
    ExpertRange,
    /// Fast speculative draft generator (draft-verify against coordinator).
    SpeculativeDraft,
    /// Expert range paged into coordinator host RAM/disk after a node drop.
    HostPaging,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertShard {
    pub shard_id: String,
    /// Inclusive start, exclusive end of the routed-expert range.
    pub expert_start: u32,
    pub expert_end: u32,
    pub node_id: String,
    pub node_name: String,
    pub size_mb: f64,
    pub role: ShardRole,
    /// Popularity rank mass (Zipf prior) — hotter shards decode faster when
    /// placed on fast nodes.
    pub heat: f64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeClusterPlan {
    pub model_id: String,
    pub architecture: String,
    pub quant: String,
    pub context: u64,
    pub routed_experts: u32,
    pub experts_per_token: u32,
    pub total_params_b: f64,
    pub active_params_b: f64,
    pub backbone_node: String,
    pub shards: Vec<ExpertShard>,
    pub draft_nodes: Vec<String>,
    pub draft_tokens: u32,
    pub wifi_rtt_ms: f64,
    pub est_decode_tps: f64,
    pub est_speedup_vs_single_node: f64,
    pub warnings: Vec<String>,
}

impl MoeClusterPlan {
    pub fn shard_for_expert(&self, expert: u32) -> Option<&ExpertShard> {
        self.shards.iter().find(|s| {
            matches!(
                s.role,
                ShardRole::ExpertRange | ShardRole::HostPaging | ShardRole::Backbone
            ) && expert >= s.expert_start
                && expert < s.expert_end
        })
    }

    pub fn expert_coverage(&self) -> u32 {
        self.shards
            .iter()
            .filter(|s| s.role != ShardRole::SpeculativeDraft)
            .map(|s| s.expert_end - s.expert_start)
            .sum()
    }
}

/// KV-cache + compute overhead for the backbone at a context length (MiB).
fn backbone_overhead_mb(spec: &MoeModelSpec, ctx: u64, quant: &str) -> f64 {
    let bpw = bits_per_weight(quant).unwrap_or(4.85);
    let weights = spec.backbone_params_b * 1e9 * bpw / 8.0 / (1024.0 * 1024.0);
    let layers = (spec.total_params_b * 3.0).clamp(12.0, 48.0);
    let kv = 2.0 * layers * 1024.0 * 2.0 * ctx as f64 / (1024.0 * 1024.0);
    weights + kv + 384.0
}

/// Bytes per routed expert at a quantization.
pub fn expert_bytes(spec: &MoeModelSpec, quant: &str) -> f64 {
    let bpw = bits_per_weight(quant).unwrap_or(4.85);
    let expert_params =
        (spec.total_params_b - spec.backbone_params_b) * 1e9 / f64::from(spec.routed_experts);
    expert_params * bpw / 8.0
}

/// Zipf-like routing popularity prior over expert indices (deterministic).
/// Real routing is input-dependent; this prior only guides *placement* of
/// hot experts on fast nodes.
fn expert_heat(expert: u32, total: u32) -> f64 {
    let rank = f64::from(expert) + 1.0;
    let norm: f64 = (1..=total).map(|i| 1.0 / f64::from(i)).sum();
    (1.0 / rank) / norm
}

/// Partition a MoE model across the fleet proportional to
/// `usable_ram_mb * speed_factor`. The coordinator keeps the backbone and
/// the hottest experts that fit; workers take contiguous expert ranges;
/// nodes too small for a full expert become speculative draft generators.
pub fn plan_moe(
    spec: &MoeModelSpec,
    nodes: &[GridNode],
    quant: &str,
    context: u64,
    wifi_rtt_ms: f64,
) -> anyhow::Result<MoeClusterPlan> {
    let mut usable: Vec<&GridNode> = nodes.iter().filter(|n| n.usable()).collect();
    anyhow::ensure!(
        !usable.is_empty(),
        "no online, unpaused nodes with usable RAM in the fleet"
    );
    // Coordinator first, then by weight descending.
    usable.sort_by(|a, b| {
        b.is_coordinator.cmp(&a.is_coordinator).then(
            b.weight()
                .partial_cmp(&a.weight())
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    let coordinator = usable[0];
    let mut warnings = Vec::new();
    if usable.len() == 1 {
        warnings.push(
            "single-node fleet: all experts stay on the coordinator (no Wi-Fi sharding)".into(),
        );
    }
    for n in &usable {
        if n.battery_pct >= 0 && n.on_battery && n.battery_pct < 30 {
            warnings.push(format!(
                "{}: battery {}% — guard will pause it mid-run",
                n.name, n.battery_pct
            ));
        }
    }

    let eb = expert_bytes(spec, quant);
    let expert_mb = eb / (1024.0 * 1024.0);
    let backbone_mb = backbone_overhead_mb(spec, context, quant);
    anyhow::ensure!(
        coordinator.usable_ram_mb as f64 + coordinator.vram_mb as f64 > backbone_mb,
        "coordinator {} cannot host the backbone ({backbone_mb:.0}MB needed)",
        coordinator.name
    );

    let total = spec.routed_experts;
    let mut shards: Vec<ExpertShard> = Vec::new();
    let mut draft_nodes: Vec<String> = Vec::new();

    // Proportional quotas: each node earns experts in proportion to
    // usable_ram_mb * speed_factor. The coordinator's quota includes the
    // backbone; nodes that cannot hold a single expert become drafters.
    let total_weight: f64 = usable.iter().map(|n| n.weight()).sum();
    let mut quotas: BTreeMap<String, u32> = BTreeMap::new();
    let mut assigned_sum = 0u32;
    for n in &usable {
        let quota = (f64::from(total) * n.weight() / total_weight).floor() as u32;
        quotas.insert(n.node_id.clone(), quota);
        assigned_sum += quota;
    }
    // Remainder to the coordinator (it hosts the backbone anyway).
    quotas
        .entry(coordinator.node_id.clone())
        .and_modify(|q| *q += total - assigned_sum);
    // Capacity clamp: a node cannot hold more experts than its RAM allows.
    let mut overflow = 0u32;
    for n in &usable {
        let cap_mb = (n.usable_ram_mb + n.vram_mb) as f64
            - if n.node_id == coordinator.node_id {
                backbone_mb
            } else {
                256.0
            };
        let max_experts = (cap_mb.max(0.0) / expert_mb).floor() as u32;
        let q = quotas[&n.node_id];
        if q > max_experts {
            overflow += q - max_experts;
            quotas.insert(n.node_id.clone(), max_experts);
        }
        if quotas[&n.node_id] == 0 && n.node_id != coordinator.node_id {
            draft_nodes.push(n.node_id.clone());
            warnings.push(format!(
                "{}: too little RAM for one expert ({expert_mb:.0}MB) — assigned speculative draft role",
                n.name
            ));
        }
    }
    if overflow > 0 {
        // Overflow pages onto the coordinator (host-RAM/disk paging).
        quotas
            .entry(coordinator.node_id.clone())
            .and_modify(|q| *q += overflow);
        warnings.push(format!(
            "{overflow} experts exceed worker capacity — coordinator host-RAM paging required"
        ));
    }

    // Experts are placed in heat order (Zipf prior: expert 0 hottest). The
    // coordinator takes the hottest block; remaining blocks go to workers
    // fastest-first. Heat order == index order, so ranges stay contiguous.
    let mut workers: Vec<&&GridNode> = usable
        .iter()
        .filter(|n| n.node_id != coordinator.node_id)
        .collect();
    workers.sort_by(|a, b| {
        b.speed_factor
            .partial_cmp(&a.speed_factor)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let placement_order: Vec<&GridNode> = std::iter::once(coordinator)
        .chain(workers.into_iter().copied())
        .collect();
    let mut cursor = 0u32;
    for (i, n) in placement_order.iter().enumerate() {
        let quota = quotas[&n.node_id];
        if quota == 0 {
            continue;
        }
        let start = cursor;
        let end = (cursor + quota).min(total);
        cursor = end;
        let heat: f64 = (start..end).map(|e| expert_heat(e, total)).sum();
        shards.push(ExpertShard {
            shard_id: format!("{}-s0", n.node_id),
            expert_start: start,
            expert_end: end,
            node_id: n.node_id.clone(),
            node_name: n.name.clone(),
            size_mb: f64::from(end - start) * expert_mb,
            role: if i == 0 {
                ShardRole::Backbone
            } else {
                ShardRole::ExpertRange
            },
            heat,
            status: "active".into(),
        });
        if cursor >= total {
            break;
        }
    }
    anyhow::ensure!(
        shards
            .iter()
            .map(|s| s.expert_end - s.expert_start)
            .sum::<u32>()
            == total,
        "internal error: expert coverage incomplete"
    );

    // Throughput estimate: decode is bounded by the slowest active expert
    // holder's effective bandwidth plus Wi-Fi round trips for routed tokens.
    let holders: Vec<&&GridNode> = usable
        .iter()
        .filter(|n| {
            shards
                .iter()
                .any(|s| s.node_id == n.node_id && s.role != ShardRole::SpeculativeDraft)
        })
        .collect();
    let min_speed = holders
        .iter()
        .map(|n| n.speed_factor.max(0.1))
        .fold(f64::MAX, f64::min);
    // Reference: ~30 tok/s dense-7B-equivalent on the reference device;
    // sparse activation moves active/total of the bytes per token.
    let sparsity = spec.active_params_b / spec.total_params_b;
    let wifi_penalty_ms = wifi_rtt_ms * f64::from(spec.experts_per_token.min(4));
    let base_tps = 30.0 * min_speed / sparsity.max(0.05);
    let est_decode_tps = (base_tps * 1000.0 / (1000.0 + wifi_penalty_ms * base_tps / 8.0)).max(0.1);
    let single_node_tps = 30.0 * coordinator.speed_factor.max(0.1) / sparsity.max(0.05);

    Ok(MoeClusterPlan {
        model_id: spec.model_id.clone(),
        architecture: spec.architecture.clone(),
        quant: quant.to_string(),
        context,
        routed_experts: total,
        experts_per_token: spec.experts_per_token,
        total_params_b: spec.total_params_b,
        active_params_b: spec.active_params_b,
        backbone_node: coordinator.node_id.clone(),
        shards,
        draft_nodes,
        draft_tokens: 4,
        wifi_rtt_ms,
        est_decode_tps,
        est_speedup_vs_single_node: est_decode_tps / single_node_tps.max(0.1),
        warnings,
    })
}

/// Re-route every shard held by a failed node to the best remaining node,
/// or to coordinator host paging when no worker can absorb it. Never fails:
/// the generation stream must survive arbitrary node churn.
pub fn reroute_after_drop(
    plan: &mut MoeClusterPlan,
    failed_node: &str,
    nodes: &[GridNode],
    spec: &MoeModelSpec,
    quant: &str,
) -> Vec<String> {
    let mut events = Vec::new();
    let expert_mb = expert_bytes(spec, quant) / (1024.0 * 1024.0);
    let survivors: Vec<&GridNode> = nodes
        .iter()
        .filter(|n| n.usable() && n.node_id != failed_node)
        .collect();
    // Headroom already committed per surviving node.
    let mut committed: BTreeMap<String, f64> = BTreeMap::new();
    for s in &plan.shards {
        if s.node_id != failed_node {
            *committed.entry(s.node_id.clone()).or_insert(0.0) += s.size_mb;
        }
    }
    let dropped: Vec<usize> = plan
        .shards
        .iter()
        .enumerate()
        .filter(|(_, s)| s.node_id == failed_node)
        .map(|(i, _)| i)
        .collect();
    for idx in dropped {
        let need = plan.shards[idx].size_mb;
        let mut best: Option<(&GridNode, f64)> = None;
        for n in &survivors {
            let free = (n.usable_ram_mb + n.vram_mb) as f64
                - committed.get(&n.node_id).copied().unwrap_or(0.0)
                - 256.0;
            if free >= need && best.is_none_or(|(_, f)| free > f) {
                best = Some((n, free));
            }
        }
        match best {
            Some((n, _)) => {
                *committed.entry(n.node_id.clone()).or_insert(0.0) += need;
                let s = &mut plan.shards[idx];
                events.push(format!(
                    "shard {} (experts {}..{}) re-routed {} → {}",
                    s.shard_id, s.expert_start, s.expert_end, failed_node, n.node_id
                ));
                s.node_id = n.node_id.clone();
                s.node_name = n.name.clone();
                s.status = "rerouted".into();
            }
            None => {
                let s = &mut plan.shards[idx];
                events.push(format!(
                    "shard {} (experts {}..{}) paged into coordinator host RAM/disk",
                    s.shard_id, s.expert_start, s.expert_end
                ));
                s.node_id = plan.backbone_node.clone();
                s.node_name = "coordinator (host paging)".into();
                s.role = ShardRole::HostPaging;
                s.status = "paging".into();
            }
        }
    }
    plan.draft_nodes.retain(|d| d != failed_node);
    if plan.backbone_node == failed_node {
        // Backbone loss: promote the strongest survivor.
        if let Some(n) = survivors.iter().max_by(|a, b| {
            a.weight()
                .partial_cmp(&b.weight())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            events.push(format!("backbone promoted {} → {}", failed_node, n.node_id));
            plan.backbone_node = n.node_id.clone();
            for s in plan
                .shards
                .iter_mut()
                .filter(|s| s.role == ShardRole::Backbone)
            {
                s.role = ShardRole::ExpertRange;
            }
        }
    }
    let _ = expert_mb;
    events
}

/// Runtime shard router: tracks node liveness and re-routes on drops.
pub struct MoeGridRouter {
    pub plan: MoeClusterPlan,
    pub spec: MoeModelSpec,
    pub quant: String,
    pub events: Vec<String>,
}

impl MoeGridRouter {
    pub fn new(plan: MoeClusterPlan, spec: MoeModelSpec, quant: &str) -> Self {
        Self {
            plan,
            spec,
            quant: quant.to_string(),
            events: Vec::new(),
        }
    }

    /// Handle a node drop (Wi-Fi loss, battery/thermal guard, crash).
    pub fn on_node_drop(&mut self, node_id: &str, nodes: &[GridNode]) -> Vec<String> {
        let events = reroute_after_drop(&mut self.plan, node_id, nodes, &self.spec, &self.quant);
        self.events.extend(events.clone());
        events
    }

    pub fn holder_of(&self, expert: u32) -> Option<&ExpertShard> {
        self.plan.shard_for_expert(expert)
    }
}

/// Page an expert range from a real GGUF file into host memory.
///
/// GGUF stores routed experts as stacked tensors (`ffn_*_exps`); each expert
/// occupies an equal slice. Returns the bytes actually read and the tensor
/// names touched — real I/O, used by the host-paging fallback.
pub fn page_in_experts(
    gguf_path: &Path,
    info: &GgufInfo,
    expert_start: u32,
    expert_end: u32,
    routed_experts: u32,
) -> anyhow::Result<(u64, Vec<String>)> {
    use std::io::{Read, Seek, SeekFrom};
    anyhow::ensure!(
        expert_end > expert_start && expert_end <= routed_experts,
        "bad expert range"
    );
    let mut file = std::fs::File::open(gguf_path)?;
    let mut total = 0u64;
    let mut touched = Vec::new();
    for tensor in &info.tensors {
        if !tensor.name.contains("_exps") {
            continue;
        }
        // Stacked expert tensor: split evenly across routed experts.
        // Saturating throughout: hostile dims/offsets must not wrap into
        // plausible-looking slice ranges.
        let n_elems = tensor.n_elements();
        let bytes_per_elem = quant_tensor_bytes(tensor.ty);
        let tensor_bytes = n_elems.saturating_mul(bytes_per_elem);
        let per_expert = tensor_bytes / u64::from(routed_experts.max(1));
        if per_expert == 0 {
            continue;
        }
        let start = info
            .data_offset
            .saturating_add(tensor.offset)
            .saturating_add(per_expert.saturating_mul(u64::from(expert_start)));
        let len = per_expert.saturating_mul(u64::from(expert_end - expert_start));
        let len = len.min(info.file_size_bytes.saturating_sub(start));
        if len == 0 {
            continue;
        }
        file.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; len as usize];
        file.read_exact(&mut buf)?;
        // Touch the pages so the OS actually residents them (real paging).
        let checksum: u64 = buf
            .chunks(4096)
            .map(|c| c.iter().map(|&b| u64::from(b)).sum::<u64>())
            .sum();
        std::hint::black_box(checksum);
        total += len;
        touched.push(tensor.name.clone());
    }
    Ok((total, touched))
}

/// Bytes per element for common GGUF tensor types (block quants averaged).
fn quant_tensor_bytes(ty: u32) -> u64 {
    match ty {
        0 => 4,       // F32
        1 => 2,       // F16
        2 | 3 => 1,   // Q4_0/Q4_1 (approx, block overhead ignored)
        6 | 7 => 1,   // Q5_0/Q5_1
        8 => 1,       // Q8_0
        10 => 1,      // Q2_K
        11..=13 => 1, // Q3_K
        14 | 15 => 1, // Q4_K
        16 | 17 => 1, // Q5_K
        18 => 1,      // Q6_K
        23 | 26 => 1, // IQ4_NL / IQ4_XS
        28 => 2,      // BF16
        _ => 1,
    }
}

/// Build GridNodes from a live cluster state snapshot.
pub fn grid_nodes_from_state(
    state: &crate::proto::StateResponse,
    coordinator_name: &str,
) -> Vec<GridNode> {
    state
        .nodes
        .iter()
        .map(|n| GridNode {
            node_id: n.id.clone(),
            node_uuid: n.node_uuid.clone(),
            name: n.name.clone(),
            usable_ram_mb: n.caps.mem_mb,
            vram_mb: 0,
            speed_factor: if n.speed_factor > 0.0 {
                n.speed_factor
            } else {
                n.caps.score.max(0.1)
            },
            battery_pct: n.caps.battery_pct,
            on_battery: n.caps.on_battery,
            temp_c: n.caps.temp_c,
            os: n.caps.os.clone(),
            is_coordinator: n.name == coordinator_name,
            online: n.online,
            paused: n.paused.clone(),
        })
        .collect()
}

/// Local single-machine fleet (used by `plan-moe` without a coordinator).
pub fn local_grid_node() -> GridNode {
    let hw = vibe_local_core::hardware::HardwareProbe::scan();
    let vram = hw.gpus.iter().filter_map(|g| g.vram_mb).max().unwrap_or(0);
    GridNode {
        node_id: "local".into(),
        node_uuid: crate::node::hardware_fingerprint_uuid(),
        name: crate::node::hostname(),
        usable_ram_mb: hw.ram_free_mb,
        vram_mb: vram,
        speed_factor: 1.0,
        battery_pct: -1,
        on_battery: false,
        temp_c: -1.0,
        os: hw.os,
        is_coordinator: true,
        online: true,
        paused: String::new(),
    }
}

/// Parse a GGUF and refine a registry spec with the file's real topology.
pub fn spec_from_gguf(
    path: &Path,
    fallback: &MoeModelSpec,
) -> anyhow::Result<(MoeModelSpec, GgufInfo)> {
    let info = gguf::parse_file(path)?;
    let mut spec = fallback.clone();
    if let Some(e) = info.expert_count {
        spec.routed_experts = e as u32;
    }
    if let Some(u) = info.expert_used_count {
        spec.experts_per_token = u as u32;
    }
    if let Some(ctx) = info.context_length {
        spec.context_length = ctx;
    }
    spec.architecture = info.architecture.clone();
    Ok((spec, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pc() -> GridNode {
        GridNode {
            node_id: "n1".into(),
            node_uuid: "uuid-pc".into(),
            name: "PC-RX580".into(),
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
        }
    }

    fn phone(id: &str, ram: u64, speed: f64) -> GridNode {
        GridNode {
            node_id: id.into(),
            node_uuid: format!("uuid-{id}"),
            name: format!("Phone-{id}"),
            usable_ram_mb: ram,
            vram_mb: 0,
            speed_factor: speed,
            battery_pct: 85,
            on_battery: false,
            temp_c: 35.0,
            os: "android".into(),
            is_coordinator: false,
            online: true,
            paused: String::new(),
        }
    }

    fn olmoe() -> MoeModelSpec {
        find_model("OLMoE").unwrap()
    }

    #[test]
    fn heterogeneous_partition_covers_all_experts() {
        let nodes = vec![pc(), phone("n2", 12000, 0.8), phone("n3", 6000, 0.6)];
        let plan = plan_moe(&olmoe(), &nodes, "Q4_K_M", 4096, 5.0).unwrap();
        assert_eq!(plan.expert_coverage(), 64);
        assert_eq!(plan.backbone_node, "n1");
        // Every expert maps to exactly one shard.
        for e in 0..64 {
            assert!(plan.shard_for_expert(e).is_some(), "expert {e} uncovered");
        }
        // Coordinator hosts the backbone shard.
        assert!(
            plan.shards
                .iter()
                .any(|s| s.role == ShardRole::Backbone && s.node_id == "n1")
        );
        // Phones hold expert ranges.
        assert!(plan.shards.iter().any(|s| s.node_id == "n2"));
        assert!(plan.shards.iter().any(|s| s.node_id == "n3"));
        // Bigger/faster phone gets more experts.
        let n2_experts: u32 = plan
            .shards
            .iter()
            .filter(|s| s.node_id == "n2")
            .map(|s| s.expert_end - s.expert_start)
            .sum();
        let n3_experts: u32 = plan
            .shards
            .iter()
            .filter(|s| s.node_id == "n3")
            .map(|s| s.expert_end - s.expert_start)
            .sum();
        assert!(n2_experts > n3_experts, "n2={n2_experts} n3={n3_experts}");
        assert!(plan.est_decode_tps > 0.0);
    }

    #[test]
    fn tiny_phone_becomes_draft_generator() {
        let nodes = vec![pc(), phone("n2", 12000, 0.8), phone("tiny", 100, 0.5)];
        let plan = plan_moe(&olmoe(), &nodes, "Q4_K_M", 4096, 5.0).unwrap();
        assert!(plan.draft_nodes.contains(&"tiny".to_string()));
        assert_eq!(plan.expert_coverage(), 64);
        assert!(
            plan.warnings
                .iter()
                .any(|w| w.contains("speculative draft"))
        );
    }

    #[test]
    fn node_drop_reroutes_without_losing_experts() {
        let nodes = vec![pc(), phone("n2", 12000, 0.8), phone("n3", 6000, 0.6)];
        let plan = plan_moe(&olmoe(), &nodes, "Q4_K_M", 4096, 5.0).unwrap();
        let mut router = MoeGridRouter::new(plan, olmoe(), "Q4_K_M");
        // Phone n2 drops (Wi-Fi loss).
        let mut surviving = nodes.clone();
        surviving.retain(|n| n.node_id != "n2");
        let events = router.on_node_drop("n2", &surviving);
        assert!(!events.is_empty());
        assert_eq!(router.plan.expert_coverage(), 64);
        for e in 0..64 {
            let holder = router.holder_of(e).unwrap();
            assert_ne!(holder.node_id, "n2");
        }
        // Drop n3 too: everything lands on the coordinator (paging allowed).
        surviving.retain(|n| n.node_id != "n3");
        let events = router.on_node_drop("n3", &surviving);
        assert!(!events.is_empty());
        assert_eq!(router.plan.expert_coverage(), 64);
        assert!(router.plan.shards.iter().all(|s| s.node_id == "n1"));
    }

    #[test]
    fn backbone_drop_promotes_survivor() {
        let nodes = vec![pc(), phone("n2", 12000, 0.8)];
        let plan = plan_moe(&olmoe(), &nodes, "Q4_K_M", 4096, 5.0).unwrap();
        let mut router = MoeGridRouter::new(plan, olmoe(), "Q4_K_M");
        let surviving = vec![phone("n2", 12000, 0.8)];
        router.on_node_drop("n1", &surviving);
        assert_eq!(router.plan.backbone_node, "n2");
        assert_eq!(router.plan.expert_coverage(), 64);
    }

    #[test]
    fn paused_nodes_excluded() {
        let mut hot = phone("n2", 12000, 0.8);
        hot.paused = "thermal 85°C".into();
        let nodes = vec![pc(), hot];
        let plan = plan_moe(&olmoe(), &nodes, "Q4_K_M", 4096, 5.0).unwrap();
        assert!(plan.shards.iter().all(|s| s.node_id == "n1"));
    }

    #[test]
    fn empty_fleet_errors() {
        assert!(plan_moe(&olmoe(), &[], "Q4_K_M", 4096, 5.0).is_err());
        let offline = vec![GridNode {
            online: false,
            ..pc()
        }];
        assert!(plan_moe(&olmoe(), &offline, "Q4_K_M", 4096, 5.0).is_err());
    }

    #[test]
    fn registry_facts() {
        let g = find_model("Granite-3.1-3B-A800M").unwrap();
        assert_eq!(g.routed_experts, 64);
        assert_eq!(g.experts_per_token, 8);
        let d = find_model("DeepSeek-V2-Lite").unwrap();
        assert_eq!(d.shared_experts, 2);
        assert!(find_model("nonexistent-model").is_none());
        // Expert bytes plausible: OLMoE ~92M params/expert at Q4_K_M ≈ 56MB.
        let mb = expert_bytes(&olmoe(), "Q4_K_M") / (1024.0 * 1024.0);
        assert!((40.0..80.0).contains(&mb), "expert size {mb}MB");
    }

    #[test]
    fn paging_reads_real_bytes() {
        // Build a synthetic GGUF with a stacked expert tensor.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("moe-Q4_K_M.gguf");
        let mut buf = Vec::new();
        buf.extend_from_slice(&gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 KV
        let key = b"general.architecture";
        let val = b"olmoe";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);
        let tname = b"blk.0.ffn_gate_exps.weight";
        buf.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        buf.extend_from_slice(tname);
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&64u64.to_le_bytes()); // 64 experts
        buf.extend_from_slice(&1024u64.to_le_bytes());
        buf.extend_from_slice(&15u32.to_le_bytes()); // Q4_K_M
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        let data_offset = (buf.len() as u64).div_ceil(32) * 32;
        buf.resize(data_offset as usize, 0);
        buf.extend_from_slice(&vec![0xABu8; 64 * 1024]); // expert data
        std::fs::write(&path, &buf).unwrap();

        let info = gguf::parse_file(&path).unwrap();
        assert_eq!(info.tensors.len(), 1);
        let (bytes, touched) = page_in_experts(&path, &info, 0, 32, 64).unwrap();
        assert_eq!(bytes, 32 * 1024);
        assert_eq!(touched, vec!["blk.0.ffn_gate_exps.weight"]);
        let (bytes2, _) = page_in_experts(&path, &info, 32, 64, 64).unwrap();
        assert_eq!(bytes2, 32 * 1024);
        assert!(page_in_experts(&path, &info, 0, 65, 64).is_err());
    }
}
