//! `vibe-local cluster …` subcommands: coordinator, agent, plan-moe,
//! submit, status, nodes, jobs, cancel. Wire-compatible with compost v0.7.0.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Subcommand;
use serde_json::json;

use vibe_local_cluster::proto::{JOB_CANCELLED, JOB_DONE, JOB_FAILED};
use vibe_local_cluster::{agent, server};

#[derive(Subcommand)]
pub enum ClusterCommand {
    /// Run the Vibe Grid coordinator (pull-based fleet brain)
    Coordinator {
        #[arg(long, default_value = "0.0.0.0:7900")]
        addr: String,
        /// Fleet shared secret (X-Compost-Key); empty = open fleet
        #[arg(long, default_value = "")]
        key: String,
        /// Require the fleet key on ALL artifact/tile reads
        #[arg(long)]
        protect_artifacts: bool,
        /// Snapshot path (enables crash-safe persistence + artifact disk offload)
        #[arg(long)]
        state: Option<PathBuf>,
        /// Allow two distinct NodeUUIDs on one IP to confirm replicas (localhost tests)
        #[arg(long)]
        allow_same_ip_replicate: bool,
    },
    /// Run a worker agent (outward-only: join/poll/heartbeat/result)
    Agent {
        #[arg(long)]
        coordinator: String,
        #[arg(long, default_value = "")]
        key: String,
        #[arg(long, default_value_t = 2)]
        workers: usize,
        #[arg(long, default_value = "")]
        name: String,
        #[arg(long, default_value = "")]
        node_id_file: String,
        /// Override the <30% battery pause (thermal guards are never overridable)
        #[arg(long)]
        battery_guard_off: bool,
    },
    /// Compute the MoE expert/RAM shard map across the live fleet
    PlanMoe {
        model_id: String,
        #[arg(long, default_value_t = 4096)]
        context: u64,
        #[arg(long)]
        json: bool,
        #[arg(long, default_value = "")]
        coordinator: String,
        #[arg(long, default_value = "")]
        key: String,
    },
    /// Submit a workload: submit [PARAM=value ...]
    Submit {
        preset: String,
        params: Vec<String>,
        #[arg(long)]
        replicate: bool,
        #[arg(long, default_value_t = 0)]
        priority: i32,
        #[arg(long, default_value = "http://127.0.0.1:7900")]
        coordinator: String,
        #[arg(long, default_value = "")]
        key: String,
        /// Input file for wordcount / llm_batch
        #[arg(long)]
        input: Option<PathBuf>,
        /// Write the finished artifact to this file
        #[arg(long, short = 'o')]
        out: Option<PathBuf>,
        #[arg(long)]
        no_wait: bool,
        #[arg(long, default_value_t = 600)]
        timeout: u64,
    },
    /// Fleet + job overview
    Status {
        #[arg(long, default_value = "http://127.0.0.1:7900")]
        coordinator: String,
        #[arg(long, default_value = "")]
        key: String,
        #[arg(long)]
        json: bool,
    },
    /// List fleet nodes
    Nodes {
        #[arg(long, default_value = "http://127.0.0.1:7900")]
        coordinator: String,
        #[arg(long, default_value = "")]
        key: String,
        #[arg(long)]
        json: bool,
    },
    /// List jobs
    Jobs {
        #[arg(long, default_value = "http://127.0.0.1:7900")]
        coordinator: String,
        #[arg(long, default_value = "")]
        key: String,
        #[arg(long)]
        json: bool,
    },
    /// Cancel a job
    Cancel {
        job_id: String,
        #[arg(long, default_value = "http://127.0.0.1:7900")]
        coordinator: String,
        #[arg(long, default_value = "")]
        key: String,
    },
}

fn client() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(30))
        .build()
}

fn get_json(base: &str, path: &str, key: &str) -> anyhow::Result<serde_json::Value> {
    let mut req = client().get(&format!("{base}{path}"));
    if !key.is_empty() {
        req = req.set("X-Compost-Key", key);
    }
    Ok(req.call()?.into_json()?)
}

fn post_json(
    base: &str,
    path: &str,
    key: &str,
    body: serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    let mut req = client()
        .post(&format!("{base}{path}"))
        .set("Content-Type", "application/json");
    if !key.is_empty() {
        req = req.set("X-Compost-Key", key);
    }
    Ok(req.send_json(body)?.into_json()?)
}

pub fn run(cmd: ClusterCommand) -> anyhow::Result<()> {
    match cmd {
        ClusterCommand::Coordinator {
            addr,
            key,
            protect_artifacts,
            state,
            allow_same_ip_replicate,
        } => {
            let handle = server::start_coordinator(server::CoordinatorOptions {
                addr: addr.clone(),
                key,
                protect_artifacts,
                allow_same_ip_replicate,
                state_path: state,
            })?;
            println!("Vibe Grid coordinator on {addr} (port {})", handle.port);
            println!(
                "Dashboard: http://localhost:{}/  ·  API: /api/v1/*  ·  metrics: /metrics",
                handle.port
            );
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        }
        ClusterCommand::Agent {
            coordinator,
            key,
            workers,
            name,
            node_id_file,
            battery_guard_off,
        } => {
            let handle = agent::start_agent(agent::AgentConfig {
                coordinator: coordinator.clone(),
                name,
                key,
                node_id_file,
                battery_guard_off,
                workers,
                ..Default::default()
            })?;
            println!(
                "Vibe Grid agent joined {coordinator} as {}",
                handle.node_id()
            );
            println!("Workers: {workers} · heartbeat 5s · battery/thermal guards active");
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        }
        ClusterCommand::PlanMoe {
            model_id,
            context,
            json,
            coordinator,
            key,
        } => {
            let base = if coordinator.is_empty() {
                crate::grid_proxy::coordinator_base(None)
            } else {
                coordinator.trim_end_matches('/').to_string()
            };
            let plan = crate::grid_proxy::plan_moe_live(
                &model_id,
                context,
                &base,
                if key.is_empty() { None } else { Some(&key) },
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan)?);
            } else {
                print_plan(&plan);
            }
            Ok(())
        }
        ClusterCommand::Submit {
            preset,
            params,
            replicate,
            priority,
            coordinator,
            key,
            input,
            out,
            no_wait,
            timeout,
        } => {
            let base = coordinator.trim_end_matches('/').to_string();
            let mut numeric = BTreeMap::new();
            let mut meta = BTreeMap::new();
            for p in &params {
                let (k, v) = p
                    .split_once('=')
                    .ok_or_else(|| anyhow::anyhow!("param must be KEY=value, got {p:?}"))?;
                if matches!(
                    k,
                    "mode" | "model" | "format" | "model_id" | "quant" | "gguf_path" | "prompt"
                ) {
                    meta.insert(k.to_string(), v.to_string());
                } else {
                    let f: f64 = v
                        .parse()
                        .map_err(|_| anyhow::anyhow!("param {k} must be numeric, got {v:?}"))?;
                    numeric.insert(k.to_string(), f);
                }
            }
            let mut body = json!({"kind": preset, "params": numeric, "meta": meta, "replicate": replicate, "priority": priority});
            if let Some(path) = &input {
                let bytes = std::fs::read(path)?;
                use base64::Engine;
                body["input"] = json!(base64::engine::general_purpose::STANDARD.encode(bytes));
            }
            let resp = post_json(&base, "/api/v1/jobs", &key, body)?;
            let job = &resp["job"];
            let job_id = job["id"].as_str().unwrap_or("?").to_string();
            println!(
                "submitted {job_id}: {} ({} tasks){}",
                job["kind"].as_str().unwrap_or("?"),
                job["tasks_total"],
                if replicate { " [replicated]" } else { "" }
            );
            if no_wait {
                return Ok(());
            }
            let deadline = Instant::now() + Duration::from_secs(timeout);
            loop {
                anyhow::ensure!(Instant::now() < deadline, "timed out waiting for {job_id}");
                let st = get_json(&base, "/api/v1/state", &key)?;
                let j = st["jobs"]
                    .as_array()
                    .and_then(|js| js.iter().find(|j| j["id"] == job_id));
                if let Some(j) = j {
                    let status = j["status"].as_str().unwrap_or("");
                    let done = j["tasks_done"].as_u64().unwrap_or(0);
                    let total = j["tasks_total"].as_u64().unwrap_or(0);
                    eprint!("\r{job_id}: {status} {done}/{total}   ");
                    match status {
                        JOB_DONE => {
                            eprintln!();
                            let mut req =
                                client().get(&format!("{base}/api/v1/jobs/{job_id}/artifact"));
                            if !key.is_empty() {
                                req = req.set("X-Compost-Key", &key);
                            }
                            let resp = req.call()?;
                            let ctype = resp.header("Content-Type").unwrap_or("").to_string();
                            let mut buf = Vec::new();
                            resp.into_reader().read_to_end(&mut buf)?;
                            match &out {
                                Some(path) => {
                                    std::fs::write(path, &buf)?;
                                    println!(
                                        "artifact saved to {} ({} bytes, {ctype})",
                                        path.display(),
                                        buf.len()
                                    );
                                }
                                None if ctype.starts_with("text/")
                                    || ctype.starts_with("application/json") =>
                                {
                                    println!("{}", String::from_utf8_lossy(&buf));
                                }
                                None => {
                                    println!(
                                        "artifact: {} bytes ({ctype}) — use -o FILE to save",
                                        buf.len()
                                    );
                                }
                            }
                            return Ok(());
                        }
                        JOB_FAILED => {
                            eprintln!();
                            anyhow::bail!(
                                "job failed: {}",
                                j["error"].as_str().unwrap_or("unknown")
                            );
                        }
                        JOB_CANCELLED => {
                            eprintln!();
                            anyhow::bail!("job cancelled");
                        }
                        _ => {}
                    }
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }
        ClusterCommand::Status {
            coordinator,
            key,
            json,
        } => {
            let base = coordinator.trim_end_matches('/').to_string();
            let health = get_json(&base, "/healthz", &key)?;
            let st = get_json(&base, "/api/v1/state", &key)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({"health": health, "state": st}))?
                );
                return Ok(());
            }
            println!(
                "Coordinator: {} v{} (uptime {}s)",
                base,
                health["version"].as_str().unwrap_or("?"),
                health["uptime_s"]
            );
            println!(
                "Nodes: {} online / {} total · with LLM: {} · active jobs: {}",
                health["nodes_online"],
                health["nodes_total"],
                health["nodes_with_llm"],
                health["jobs_active"]
            );
            print_nodes(&st);
            print_jobs(&st);
            Ok(())
        }
        ClusterCommand::Nodes {
            coordinator,
            key,
            json,
        } => {
            let st = get_json(coordinator.trim_end_matches('/'), "/api/v1/state", &key)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&st["nodes"])?);
            } else {
                print_nodes(&st);
            }
            Ok(())
        }
        ClusterCommand::Jobs {
            coordinator,
            key,
            json,
        } => {
            let st = get_json(coordinator.trim_end_matches('/'), "/api/v1/state", &key)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&st["jobs"])?);
            } else {
                print_jobs(&st);
            }
            Ok(())
        }
        ClusterCommand::Cancel {
            job_id,
            coordinator,
            key,
        } => {
            let resp = post_json(
                coordinator.trim_end_matches('/'),
                &format!("/api/v1/jobs/{job_id}/cancel"),
                &key,
                json!({}),
            )?;
            println!("{resp}");
            Ok(())
        }
    }
}

fn print_nodes(st: &serde_json::Value) {
    let nodes = st["nodes"].as_array().cloned().unwrap_or_default();
    if nodes.is_empty() {
        println!("No nodes have joined yet.");
        return;
    }
    println!(
        "{:<6} {:<20} {:<14} {:>6} {:>9} {:>7} {:>6} {:>6}  STATUS",
        "ID", "NAME", "OS/ARCH", "CORES", "RAM MB", "SPEED", "DONE", "FAIL"
    );
    for n in nodes {
        let caps = &n["caps"];
        let status = if !n["online"].as_bool().unwrap_or(false) {
            "offline"
        } else if !n["paused"].as_str().unwrap_or("").is_empty() {
            n["paused"].as_str().unwrap()
        } else {
            "ready"
        };
        println!(
            "{:<6} {:<20} {:<14} {:>6} {:>9} {:>7.2} {:>6} {:>6}  {}",
            n["id"].as_str().unwrap_or("?"),
            truncate(n["name"].as_str().unwrap_or("?"), 20),
            format!(
                "{}/{}",
                caps["os"].as_str().unwrap_or("?"),
                caps["arch"].as_str().unwrap_or("?")
            ),
            caps["cores"],
            caps["mem_mb"],
            n["speed_factor"].as_f64().unwrap_or(0.0),
            n["done"],
            n["failed"],
            status
        );
    }
}

fn print_jobs(st: &serde_json::Value) {
    let jobs = st["jobs"].as_array().cloned().unwrap_or_default();
    if jobs.is_empty() {
        println!("No jobs.");
        return;
    }
    println!(
        "{:<10} {:<16} {:<10} {:>9} {:>7}  WAITING/ERROR",
        "ID", "KIND", "STATUS", "TASKS", "ETA s"
    );
    for j in jobs {
        let note = if !j["error"].as_str().unwrap_or("").is_empty() {
            j["error"].as_str().unwrap()
        } else {
            j["waiting"].as_str().unwrap_or("")
        };
        let eta = j["eta_seconds"].as_i64().unwrap_or(0);
        println!(
            "{:<10} {:<16} {:<10} {:>4}/{:<4} {:>7}  {}",
            j["id"].as_str().unwrap_or("?"),
            j["kind"].as_str().unwrap_or("?"),
            j["status"].as_str().unwrap_or("?"),
            j["tasks_done"],
            j["tasks_total"],
            if eta > 0 {
                format!("{eta}s")
            } else {
                "—".to_string()
            },
            truncate(note, 48)
        );
    }
}

fn print_plan(plan: &serde_json::Value) {
    println!(
        "MoE fleet plan: {} ({}, {})",
        plan["model_id"].as_str().unwrap_or("?"),
        plan["architecture"].as_str().unwrap_or("?"),
        plan["quant"].as_str().unwrap_or("?")
    );
    println!(
        "Experts: {} routed, top-{} per token · context {} · backbone on {}",
        plan["routed_experts"],
        plan["experts_per_token"],
        plan["context"],
        plan["backbone_node"].as_str().unwrap_or("?")
    );
    println!(
        "Est. decode: {:.1} tok/s ({:.2}x vs single node) · Wi-Fi RTT {}ms",
        plan["est_decode_tps"].as_f64().unwrap_or(0.0),
        plan["est_speedup_vs_single_node"].as_f64().unwrap_or(0.0),
        plan["wifi_rtt_ms"]
    );
    println!(
        "{:<12} {:<20} {:>13} {:>9}  {:<18} STATUS",
        "SHARD", "NODE", "EXPERTS", "SIZE MB", "ROLE"
    );
    if let Some(shards) = plan["shards"].as_array() {
        for s in shards {
            println!(
                "{:<12} {:<20} {:>5}..{:<5} {:>9.1}  {:<18} {}",
                s["shard_id"].as_str().unwrap_or("?"),
                truncate(s["node_name"].as_str().unwrap_or("?"), 20),
                s["expert_start"],
                s["expert_end"],
                s["size_mb"].as_f64().unwrap_or(0.0),
                s["role"].as_str().unwrap_or("?"),
                s["status"].as_str().unwrap_or("?")
            );
        }
    }
    if let Some(drafts) = plan["draft_nodes"].as_array()
        && !drafts.is_empty()
    {
        println!(
            "Speculative draft nodes ({} draft tokens): {}",
            plan["draft_tokens"],
            drafts
                .iter()
                .map(|d| d.as_str().unwrap_or("?"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if let Some(warnings) = plan["warnings"].as_array() {
        for w in warnings {
            println!("warning: {}", w.as_str().unwrap_or(""));
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let t: String = s.chars().take(n - 1).collect();
        format!("{t}…")
    }
}

use std::io::Read;
