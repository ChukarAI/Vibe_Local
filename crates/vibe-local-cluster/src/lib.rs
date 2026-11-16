//! # vibe-local-cluster — Vibe Grid
//!
//! Pull-based distributed compute for the Vibe Local ecosystem: a Rust port
//! of compost v0.7.0 (wire-compatible `/api/v1/*`) plus distributed MoE
//! Wi-Fi expert sharding across phones + PC.
//!
//! - [`scheduler`]: fair + priority batch leasing, LPT ordering, adaptive
//!   leases, LeaseEpoch heartbeats, Sybil-safe replication, reclamation.
//! - [`server`]: coordinator HTTP API, artifact LRU + disk offload, gzip-bomb
//!   guard, metrics, ETA, persistence, embedded dashboard.
//! - [`agent`]: outward-only worker loop with battery/thermal guards.
//! - [`workload`]: mandelbrot, primes, wordcount, llm_batch, moe_distributed.
//! - [`moe_grid`]: fleet expert partitioning, shard routing, drop recovery.
//!
//! Licensed under Apache-2.0.

pub mod agent;
pub mod auth;
pub mod metrics;
pub mod moe_grid;
pub mod node;
pub mod persist;
pub mod proto;
pub mod scheduler;
pub mod server;
pub mod workload;

pub use agent::{AgentConfig, AgentHandle, start_agent};
pub use server::{CoordinatorHandle, CoordinatorOptions, start_coordinator};
