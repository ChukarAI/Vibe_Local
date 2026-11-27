//! Pluggable workload presets: validate → split → execute → aggregate.
//! Adding a workload never touches the scheduler or server.

pub mod embed_batch;
pub mod llm_batch;
pub mod mandelbrot;
pub mod moe;
pub mod primes;
pub mod vision_batch;
pub mod wordcount;

use std::collections::BTreeMap;

use crate::proto::{Job, Task, WorkloadInfo};

pub const ENC_RGBA: &str = "rgba";
pub const ENC_TEXT: &str = "text";
pub const ENC_JSONL: &str = "jsonl";

pub type AggregateOut = (Vec<u8>, String, String); // bytes, content-type, ext

pub trait Preset: Send + Sync {
    fn name(&self) -> &'static str;
    fn describe(&self) -> &'static str;
    fn validate(&self, params: &BTreeMap<String, f64>) -> anyhow::Result<Job>;
    fn split(&self, job: &Job) -> anyhow::Result<Vec<Task>>;
    /// Deterministic, side-effect-free local execution (idempotency).
    fn execute(&self, t: &Task) -> anyhow::Result<(Vec<u8>, &'static str)>;
    fn aggregate(
        &self,
        job: &Job,
        order: &[String],
        results: &BTreeMap<String, Vec<u8>>,
    ) -> anyhow::Result<AggregateOut>;
    /// Expected payload length; 0 = no cheap check.
    fn result_size(&self, t: &Task) -> usize;
    fn verify_replica(&self, primary: &[u8], replica: &[u8]) -> bool;
}

pub struct Registry {
    presets: Vec<Box<dyn Preset>>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Registry {
        Registry {
            presets: Vec::new(),
        }
    }

    /// The full Vibe Grid registry: four compost classics + distributed MoE
    /// + heterogeneous capability workloads (`embed_batch`, `vision_batch`).
    pub fn with_defaults() -> Registry {
        let mut r = Registry::new();
        r.register(Box::new(mandelbrot::Mandelbrot));
        r.register(Box::new(primes::Primes));
        r.register(Box::new(wordcount::Wordcount));
        r.register(Box::new(llm_batch::LlmBatch));
        r.register(Box::new(embed_batch::EmbedBatch));
        r.register(Box::new(vision_batch::VisionBatch));
        r.register(Box::new(moe::MoeDistributed));
        r
    }

    pub fn register(&mut self, p: Box<dyn Preset>) {
        assert!(
            !self.presets.iter().any(|x| x.name() == p.name()),
            "duplicate preset {}",
            p.name()
        );
        self.presets.push(p);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Preset> {
        self.presets
            .iter()
            .find(|p| p.name() == name)
            .map(|p| p.as_ref())
    }

    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<String> = self.presets.iter().map(|p| p.name().to_string()).collect();
        n.sort();
        n
    }

    pub fn info(&self) -> Vec<WorkloadInfo> {
        let mut info: Vec<WorkloadInfo> = self
            .presets
            .iter()
            .map(|p| WorkloadInfo {
                name: p.name().to_string(),
                description: p.describe().to_string(),
            })
            .collect();
        info.sort_by(|a, b| a.name.cmp(&b.name));
        info
    }
}

pub(crate) fn get_param(params: &BTreeMap<String, f64>, key: &str, default: f64) -> f64 {
    params.get(key).copied().unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_defaults() {
        let r = Registry::with_defaults();
        assert_eq!(
            r.names(),
            vec![
                "embed_batch",
                "llm_batch",
                "mandelbrot",
                "moe_distributed",
                "primes",
                "vision_batch",
                "wordcount"
            ]
        );
        assert!(r.get("primes").is_some());
        assert!(r.get("nope").is_none());
        assert_eq!(r.info().len(), 7);
    }
}
