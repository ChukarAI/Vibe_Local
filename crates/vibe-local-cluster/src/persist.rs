//! Crash-safe coordinator snapshots: gzipped JSON, version 4 semantics
//! (jobs/tasks/payloads/reclaimed sizes/artifacts/NodeUUID speed profiles).
//! The HTTP wire protocol is compost v0.7.0-compatible; the on-disk format
//! is Vibe Local's JSON+gzip equivalent of compost's gob snapshots (v3-era
//! fields load with zero-value defaults).

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::proto::{Job, Task};

pub const SNAPSHOT_VERSION: u32 = 4;
const MIN_SUPPORTED: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetSpeedSample {
    pub factor: f64,
    pub n: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedSpeed {
    #[serde(default)]
    pub node_uuid: String,
    #[serde(default)]
    pub name: String,
    pub factor: f64,
    pub n: u32,
    #[serde(default)]
    pub preset_speeds: BTreeMap<String, PresetSpeedSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
    pub task: Task,
    pub state: u8,
    pub attempts: u32,
    #[serde(default)]
    pub epoch: u64,
    #[serde(default)]
    pub pending_from: String,
    #[serde(default)]
    pub pending_from_uuid: String,
    #[serde(default)]
    pub pending_from_ip: String,
    #[serde(default)]
    pub replica_payload: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub job: Job,
    pub order: Vec<String>,
    #[serde(default)]
    pub payloads: BTreeMap<String, Vec<u8>>,
    #[serde(default)]
    pub payload_sizes: BTreeMap<String, usize>,
    #[serde(default)]
    pub reclaimed: bool,
    #[serde(default)]
    pub artifact: Option<Vec<u8>>,
    #[serde(default)]
    pub artifact_type: String,
    #[serde(default)]
    pub artifact_ext: String,
    pub tasks: Vec<TaskRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub version: u32,
    pub job_seq: u64,
    #[serde(default)]
    pub id_prefix: String,
    #[serde(default)]
    pub learned_speeds: Vec<LearnedSpeed>,
    pub jobs: Vec<JobRecord>,
}

pub fn save(path: &Path, snap: &Snapshot) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_vec(snap)?;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&json)?;
    let compressed = encoder.finish()?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &compressed)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn load(path: &Path) -> anyhow::Result<Option<Snapshot>> {
    if !path.is_file() {
        return Ok(None);
    }
    let raw = std::fs::read(path)?;
    let mut decoder = flate2::read::GzDecoder::new(&raw[..]);
    let mut json = Vec::new();
    decoder.read_to_end(&mut json)?;
    let snap: Snapshot = serde_json::from_slice(&json)?;
    if snap.version < MIN_SUPPORTED || snap.version > SNAPSHOT_VERSION {
        anyhow::bail!(
            "snapshot version {} unsupported (need {}..={})",
            snap.version,
            MIN_SUPPORTED,
            SNAPSHOT_VERSION
        );
    }
    Ok(Some(snap))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.json.gz");
        let snap = Snapshot {
            version: SNAPSHOT_VERSION,
            job_seq: 7,
            id_prefix: "job".into(),
            learned_speeds: vec![LearnedSpeed {
                node_uuid: "uuid-1".into(),
                name: "pc".into(),
                factor: 1.5,
                n: 4,
                preset_speeds: BTreeMap::from([(
                    "primes".to_string(),
                    PresetSpeedSample { factor: 2.0, n: 3 },
                )]),
            }],
            jobs: vec![JobRecord {
                job: Job {
                    id: "job-1".into(),
                    kind: "primes".into(),
                    status: "done".into(),
                    ..Default::default()
                },
                order: vec!["t1".into()],
                payloads: BTreeMap::from([("t1".to_string(), b"42".to_vec())]),
                payload_sizes: BTreeMap::from([("t1".to_string(), 2usize)]),
                reclaimed: false,
                artifact: Some(b"primes below 100: 25\n".to_vec()),
                artifact_type: "text/plain".into(),
                artifact_ext: ".txt".into(),
                tasks: vec![TaskRecord {
                    task: Task {
                        id: "t1".into(),
                        job_id: "job-1".into(),
                        kind: "primes".into(),
                        ..Default::default()
                    },
                    state: 2,
                    attempts: 1,
                    epoch: 1,
                    pending_from: String::new(),
                    pending_from_uuid: String::new(),
                    pending_from_ip: String::new(),
                    replica_payload: None,
                }],
            }],
        };
        save(&path, &snap).unwrap();
        let loaded = load(&path).unwrap().unwrap();
        assert_eq!(loaded.job_seq, 7);
        assert_eq!(loaded.jobs[0].payloads["t1"], b"42");
        assert_eq!(loaded.learned_speeds[0].factor, 1.5);
        assert!(load(&dir.path().join("missing.gz")).unwrap().is_none());
    }

    #[test]
    fn v3_defaults_load() {
        // v3-era records lack epoch/uuid/artifact fields — zero defaults.
        let json = r#"{"version":3,"job_seq":2,"jobs":[{"job":{"id":"j","kind":"primes","status":"queued","created_at":"","width":0,"height":0,"max_iter":0,"cols":0,"rows":0,"tile_w":0,"tile_h":0,"tasks_total":1,"tasks_done":0},"order":["t"],"tasks":[{"task":{"id":"t","job_id":"j","kind":"primes"},"state":0,"attempts":0}]}]}"#;
        let snap: Snapshot = serde_json::from_str(json).unwrap();
        assert_eq!(snap.version, 3);
        assert_eq!(snap.jobs[0].tasks[0].epoch, 0);
        assert!(snap.jobs[0].artifact.is_none());
    }
}
