//! Rescheduling work queue — a faithful Rust port of compost v0.7.0's
//! scheduler: capacity-aware batch leasing with per-job fairness + priority,
//! LPT biggest-first ordering, adaptive leases scaled by learned per-preset
//! speed, monotonic LeaseEpoch tokens with active heartbeat renewal,
//! Sybil-safe replication, poison-task caps and post-assembly reclamation.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use crate::persist::{JobRecord, LearnedSpeed, PresetSpeedSample, Snapshot, TaskRecord};
use crate::proto::{
    Caps, JOB_CANCELLED, JOB_DONE, JOB_FAILED, JOB_QUEUED, JOB_RUNNING, Job, Node, ResultMsg,
    StateResponse, TASK_LLM_BATCH, Task, TaskLeaseRef,
};
use vibe_local_core::util::iso8601_from_unix;

pub const LEASE_BASE: Duration = Duration::from_secs(15);
pub const LEASE_SLACK: u64 = 4;
pub const LEASE_CAP: Duration = Duration::from_secs(30 * 60);
pub const OFFLINE_AFTER: Duration = Duration::from_secs(20);
pub const FORGET_AFTER: Duration = Duration::from_secs(600);
pub const MAX_ATTEMPTS: u32 = 5;
const RATE_ALPHA: f64 = 0.3;
pub const SPEED_PRIOR_N: f64 = 3.0;
pub const SPEED_ALPHA_EWMA: f64 = 0.3;

pub type LeaseFn = Box<dyn Fn(&str, &Task) -> Duration + Send + Sync>;
pub type ReplicaVerifier = Box<dyn Fn(&str, &[u8], &[u8]) -> bool + Send + Sync>;
/// Everything a preset's Aggregate needs: job, task order, payloads by ID.
pub type AssembleInputs = (Job, Vec<String>, BTreeMap<String, Vec<u8>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    Queued = 0,
    Leased = 1,
    Done = 2,
}

#[derive(Clone)]
struct QueuedTask {
    task: Task,
    state: TaskState,
    attempts: u32,
    node_id: String,
    epoch: u64,
    leased_at: Option<Instant>,
    deadline: Option<Instant>,
    pending_from: String,
    pending_from_uuid: String,
    pending_from_ip: String,
    replica_payload: Option<Vec<u8>>,
}

struct JobState {
    j: Job,
    order: Vec<String>,
    payloads: BTreeMap<String, Vec<u8>>,
    payload_sizes: BTreeMap<String, usize>,
    reclaimed: bool,
    done: bool,
    failed: bool,
    replicate: bool,
}

impl JobState {
    fn open(&self) -> bool {
        !self.failed
            && self.j.status != JOB_FAILED
            && self.j.status != JOB_CANCELLED
            && self.j.status != JOB_DONE
    }
}

#[derive(Clone, Copy, Default)]
struct SpeedSample {
    factor: f64,
    n: u32,
}

fn fold_speed_sample(cur: f64, n: u32, raw: f64) -> (f64, u32) {
    if n == 0 {
        ((raw + SPEED_PRIOR_N * 1.0) / (1.0 + SPEED_PRIOR_N), 1)
    } else {
        (
            cur * (1.0 - SPEED_ALPHA_EWMA) + raw * SPEED_ALPHA_EWMA,
            n + 1,
        )
    }
}

#[derive(Clone)]
struct NodeStat {
    id: String,
    uuid: String,
    remote_ip: String,
    name: String,
    caps: Caps,
    joined: SystemTime,
    seen: Instant,
    rate: f64,
    runs: u64,
    running: usize,
    done: u64,
    failed: u64,
    paused: String,
    speed_factor: f64,
    speed_n: u32,
    preset_speed: BTreeMap<String, SpeedSample>,
}

impl NodeStat {
    fn observe_speed(&mut self, est_millis: i64, actual_millis: i64) {
        if est_millis <= 0 || actual_millis <= 0 {
            return;
        }
        let raw = est_millis as f64 / actual_millis as f64;
        let (f, n) = fold_speed_sample(self.speed_factor, self.speed_n, raw);
        self.speed_factor = f;
        self.speed_n = n;
    }

    fn observe_preset_speed(&mut self, kind: &str, est_millis: i64, actual_millis: i64) {
        if est_millis <= 0 || actual_millis <= 0 {
            return;
        }
        self.observe_speed(est_millis, actual_millis);
        if kind.is_empty() {
            return;
        }
        let cur = self.preset_speed.get(kind).copied().unwrap_or_default();
        let (f, n) = fold_speed_sample(cur.factor, cur.n, est_millis as f64 / actual_millis as f64);
        self.preset_speed
            .insert(kind.to_string(), SpeedSample { factor: f, n });
    }

    /// Battery safety is a feature, not a suggestion: old lithium cells +
    /// sustained load is how fires start. Android battery thermal is stricter.
    fn guard_reason(&self) -> String {
        if self.caps.battery_pct >= 0 && self.caps.on_battery && self.caps.battery_pct < 30 {
            return format!("battery {}% (<30%, on battery)", self.caps.battery_pct);
        }
        if self.caps.temp_c > 80.0 {
            return format!("thermal {:.0}°C (>80°C)", self.caps.temp_c);
        }
        if self.caps.os == "android" && self.caps.temp_c > 45.0 {
            return format!("battery thermal {:.0}°C (>45°C Android)", self.caps.temp_c);
        }
        String::new()
    }
}

/// SpeedProfile is a persisted learning record keyed by NodeUUID.
#[derive(Debug, Clone)]
pub struct SpeedProfile {
    pub node_uuid: String,
    pub name: String,
    pub factor: f64,
    pub n: u32,
    pub preset_speeds: BTreeMap<String, PresetSpeedSample>,
}

fn profile_key(uuid: &str, name: &str) -> String {
    if !uuid.is_empty() {
        format!("uuid:{uuid}")
    } else {
        format!("name:{name}")
    }
}

struct Inner {
    tasks: Vec<QueuedTask>,
    tasks_by_id: BTreeMap<String, usize>,
    jobs: BTreeMap<String, JobState>,
    nodes: BTreeMap<String, NodeStat>,
    seq: u64,
    job_seq: u64,
    id_prefix: String,
    allow_same_ip_replicate: bool,
    learn_by_key: BTreeMap<String, SpeedProfile>,
    requeued_total: u64,
    unsupported_total: u64,
}

pub struct Queue {
    inner: Mutex<Inner>,
    /// None = adaptive default leases; Some = test/custom override.
    lease_fn: Mutex<Option<LeaseFn>>,
    verify_fn: Mutex<ReplicaVerifier>,
}

impl Default for Queue {
    fn default() -> Self {
        Self::new()
    }
}

impl Queue {
    pub fn new() -> Queue {
        Queue {
            inner: Mutex::new(Inner {
                tasks: Vec::new(),
                tasks_by_id: BTreeMap::new(),
                jobs: BTreeMap::new(),
                nodes: BTreeMap::new(),
                seq: 0,
                job_seq: 0,
                id_prefix: "job".into(),
                allow_same_ip_replicate: false,
                learn_by_key: BTreeMap::new(),
                requeued_total: 0,
                unsupported_total: 0,
            }),
            lease_fn: Mutex::new(None),
            verify_fn: Mutex::new(Box::new(|_, a, b| a == b)),
        }
    }

    pub fn set_allow_same_ip_replicate(&self, allow: bool) {
        self.inner.lock().unwrap().allow_same_ip_replicate = allow;
    }

    pub fn set_replica_verifier(&self, f: ReplicaVerifier) {
        *self.verify_fn.lock().unwrap() = f;
    }

    pub fn set_lease_fn(&self, f: LeaseFn) {
        *self.lease_fn.lock().unwrap() = Some(f);
    }

    fn speed_of(inner: &Inner, node_id: &str, kind: &str) -> f64 {
        if let Some(n) = inner.nodes.get(node_id) {
            if let Some(ps) = n.preset_speed.get(kind)
                && ps.n > 0
                && ps.factor > 0.0
            {
                return ps.factor;
            }
            if n.speed_factor > 0.0 {
                return n.speed_factor;
            }
        }
        1.0
    }

    fn lease_duration(&self, inner: &Inner, node_id: &str, t: &Task) -> Duration {
        if let Some(f) = self.lease_fn.lock().unwrap().as_ref() {
            return f(node_id, t);
        }
        let speed = Self::speed_of(inner, node_id, &t.kind);
        let est = Duration::from_millis(t.est_millis.max(0) as u64);
        let d = LEASE_BASE + est.mul_f64(LEASE_SLACK as f64 / speed);
        d.min(LEASE_CAP)
    }

    pub fn requeued_total(&self) -> u64 {
        self.inner.lock().unwrap().requeued_total
    }

    pub fn unsupported_total(&self) -> u64 {
        self.inner.lock().unwrap().unsupported_total
    }

    // --- nodes ---

    pub fn join(&self, name: &str, caps: Caps) -> String {
        self.join_with_meta(name, "", "", caps)
    }

    pub fn join_with_meta(&self, name: &str, uuid: &str, remote_ip: &str, caps: Caps) -> String {
        let mut inner = self.inner.lock().unwrap();
        let now = Instant::now();
        for n in inner.nodes.values() {
            if now.duration_since(n.seen) >= OFFLINE_AFTER {
                continue;
            }
            if !uuid.is_empty() && n.uuid == uuid {
                eprintln!(
                    "[join] WARNING: {name} (uuid={uuid}) joined while another session with same UUID is online"
                );
            } else if uuid.is_empty() && n.name == name {
                eprintln!(
                    "[join] WARNING: {name} joined while another node of the same name is online"
                );
            }
        }
        inner.seq += 1;
        let id = format!("n{}", inner.seq);
        let mut n = NodeStat {
            id: id.clone(),
            uuid: uuid.to_string(),
            remote_ip: remote_ip.to_string(),
            name: name.to_string(),
            caps,
            joined: SystemTime::now(),
            seen: now,
            rate: 0.0,
            runs: 0,
            running: 0,
            done: 0,
            failed: 0,
            paused: String::new(),
            speed_factor: 0.0,
            speed_n: 0,
            preset_speed: BTreeMap::new(),
        };
        // Resume learned profile by NodeUUID first, then legacy name.
        let keys = if !uuid.is_empty() {
            vec![profile_key(uuid, ""), profile_key("", name)]
        } else {
            vec![profile_key("", name)]
        };
        for k in keys {
            if let Some(p) = inner.learn_by_key.get(&k).cloned()
                && (p.factor > 0.0 || !p.preset_speeds.is_empty())
            {
                n.speed_factor = p.factor;
                n.speed_n = p.n;
                for (kind, s) in &p.preset_speeds {
                    n.preset_speed.insert(
                        kind.clone(),
                        SpeedSample {
                            factor: s.factor,
                            n: s.n,
                        },
                    );
                }
                inner.learn_by_key.remove(&k);
                eprintln!(
                    "[join] {name} resumes learned speed ×{:.2} ({} samples)",
                    p.factor, p.n
                );
                break;
            }
        }
        eprintln!(
            "[join] {name} ({}/{}) id={id} uuid={uuid} score={:.1} cores={} batt={}% llm={}",
            n.caps.os,
            n.caps.arch,
            n.caps.score,
            n.caps.cores,
            n.caps.battery_pct,
            n.caps.has_local_llm
        );
        inner.nodes.insert(id.clone(), n);
        id
    }

    pub fn touch_with_meta(
        &self,
        id: &str,
        name: &str,
        uuid: &str,
        remote_ip: &str,
        caps: Caps,
    ) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(n) = inner.nodes.get_mut(id) else {
            return false;
        };
        n.name = name.to_string();
        n.caps = caps;
        n.seen = Instant::now();
        if !uuid.is_empty() && n.uuid.is_empty() {
            n.uuid = uuid.to_string();
        }
        if !remote_ip.is_empty() {
            n.remote_ip = remote_ip.to_string();
        }
        true
    }

    pub fn note_running(&self, id: &str, running: usize) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(n) = inner.nodes.get_mut(id) {
            n.running = running;
        }
    }

    pub fn node_exists(&self, id: &str) -> bool {
        self.inner.lock().unwrap().nodes.contains_key(id)
    }

    // --- jobs ---

    pub fn next_job_id(&self) -> String {
        let mut inner = self.inner.lock().unwrap();
        inner.job_seq += 1;
        format!("{}-{}", inner.id_prefix, inner.job_seq)
    }

    pub fn add_job(&self, j: Job, tasks: Vec<Task>) -> Job {
        let mut inner = self.inner.lock().unwrap();
        let mut js = JobState {
            replicate: j.replicate,
            j,
            order: Vec::new(),
            payloads: BTreeMap::new(),
            payload_sizes: BTreeMap::new(),
            reclaimed: false,
            done: false,
            failed: false,
        };
        for t in tasks {
            let idx = inner.tasks.len();
            inner.tasks.push(QueuedTask {
                task: t.clone(),
                state: TaskState::Queued,
                attempts: 0,
                node_id: String::new(),
                epoch: 0,
                leased_at: None,
                deadline: None,
                pending_from: String::new(),
                pending_from_uuid: String::new(),
                pending_from_ip: String::new(),
                replica_payload: None,
            });
            inner.tasks_by_id.insert(t.id.clone(), idx);
            js.order.push(t.id);
        }
        js.j.tasks_total = js.order.len();
        let out = js.j.clone();
        eprintln!(
            "[job] created {}: {} ({} tasks){}",
            out.id,
            out.kind,
            out.tasks_total,
            if js.replicate { " [replicated]" } else { "" }
        );
        inner.jobs.insert(out.id.clone(), js);
        out
    }

    fn can_execute(n: &NodeStat, t: &Task) -> bool {
        if t.kind == TASK_LLM_BATCH && !t.verify_mode {
            return n.caps.has_local_llm;
        }
        true
    }

    fn can_confirm_replica(inner: &Inner, js: &JobState, qt: &QueuedTask, n: &NodeStat) -> bool {
        if !js.replicate || qt.pending_from.is_empty() {
            return true;
        }
        if n.id == qt.pending_from {
            return false;
        }
        // Sybil guard 1: same persistent NodeUUID (even after agent restart).
        if !n.uuid.is_empty() && !qt.pending_from_uuid.is_empty() && n.uuid == qt.pending_from_uuid
        {
            return false;
        }
        // Sybil guard 2: same source IP unless explicitly permitted.
        if !inner.allow_same_ip_replicate
            && !n.remote_ip.is_empty()
            && !qt.pending_from_ip.is_empty()
            && n.remote_ip == qt.pending_from_ip
        {
            return false;
        }
        true
    }

    // --- leasing ---

    pub fn lease(&self, node_id: &str, paused: &str) -> Option<Task> {
        let batch = self.lease_batch(node_id, paused, 0, 1);
        batch.into_iter().next()
    }

    pub fn lease_batch(
        &self,
        node_id: &str,
        paused: &str,
        running: usize,
        workers: usize,
    ) -> Vec<Task> {
        let workers = workers.max(1);
        let want = (workers.saturating_sub(running)).max(1);
        let mut inner = self.inner.lock().unwrap();
        let Some(n) = inner.nodes.get_mut(node_id) else {
            return Vec::new();
        };
        n.paused = paused.to_string();
        let r = n.guard_reason();
        if !r.is_empty() {
            n.paused = r;
        }
        if !n.paused.is_empty() {
            return Vec::new();
        }
        let now = Instant::now();
        let mut out: Vec<usize> = Vec::new();
        let mut pass = 1u32;
        while out.len() < want {
            let mut progressed = false;
            let mut taken_this_pass: BTreeMap<String, usize> = BTreeMap::new();
            for idx in 0..inner.tasks.len() {
                if out.len() >= want {
                    break;
                }
                let (eligible, job_id, priority) = {
                    let qt = &inner.tasks[idx];
                    if qt.state != TaskState::Queued {
                        continue;
                    }
                    let Some(js) = inner.jobs.get(&qt.task.job_id) else {
                        continue;
                    };
                    if !js.open() {
                        continue;
                    }
                    let Some(n) = inner.nodes.get(node_id) else {
                        continue;
                    };
                    if !Self::can_execute(n, &qt.task) {
                        continue;
                    }
                    if !Self::can_confirm_replica(&inner, js, qt, n) {
                        continue;
                    }
                    (true, js.j.id.clone(), js.j.priority.max(1))
                };
                if !eligible {
                    continue;
                }
                let cap_this_pass = if pass > 1 { priority as usize } else { 1 };
                let taken = taken_this_pass.get(&job_id).copied().unwrap_or(0);
                if taken >= cap_this_pass {
                    continue;
                }
                taken_this_pass.insert(job_id.clone(), taken + 1);
                let qt = &mut inner.tasks[idx];
                qt.state = TaskState::Leased;
                qt.node_id = node_id.to_string();
                qt.attempts += 1;
                qt.epoch += 1;
                qt.task.lease_epoch = qt.epoch;
                qt.leased_at = Some(now);
                out.push(idx);
                progressed = true;
            }
            if !progressed {
                break;
            }
            pass += 1;
        }
        // LPT: biggest estimated tasks first.
        out.sort_by(|&a, &b| {
            inner.tasks[b]
                .task
                .est_millis
                .cmp(&inner.tasks[a].task.est_millis)
        });
        let mut tasks = Vec::with_capacity(out.len());
        for idx in out {
            let d = self.lease_duration(&inner, node_id, &inner.tasks[idx].task);
            let qt = &mut inner.tasks[idx];
            qt.deadline = Some(now + d);
            tasks.push(qt.task.clone());
        }
        tasks
    }

    pub fn renew_leases(&self, node_id: &str, refs: &[TaskLeaseRef]) -> usize {
        let now = Instant::now();
        let mut inner = self.inner.lock().unwrap();
        let Some(n) = inner.nodes.get_mut(node_id) else {
            return 0;
        };
        n.seen = now;
        let mut extended = 0;
        for r in refs {
            let Some(&idx) = inner.tasks_by_id.get(&r.task_id) else {
                continue;
            };
            let ok = {
                let qt = &inner.tasks[idx];
                qt.state == TaskState::Leased
                    && qt.node_id == node_id
                    && (r.lease_epoch == 0 || r.lease_epoch == qt.epoch)
                    && inner.jobs.get(&qt.task.job_id).is_some_and(|js| js.open())
                    && qt
                        .leased_at
                        .is_none_or(|la| now.duration_since(la) < 4 * LEASE_CAP)
            };
            if !ok {
                continue;
            }
            let d = self.lease_duration(&inner, node_id, &inner.tasks[idx].task);
            inner.tasks[idx].deadline = Some(now + d);
            extended += 1;
        }
        extended
    }

    // --- results ---

    pub fn result_in(&self, res: ResultMsg) -> bool {
        let now = Instant::now();
        let mut inner = self.inner.lock().unwrap();
        let Some(&idx) = inner.tasks_by_id.get(&res.task_id) else {
            return false;
        };
        let job_id = inner.tasks[idx].task.job_id.clone();
        if !inner.jobs.contains_key(&job_id) {
            return false;
        }
        {
            let qt = &inner.tasks[idx];
            if qt.state == TaskState::Done {
                return false;
            }
            if res.lease_epoch != 0 && res.lease_epoch != qt.epoch {
                return false; // stale epoch — never touch the active lessee
            }
            if qt.state == TaskState::Leased && !res.node_id.is_empty() && res.node_id != qt.node_id
            {
                return false;
            }
        }

        if !res.error.is_empty() {
            if res.unsupported {
                let kind = inner.tasks[idx].task.kind.clone();
                let msg = format!(
                    "no node can run workload {kind:?} (agent older than coordinator?): {}",
                    res.error
                );
                if let Some(js) = inner.jobs.get_mut(&job_id) {
                    js.failed = true;
                    js.j.error = msg.clone();
                    js.j.status = JOB_FAILED.into();
                }
                inner.tasks[idx].state = TaskState::Done;
                inner.tasks[idx].deadline = None;
                inner.unsupported_total += 1;
                eprintln!("[job] {job_id} FAILED: {msg}");
                return true;
            }
            return self.fail_task(&mut inner, idx, &res.error, "error report");
        }
        let payload = res.payload();
        if payload.is_empty() {
            return self.fail_task(&mut inner, idx, "empty payload", "empty result");
        }

        // Compute-only ExecMillis clamped to [1, wall elapsed].
        let mut actual_ms = if res.exec_millis > 0 {
            res.exec_millis
        } else {
            res.millis
        };
        if actual_ms > 0
            && let Some(la) = inner.tasks[idx].leased_at
        {
            let wall = (now.duration_since(la).as_millis() as i64).max(1);
            actual_ms = actual_ms.min(wall).max(1);
        }

        let replicate = inner.jobs.get(&job_id).is_some_and(|js| js.replicate);
        if replicate {
            let pending_from = inner.tasks[idx].pending_from.clone();
            if pending_from.is_empty() {
                let (uuid, ip) = inner
                    .nodes
                    .get(&res.node_id)
                    .map(|n| (n.uuid.clone(), n.remote_ip.clone()))
                    .unwrap_or_default();
                let qt = &mut inner.tasks[idx];
                qt.pending_from = res.node_id.clone();
                qt.pending_from_uuid = uuid;
                qt.pending_from_ip = ip;
                qt.replica_payload = Some(payload);
                self.requeue(&mut inner, idx);
                let kind = inner.tasks[idx].task.kind.clone();
                let est = inner.tasks[idx].task.est_millis;
                if let Some(n) = inner.nodes.get_mut(&res.node_id) {
                    n.running = n.running.saturating_sub(1);
                    if actual_ms > 0 {
                        n.observe_preset_speed(&kind, est, actual_ms);
                    }
                }
                eprintln!(
                    "[replica] {} awaiting second opinion (first from {})",
                    res.task_id, res.node_id
                );
                return true;
            }
            let confirm_ok = {
                let js = inner.jobs.get(&job_id).unwrap();
                let qt = &inner.tasks[idx];
                inner
                    .nodes
                    .get(&res.node_id)
                    .is_some_and(|n| Self::can_confirm_replica(&inner, js, qt, n))
            };
            if !confirm_ok {
                if res.node_id == pending_from {
                    return true; // same session re-sending its first result
                }
                eprintln!(
                    "[replica] {} ignored duplicate-host confirmation from {}",
                    res.task_id, res.node_id
                );
                self.requeue(&mut inner, idx);
                return true;
            }
            if res.node_id != pending_from {
                let primary = inner.tasks[idx].replica_payload.clone().unwrap_or_default();
                let verified = (self.verify_fn.lock().unwrap())(
                    &inner.tasks[idx].task.kind,
                    &primary,
                    &payload,
                );
                if verified {
                    let qt = &mut inner.tasks[idx];
                    qt.pending_from.clear();
                    qt.pending_from_uuid.clear();
                    qt.pending_from_ip.clear();
                    qt.replica_payload = None;
                    eprintln!(
                        "[replica] {} confirmed by two independent nodes",
                        res.task_id
                    );
                } else {
                    let first = pending_from.clone();
                    {
                        let qt = &mut inner.tasks[idx];
                        qt.pending_from.clear();
                        qt.pending_from_uuid.clear();
                        qt.pending_from_ip.clear();
                        qt.replica_payload = None;
                    }
                    for id in [first, res.node_id.clone()] {
                        if let Some(nd) = inner.nodes.get_mut(&id) {
                            nd.failed += 1;
                            nd.running = nd.running.saturating_sub(1);
                        }
                    }
                    eprintln!("[replica] {} MISMATCH — requeueing", res.task_id);
                    return self.fail_task(&mut inner, idx, "replica mismatch", "disagreement");
                }
            } else {
                return true;
            }
        }

        let len = payload.len();
        {
            let js = inner.jobs.get_mut(&job_id).unwrap();
            js.payloads.insert(res.task_id.clone(), payload);
            js.payload_sizes.insert(res.task_id.clone(), len);
            js.j.payload_bytes += len as i64;
            js.j.tasks_done += 1;
        }
        inner.tasks[idx].state = TaskState::Done;

        let kind = inner.tasks[idx].task.kind.clone();
        let est = inner.tasks[idx].task.est_millis;
        if let Some(n) = inner.nodes.get_mut(&res.node_id) {
            n.done += 1;
            n.running = n.running.saturating_sub(1);
            if actual_ms > 0 {
                let r = 1000.0 / actual_ms as f64;
                if n.runs == 0 {
                    n.rate = r;
                } else {
                    n.rate = n.rate * (1.0 - RATE_ALPHA) + r * RATE_ALPHA;
                }
                n.runs += 1;
                n.observe_preset_speed(&kind, est, actual_ms);
            }
        }

        let js = inner.jobs.get_mut(&job_id).unwrap();
        if js.failed || js.j.status == JOB_CANCELLED {
            // settled elsewhere
        } else if js.j.tasks_done == js.j.tasks_total {
            js.done = true;
            js.j.status = JOB_DONE.into();
            eprintln!("[job] {job_id} complete ({} tasks)", js.j.tasks_total);
        } else if js.j.status == JOB_QUEUED {
            js.j.status = JOB_RUNNING.into();
        }
        true
    }

    fn fail_task(&self, inner: &mut Inner, idx: usize, cause: &str, kind: &str) -> bool {
        let job_id = inner.tasks[idx].task.job_id.clone();
        let settled = inner
            .jobs
            .get(&job_id)
            .is_some_and(|js| js.j.status == JOB_CANCELLED || js.done);
        if settled {
            inner.tasks[idx].state = TaskState::Done;
            inner.tasks[idx].deadline = None;
            return true;
        }
        if inner.tasks[idx].attempts >= MAX_ATTEMPTS {
            let task_id = inner.tasks[idx].task.id.clone();
            let attempts = inner.tasks[idx].attempts;
            let msg = format!("task {task_id} {kind} after {attempts} attempts: {cause}");
            if let Some(js) = inner.jobs.get_mut(&job_id) {
                js.failed = true;
                js.j.error = msg.clone();
                js.j.status = JOB_FAILED.into();
            }
            eprintln!("[job] {job_id} FAILED: {msg}");
            inner.tasks[idx].state = TaskState::Done;
            inner.tasks[idx].deadline = None;
            return true;
        }
        self.requeue(inner, idx);
        inner.requeued_total += 1;
        eprintln!(
            "[task] {} {kind} on {} — retrying (attempt {}/{}): {cause}",
            inner.tasks[idx].task.id,
            inner.tasks[idx].node_id,
            inner.tasks[idx].attempts + 1,
            MAX_ATTEMPTS
        );
        true
    }

    fn requeue(&self, inner: &mut Inner, idx: usize) {
        inner.tasks[idx].state = TaskState::Queued;
        inner.tasks[idx].deadline = None;
        // Move to the front of the FIFO (matches compost requeueLocked).
        if idx > 0 {
            let qt = inner.tasks.remove(idx);
            inner.tasks.insert(0, qt);
            // Rebuild index map positions (cheap: indices shifted by one).
            for (i, t) in inner.tasks.iter().enumerate() {
                inner.tasks_by_id.insert(t.task.id.clone(), i);
            }
        }
    }

    // --- reclamation ---

    pub fn reclaim_job_payloads(&self, job_id: &str) -> i64 {
        let mut inner = self.inner.lock().unwrap();
        let Some(js) = inner.jobs.get_mut(job_id) else {
            return 0;
        };
        if js.reclaimed {
            return 0;
        }
        let mut freed = 0i64;
        for id in &js.order {
            if let Some(buf) = js.payloads.get(id) {
                let entry = js.payload_sizes.entry(id.clone()).or_insert(0);
                if *entry == 0 {
                    *entry = buf.len();
                }
                freed += buf.len() as i64;
            }
        }
        js.payloads.clear();
        js.reclaimed = true;
        js.j.reclaimed = true;
        for qt in inner.tasks.iter_mut() {
            if qt.task.job_id == job_id
                && let Some(rp) = qt.replica_payload.take()
            {
                freed += rp.len() as i64;
            }
        }
        if freed > 0 {
            eprintln!("[reclaim] {job_id} freed {freed} bytes of raw task payloads");
        }
        freed
    }

    pub fn is_reclaimed(&self, job_id: &str) -> bool {
        self.inner
            .lock()
            .unwrap()
            .jobs
            .get(job_id)
            .is_some_and(|js| js.reclaimed)
    }

    pub fn payload_size(&self, job_id: &str, task_id: &str) -> Option<usize> {
        let inner = self.inner.lock().unwrap();
        inner
            .jobs
            .get(job_id)
            .and_then(|js| js.payload_sizes.get(task_id).copied())
    }

    // --- reaper ---

    pub fn reap(&self) {
        let now = Instant::now();
        let mut inner = self.inner.lock().unwrap();
        let expired: Vec<usize> = inner
            .tasks
            .iter()
            .enumerate()
            .filter(|(_, qt)| qt.state == TaskState::Leased && qt.deadline.is_some_and(|d| now > d))
            .map(|(i, _)| i)
            .collect();
        for idx in expired {
            if inner.tasks[idx].state != TaskState::Leased {
                continue;
            }
            let holder = inner.tasks[idx].node_id.clone();
            if let Some(h) = inner.nodes.get_mut(&holder) {
                h.failed += 1;
                h.running = h.running.saturating_sub(1);
            }
            self.fail_task(&mut inner, idx, "lease expired", "timed out");
        }
        let stale: Vec<String> = inner
            .nodes
            .iter()
            .filter(|(_, n)| now.duration_since(n.seen) > FORGET_AFTER)
            .map(|(id, _)| id.clone())
            .collect();
        for id in stale {
            inner.nodes.remove(&id);
        }
        for n in inner.nodes.values_mut() {
            if now.duration_since(n.seen) > OFFLINE_AFTER {
                n.running = 0;
            }
        }
    }

    pub fn cancel_job(&self, job_id: &str) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(js) = inner.jobs.get(job_id) else {
            return false;
        };
        if matches!(js.j.status.as_str(), JOB_DONE | JOB_FAILED | JOB_CANCELLED) {
            return true;
        }
        for qt in inner.tasks.iter_mut() {
            if qt.task.job_id == job_id && qt.state != TaskState::Done {
                qt.state = TaskState::Done;
            }
        }
        inner.jobs.get_mut(job_id).unwrap().j.status = JOB_CANCELLED.into();
        eprintln!("[job] {job_id} cancelled by user");
        true
    }

    // --- views ---

    pub fn snapshot(&self) -> StateResponse {
        let inner = self.inner.lock().unwrap();
        let now = Instant::now();
        let mut nodes = Vec::new();
        for n in inner.nodes.values() {
            let mut paused = n.paused.clone();
            let r = n.guard_reason();
            if !r.is_empty() {
                paused = r;
            }
            let preset_speeds: BTreeMap<String, f64> = n
                .preset_speed
                .iter()
                .map(|(k, v)| (k.clone(), v.factor))
                .collect();
            nodes.push(Node {
                id: n.id.clone(),
                node_uuid: n.uuid.clone(),
                name: n.name.clone(),
                caps: n.caps.clone(),
                joined_at: iso8601_from_unix(
                    n.joined
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                ),
                last_seen: iso8601_from_unix(
                    SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                ),
                online: now.duration_since(n.seen) < OFFLINE_AFTER,
                rate: n.rate,
                running: n.running,
                done: n.done,
                failed: n.failed,
                speed_factor: n.speed_factor,
                preset_speeds,
                paused,
            });
        }
        let mut jobs: Vec<Job> = inner
            .jobs
            .values()
            .map(|js| {
                let mut j = js.j.clone();
                j.input = None; // never leak job input over /state
                j
            })
            .collect();
        jobs.sort_by(|a, b| a.id.cmp(&b.id));
        StateResponse {
            nodes,
            jobs,
            workloads: Vec::new(),
        }
    }

    pub fn assemble_inputs(&self, job_id: &str) -> Option<AssembleInputs> {
        let inner = self.inner.lock().unwrap();
        let js = inner.jobs.get(job_id)?;
        Some((js.j.clone(), js.order.clone(), js.payloads.clone()))
    }

    pub fn tile_states(&self, job_id: &str) -> Option<Vec<String>> {
        let inner = self.inner.lock().unwrap();
        let js = inner.jobs.get(job_id)?;
        Some(
            js.order
                .iter()
                .map(|id| {
                    inner
                        .tasks_by_id
                        .get(id)
                        .map(|&idx| match inner.tasks[idx].state {
                            TaskState::Done => "done",
                            TaskState::Leased => "busy",
                            TaskState::Queued => "queued",
                        })
                        .unwrap_or("queued")
                        .to_string()
                })
                .collect(),
        )
    }

    pub fn job_exists(&self, id: &str) -> bool {
        self.inner.lock().unwrap().jobs.contains_key(id)
    }

    pub fn job_kind(&self, id: &str) -> Option<String> {
        self.inner
            .lock()
            .unwrap()
            .jobs
            .get(id)
            .map(|js| js.j.kind.clone())
    }

    pub fn task_by_id(&self, task_id: &str) -> Option<Task> {
        let inner = self.inner.lock().unwrap();
        inner
            .tasks_by_id
            .get(task_id)
            .map(|&idx| inner.tasks[idx].task.clone())
    }

    pub fn active_jobs(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner
            .jobs
            .values()
            .filter(|js| matches!(js.j.status.as_str(), JOB_QUEUED | JOB_RUNNING))
            .count()
    }

    // --- speed profiles ---

    pub fn learn_profiles(&self) -> Vec<SpeedProfile> {
        let inner = self.inner.lock().unwrap();
        self.learn_profiles_locked(&inner)
    }

    pub fn seed_learn_profiles(&self, profiles: Vec<SpeedProfile>) {
        let mut inner = self.inner.lock().unwrap();
        for p in profiles {
            inner
                .learn_by_key
                .insert(profile_key(&p.node_uuid, &p.name), p);
        }
    }

    // --- persistence ---

    pub fn save(&self) -> Snapshot {
        let inner = self.inner.lock().unwrap();
        let mut snap = Snapshot {
            version: crate::persist::SNAPSHOT_VERSION,
            job_seq: inner.job_seq,
            id_prefix: inner.id_prefix.clone(),
            learned_speeds: Vec::new(),
            jobs: Vec::new(),
        };
        for p in self.learn_profiles_locked(&inner) {
            snap.learned_speeds.push(LearnedSpeed {
                node_uuid: p.node_uuid,
                name: p.name,
                factor: p.factor,
                n: p.n,
                preset_speeds: p.preset_speeds,
            });
        }
        for js in inner.jobs.values() {
            let mut rec = JobRecord {
                job: js.j.clone(),
                order: js.order.clone(),
                payloads: js.payloads.clone(),
                payload_sizes: js.payload_sizes.clone(),
                reclaimed: js.reclaimed,
                artifact: None,
                artifact_type: String::new(),
                artifact_ext: String::new(),
                tasks: Vec::new(),
            };
            rec.job.input = js.j.input.clone();
            for qt in inner.tasks.iter().filter(|t| t.task.job_id == js.j.id) {
                rec.tasks.push(TaskRecord {
                    task: qt.task.clone(),
                    state: qt.state as u8,
                    attempts: qt.attempts,
                    epoch: qt.epoch,
                    pending_from: qt.pending_from.clone(),
                    pending_from_uuid: qt.pending_from_uuid.clone(),
                    pending_from_ip: qt.pending_from_ip.clone(),
                    replica_payload: qt.replica_payload.clone(),
                });
            }
            snap.jobs.push(rec);
        }
        snap
    }

    fn learn_profiles_locked(&self, inner: &Inner) -> Vec<SpeedProfile> {
        let mut out = Vec::new();
        let mut emitted = std::collections::HashSet::new();
        for n in inner.nodes.values() {
            if n.speed_n == 0 && n.preset_speed.is_empty() {
                continue;
            }
            let k = profile_key(&n.uuid, &n.name);
            emitted.insert(k);
            out.push(SpeedProfile {
                node_uuid: n.uuid.clone(),
                name: n.name.clone(),
                factor: n.speed_factor,
                n: n.speed_n,
                preset_speeds: n
                    .preset_speed
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            PresetSpeedSample {
                                factor: v.factor,
                                n: v.n,
                            },
                        )
                    })
                    .collect(),
            });
        }
        for (k, p) in &inner.learn_by_key {
            if !emitted.contains(k) {
                out.push(p.clone());
            }
        }
        out
    }

    pub fn restore(&self, snap: &Snapshot) {
        let mut inner = self.inner.lock().unwrap();
        if !snap.id_prefix.is_empty() {
            inner.id_prefix = snap.id_prefix.clone();
        }
        inner.job_seq = snap.job_seq;
        for rec in &snap.jobs {
            let mut payload_sizes = rec.payload_sizes.clone();
            let mut payload_bytes = 0i64;
            for (id, payload) in &rec.payloads {
                payload_sizes.entry(id.clone()).or_insert(payload.len());
            }
            for sz in payload_sizes.values() {
                payload_bytes += *sz as i64;
            }
            let mut js = JobState {
                failed: rec.job.status == JOB_FAILED,
                done: rec.job.status == JOB_DONE,
                replicate: rec.job.replicate,
                j: rec.job.clone(),
                order: rec.order.clone(),
                payloads: rec.payloads.clone(),
                payload_sizes,
                reclaimed: rec.reclaimed,
            };
            js.j.payload_bytes = payload_bytes;
            for tr in &rec.tasks {
                let mut state = match tr.state {
                    1 => TaskState::Leased,
                    2 => TaskState::Done,
                    _ => TaskState::Queued,
                };
                if state == TaskState::Leased {
                    state = TaskState::Queued; // holder vanished with the old process
                }
                let epoch = if tr.epoch != 0 {
                    tr.epoch
                } else {
                    tr.task.lease_epoch
                };
                let idx = inner.tasks.len();
                inner.tasks.push(QueuedTask {
                    task: tr.task.clone(),
                    state,
                    attempts: tr.attempts,
                    node_id: String::new(),
                    epoch,
                    leased_at: None,
                    deadline: None,
                    pending_from: tr.pending_from.clone(),
                    pending_from_uuid: tr.pending_from_uuid.clone(),
                    pending_from_ip: tr.pending_from_ip.clone(),
                    replica_payload: tr.replica_payload.clone(),
                });
                inner.tasks_by_id.insert(tr.task.id.clone(), idx);
            }
            inner.jobs.insert(js.j.id.clone(), js);
        }
        eprintln!("[persist] restored {} jobs", snap.jobs.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::Caps;

    fn caps() -> Caps {
        Caps {
            cores: 4,
            mem_mb: 8192,
            score: 1.0,
            battery_pct: -1,
            temp_c: -1.0,
            arch: "x86_64".into(),
            os: "linux".into(),
            version: "0.7.0".into(),
            ..Default::default()
        }
    }

    fn task(job: &str, id: &str, est: i64) -> Task {
        Task {
            id: id.into(),
            job_id: job.into(),
            kind: "primes".into(),
            est_millis: est,
            ..Default::default()
        }
    }

    fn job(id: &str) -> Job {
        Job {
            id: id.into(),
            kind: "primes".into(),
            status: JOB_QUEUED.into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            ..Default::default()
        }
    }

    #[test]
    fn lease_epoch_monotonic_and_stale_rejected() {
        let q = Queue::new();
        let n = q.join("pc", caps());
        q.add_job(job("j1"), vec![task("j1", "t1", 100)]);
        let t1 = q.lease(&n, "").unwrap();
        assert_eq!(t1.lease_epoch, 1);
        // Stale epoch result is rejected without touching state.
        assert!(!q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n.clone(),
            lease_epoch: 99,
            data: Some(b"x".to_vec()),
            millis: 10,
            ..Default::default()
        }));
        // Correct epoch accepted.
        assert!(q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n.clone(),
            lease_epoch: 1,
            data: Some(b"42".to_vec()),
            millis: 10,
            exec_millis: 10,
            ..Default::default()
        }));
        // Duplicate after settle rejected.
        assert!(!q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n,
            lease_epoch: 1,
            data: Some(b"42".to_vec()),
            millis: 10,
            ..Default::default()
        }));
    }

    #[test]
    fn heartbeat_renews_only_matching_epoch() {
        let q = Queue::new();
        q.set_lease_fn(Box::new(|_, _| Duration::from_millis(50)));
        let n = q.join("phone", caps());
        q.add_job(job("j1"), vec![task("j1", "t1", 10)]);
        let t = q.lease(&n, "").unwrap();
        std::thread::sleep(Duration::from_millis(30));
        let extended = q.renew_leases(
            &n,
            &[TaskLeaseRef {
                task_id: "t1".into(),
                lease_epoch: t.lease_epoch,
            }],
        );
        assert_eq!(extended, 1);
        // Wrong epoch does not extend.
        let extended = q.renew_leases(
            &n,
            &[TaskLeaseRef {
                task_id: "t1".into(),
                lease_epoch: 999,
            }],
        );
        assert_eq!(extended, 0);
        std::thread::sleep(Duration::from_millis(30));
        q.reap(); // renewed lease must survive
        let snap = q.snapshot();
        assert_eq!(snap.jobs[0].status, JOB_QUEUED);
        assert_eq!(snap.jobs[0].tasks_done, 0);
    }

    #[test]
    fn lease_expiry_requeues_then_poison_fails_job() {
        let q = Queue::new();
        q.set_lease_fn(Box::new(|_, _| Duration::from_millis(10)));
        let n = q.join("pc", caps());
        q.add_job(job("j1"), vec![task("j1", "t1", 10)]);
        for _ in 0..MAX_ATTEMPTS {
            let _ = q.lease(&n, "").unwrap();
            std::thread::sleep(Duration::from_millis(15));
            q.reap();
        }
        let snap = q.snapshot();
        assert_eq!(snap.jobs[0].status, JOB_FAILED);
        assert!(snap.jobs[0].error.contains("5 attempts"));
    }

    #[test]
    fn batch_fairness_priority_and_lpt() {
        let q = Queue::new();
        let n = q.join("pc", caps());
        let mut j1 = job("j1");
        j1.priority = 0;
        let mut j2 = job("j2");
        j2.priority = 3;
        q.add_job(
            j1,
            vec![
                task("j1", "a1", 10),
                task("j1", "a2", 20),
                task("j1", "a3", 30),
            ],
        );
        q.add_job(
            j2,
            vec![
                task("j2", "b1", 40),
                task("j2", "b2", 50),
                task("j2", "b3", 60),
            ],
        );
        let batch = q.lease_batch(&n, "", 0, 5);
        assert_eq!(batch.len(), 5);
        // Pass 1: one per job (a1, b1). Pass 2: j1 cap 1 (a2), j2 cap 3 (b2, b3).
        let ids: Vec<&str> = batch.iter().map(|t| t.id.as_str()).collect();
        assert!(ids.contains(&"a1"));
        assert!(ids.contains(&"a2"));
        assert_eq!(ids.iter().filter(|i| i.starts_with('b')).count(), 3);
        // LPT: biggest first.
        assert_eq!(batch[0].id, "b3");
    }

    #[test]
    fn sybil_replication_guard() {
        let q = Queue::new();
        let n1 = q.join_with_meta("pixel", "uuid-A", "10.0.0.5", caps());
        let n2 = q.join_with_meta("pixel", "uuid-A", "10.0.0.9", caps()); // same UUID, different IP
        let n3 = q.join_with_meta("pixel2", "uuid-B", "10.0.0.5", caps()); // same IP as n1
        let n4 = q.join_with_meta("pc", "uuid-C", "10.0.0.7", caps());
        let mut j = job("j1");
        j.replicate = true;
        q.add_job(j, vec![task("j1", "t1", 10)]);
        let t = q.lease(&n1, "").unwrap();
        assert!(q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n1.clone(),
            lease_epoch: t.lease_epoch,
            data: Some(b"7".to_vec()),
            millis: 5,
            ..Default::default()
        }));
        // Same UUID cannot even lease the replica task.
        assert!(q.lease(&n2, "").is_none());
        // Same IP cannot lease it either.
        assert!(q.lease(&n3, "").is_none());
        // Result-time Sybil guard: a raced result from the same UUID is
        // ignored without settling the task.
        assert!(q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n2.clone(),
            lease_epoch: t.lease_epoch,
            data: Some(b"7".to_vec()),
            millis: 5,
            ..Default::default()
        }));
        assert_eq!(q.snapshot().jobs[0].tasks_done, 0);
        // Genuinely distinct node confirms.
        let t4 = q.lease(&n4, "").unwrap();
        assert!(q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n4,
            lease_epoch: t4.lease_epoch,
            data: Some(b"7".to_vec()),
            millis: 5,
            ..Default::default()
        }));
        let snap = q.snapshot();
        assert_eq!(snap.jobs[0].tasks_done, 1);
        assert_eq!(snap.jobs[0].status, JOB_DONE);
    }

    #[test]
    fn replica_mismatch_fails() {
        let q = Queue::new();
        let n1 = q.join_with_meta("a", "uuid-A", "10.0.0.1", caps());
        let n2 = q.join_with_meta("b", "uuid-B", "10.0.0.2", caps());
        let mut j = job("j1");
        j.replicate = true;
        q.add_job(j, vec![task("j1", "t1", 10)]);
        let t = q.lease(&n1, "").unwrap();
        q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n1,
            lease_epoch: t.lease_epoch,
            data: Some(b"7".to_vec()),
            millis: 5,
            ..Default::default()
        });
        let t2 = q.lease(&n2, "").unwrap();
        q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n2,
            lease_epoch: t2.lease_epoch,
            data: Some(b"8".to_vec()),
            millis: 5,
            ..Default::default()
        });
        // Mismatch requeues; after enough disagreements the job fails.
        let snap = q.snapshot();
        assert!(snap.jobs[0].tasks_done == 0);
    }

    #[test]
    fn speed_learning_ewma_with_prior() {
        let q = Queue::new();
        let n = q.join("pc", caps());
        q.add_job(job("j1"), vec![task("j1", "t1", 100)]);
        let t = q.lease(&n, "").unwrap();
        // ExecMillis is clamped to wall elapsed: give the lease real time.
        std::thread::sleep(Duration::from_millis(80));
        q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n.clone(),
            lease_epoch: t.lease_epoch,
            data: Some(b"ok".to_vec()),
            millis: 50,
            exec_millis: 50,
            ..Default::default()
        });
        let snap = q.snapshot();
        let node = &snap.nodes[0];
        // First sample: raw = 100/50 = 2.0 → (2.0 + 3*1)/(1+3) = 1.25
        assert!(
            (node.speed_factor - 1.25).abs() < 1e-9,
            "factor {}",
            node.speed_factor
        );
        assert!((node.preset_speeds["primes"] - 1.25).abs() < 1e-9);
    }

    #[test]
    fn battery_and_thermal_guards() {
        let q = Queue::new();
        let mut c = caps();
        c.battery_pct = 20;
        c.on_battery = true;
        let n = q.join("phone", c);
        q.add_job(job("j1"), vec![task("j1", "t1", 10)]);
        assert!(q.lease(&n, "").is_none());
        assert!(q.snapshot().nodes[0].paused.contains("battery 20%"));
        // Thermal guard (CPU > 80).
        let mut c2 = caps();
        c2.temp_c = 85.0;
        let n2 = q.join("hot-pc", c2);
        assert!(q.lease(&n2, "").is_none());
        // Android battery thermal > 45.
        let mut c3 = caps();
        c3.os = "android".into();
        c3.temp_c = 46.0;
        let n3 = q.join("hot-phone", c3);
        assert!(q.lease(&n3, "").is_none());
        assert!(
            q.snapshot()
                .nodes
                .iter()
                .find(|x| x.id == n3)
                .unwrap()
                .paused
                .contains("45°C")
        );
    }

    #[test]
    fn llm_gating() {
        let q = Queue::new();
        let n = q.join("pc", caps());
        let mut t = task("j1", "t1", 10);
        t.kind = TASK_LLM_BATCH.into();
        q.add_job(job("j1"), vec![t.clone()]);
        assert!(q.lease(&n, "").is_none()); // no local LLM
        // Verify-mode tasks bypass the gate.
        let mut tv = t.clone();
        tv.id = "t2".into();
        tv.verify_mode = true;
        q.add_job(job("j2"), vec![tv]);
        assert!(q.lease(&n, "").is_some());
    }

    #[test]
    fn reclamation_preserves_sizes() {
        let q = Queue::new();
        let n = q.join("pc", caps());
        q.add_job(job("j1"), vec![task("j1", "t1", 10)]);
        let t = q.lease(&n, "").unwrap();
        q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n,
            lease_epoch: t.lease_epoch,
            data: Some(vec![0u8; 4096]),
            millis: 5,
            ..Default::default()
        });
        assert_eq!(q.payload_size("j1", "t1"), Some(4096));
        let freed = q.reclaim_job_payloads("j1");
        assert_eq!(freed, 4096);
        assert!(q.is_reclaimed("j1"));
        assert_eq!(q.payload_size("j1", "t1"), Some(4096));
        assert_eq!(q.reclaim_job_payloads("j1"), 0); // idempotent
        let (_, _, payloads) = q.assemble_inputs("j1").unwrap();
        assert!(payloads.is_empty());
    }

    #[test]
    fn save_restore_roundtrip() {
        let q = Queue::new();
        let n = q.join("pc", caps());
        let id = q.next_job_id();
        q.add_job(job(&id), vec![task(&id, "t1", 10), task(&id, "t2", 20)]);
        let t = q.lease(&n, "").unwrap();
        q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n,
            lease_epoch: t.lease_epoch,
            data: Some(b"5".to_vec()),
            millis: 5,
            exec_millis: 5,
            ..Default::default()
        });
        let snap = q.save();
        let q2 = Queue::new();
        q2.restore(&snap);
        let s2 = q2.snapshot();
        assert_eq!(s2.jobs[0].tasks_done, 1);
        assert_eq!(s2.jobs[0].tasks_total, 2);
        // Leased task became queued again; IDs continue after restore.
        assert_eq!(q2.next_job_id(), "job-2");
        let n2 = q2.join("pc", caps());
        let t2 = q2.lease(&n2, "").unwrap();
        assert_eq!(t2.id, "t2");
    }

    #[test]
    fn unsupported_fails_job_immediately() {
        let q = Queue::new();
        let n = q.join("pc", caps());
        q.add_job(job("j1"), vec![task("j1", "t1", 10)]);
        let t = q.lease(&n, "").unwrap();
        q.result_in(ResultMsg {
            task_id: "t1".into(),
            node_id: n,
            lease_epoch: t.lease_epoch,
            error: "no preset".into(),
            unsupported: true,
            millis: 1,
            ..Default::default()
        });
        let snap = q.snapshot();
        assert_eq!(snap.jobs[0].status, JOB_FAILED);
        assert!(snap.jobs[0].error.contains("no node can run workload"));
        assert_eq!(q.unsupported_total(), 1);
    }
}
