//! Prime counting by trial division on the 6k±1 wheel. Correctness is
//! locally checkable against known π values — ideal for distributed
//! agreement tests. π(10^5)=9592, π(10^6)=78498, π(10^7)=664579.

use std::collections::BTreeMap;

use crate::proto::{JOB_QUEUED, Job, TASK_PRIMES, Task};
use crate::workload::{AggregateOut, ENC_TEXT, Preset, get_param};

pub struct Primes;

const DEFAULT_LIMIT: i64 = 10_000_000;
const MAX_LIMIT: i64 = 1_000_000_000;
const CHUNK: i64 = 250_000;

pub fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5i64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

pub fn count_primes_in_range(start: i64, count: i64) -> i64 {
    (start..start + count).filter(|&i| is_prime(i)).count() as i64
}

impl Preset for Primes {
    fn name(&self) -> &'static str {
        TASK_PRIMES
    }
    fn describe(&self) -> &'static str {
        "count primes in a range (trial division + 6k±1) → text"
    }

    fn validate(&self, params: &BTreeMap<String, f64>) -> anyhow::Result<Job> {
        let limit = get_param(params, "limit", DEFAULT_LIMIT as f64) as i64;
        anyhow::ensure!(limit >= 100, "primes: limit must be ≥100, got {limit}");
        anyhow::ensure!(
            limit <= MAX_LIMIT,
            "primes: limit ≤{MAX_LIMIT}, got {limit}"
        );
        Ok(Job {
            id: String::new(),
            kind: self.name().into(),
            status: JOB_QUEUED.into(),
            width: limit as i32, // reuse Width as the scan limit (compost convention)
            params: BTreeMap::from([("limit".into(), limit as f64)]),
            ..Default::default()
        })
    }

    fn split(&self, job: &Job) -> anyhow::Result<Vec<Task>> {
        let limit = i64::from(job.width);
        let mut tasks = Vec::new();
        let mut start = 2i64;
        while start <= limit {
            let end = (start + CHUNK - 1).min(limit);
            tasks.push(Task {
                id: format!("{}-p{start}-{end}", job.id),
                job_id: job.id.clone(),
                kind: self.name().into(),
                start,
                count: end - start + 1,
                end,
                // ~0.9 ms per 1k candidates on reference hardware.
                est_millis: ((end - start + 1) as f64 * 0.0009) as i64,
                ..Default::default()
            });
            start += CHUNK;
        }
        Ok(tasks)
    }

    fn execute(&self, t: &Task) -> anyhow::Result<(Vec<u8>, &'static str)> {
        anyhow::ensure!(
            t.start >= 2 && t.count > 0,
            "primes: bad range {}+{}",
            t.start,
            t.count
        );
        let n = count_primes_in_range(t.start, t.count);
        Ok((n.to_string().into_bytes(), ENC_TEXT))
    }

    fn aggregate(
        &self,
        job: &Job,
        order: &[String],
        results: &BTreeMap<String, Vec<u8>>,
    ) -> anyhow::Result<AggregateOut> {
        let mut total = 0i64;
        for id in order {
            let buf = results
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("primes: missing chunk {id}"))?;
            let text = String::from_utf8_lossy(buf);
            let v: i64 = text
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("primes: chunk {id} bad payload {text:?}"))?;
            total += v;
        }
        let text = format!("primes below {}: {total}\n", job.width);
        Ok((
            text.into_bytes(),
            "text/plain; charset=utf-8".into(),
            ".txt".into(),
        ))
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
    fn known_pi_values() {
        assert_eq!(count_primes_in_range(2, 99_999), 9592); // π(10^5)
        assert_eq!(count_primes_in_range(2, 999_999), 78498); // π(10^6)
    }

    #[test]
    fn split_execute_agree() {
        let mut job = Primes
            .validate(&BTreeMap::from([("limit".into(), 100_000.0)]))
            .unwrap();
        job.id = "job-p".into();
        let tasks = Primes.split(&job).unwrap();
        let mut results = BTreeMap::new();
        let order: Vec<String> = tasks.iter().map(|t| t.id.clone()).collect();
        for t in &tasks {
            let (data, _) = Primes.execute(t).unwrap();
            results.insert(t.id.clone(), data);
        }
        let (artifact, _, _) = Primes.aggregate(&job, &order, &results).unwrap();
        assert_eq!(
            String::from_utf8_lossy(&artifact),
            "primes below 100000: 9592\n"
        );
    }

    #[test]
    fn validation_bounds() {
        assert!(
            Primes
                .validate(&BTreeMap::from([("limit".into(), 50.0)]))
                .is_err()
        );
        assert!(
            Primes
                .validate(&BTreeMap::from([("limit".into(), 2e9)]))
                .is_err()
        );
        assert!(Primes.validate(&BTreeMap::new()).is_ok());
    }

    #[test]
    fn is_prime_edges() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(97));
        assert!(!is_prime(100));
    }
}
