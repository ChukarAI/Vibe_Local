//! Data-parallel word-frequency count: word-aligned chunks (map) merged into
//! a grand tally (reduce). Deterministic, replication-friendly.

use std::collections::BTreeMap;

use crate::proto::{JOB_QUEUED, Job, TASK_WORDCOUNT, Task};
use crate::workload::{AggregateOut, ENC_TEXT, Preset, get_param};

pub struct Wordcount;

const DEFAULT_CHUNK: usize = 64 * 1024;
pub const MAX_INPUT: usize = 8 << 20;

/// Last whitespace boundary at or before `i`; hard-cut at a UTF-8 rune
/// boundary when a single token exceeds the chunk window.
fn split_at_word_break(text: &[u8], i: usize) -> usize {
    let s = String::from_utf8_lossy(&text[..i]);
    let mut j = s.len();
    for (idx, ch) in s.char_indices().rev() {
        if ch.is_whitespace() {
            return idx + ch.len_utf8();
        }
        j = idx;
    }
    let _ = j;
    // No break found: hard-cut at the last rune boundary ≤ i.
    let mut end = i;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

fn count_words(text: &[u8]) -> BTreeMap<String, u64> {
    let s = String::from_utf8_lossy(text);
    let mut counts = BTreeMap::new();
    for word in s.split_whitespace() {
        *counts.entry(word.to_lowercase()).or_insert(0) += 1;
    }
    counts
}

impl Preset for Wordcount {
    fn name(&self) -> &'static str {
        TASK_WORDCOUNT
    }
    fn describe(&self) -> &'static str {
        "parallel word-frequency count over raw text → TSV"
    }

    fn validate(&self, params: &BTreeMap<String, f64>) -> anyhow::Result<Job> {
        let chunk = get_param(params, "chunk", DEFAULT_CHUNK as f64);
        anyhow::ensure!(
            chunk >= 1024.0,
            "wordcount: chunk must be ≥1024, got {chunk}"
        );
        anyhow::ensure!(
            chunk <= (1 << 20) as f64,
            "wordcount: chunk ≤1MiB, got {chunk}"
        );
        Ok(Job {
            id: String::new(),
            kind: self.name().into(),
            status: JOB_QUEUED.into(),
            width: chunk as i32,
            params: BTreeMap::from([("chunk".into(), chunk)]),
            ..Default::default()
        })
    }

    fn split(&self, job: &Job) -> anyhow::Result<Vec<Task>> {
        let text = job
            .input
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("wordcount: job has no input"))?;
        anyhow::ensure!(!text.is_empty(), "wordcount: empty input");
        let chunk = job.width.max(1024) as usize;
        let mut tasks = Vec::new();
        let mut off = 0usize;
        while off < text.len() {
            let mut end = off + chunk;
            if end >= text.len() {
                end = text.len();
            } else {
                end = split_at_word_break(text, end);
                if end <= off {
                    // Degenerate no-whitespace window: advance one rune.
                    let s = String::from_utf8_lossy(&text[off..]);
                    let size = s.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                    end = off + size;
                }
            }
            let slice = &text[off..end];
            tasks.push(Task {
                id: format!("{}-w{}", job.id, tasks.len()),
                job_id: job.id.clone(),
                kind: self.name().into(),
                input: Some(slice.to_vec()),
                start: off as i64,
                count: slice.len() as i64,
                est_millis: (slice.len() / 40) as i64,
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
            .ok_or_else(|| anyhow::anyhow!("wordcount: task has no input chunk"))?;
        let counts = count_words(input);
        let mut buf = String::new();
        for (word, n) in &counts {
            buf.push_str(&format!("{word}\t{n}\n"));
        }
        Ok((buf.into_bytes(), ENC_TEXT))
    }

    fn aggregate(
        &self,
        _job: &Job,
        order: &[String],
        results: &BTreeMap<String, Vec<u8>>,
    ) -> anyhow::Result<AggregateOut> {
        let mut total: BTreeMap<String, u64> = BTreeMap::new();
        for id in order {
            let buf = results
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("wordcount: missing chunk {id}"))?;
            let text = String::from_utf8_lossy(buf);
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let (word, count) = line.split_once('\t').ok_or_else(|| {
                    anyhow::anyhow!("wordcount: bad payload line in {id}: {line:?}")
                })?;
                let n: u64 = count
                    .parse()
                    .map_err(|_| anyhow::anyhow!("wordcount: bad count in {id}: {count:?}"))?;
                *total.entry(word.to_string()).or_insert(0) += n;
            }
        }
        let sum: u64 = total.values().sum();
        let mut words: Vec<(&String, &u64)> = total.iter().collect();
        words.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let mut out = format!("# total_words\t{sum}\n# unique_words\t{}\n", total.len());
        for (word, n) in words {
            out.push_str(&format!("{word}\t{n}\n"));
        }
        Ok((
            out.into_bytes(),
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
    fn map_reduce_matches_local_count() {
        let text = b"the quick brown fox jumps over the lazy dog. The FOX! the fox.";
        let mut job = Wordcount
            .validate(&BTreeMap::from([("chunk".into(), 1024.0)]))
            .unwrap();
        job.id = "job-w".into();
        job.input = Some(text.to_vec());
        let tasks = Wordcount.split(&job).unwrap();
        let mut results = BTreeMap::new();
        let order: Vec<String> = tasks.iter().map(|t| t.id.clone()).collect();
        for t in &tasks {
            let (data, _) = Wordcount.execute(t).unwrap();
            results.insert(t.id.clone(), data);
        }
        let (artifact, _, _) = Wordcount.aggregate(&job, &order, &results).unwrap();
        let text = String::from_utf8_lossy(&artifact);
        assert!(text.contains("# total_words\t13"));
        assert!(text.contains("the\t4"));
        // Whitespace tokenization keeps punctuation attached (wc -w style).
        assert!(text.contains("fox\t1"));
        assert!(text.contains("fox!\t1"));
    }

    #[test]
    fn chunking_never_splits_words() {
        let mut job = Wordcount
            .validate(&BTreeMap::from([("chunk".into(), 1024.0)]))
            .unwrap();
        job.id = "j".into();
        let text = "word ".repeat(5000);
        job.input = Some(text.clone().into_bytes());
        let tasks = Wordcount.split(&job).unwrap();
        assert!(tasks.len() > 1);
        let mut reconstructed = Vec::new();
        for t in &tasks {
            let chunk = t.input.as_ref().unwrap();
            // Every chunk ends at a word boundary (whitespace) or EOF.
            let last = *chunk.last().unwrap();
            assert!(
                last == b' ' || std::ptr::eq(chunk, tasks.last().unwrap().input.as_ref().unwrap())
            );
            reconstructed.extend_from_slice(chunk);
        }
        assert_eq!(String::from_utf8_lossy(&reconstructed), text);
    }

    #[test]
    fn unicode_hard_cut() {
        // One giant token with multibyte chars: must not cut mid-rune.
        let text = "é".repeat(3000);
        let mut job = Wordcount
            .validate(&BTreeMap::from([("chunk".into(), 1024.0)]))
            .unwrap();
        job.id = "j".into();
        job.input = Some(text.into_bytes());
        let tasks = Wordcount.split(&job).unwrap();
        for t in &tasks {
            assert!(std::str::from_utf8(t.input.as_ref().unwrap()).is_ok());
        }
    }

    #[test]
    fn empty_input_rejected() {
        let mut job = Wordcount.validate(&BTreeMap::new()).unwrap();
        job.input = Some(Vec::new());
        assert!(Wordcount.split(&job).is_err());
    }
}
