//! Zero-dependency Prometheus `/metrics` with compost-compatible metric
//! names, so existing scrape configs keep working against Vibe Grid.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::proto::{JOB_CANCELLED, JOB_DONE, JOB_FAILED, StateResponse};

static API_JOIN: AtomicU64 = AtomicU64::new(0);
static API_POLL: AtomicU64 = AtomicU64::new(0);
static API_RESULT: AtomicU64 = AtomicU64::new(0);
static API_SUBMIT: AtomicU64 = AtomicU64::new(0);
static API_CANCEL: AtomicU64 = AtomicU64::new(0);
static API_STATE: AtomicU64 = AtomicU64::new(0);
static API_ARTIFACT: AtomicU64 = AtomicU64::new(0);
static RESULTS_TOTAL: AtomicU64 = AtomicU64::new(0);
static NODES_ONLINE: AtomicU64 = AtomicU64::new(0);
static JOBS_DONE: AtomicU64 = AtomicU64::new(0);
static JOBS_FAILED: AtomicU64 = AtomicU64::new(0);
static JOBS_CANCELLED: AtomicU64 = AtomicU64::new(0);
static TASKS_REQUEUED: AtomicU64 = AtomicU64::new(0);
static UNSUPPORTED: AtomicU64 = AtomicU64::new(0);

pub fn count_route(route: &str) {
    let c = match route {
        "join" => &API_JOIN,
        "poll" => &API_POLL,
        "result" => &API_RESULT,
        "submit" => &API_SUBMIT,
        "cancel" => &API_CANCEL,
        "state" => &API_STATE,
        "artifact" | "tiles" => &API_ARTIFACT,
        _ => return,
    };
    c.fetch_add(1, Ordering::Relaxed);
}

pub fn count_result() {
    RESULTS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn job_settled(status: &str) {
    match status {
        JOB_DONE => {
            JOBS_DONE.fetch_add(1, Ordering::Relaxed);
        }
        JOB_FAILED => {
            JOBS_FAILED.fetch_add(1, Ordering::Relaxed);
        }
        JOB_CANCELLED => {
            JOBS_CANCELLED.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

pub fn set_requeued(n: u64) {
    TASKS_REQUEUED.store(n, Ordering::Relaxed);
}

pub fn set_unsupported(n: u64) {
    UNSUPPORTED.store(n, Ordering::Relaxed);
}

pub fn publish_gauges(st: &StateResponse) {
    let online = st.nodes.iter().filter(|n| n.online).count() as u64;
    NODES_ONLINE.store(online, Ordering::Relaxed);
}

pub fn render() -> String {
    let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
    format!(
        "# HELP compost_api_requests_total API requests by route.\n\
         # TYPE compost_api_requests_total counter\n\
         compost_api_requests_total{{route=\"join\"}} {}\n\
         compost_api_requests_total{{route=\"poll\"}} {}\n\
         compost_api_requests_total{{route=\"result\"}} {}\n\
         compost_api_requests_total{{route=\"submit\"}} {}\n\
         compost_api_requests_total{{route=\"cancel\"}} {}\n\
         compost_api_requests_total{{route=\"state\"}} {}\n\
         compost_api_requests_total{{route=\"artifact\"}} {}\n\
         # HELP compost_results_total task results received.\n\
         # TYPE compost_results_total counter\n\
         compost_results_total {}\n\
         # HELP compost_nodes_online nodes seen in the last offline window.\n\
         # TYPE compost_nodes_online gauge\n\
         compost_nodes_online {}\n\
         # HELP compost_jobs_total jobs by terminal outcome.\n\
         # TYPE compost_jobs_total counter\n\
         compost_jobs_total{{status=\"done\"}} {}\n\
         compost_jobs_total{{status=\"failed\"}} {}\n\
         compost_jobs_total{{status=\"cancelled\"}} {}\n\
         # HELP compost_tasks_requeued_total lease expiries and error retries.\n\
         # TYPE compost_tasks_requeued_total counter\n\
         compost_tasks_requeued_total {}\n\
         # HELP compost_tasks_unsupported_total results reporting an unknown workload.\n\
         # TYPE compost_tasks_unsupported_total counter\n\
         compost_tasks_unsupported_total {}\n",
        g(&API_JOIN),
        g(&API_POLL),
        g(&API_RESULT),
        g(&API_SUBMIT),
        g(&API_CANCEL),
        g(&API_STATE),
        g(&API_ARTIFACT),
        g(&RESULTS_TOTAL),
        g(&NODES_ONLINE),
        g(&JOBS_DONE),
        g(&JOBS_FAILED),
        g(&JOBS_CANCELLED),
        g(&TASKS_REQUEUED),
        g(&UNSUPPORTED),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_render_prometheus_text() {
        count_route("join");
        count_route("poll");
        count_result();
        job_settled(JOB_DONE);
        let text = render();
        assert!(text.contains("# TYPE compost_api_requests_total counter"));
        assert!(text.contains("compost_results_total"));
        assert!(text.contains("compost_jobs_total{status=\"done\"}"));
    }
}
