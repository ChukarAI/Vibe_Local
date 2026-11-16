//! Distributed escape-time fractal renderer — the original compost workload.
//! Exact algorithm port (tile geometry, palette, cross-arch replica tolerance).

use std::collections::BTreeMap;

use crate::proto::{JOB_QUEUED, Job, TASK_MANDELBROT, Task};
use crate::workload::{AggregateOut, ENC_RGBA, Preset, get_param};

pub struct Mandelbrot;

fn plane_bounds(px: i32, py: i32, width: i32, height: i32) -> (f64, f64) {
    const XMIN: f64 = -2.5;
    const XMAX: f64 = 1.0;
    const YMIN: f64 = -1.0;
    const YMAX: f64 = 1.0;
    let x = XMIN + (XMAX - XMIN) * f64::from(px) / f64::from(width);
    let y = YMIN + (YMAX - YMIN) * f64::from(py) / f64::from(height);
    (x, y)
}

/// Factor n into (cols, rows) whose tiles are closest to square.
fn split_grid(n: i32) -> (i32, i32) {
    let mut best_cols = 1;
    let mut best_score = f64::MAX;
    for c in 1..=n {
        if n % c != 0 {
            continue;
        }
        let r = n / c;
        let mut ar = f64::from(c) / f64::from(r);
        if ar < 1.0 {
            ar = 1.0 / ar;
        }
        if ar < best_score {
            best_score = ar;
            best_cols = c;
        }
    }
    (best_cols, n / best_cols)
}

fn tile_size(job: &Job, col: i32, row: i32) -> (i32, i32) {
    let mut tw = job.tile_w;
    let rem = job.width - job.tile_w * job.cols;
    if rem > 0 && col < rem {
        tw += 1;
    }
    let mut th = job.tile_h;
    let rem = job.height - job.tile_h * job.rows;
    if rem > 0 && row < rem {
        th += 1;
    }
    (tw, th)
}

fn tile_origin(job: &Job, col: i32, row: i32) -> (i32, i32) {
    let mut x = col * job.tile_w;
    let extra = col.min((job.width - job.tile_w * job.cols).max(0));
    if extra > 0 {
        x += extra;
    }
    let mut y = row * job.tile_h;
    let extra = row.min((job.height - job.tile_h * job.rows).max(0));
    if extra > 0 {
        y += extra;
    }
    (x, y)
}

fn palette(i: i32, max_iter: i32) -> (u8, u8, u8) {
    if i >= max_iter {
        return (8, 12, 24); // deep-space navy for in-set points
    }
    let t = f64::from(i) / f64::from(max_iter);
    if t < 0.5 {
        let u = t / 0.5;
        (
            (30.0 + 40.0 * u) as u8,
            (60.0 + 150.0 * u) as u8,
            (180.0 - 60.0 * u) as u8,
        )
    } else {
        let u = (t - 0.5) / 0.5;
        (
            (70.0 + 185.0 * u) as u8,
            (210.0 + 45.0 * u) as u8,
            (120.0 - 90.0 * u) as u8,
        )
    }
}

fn render_rows(px: &mut [u8], t: &Task, y_start: i32, y_end: i32) {
    for y in y_start..y_end {
        let ci = t.y0 + (t.y1 - t.y0) * f64::from(y) / f64::from(t.height);
        let row = y as usize * t.width as usize * 4;
        for x in 0..t.width {
            let cr = t.x0 + (t.x1 - t.x0) * f64::from(x) / f64::from(t.width);
            let mut zx = 0.0f64;
            let mut zy = 0.0f64;
            let mut i = 0;
            while i < t.max_iter {
                let zx2 = zx * zx;
                let zy2 = zy * zy;
                if zx2 + zy2 > 4.0 {
                    break;
                }
                zy = 2.0 * zx * zy + ci;
                zx = zx2 - zy2 + cr;
                i += 1;
            }
            let (r, g, b) = palette(i, t.max_iter);
            let o = row + x as usize * 4;
            px[o] = r;
            px[o + 1] = g;
            px[o + 2] = b;
            px[o + 3] = 255;
        }
    }
}

impl Preset for Mandelbrot {
    fn name(&self) -> &'static str {
        TASK_MANDELBROT
    }
    fn describe(&self) -> &'static str {
        "distributed escape-time fractal render → PNG"
    }

    fn validate(&self, params: &BTreeMap<String, f64>) -> anyhow::Result<Job> {
        let width = get_param(params, "width", 1280.0).clamp(16.0, 8192.0) as i32;
        let height = get_param(params, "height", 720.0).clamp(16.0, 8192.0) as i32;
        let iter = get_param(params, "max_iter", 800.0).clamp(32.0, 100_000.0) as i32;
        let (cols, rows) = split_grid(96);
        let tw = width / cols;
        let th = height / rows;
        anyhow::ensure!(
            tw >= 4 && th >= 4,
            "resolution {width}x{height} too small for tiling"
        );
        Ok(Job {
            id: String::new(),
            kind: self.name().into(),
            status: JOB_QUEUED.into(),
            width,
            height,
            max_iter: iter,
            cols,
            rows,
            tile_w: tw,
            tile_h: th,
            params: BTreeMap::from([
                ("width".into(), f64::from(width)),
                ("height".into(), f64::from(height)),
                ("max_iter".into(), f64::from(iter)),
            ]),
            ..Default::default()
        })
    }

    fn split(&self, job: &Job) -> anyhow::Result<Vec<Task>> {
        let mut tasks = Vec::with_capacity((job.cols * job.rows) as usize);
        for r in 0..job.rows {
            for c in 0..job.cols {
                let (tw, th) = tile_size(job, c, r);
                let (x_off, y_off) = tile_origin(job, c, r);
                let (x0, y0) = plane_bounds(x_off, y_off, job.width, job.height);
                let (x1, _) = plane_bounds(x_off + tw, y_off, job.width, job.height);
                let (_, y1) = plane_bounds(x_off, y_off + th, job.width, job.height);
                tasks.push(Task {
                    id: format!("{}-r{r}c{c}", job.id),
                    job_id: job.id.clone(),
                    kind: self.name().into(),
                    width: tw,
                    height: th,
                    x0,
                    y0,
                    x1,
                    y1,
                    max_iter: job.max_iter,
                    // ~150k px·iter per single-core millisecond on reference
                    // hardware; good to ±3× for lease deadlines.
                    est_millis: (f64::from(tw * th * job.max_iter) / 150_000.0) as i64,
                    ..Default::default()
                });
            }
        }
        Ok(tasks)
    }

    fn execute(&self, t: &Task) -> anyhow::Result<(Vec<u8>, &'static str)> {
        anyhow::ensure!(
            t.width > 0 && t.height > 0 && t.max_iter > 0,
            "render: bad task dims {}x{} iter {}",
            t.width,
            t.height,
            t.max_iter
        );
        let mut px = vec![0u8; (t.width * t.height * 4) as usize];
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let chunk = (t.height as usize).div_ceil(workers);
        std::thread::scope(|scope| {
            let mut rest: &mut [u8] = &mut px;
            for w in 0..workers {
                let y0 = (w * chunk) as i32;
                let y1 = (((w + 1) * chunk) as i32).min(t.height);
                if y0 >= y1 {
                    break;
                }
                let band = (y1 - y0) as usize * t.width as usize * 4;
                let (slice, tail) = rest.split_at_mut(band);
                rest = tail;
                scope.spawn(move || {
                    // Render into a local buffer then copy, keeping row math
                    // relative to the band start.
                    let mut local = vec![0u8; band];
                    let mut shifted = t.clone();
                    shifted.height = y1 - y0;
                    shifted.y0 = t.y0 + (t.y1 - t.y0) * f64::from(y0) / f64::from(t.height);
                    shifted.y1 = t.y0 + (t.y1 - t.y0) * f64::from(y1) / f64::from(t.height);
                    render_rows(&mut local, &shifted, 0, y1 - y0);
                    slice.copy_from_slice(&local);
                });
            }
        });
        Ok((px, ENC_RGBA))
    }

    fn aggregate(
        &self,
        job: &Job,
        order: &[String],
        results: &BTreeMap<String, Vec<u8>>,
    ) -> anyhow::Result<AggregateOut> {
        let mut img = vec![0u8; (job.width * job.height * 4) as usize];
        for (idx, task_id) in order.iter().enumerate() {
            let col = (idx as i32) % job.cols;
            let row = (idx as i32) / job.cols;
            let (tw, th) = tile_size(job, col, row);
            let buf = results
                .get(task_id)
                .ok_or_else(|| anyhow::anyhow!("render: missing tile {task_id}"))?;
            anyhow::ensure!(
                buf.len() >= (tw * th * 4) as usize,
                "render: tile {task_id} payload {} bytes < {tw}x{th} expected",
                buf.len()
            );
            let (x_off, y_off) = tile_origin(job, col, row);
            for y in 0..th {
                let src = &buf[(y * tw * 4) as usize..((y + 1) * tw * 4) as usize];
                let dst_off = ((y_off + y) * job.width * 4 + x_off * 4) as usize;
                img[dst_off..dst_off + (tw * 4) as usize].copy_from_slice(src);
            }
        }
        // Encode PNG (RGBA8).
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, job.width as u32, job.height as u32);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_image_data(&img)?;
        }
        Ok((out, "image/png".into(), ".png".into()))
    }

    fn result_size(&self, t: &Task) -> usize {
        (t.width * t.height * 4).max(0) as usize
    }

    /// Exact equality, or ≤0.05% edge pixels differing by ≤1 palette step
    /// (RGB delta ≤12, identical alpha) — absorbs cross-architecture
    /// IEEE-754 FMA/NEON 1-ULP differences while rejecting corruption.
    fn verify_replica(&self, primary: &[u8], replica: &[u8]) -> bool {
        if primary.is_empty() || primary.len() != replica.len() || !primary.len().is_multiple_of(4)
        {
            return false;
        }
        if primary == replica {
            return true;
        }
        const MAX_CHANNEL_DELTA: i32 = 12;
        let total_pixels = primary.len() / 4;
        let mut diff_pixels = 0usize;
        for i in (0..primary.len()).step_by(4) {
            if primary[i] == replica[i]
                && primary[i + 1] == replica[i + 1]
                && primary[i + 2] == replica[i + 2]
                && primary[i + 3] == replica[i + 3]
            {
                continue;
            }
            if primary[i + 3] != replica[i + 3] {
                return false; // alpha corruption is never a 1-ULP edge diff
            }
            for c in 0..3 {
                let d = (i32::from(primary[i + c]) - i32::from(replica[i + c])).abs();
                if d > MAX_CHANNEL_DELTA {
                    return false;
                }
            }
            diff_pixels += 1;
        }
        diff_pixels * 2000 <= total_pixels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_job() -> Job {
        Mandelbrot
            .validate(&BTreeMap::from([
                ("width".into(), 128.0),
                ("height".into(), 96.0),
                ("max_iter".into(), 64.0),
            ]))
            .unwrap()
    }

    #[test]
    fn split_covers_plane() {
        let job = small_job();
        let tasks = Mandelbrot.split(&job).unwrap();
        assert_eq!(tasks.len(), 96);
        assert_eq!(tasks[0].id, format!("{}-r0c0", job.id));
        let total_px: i64 = tasks.iter().map(|t| i64::from(t.width * t.height)).sum();
        assert_eq!(total_px, i64::from(job.width * job.height));
    }

    #[test]
    fn render_and_aggregate_png() {
        let mut job = small_job();
        job.id = "job-test".into();
        let tasks = Mandelbrot.split(&job).unwrap();
        let mut results = BTreeMap::new();
        let mut order = Vec::new();
        for t in &tasks {
            let (px, enc) = Mandelbrot.execute(t).unwrap();
            assert_eq!(enc, ENC_RGBA);
            assert_eq!(px.len(), Mandelbrot.result_size(t));
            order.push(t.id.clone());
            results.insert(t.id.clone(), px);
        }
        let (png_bytes, ct, ext) = Mandelbrot.aggregate(&job, &order, &results).unwrap();
        assert_eq!(ct, "image/png");
        assert_eq!(ext, ".png");
        assert_eq!(&png_bytes[1..4], b"PNG");
        // Determinism: re-render one tile identically.
        let (a, _) = Mandelbrot.execute(&tasks[0]).unwrap();
        let (b, _) = Mandelbrot.execute(&tasks[0]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn replica_tolerance() {
        let mut a = vec![0u8; 4 * 100_000];
        for p in a.as_chunks_mut::<4>().0 {
            p.copy_from_slice(&[100, 150, 200, 255]);
        }
        let b = a.clone();
        assert!(Mandelbrot.verify_replica(&a, &b));
        // One pixel off by one palette step: allowed.
        let mut c = a.clone();
        c[0] += 12;
        assert!(Mandelbrot.verify_replica(&a, &c));
        // One pixel off by 13: rejected.
        let mut d = a.clone();
        d[0] += 13;
        assert!(!Mandelbrot.verify_replica(&a, &d));
        // Alpha corruption: rejected.
        let mut e = a.clone();
        e[3] = 254;
        assert!(!Mandelbrot.verify_replica(&a, &e));
        // >0.05% differing pixels: rejected (60 of 100k > 50).
        let mut f = a.clone();
        for i in 0..60 {
            f[i * 4] += 1;
        }
        assert!(!Mandelbrot.verify_replica(&a, &f));
        // Exactly 0.05% (50 of 100k): allowed.
        let mut g = a.clone();
        for i in 0..50 {
            g[i * 4] += 1;
        }
        assert!(Mandelbrot.verify_replica(&a, &g));
        assert!(!Mandelbrot.verify_replica(&[], &[]));
    }

    #[test]
    fn grid_split_shapes() {
        // 8x12 wins the aspect-ratio tie (first best factor encountered).
        assert_eq!(split_grid(96), (8, 12));
        assert_eq!(split_grid(1), (1, 1));
        let (c, r) = split_grid(97); // prime → 1x97
        assert_eq!(c * r, 97);
    }
}
