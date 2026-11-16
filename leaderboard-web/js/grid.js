/**
 * Vibe Grid — Wi-Fi MoE & Cluster view.
 * Shows the distributed fleet (phones + PCs), expert shard allocation map,
 * battery/thermal guard badges and live job/tile progress.
 *
 * Data flows through the same-origin Vibe-Link proxy (/api/grid/*) so the
 * strict CSP (connect-src 'self') holds. All DOM via createElement/textContent.
 */
import { el, clear } from './dom.js';

const MOE_MODELS = [
  'OLMoE-1B-7B-0924-Instruct',
  'Granite-3.1-3B-A800M-Instruct',
  'Qwen1.5-MoE-A2.7B-Chat',
  'DeepSeek-V2-Lite-16B',
];

const NODE_COLORS = ['#7C5CFF', '#22D3EE', '#4ade80', '#fbbf24', '#f472b6', '#60a5fa', '#f87171', '#a3e635'];

function nodeColor(id, registry) {
  if (!registry.has(id)) registry.set(id, NODE_COLORS[registry.size % NODE_COLORS.length]);
  return registry.get(id);
}

async function gridFetch(path, { method = 'GET', body, key, signal } = {}) {
  const headers = { Accept: 'application/json' };
  if (key) headers['X-Compost-Key'] = key;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const res = await fetch(path, { method, headers, body: body !== undefined ? JSON.stringify(body) : undefined, signal, cache: 'no-store' });
  const text = await res.text();
  let data = null;
  try { data = JSON.parse(text); } catch { data = text; }
  if (!res.ok) {
    const msg = data && data.error ? data.error : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

export function createGridView(panel) {
  const state = {
    coordinator: localStorage.getItem('vibe-grid-coordinator') || '',
    key: localStorage.getItem('vibe-grid-key') || '',
    active: false,
    timer: null,
    lastState: null,
    plan: null,
    colors: new Map(),
    busy: false,
  };

  // ----- connection bar -----
  const coordInput = el('input', { type: 'text', class: 'input', placeholder: 'http://127.0.0.1:7900 (default)', value: state.coordinator, 'aria-label': 'Coordinator URL' });
  const keyInput = el('input', { type: 'password', class: 'input', placeholder: 'fleet key (X-Compost-Key)', value: state.key, 'aria-label': 'Fleet key' });
  const connectBtn = el('button', { type: 'button', class: 'btn btn-primary btn-sm', text: 'Connect' });
  const connStatus = el('span', { class: 'grid-conn-status', text: 'Not connected' });
  const connBar = el('div', { class: 'grid-conn-bar' },
    el('label', { class: 'grid-field' }, el('span', { class: 'grid-field-label', text: 'Coordinator' }), coordInput),
    el('label', { class: 'grid-field' }, el('span', { class: 'grid-field-label', text: 'Fleet key' }), keyInput),
    connectBtn, connStatus);

  // ----- sections -----
  const fleetBox = el('div', { class: 'grid-fleet' });
  const planControls = el('div', { class: 'grid-plan-controls' });
  const planBox = el('div', { class: 'grid-plan' });
  const jobsBox = el('div', { class: 'grid-jobs' });
  const submitBox = el('div', { class: 'grid-submit' });
  const notice = el('div', { class: 'grid-notice', hidden: true });

  panel.appendChild(el('div', { class: 'panel-head' },
    el('h2', { text: 'Vibe Grid — Wi-Fi MoE & Cluster' }),
    el('p', { class: 'panel-desc', text: 'Distributed compute across phones + PCs (compost v0.7.0 wire-compatible). Workers connect outward over Wi-Fi — no inbound ports. Sparse MoE experts shard across the fleet; the coordinator keeps the backbone and hottest experts.' })));
  panel.appendChild(connBar);
  panel.appendChild(notice);
  panel.appendChild(el('h3', { class: 'grid-section-title', text: 'Fleet' }));
  panel.appendChild(fleetBox);
  panel.appendChild(el('h3', { class: 'grid-section-title', text: 'MoE expert shard plan' }));
  panel.appendChild(planControls);
  panel.appendChild(planBox);
  panel.appendChild(el('h3', { class: 'grid-section-title', text: 'Jobs' }));
  panel.appendChild(jobsBox);
  panel.appendChild(el('h3', { class: 'grid-section-title', text: 'Submit workload' }));
  panel.appendChild(submitBox);
  panel.appendChild(el('p', { class: 'grid-footer-note', text: 'Vibe Grid is wire-compatible with compost v0.7.0. Phones connect outward over Wi-Fi; no inbound ports required. Battery <30% and thermal guards pause work automatically.' }));

  function showNotice(msg, kind) {
    notice.hidden = !msg;
    notice.textContent = msg || '';
    notice.className = `grid-notice grid-notice-${kind || 'info'}`;
  }

  function queryParams() {
    const c = coordInput.value.trim();
    return c ? `?coordinator=${encodeURIComponent(c)}` : '';
  }

  // ----- fleet rendering -----
  function guardBadge(n) {
    const paused = n.paused || '';
    if (!paused) return null;
    const lower = paused.toLowerCase();
    const cls = lower.includes('thermal') ? 'badge-red' : 'badge-amber';
    return el('span', { class: `grid-badge ${cls}`, text: paused, title: paused });
  }

  function renderFleet(st) {
    clear(fleetBox);
    const nodes = (st && st.nodes) || [];
    if (!nodes.length) {
      fleetBox.appendChild(el('p', { class: 'grid-empty', text: 'No nodes have joined yet — start one with `vibe-local cluster agent --coordinator <url>`.' }));
      return;
    }
    for (const n of nodes) {
      const c = n.caps || {};
      const card = el('div', { class: 'grid-node-card' });
      const head = el('div', { class: 'grid-node-head' },
        el('span', { class: 'grid-node-dot', style: { background: nodeColor(n.id, state.colors) } }),
        el('strong', { text: n.name || n.id }),
        el('span', { class: `grid-badge ${n.online ? 'badge-green' : 'badge-gray'}`, text: n.online ? 'online' : 'offline' }));
      const gb = guardBadge(n);
      if (gb) head.appendChild(gb);
      if (c.has_local_llm) head.appendChild(el('span', { class: 'grid-badge badge-violet', text: 'LLM :20129' }));
      card.appendChild(head);
      const batt = (typeof c.battery_pct === 'number' && c.battery_pct >= 0) ? ` · battery ${c.battery_pct}%${c.on_battery ? ' (on battery)' : ''}` : '';
      const temp = (typeof c.temp_c === 'number' && c.temp_c >= 0) ? ` · ${c.temp_c.toFixed(0)}°C` : '';
      card.appendChild(el('div', { class: 'grid-node-meta', text: `${c.os || '?'}/${c.arch || '?'} · ${c.cores || '?'} cores · ${c.mem_mb || '?'} MB RAM${batt}${temp}` }));
      card.appendChild(el('div', { class: 'grid-node-meta', text: `speed ×${(n.speed_factor || 0).toFixed(2)} · rate ${(n.rate || 0).toFixed(2)} tasks/s · done ${n.done || 0} · failed ${n.failed || 0} · running ${n.running || 0}` }));
      fleetBox.appendChild(card);
    }
  }

  // ----- MoE planner -----
  const modelSelect = el('select', { class: 'input', 'aria-label': 'MoE model' });
  for (const m of MOE_MODELS) modelSelect.appendChild(el('option', { value: m, text: m }));
  const ctxInput = el('input', { type: 'number', class: 'input input-narrow', value: '4096', min: '512', max: '131072', 'aria-label': 'Context length' });
  const planBtn = el('button', { type: 'button', class: 'btn btn-primary btn-sm', text: 'Plan MoE' });
  planControls.appendChild(el('label', { class: 'grid-field' }, el('span', { class: 'grid-field-label', text: 'Model' }), modelSelect));
  planControls.appendChild(el('label', { class: 'grid-field' }, el('span', { class: 'grid-field-label', text: 'Context' }), ctxInput));
  planControls.appendChild(planBtn);

  function renderPlan(plan) {
    clear(planBox);
    if (!plan) {
      planBox.appendChild(el('p', { class: 'grid-empty', text: 'Choose a model and press Plan MoE to compute the expert/RAM shard map across the live fleet.' }));
      return;
    }
    const summary = el('div', { class: 'grid-plan-summary' },
      el('span', { class: 'grid-badge badge-violet', text: `${plan.routed_experts} experts · top-${plan.experts_per_token}` }),
      el('span', { class: 'grid-badge badge-cyan', text: `est ${Number(plan.est_decode_tps || 0).toFixed(1)} tok/s` }),
      el('span', { class: 'grid-badge badge-green', text: `${Number(plan.est_speedup_vs_single_node || 0).toFixed(2)}× vs single node` }),
      el('span', { class: 'grid-plan-meta', text: `backbone: ${plan.backbone_node} · Wi-Fi RTT ${plan.wifi_rtt_ms}ms · ${plan.quant} · ctx ${plan.context}` }));
    planBox.appendChild(summary);

    // Segmented expert bar.
    const total = plan.routed_experts || 64;
    const bar = el('div', { class: 'grid-shard-bar', role: 'img', 'aria-label': `Expert allocation map: ${total} experts across ${(plan.shards || []).length} shards` });
    const tooltip = el('div', { class: 'grid-tooltip', hidden: true });
    for (const s of plan.shards || []) {
      const span = s.expert_end - s.expert_start;
      const color = nodeColor(s.node_id, state.colors);
      const seg = el('div', {
        class: 'grid-shard-seg',
        style: { width: `${(span / total) * 100}%`, background: color, opacity: s.role === 'host_paging' ? '0.45' : '1' },
        tabindex: '0',
        'aria-label': `${s.node_name}: experts ${s.expert_start}..${s.expert_end}, ${s.role}`,
      });
      const show = () => {
        tooltip.hidden = false;
        tooltip.textContent = `${s.node_name} · experts ${s.expert_start}..${s.expert_end} · ${Number(s.size_mb || 0).toFixed(0)} MB · ${s.role} · heat ${Number(s.heat || 0).toFixed(3)}`;
      };
      seg.addEventListener('mouseenter', show);
      seg.addEventListener('focus', show);
      seg.addEventListener('mouseleave', () => { tooltip.hidden = true; });
      seg.addEventListener('blur', () => { tooltip.hidden = true; });
      bar.appendChild(seg);
    }
    planBox.appendChild(bar);
    planBox.appendChild(tooltip);

    // Legend / shard cards.
    const legend = el('div', { class: 'grid-shard-legend' });
    for (const s of plan.shards || []) {
      legend.appendChild(el('div', { class: 'grid-shard-card' },
        el('span', { class: 'grid-node-dot', style: { background: nodeColor(s.node_id, state.colors) } }),
        el('div', {},
          el('strong', { text: s.node_name }),
          el('div', { class: 'grid-node-meta', text: `experts ${s.expert_start}..${s.expert_end} · ${Number(s.size_mb || 0).toFixed(0)} MB` })),
        el('span', { class: `grid-badge ${s.role === 'backbone' ? 'badge-violet' : s.role === 'host_paging' ? 'badge-amber' : 'badge-cyan'}`, text: s.role }),
        el('span', { class: 'grid-badge badge-gray', text: s.status })));
    }
    planBox.appendChild(legend);
    if ((plan.draft_nodes || []).length) {
      planBox.appendChild(el('p', { class: 'grid-plan-meta', text: `Speculative draft nodes (${plan.draft_tokens} draft tokens, verified by coordinator): ${plan.draft_nodes.join(', ')}` }));
    }
    for (const w of plan.warnings || []) {
      planBox.appendChild(el('p', { class: 'grid-warning', text: `⚠ ${w}` }));
    }
  }

  planBtn.addEventListener('click', async () => {
    planBtn.disabled = true;
    try {
      const ctx = Math.max(512, Number(ctxInput.value) || 4096);
      const q = queryParams();
      const sep = q ? '&' : '?';
      const plan = await gridFetch(`/api/grid/plan${q}${sep}model_id=${encodeURIComponent(modelSelect.value)}&context=${ctx}`, { key: keyInput.value.trim() });
      state.plan = plan;
      renderPlan(plan);
      showNotice('', 'info');
    } catch (e) {
      showNotice(`Plan failed: ${e.message}`, 'error');
    } finally {
      planBtn.disabled = false;
    }
  });

  // ----- jobs -----
  function renderJobs(st) {
    clear(jobsBox);
    const jobs = (st && st.jobs) || [];
    if (!jobs.length) {
      jobsBox.appendChild(el('p', { class: 'grid-empty', text: 'No jobs yet.' }));
      return;
    }
    const table = el('table', { class: 'grid-jobs-table' });
    const thead = el('thead', {}, el('tr', {},
      ...['Job', 'Kind', 'Status', 'Progress', 'ETA', 'Waiting / Error', ''].map(h => el('th', { text: h }))));
    table.appendChild(thead);
    const tbody = el('tbody');
    for (const j of jobs) {
      const statusCls = j.status === 'done' ? 'badge-green' : j.status === 'failed' ? 'badge-red' : j.status === 'cancelled' ? 'badge-gray' : 'badge-amber';
      const total = j.tasks_total || 0;
      const done = j.tasks_done || 0;
      const pct = total ? Math.round((done / total) * 100) : 0;
      const progress = el('div', { class: 'grid-progress', title: `${done}/${total}` },
        el('div', { class: 'grid-progress-fill', style: { width: `${pct}%` } }),
        el('span', { class: 'grid-progress-text', text: `${done}/${total}` }));
      const cancelBtn = el('button', { type: 'button', class: 'btn btn-ghost btn-sm', text: 'Cancel' });
      cancelBtn.addEventListener('click', async () => {
        cancelBtn.disabled = true;
        try {
          await gridFetch(`/api/grid/cancel/${encodeURIComponent(j.id)}${queryParams()}`, { method: 'POST', body: {}, key: keyInput.value.trim() });
          refresh();
        } catch (e) {
          showNotice(`Cancel failed: ${e.message}`, 'error');
          cancelBtn.disabled = false;
        }
      });
      const note = j.error || j.waiting || '';
      const tr = el('tr', {},
        el('td', { text: j.id }),
        el('td', { text: j.kind + (j.replicate ? ' ×2' : '') }),
        el('td', {}, el('span', { class: `grid-badge ${statusCls}`, text: j.status })),
        el('td', {}, progress),
        el('td', { text: j.eta_seconds ? `${j.eta_seconds}s` : '—' }),
        el('td', { class: 'grid-job-note', text: note, title: note }),
        el('td', {}, (j.status === 'queued' || j.status === 'running') ? cancelBtn : null));
      tbody.appendChild(tr);
      // Per-task tile strip for active jobs (done/busy/queued).
      if (j.status === 'queued' || j.status === 'running') {
        const tileCell = el('td', { colspan: '7' });
        const strip = el('div', { class: 'grid-tile-strip', 'aria-label': `Task tiles for ${j.id}` });
        tileCell.appendChild(strip);
        tbody.appendChild(el('tr', { class: 'grid-tile-row' }, tileCell));
        gridFetch(`/api/grid/tiles/${encodeURIComponent(j.id)}${queryParams()}`, { key: keyInput.value.trim() })
          .then((tiles) => {
            if (!Array.isArray(tiles)) return;
            clear(strip);
            for (const t of tiles.slice(0, 512)) {
              strip.appendChild(el('span', {
                class: `grid-tile grid-tile-${t}`,
                title: t,
              }));
            }
            if (tiles.length > 512) strip.appendChild(el('span', { class: 'grid-tile-more', text: `+${tiles.length - 512}` }));
          })
          .catch(() => { /* tiles are best-effort */ });
      }
    }
    table.appendChild(tbody);
    jobsBox.appendChild(table);
  }

  // ----- submit form -----
  const kindSelect = el('select', { class: 'input', 'aria-label': 'Workload kind' });
  for (const k of ['primes', 'mandelbrot', 'moe_distributed', 'wordcount', 'llm_batch']) {
    kindSelect.appendChild(el('option', { value: k, text: k }));
  }
  const paramsInput = el('input', { type: 'text', class: 'input', placeholder: 'limit=1000000', 'aria-label': 'Parameters (KEY=value, space separated)' });
  const inputArea = el('textarea', { class: 'input grid-textarea', placeholder: 'input text (wordcount / llm_batch)', rows: '3', 'aria-label': 'Job input text' });
  const verifyChk = el('input', { type: 'checkbox', id: 'grid-verify' });
  const verifyLabel = el('label', { class: 'grid-check' }, verifyChk, el('span', { text: 'llm_batch verify mode (offline)' }));
  const replicateChk = el('input', { type: 'checkbox', id: 'grid-replicate' });
  const priorityInput = el('input', { type: 'number', class: 'input input-narrow', value: '0', min: '0', max: '10', 'aria-label': 'Priority' });
  const submitBtn = el('button', { type: 'button', class: 'btn btn-primary btn-sm', text: 'Submit job' });
  const submitStatus = el('span', { class: 'grid-conn-status' });

  function syncSubmitFields() {
    const k = kindSelect.value;
    inputArea.hidden = !(k === 'wordcount' || k === 'llm_batch');
    verifyLabel.hidden = k !== 'llm_batch';
    paramsInput.placeholder = k === 'primes' ? 'limit=1000000'
      : k === 'mandelbrot' ? 'width=1280 height=720 max_iter=800'
      : k === 'moe_distributed' ? 'model_id=OLMoE-1B-7B-0924-Instruct context=4096'
      : 'chunk=65536';
  }
  kindSelect.addEventListener('change', syncSubmitFields);
  syncSubmitFields();

  submitBox.appendChild(el('div', { class: 'grid-submit-row' },
    el('label', { class: 'grid-field' }, el('span', { class: 'grid-field-label', text: 'Kind' }), kindSelect),
    el('label', { class: 'grid-field grid-field-grow' }, el('span', { class: 'grid-field-label', text: 'Params' }), paramsInput),
    el('label', { class: 'grid-field' }, el('span', { class: 'grid-field-label', text: 'Priority' }), priorityInput)));
  submitBox.appendChild(inputArea);
  submitBox.appendChild(el('div', { class: 'grid-submit-row' },
    verifyLabel,
    el('label', { class: 'grid-check' }, replicateChk, el('span', { text: 'replicate (2-node agreement)' })),
    submitBtn, submitStatus));

  submitBtn.addEventListener('click', async () => {
    submitBtn.disabled = true;
    submitStatus.textContent = 'Submitting…';
    try {
      const kind = kindSelect.value;
      const params = {};
      const meta = {};
      for (const pair of paramsInput.value.trim().split(/\s+/).filter(Boolean)) {
        const eq = pair.indexOf('=');
        if (eq < 0) throw new Error(`param must be KEY=value: ${pair}`);
        const k = pair.slice(0, eq);
        const v = pair.slice(eq + 1);
        if (['mode', 'model', 'format', 'model_id', 'quant', 'gguf_path', 'prompt'].includes(k)) meta[k] = v;
        else {
          const f = Number(v);
          if (!Number.isFinite(f)) throw new Error(`param ${k} must be numeric`);
          params[k] = f;
        }
      }
      if (kind === 'llm_batch' && verifyChk.checked) meta.mode = 'verify';
      const body = { kind, params, meta, replicate: replicateChk.checked, priority: Math.max(0, Math.min(10, Number(priorityInput.value) || 0)) };
      if ((kind === 'wordcount' || kind === 'llm_batch')) {
        if (!inputArea.value.trim()) throw new Error(`${kind} requires input text`);
        body.input = btoa(unescape(encodeURIComponent(inputArea.value)));
      }
      const resp = await gridFetch(`/api/grid/jobs${queryParams()}`, { method: 'POST', body, key: keyInput.value.trim() });
      submitStatus.textContent = `Submitted ${resp.job ? resp.job.id : ''}`;
      refresh();
    } catch (e) {
      submitStatus.textContent = `Error: ${e.message}`;
    } finally {
      submitBtn.disabled = false;
    }
  });

  // ----- refresh loop -----
  async function refresh() {
    if (state.busy) return;
    state.busy = true;
    try {
      const st = await gridFetch(`/api/grid/state${queryParams()}`, { key: keyInput.value.trim() });
      state.lastState = st;
      connStatus.textContent = `Connected · ${(st.nodes || []).length} nodes · ${(st.jobs || []).length} jobs`;
      connStatus.className = 'grid-conn-status grid-conn-ok';
      showNotice('', 'info');
      renderFleet(st);
      renderJobs(st);
    } catch (e) {
      connStatus.textContent = 'Coordinator unreachable';
      connStatus.className = 'grid-conn-status grid-conn-err';
      showNotice(`Coordinator unreachable — start one with \`vibe-local cluster coordinator\`. (${e.message})`, 'error');
      if (!state.lastState) {
        renderFleet(null);
        renderJobs(null);
      }
    } finally {
      state.busy = false;
    }
  }

  connectBtn.addEventListener('click', () => {
    state.coordinator = coordInput.value.trim();
    state.key = keyInput.value.trim();
    localStorage.setItem('vibe-grid-coordinator', state.coordinator);
    localStorage.setItem('vibe-grid-key', state.key);
    refresh();
  });

  function setActive(active) {
    state.active = active;
    if (active) {
      refresh();
      if (!state.timer) state.timer = setInterval(() => { if (state.active) refresh(); }, 3000);
      renderPlan(state.plan);
    } else if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
  }

  renderFleet(null);
  renderJobs(null);
  renderPlan(null);

  return {
    refresh: () => { if (state.active) refresh(); },
    update: () => {},
    setActive,
  };
}