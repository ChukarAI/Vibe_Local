package com.vibelocal.app.ui.grid

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.vibelocal.app.AppContainer
import com.vibelocal.app.VibeLocalApp
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONObject

/** One expert shard of the fleet MoE plan. */
data class GridShard(
    val shardId: String,
    val expertStart: Int,
    val expertEnd: Int,
    val nodeId: String,
    val nodeName: String,
    val sizeMb: Double,
    val role: String,
    val heat: Double,
    val status: String,
)

data class GridPlan(
    val modelId: String,
    val routedExperts: Int,
    val expertsPerToken: Int,
    val backboneNode: String,
    val estDecodeTps: Double,
    val estSpeedup: Double,
    val wifiRttMs: Double,
    val quant: String,
    val context: Int,
    val shards: List<GridShard>,
    val draftNodes: List<String>,
    val warnings: List<String>,
) {
    companion object {
        fun fromJson(o: JSONObject): GridPlan {
            val shards = mutableListOf<GridShard>()
            val arr = o.optJSONArray("shards")
            if (arr != null) for (i in 0 until arr.length()) {
                val s = arr.getJSONObject(i)
                shards.add(
                    GridShard(
                        shardId = s.optString("shard_id"),
                        expertStart = s.optInt("expert_start"),
                        expertEnd = s.optInt("expert_end"),
                        nodeId = s.optString("node_id"),
                        nodeName = s.optString("node_name"),
                        sizeMb = s.optDouble("size_mb", 0.0),
                        role = s.optString("role"),
                        heat = s.optDouble("heat", 0.0),
                        status = s.optString("status"),
                    )
                )
            }
            val drafts = mutableListOf<String>()
            val da = o.optJSONArray("draft_nodes")
            if (da != null) for (i in 0 until da.length()) drafts.add(da.optString(i))
            val warnings = mutableListOf<String>()
            val wa = o.optJSONArray("warnings")
            if (wa != null) for (i in 0 until wa.length()) warnings.add(wa.optString(i))
            return GridPlan(
                modelId = o.optString("model_id"),
                routedExperts = o.optInt("routed_experts", 64),
                expertsPerToken = o.optInt("experts_per_token", 8),
                backboneNode = o.optString("backbone_node"),
                estDecodeTps = o.optDouble("est_decode_tps", 0.0),
                estSpeedup = o.optDouble("est_speedup_vs_single_node", 0.0),
                wifiRttMs = o.optDouble("wifi_rtt_ms", 0.0),
                quant = o.optString("quant"),
                context = o.optInt("context", 4096),
                shards = shards,
                draftNodes = drafts,
                warnings = warnings,
            )
        }
    }
}

data class GridNode(
    val id: String,
    val name: String,
    val os: String,
    val arch: String,
    val cores: Int,
    val memMb: Long,
    val batteryPct: Int,
    val onBattery: Boolean,
    val tempC: Double,
    val hasLlm: Boolean,
    val online: Boolean,
    val paused: String,
    val speedFactor: Double,
    val done: Long,
    val failed: Long,
)

data class GridJob(
    val id: String,
    val kind: String,
    val status: String,
    val tasksDone: Int,
    val tasksTotal: Int,
    val etaSeconds: Long,
    val waiting: String,
    val error: String,
    val replicate: Boolean,
)

data class GridUiState(
    val coordinator: String = "",
    val key: String = "",
    val workers: Int = 2,
    /** Explicit capability labels to advertise; empty = auto-detect. */
    val capabilities: Set<String> = emptySet(),
    /** Coordinator base URLs found by the last LAN discovery run. */
    val discoveredUrls: List<String> = emptyList(),
    val discovering: Boolean = false,
    /** Set after a discovery run that found nobody. */
    val discoverNote: String? = null,
    val agentRunning: Boolean = false,
    val agentNodeId: String? = null,
    val coordinatorRunning: Boolean = false,
    val coordinatorPort: Int? = null,
    val nodes: List<GridNode> = emptyList(),
    val jobs: List<GridJob> = emptyList(),
    val plan: GridPlan? = null,
    val planModel: String = MOE_MODELS[0],
    val planContext: Int = 4096,
    val connected: Boolean = false,
    val busy: Boolean = false,
    val error: String? = null,
    val submitStatus: String? = null,
) {
    companion object {
        /** Capability labels the worker can advertise (Wave 4 Task 2). */
        val CAPABILITY_OPTIONS = listOf("llm", "vision", "embed_rag", "stt", "tts")

        val MOE_MODELS = listOf(
            "OLMoE-1B-7B-0924-Instruct",
            "Granite-3.1-3B-A800M-Instruct",
            "Qwen1.5-MoE-A2.7B-Chat",
            "DeepSeek-V2-Lite-16B",
        )
    }
}

class GridViewModel(private val container: AppContainer) : ViewModel() {

    private val _state = MutableStateFlow(
        GridUiState(
            coordinator = container.prefs.gridCoordinator,
            key = container.prefs.gridKey,
            workers = container.prefs.gridWorkers,
        )
    )
    val state: StateFlow<GridUiState> = _state.asStateFlow()

    private var pollJob: Job? = null

    fun setCoordinator(url: String) {
        container.prefs.gridCoordinator = url.trim()
        _state.update { it.copy(coordinator = url.trim()) }
    }

    fun setKey(key: String) {
        container.prefs.gridKey = key.trim()
        _state.update { it.copy(key = key.trim()) }
    }

    fun setWorkers(n: Int) {
        container.prefs.gridWorkers = n
        _state.update { it.copy(workers = n.coerceIn(1, 16)) }
    }

    fun toggleCapability(label: String) {
        _state.update {
            it.copy(
                capabilities = if (label in it.capabilities) it.capabilities - label else it.capabilities + label,
            )
        }
    }

    /** UDP-broadcast for LAN coordinators (VIBE_GRID_DISCOVER_V1, :20132). */
    fun discoverCoordinators() {
        viewModelScope.launch {
            _state.update { it.copy(discovering = true, discoverNote = null) }
            container.repository.gridDiscover()
                .onSuccess { urls ->
                    _state.update {
                        it.copy(
                            discovering = false,
                            discoveredUrls = urls,
                            discoverNote = if (urls.isEmpty()) "none" else null,
                        )
                    }
                    // A single answer is unambiguous: fill the field.
                    if (urls.size == 1) setCoordinator(urls[0])
                }
                .onFailure { e ->
                    _state.update { it.copy(discovering = false, error = e.message) }
                }
        }
    }

    fun setPlanModel(model: String) = _state.update { it.copy(planModel = model) }

    fun setPlanContext(ctx: Int) = _state.update { it.copy(planContext = ctx.coerceIn(512, 131072)) }

    /** Called when the tab becomes visible: start polling fleet state. */
    fun onVisible() {
        refresh()
        if (pollJob == null) {
            pollJob = viewModelScope.launch {
                while (isActive) {
                    delay(5000)
                    refresh(silent = true)
                }
            }
        }
    }

    fun onHidden() {
        pollJob?.cancel()
        pollJob = null
    }

    fun refresh(silent: Boolean = false) {
        viewModelScope.launch {
            if (!silent) _state.update { it.copy(busy = true) }
            container.repository.gridStatus()
                .onSuccess { data ->
                    val st = data.optJSONObject("state") ?: JSONObject()
                    val nodes = mutableListOf<GridNode>()
                    val na = st.optJSONArray("nodes")
                    if (na != null) for (i in 0 until na.length()) {
                        val n = na.getJSONObject(i)
                        val c = n.optJSONObject("caps") ?: JSONObject()
                        nodes.add(
                            GridNode(
                                id = n.optString("id"),
                                name = n.optString("name"),
                                os = c.optString("os"),
                                arch = c.optString("arch"),
                                cores = c.optInt("cores"),
                                memMb = c.optLong("mem_mb"),
                                batteryPct = c.optInt("battery_pct", -1),
                                onBattery = c.optBoolean("on_battery"),
                                tempC = c.optDouble("temp_c", -1.0),
                                hasLlm = c.optBoolean("has_local_llm"),
                                online = n.optBoolean("online"),
                                paused = n.optString("paused"),
                                speedFactor = n.optDouble("speed_factor", 0.0),
                                done = n.optLong("done"),
                                failed = n.optLong("failed"),
                            )
                        )
                    }
                    val jobs = mutableListOf<GridJob>()
                    val ja = st.optJSONArray("jobs")
                    if (ja != null) for (i in 0 until ja.length()) {
                        val j = ja.getJSONObject(i)
                        jobs.add(
                            GridJob(
                                id = j.optString("id"),
                                kind = j.optString("kind"),
                                status = j.optString("status"),
                                tasksDone = j.optInt("tasks_done"),
                                tasksTotal = j.optInt("tasks_total"),
                                etaSeconds = j.optLong("eta_seconds"),
                                waiting = j.optString("waiting"),
                                error = j.optString("error"),
                                replicate = j.optBoolean("replicate"),
                            )
                        )
                    }
                    _state.update {
                        it.copy(nodes = nodes, jobs = jobs, connected = true, busy = false, error = null)
                    }
                }
                .onFailure { e ->
                    _state.update {
                        it.copy(connected = false, busy = false, error = if (silent) it.error else e.message)
                    }
                }
        }
    }

    fun startAgent() {
        viewModelScope.launch {
            _state.update { it.copy(busy = true, error = null) }
            container.repository.gridAgentStart(_state.value.workers, _state.value.capabilities.toList())
                .onSuccess { data ->
                    _state.update {
                        it.copy(agentRunning = true, agentNodeId = data.optString("node_id"), busy = false)
                    }
                    refresh()
                }
                .onFailure { e -> _state.update { it.copy(busy = false, error = e.message) } }
        }
    }

    fun stopAgent() {
        viewModelScope.launch {
            container.repository.gridAgentStop()
            _state.update { it.copy(agentRunning = false, agentNodeId = null) }
            refresh()
        }
    }

    fun startCoordinator(addr: String, protectArtifacts: Boolean) {
        viewModelScope.launch {
            _state.update { it.copy(busy = true, error = null) }
            container.repository.gridCoordinatorStart(addr, protectArtifacts)
                .onSuccess { data ->
                    _state.update {
                        it.copy(coordinatorRunning = true, coordinatorPort = data.optInt("port"), busy = false)
                    }
                }
                .onFailure { e -> _state.update { it.copy(busy = false, error = e.message) } }
        }
    }

    fun stopCoordinator() {
        viewModelScope.launch {
            container.repository.gridCoordinatorStop()
            _state.update { it.copy(coordinatorRunning = false, coordinatorPort = null) }
        }
    }

    fun planMoe() {
        viewModelScope.launch {
            _state.update { it.copy(busy = true, error = null) }
            val s = _state.value
            container.repository.gridPlanMoe(s.planModel, s.planContext)
                .onSuccess { data ->
                    _state.update { it.copy(plan = GridPlan.fromJson(data), busy = false) }
                }
                .onFailure { e -> _state.update { it.copy(busy = false, error = e.message) } }
        }
    }

    fun submitJob(kind: String, paramsText: String, input: String, verify: Boolean, replicate: Boolean, priority: Int) {
        viewModelScope.launch {
            _state.update { it.copy(busy = true, submitStatus = null, error = null) }
            try {
                val params = JSONObject()
                val meta = JSONObject()
                for (pair in paramsText.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }) {
                    val eq = pair.indexOf('=')
                    require(eq > 0) { "param must be KEY=value: $pair" }
                    val k = pair.substring(0, eq)
                    val v = pair.substring(eq + 1)
                    if (k in listOf("mode", "model", "format", "model_id", "quant", "gguf_path", "prompt")) {
                        meta.put(k, v)
                    } else {
                        params.put(k, v.toDouble())
                    }
                }
                if (kind == "llm_batch" && verify) meta.put("mode", "verify")
                val body = JSONObject()
                    .put("kind", kind)
                    .put("params", params)
                    .put("meta", meta)
                    .put("replicate", replicate)
                    .put("priority", priority.coerceIn(0, 10))
                if (kind == "wordcount" || kind == "llm_batch") {
                    require(input.isNotBlank()) { "$kind requires input text" }
                    body.put("input", android.util.Base64.encodeToString(input.toByteArray(Charsets.UTF_8), android.util.Base64.NO_WRAP))
                }
                container.repository.gridSubmitJob(body)
                    .onSuccess { resp ->
                        val job = resp.optJSONObject("job")
                        _state.update { it.copy(busy = false, submitStatus = "Submitted ${job?.optString("id") ?: ""}") }
                        refresh()
                    }
                    .onFailure { e -> _state.update { it.copy(busy = false, error = e.message) } }
            } catch (e: Exception) {
                _state.update { it.copy(busy = false, error = e.message) }
            }
        }
    }

    fun cancelJob(jobId: String) {
        viewModelScope.launch {
            container.repository.gridCancelJob(jobId)
                .onFailure { e -> _state.update { it.copy(error = e.message) } }
            refresh()
        }
    }

    override fun onCleared() {
        onHidden()
        super.onCleared()
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app = this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as VibeLocalApp
                GridViewModel(app.container)
            }
        }
    }
}