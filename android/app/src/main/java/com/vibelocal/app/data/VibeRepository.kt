package com.vibelocal.app.data

import android.content.Context
import com.vibelocal.app.R
import com.vibelocal.app.core.BridgeResult
import com.vibelocal.app.core.ChatMessage
import com.vibelocal.app.core.HardwareInfo
import com.vibelocal.app.core.LeaderboardData
import com.vibelocal.app.core.ModelFile
import com.vibelocal.app.core.NativeLib
import com.vibelocal.app.core.Recommendations
import com.vibelocal.app.core.SafeBridge
import com.vibelocal.app.core.ServerStatus
import com.vibelocal.app.core.VibeMarkResult
import com.vibelocal.app.core.buildMessagesArray
import com.vibelocal.app.core.extractChatContent
import com.vibelocal.app.util.NetUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import org.json.JSONException
import org.json.JSONObject
import java.io.IOException

/**
 * Single source of truth for engine access.
 *
 * Transport selection:
 *  - native lib loaded  -> JNI via [SafeBridge] (loadModel / chatMessages
 *    included — model loading and multi-turn chat never leave the process);
 *  - native lib missing -> HTTP fallback against the user-entered Vibe-Link
 *    host ([remoteHost]), with the optional remote token header.
 *
 * All calls are suspend + Dispatchers.IO and never throw: failures surface as
 * [Result.failure] so ViewModels can render proper error states.
 */
class VibeRepository(
    private val context: Context,
    private val prefs: AppPrefs,
    val dataDirs: DataDirManager,
    private val http: VibeHttpClient,
) {

    val nativeAvailable: Boolean get() = NativeLib.isAvailable
    val nativeLoadError: String? get() = NativeLib.loadError

    private val _remoteHost = MutableStateFlow(prefs.remoteHost)
    val remoteHost: StateFlow<String> = _remoteHost.asStateFlow()

    fun setRemoteHost(raw: String): Boolean {
        val normalized = NetUtils.normalizeBaseUrl(raw) ?: return false
        prefs.remoteHost = normalized
        _remoteHost.value = normalized
        return true
    }

    fun setRemoteToken(token: String) {
        prefs.remoteToken = token.trim()
    }

    val remoteToken: String get() = prefs.remoteToken

    /** Base URL + token for HTTP calls in the current transport mode. */
    private fun prepareHttp(): String {
        return if (nativeAvailable) {
            http.token = prefs.lanToken
            LOCAL_BASE
        } else {
            http.token = prefs.remoteToken.takeIf { it.isNotBlank() }
            prefs.remoteHost
        }
    }

    private fun <T> bridgeToResult(r: BridgeResult, parse: (Any?) -> T): Result<T> = when (r) {
        is BridgeResult.Ok -> runCatching { parse(r.data) }
        is BridgeResult.Err -> Result.failure(IOException(r.message))
    }

    // ------------------------------------------------------------- discovery

    suspend fun scanHardware(): Result<HardwareInfo> = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            bridgeToResult(SafeBridge.scanHardware()) { HardwareInfo.fromJson(it as JSONObject) }
        } else {
            http.hardware(prepareHttp())
        }
    }

    suspend fun recommendations(): Result<Recommendations> = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            bridgeToResult(SafeBridge.recommend()) { Recommendations.fromJson(it as JSONObject) }
        } else {
            http.recommendations(prepareHttp())
        }
    }

    suspend fun leaderboard(profile: String): Result<LeaderboardData> = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            bridgeToResult(SafeBridge.leaderboard(profile)) { LeaderboardData.fromJson(it as JSONObject) }
        } else {
            http.leaderboard(prepareHttp(), profile)
        }
    }

    suspend fun inspectGguf(path: String): Result<JSONObject> = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            bridgeToResult(SafeBridge.inspectGguf(path)) { it as JSONObject }
        } else {
            Result.failure(IOException("inspectGguf requires the native engine"))
        }
    }

    // ---------------------------------------------------------------- server

    suspend fun status(): Result<ServerStatus> = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            bridgeToResult(SafeBridge.serverStatus()) { data ->
                when (data) {
                    is JSONObject -> ServerStatus.fromJson(data)
                    else -> ServerStatus(
                        ok = false, loadedModel = null, backend = null, engine = null,
                        idleTimeoutSecs = 0, thermal = null, lan = false, port = 0,
                    )
                }
            }
        } else {
            http.status(prepareHttp())
        }
    }

    /**
     * Makes sure the loopback Vibe-Link server is up in native mode
     * (starting it with lan=false when needed). Used by the HTTP fallback
     * path of [loadModel] and by features that serve LAN clients; native
     * model loading itself goes through JNI and does not need the server.
     */
    suspend fun ensureLocalServer(): Result<Unit> = withContext(Dispatchers.IO) {
        if (!nativeAvailable) return@withContext Result.success(Unit)
        val st = status().getOrNull()
        if (st != null && st.ok && st.port > 0) return@withContext Result.success(Unit)
        bridgeToResult(
            SafeBridge.startServer(DEFAULT_PORT, false, "", dataDirs.dataDirPath),
        ) { }
    }

    suspend fun startServer(port: Int, lan: Boolean, token: String): Result<Unit> =
        withContext(Dispatchers.IO) {
            if (!nativeAvailable) {
                return@withContext Result.failure(IOException(context.getString(R.string.lan_native_required)))
            }
            bridgeToResult(
                SafeBridge.startServer(port, lan, token, dataDirs.dataDirPath),
            ) { }
        }

    suspend fun stopServer(): Result<Unit> = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            bridgeToResult(SafeBridge.stopServer()) { }
        } else {
            Result.success(Unit)
        }
    }

    // ----------------------------------------------------------------- model

    /**
     * Native mode: direct JNI `loadModel` with a LoadRequest JSON
     * (path / backend / ngl / ctx plus optional draft_model_path,
     * mmproj_path and embedding). HTTP fallback (no native lib):
     * POST /api/load against the configured Vibe-Link host.
     */
    suspend fun loadModel(
        path: String,
        backend: String?,
        ngl: Int,
        ctx: Int,
        draftModelPath: String? = null,
        mmprojPath: String? = null,
        embedding: Boolean = false,
    ): Result<Unit> =
        withContext(Dispatchers.IO) {
            if (nativeAvailable) {
                val config = JSONObject()
                    .put("path", path)
                    .put("ngl", ngl)
                    .put("ctx", ctx)
                if (!backend.isNullOrBlank()) config.put("backend", backend)
                if (!draftModelPath.isNullOrBlank()) config.put("draft_model_path", draftModelPath)
                if (!mmprojPath.isNullOrBlank()) config.put("mmproj_path", mmprojPath)
                if (embedding) config.put("embedding", true)
                return@withContext bridgeToResult(SafeBridge.loadModel(config.toString())) { }
            }
            val base = prepareHttp()
            val ensured = ensureLocalServer()
            if (ensured.isFailure) {
                return@withContext Result.failure(ensured.exceptionOrNull() ?: IOException("Server start failed"))
            }
            http.load(base, path, backend, ngl, ctx)
        }

    suspend fun unloadModel(): Result<Unit> = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            bridgeToResult(SafeBridge.unloadModel()) { }
        } else {
            http.unload(prepareHttp())
        }
    }

    fun listModels(): List<ModelFile> = dataDirs.listModels()

    // -------------------------------------------------------------- vibemark

    suspend fun runVibeMark(path: String, budgetSecs: Int): Result<VibeMarkResult> =
        withContext(Dispatchers.IO) {
            if (nativeAvailable) {
                bridgeToResult(SafeBridge.runVibeMark(path, budgetSecs)) {
                    VibeMarkResult.fromJson(it as JSONObject)
                }
            } else {
                http.vibemark(prepareHttp(), path, budgetSecs)
            }
        }

    suspend fun submitLeaderboard(entryJson: String): Result<Unit> = withContext(Dispatchers.IO) {
        if (!nativeAvailable) {
            Result.failure(IOException(context.getString(R.string.vm_submit_unavailable_http)))
        } else {
            bridgeToResult(SafeBridge.submitLeaderboard(entryJson)) { }
        }
    }

    // ------------------------------------------------------------------ chat

    /**
     * Both modes send the structured OpenAI messages array built from
     * [history]: native mode via JNI `chatMessages`, HTTP mode via
     * POST /v1/chat/completions with the extra strip_think flag.
     * [history] must already include the newest user message. [prompt] is
     * the latest user text, kept for call-site compatibility.
     */
    suspend fun chat(prompt: String, history: List<ChatMessage>, stripThink: Boolean): Result<String> =
        withContext(Dispatchers.IO) {
            if (nativeAvailable) {
                bridgeToResult(
                    SafeBridge.chatMessages(buildMessagesArray(history).toString(), stripThink),
                ) { data ->
                    extractChatContent(data) ?: throw JSONException("No content in chat response")
                }
            } else {
                http.chat(prepareHttp(), buildMessagesArray(history), model = null, stripThink = stripThink)
            }
        }

    // ---------------------------------------------------------------- guards

    /** Best-effort: feeds device temperature into the native governor. */
    suspend fun setThermalReading(celsius: Double) {
        if (nativeAvailable) {
            withContext(Dispatchers.IO) { SafeBridge.setThermalReading(celsius) }
        }
    }

    /** Best-effort: keeps the native idle timer in sync with the UI setting. */
    suspend fun setIdleTimeout(seconds: Int) {
        if (nativeAvailable) {
            withContext(Dispatchers.IO) { SafeBridge.setIdleTimeout(seconds) }
        }
    }

    // ------------------------------------------------------------- vibe grid

    /**
     * Start the pull-based worker agent (native only; outward connections).
     * [capabilities] are the explicit labels to advertise (llm, vision,
     * embed_rag, stt, tts, ...); empty = auto-detect on the Rust side.
     */
    suspend fun gridAgentStart(workers: Int, capabilities: List<String> = emptyList()): Result<JSONObject> = withContext(Dispatchers.IO) {
        if (!nativeAvailable) {
            return@withContext Result.failure(IOException("Grid worker requires the native engine"))
        }
        val dir = dataDirs.dataDirPath
        bridgeToResult(
            SafeBridge.clusterAgentStart(
                prefs.gridCoordinator, prefs.gridKey, workers,
                android.os.Build.MODEL ?: "android", dir,
                capabilities.joinToString(","),
            )
        ) { it as JSONObject }
    }

    /**
     * Discover Vibe Grid coordinators on the LAN via the
     * VIBE_GRID_DISCOVER_V1 UDP broadcast (:20132). Returns base URLs
     * ("http://host:port") ready to paste into the coordinator field;
     * an empty list means nobody answered within [timeoutMs].
     */
    suspend fun gridDiscover(timeoutMs: Long = 1500): Result<List<String>> = withContext(Dispatchers.IO) {
        if (!nativeAvailable) {
            return@withContext Result.failure(IOException("Grid discovery requires the native engine"))
        }
        bridgeToResult(SafeBridge.clusterDiscoverJson(timeoutMs)) { data ->
            val urls = mutableListOf<String>()
            val arr = data as? org.json.JSONArray
            if (arr != null) for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue
                val host = o.optString("host")
                val port = o.optInt("coordinator_port", 7900)
                if (host.isNotBlank()) urls.add("http://$host:$port")
            }
            urls.distinct()
        }
    }

    suspend fun gridAgentStop(): Result<Unit> = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            bridgeToResult(SafeBridge.clusterAgentStop()) { Unit }
        } else {
            Result.failure(IOException("Grid worker requires the native engine"))
        }
    }

    /** Fleet health + state; HTTP fallback talks to the coordinator directly. */
    suspend fun gridStatus(): Result<JSONObject> = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            bridgeToResult(SafeBridge.clusterStatusJson(prefs.gridCoordinator, prefs.gridKey)) { it as JSONObject }
        } else {
            gridHttpGet("/healthz").mapCatching { health ->
                val state = gridHttpGet("/api/v1/state").getOrThrow()
                JSONObject().put("health", health).put("state", state)
            }
        }
    }

    /** MoE expert shard plan across the live fleet. */
    suspend fun gridPlanMoe(modelId: String, context: Int): Result<JSONObject> = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            bridgeToResult(SafeBridge.clusterPlanMoeJson(modelId, context, prefs.gridCoordinator, prefs.gridKey)) { it as JSONObject }
        } else {
            // Without native, plan against the local device only via the
            // Vibe-Link proxy if reachable; otherwise report the limitation.
            Result.failure(IOException("MoE planning requires the native engine"))
        }
    }

    suspend fun gridSubmitJob(body: JSONObject): Result<JSONObject> = withContext(Dispatchers.IO) {
        gridHttpPost("/api/v1/jobs", body).mapCatching { it as JSONObject }
    }

    suspend fun gridCancelJob(jobId: String): Result<Unit> = withContext(Dispatchers.IO) {
        gridHttpPost("/api/v1/jobs/$jobId/cancel", JSONObject()).map { Unit }
    }

    suspend fun gridCoordinatorStart(addr: String, protectArtifacts: Boolean): Result<JSONObject> =
        withContext(Dispatchers.IO) {
            if (!nativeAvailable) {
                return@withContext Result.failure(IOException("Coordinator requires the native engine"))
            }
            val statePath = java.io.File(dataDirs.dataDirPath, "grid-state.json.gz").absolutePath
            bridgeToResult(SafeBridge.clusterCoordinatorStart(addr, prefs.gridKey, protectArtifacts, statePath)) { it as JSONObject }
        }

    suspend fun gridCoordinatorStop(): Result<Unit> = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            bridgeToResult(SafeBridge.clusterCoordinatorStop()) { Unit }
        } else {
            Result.failure(IOException("Coordinator requires the native engine"))
        }
    }

    private suspend fun gridHttpGet(path: String): Result<Any> {
        val saved = http.token
        http.token = prefs.gridKey
        try {
            return http.getJson(prefs.gridCoordinator, path)
        } finally {
            http.token = saved
        }
    }

    private suspend fun gridHttpPost(path: String, body: JSONObject): Result<Any> {
        val saved = http.token
        http.token = prefs.gridKey
        try {
            return http.postJson(prefs.gridCoordinator, path, body)
        } finally {
            http.token = saved
        }
    }

    companion object {
        const val DEFAULT_PORT = 20129
        const val LOCAL_BASE = "http://127.0.0.1:$DEFAULT_PORT"
    }
}