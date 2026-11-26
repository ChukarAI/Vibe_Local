package com.vibelocal.app.core

/**
 * Raw JNI surface of the Vibe Local engine.
 *
 * Contract (must match the Rust `vibe-local-jni` crate exactly):
 * every function returns a JSON string of shape
 *   {"ok":true,"data":...}   or   {"ok":false,"error":"..."}
 *
 * These are instance methods on the Kotlin `object` singleton (INSTANCE);
 * the Rust side binds them via JNI RegisterNatives or the
 * Java_com_vibelocal_app_core_VibeLocalBridge_<name> symbol convention.
 *
 * NEVER call these directly from UI code — go through [SafeBridge], which
 * catches UnsatisfiedLinkError / JSON errors and wraps results in [BridgeResult].
 */
object VibeLocalBridge {

    init {
        // Trigger the one-time System.loadLibrary attempt. Failure is tolerated:
        // SafeBridge checks NativeLib.isAvailable before every call.
        NativeLib.ensureLoaded()
    }

    external fun scanHardware(): String

    external fun recommend(): String

    external fun inspectGguf(path: String): String

    external fun runVibeMark(path: String, budgetSecs: Int): String

    external fun leaderboard(profile: String): String

    external fun submitLeaderboard(entryJson: String): String

    external fun startServer(port: Int, lan: Boolean, token: String, dataDir: String): String

    external fun stopServer(): String

    external fun serverStatus(): String

    external fun chat(prompt: String, stripThink: Boolean): String

    /** LoadRequest JSON: path required; ctx 0/omitted auto-sizes; optional
     *  draft_model_path / draft_ngl / draft_max for speculative decoding,
     *  mmproj_path for vision models, embedding=true for embedding mode. */
    external fun loadModel(configJson: String): String

    /** JSON array of OpenAI-style {"role","content"} messages (multi-turn). */
    external fun chatMessages(messagesJson: String, stripThink: Boolean): String

    external fun unloadModel(): String

    external fun setThermalReading(celsius: Double): String

    external fun setIdleTimeout(seconds: Int): String

    // --- Vibe Grid cluster (compost v0.7.0 wire-compatible) ---

    /** [capabilities]: comma-separated labels ("llm,embed_rag") or a JSON
     *  array; empty string = auto-detect on the Rust side. */
    external fun clusterAgentStart(coordinatorUrl: String, key: String, workers: Int, name: String, dataDir: String, capabilities: String): String

    /** Broadcasts VIBE_GRID_DISCOVER_V1 on UDP :20132 and returns the
     *  discovered coordinators as a JSON array in the data envelope. */
    external fun clusterDiscoverJson(timeoutMs: Long): String

    external fun clusterAgentStop(): String

    external fun clusterStatusJson(coordinatorUrl: String, key: String): String

    external fun clusterPlanMoeJson(modelId: String, context: Int, coordinatorUrl: String, key: String): String

    external fun clusterCoordinatorStart(addr: String, key: String, protectArtifacts: Boolean, statePath: String): String

    external fun clusterCoordinatorStop(): String
}