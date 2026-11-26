package com.vibelocal.app.core

import org.json.JSONException
import org.json.JSONObject

/**
 * Safe wrapper around [VibeLocalBridge]. Every external function is invoked
 * through [invoke], which:
 *  - refuses to call into JNI when the native library failed to load,
 *  - catches UnsatisfiedLinkError (late binding failures),
 *  - catches JSON parse errors and any other throwable,
 *  - unwraps the {"ok":..,"data":..,"error":..} envelope into [BridgeResult].
 */
object SafeBridge {

    fun scanHardware(): BridgeResult = invoke { VibeLocalBridge.scanHardware() }

    fun recommend(): BridgeResult = invoke { VibeLocalBridge.recommend() }

    fun inspectGguf(path: String): BridgeResult = invoke { VibeLocalBridge.inspectGguf(path) }

    fun runVibeMark(path: String, budgetSecs: Int): BridgeResult =
        invoke { VibeLocalBridge.runVibeMark(path, budgetSecs) }

    fun leaderboard(profile: String): BridgeResult = invoke { VibeLocalBridge.leaderboard(profile) }

    fun submitLeaderboard(entryJson: String): BridgeResult =
        invoke { VibeLocalBridge.submitLeaderboard(entryJson) }

    fun startServer(port: Int, lan: Boolean, token: String, dataDir: String): BridgeResult =
        invoke { VibeLocalBridge.startServer(port, lan, token, dataDir) }

    fun stopServer(): BridgeResult = invoke { VibeLocalBridge.stopServer() }

    fun serverStatus(): BridgeResult = invoke { VibeLocalBridge.serverStatus() }

    fun chat(prompt: String, stripThink: Boolean): BridgeResult =
        invoke { VibeLocalBridge.chat(prompt, stripThink) }

    fun loadModel(configJson: String): BridgeResult =
        invoke { VibeLocalBridge.loadModel(configJson) }

    fun chatMessages(messagesJson: String, stripThink: Boolean): BridgeResult =
        invoke { VibeLocalBridge.chatMessages(messagesJson, stripThink) }

    fun unloadModel(): BridgeResult = invoke { VibeLocalBridge.unloadModel() }

    fun setThermalReading(celsius: Double): BridgeResult =
        invoke { VibeLocalBridge.setThermalReading(celsius) }

    fun setIdleTimeout(seconds: Int): BridgeResult =
        invoke { VibeLocalBridge.setIdleTimeout(seconds) }

    fun clusterAgentStart(coordinatorUrl: String, key: String, workers: Int, name: String, dataDir: String, capabilities: String): BridgeResult =
        invoke { VibeLocalBridge.clusterAgentStart(coordinatorUrl, key, workers, name, dataDir, capabilities) }

    fun clusterDiscoverJson(timeoutMs: Long): BridgeResult =
        invoke { VibeLocalBridge.clusterDiscoverJson(timeoutMs) }

    fun clusterAgentStop(): BridgeResult = invoke { VibeLocalBridge.clusterAgentStop() }

    fun clusterStatusJson(coordinatorUrl: String, key: String): BridgeResult =
        invoke { VibeLocalBridge.clusterStatusJson(coordinatorUrl, key) }

    fun clusterPlanMoeJson(modelId: String, context: Int, coordinatorUrl: String, key: String): BridgeResult =
        invoke { VibeLocalBridge.clusterPlanMoeJson(modelId, context, coordinatorUrl, key) }

    fun clusterCoordinatorStart(addr: String, key: String, protectArtifacts: Boolean, statePath: String): BridgeResult =
        invoke { VibeLocalBridge.clusterCoordinatorStart(addr, key, protectArtifacts, statePath) }

    fun clusterCoordinatorStop(): BridgeResult = invoke { VibeLocalBridge.clusterCoordinatorStop() }

    private fun invoke(call: () -> String): BridgeResult {
        if (!NativeLib.isAvailable) {
            return BridgeResult.Err(
                "Native library not loaded: ${NativeLib.loadError ?: "unknown reason"}"
            )
        }
        return try {
            val raw = call()
            val envelope = JSONObject(raw)
            if (envelope.optBoolean("ok", false)) {
                BridgeResult.Ok(envelope.opt("data"))
            } else {
                BridgeResult.Err(envelope.optString("error", "Native call reported ok=false"))
            }
        } catch (e: UnsatisfiedLinkError) {
            BridgeResult.Err("JNI symbol missing: ${e.message}")
        } catch (e: JSONException) {
            BridgeResult.Err("Malformed native JSON: ${e.message}")
        } catch (t: Throwable) {
            BridgeResult.Err("Native call failed: ${t.message ?: t.javaClass.simpleName}")
        }
    }
}