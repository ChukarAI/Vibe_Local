package com.vibelocal.app.ui.grid

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.foundation.rememberScrollState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.vibelocal.app.R
import com.vibelocal.app.ui.common.ErrorBanner
import com.vibelocal.app.ui.common.InfoBanner
import com.vibelocal.app.ui.common.SectionCard
import com.vibelocal.app.ui.theme.AccentCyan
import com.vibelocal.app.ui.theme.AccentViolet
import com.vibelocal.app.ui.theme.DangerRed
import com.vibelocal.app.ui.theme.OkGreen
import com.vibelocal.app.ui.theme.TextSecondary
import com.vibelocal.app.ui.theme.WarnAmber

private val shardPalette = listOf(
    Color(0xFF7C5CFF), Color(0xFF22D3EE), Color(0xFF3DDC97), Color(0xFFFFB454),
    Color(0xFFF472B6), Color(0xFF60A5FA), Color(0xFFFF5470), Color(0xFFA3E635),
)

@Composable
fun GridScreen(vm: GridViewModel) {
    val state by vm.state.collectAsStateWithLifecycle()

    DisposableEffect(Unit) {
        vm.onVisible()
        onDispose { vm.onHidden() }
    }

    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            Text(
                stringResource(R.string.grid_title),
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onBackground,
            )
            Text(
                stringResource(R.string.grid_subtitle),
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
            )
        }

        state.error?.let { item { ErrorBanner(it) } }

        item { ConnectionCard(state, vm) }
        item { WorkerCard(state, vm) }
        item { FleetCard(state) }
        item { PlannerCard(state, vm) }
        item { JobsCard(state, vm) }
        item { SubmitCard(state, vm) }
        item { CoordinatorCard(state, vm) }
        item {
            Text(
                stringResource(R.string.grid_footer),
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
                modifier = Modifier.padding(bottom = 16.dp),
            )
        }
    }
}

@Composable
private fun ConnectionCard(state: GridUiState, vm: GridViewModel) {
    // Keyed on state.coordinator so LAN discovery can fill the field.
    var url by rememberSaveable(state.coordinator) { mutableStateOf(state.coordinator) }
    var key by rememberSaveable { mutableStateOf(state.key) }
    SectionCard(title = stringResource(R.string.grid_connection)) {
        OutlinedTextField(
            value = url,
            onValueChange = { url = it },
            label = { Text(stringResource(R.string.grid_coordinator_url)) },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = key,
            onValueChange = { key = it },
            label = { Text(stringResource(R.string.grid_fleet_key)) },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            Button(onClick = {
                vm.setCoordinator(url)
                vm.setKey(key)
                vm.refresh()
            }) { Text(stringResource(R.string.grid_connect)) }
            Text(
                if (state.connected) stringResource(R.string.grid_connected) else stringResource(R.string.grid_disconnected),
                color = if (state.connected) OkGreen else DangerRed,
                fontSize = 12.sp,
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedButton(
                onClick = { vm.discoverCoordinators() },
                enabled = !state.discovering,
            ) { Text(stringResource(R.string.grid_discover)) }
            if (state.discovering) {
                Text(
                    stringResource(R.string.grid_discovering),
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
            }
        }
        if (state.discoverNote == "none") {
            Text(
                stringResource(R.string.grid_discovered_none),
                style = MaterialTheme.typography.bodySmall,
                color = WarnAmber,
            )
        }
        state.discoveredUrls.forEach { found ->
            OutlinedButton(
                onClick = {
                    vm.setCoordinator(found)
                    vm.setKey(key)
                    vm.refresh()
                },
                modifier = Modifier.fillMaxWidth(),
            ) { Text(found) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun WorkerCard(state: GridUiState, vm: GridViewModel) {
    SectionCard(title = stringResource(R.string.grid_worker)) {
        Text(
            stringResource(R.string.grid_worker_help),
            style = MaterialTheme.typography.bodySmall,
            color = TextSecondary,
        )
        Text(
            stringResource(R.string.grid_capabilities),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Row(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            modifier = Modifier.horizontalScroll(rememberScrollState()),
        ) {
            GridUiState.CAPABILITY_OPTIONS.forEach { label ->
                FilterChip(
                    selected = label in state.capabilities,
                    onClick = { vm.toggleCapability(label) },
                    label = { Text(label, fontSize = 11.sp) },
                )
            }
        }
        Text(
            stringResource(R.string.grid_capabilities_help),
            style = MaterialTheme.typography.bodySmall,
            color = TextSecondary,
        )
        Text(
            stringResource(R.string.grid_workers_label, state.workers),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Slider(
            value = state.workers.toFloat(),
            onValueChange = { vm.setWorkers(it.toInt()) },
            valueRange = 1f..16f,
            steps = 14,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = { vm.startAgent() },
                enabled = !state.agentRunning && !state.busy,
            ) { Text(stringResource(R.string.grid_start_worker)) }
            OutlinedButton(
                onClick = { vm.stopAgent() },
                enabled = state.agentRunning,
            ) { Text(stringResource(R.string.grid_stop_worker)) }
        }
        if (state.agentRunning) {
            InfoBanner(stringResource(R.string.grid_worker_running, state.agentNodeId ?: "?"))
        }
    }
}

@Composable
private fun FleetCard(state: GridUiState) {
    SectionCard(title = stringResource(R.string.grid_fleet)) {
        if (state.nodes.isEmpty()) {
            Text(
                stringResource(R.string.grid_no_nodes),
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
            )
        }
        state.nodes.forEach { n ->
            Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(n.name.ifEmpty { n.id }, fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.onSurface)
                    Spacer(Modifier.width(8.dp))
                    GuardBadge(
                        text = if (n.online) "online" else "offline",
                        color = if (n.online) OkGreen else TextSecondary,
                    )
                    if (n.paused.isNotEmpty()) {
                        Spacer(Modifier.width(6.dp))
                        GuardBadge(
                            text = n.paused,
                            color = if (n.paused.contains("thermal", true)) DangerRed else WarnAmber,
                        )
                    }
                    if (n.hasLlm) {
                        Spacer(Modifier.width(6.dp))
                        GuardBadge("LLM :20129", AccentViolet)
                    }
                }
                val battery = if (n.batteryPct >= 0) " · battery ${n.batteryPct}%${if (n.onBattery) " (on battery)" else ""}" else ""
                val temp = if (n.tempC >= 0) " · ${n.tempC.toInt()}°C" else ""
                Text(
                    "${n.os}/${n.arch} · ${n.cores} cores · ${n.memMb} MB$battery$temp",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
                Text(
                    "speed ×${"%.2f".format(n.speedFactor)} · done ${n.done} · failed ${n.failed}",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
            }
        }
    }
}

@Composable
private fun GuardBadge(text: String, color: Color) {
    Text(
        text,
        fontSize = 10.sp,
        color = color,
        modifier = Modifier
            .clip(RoundedCornerShape(10.dp))
            .background(color.copy(alpha = 0.14f))
            .padding(horizontal = 7.dp, vertical = 2.dp),
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PlannerCard(state: GridUiState, vm: GridViewModel) {
    var expanded by remember { mutableStateOf(false) }
    var ctxText by rememberSaveable { mutableStateOf(state.planContext.toString()) }
    SectionCard(title = stringResource(R.string.grid_planner)) {
        ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
            OutlinedTextField(
                value = state.planModel,
                onValueChange = {},
                readOnly = true,
                label = { Text(stringResource(R.string.grid_model)) },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                modifier = Modifier.menuAnchor().fillMaxWidth(),
            )
            ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                GridUiState.MOE_MODELS.forEach { m ->
                    DropdownMenuItem(
                        text = { Text(m) },
                        onClick = { vm.setPlanModel(m); expanded = false },
                    )
                }
            }
        }
        OutlinedTextField(
            value = ctxText,
            onValueChange = {
                ctxText = it
                it.toIntOrNull()?.let { c -> vm.setPlanContext(c) }
            },
            label = { Text(stringResource(R.string.grid_context)) },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Button(onClick = { vm.planMoe() }, enabled = !state.busy) {
            Text(stringResource(R.string.grid_plan_button))
        }
        state.plan?.let { plan ->
            Text(
                stringResource(
                    R.string.grid_plan_summary,
                    plan.routedExperts, plan.expertsPerToken,
                    "%.1f".format(plan.estDecodeTps), "%.2f".format(plan.estSpeedup),
                ),
                style = MaterialTheme.typography.bodySmall,
                color = AccentCyan,
            )
            // Expert shard allocation bar.
            val colorOf = remember(plan) {
                val map = mutableMapOf<String, Color>()
                plan.shards.forEach { s ->
                    map.getOrPut(s.nodeId) { shardPalette[map.size % shardPalette.size] }
                }
                map
            }
            Row(modifier = Modifier.fillMaxWidth().height(26.dp).clip(RoundedCornerShape(6.dp))) {
                plan.shards.forEach { s ->
                    val span = (s.expertEnd - s.expertStart).coerceAtLeast(0)
                    if (span > 0) {
                        Box(
                            modifier = Modifier
                                .weight(span.toFloat())
                                .height(26.dp)
                                .background(
                                    (colorOf[s.nodeId] ?: AccentViolet).copy(
                                        alpha = if (s.role == "host_paging") 0.45f else 1f
                                    )
                                )
                        )
                    }
                }
            }
            plan.shards.forEach { s ->
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 2.dp)) {
                    Box(
                        Modifier.width(10.dp).height(10.dp).clip(RoundedCornerShape(3.dp))
                            .background(colorOf[s.nodeId] ?: AccentViolet)
                    )
                    Spacer(Modifier.width(8.dp))
                    Column(Modifier.weight(1f)) {
                        Text(s.nodeName, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface)
                        Text(
                            "experts ${s.expertStart}..${s.expertEnd} · ${s.sizeMb.toInt()} MB",
                            fontSize = 11.sp, color = TextSecondary,
                        )
                    }
                    GuardBadge(
                        s.role,
                        when (s.role) {
                            "backbone" -> AccentViolet
                            "host_paging" -> WarnAmber
                            else -> AccentCyan
                        },
                    )
                }
            }
            if (plan.draftNodes.isNotEmpty()) {
                Text(
                    stringResource(R.string.grid_draft_nodes, plan.draftNodes.joinToString(", ")),
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
            }
            plan.warnings.forEach { w ->
                Text("⚠ $w", style = MaterialTheme.typography.bodySmall, color = WarnAmber)
            }
        }
    }
}

@Composable
private fun JobsCard(state: GridUiState, vm: GridViewModel) {
    SectionCard(title = stringResource(R.string.grid_jobs)) {
        if (state.jobs.isEmpty()) {
            Text(stringResource(R.string.grid_no_jobs), style = MaterialTheme.typography.bodySmall, color = TextSecondary)
        }
        state.jobs.forEach { j ->
            Column(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("${j.id} · ${j.kind}${if (j.replicate) " ×2" else ""}", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurface)
                    Spacer(Modifier.width(8.dp))
                    GuardBadge(
                        j.status,
                        when (j.status) {
                            "done" -> OkGreen
                            "failed" -> DangerRed
                            "cancelled" -> TextSecondary
                            else -> WarnAmber
                        },
                    )
                }
                LinearProgressIndicator(
                    progress = { if (j.tasksTotal > 0) j.tasksDone.toFloat() / j.tasksTotal else 0f },
                    modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp),
                )
                Text(
                    "${j.tasksDone}/${j.tasksTotal}" +
                        (if (j.etaSeconds > 0) " · ETA ${j.etaSeconds}s" else "") +
                        (if (j.waiting.isNotEmpty()) " · ${j.waiting}" else "") +
                        (if (j.error.isNotEmpty()) " · ${j.error}" else ""),
                    fontSize = 11.sp,
                    color = TextSecondary,
                )
                if (j.status == "queued" || j.status == "running") {
                    OutlinedButton(onClick = { vm.cancelJob(j.id) }) {
                        Text(stringResource(R.string.grid_cancel), fontSize = 12.sp)
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SubmitCard(state: GridUiState, vm: GridViewModel) {
    var kindExpanded by remember { mutableStateOf(false) }
    var kind by rememberSaveable { mutableStateOf("primes") }
    var params by rememberSaveable { mutableStateOf("limit=1000000") }
    var input by rememberSaveable { mutableStateOf("") }
    var verify by rememberSaveable { mutableStateOf(false) }
    var replicate by rememberSaveable { mutableStateOf(false) }
    var priority by rememberSaveable { mutableStateOf(0f) }
    val kinds = listOf("primes", "mandelbrot", "moe_distributed", "wordcount", "llm_batch")

    SectionCard(title = stringResource(R.string.grid_submit)) {
        ExposedDropdownMenuBox(expanded = kindExpanded, onExpandedChange = { kindExpanded = it }) {
            OutlinedTextField(
                value = kind,
                onValueChange = {},
                readOnly = true,
                label = { Text(stringResource(R.string.grid_kind)) },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = kindExpanded) },
                modifier = Modifier.menuAnchor().fillMaxWidth(),
            )
            ExposedDropdownMenu(expanded = kindExpanded, onDismissRequest = { kindExpanded = false }) {
                kinds.forEach { k ->
                    DropdownMenuItem(text = { Text(k) }, onClick = {
                        kind = k
                        params = when (k) {
                            "primes" -> "limit=1000000"
                            "mandelbrot" -> "width=1280 height=720 max_iter=800"
                            "moe_distributed" -> "model_id=OLMoE-1B-7B-0924-Instruct context=4096"
                            else -> ""
                        }
                        kindExpanded = false
                    })
                }
            }
        }
        OutlinedTextField(
            value = params,
            onValueChange = { params = it },
            label = { Text(stringResource(R.string.grid_params)) },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        if (kind == "wordcount" || kind == "llm_batch") {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                label = { Text(stringResource(R.string.grid_input_text)) },
                minLines = 3,
                modifier = Modifier.fillMaxWidth(),
            )
        }
        if (kind == "llm_batch") {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(checked = verify, onCheckedChange = { verify = it })
                Text(stringResource(R.string.grid_verify_mode), fontSize = 12.sp, color = TextSecondary)
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(checked = replicate, onCheckedChange = { replicate = it })
            Text(stringResource(R.string.grid_replicate), fontSize = 12.sp, color = TextSecondary)
        }
        Text(stringResource(R.string.grid_priority, priority.toInt()), fontSize = 12.sp, color = TextSecondary)
        Slider(value = priority, onValueChange = { priority = it }, valueRange = 0f..10f, steps = 9)
        Button(
            onClick = { vm.submitJob(kind, params, input, verify, replicate, priority.toInt()) },
            enabled = !state.busy,
        ) { Text(stringResource(R.string.grid_submit_button)) }
        state.submitStatus?.let { InfoBanner(it) }
    }
}

@Composable
private fun CoordinatorCard(state: GridUiState, vm: GridViewModel) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    var addr by rememberSaveable { mutableStateOf("0.0.0.0:7900") }
    var protect by rememberSaveable { mutableStateOf(true) }
    SectionCard(title = stringResource(R.string.grid_coordinator_mode)) {
        OutlinedButton(onClick = { expanded = !expanded }) {
            Text(if (expanded) stringResource(R.string.grid_hide) else stringResource(R.string.grid_show))
        }
        if (expanded) {
            Text(
                stringResource(R.string.grid_coordinator_help),
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
            )
            OutlinedTextField(
                value = addr,
                onValueChange = { addr = it },
                label = { Text(stringResource(R.string.grid_listen_addr)) },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(checked = protect, onCheckedChange = { protect = it })
                Text(stringResource(R.string.grid_protect_artifacts), fontSize = 12.sp, color = TextSecondary)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = { vm.startCoordinator(addr, protect) },
                    enabled = !state.coordinatorRunning && !state.busy,
                ) { Text(stringResource(R.string.grid_start_coordinator)) }
                OutlinedButton(
                    onClick = { vm.stopCoordinator() },
                    enabled = state.coordinatorRunning,
                ) { Text(stringResource(R.string.grid_stop_coordinator)) }
            }
            if (state.coordinatorRunning) {
                InfoBanner(stringResource(R.string.grid_coordinator_running, state.coordinatorPort ?: 0))
            }
        }
    }
}