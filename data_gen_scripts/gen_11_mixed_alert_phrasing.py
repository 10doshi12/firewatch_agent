"""
gen_11_mixed_alert_phrasing.py — Mixed Batch: Alert Phrasing Style Variation

Script: gen_11_mixed_alert_phrasing.py
Batch: 010 (script_num = 11, batch = 010)
Primary axes: alert_phrasing + metric_value + noise_injection
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-11
Bootstrap: CONTEXT-BOOTSTRAP.md
"""

import random
from typing import Any

HEALTHY_ERROR_RATE = 0.03


def _derive_status(e: float) -> str:
    if e >= 0.5: return "critical"
    if e >= 0.2: return "degraded"
    return "healthy"


def _calculate_budget(tier: str, tick: int) -> float:
    if tier == "easy": return round(30.0 - tick * 1.5, 2)
    if tier == "medium": return round(60.0 - tick * 2.0, 2)
    return round(120.0 - tick * 3.0, 2)


# Four alert phrasing styles
def style1(svc: str, metric: str, value: float, threshold: float, severity: str) -> str:
    return f"[{severity}] {svc}: {metric}={value:.2f} exceeded threshold={threshold}"


def style2(svc: str, value: float, threshold: float, severity: str) -> str:
    return f"[{severity}] {svc} is experiencing {value:.1f}% error rate (threshold: {threshold:.0f}%). Immediate action required."


def style3(svc: str, value: float, threshold: float, severity: str, tick: int) -> str:
    abbrev = {"api-gateway": "agw", "auth-service": "auth", "payment-service": "pay"}.get(svc, svc[:4])
    return f"[{severity[:3]}] svc:{abbrev} err={value:.2f} thr={threshold:.2f} dur={tick}t"


def style4(svc: str, value: float, threshold: float, slo_rate: float) -> str:
    return f"Incident: {svc} degraded — {value:.2f} errors above {threshold:.0f} baseline (SLO burn: {slo_rate:.1f}×/hr)"


ALERT_STYLES = [style1, style2, style3, style4]


def _format_alert(style_idx: int, svc: str, value: float, threshold: float, severity: str, tick: int) -> str:
    if style_idx == 0:
        return style1(svc, "error_rate", value, threshold, severity)
    elif style_idx == 1:
        return style2(svc, value * 100, threshold * 100, severity)
    elif style_idx == 2:
        return style3(svc, value, threshold, severity, tick)
    else:
        slo_rate = value / 0.05
        return style4(svc, value, threshold, slo_rate)


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = ["task_easy_thread_deadlock", "task_easy_fail_slow_memleak", "task_easy_liveness_probe_flap",
                "task_easy_cronjob_spike", "task_easy_noisy_neighbor"]
    medium_ids = ["task_medium_cascade_memleak", "task_medium_replica_lag",
                  "task_medium_mtls_rotation", "task_medium_bg_traffic_leak"]
    hard_ids = ["task_hard_config_drift_noise", "task_hard_consensus_degradation", "task_hard_multiz_failover"]

    GOLD = {
        "task_easy_thread_deadlock": ["thread_dump(order-service)", "restart_thread_pool(order-service)", "declare_resolved"],
        "task_easy_fail_slow_memleak": ["get_metrics_detail(payment-service)", "scale_replicas(payment-service)", "declare_resolved"],
        "task_easy_liveness_probe_flap": ["get_metrics_detail(payment-processor)", "fetch_logs(payment-processor)", "adjust_probe_timing(payment-processor)", "declare_resolved"],
        "task_easy_cronjob_spike": ["get_metrics_detail(analytics-service)", "fetch_logs(analytics-service)", "scale_replicas(analytics-service)", "declare_resolved"],
        "task_easy_noisy_neighbor": [
            "get_metrics_detail(batch-processor)",
            "evict_noisy_pod(batch-processor)",
            "declare_resolved",
        ],
        "task_medium_cascade_memleak": ["trace_dependencies(checkout-service)", "get_metrics_detail(payment-service)", "scale_replicas(payment-service)", "declare_resolved"],
        "task_medium_replica_lag": ["fetch_logs(user-service)", "get_metrics_detail(user-service)", "redirect_reads_to_primary(user-service)", "force_replica_resync(user-service)", "declare_resolved"],
        "task_medium_mtls_rotation": ["inspect_mtls_status(payment-service)", "force_cert_rotation(payment-service)", "declare_resolved"],
        "task_medium_bg_traffic_leak": ["get_metrics_detail(api-gateway)", "complete_traffic_switch(api-gateway)", "declare_resolved"],
        "task_hard_config_drift_noise": ["get_metrics_detail(api-gateway)", "fetch_logs(api-gateway)", "revert_config(api-gateway)", "declare_resolved"],
        "task_hard_consensus_degradation": ["inspect_consensus_state(config-service)", "isolate_minority_nodes(config-service)", "force_leader_election(config-service)", "declare_resolved"],
        "task_hard_multiz_failover": ["get_metrics_detail(api-gateway-az-a)", "rebalance_az_traffic(api-gateway-az-a)", "scale_az_capacity(api-gateway-az-a)", "declare_resolved"],
    }

    def _build_easy_metrics(tid: str, rng: random.Random) -> dict[str, Any]:
        if tid == "task_easy_thread_deadlock":
            return {"status": "critical", "http_server_error_rate": round(rng.choice([0.88, 0.92, 0.97, 0.99, 1.00]), 4),
                    "runtime_blocked_thread_count": rng.randint(42, 60), "wait_ratio": round(rng.uniform(0.88, 1.00), 2)}
        if tid == "task_easy_fail_slow_memleak":
            mem = rng.choice([0.62, 0.65, 0.68, 0.71, 0.74, 0.77])
            return {"status": "critical", "http_server_error_rate": round(rng.choice([0.04, 0.06, 0.08, 0.10, 0.12]), 4),
                    "process_memory_utilization": mem, "memory_trend": [0.62, mem - 0.03, mem]}
        if tid == "task_easy_liveness_probe_flap":
            return {"status": "critical", "http_server_error_rate": round(rng.choice([0.65, 0.72, 0.80, 0.88, 0.95, 1.00]), 4),
                    "restart_count": rng.randint(4, 10), "startup_duration_s": rng.uniform(3.8, 6.1)}
        if tid == "task_easy_cronjob_spike":
            return {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.10, 0.30), 4),
                    "process_memory_utilization": rng.choice([0.88, 0.91, 0.93, 0.95, 0.97]),
                    "baseline_memory": rng.choice([0.28, 0.32, 0.35, 0.38, 0.42])}
        if tid == "task_easy_noisy_neighbor":
            return {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.10, 0.25), 4),
                    "noisy_pod_cpu_utilization": rng.choice([0.72, 0.78, 0.82, 0.88, 0.92]),
                    "node_memory_pressure": rng.choice([0.85, 0.88, 0.91, 0.94])}
        return {}

    def _build_medium_metrics(tid: str, rng: random.Random) -> dict[str, Any]:
        if tid == "task_medium_cascade_memleak":
            return {"status": "critical", "http_server_error_rate": round(rng.uniform(0.35, 0.60), 4),
                    "process_memory_utilization": rng.choice([0.68, 0.72, 0.74, 0.78, 0.82, 0.86])}
        if tid == "task_medium_replica_lag":
            return {"status": "critical", "http_server_error_rate": round(rng.uniform(0.28, 0.58), 4),
                    "db_replication_lag_seconds": rng.randint(22, 80),
                    "http_server_write_path_error_rate": round(rng.uniform(0.0, 0.02), 4)}
        if tid == "task_medium_mtls_rotation":
            return {"status": "critical", "http_server_error_rate": round(rng.uniform(0.55, 0.95), 4),
                    "sidecar_cert_rotation_status": "stale", "mtls_handshake_failure_rate": round(rng.uniform(0.55, 0.95), 4)}
        if tid == "task_medium_bg_traffic_leak":
            blue_frac = rng.choice([0.05, 0.10, 0.15, 0.20, 0.25])
            old_err = rng.choice([0.45, 0.55, 0.65, 0.75])
            return {"status": "degraded", "http_server_error_rate": round(blue_frac * old_err, 4),
                    "blue_environment_traffic_fraction": blue_frac, "blue_environment_error_rate": old_err}
        return {}

    def _build_hard_metrics(tid: str, rng: random.Random) -> dict[str, Any]:
        if tid == "task_hard_config_drift_noise":
            return {"status": "critical", "http_server_error_rate": round(rng.uniform(0.52, 0.78), 4),
                    "process_open_file_descriptors": rng.randint(2, 6)}
        if tid == "task_hard_consensus_degradation":
            return {"status": "critical", "http_server_error_rate": round(rng.uniform(0.45, 0.75), 4),
                    "config_data_age_seconds": rng.randint(420, 720),
                    "consensus_leader_election_count": rng.randint(3, 8)}
        if tid == "task_hard_multiz_failover":
            return {"status": "critical", "http_server_error_rate": round(rng.uniform(0.60, 0.90), 4),
                    "az_a_load_factor": rng.uniform(1.6, 2.5), "hikaricp_timeout_rate": round(rng.uniform(0.15, 0.45), 2)}
        return {}

    examples = []
    easy_ticks = [0, 2, 4, 6]
    for tid in easy_ids:
        task = task_map[tid]
        for i in range(4):
            tick = easy_ticks[i]
            style_idx = i % 4  # One style per tick deterministically
            m = _build_easy_metrics(tid, rng)
            severity = "CRITICAL" if m.get("http_server_error_rate", 0) > 0.5 else "WARNING"
            alert = _format_alert(style_idx, task["fault_service"], m["http_server_error_rate"], 0.05, severity, tick)
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "easy",
                "fault_type": task["fault_type"],
                "variation_strategy": "alert_phrasing,metric_value,noise_injection",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("easy", tick),
                    "alerts": [alert], "service_metrics": {task["fault_service"]: m},
                    "logs": {task["fault_service"]: [f"fault: {tid}"]},
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [], "expected_score_range": [0.70, 1.0], "suboptimal_paths": [],
            })

    medium_ticks = [0, 1, 3, 5]
    medium_counts = [5, 4, 4, 5]
    for idx, tid in enumerate(medium_ids):
        task = task_map[tid]
        count = medium_counts[idx]
        for i in range(count):
            tick = medium_ticks[i % len(medium_ticks)]
            style_idx = i % 4
            m = _build_medium_metrics(tid, rng)
            severity = "CRITICAL" if m.get("http_server_error_rate", 0) > 0.5 else "WARNING"
            alert = _format_alert(style_idx, task["fault_service"], m["http_server_error_rate"], 0.05, severity, tick)
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "medium",
                "fault_type": task["fault_type"],
                "variation_strategy": "alert_phrasing,metric_value,noise_injection,red_herring_salience",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("medium", tick),
                    "alerts": [alert], "service_metrics": {task["fault_service"]: m},
                    "logs": {task["fault_service"]: [f"fault: {tid}"]},
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [], "expected_score_range": [0.50, 0.90], "suboptimal_paths": [],
            })

    hard_ticks = [0, 2, 5, 7]
    for tid in hard_ids:
        task = task_map[tid]
        for i in range(4):
            tick = hard_ticks[i]
            style_idx = i % 4
            m = _build_hard_metrics(tid, rng)
            severity = "CRITICAL" if m.get("http_server_error_rate", 0) > 0.5 else "WARNING"
            alert = _format_alert(style_idx, task["fault_service"], m["http_server_error_rate"], 0.05, severity, tick)
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "hard",
                "fault_type": task["fault_type"],
                "variation_strategy": "alert_phrasing,metric_value,adversarial_content",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("hard", tick),
                    "alerts": [alert], "service_metrics": {task["fault_service"]: m},
                    "logs": {task["fault_service"]: [f"fault: {tid}"]},
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [], "expected_score_range": [0.30, 0.80], "suboptimal_paths": [],
            })

    if len(examples) != 50:
        raise ValueError(f"Expected 50, got {len(examples)}")
    rng.shuffle(examples)
    return examples


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "firewatch_env"))
    from config import TASKS
    task_list = [{"task_id": tc.task_id, "difficulty": tc.difficulty, "fault_type": tc.fault_type,
                  "fault_service": tc.fault_service, "services": tc.services}
                 for tc in TASKS.values()]
    exs = generate(task_list, rng_seed=11000)
    print(f"Generated {len(exs)} examples")
    for i, ex in enumerate(exs[:4]):
        print(f"\nEx {i}: {ex['task_seed_id']} tier={ex['tier']}")
        print(f"  Alert: {ex['observation']['alerts']}")
