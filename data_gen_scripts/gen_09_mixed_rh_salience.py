"""
gen_09_mixed_rh_salience.py — Mixed Batch: Red Herring Salience Manipulation

Script: gen_09_mixed_rh_salience.py
Batch: 008 (script_num = 9, batch = 008)
Primary axes: red_herring_salience + metric_value
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-09
Bootstrap: CONTEXT-BOOTSTRAP.md
"""

import random
from typing import Any

HEALTHY_ERROR_RATE = 0.03


def _derive_status(e: float) -> str:
    if e >= 0.5: return "critical"
    if e >= 0.2: return "degraded"
    if e >= HEALTHY_ERROR_RATE: return "degraded"
    return "healthy"


def _calculate_budget(tier: str, tick: int) -> float:
    if tier == "easy": return round(30.0 - tick * 1.5, 2)
    if tier == "medium": return round(60.0 - tick * 2.0, 2)
    return round(120.0 - tick * 3.0, 2)


# Salience configs: A=first_in_dict, B=severity_inversion, C=log_volume, D=metric_mimic
SALIENT_CONFIGS = ["A", "B", "C", "D"]


def _apply_salience_config(
    config: str,
    root_service: str,
    rh_service: str,
    rng: random.Random,
    tier: str,
) -> tuple[dict[str, dict], dict[str, list[str]], list[str]]:
    """Apply salience config, return (metrics, logs, alerts) with RH promoted."""
    metrics = {}
    logs = {}
    alerts = []

    if tier == "easy":
        # Noise service injection: one healthy service gets a plausible WARNING alert first
        noise_svc = rng.choice(["db-proxy", "cache", "metrics-exporter"])
        noise_err = rng.uniform(0.06, 0.12)
        metrics[noise_svc] = {"status": "degraded", "http_server_error_rate": round(noise_err, 4)}
        logs[noise_svc] = [f"[WARNING] {noise_svc} showing elevated latency (non-critical)"]
        alerts.append(f"[WARNING] {noise_svc} elevated — investigate")
        return metrics, logs, alerts

    # Medium/Hard: actual red herring manipulation
    if config == "A":
        # First-in-dict: RH appears first, highest error
        rh_err = rng.uniform(0.12, 0.22)
        metrics[rh_service] = {"status": "critical", "http_server_error_rate": round(rh_err, 4)}
        metrics[root_service] = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.04, 0.08), 4)}
        logs[rh_service] = [f"RED HERRING: {rh_service} showing elevated errors — investigate first"]
        alerts.append(f"[CRITICAL] {rh_service} elevated — root cause?")

    elif config == "B":
        # Severity inversion: RH fires CRITICAL, root fires WARNING
        root_err = rng.uniform(0.04, 0.08)
        rh_err = rng.uniform(0.50, 0.80)
        metrics[rh_service] = {"status": "critical", "http_server_error_rate": round(rh_err, 4)}
        metrics[root_service] = {"status": "degraded", "http_server_error_rate": round(root_err, 4)}
        logs[rh_service] = [f"RED HERRING: {rh_service} CRITICAL alert — appears severe"]
        logs[root_service] = [f"Real root: {root_service} WARNING — less visible but actual cause"]
        alerts.append(f"[CRITICAL] {rh_service} — immediate action required!")
        alerts.append(f"[WARNING] {root_service} slight degradation")

    elif config == "C":
        # Log volume dominance: RH has 4-6 lines, root has 1-2
        metrics[rh_service] = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.06, 0.12), 4)}
        metrics[root_service] = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.55, 0.90), 4)}
        logs[rh_service] = [
            f"[INFO] {rh_service} routine check 1",
            f"[INFO] {rh_service} routine check 2",
            f"[DEBUG] {rh_service} metrics nominal",
            f"[INFO] {rh_service} health check passed",
            f"[DEBUG] {rh_service} connection pool OK",
        ]
        logs[root_service] = [
            f"FAULT: {root_service} critical error",
            f"DIAG: {root_service} diagnostic token present",
        ]
        alerts.append(f"[WARNING] {root_service} fault detected")

    elif config == "D":
        # Metric mimic: RH mimics root cause's metric at lower value, no diagnostic token
        rh_err = rng.uniform(0.30, 0.45)  # Elevated but not as high
        root_err = rng.uniform(0.70, 0.95)
        metrics[rh_service] = {"status": "degraded", "http_server_error_rate": round(rh_err, 4), "metric_like_root": round(rh_err, 4)}
        metrics[root_service] = {"status": "critical", "http_server_error_rate": round(root_err, 4), "diagnostic_token": True}
        logs[rh_service] = [f"MIMIC: {rh_service} elevated but no diagnostic token"]
        logs[root_service] = [f"FAULT: {root_service} diagnostic token confirmed"]
        alerts.append(f"[CRITICAL] {rh_service} mimicking fault signal")
        alerts.append(f"[CRITICAL] {root_service} actual root cause")

    return metrics, logs, alerts


def _build_task_metrics(task_id: str, fault_service: str, rng: random.Random) -> dict[str, Any]:
    """Build task-specific metrics."""
    if task_id == "task_easy_thread_deadlock":
        return {
            "status": "critical", "http_server_error_rate": round(rng.uniform(0.55, 0.85), 4),
            "blocked_thread_count": rng.randint(42, 60), "wait_ratio": round(rng.uniform(0.88, 1.00), 2),
        }
    if task_id == "task_easy_jwt_clock_skew":
        return {
            "status": "degraded", "http_server_error_rate": round(rng.uniform(0.40, 0.70), 4),
            "system_clock_offset_seconds": -abs(rng.choice([240, 270, 305, 340, 380, 420])),
        }
    if task_id == "task_easy_rbac_403":
        return {
            "status": "degraded", "http_server_error_rate": round(rng.uniform(0.50, 0.80), 4),
            "rbac_permission_status": "forbidden",
        }
    if task_id == "task_easy_cronjob_spike":
        return {
            "status": "degraded", "http_server_error_rate": round(rng.uniform(0.10, 0.30), 4),
            "process_memory_utilization": rng.choice([0.88, 0.91, 0.93, 0.95, 0.97]),
        }
    if task_id == "task_easy_noisy_neighbor":
        return {
            "status": "degraded", "http_server_error_rate": round(rng.uniform(0.10, 0.25), 4),
            "noisy_pod_cpu_utilization": rng.choice([0.72, 0.78, 0.82, 0.88, 0.92]),
        }
    if task_id == "task_medium_canary_false_alert":
        return {
            "status": "degraded", "http_server_error_rate": round(rng.uniform(0.38, 0.60), 4),
            "canary_error_rate": round(rng.uniform(0.38, 0.60), 4), "stable_error_rate": round(rng.uniform(0.01, 0.02), 4),
        }
    if task_id == "task_medium_replica_lag":
        return {
            "status": "critical", "http_server_error_rate": round(rng.uniform(0.45, 0.75), 4),
            "db_replication_lag_seconds": rng.randint(22, 80),
        }
    if task_id == "task_medium_mtls_rotation":
        return {
            "status": "critical", "http_server_error_rate": round(rng.uniform(0.55, 0.95), 4),
            "sidecar_cert_rotation_status": "stale", "mtls_handshake_failure_rate": round(rng.uniform(0.55, 0.95), 4),
        }
    if task_id == "task_medium_gateway_rate_limit":
        return {
            "status": "critical", "http_server_error_rate": round(rng.uniform(0.88, 0.97), 4),
            "rate_limit_rpm": rng.choice([8, 10, 12, 15, 20]),
        }
    if task_id == "task_hard_consensus_degradation":
        return {
            "status": "critical", "http_server_error_rate": round(rng.uniform(0.45, 0.75), 4),
            "config_data_age_seconds": rng.randint(420, 720), "consensus_leader_election_count": rng.randint(3, 8),
        }
    if task_id == "task_hard_pipeline_freshness":
        return {
            "status": "degraded", "http_server_error_rate": 0.0,
            "freshness_lag_seconds": rng.randint(380, 650), "queue_depth": rng.randint(12000, 38000),
        }
    if task_id == "task_hard_multiz_failover":
        return {
            "status": "critical", "http_server_error_rate": round(rng.uniform(0.60, 0.90), 4),
            "az_a_load_factor": rng.uniform(1.6, 2.5), "hikaricp_timeout_rate": round(rng.uniform(0.15, 0.45), 2),
        }
    return {}


GOLD_ACTIONS = {
    "task_easy_thread_deadlock": ["thread_dump(order-service)", "restart_thread_pool(order-service)", "declare_resolved"],
    "task_easy_jwt_clock_skew": ["fetch_logs(auth-service)", "force_ntp_sync(auth-service)", "declare_resolved"],
    "task_easy_rbac_403": ["fetch_logs(notification-service)", "grant_rbac_permission(notification-service)", "declare_resolved"],
    "task_easy_cronjob_spike": ["get_metrics_detail(analytics-service)", "scale_replicas(analytics-service)", "declare_resolved"],
    "task_easy_noisy_neighbor": [
        "get_metrics_detail(batch-processor)",
        "evict_noisy_pod(batch-processor)",
        "declare_resolved",
    ],
    "task_medium_canary_false_alert": ["get_metrics_detail(api-gateway)", "fetch_logs(api-gateway)", "declare_resolved"],
    "task_medium_replica_lag": ["fetch_logs(user-service)", "redirect_reads_to_primary(user-service)", "force_replica_resync(user-service)", "declare_resolved"],
    "task_medium_mtls_rotation": ["inspect_mtls_status(payment-service)", "force_cert_rotation(payment-service)", "declare_resolved"],
    "task_medium_gateway_rate_limit": ["fetch_logs(api-gateway)", "revert_config(api-gateway)", "declare_resolved"],
    "task_hard_consensus_degradation": ["inspect_consensus_state(config-service)", "isolate_minority_nodes(config-service)", "force_leader_election(config-service)", "declare_resolved"],
    "task_hard_pipeline_freshness": ["inspect_pipeline_topology(feature-pipeline)", "get_metrics_detail(feature-pipeline)", "restart_pipeline_job(feature-pipeline)", "declare_resolved"],
    "task_hard_multiz_failover": ["get_metrics_detail(api-gateway-az-a)", "rebalance_az_traffic(api-gateway-az-a)", "scale_az_capacity(api-gateway-az-a)", "declare_resolved"],
}


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = ["task_easy_thread_deadlock", "task_easy_jwt_clock_skew", "task_easy_rbac_403",
                "task_easy_cronjob_spike", "task_easy_noisy_neighbor"]
    medium_ids = ["task_medium_canary_false_alert", "task_medium_replica_lag",
                  "task_medium_mtls_rotation", "task_medium_gateway_rate_limit"]
    hard_ids = ["task_hard_consensus_degradation", "task_hard_pipeline_freshness", "task_hard_multiz_failover"]

    RH_SERVICES = {
        "task_medium_canary_false_alert": "fraud-detection-service",
        "task_medium_replica_lag": "session-service",
        "task_medium_mtls_rotation": "db-proxy",
        "task_medium_gateway_rate_limit": "auth-service",
        "task_hard_consensus_degradation": "redis-cluster",
        "task_hard_pipeline_freshness": "ranking-service",
        "task_hard_multiz_failover": "api-gateway-az-c",
    }

    examples = []
    easy_ticks = [0, 2, 4, 6]
    for tid in easy_ids:
        task = task_map[tid]
        fault_service = task["fault_service"]
        for i in range(4):
            tick = easy_ticks[i]
            config = rng.choice(SALIENT_CONFIGS)
            rh_metrics, rh_logs, rh_alerts = _apply_salience_config(config, fault_service, "", rng, "easy")
            task_m = _build_task_metrics(tid, fault_service, rng)
            metrics = dict(rh_metrics)
            metrics[fault_service] = task_m
            gold = GOLD_ACTIONS.get(tid, ["declare_resolved"])
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "easy",
                "fault_type": task["fault_type"],
                "variation_strategy": "red_herring_salience,metric_value",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("easy", tick),
                    "alerts": rh_alerts + [f"[WARNING] {fault_service} fault detected"],
                    "service_metrics": metrics, "logs": dict(rh_logs),
                },
                "gold_action_sequence": gold.copy(), "gold_alternatives": [],
                "expected_score_range": [0.70, 1.0], "suboptimal_paths": [],
            })

    medium_ticks = [0, 1, 3, 5]
    medium_counts = [5, 4, 4, 5]
    for idx, tid in enumerate(medium_ids):
        task = task_map[tid]
        fault_service = task["fault_service"]
        rh_service = RH_SERVICES.get(tid, "db-proxy")
        count = medium_counts[idx]
        for i in range(count):
            tick = medium_ticks[i % len(medium_ticks)]
            config = rng.choice(SALIENT_CONFIGS)
            rh_metrics, rh_logs, rh_alerts = _apply_salience_config(config, fault_service, rh_service, rng, "medium")
            task_m = _build_task_metrics(tid, fault_service, rng)
            metrics = dict(rh_metrics)
            metrics[fault_service] = task_m
            gold = GOLD_ACTIONS.get(tid, ["declare_resolved"])
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "medium",
                "fault_type": task["fault_type"],
                "variation_strategy": "red_herring_salience,metric_value",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("medium", tick),
                    "alerts": rh_alerts + [f"[WARNING] {fault_service} actual fault"],
                    "service_metrics": metrics, "logs": rh_logs,
                },
                "gold_action_sequence": gold.copy(), "gold_alternatives": [],
                "expected_score_range": [0.50, 0.90], "suboptimal_paths": [],
            })

    hard_ticks = [0, 2, 5, 7]
    for tid in hard_ids:
        task = task_map[tid]
        fault_service = task["fault_service"]
        rh_service = RH_SERVICES.get(tid, "db-proxy")
        for i in range(4):
            tick = hard_ticks[i]
            config = rng.choice(SALIENT_CONFIGS)
            rh_metrics, rh_logs, rh_alerts = _apply_salience_config(config, fault_service, rh_service, rng, "hard")
            task_m = _build_task_metrics(tid, fault_service, rng)
            metrics = dict(rh_metrics)
            metrics[fault_service] = task_m
            gold = GOLD_ACTIONS.get(tid, ["declare_resolved"])
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "hard",
                "fault_type": task["fault_type"],
                "variation_strategy": "red_herring_salience,metric_value,adversarial_content",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("hard", tick),
                    "alerts": rh_alerts + [f"[CRITICAL] {fault_service} root cause"],
                    "service_metrics": metrics, "logs": rh_logs,
                },
                "gold_action_sequence": gold.copy(), "gold_alternatives": [],
                "expected_score_range": [0.30, 0.80], "suboptimal_paths": [],
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
                  "fault_service": tc.fault_service, "services": tc.services, "red_herrings": tc.red_herrings}
                 for tc in TASKS.values()]
    examples = generate(task_list, rng_seed=9000)
    print(f"Generated {len(examples)} examples")
    for i, ex in enumerate(examples[:3]):
        print(f"\nEx {i}: {ex['task_seed_id']} tier={ex['tier']}")
        print(f"  Keys order: {list(ex['observation']['service_metrics'].keys())[:3]}")
