"""
gen_19_mixed_high_noise.py — Mixed Batch: High-Noise Environments

Script: gen_19_mixed_high_noise.py
Batch: 018 (script_num = 19, batch = 018)
Primary axes: noise_injection (maximum) + metric_value + red_herring_salience
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-19
Bootstrap: CONTEXT-BOOTSTRAP.md
High-noise environments with degraded signal-to-noise ratio.
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


# Noise pools from SPEC-19
NOISE_ALERTS = [
    "[WARNING] analytics-service: scheduled backup running — disk I/O elevated (expected)",
    "[INFO] cache-service: eviction rate 12/s — within normal traffic variance",
    "[WARNING] db-proxy: slow query log volume high — non-critical, monitoring only",
    "[INFO] api-gateway: connection pool refresh in progress — normal rolling refresh",
    "[WARNING] user-service: GC collection triggered — heap at 58% (normal)",
    "[CRITICAL] monitoring-agent: failed to scrape metrics from 2 pods (transient network issue)",
    "[WARNING] load-balancer: health check latency 45ms (threshold 40ms) — 1 pod",
    "[INFO] notification-service: batch job running — CPU spike expected (every 4h)",
    "[WARNING] auth-service: token refresh rate elevated 1.2× baseline (high traffic event)",
    "[INFO] payment-service: PCI compliance scan running — latency may be slightly elevated",
]

NOISE_LOGS = [
    "INFO [service-X] Heartbeat check passed. Latency: 0.8ms. All connections healthy.",
    "DEBUG [service-Y] Cache warm-up complete. 14,200 keys loaded.",
    "WARN [service-Z] Scheduled maintenance window starting in 30 minutes.",
    "INFO [monitoring] Metric scrape succeeded for 47/49 pods (2 pods in restart).",
    "DEBUG [service-W] Connection pool health check: 48/50 connections active.",
]


GOLD = {
    "task_easy_alert_fatigue": ["get_metrics_detail(db-proxy)", "fetch_logs(db-proxy)", "revert_config(db-proxy)", "declare_resolved"],
    "task_easy_oom_baseline": ["fetch_logs(auth-service)", "scale_replicas(auth-service)", "declare_resolved"],
    "task_easy_quota_runaway": ["trace_dependencies(user-service)", "get_metrics_detail(notification-service)", "rollback_deploy(notification-service)", "declare_resolved"],
    "task_easy_log_debug_disk": ["fetch_logs(api-gateway)", "set_log_level(api-gateway, level=\"INFO\")", "declare_resolved"],
    "task_easy_jwt_clock_skew": ["fetch_logs(auth-service)", "force_ntp_sync(auth-service)", "declare_resolved"],
    "task_medium_ntp_clock_drift": ["trace_dependencies(auth-service)", "trace_dependencies(payment-service)", "get_metrics_detail(db-proxy)", "fetch_logs(db-proxy)", "revert_config(db-proxy)", "declare_resolved"],
    "task_medium_corrupted_external_dep": ["fetch_logs(user-service)", "rollback_deploy(user-service)", "declare_resolved"],
    "task_medium_gateway_rate_limit": ["fetch_logs(api-gateway)", "revert_config(api-gateway)", "declare_resolved"],
    "task_medium_grpc_deadline": ["get_metrics_detail(payment-service)", "trace_dependencies(order-service)", "enable_deadline_propagation(order-service)", "declare_resolved"],
    "task_hard_config_drift_noise": ["get_metrics_detail(api-gateway)", "fetch_logs(api-gateway)", "revert_config(api-gateway)", "declare_resolved"],
    "task_hard_quota_cascade": ["inspect_quota_usage(ml-inference-service)", "request_quota_increase(ml-inference-service, resource=\"gpu_compute\")", "declare_resolved"],
    "task_hard_partial_infra_asymmetric": ["inspect_infrastructure_topology()", "get_metrics_detail(infrastructure)", "remediate_infrastructure()", "declare_resolved"],
}


def _add_noise(n_alerts: int, n_logs: int, rng: random.Random) -> tuple[list[str], dict[str, list[str]]]:
    """Generate noise alerts and noise log lines."""
    alerts = rng.sample(NOISE_ALERTS, min(n_alerts, len(NOISE_ALERTS)))
    logs = {}
    for _ in range(n_logs):
        line = rng.choice(NOISE_LOGS)
        svc = rng.choice(["analytics-service", "cache-service", "db-proxy", "monitoring-agent", "load-balancer"])
        if svc not in logs:
            logs[svc] = []
        logs[svc].append(line)
    return alerts, logs


def _build_easy_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_easy_alert_fatigue":
        pool_size = rng.randint(3, 6)
        fds = rng.choice([3200, 3800, 4200, 4987])
        cache_mem = rng.choice([0.68, 0.72, 0.78, 0.82])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.15, 0.35),
            "db_connection_pool_size": pool_size,
            "process_open_file_descriptors": fds,
            "cache_memory_utilization": cache_mem,
        }
    if tid == "task_easy_oom_baseline":
        mem = rng.choice([0.93, 0.95, 0.96, 0.97, 0.98, 0.99])
        return {
            "status": "critical",
            "http_server_error_rate": rng.choice([0.12, 0.15, 0.18, 0.22, 0.27, 0.31]),
            "process_memory_utilization": mem,
            "restart_count": rng.randint(2, 8),
        }
    if tid == "task_easy_quota_runaway":
        deploy_age = rng.choice([60, 90, 120, 180, 240, 300])
        queue_depth = rng.choice([420, 620, 847, 1100, 1400])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.45, 0.70),
            "last_deployment_age_seconds": deploy_age,
            "runtime_thread_pool_queue_depth": queue_depth,
        }
    if tid == "task_easy_log_debug_disk":
        disk = rng.choice([0.91, 0.94, 0.96, 0.97, 0.98])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.55, 0.85),
            "process_disk_usage_ratio": disk,
            "application_log_level": "DEBUG",
        }
    if tid == "task_easy_jwt_clock_skew":
        offset = -rng.choice([240, 270, 305, 340, 380, 420])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.30, 0.65),
            "system_clock_offset_seconds": offset,
        }
    return {}


def _build_medium_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_medium_ntp_clock_drift":
        offset = rng.choice([38, 42, 45, 52, 58, 68])
        thread_pool_depth = rng.choice([200, 350, 520, 740])
        jwt_reject = rng.choice([0.40, 0.55, 0.65, 0.75, 0.82])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.50, 0.78),
            "system_clock_offset_seconds": offset,
            "auth_thread_pool_depth": thread_pool_depth,
            "jwt_rejection_rate": jwt_reject,
        }
    if tid == "task_medium_corrupted_external_dep":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.55, 0.88),
            "dependency_health_score": rng.uniform(0.25, 0.45),
        }
    if tid == "task_medium_gateway_rate_limit":
        rate = rng.choice([8, 10, 12, 15, 20])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.05, 0.15),
            "rate_limit_rpm": rate,
        }
    if tid == "task_medium_grpc_deadline":
        orphaned = rng.choice([0.55, 0.65, 0.72, 0.80, 0.88])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.50, 0.88),
            "grpc_orphaned_call_rate": orphaned,
            "grpc_deadline_propagation_rate": rng.choice([0.00, 0.05, 0.10, 0.15]),
        }
    return {}


def _build_hard_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_hard_config_drift_noise":
        fds = rng.randint(2, 6)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.52, 0.78),
            "process_open_file_descriptors": fds,
        }
    if tid == "task_hard_quota_cascade":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.08, 0.25),
            "gpu_quota_utilization": 0.0,
            "cpu_fallback_response_bytes": rng.randint(95, 160),
        }
    if tid == "task_hard_partial_infra_asymmetric":
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.12, 0.40),
            "affected_service_count": rng.choice([2, 3, 4]),
        }
    return {}


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = [
        "task_easy_alert_fatigue", "task_easy_oom_baseline", "task_easy_quota_runaway",
        "task_easy_log_debug_disk", "task_easy_jwt_clock_skew",
    ]
    medium_ids = [
        "task_medium_ntp_clock_drift", "task_medium_corrupted_external_dep",
        "task_medium_gateway_rate_limit", "task_medium_grpc_deadline",
    ]
    hard_ids = [
        "task_hard_config_drift_noise", "task_hard_quota_cascade", "task_hard_partial_infra_asymmetric",
    ]

    examples = []

    # Easy: 5 tasks × 4 = 20 (2-3 noise alerts, 2-3 noise logs)
    for tid in easy_ids:
        task = task_map[tid]
        for i in range(4):
            tick = [0, 2, 4, 6][i]
            m = _build_easy_metrics(tid, rng, tick)
            slo_burn = rng.choice([1.8, 2.5, 3.2, 4.0, 5.1])

            # Add noise: 2-3 alerts, 2-3 logs
            n_noise_alerts = rng.randint(2, 3)
            n_noise_logs = rng.randint(2, 3)
            noise_alerts, noise_logs = _add_noise(n_noise_alerts, n_noise_logs, rng)

            fault_alert = f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"
            all_alerts = noise_alerts + [fault_alert]

            logs = {task["fault_service"]: [f"fault: {tid}"]}
            logs.update(noise_logs)

            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "easy",
                "fault_type": task["fault_type"],
                "variation_strategy": "noise_injection,metric_value,red_herring_salience",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("easy", tick),
                    "alerts": all_alerts,
                    "service_metrics": {task["fault_service"]: m},
                    "logs": logs,
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [], "expected_score_range": [0.70, 1.0], "suboptimal_paths": [],
            })

    # Medium: 4 tasks × ~4-5 = 18 (3-5 noise alerts, 3-4 noise logs, 1 critical noise)
    medium_counts = [5, 4, 4, 5]
    for idx, tid in enumerate(medium_ids):
        task = task_map[tid]
        count = medium_counts[idx]
        for i in range(count):
            tick = [0, 1, 3, 5][i % 4]
            m = _build_medium_metrics(tid, rng, tick)
            slo_burn = rng.choice([2.2, 3.0, 4.5, 5.8, 7.0])

            n_noise_alerts = rng.randint(3, 5)
            n_noise_logs = rng.randint(3, 4)
            noise_alerts, noise_logs = _add_noise(n_noise_alerts, n_noise_logs, rng)

            fault_alert = f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"
            all_alerts = noise_alerts + [fault_alert]

            logs = {task["fault_service"]: [f"fault: {tid}"]}
            logs.update(noise_logs)

            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "medium",
                "fault_type": task["fault_type"],
                "variation_strategy": "noise_injection,metric_value,red_herring_salience",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("medium", tick),
                    "alerts": all_alerts,
                    "service_metrics": {task["fault_service"]: m},
                    "logs": logs,
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [], "expected_score_range": [0.50, 0.90], "suboptimal_paths": [],
            })

    # Hard: 3 tasks × 4 = 12 (5-8 noise alerts, 4-6 noise logs)
    for tid in hard_ids:
        task = task_map[tid]
        for i in range(4):
            tick = [0, 2, 5, 7][i]
            m = _build_hard_metrics(tid, rng, tick)
            slo_burn = rng.choice([3.0, 4.5, 6.0, 8.0, 10.0])

            n_noise_alerts = rng.randint(5, 8)
            n_noise_logs = rng.randint(4, 6)
            noise_alerts, noise_logs = _add_noise(n_noise_alerts, n_noise_logs, rng)

            fault_alert = f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"
            all_alerts = noise_alerts + [fault_alert]

            logs = {task["fault_service"]: [f"fault: {tid}"]}
            logs.update(noise_logs)

            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "hard",
                "fault_type": task["fault_type"],
                "variation_strategy": "noise_injection,metric_value,adversarial_content",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("hard", tick),
                    "alerts": all_alerts,
                    "service_metrics": {task["fault_service"]: m},
                    "logs": logs,
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
    exs = generate(task_list, rng_seed=19000)
    print(f"Generated {len(exs)} examples")
    for i, ex in enumerate(exs[:3]):
        print(f"\nEx {i}: {ex['task_seed_id']} tier={ex['tier']}")
        print(f"  Alerts count: {len(ex['observation']['alerts'])}")
        print(f"  Logs count: {len(ex['observation']['logs'])}")