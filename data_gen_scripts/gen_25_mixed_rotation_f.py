"""
gen_25_mixed_rotation_f.py — Mixed Batch: Complete Task Rotation F

Script: gen_25_mixed_rotation_f.py
Batch: 024 (script_num = 25, batch = 024)
Primary axes: metric_value + alert_phrasing + red_herring_salience
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-25
Bootstrap: CONTEXT-BOOTSTRAP.md
Fresh cross-tier metric randomisation pass.
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


GOLD = {
    "task_easy_log_storm_disk": ["fetch_logs(notification-service)", "set_log_level(notification-service, level=\"INFO\")", "declare_resolved"],
    "task_easy_cronjob_spike": ["get_metrics_detail(analytics-service)", "fetch_logs(analytics-service)", "scale_replicas(analytics-service)", "declare_resolved"],
    "task_easy_rollout_stuck": ["fetch_logs(checkout-service)", "rollback_deployment_rollout(checkout-service)", "declare_resolved"],
    "task_easy_noisy_neighbor": ["get_metrics_detail()", "evict_noisy_pod()", "declare_resolved"],
    "task_easy_alert_fatigue": ["get_metrics_detail(db-proxy)", "fetch_logs(db-proxy)", "revert_config(db-proxy)", "declare_resolved"],
    "task_medium_hpa_cold_start": ["fetch_logs(recommendation-engine)", "get_metrics_detail(recommendation-engine)", "pre_warm_service(recommendation-engine)", "declare_resolved"],
    "task_medium_stale_registry": ["get_metrics_detail(recommendation-engine)", "deregister_stale_instances(recommendation-engine)", "declare_resolved"],
    "task_medium_configmap_reload": ["fetch_logs(notification-service)", "restart_service(notification-service)", "declare_resolved"],
    "task_medium_single_az_partition": ["get_metrics_detail(api-gateway-az-b)", "drain_availability_zone(az-b)", "declare_resolved"],
    "task_hard_quota_cascade": ["inspect_quota_usage(ml-inference-service)", "request_quota_increase(ml-inference-service, resource=\"gpu_compute\")", "declare_resolved"],
    "task_hard_cache_corruption": ["get_metrics_detail(cache)", "evict_corrupted_keys(cache)", "declare_resolved"],
}


def _build_easy_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_easy_log_storm_disk":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.65, 0.92),
            "process_disk_usage_ratio": rng.choice([0.91, 0.94, 0.96, 0.97, 0.98]),
            "application_log_level": "DEBUG",
            "logs": {"notification-service": [f"INFO logging: disk at {rng.choice([0.91, 0.94, 0.96])}"]},
        }
    if tid == "task_easy_cronjob_spike":
        peak = rng.choice([0.88, 0.91, 0.93, 0.95, 0.97])
        baseline = rng.choice([0.28, 0.32, 0.35, 0.38, 0.42])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.02, 0.08),
            "process_memory_utilization": peak,
            "baseline_memory": baseline,
            "cronjob_runtime_seconds": rng.randint(420, 900),
            "dataset_size_gb": rng.randint(28, 68),
        }
    if tid == "task_easy_rollout_stuck":
        rollout_pct = rng.choice([0.30, 0.40, 0.50, 0.60, 0.70])
        missing = rng.choice(["CHECKOUT_FEATURE_FLAG_ENDPOINT", "CHECKOUT_SERVICE_KEY", "CHECKOUT_API_URL"])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.30, 0.60),
            "deployment_rollout_progress_pct": rollout_pct,
            "logs": {"checkout-service": [f"Rollout stuck at {rollout_pct:.0%}", f"Missing: {missing}"]},
        }
    if tid == "task_easy_noisy_neighbor":
        noisy_cpu = rng.choice([0.72, 0.78, 0.82, 0.88, 0.92])
        node_mem = rng.choice([0.85, 0.88, 0.91, 0.94])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.10, 0.25),
            "noisy_pod_cpu_utilization": noisy_cpu,
            "node_memory_pressure": node_mem,
        }
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
    return {}


def _build_medium_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_medium_hpa_cold_start":
        startup_dur = rng.choice([38, 42, 45, 50, 55, 62])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.05, 0.15),
            "deployment_ready_replicas": 0,
            "startup_duration_s": startup_dur,
            "process_memory_utilization": rng.uniform(0.85, 0.95),
        }
    if tid == "task_medium_stale_registry":
        stale = rng.choice([1, 2, 3])
        total = stale + rng.randint(5, 12)
        return {
            "status": "degraded",
            "http_server_error_rate": round(stale / total, 4),
            "registry_stale_instance_count": stale,
            "registry_health_check_age_seconds": rng.randint(120, 420),
        }
    if tid == "task_medium_configmap_reload":
        config_age = rng.choice(["1 min ago", "2 min ago", "3 min ago", "5 min ago"])
        missing = rng.choice(["email_templates.conf", "notification_config.yaml", "rate_limit_map.json"])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.25, 0.55),
            "configmap_update_age_seconds": rng.randint(60, 300),
            "logs": {
                "notification-service": [f"ConfigMap update {config_age}", f"Missing: /etc/templates/{missing}"],
                "auth-service": [f"Auth-service showing memory {rng.choice([0.62, 0.68, 0.72, 0.76])} — RED HERRING"],
            },
        }
    if tid == "task_medium_single_az_partition":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.82, 0.95),
            "az_b_error_rate": rng.uniform(0.82, 0.95),
            "az_b_weight": 0.33,
            "logs": {"api-gateway-az-b": ["AZ-B partition"]},
        }
    return {}


def _build_hard_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_hard_quota_cascade":
        fallback = rng.choice([95, 110, 130, 140, 160])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.08, 0.25),
            "gpu_quota_utilization": 0.0,
            "cpu_fallback_response_bytes": fallback,
            "logs": {"ml-inference-service": [f"GPU quota 0.00, fallback {fallback}KB"]},
        }
    if tid == "task_hard_cache_corruption":
        miss_rate = rng.uniform(0.35, 0.72)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.40, 0.72),
            "cache_checksum_errors": rng.randint(80, 350),
            "cache_miss_rate": miss_rate,
            "logs": {"cache": ["Cache data integrity error — CRC mismatch detected"]},
        }
    return {}


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = [
        "task_easy_log_storm_disk", "task_easy_cronjob_spike", "task_easy_rollout_stuck",
        "task_easy_noisy_neighbor", "task_easy_alert_fatigue",
    ]
    medium_ids = [
        "task_medium_hpa_cold_start", "task_medium_stale_registry",
        "task_medium_configmap_reload", "task_medium_single_az_partition",
    ]
    hard_ids = [
        "task_hard_quota_cascade", "task_hard_cache_corruption",
    ]

    examples = []

    # Easy: 5 tasks × 4 = 20
    for tid in easy_ids:
        task = task_map[tid]
        for i in range(4):
            tick = [0, 2, 4, 6][i]
            m = _build_easy_metrics(tid, rng, tick)
            slo_burn = rng.choice([1.8, 2.5, 3.2, 4.0, 5.1])
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "easy",
                "fault_type": task["fault_type"],
                "variation_strategy": "metric_value,alert_phrasing,red_herring_salience",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("easy", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {task["fault_service"]: m},
                    "logs": m.get("logs", {task["fault_service"]: [f"fault: {tid}"]}),
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [], "expected_score_range": [0.70, 1.0], "suboptimal_paths": [],
            })

    # Medium: 4 tasks × ~4-5 = 18
    medium_counts = [5, 4, 4, 5]
    for idx, tid in enumerate(medium_ids):
        task = task_map[tid]
        count = medium_counts[idx]
        for i in range(count):
            tick = [0, 1, 3, 5][i % 4]
            m = _build_medium_metrics(tid, rng, tick)
            slo_burn = rng.choice([2.2, 3.0, 4.5, 5.8, 7.0])
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "medium",
                "fault_type": task["fault_type"],
                "variation_strategy": "metric_value,alert_phrasing,red_herring_salience",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("medium", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {task["fault_service"]: m},
                    "logs": m.get("logs", {task["fault_service"]: [f"fault: {tid}"]}),
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [], "expected_score_range": [0.50, 0.90], "suboptimal_paths": [],
            })

    # Hard: 2 tasks × 4 = 8 (need 12 — add gray_failure)
    hard_ids_full = ["task_hard_quota_cascade", "task_hard_cache_corruption", "task_hard_gray_failure"]
    GOLD["task_hard_gray_failure"] = ["get_metrics_detail(auth-service)", "fetch_logs(auth-service)", "inspect_network_policy(auth-service)", "revert_network_policy(auth-service)", "declare_resolved"]
    for tid in hard_ids_full:
        task = task_map[tid]
        for i in range(4):
            tick = [0, 2, 5, 7][i]
            if tid == "task_hard_gray_failure":
                p99 = rng.uniform(5.5, 9.0)
                m = {
                    "status": "degraded",
                    "http_server_error_rate": rng.uniform(0.12, 0.25),
                    "network_packet_loss_rate_inbound": rng.uniform(0.12, 0.25),
                    "http_server_request_duration_p99": p99,
                    "http_server_request_duration_p50": rng.uniform(0.08, 0.12),
                    "logs": {"auth-service": [f"Gray failure: p99={p99:.1f}s"]},
                }
            else:
                m = _build_hard_metrics(tid, rng, tick)
            slo_burn = rng.choice([3.0, 4.5, 6.0, 8.0, 10.0])
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "hard",
                "fault_type": task["fault_type"],
                "variation_strategy": "metric_value,alert_phrasing,adversarial_content",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("hard", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {task["fault_service"]: m},
                    "logs": m.get("logs", {task["fault_service"]: [f"fault: {tid}"]}),
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
    exs = generate(task_list, rng_seed=25000)
    print(f"Generated {len(exs)} examples")