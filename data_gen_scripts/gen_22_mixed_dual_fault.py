"""
gen_22_mixed_dual_fault.py — Mixed Batch: dual_fault Thematic

Script: gen_22_mixed_dual_fault.py
Batch: 021 (script_num = 22, batch = 021)
Primary axes: metric_value + suboptimal_path (partial fix) + noise_injection
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-22
Bootstrap: CONTEXT-BOOTSTRAP.md
All tasks involve two simultaneous/compound fault sources.
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
    "task_easy_alert_fatigue": ["get_metrics_detail(db-proxy)", "fetch_logs(db-proxy)", "revert_config(db-proxy)", "declare_resolved"],
    "task_easy_quota_runaway": ["trace_dependencies(user-service)", "get_metrics_detail(notification-service)", "rollback_deploy(notification-service)", "declare_resolved"],
    "task_easy_pool_restart_cycle": ["fetch_logs(auth-service)", "revert_config(auth-service)", "declare_resolved"],
    "task_easy_log_storm_disk": ["fetch_logs(notification-service)", "set_log_level(notification-service, level=\"INFO\")", "declare_resolved"],
    "task_easy_liveness_probe_flap": ["get_metrics_detail(payment-processor)", "fetch_logs(payment-processor)", "adjust_probe_timing(payment-processor)", "declare_resolved"],
    "task_medium_ntp_clock_drift": ["trace_dependencies(auth-service)", "trace_dependencies(payment-service)", "get_metrics_detail(db-proxy)", "fetch_logs(db-proxy)", "revert_config(db-proxy)", "declare_resolved"],
    "task_medium_retry_storm": ["get_metrics_detail(api-gateway)", "trace_dependencies(api-gateway)", "disable_retries(api-gateway)", "configure_retry_backoff(api-gateway)", "declare_resolved"],
    "task_medium_cache_eviction_storm": ["trace_dependencies(user-db)", "get_metrics_detail(cache-service)", "fetch_logs(cache-service)", "increase_cache_memory(cache-service)", "declare_resolved"],
    "task_medium_rollout_quota_exhaustion": ["trace_dependencies(auth-service)", "get_metrics_detail(api-gateway)", "rollback_deploy(api-gateway)", "declare_resolved"],
    "task_hard_dual_fault_shared_cascade": ["trace_dependencies(checkout-service)", "rollback_deploy(auth-service)", "scale_replicas(payment-service)", "declare_resolved"],
    "task_hard_multiteam_dual_fault": ["trace_dependencies(checkout-service)", "rollback_deploy(auth-service)", "scale_replicas(payment-service)", "declare_resolved"],
    "task_hard_metastable_failure": ["get_metrics_detail(search-service)", "disable_retries(api-gateway)", "declare_resolved"],
}


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
            "logs": {"db-proxy": [f"Pool size {pool_size} — real fault (6 noise alerts)"]},
        }
    if tid == "task_easy_quota_runaway":
        deploy_age = rng.choice([60, 90, 120, 180, 240, 300])
        queue_depth = rng.choice([420, 620, 847, 1100, 1400])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.45, 0.70),
            "last_deployment_age_seconds": deploy_age,
            "runtime_thread_pool_queue_depth": queue_depth,
            "logs": {"notification-service": [f"notification-service v2.1.4 deployed {deploy_age}s ago"]},
        }
    if tid == "task_easy_pool_restart_cycle":
        fds = rng.randint(2, 6)
        hikari_pool = rng.randint(2, 6)
        return {
            "status": "critical",
            "http_server_error_rate": rng.choice([0.52, 0.58, 0.61, 0.66, 0.72, 0.78]),
            "process_open_file_descriptors": fds,
            "logs": {"auth-service": [f"HikariCP: pool_size={hikari_pool} (recommended: 20)"]},
        }
    if tid == "task_easy_log_storm_disk":
        disk = rng.choice([0.91, 0.94, 0.96, 0.97, 0.98])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.65, 0.92),
            "process_disk_usage_ratio": disk,
            "logs": {"notification-service": [f"INFO logging: disk {disk:.2f}"]},
        }
    if tid == "task_easy_liveness_probe_flap":
        startup_dur = rng.uniform(3.8, 6.1)
        return {
            "status": "critical",
            "http_server_error_rate": rng.choice([0.65, 0.72, 0.80, 0.88, 0.95, 1.00]),
            "startup_duration_s": startup_dur,
            "liveness_probe_timeout_s": round(startup_dur * rng.uniform(0.6, 0.85), 2),
            "logs": {"payment-processor": [f"Probe timeout < startup — interacting config values"]},
        }
    return {}


def _build_medium_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_medium_ntp_clock_drift":
        offset = rng.choice([38, 42, 45, 52, 58, 68])
        thread_pool_depth = rng.choice([200, 350, 520, 740])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.50, 0.78),
            "system_clock_offset_seconds": offset,
            "auth_thread_pool_depth": thread_pool_depth,
            "logs": {"db-proxy": [f"Clock offset {offset}s — root cause"]},
        }
    if tid == "task_medium_retry_storm":
        retry_count = rng.randint(4, 8)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.60, 0.90),
            "http_client_retry_count": retry_count,
            "effective_rps_multiplier": round(retry_count * 0.85, 2),
            "logs": {"api-gateway": [f"Retry storm: {retry_count} retries"]},
        }
    if tid == "task_medium_cache_eviction_storm":
        cache_mem = rng.choice([0.95, 0.97, 0.98, 0.99, 1.00])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.18, 0.42),
            "cache_memory_utilization": cache_mem,
            "cache_evictions_per_second": rng.choice([180, 280, 380, 450, 550, 680]),
            "logs": {"cache-service": [f"Cache maxmemory — eviction storm"]},
        }
    if tid == "task_medium_rollout_quota_exhaustion":
        deploy_age = rng.choice([420, 540, 660, 720])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.55, 0.85),
            "last_deployment_age_seconds": deploy_age,
            "http_client_retry_count": 10,
            "logs": {"api-gateway": [f"Rollout quota bug — retry count 10"]},
        }
    return {}


def _build_hard_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_hard_dual_fault_shared_cascade":
        auth_err = rng.choice([0.38, 0.50, 0.55, 0.60, 0.65])
        mem_trend = [round(rng.uniform(0.65, 0.72), 4), round(rng.uniform(0.72, 0.80), 4), round(rng.uniform(0.80, 0.86), 4)]
        checkout_err = rng.uniform(0.30, 0.55)
        return {
            "status": "critical",
            "http_server_error_rate": round(checkout_err, 4),
            "process_memory_utilization": mem_trend[-1],
            "memory_trend": mem_trend,
            "auth_error_rate": auth_err,
            "logs": {"checkout-service": ["Dual fault: auth bad_deploy + payment memory_leak"]},
        }
    if tid == "task_hard_multiteam_dual_fault":
        auth_err = rng.choice([0.38, 0.50, 0.55, 0.60, 0.65])
        mem_trend = [round(rng.uniform(0.65, 0.72), 4), round(rng.uniform(0.72, 0.80), 4), round(rng.uniform(0.80, 0.86), 4)]
        checkout_err = rng.uniform(0.30, 0.55)
        return {
            "status": "critical",
            "http_server_error_rate": round(checkout_err, 4),
            "process_memory_utilization": mem_trend[-1],
            "memory_trend": mem_trend,
            "auth_error_rate": auth_err,
            "logs": {"checkout-service": ["Multi-team dual fault"]},
        }
    if tid == "task_hard_metastable_failure":
        queue_depth = rng.randint(650, 900)
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.18, 0.28),
            "http_server_request_queue_depth": queue_depth,
            "metastable_feedback_loop_active": True,
            "logs": {"search-service": [f"Queue depth {queue_depth} — feedback loop active"]},
        }
    return {}


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = [
        "task_easy_alert_fatigue", "task_easy_quota_runaway", "task_easy_pool_restart_cycle",
        "task_easy_log_storm_disk", "task_easy_liveness_probe_flap",
    ]
    medium_ids = [
        "task_medium_ntp_clock_drift", "task_medium_retry_storm",
        "task_medium_cache_eviction_storm", "task_medium_rollout_quota_exhaustion",
    ]
    hard_ids = [
        "task_hard_dual_fault_shared_cascade", "task_hard_multiteam_dual_fault",
        "task_hard_metastable_failure",
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
                "variation_strategy": "metric_value,suboptimal_path,noise_injection",
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
                "variation_strategy": "metric_value,suboptimal_path,noise_injection",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("medium", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {task["fault_service"]: m},
                    "logs": m.get("logs", {task["fault_service"]: [f"fault: {tid}"]}),
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [], "expected_score_range": [0.50, 0.90], "suboptimal_paths": [],
            })

    # Hard: 3 tasks × 4 = 12
    for tid in hard_ids:
        task = task_map[tid]
        for i in range(4):
            tick = [0, 2, 5, 7][i]
            m = _build_hard_metrics(tid, rng, tick)
            slo_burn = rng.choice([3.0, 4.5, 6.0, 8.0, 10.0])
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "hard",
                "fault_type": task["fault_type"],
                "variation_strategy": "metric_value,suboptimal_path,adversarial_content",
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
    exs = generate(task_list, rng_seed=22000)
    print(f"Generated {len(exs)} examples")