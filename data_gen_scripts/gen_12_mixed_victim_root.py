"""
gen_12_mixed_victim_root.py — Mixed Batch: Victim-vs-Root Confusion

Script: gen_12_mixed_victim_root.py
Batch: 011 (script_num = 12, batch = 011)
Primary axes: metric_value + red_herring_salience (victim prominence)
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-12
Bootstrap: CONTEXT-BOOTSTRAP.md
Victim always appears first with highest error rate, root is different service.
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


# Task configs: (victim, root, gold_actions)
TASK_CONFIG = {
    "task_easy_quota_runaway": ("user-service", "notification-service",
        ["trace_dependencies(user-service)", "get_metrics_detail(notification-service)", "rollback_deploy(notification-service)", "declare_resolved"]),
    "task_easy_timeout_propagation": ("order-service", "inventory-service",
        ["trace_dependencies(order-service)", "fetch_logs(inventory-service)", "optimize_query(inventory-service)", "declare_resolved"]),
    "task_easy_thundering_herd": ("api-gateway", "session-service",
        ["get_metrics_detail(session-service)", "enable_connection_throttle(session-service)", "declare_resolved"]),
    "task_easy_slow_db_query": ("checkout-service", "user-service",
        ["trace_dependencies(checkout-service)", "get_metrics_detail(user-service)", "rollback_deploy(user-service)", "declare_resolved"]),
    "task_easy_image_pull_backoff": ("api-gateway", "recommendation-engine",
        ["fetch_logs(recommendation-engine)", "rollback_deploy(recommendation-engine)", "declare_resolved"]),
    "task_medium_asymmetric_blast": ("auth-service", "db-proxy",
        ["trace_dependencies(auth-service)", "trace_dependencies(payment-service)", "get_metrics_detail(db-proxy)", "restart_service(db-proxy)", "declare_resolved"]),
    "task_medium_circuit_breaker_masking": ("product-catalog-service", "pricing-service",
        ["trace_dependencies(product-catalog-service)", "get_metrics_detail(pricing-service)", "scale_replicas(pricing-service)", "declare_resolved"]),
    "task_medium_rollout_quota_exhaustion": ("auth-service", "api-gateway",
        ["trace_dependencies(auth-service)", "get_metrics_detail(api-gateway)", "rollback_deploy(api-gateway)", "declare_resolved"]),
    "task_medium_retry_storm": ("notification-service", "api-gateway",
        ["get_metrics_detail(api-gateway)", "trace_dependencies(api-gateway)", "disable_retries(api-gateway)", "configure_retry_backoff(api-gateway)", "declare_resolved"]),
    "task_hard_dual_fault_shared_cascade": ("checkout-service", "auth-service",
        ["trace_dependencies(checkout-service)", "rollback_deploy(auth-service)", "scale_replicas(payment-service)", "declare_resolved"]),
    "task_hard_metastable_failure": ("search-service", "api-gateway",
        ["get_metrics_detail(search-service)", "disable_retries(api-gateway)", "declare_resolved"]),
    "task_hard_multiteam_dual_fault": ("checkout-service", "auth-service",
        ["trace_dependencies(checkout-service)", "rollback_deploy(auth-service)", "scale_replicas(payment-service)", "declare_resolved"]),
}


def _build_easy_metrics(tid: str, victim: str, root: str, rng: random.Random) -> tuple[dict, dict]:
    """Return (victim_metrics, root_metrics)"""
    if tid == "task_easy_quota_runaway":
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.45, 0.70), 4), "queue_depth": rng.randint(500, 800)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.05, 0.12), 4)}
    elif tid == "task_easy_timeout_propagation":
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.30, 0.45), 4), "http_server_request_duration_p99": rng.choice([5.2, 6.4, 7.8, 8.4, 9.1, 10.2])}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.02, 0.08), 4)}
    elif tid == "task_easy_thundering_herd":
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.55, 0.85), 4), "session_connection_count": rng.randint(500, 900)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.04, 0.10), 4)}
    elif tid == "task_easy_slow_db_query":
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.35, 0.70), 4)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.02, 0.08), 4), "http_server_request_duration_p99": rng.choice([5.2, 6.4, 7.8, 8.4, 9.1, 10.2])}
    elif tid == "task_easy_image_pull_backoff":
        v = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.40, 0.65), 4)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.02, 0.10), 4), "image_pull_backoff_seconds": rng.randint(30, 300)}
    else:
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.50, 0.80), 4)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.05, 0.15), 4)}
    return v, r


def _build_medium_metrics(tid: str, victim: str, root: str, rng: random.Random) -> tuple[dict, dict]:
    if tid == "task_medium_asymmetric_blast":
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.75, 0.92), 4)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.02, 0.10), 4), "process_open_file_descriptors": rng.randint(2, 6)}
    elif tid == "task_medium_circuit_breaker_masking":
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.60, 0.85), 4), "circuit_breaker_state": "open"}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.04, 0.12), 4), "process_memory_utilization": rng.uniform(0.78, 0.92)}
    elif tid == "task_medium_rollout_quota_exhaustion":
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.65, 0.88), 4), "thread_pool_depth": rng.randint(400, 700)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.04, 0.15), 4)}
    elif tid == "task_medium_retry_storm":
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.70, 0.90), 4)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.02, 0.10), 4), "effective_rps_multiplier": rng.uniform(1.8, 5.5)}
    else:
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.60, 0.90), 4)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.04, 0.15), 4)}
    return v, r


def _build_hard_metrics(tid: str, victim: str, root: str, rng: random.Random) -> tuple[dict, dict]:
    if tid == "task_hard_dual_fault_shared_cascade":
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.50, 0.80), 4)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.38, 0.65), 4), "last_deployment_age_seconds": rng.randint(120, 540)}
    elif tid == "task_hard_metastable_failure":
        v = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.18, 0.28), 4), "http_server_request_queue_depth": rng.randint(650, 900)}
        r = {"status": "degraded", "http_server_error_rate": 0.0, "effective_rps_multiplier": rng.uniform(1.8, 5.5)}
    elif tid == "task_hard_multiteam_dual_fault":
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.50, 0.80), 4)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.38, 0.65), 4)}
    else:
        v = {"status": "critical", "http_server_error_rate": round(rng.uniform(0.50, 0.85), 4)}
        r = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.30, 0.60), 4)}
    return v, r


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = list(TASK_CONFIG.keys())[:5]  # First 5
    medium_ids = list(TASK_CONFIG.keys())[5:9]  # Next 4
    hard_ids = list(TASK_CONFIG.keys())[9:]  # Last 3

    examples = []
    easy_ticks = [0, 2, 4, 6]
    for tid in easy_ids:
        victim, root_svc, gold = TASK_CONFIG[tid]
        task = task_map[tid]
        for i in range(4):
            tick = easy_ticks[i]
            v_m, r_m = _build_easy_metrics(tid, victim, root_svc, rng)
            # Build metrics dict: victim FIRST
            metrics = {}
            metrics[victim] = v_m
            metrics[root_svc] = r_m
            # Bystanders
            for svc in task.get("services", [])[:2]:
                if svc not in metrics:
                    metrics[svc] = {"status": "healthy", "http_server_error_rate": round(rng.uniform(0.0, HEALTHY_ERROR_RATE), 4)}
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "easy",
                "fault_type": task["fault_type"],
                "variation_strategy": "metric_value,red_herring_salience",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("easy", tick),
                    "alerts": [f"[CRITICAL] {victim} critical — investigate first", f"[WARNING] {root_svc} slight degradation"],
                    "service_metrics": metrics,
                    "logs": {victim: [f"Victim {victim} showing critical errors"], root_svc: [f"Root {root_svc} root cause"]},
                },
                "gold_action_sequence": gold.copy(), "gold_alternatives": [],
                "expected_score_range": [0.70, 1.0], "suboptimal_paths": [],
            })

    medium_ticks = [0, 1, 3, 5]
    medium_counts = [5, 4, 4, 5]
    for idx, tid in enumerate(medium_ids):
        victim, root_svc, gold = TASK_CONFIG[tid]
        task = task_map[tid]
        count = medium_counts[idx]
        for i in range(count):
            tick = medium_ticks[i % len(medium_ticks)]
            v_m, r_m = _build_medium_metrics(tid, victim, root_svc, rng)
            metrics = {}
            metrics[victim] = v_m
            metrics[root_svc] = r_m
            for svc in task.get("services", [])[:3]:
                if svc not in metrics:
                    metrics[svc] = {"status": "healthy", "http_server_error_rate": round(rng.uniform(0.0, HEALTHY_ERROR_RATE), 4)}
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "medium",
                "fault_type": task["fault_type"],
                "variation_strategy": "metric_value,red_herring_salience",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("medium", tick),
                    "alerts": [f"[CRITICAL] {victim} critical — investigate first", f"[WARNING] {root_svc} slight degradation"],
                    "service_metrics": metrics,
                    "logs": {victim: [f"Victim {victim} showing critical errors"], root_svc: [f"Root {root_svc} root cause"]},
                },
                "gold_action_sequence": gold.copy(), "gold_alternatives": [],
                "expected_score_range": [0.50, 0.90], "suboptimal_paths": [],
            })

    hard_ticks = [0, 2, 5, 7]
    for tid in hard_ids:
        victim, root_svc, gold = TASK_CONFIG[tid]
        task = task_map[tid]
        for i in range(4):
            tick = hard_ticks[i]
            v_m, r_m = _build_hard_metrics(tid, victim, root_svc, rng)
            metrics = {}
            metrics[victim] = v_m
            metrics[root_svc] = r_m
            for svc in task.get("services", [])[:4]:
                if svc not in metrics:
                    metrics[svc] = {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.04, 0.10), 4)}
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "hard",
                "fault_type": task["fault_type"],
                "variation_strategy": "metric_value,red_herring_salience,adversarial_content",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("hard", tick),
                    "alerts": [f"[CRITICAL] {victim} critical — investigate first", f"[WARNING] {root_svc} root cause"],
                    "service_metrics": metrics,
                    "logs": {victim: [f"Victim {victim} showing critical errors"], root_svc: [f"Root {root_svc} root cause"]},
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
                  "fault_service": tc.fault_service, "services": tc.services}
                 for tc in TASKS.values()]
    exs = generate(task_list, rng_seed=12000)
    print(f"Generated {len(exs)} examples")
    for i, ex in enumerate(exs[:3]):
        print(f"\nEx {i}: {ex['task_seed_id']} tier={ex['tier']}")
        keys = list(ex['observation']['service_metrics'].keys())
        print(f"  First metric (victim): {keys[0]}")
        print(f"  Alert order: {[a[:20] for a in ex['observation']['alerts']]}")
