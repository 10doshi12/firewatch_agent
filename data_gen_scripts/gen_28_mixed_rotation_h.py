"""
gen_28_mixed_rotation_h.py — Mixed Batch: Complete Task Rotation H

Script: gen_28_mixed_rotation_h.py
Batch: 027 (script_num = 28, batch = 027)
Primary axes: metric_value + service_pool_substitution + alert_phrasing
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-28
Bootstrap: CONTEXT-BOOTSTRAP.md
"""

import random
from typing import Any

HEALTHY_ERROR_RATE = 0.03

SUBSTITUTION_POOL = {
    "api-gateway": ["ingress-controller", "edge-proxy", "frontend-gateway"],
    "auth-service": ["identity-service", "sso-service", "token-service"],
    "payment-service": ["billing-service", "transaction-service", "payment-processor"],
    "user-service": ["profile-service", "account-service", "member-service"],
    "db-proxy": ["data-proxy", "query-router", "db-gateway"],
    "checkout-service": ["order-service", "cart-service", "purchase-service"],
    "notification-service": ["alert-service", "messaging-service", "comms-service"],
}


def _substitute(svc: str, rng: random.Random) -> str:
    if svc in SUBSTITUTION_POOL:
        return rng.choice(SUBSTITUTION_POOL[svc])
    return svc


def _apply_subst(text: str, sub_map: dict[str, str]) -> str:
    for canonical, sub in sub_map.items():
        text = text.replace(canonical, sub)
    return text


def _calculate_budget(tier: str, tick: int) -> float:
    if tier == "easy": return round(30.0 - tick * 1.5, 2)
    if tier == "medium": return round(60.0 - tick * 2.0, 2)
    return round(120.0 - tick * 3.0, 2)


GOLD = {
    "task_easy_image_pull_backoff": ["fetch_logs(recommendation-engine)", "rollback_deploy(recommendation-engine)", "declare_resolved"],
    "task_easy_thread_deadlock": ["thread_dump(order-service)", "restart_thread_pool(order-service)", "declare_resolved"],
    "task_easy_log_debug_disk": ["fetch_logs(api-gateway)", "set_log_level(api-gateway, level=\"INFO\")", "declare_resolved"],
    "task_easy_rollout_stuck": ["fetch_logs(checkout-service)", "rollback_deployment_rollout(checkout-service)", "declare_resolved"],
    "task_easy_noisy_neighbor": [
        "get_metrics_detail(batch-processor)",
        "evict_noisy_pod(batch-processor)",
        "declare_resolved",
    ],
    "task_medium_corrupted_external_dep": ["fetch_logs(user-service)", "rollback_deploy(user-service)", "declare_resolved"],
    "task_medium_db_connection_herd": ["fetch_logs(db-proxy)", "stagger_connection_pool_reconnect(db-proxy)", "declare_resolved"],
    "task_medium_config_race": ["get_metrics_detail(api-gateway)", "trace_dependencies(api-gateway)", "revert_config(api-gateway)", "declare_resolved"],
    "task_medium_stale_registry": ["get_metrics_detail(recommendation-engine)", "deregister_stale_instances(recommendation-engine)", "declare_resolved"],
    "task_hard_dual_fault_shared_cascade": ["trace_dependencies(checkout-service)", "rollback_deploy(auth-service)", "scale_replicas(payment-service)", "declare_resolved"],
    "task_hard_redis_split_brain": ["inspect_cluster_topology(redis-cluster)", "flush_diverged_keys(redis-cluster)", "force_cluster_resync(redis-cluster)", "declare_resolved"],
    "task_hard_cache_corruption": ["get_metrics_detail(cache)", "evict_corrupted_keys(cache)", "declare_resolved"],
}


def _build_easy_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_easy_image_pull_backoff":
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.40, 0.65),
            "image_pull_backoff_seconds": rng.randint(30, 300),
            "restart_count": rng.randint(3, 7),
        }
    if tid == "task_easy_thread_deadlock":
        blocked = rng.choice([42, 50, 55, 58, 60])
        return {
            "status": "critical",
            "http_server_error_rate": rng.choice([0.88, 0.92, 0.97, 0.99, 1.00]),
            "runtime_blocked_thread_count": blocked,
            "wait_ratio": round(rng.uniform(0.88, 1.00), 2),
        }
    if tid == "task_easy_log_debug_disk":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.55, 0.85),
            "process_disk_usage_ratio": rng.choice([0.91, 0.94, 0.96, 0.97, 0.98]),
            "application_log_level": "DEBUG",
        }
    if tid == "task_easy_rollout_stuck":
        rollout_pct = rng.choice([0.30, 0.40, 0.50, 0.60, 0.70])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.30, 0.60),
            "deployment_rollout_progress_pct": rollout_pct,
        }
    if tid == "task_easy_noisy_neighbor":
        noisy_cpu = rng.choice([0.72, 0.78, 0.82, 0.88, 0.92])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.10, 0.25),
            "noisy_pod_cpu_utilization": noisy_cpu,
            "node_memory_pressure": rng.choice([0.85, 0.88, 0.91, 0.94]),
        }
    return {}


def _build_medium_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_medium_corrupted_external_dep":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.42, 0.75),
            "dependency_health_score": rng.uniform(0.25, 0.45),
            "last_deployment_age_seconds": rng.randint(420, 900),
        }
    if tid == "task_medium_db_connection_herd":
        active = rng.choice([210, 228, 238, 248, 255, 270])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.35, 0.68),
            "db_active_connections": active,
            "db_max_connections": 200,
        }
    if tid == "task_medium_config_race":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.55, 0.85),
            "config_version_mismatch": True,
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
    return {}


def _build_hard_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_hard_dual_fault_shared_cascade":
        auth_err = rng.choice([0.38, 0.50, 0.55, 0.60, 0.65])
        mem_trend = [round(rng.uniform(0.65, 0.72), 4), round(rng.uniform(0.72, 0.80), 4), round(rng.uniform(0.80, 0.86), 4)]
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.30, 0.55),
            "process_memory_utilization": mem_trend[-1],
            "memory_trend": mem_trend,
            "auth_error_rate": auth_err,
        }
    if tid == "task_hard_redis_split_brain":
        diverged = rng.randint(12000, 80000)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.35, 0.65),
            "redis_diverged_key_count": diverged,
            "data_inconsistency_rate": rng.uniform(0.08, 0.32),
        }
    if tid == "task_hard_cache_corruption":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.40, 0.72),
            "cache_checksum_errors": rng.randint(80, 350),
            "cache_hit_rate": rng.uniform(0.08, 0.22),
        }
    return {}


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = ["task_easy_image_pull_backoff", "task_easy_thread_deadlock", "task_easy_log_debug_disk",
                "task_easy_rollout_stuck", "task_easy_noisy_neighbor"]
    medium_ids = ["task_medium_corrupted_external_dep", "task_medium_db_connection_herd",
                  "task_medium_config_race", "task_medium_stale_registry"]
    hard_ids = ["task_hard_dual_fault_shared_cascade", "task_hard_redis_split_brain", "task_hard_cache_corruption"]

    examples = []

    # Easy: 5 tasks × 4 = 20 (substitute root + cascade services)
    easy_sub_svcs_map = {
        "task_easy_image_pull_backoff": ["recommendation-engine"],
        "task_easy_thread_deadlock": ["order-service"],
        "task_easy_log_debug_disk": ["api-gateway", "notification-service"],
        "task_easy_rollout_stuck": ["checkout-service"],
        "task_easy_noisy_neighbor": [],
    }
    for tid in easy_ids:
        task = task_map[tid]
        sub_svcs = easy_sub_svcs_map.get(tid, [])
        for i in range(4):
            tick = [0, 2, 4, 6][i]
            sub_map = {svc: _substitute(svc, rng) for svc in sub_svcs}
            m = _build_easy_metrics(tid, rng, tick)
            slo_burn = rng.choice([1.8, 2.5, 3.2, 4.0, 5.1])
            fault_svc_sub = _apply_subst(task["fault_service"], sub_map) if sub_svcs else task["fault_service"]
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "easy",
                "fault_type": task["fault_type"],
                "variation_strategy": "service_pool_substitution,metric_value,alert_phrasing",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("easy", tick),
                    "alerts": [f"[WARNING] {fault_svc_sub} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {fault_svc_sub: m},
                    "logs": {fault_svc_sub: [f"fault: {_apply_subst(tid, sub_map) if sub_svcs else tid}"]},
                },
                "gold_action_sequence": [(_apply_subst(a, sub_map) if '(' in a else a) for a in GOLD.get(tid, ["declare_resolved"])],
                "gold_alternatives": [], "expected_score_range": [0.70, 1.0], "suboptimal_paths": [],
            })

    # Medium: 4 tasks × ~4-5 = 18
    medium_counts = [5, 4, 4, 5]
    medium_sub_map = {
        "task_medium_corrupted_external_dep": ["user-service", "auth-service", "payment-service"],
        "task_medium_db_connection_herd": ["db-proxy", "auth-service", "payment-service"],
        "task_medium_config_race": ["api-gateway", "auth-service"],
        "task_medium_stale_registry": ["recommendation-engine", "product-catalog"],
    }
    for idx, tid in enumerate(medium_ids):
        task = task_map[tid]
        count = medium_counts[idx]
        sub_svcs = medium_sub_map.get(tid, [])
        for i in range(count):
            tick = [0, 1, 3, 5][i % 4]
            sub_map = {svc: _substitute(svc, rng) for svc in sub_svcs}
            m = _build_medium_metrics(tid, rng, tick)
            slo_burn = rng.choice([2.2, 3.0, 4.5, 5.8, 7.0])
            fault_svc_sub = _apply_subst(task["fault_service"], sub_map) if sub_svcs else task["fault_service"]
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "medium",
                "fault_type": task["fault_type"],
                "variation_strategy": "service_pool_substitution,metric_value,alert_phrasing",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("medium", tick),
                    "alerts": [f"[WARNING] {fault_svc_sub} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {fault_svc_sub: m},
                    "logs": {fault_svc_sub: [f"fault: {_apply_subst(tid, sub_map) if sub_svcs else tid}"]},
                },
                "gold_action_sequence": [(_apply_subst(a, sub_map) if '(' in a else a) for a in GOLD.get(tid, ["declare_resolved"])],
                "gold_alternatives": [], "expected_score_range": [0.50, 0.90], "suboptimal_paths": [],
            })

    # Hard: 3 tasks × 4 = 12
    hard_sub_map = {
        "task_hard_dual_fault_shared_cascade": ["auth-service", "payment-service", "checkout-service"],
        "task_hard_redis_split_brain": ["redis-cluster"],
        "task_hard_cache_corruption": ["cache", "checkout-service"],
    }
    for tid in hard_ids:
        task = task_map[tid]
        sub_svcs = hard_sub_map.get(tid, [])
        for i in range(4):
            tick = [0, 2, 5, 7][i]
            sub_map = {svc: _substitute(svc, rng) for svc in sub_svcs}
            m = _build_hard_metrics(tid, rng, tick)
            slo_burn = rng.choice([3.0, 4.5, 6.0, 8.0, 10.0])
            fault_svc_sub = _apply_subst(task["fault_service"], sub_map) if sub_svcs else task["fault_service"]
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "hard",
                "fault_type": task["fault_type"],
                "variation_strategy": "service_pool_substitution,metric_value,adversarial_content",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("hard", tick),
                    "alerts": [f"[WARNING] {fault_svc_sub} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {fault_svc_sub: m},
                    "logs": {fault_svc_sub: [f"fault: {_apply_subst(tid, sub_map) if sub_svcs else tid}"]},
                },
                "gold_action_sequence": [(_apply_subst(a, sub_map) if '(' in a else a) for a in GOLD.get(tid, ["declare_resolved"])],
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
    exs = generate(task_list, rng_seed=28000)
    print(f"Generated {len(exs)} examples")