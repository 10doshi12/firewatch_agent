"""
gen_08_mixed_suboptimal.py — Mixed Batch: Suboptimal Path Teaching

Script: gen_08_mixed_suboptimal.py
Batch: 007 (script_num = 8, batch = 007)
Primary axes: suboptimal_path + metric_value
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-08
Bootstrap: CONTEXT-BOOTSTRAP.md
"""

import random
from typing import Any

HEALTHY_ERROR_RATE = 0.03


def _derive_status(error_rate: float) -> str:
    if error_rate >= 0.5:
        return "critical"
    elif error_rate >= 0.2:
        return "degraded"
    elif error_rate >= HEALTHY_ERROR_RATE:
        return "degraded"
    return "healthy"


def _calculate_budget(tier: str, tick: int) -> float:
    if tier == "easy":
        return round(30.0 - (tick * 1.5), 2)
    elif tier == "medium":
        return round(60.0 - (tick * 2.0), 2)
    return round(120.0 - (tick * 3.0), 2)


SUBOPTIMAL_PATHS = {
    "task_easy_crashloop_backoff": [{
        "wrong_action": "restart_service(payment-service)",
        "consequence": "Service re-enters CrashLoopBackOff immediately. restart_count increments. Missing env var unchanged. Zero improvement.",
        "partial_credit": False,
        "score_penalty": 0.25,
    }],
    "task_easy_lb_hotspot": [{
        "wrong_action": "scale_replicas(user-profile-service)",
        "consequence": "New replicas added to same broken weight table. Hot replica weight=4.0 unchanged. New replicas receive ~0% traffic. CPU still 0.87 on hot replica.",
        "partial_credit": False,
        "score_penalty": 0.25,
    }],
    "task_easy_liveness_probe_flap": [{
        "wrong_action": "restart_service(payment-processor)",
        "consequence": "Service restarts and immediately re-enters same probe-fail loop. restart_count increments. Flap continues unchanged.",
        "partial_credit": False,
        "score_penalty": 0.25,
    }],
    "task_easy_rate_limiter_misconfig": [{
        "wrong_action": "scale_replicas(api-gateway)",
        "consequence": "More api-gateway instances each enforcing same misconfigured 100-rpm limit. 429 error rate unchanged. Backend services still idle.",
        "partial_credit": False,
        "score_penalty": 0.25,
    }],
    "task_easy_quota_runaway": [{
        "wrong_action": "scale_replicas(user-service)",
        "consequence": "user-service absorbs excess calls for 2-3 ticks. Error rate drops to ~0.08. notification-service retry bug persists. At tick+3, user-service fills up again. Errors return.",
        "partial_credit": True,
        "score_penalty": 0.20,
    }],
    "task_medium_asymmetric_blast": [{
        "wrong_action": "restart_service(auth-service)",
        "consequence": "auth-service restarts, errors clear momentarily, return within 2 ticks — db-proxy is still partitioned. Precision failure.",
        "partial_credit": False,
        "score_penalty": 0.30,
    }],
    "task_medium_circuit_breaker_masking": [{
        "wrong_action": "circuit_break(product-catalog-service)",
        "consequence": "Action is a no-op. Circuit breaker already open. pricing-service memory leak continues. Worse outcome in remaining ticks.",
        "partial_credit": False,
        "score_penalty": 0.30,
    }],
    "task_medium_cache_eviction_storm": [{
        "wrong_action": "revert_config(user-db)",
        "consequence": "user-db config reverted to same state (no change). Cache continues evicting. Fallthrough load unchanged. Error rate unchanged.",
        "partial_credit": False,
        "score_penalty": 0.25,
    }],
    "task_medium_rollout_quota_exhaustion": [{
        "wrong_action": "scale_replicas(auth-service)",
        "consequence": "Temporarily absorbs api-gateway excess retries for ~3 ticks. auth error_rate drops to ~0.15. api-gateway retry count=10 unchanged. Errors return in 4 ticks.",
        "partial_credit": True,
        "score_penalty": 0.20,
    }],
    "task_hard_config_drift_noise": [{
        "wrong_action": "restart_service(auth-service)",
        "consequence": "auth-service restarts — it is a victim service. api-gateway pool exhaustion unchanged. api-gateway errors continue. Precision penalty + acted on adversarial.",
        "partial_credit": False,
        "score_penalty": 0.35,
    }],
    "task_hard_redis_split_brain": [{
        "wrong_action": "restart_service(redis-cluster)",
        "consequence": "Restart merges cluster without flushing diverged keys. Last-write-wins means some keys have wrong values. Data inconsistency persists.",
        "partial_credit": False,
        "score_penalty": 0.30,
    }],
    "task_hard_stampeding_herd": [{
        "wrong_action": "scale_replicas(origin-db)",
        "consequence": "More DB replicas handle immediate stampede load. cache_hit_rate still 0.02-0.15. Next traffic spike triggers identical stampede. Root cause unaddressed.",
        "partial_credit": True,
        "score_penalty": 0.20,
    }],
}


def _generate_examples(
    task: dict[str, Any],
    count: int,
    ticks: list[int],
    tier: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    examples = []
    task_id = task["task_id"]
    fault_service = task["fault_service"]
    fault_type = task["fault_type"]

    gold_actions = {
        "task_easy_crashloop_backoff": ["fetch_logs(payment-service)", "rollback_deploy(payment-service)", "declare_resolved"],
        "task_easy_lb_hotspot": ["get_metrics_detail(user-profile-service)", "rebalance_load(user-profile-service)", "declare_resolved"],
        "task_easy_liveness_probe_flap": ["get_metrics_detail(payment-processor)", "fetch_logs(payment-processor)", "adjust_probe_timing(payment-processor)", "declare_resolved"],
        "task_easy_rate_limiter_misconfig": ["fetch_logs(api-gateway)", "revert_config(api-gateway)", "declare_resolved"],
        "task_easy_quota_runaway_client": ["trace_dependencies(user-service)", "get_metrics_detail(notification-service)", "rollback_deploy(notification-service)", "declare_resolved"],
        "task_medium_asymmetric_blast": ["trace_dependencies(auth-service)", "trace_dependencies(payment-service)", "get_metrics_detail(db-proxy)", "restart_service(db-proxy)", "declare_resolved"],
        "task_medium_circuit_breaker_masking": ["get_metrics_detail(pricing-service)", "fetch_logs(pricing-service)", "declare_resolved"],
        "task_medium_cache_eviction_storm": ["trace_dependencies(user-db)", "get_metrics_detail(cache)", "increase_cache_memory(cache)", "declare_resolved"],
        "task_medium_rollout_quota_exhaustion": ["trace_dependencies(auth-service)", "get_metrics_detail(api-gateway)", "rollback_deploy(api-gateway)", "declare_resolved"],
        "task_hard_config_drift_noise": ["get_metrics_detail(api-gateway)", "fetch_logs(api-gateway)", "revert_config(api-gateway)", "declare_resolved"],
        "task_hard_redis_split_brain": ["inspect_cluster_topology(cache)", "flush_diverged_keys(cache)", "force_cluster_resync(cache)", "declare_resolved"],
        "task_hard_stampeding_herd": ["get_metrics_detail(cache)", "fetch_logs(cache)", "enable_cache_warming(cache)", "rate_limit_cache_misses(cache)", "declare_resolved"],
    }.get(task_id, ["declare_resolved"])

    subopt = SUBOPTIMAL_PATHS.get(task_id, [])

    for i in range(count):
        tick = ticks[i % len(ticks)]
        service_names = list(task.get("services", (fault_service,)))

        # Sample metrics from ranges
        metrics = {}
        for svc in service_names:
            if svc == fault_service:
                if fault_type == "oom":
                    mem = rng.choice([0.93, 0.95, 0.96, 0.97, 0.98, 0.99])
                    err = rng.uniform(0.12, 0.31)
                    m = {"status": _derive_status(err), "http_server_error_rate": round(err, 4),
                         "process_memory_utilization": mem, "restart_count": rng.randint(3, 7)}
                elif fault_type == "config_drift":
                    fd = rng.randint(2, 6)
                    err = rng.uniform(0.52, 0.78)
                    m = {"status": _derive_status(err), "http_server_error_rate": round(err, 4),
                         "process_open_file_descriptors": fd}
                elif fault_type == "memory_leak":
                    mem = rng.uniform(0.62, 0.88)
                    err = rng.uniform(0.05, 0.45)
                    m = {"status": _derive_status(err), "http_server_error_rate": round(err, 4),
                         "process_memory_utilization": round(mem, 4), "runtime_gc_pause_duration": round(rng.uniform(0.25, 0.65), 2)}
                elif fault_type == "bad_deploy":
                    err = rng.uniform(0.30, 0.75)
                    m = {"status": _derive_status(err), "http_server_error_rate": round(err, 4),
                         "last_deployment_age_seconds": rng.choice([120, 180, 240, 300, 420, 540]), "restart_count": rng.randint(0, 5)}
                else:
                    err = rng.uniform(0.40, 0.80)
                    m = {"status": _derive_status(err), "http_server_error_rate": round(err, 4)}
            else:
                err = rng.uniform(0.0, HEALTHY_ERROR_RATE) if rng.random() < 0.7 else rng.uniform(0.04, 0.10)
                m = {"status": _derive_status(err), "http_server_error_rate": round(err, 4)}
            metrics[svc] = m

        example = {
            "example_id": "",
            "source_script": "",
            "task_seed_id": task_id,
            "tier": tier,
            "fault_type": fault_type,
            "variation_strategy": "suboptimal_path,metric_value",
            "observation": {
                "tick": tick,
                "budget": _calculate_budget(tier, tick),
                "alerts": [f"[WARNING] {fault_service} showing elevated errors"],
                "service_metrics": metrics,
                "logs": {fault_service: [f"{fault_type} fault signature detected"]},
            },
            "gold_action_sequence": gold_actions.copy(),
            "gold_alternatives": [],
            "expected_score_range": [0.70, 1.0] if tier == "easy" else ([0.50, 0.90] if tier == "medium" else [0.30, 0.80]),
            "suboptimal_paths": subopt.copy(),
        }
        examples.append(example)
    return examples


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = ["task_easy_crashloop_backoff", "task_easy_lb_hotspot", "task_easy_liveness_probe_flap",
               "task_easy_rate_limiter_misconfig", "task_easy_quota_runaway"]
    medium_ids = ["task_medium_asymmetric_blast", "task_medium_circuit_breaker_masking",
                  "task_medium_cache_eviction_storm", "task_medium_rollout_quota_exhaustion"]
    hard_ids = ["task_hard_config_drift_noise", "task_hard_redis_split_brain", "task_hard_stampeding_herd"]

    examples = []
    easy_ticks = [0, 2, 4, 6]
    for tid in easy_ids:
        if tid not in task_map:
            raise ValueError(f"Task {tid} not found")
        examples.extend(_generate_examples(task_map[tid], 4, easy_ticks, "easy", rng))

    medium_ticks = [0, 1, 3, 5]
    medium_counts = [5, 4, 4, 5]
    for idx, tid in enumerate(medium_ids):
        if tid not in task_map:
            raise ValueError(f"Task {tid} not found")
        ticks_for = [medium_ticks[i % len(medium_ticks)] for i in range(medium_counts[idx])]
        examples.extend(_generate_examples(task_map[tid], medium_counts[idx], ticks_for, "medium", rng))

    hard_ticks = [0, 2, 5, 7]
    for tid in hard_ids:
        if tid not in task_map:
            raise ValueError(f"Task {tid} not found")
        examples.extend(_generate_examples(task_map[tid], 4, hard_ticks, "hard", rng))

    if len(examples) != 50:
        raise ValueError(f"Expected 50 examples, got {len(examples)}")
    rng.shuffle(examples)
    return examples


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "firewatch_env"))
    from config import TASKS
    task_list = [{"task_id": tc.task_id, "difficulty": tc.difficulty, "fault_type": tc.fault_type,
                  "fault_service": tc.fault_service, "services": tc.services, "red_herrings": tc.red_herrings}
                 for tc in TASKS.values()]
    examples = generate(task_list, rng_seed=8000)
    print(f"Generated {len(examples)} examples")
    for i, ex in enumerate(examples[:3]):
        print(f"\nExample {i}: task={ex['task_seed_id']}, tier={ex['tier']}")
        print(f"  Gold: {ex['gold_action_sequence']}")
        print(f"  Subopt: {ex['suboptimal_paths'][0]['wrong_action'] if ex['suboptimal_paths'] else 'none'}")
