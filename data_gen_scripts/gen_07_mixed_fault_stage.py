"""
gen_07_mixed_fault_stage.py — Mixed Batch: Fault Stage Progression Sampling

Script: gen_07_mixed_fault_stage.py
Batch: 006 (script_num = 7, batch = 006)
Primary axes: fault_stage + metric_value
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-07
Bootstrap: CONTEXT-BOOTSTRAP.md

NOTE: No tick=0 examples. All observed mid-incident at varying stages.
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


def _scale_metric(lower: float, upper: float, p: float, jitter: float) -> float:
    """Scale metric from lower->upper by proportion p, with ±jitter."""
    val = lower + p * (upper - lower)
    return round(val * jitter, 4)


def _calculate_budget(tier: str, tick: int) -> float:
    if tier == "easy":
        return round(30.0 - (tick * 1.5), 2)
    elif tier == "medium":
        return round(60.0 - (tick * 2.0), 2)
    else:
        return round(120.0 - (tick * 3.0), 2)


def _build_alerts_for_stage(fault_service: str, fault_type: str, p: float, rng: random.Random) -> list[str]:
    """Build alert list based on fault progression stage p (0.0-1.0)."""
    alerts = []
    if p < 0.3:
        alerts.append(f"[WARNING] {fault_service} showing elevated errors")
    elif p < 0.6:
        alerts.append(f"[WARNING] {fault_service} degraded")
        alerts.append(f"[CRITICAL] {fault_service} error rate elevated")
    else:
        alerts.append(f"[CRITICAL] {fault_service} critical — immediate action required")
        alerts.append(f"[PAGE] {fault_service} outage escalation")
    return alerts


def _generate_easy_examples(
    task: dict[str, Any],
    count: int,
    ticks: list[int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate Easy tier examples at non-zero ticks."""
    examples = []
    task_id = task["task_id"]
    fault_service = task["fault_service"]
    fault_type = task["fault_type"]

    max_ticks = 20  # Easy

    gold_actions_map = {
        "task_easy_pool_restart_cycle": [
            "fetch_logs(auth-service)", "revert_config(auth-service)", "declare_resolved",
        ],
        "task_easy_alert_fatigue": [
            "get_metrics_detail(db-proxy)", "fetch_logs(db-proxy)", "revert_config(db-proxy)", "declare_resolved",
        ],
        "task_easy_image_pull_backoff": [
            "fetch_logs(recommendation-engine)", "rollback_deploy(recommendation-engine)", "declare_resolved",
        ],
        "task_easy_slow_db_query": [
            "trace_dependencies(checkout-service)", "get_metrics_detail(user-service)",
            "rollback_deploy(user-service)", "declare_resolved",
        ],
        "task_easy_rollout_stuck": [
            "fetch_logs(checkout-service)", "rollback_deployment_rollout(checkout-service)", "declare_resolved",
        ],
    }
    gold_template = gold_actions_map.get(task_id, ["declare_resolved"])

    for i in range(count):
        tick = ticks[i]
        p = tick / max_ticks
        jitter = rng.uniform(0.90, 1.10)

        task_metrics = None
        task_logs = None

        if task_id == "task_easy_pool_restart_cycle":
            restart_count = int(_scale_metric(1, 5, p, jitter))
            err_rate = _scale_metric(0.30, 0.65, p, jitter)
            task_metrics = {
                "status": _derive_status(err_rate),
                "http_server_error_rate": err_rate,
                "process_open_file_descriptors": int(_scale_metric(2, 6, p, jitter)),
                "restart_count": restart_count,
            }
            pool_size = int(_scale_metric(2, 4, p, jitter))
            task_logs = {
                fault_service: [
                    f"HikariPool-1 - Connection pool exhausted. Total={pool_size}, Idle=0, Waiting=50",
                    f"restart_count: {restart_count}",
                ],
            }
        elif task_id == "task_easy_alert_fatigue":
            err_rate = _scale_metric(0.52, 0.78, p, jitter)
            noise_count = int(_scale_metric(2, 5, p, jitter))
            task_metrics = {
                "status": "critical",
                "http_server_error_rate": err_rate,
                "process_open_file_descriptors": int(_scale_metric(2, 6, p, jitter)),
            }
            task_logs = {
                "db-proxy": [f"FD exhaustion: {int(_scale_metric(3200, 4987, p, jitter))} FDs used"],
            }
        elif task_id == "task_easy_image_pull_backoff":
            backoff = int(_scale_metric(30, 300, p, jitter))
            restart_count = int(_scale_metric(1, 5, p, jitter))
            err_rate = _scale_metric(0.40, 0.80, p, jitter)
            task_metrics = {
                "status": "degraded",
                "http_server_error_rate": err_rate,
                "image_pull_backoff_seconds": backoff,
                "restart_count": restart_count,
            }
            task_logs = {
                fault_service: [
                    f"image_pull_backoff_seconds: {backoff}",
                    "Image pull backoff: manifest unknown",
                    f"restart_count: {restart_count}",
                ],
            }
        elif task_id == "task_easy_slow_db_query":
            checkout_err = _scale_metric(0.15, 0.55, p, jitter)
            user_p99 = _scale_metric(3.0, 9.0, p, jitter)
            task_metrics = {
                "status": _derive_status(checkout_err),
                "http_server_error_rate": checkout_err,
                "http_server_request_duration_p99": user_p99,
            }
            task_logs = {
                "user-service": [f"p99 latency: {user_p99:.1f}s"],
                "checkout-service": [f"error_rate: {checkout_err:.2f} (timeout dependency)"],
            }
        elif task_id == "task_easy_rollout_stuck":
            progress = rng.choice([0.30, 0.40, 0.50, 0.60, 0.70])  # Stuck stays constant
            err_rate = _scale_metric(0.30, 0.75, p, jitter)
            task_metrics = {
                "status": "degraded",
                "http_server_error_rate": err_rate,
                "deployment_rollout_progress_pct": progress,
            }
            task_logs = {
                fault_service: [
                    f"Rollout stuck at {progress:.0%}",
                    f"error_rate: {err_rate:.2f} (503 cascade)",
                ],
            }

        # Scale cascade/red herring independently
        service_names = list(task.get("services", (fault_service,)))
        metrics = {}
        for svc in service_names:
            if task_metrics and svc == fault_service:
                m = dict(task_metrics)
            else:
                # Bystander: independent of fault stage
                err = rng.uniform(0.0, HEALTHY_ERROR_RATE) if rng.random() < 0.7 else rng.uniform(0.04, 0.10)
                m = {"status": _derive_status(err), "http_server_error_rate": round(err, 4)}
            metrics[svc] = m

        example = {
            "example_id": "",
            "source_script": "",
            "task_seed_id": task_id,
            "tier": "easy",
            "fault_type": fault_type,
            "variation_strategy": "fault_stage,metric_value",
            "observation": {
                "tick": tick,
                "budget": _calculate_budget("easy", tick),
                "alerts": _build_alerts_for_stage(fault_service, fault_type, p, rng),
                "service_metrics": metrics,
                "logs": task_logs or {},
            },
            "gold_action_sequence": gold_template.copy(),
            "gold_alternatives": [],
            "expected_score_range": [0.70, 1.0],
            "suboptimal_paths": [],
        }
        examples.append(example)

    return examples


def _generate_medium_examples(
    task: dict[str, Any],
    count: int,
    ticks: list[int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate Medium tier examples at non-zero ticks."""
    examples = []
    task_id = task["task_id"]
    fault_service = task["fault_service"]
    fault_type = task["fault_type"]

    max_ticks = 30  # Medium

    gold_actions_map = {
        "task_medium_config_race": [
            "fetch_logs(api-gateway)", "get_metrics_detail(api-gateway)",
            "revert_config(api-gateway)", "declare_resolved",
        ],
        "task_medium_retry_storm": [
            "get_metrics_detail(api-gateway)", "trace_dependencies(api-gateway)",
            "disable_retries(api-gateway)", "configure_retry_backoff(api-gateway)", "declare_resolved",
        ],
        "task_medium_corrupted_dependency": [
            "get_metrics_detail(user-service)", "fetch_logs(user-service)",
            "rollback_deploy(user-service)", "declare_resolved",
        ],
        "task_medium_configmap_reload": [
            "fetch_logs(notification-service)", "restart_service(notification-service)", "declare_resolved",
        ],
    }
    gold_template = gold_actions_map.get(task_id, ["declare_resolved"])

    for i in range(count):
        tick = ticks[i]
        p = tick / max_ticks
        jitter = rng.uniform(0.90, 1.10)

        task_metrics = None
        task_logs = None

        if task_id == "task_medium_config_race":
            wrong_frac = int(_scale_metric(0.2, 0.4, p, jitter))  # 1/5 to 2/5
            err_rate = _scale_metric(0.20, 0.45, p, jitter)
            task_metrics = {
                "status": "degraded",
                "http_server_error_rate": err_rate,
                "wrong_config_replica_fraction": wrong_frac / 5,
            }
            task_logs = {
                "api-gateway": [
                    f"Auth header version mismatch on {wrong_frac}/5 replicas",
                    f"aggregate error_rate: {err_rate:.2f}",
                ],
            }
        elif task_id == "task_medium_retry_storm":
            multiplier = _scale_metric(1.8, 5.5, p, jitter)
            notif_err = _scale_metric(0.20, 0.70, p, jitter)
            task_metrics = {
                "status": "critical",
                "http_server_error_rate": notif_err,
                "effective_rps_multiplier": round(multiplier, 1),
            }
            task_logs = {
                "api-gateway": [
                    f"Retry storm: effective_rps_multiplier={multiplier:.1f}x",
                    "Retries overwhelming downstream services",
                ],
            }
        elif task_id == "task_medium_corrupted_external_dep":
            err_rate = _scale_metric(0.42, 0.75, p, jitter)
            affected = int(_scale_metric(1, 4, p, jitter))
            task_metrics = {
                "status": "critical",
                "http_server_error_rate": err_rate,
                "user_service_instances_affected": affected,
            }
            task_logs = {
                fault_service: [
                    f"corrupted_dependency — {affected} instances affected",
                    "Checksum validation failed — corrupted binary",
                ],
            }
        elif task_id == "task_medium_configmap_reload":
            err_rate = 1.0  # Hard failure — always
            log_count = int(_scale_metric(2, 8, p, jitter))
            task_metrics = {
                "status": "critical",
                "http_server_error_rate": err_rate,
                "configmap_update_age_seconds": int(_scale_metric(60, 300, p, jitter)),
            }
            task_logs = {
                fault_service: [
                    f"ConfigMap path: /templates/v1/ (expected v2/)",
                    f"Log entries: {log_count} in past tick",
                    "Missing template: order_confirmation.html",
                ],
            }

        service_names = list(task.get("services", (fault_service,)))
        metrics = {}
        for svc in service_names:
            if task_metrics and svc == fault_service:
                m = dict(task_metrics)
            else:
                err = rng.uniform(0.0, HEALTHY_ERROR_RATE) if rng.random() < 0.7 else rng.uniform(0.04, 0.10)
                m = {"status": _derive_status(err), "http_server_error_rate": round(err, 4)}
            metrics[svc] = m

        example = {
            "example_id": "",
            "source_script": "",
            "task_seed_id": task_id,
            "tier": "medium",
            "fault_type": fault_type,
            "variation_strategy": "fault_stage,metric_value,red_herring_salience",
            "observation": {
                "tick": tick,
                "budget": _calculate_budget("medium", tick),
                "alerts": _build_alerts_for_stage(fault_service, fault_type, p, rng),
                "service_metrics": metrics,
                "logs": task_logs or {},
            },
            "gold_action_sequence": gold_template.copy(),
            "gold_alternatives": [],
            "expected_score_range": [0.50, 0.90],
            "suboptimal_paths": [],
        }
        examples.append(example)

    return examples


def _generate_hard_examples(
    task: dict[str, Any],
    count: int,
    ticks: list[int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate Hard tier examples at non-zero ticks."""
    examples = []
    task_id = task["task_id"]
    fault_service = task["fault_service"]
    fault_type = task["fault_type"]

    max_ticks = 40  # Hard

    gold_actions_map = {
        "task_hard_dual_fault_shared_cascade": [
            "trace_dependencies(checkout-service)", "rollback_deploy(auth-service)",
            "scale_replicas(payment-service)", "declare_resolved",
        ],
        "task_hard_metastable_failure": [
            "get_metrics_detail(search-service)", "disable_retries(api-gateway)", "declare_resolved",
        ],
        "task_hard_cache_corruption": [
            "get_metrics_detail(cache)", "inspect_cache_corruption_layers(cache)",
            "revert_config(cache)", "declare_resolved",
        ],
    }
    gold_template = gold_actions_map.get(task_id, ["declare_resolved"])

    for i in range(count):
        tick = ticks[i]
        p = tick / max_ticks
        jitter = rng.uniform(0.90, 1.10)

        task_metrics = None
        task_logs = None

        if task_id == "task_hard_dual_fault_shared_cascade":
            auth_err = _scale_metric(0.38, 0.65, p, jitter)
            pay_mem = _scale_metric(0.68, 0.90, p, jitter)
            checkout_err = _scale_metric(0.50, 0.80, p, jitter)
            task_metrics = {
                "status": "critical",
                "http_server_error_rate": checkout_err,
                "auth_service_error_rate": auth_err,
                "payment_service_memory_utilization": pay_mem,
            }
            task_logs = {
                "checkout-service": [
                    f"Dual fault cascade — auth_err={auth_err:.2f}, pay_mem={pay_mem:.2f}",
                    f"checkout error_rate: {checkout_err:.2f}",
                ],
            }
        elif task_id == "task_hard_metastable_failure":
            queue_depth = int(_scale_metric(650, 900, p, jitter))
            multiplier = _scale_metric(1.8, 5.5, p, jitter)
            task_metrics = {
                "status": "critical",
                "http_server_error_rate": _scale_metric(0.18, 0.28, p, jitter),
                "http_server_request_queue_depth": queue_depth,
                "effective_rps_multiplier": round(multiplier, 1),
                "metastable_feedback_loop_active": True,
            }
            task_logs = {
                "search-service": [
                    f"Queue depth: {queue_depth}. Processing stalled.",
                    f"Feedback loop active. Retry multiplier: {multiplier:.1f}x.",
                ],
            }
        elif task_id == "task_hard_cache_corruption":
            miss_rate = _scale_metric(0.35, 0.72, p, jitter)
            layers = int(_scale_metric(1, 2, p, jitter))
            affected = int(_scale_metric(2, 4, p, jitter))
            task_metrics = {
                "status": "critical",
                "http_server_error_rate": miss_rate,
                "cache_miss_rate": miss_rate,
                "cache_corruption_layers": layers,
                "services_affected": affected,
            }
            task_logs = {
                "cache": [
                    f"Cache miss rate: {miss_rate:.2f}",
                    f"Corruption layers: {layers} — {affected} services receiving corrupt data",
                    "Checksum failure: CRC mismatch on cached data",
                ],
            }

        service_names = list(task.get("services", (fault_service,)))
        metrics = {}
        for svc in service_names:
            if task_metrics and svc == fault_service:
                m = dict(task_metrics)
            else:
                err = rng.uniform(0.0, HEALTHY_ERROR_RATE) if rng.random() < 0.6 else rng.uniform(0.04, 0.07)
                m = {"status": _derive_status(err), "http_server_error_rate": round(err, 4)}
            metrics[svc] = m

        example = {
            "example_id": "",
            "source_script": "",
            "task_seed_id": task_id,
            "tier": "hard",
            "fault_type": fault_type,
            "variation_strategy": "fault_stage,metric_value,red_herring_salience,adversarial_content",
            "observation": {
                "tick": tick,
                "budget": _calculate_budget("hard", tick),
                "alerts": _build_alerts_for_stage(fault_service, fault_type, p, rng),
                "service_metrics": metrics,
                "logs": task_logs or {},
            },
            "gold_action_sequence": gold_template.copy(),
            "gold_alternatives": [],
            "expected_score_range": [0.30, 0.80],
            "suboptimal_paths": [],
        }
        examples.append(example)

    return examples


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_task_ids = [
        "task_easy_pool_restart_cycle", "task_easy_alert_fatigue", "task_easy_image_pull_backoff",
        "task_easy_slow_db_query", "task_easy_rollout_stuck",
    ]
    medium_task_ids = [
        "task_medium_config_race", "task_medium_retry_storm",
        "task_medium_corrupted_external_dep", "task_medium_configmap_reload",
    ]
    hard_task_ids = [
        "task_hard_dual_fault_shared_cascade", "task_hard_metastable_failure",
        "task_hard_cache_corruption",
    ]

    examples = []

    # Easy ticks {1, 3, 6, 10} — no tick=0
    easy_ticks = [1, 3, 6, 10]
    for task_id in easy_task_ids:
        if task_id not in task_map:
            raise ValueError(f"Task {task_id} not found")
        task = task_map[task_id]
        examples.extend(_generate_easy_examples(task, 4, easy_ticks, rng))

    # Medium ticks {2, 4, 7, 12}
    medium_ticks = [2, 4, 7, 12]
    medium_counts = [5, 4, 4, 5]  # sum = 18
    for idx, task_id in enumerate(medium_task_ids):
        if task_id not in task_map:
            raise ValueError(f"Task {task_id} not found")
        task = task_map[task_id]
        count = medium_counts[idx]
        ticks_for_task = [medium_ticks[i % len(medium_ticks)] for i in range(count)]
        examples.extend(_generate_medium_examples(task, count, ticks_for_task, rng))

    # Hard ticks {3, 6, 10, 18}
    hard_ticks = [3, 6, 10, 18]
    for task_id in hard_task_ids:
        if task_id not in task_map:
            raise ValueError(f"Task {task_id} not found")
        task = task_map[task_id]
        examples.extend(_generate_hard_examples(task, 4, hard_ticks, rng))

    if len(examples) != 50:
        raise ValueError(f"Expected 50 examples, got {len(examples)}")

    rng.shuffle(examples)
    return examples


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "firewatch_env"))
    from config import TASKS
    task_list = [
        {"task_id": tc.task_id, "difficulty": tc.difficulty, "fault_type": tc.fault_type,
         "fault_service": tc.fault_service, "services": tc.services, "red_herrings": tc.red_herrings}
        for tc in TASKS.values()
    ]
    examples = generate(task_list, rng_seed=7000)
    print(f"Generated {len(examples)} examples")
    for i, ex in enumerate(examples[:3]):
        print(f"\nExample {i}: task={ex['task_seed_id']}, tier={ex['tier']}, tick={ex['observation']['tick']}, budget={ex['observation']['budget']}")
