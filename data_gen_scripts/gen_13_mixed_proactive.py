"""
gen_13_mixed_proactive.py — Mixed Batch: Proactive and Warning-Phase Scenarios

Script: gen_13_mixed_proactive.py
Batch: 012 (script_num = 13, batch = 012)
Primary axes: fault_stage (early/warning only) + metric_value + alert_phrasing
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-13
Bootstrap: CONTEXT-BOOTSTRAP.md
All examples observed early, error_rate <= 0.08 (Easy) or <= 0.15 (Medium) or <= 0.25 (Hard).
[WARNING] only primary alert. SLO burn as primary signal.
"""

import random
from typing import Any

HEALTHY_ERROR_RATE = 0.03


def _calculate_budget(tier: str, tick: int) -> float:
    if tier == "easy": return round(30.0 - tick * 1.5, 2)
    if tier == "medium": return round(60.0 - tick * 2.0, 2)
    return round(120.0 - tick * 3.0, 2)


SLO_BURN_EASY = [1.8, 2.5, 3.2, 4.0, 5.1]
SLO_BURN_MEDIUM = [2.2, 3.0, 4.5, 5.8, 7.0]
SLO_BURN_HARD = [3.0, 4.5, 6.0, 8.0, 10.0]


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = ["task_easy_cert_expiry", "task_easy_http2_streams", "task_easy_thundering_herd",
                "task_easy_cpu_throttling", "task_easy_cronjob_spike"]
    medium_ids = ["task_medium_hpa_cold_start", "task_medium_canary_false_alert",
                  "task_medium_gateway_rate_limit", "task_medium_circuit_breaker_masking"]
    hard_ids = ["task_hard_pipeline_freshness", "task_hard_quota_cascade", "task_hard_metastable_failure"]

    GOLD = {
        "task_easy_cert_expiry": ["fetch_logs(payment-service)", "rotate_tls_certificate(payment-service)", "declare_resolved"],
        "task_easy_http2_streams": ["get_metrics_detail(api-gateway)", "increase_max_streams(api-gateway)", "declare_resolved"],
        "task_easy_thundering_herd": ["get_metrics_detail(session-service)", "enable_connection_throttle(session-service)", "declare_resolved"],
        "task_easy_cpu_throttling": ["get_metrics_detail(payment-service)", "increase_cpu_limit(payment-service)", "declare_resolved"],
        "task_easy_cronjob_spike": ["get_metrics_detail(analytics-service)", "fetch_logs(analytics-service)", "scale_replicas(analytics-service)", "declare_resolved"],
        "task_medium_hpa_cold_start": ["fetch_logs(recommendation-engine)", "get_metrics_detail(recommendation-engine)", "pre_warm_service(recommendation-engine)", "declare_resolved"],
        "task_medium_canary_false_alert": ["get_metrics_detail(checkout-service)", "rollback_canary(checkout-service)", "declare_resolved"],
        "task_medium_gateway_rate_limit": ["fetch_logs(api-gateway)", "revert_config(api-gateway)", "declare_resolved"],
        "task_medium_circuit_breaker_masking": ["trace_dependencies(product-catalog-service)", "get_metrics_detail(pricing-service)", "scale_replicas(pricing-service)", "declare_resolved"],
        "task_hard_pipeline_freshness": ["inspect_pipeline_topology(feature-pipeline)", "get_metrics_detail(feature-pipeline)", "restart_pipeline_job(feature-pipeline)", "declare_resolved"],
        "task_hard_quota_cascade": ["inspect_quota_usage(ml-inference-service)", "request_quota_increase(ml-inference-service, resource=\"gpu_compute\")", "declare_resolved"],
        "task_hard_metastable_failure": ["get_metrics_detail(search-service)", "disable_retries(api-gateway)", "declare_resolved"],
    }

    def _easy_metrics(tid: str, rng: random.Random) -> dict[str, Any]:
        slo = rng.choice(SLO_BURN_EASY)
        if tid == "task_easy_cert_expiry":
            return {"status": "healthy", "http_server_error_rate": 0.0,
                    "tls_certificate_expiry_seconds": rng.choice([3600, 14400, 28800, 43200, 82800]),
                    "latency_slo_burn_rate_per_hour": slo}
        if tid == "task_easy_http2_streams":
            return {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.0, 0.08), 4),
                    "http2_max_concurrent_streams": rng.choice([80, 100, 128, 150]),
                    "http2_stream_utilization": round(rng.uniform(0.95, 1.00), 2),
                    "http_server_request_duration_p99": rng.choice([8, 10, 12, 15, 18]),
                    "latency_slo_burn_rate_per_hour": slo}
        if tid == "task_easy_thundering_herd":
            return {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.02, 0.08), 4),
                    "session_connection_count": rng.randint(500, 900),
                    "latency_slo_burn_rate_per_hour": slo}
        if tid == "task_easy_cpu_throttling":
            return {"status": "degraded", "http_server_error_rate": 0.0,
                    "process_cpu_throttle_rate": rng.choice([0.72, 0.78, 0.82, 0.87, 0.91, 0.94]),
                    "latency_slo_burn_rate_per_hour": slo}
        if tid == "task_easy_cronjob_spike":
            return {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.02, 0.08), 4),
                    "process_memory_utilization": rng.choice([0.88, 0.91, 0.93, 0.95, 0.97]),
                    "latency_slo_burn_rate_per_hour": slo}
        return {}

    def _medium_metrics(tid: str, rng: random.Random) -> dict[str, Any]:
        slo = rng.choice(SLO_BURN_MEDIUM)
        if tid == "task_medium_hpa_cold_start":
            return {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.05, 0.15), 4),
                    "deployment_ready_replicas": 0, "latency_slo_burn_rate_per_hour": slo}
        if tid == "task_medium_canary_false_alert":
            return {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.03, 0.08), 4),
                    "canary_error_rate": round(rng.uniform(0.05, 0.15), 4),
                    "latency_slo_burn_rate_per_hour": slo}
        if tid == "task_medium_gateway_rate_limit":
            return {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.05, 0.15), 4),
                    "rate_limit_rpm": rng.choice([8, 10, 12, 15, 20]),
                    "latency_slo_burn_rate_per_hour": slo}
        if tid == "task_medium_circuit_breaker_masking":
            return {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.04, 0.12), 4),
                    "circuit_breaker_state": "open", "pricing_service_memory": rng.uniform(0.78, 0.92),
                    "latency_slo_burn_rate_per_hour": slo}
        return {}

    def _hard_metrics(tid: str, rng: random.Random) -> dict[str, Any]:
        slo = rng.choice(SLO_BURN_HARD)
        if tid == "task_hard_pipeline_freshness":
            return {"status": "degraded", "http_server_error_rate": 0.0,
                    "freshness_lag_seconds": rng.randint(380, 650),
                    "queue_depth": rng.randint(12000, 38000),
                    "data_freshness_slo_burn_rate_per_hour": slo}
        if tid == "task_hard_quota_cascade":
            return {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.08, 0.25), 4),
                    "gpu_quota_utilization": 0.0, "cpu_fallback_response_bytes": rng.randint(95, 160),
                    "error_budget_burn_rate": slo}
        if tid == "task_hard_metastable_failure":
            return {"status": "degraded", "http_server_error_rate": round(rng.uniform(0.18, 0.28), 4),
                    "http_server_request_queue_depth": rng.randint(650, 900),
                    "metastable_feedback_loop_active": True,
                    "error_budget_burn_rate": slo}
        return {}

    examples = []
    easy_ticks = [0, 2, 4, 6]
    for tid in easy_ids:
        task = task_map[tid]
        for i in range(4):
            tick = easy_ticks[i]
            m = _easy_metrics(tid, rng)
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "easy",
                "fault_type": task["fault_type"],
                "variation_strategy": "fault_stage,metric_value,alert_phrasing",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("easy", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {m.get('latency_slo_burn_rate_per_hour', 2.0):.1f}×/hr — investigate"],
                    "service_metrics": {task["fault_service"]: m},
                    "logs": {task["fault_service"]: [f"Proactive signal: {tid}"]},
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
            m = _medium_metrics(tid, rng)
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "medium",
                "fault_type": task["fault_type"],
                "variation_strategy": "fault_stage,metric_value,alert_phrasing",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("medium", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {m.get('latency_slo_burn_rate_per_hour', 3.0):.1f}×/hr"],
                    "service_metrics": {task["fault_service"]: m},
                    "logs": {task["fault_service"]: [f"Proactive signal: {tid}"]},
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [], "expected_score_range": [0.50, 0.90], "suboptimal_paths": [],
            })

    hard_ticks = [0, 2, 5, 7]
    for tid in hard_ids:
        task = task_map[tid]
        for i in range(4):
            tick = hard_ticks[i]
            m = _hard_metrics(tid, rng)
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "hard",
                "fault_type": task["fault_type"],
                "variation_strategy": "fault_stage,metric_value,alert_phrasing",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("hard", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {m.get('error_budget_burn_rate', m.get('data_freshness_slo_burn_rate_per_hour', 4.0)):.1f}×/hr"],
                    "service_metrics": {task["fault_service"]: m},
                    "logs": {task["fault_service"]: [f"Proactive signal: {tid}"]},
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
    exs = generate(task_list, rng_seed=13000)
    print(f"Generated {len(exs)} examples")
    for i, ex in enumerate(exs[:3]):
        print(f"\nEx {i}: {ex['task_seed_id']} tier={ex['tier']}")
        svc = ex['task_seed_id'].replace('task_', 'task_')
        # find fault service key
        fault_svc = list(ex['observation']['service_metrics'].keys())[0]
        print(f"  error_rate: {ex['observation']['service_metrics'][fault_svc].get('http_server_error_rate', 'N/A')}")
