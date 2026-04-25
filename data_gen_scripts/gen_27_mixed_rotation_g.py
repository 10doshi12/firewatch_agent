"""
gen_27_mixed_rotation_g.py — Mixed Batch: Complete Task Rotation G

Script: gen_27_mixed_rotation_g.py
Batch: 026 (script_num = 27, batch = 026)
Primary axes: metric_value + fault_stage + noise_injection
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-27
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


GOLD = {
    "task_easy_rbac_403": ["fetch_logs(notification-service)", "grant_rbac_permission(notification-service)", "declare_resolved"],
    "task_easy_cpu_throttling": ["get_metrics_detail(payment-service)", "increase_cpu_limit(payment-service)", "declare_resolved"],
    "task_easy_slow_db_query": ["trace_dependencies(checkout-service)", "get_metrics_detail(user-service)", "rollback_deploy(user-service)", "declare_resolved"],
    "task_easy_http2_streams": ["get_metrics_detail(api-gateway)", "increase_max_streams(api-gateway)", "declare_resolved"],
    "task_easy_cert_expiry": ["fetch_logs(payment-service)", "rotate_tls_certificate(payment-service)", "declare_resolved"],
    "task_medium_canary_false_alert": ["get_metrics_detail(checkout-service)", "rollback_canary(checkout-service)", "declare_resolved"],
    "task_medium_circuit_breaker_masking": ["trace_dependencies(product-catalog-service)", "get_metrics_detail(pricing-service)", "scale_replicas(pricing-service)", "declare_resolved"],
    "task_medium_bg_traffic_leak": ["get_metrics_detail(api-gateway)", "complete_traffic_switch(api-gateway)", "declare_resolved"],
    "task_medium_grpc_deadline": ["get_metrics_detail(payment-service)", "trace_dependencies(order-service)", "enable_deadline_propagation(order-service)", "declare_resolved"],
    "task_hard_gray_failure": ["get_metrics_detail(auth-service)", "fetch_logs(auth-service)", "inspect_network_policy(auth-service)", "revert_network_policy(auth-service)", "declare_resolved"],
    "task_hard_pipeline_freshness": ["inspect_pipeline_topology(feature-pipeline)", "get_metrics_detail(feature-pipeline)", "restart_pipeline_job(feature-pipeline)", "declare_resolved"],
}


def _build_easy_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_easy_rbac_403":
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.50, 0.80),
            "rbac_permission_status": "forbidden",
            "logs": {"notification-service": [f"RBAC: forbidden — ServiceAccount project-sa lacking configmaps read"]},
        }
    if tid == "task_easy_cpu_throttling":
        return {
            "status": "degraded",
            "http_server_error_rate": 0.0,
            "process_cpu_throttle_rate": rng.choice([0.72, 0.78, 0.82, 0.87, 0.91, 0.94]),
            "logs": {"payment-service": [f"CPU throttle rate {rng.choice([0.72, 0.78, 0.82])} — latency-only invariant"]},
        }
    if tid == "task_easy_slow_db_query":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.35, 0.70),
            "http_server_request_duration_p99": rng.choice([5.2, 6.4, 7.8, 8.4, 9.1, 10.2]),
            "db_query_duration_ms": rng.randint(2500, 8000),
            "logs": {"checkout-service": ["checkout-service timeout waiting on user-service"]},
        }
    if tid == "task_easy_http2_streams":
        max_streams = rng.choice([80, 100, 128, 150])
        return {
            "status": "degraded",
            "http_server_error_rate": round(rng.uniform(0.0, 0.08), 4),
            "http2_max_concurrent_streams": max_streams,
            "http2_stream_utilization": round(rng.uniform(0.95, 1.00), 2),
            "logs": {"api-gateway": [f"stream limit {max_streams} — bimodal invariant"]},
        }
    if tid == "task_easy_cert_expiry":
        expiry = rng.choice([3600, 14400, 28800, 43200, 82800])
        return {
            "status": "degraded",
            "http_server_error_rate": 0.0,
            "tls_certificate_expiry_seconds": expiry,
            "logs": {"payment-service": [f"TLS cert expires in {expiry}s"]},
        }
    return {}


def _build_medium_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_medium_canary_false_alert":
        canary_weight = rng.choice([0.08, 0.10, 0.12, 0.15, 0.20])
        canary_err = rng.choice([0.38, 0.42, 0.45, 0.50, 0.55])
        stable_err = rng.uniform(0.01, 0.02)
        agg_err = round(canary_weight * canary_err + (1 - canary_weight) * stable_err, 4)
        return {
            "status": "degraded",
            "http_server_error_rate": agg_err,
            "canary_traffic_weight": canary_weight,
            "canary_error_rate": canary_err,
            "stable_error_rate": stable_err,
        }
    if tid == "task_medium_circuit_breaker_masking":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.00, 0.01),
            "process_memory_utilization": rng.choice([0.78, 0.82, 0.86, 0.88, 0.92]),
            "circuit_breaker_state": "open",
        }
    if tid == "task_medium_bg_traffic_leak":
        blue_frac = rng.choice([0.05, 0.10, 0.15, 0.20, 0.25])
        old_err = rng.choice([0.45, 0.55, 0.65, 0.75])
        return {
            "status": "degraded",
            "http_server_error_rate": round(blue_frac * old_err, 4),
            "blue_environment_traffic_fraction": blue_frac,
            "blue_environment_error_rate": old_err,
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
    if tid == "task_hard_gray_failure":
        p99 = rng.uniform(5.5, 9.0)
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.12, 0.25),
            "network_packet_loss_rate_inbound": rng.uniform(0.12, 0.25),
            "http_server_request_duration_p99": p99,
            "http_server_request_duration_p50": rng.uniform(0.08, 0.12),
        }
    if tid == "task_hard_pipeline_freshness":
        return {
            "status": "degraded",
            "http_server_error_rate": 0.0,
            "freshness_lag_seconds": rng.randint(380, 650),
            "pipeline_throughput_ratio": rng.choice([0.72, 0.78, 0.82, 0.86, 0.90]),
        }
    return {}


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = ["task_easy_rbac_403", "task_easy_cpu_throttling", "task_easy_slow_db_query",
                "task_easy_http2_streams", "task_easy_cert_expiry"]
    medium_ids = ["task_medium_canary_false_alert", "task_medium_circuit_breaker_masking",
                  "task_medium_bg_traffic_leak", "task_medium_grpc_deadline"]
    hard_ids = ["task_hard_gray_failure", "task_hard_pipeline_freshness"]

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
                "variation_strategy": "metric_value,fault_stage,noise_injection",
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
                "variation_strategy": "metric_value,fault_stage,noise_injection",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("medium", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {task["fault_service"]: m},
                    "logs": m.get("logs", {task["fault_service"]: [f"fault: {tid}"]}),
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [], "expected_score_range": [0.50, 0.90], "suboptimal_paths": [],
            })

    # Hard: 3 tasks × 4 = 12 (need exactly 12)
    hard_ids_full = ["task_hard_gray_failure", "task_hard_pipeline_freshness",
                     "task_hard_consensus_degradation"]
    GOLD["task_hard_consensus_degradation"] = ["inspect_consensus_state(config-service)", "isolate_minority_nodes(config-service)", "force_leader_election(config-service)", "declare_resolved"]
    GOLD["task_hard_cache_corruption"] = ["get_metrics_detail(cache)", "evict_corrupted_keys(cache)", "declare_resolved"]
    for tid in hard_ids_full:
        task = task_map[tid]
        for i in range(4):
            tick = [0, 2, 5, 7][i]
            if tid == "task_hard_consensus_degradation":
                m = {
                    "status": "critical",
                    "http_server_error_rate": rng.uniform(0.45, 0.75),
                    "config_data_age_seconds": rng.choice([420, 480, 540, 600, 720]),
                    "consensus_leader_election_count": rng.choice([3, 4, 5, 6, 8]),
                }
            elif tid == "task_hard_cache_corruption":
                m = {
                    "status": "critical",
                    "http_server_error_rate": rng.uniform(0.40, 0.72),
                    "cache_checksum_errors": rng.randint(80, 350),
                    "cache_hit_rate": rng.uniform(0.08, 0.22),
                }
            else:
                m = _build_hard_metrics(tid, rng, tick)
            slo_burn = rng.choice([3.0, 4.5, 6.0, 8.0, 10.0])
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "hard",
                "fault_type": task["fault_type"],
                "variation_strategy": "metric_value,fault_stage,adversarial_content",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("hard", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {task["fault_service"]: m},
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
    exs = generate(task_list, rng_seed=27000)
    print(f"Generated {len(exs)} examples")