"""
gen_21_mixed_network_partition.py — Mixed Batch: network_partition Fault Type Thematic

Script: gen_21_mixed_network_partition.py
Batch: 020 (script_num = 21, batch = 020)
Primary axes: metric_value + fault_stage + noise_injection
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-21
Bootstrap: CONTEXT-BOOTSTRAP.md
Tasks involve service-to-service communication failures.
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
    "task_easy_timeout_propagation": ["trace_dependencies(order-service)", "fetch_logs(inventory-service)", "optimize_query(inventory-service)", "declare_resolved"],
    "task_easy_dns_nxdomain": ["fetch_logs(payment-service)", "update_service_endpoint(payment-service)", "declare_resolved"],
    "task_easy_liveness_probe_flap": ["get_metrics_detail(payment-processor)", "fetch_logs(payment-processor)", "adjust_probe_timing(payment-processor)", "declare_resolved"],
    "task_easy_image_pull_backoff": ["fetch_logs(recommendation-engine)", "rollback_deploy(recommendation-engine)", "declare_resolved"],
    "task_easy_jwt_clock_skew": ["fetch_logs(auth-service)", "force_ntp_sync(auth-service)", "declare_resolved"],
    "task_medium_asymmetric_blast": ["trace_dependencies(auth-service)", "trace_dependencies(payment-service)", "get_metrics_detail(db-proxy)", "restart_service(db-proxy)", "declare_resolved"],
    "task_medium_replica_lag": ["fetch_logs(user-service)", "get_metrics_detail(user-service)", "redirect_reads_to_primary(user-service)", "force_replica_resync(user-service)", "declare_resolved"],
    "task_medium_single_az_partition": ["get_metrics_detail(api-gateway-az-b)", "drain_availability_zone(az-b)", "declare_resolved"],
    "task_medium_mtls_rotation": ["inspect_mtls_status(payment-service)", "force_cert_rotation(payment-service)", "declare_resolved"],
    "task_hard_gray_failure": ["get_metrics_detail(auth-service)", "fetch_logs(auth-service)", "inspect_network_policy(auth-service)", "revert_network_policy(auth-service)", "declare_resolved"],
    "task_hard_redis_split_brain": ["inspect_cluster_topology(redis-cluster)", "flush_diverged_keys(redis-cluster)", "force_cluster_resync(redis-cluster)", "declare_resolved"],
    "task_hard_partial_infra_asymmetric": ["inspect_infrastructure_topology()", "get_metrics_detail(infrastructure)", "remediate_infrastructure()", "declare_resolved"],
}


def _build_easy_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_easy_timeout_propagation":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.30, 0.45),
            "http_server_request_duration_p99": rng.choice([5.2, 6.4, 7.8, 8.4, 9.1, 10.2]),
            "logs": {"order-service": ["order-service timeout: inventory-service not responding"]},
        }
    if tid == "task_easy_dns_nxdomain":
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.0, 0.08),
            "dns_lookup_failure_rate": rng.uniform(0.55, 0.88),
            "logs": {"payment-service": ["DNS lookup failed: NXDOMAIN for payment-service.prod.svc.cluster.local"]},
        }
    if tid == "task_easy_liveness_probe_flap":
        restart_count = rng.randint(4, 10)
        startup_dur = rng.uniform(3.8, 6.1)
        return {
            "status": "critical",
            "http_server_error_rate": rng.choice([0.65, 0.72, 0.80, 0.88, 0.95, 1.00]),
            "restart_count": restart_count,
            "startup_duration_s": startup_dur,
            "liveness_probe_timeout_s": round(startup_dur * rng.uniform(0.6, 0.85), 2),
            "logs": {"payment-processor": [f"Liveness probe failing — restart #{restart_count}"]},
        }
    if tid == "task_easy_image_pull_backoff":
        backoff_s = rng.randint(30, 300)
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.40, 0.65),
            "image_pull_backoff_seconds": backoff_s,
            "logs": {"recommendation-engine": [f"Image pull backoff: {backoff_s}s — unable to pull v{rng.randint(2,5)}.X.Y"]},
        }
    if tid == "task_easy_jwt_clock_skew":
        offset = -rng.choice([240, 270, 305, 340, 380, 420])
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.30, 0.65),
            "system_clock_offset_seconds": offset,
            "logs": {"auth-service": [f"system clock {abs(offset)}s BEHIND UTC — JWT validation failing"]},
        }
    return {}


def _build_medium_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_medium_asymmetric_blast":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.72, 0.92),
            "auth_error_rate": rng.uniform(0.72, 0.92),
            "payment_error_rate": rng.uniform(0.16, 0.30),
            "user_error_rate": rng.uniform(0.04, 0.12),
            "dbproxy_error_rate": rng.uniform(0.82, 0.98),
            "logs": {"db-proxy": ["db-proxy connection pool exhausted"]},
        }
    if tid == "task_medium_replica_lag":
        db_lag = rng.randint(22, 80)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.28, 0.58),
            "db_replication_lag_seconds": db_lag,
            "http_server_write_path_error_rate": round(rng.uniform(0.0, 0.02), 4),
            "logs": {"user-service": [f"DB replication lag: {db_lag}s"]},
        }
    if tid == "task_medium_single_az_partition":
        az_b_weight = 0.33
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.82, 0.95),
            "az_b_error_rate": rng.uniform(0.82, 0.95),
            "az_b_weight": az_b_weight,
            "packet_loss_pct": rng.randint(88, 97),
            "logs": {"api-gateway-az-b": ["AZ-B partition: all requests timing out"]},
        }
    if tid == "task_medium_mtls_rotation":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.55, 0.95),
            "sidecar_cert_rotation_status": "stale",
            "mtls_handshake_failure_rate": rng.choice([0.55, 0.68, 0.78, 0.88, 0.95]),
            "logs": {"payment-service": ["mTLS handshake failure — cert stale"]},
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
            "logs": {
                "auth-service": [f"Gray failure: auth-service responding slowly (p99={p99:.1f}s)"],
                "user-service": [f"Error rate {rng.uniform(0.04, 0.07):.2f} — red herring"],
                "payment-service": [f"Error rate {rng.uniform(0.04, 0.07):.2f} — red herring"],
            },
        }
    if tid == "task_hard_redis_split_brain":
        diverged = rng.randint(12000, 80000)
        lag = rng.randint(45, 240)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.35, 0.65),
            "redis_diverged_key_count": diverged,
            "data_inconsistency_rate": rng.uniform(0.08, 0.32),
            "db_replication_lag_seconds": lag,
            "logs": {"redis-cluster": [f"Split brain: {diverged} keys diverged, lag {lag}s"]},
        }
    if tid == "task_hard_partial_infra_asymmetric":
        affected = rng.randint(2, 4)
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.12, 0.40),
            "affected_service_count": affected,
            "io_error_rate": rng.uniform(0.12, 0.40),
            "logs": {
                "infrastructure": ["UPS/I/O error: infrastructure write failures detected"],
                "api-gateway": [f"{affected} services affected by I/O errors"],
            },
        }
    return {}


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = [
        "task_easy_timeout_propagation", "task_easy_dns_nxdomain", "task_easy_liveness_probe_flap",
        "task_easy_image_pull_backoff", "task_easy_jwt_clock_skew",
    ]
    medium_ids = [
        "task_medium_asymmetric_blast", "task_medium_replica_lag",
        "task_medium_single_az_partition", "task_medium_mtls_rotation",
    ]
    hard_ids = [
        "task_hard_gray_failure", "task_hard_redis_split_brain", "task_hard_partial_infra_asymmetric",
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
                "variation_strategy": "metric_value,fault_stage,adversarial_content",
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
    exs = generate(task_list, rng_seed=21000)
    print(f"Generated {len(exs)} examples")
    for i, ex in enumerate(exs[:3]):
        print(f"\nEx {i}: {ex['task_seed_id']} tier={ex['tier']}")
        svc = list(ex['observation']['service_metrics'].keys())[0]
        m = ex['observation']['service_metrics'][svc]
        print(f"  error_rate: {m.get('http_server_error_rate', 'N/A')}")