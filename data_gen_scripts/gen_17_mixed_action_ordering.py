"""
gen_17_mixed_action_ordering.py — Mixed Batch: Multi-Step Action Ordering Teaching

Script: gen_17_mixed_action_ordering.py
Batch: 016 (script_num = 17, batch = 016)
Primary axes: suboptimal_path (wrong order) + metric_value
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-17
Bootstrap: CONTEXT-BOOTSTRAP.md
Every task has gold sequence where action ORDER matters.
suboptimal_paths documents consequence of wrong ordering.
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
    "task_easy_slow_db_query": ["trace_dependencies(checkout-service)", "get_metrics_detail(user-service)", "rollback_deploy(user-service)", "declare_resolved"],
    "task_easy_cert_expiry": ["fetch_logs(payment-service)", "rotate_tls_certificate(payment-service)", "declare_resolved"],
    "task_easy_liveness_probe_flap": ["get_metrics_detail(payment-processor)", "fetch_logs(payment-processor)", "adjust_probe_timing(payment-processor)", "declare_resolved"],
    "task_easy_timeout_propagation": ["trace_dependencies(order-service)", "fetch_logs(inventory-service)", "optimize_query(inventory-service)", "declare_resolved"],
    "task_easy_thread_deadlock": ["thread_dump(order-service)", "restart_thread_pool(order-service)", "declare_resolved"],
    "task_medium_replica_lag": ["fetch_logs(user-service)", "get_metrics_detail(user-service)", "redirect_reads_to_primary(user-service)", "force_replica_resync(user-service)", "declare_resolved"],
    "task_medium_retry_storm": ["get_metrics_detail(api-gateway)", "trace_dependencies(api-gateway)", "disable_retries(api-gateway)", "configure_retry_backoff(api-gateway)", "declare_resolved"],
    "task_medium_mtls_rotation": ["inspect_mtls_status(payment-service)", "force_cert_rotation(payment-service)", "declare_resolved"],
    "task_medium_single_az_partition": ["get_metrics_detail(api-gateway-az-b)", "drain_availability_zone(az-b)", "declare_resolved"],
    "task_hard_consensus_degradation": ["inspect_consensus_state(config-service)", "isolate_minority_nodes(config-service)", "force_leader_election(config-service)", "declare_resolved"],
    "task_hard_pipeline_freshness": ["inspect_pipeline_topology(feature-pipeline)", "get_metrics_detail(feature-pipeline)", "restart_pipeline_job(feature-pipeline)", "declare_resolved"],
    "task_hard_redis_split_brain": ["inspect_cluster_topology(redis-cluster)", "flush_diverged_keys(redis-cluster)", "force_cluster_resync(redis-cluster)", "declare_resolved"],
}


SUBOPTIMAL = {
    "task_easy_slow_db_query": [{
        "wrong_action_sequence": ["rollback_deploy(checkout-service)", "declare_resolved"],
        "consequence": "checkout-service reverted but user-service N+1 query persists. checkout errors continue from its timeout dependency on user-service. Zero improvement.",
        "score_penalty": 0.30,
    }],
    "task_easy_cert_expiry": [{
        "wrong_action_sequence": ["rotate_tls_certificate(payment-service)", "declare_resolved"],
        "consequence": "Rotation works but investigation skipped. Score 0.85 not 1.00 — acceptable but suboptimal.",
        "partial_credit": True,
        "score_penalty": 0.05,
    }],
    "task_easy_liveness_probe_flap": [{
        "wrong_action_sequence": ["adjust_probe_timing(payment-processor)", "declare_resolved"],
        "consequence": "Action correct but no diagnostic evidence collected. Score capped.",
        "partial_credit": True,
        "score_penalty": 0.08,
    }],
    "task_easy_timeout_propagation": [{
        "wrong_action_sequence": ["optimize_query(order-service)", "declare_resolved"],
        "consequence": "order-service has no query to optimise. Action has no effect. inventory-service slow query persists.",
        "score_penalty": 0.25,
    }],
    "task_easy_thread_deadlock": [{
        "wrong_action_sequence": ["restart_thread_pool(order-service)", "declare_resolved"],
        "consequence": "Thread pool restarts, deadlock clears. But no confirmation of deadlock cycle — if it recurs, agent has no evidence.",
        "partial_credit": True,
        "score_penalty": 0.10,
    }],
    "task_medium_replica_lag": [{
        "wrong_action_sequence": ["force_replica_resync(user-service)", "redirect_reads_to_primary(user-service)", "declare_resolved"],
        "consequence": "Resync takes 8–10 ticks. Stale reads continue during resync. Errors persist for entire resync window. redirect_reads_to_primary would have provided immediate relief.",
        "score_penalty": 0.20,
    }],
    "task_medium_retry_storm": [{
        "wrong_action_sequence": ["configure_retry_backoff(api-gateway)", "disable_retries(api-gateway)", "declare_resolved"],
        "consequence": "Backoff configured but retries still active. Amplification reduced but not eliminated. error_rate drops from 0.80 to ~0.45 but does not clear.",
        "partial_credit": True,
        "score_penalty": 0.15,
    }],
    "task_medium_mtls_rotation": [{
        "wrong_action_sequence": ["force_cert_rotation(payment-service)", "declare_resolved"],
        "consequence": "Rotation succeeds but without inspection, agent doesn't know which service had stale cert. If wrong service rotated, failures persist.",
        "partial_credit": True,
        "score_penalty": 0.10,
    }],
    "task_medium_single_az_partition": [{
        "wrong_action_sequence": ["drain_availability_zone(az-a)", "declare_resolved"],
        "consequence": "Healthy AZ drained. Traffic sent entirely to failing AZ-B. Error rate goes to 90%+. Catastrophic.",
        "score_penalty": 0.40,
    }],
    "task_hard_consensus_degradation": [{
        "wrong_action_sequence": ["force_leader_election(config-service)", "declare_resolved"],
        "consequence": "Election storm involving all 5 nodes including 2 minority. Leader election count spikes to 8+/hour. Stale reads worsen for 2 ticks. Must isolate minority first.",
        "score_penalty": 0.20,
    }],
    "task_hard_pipeline_freshness": [{
        "wrong_action_sequence": ["restart_pipeline_job(feature-pipeline)", "declare_resolved"],
        "consequence": "Job restarts, memory leak clears. But without topology inspection, agent doesn't know if restart fixed the bottleneck stage.",
        "partial_credit": True,
        "score_penalty": 0.08,
    }],
    "task_hard_redis_split_brain": [{
        "wrong_action_sequence": ["force_cluster_resync(redis-cluster)", "declare_resolved"],
        "consequence": "Resync merges cluster with diverged keys intact. Last-write-wins means ~half diverged keys have wrong values permanently. Flush must come before resync.",
        "score_penalty": 0.35,
    }],
}


def _build_easy_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_easy_slow_db_query":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.35, 0.70),
            "http_server_request_duration_p99": rng.choice([5.2, 6.4, 7.8, 8.4, 9.1, 10.2]),
            "db_query_duration_ms": rng.randint(2500, 8000),
            "logs": {"checkout-service": ["checkout-service timeout waiting on user-service DB query"]},
        }
    if tid == "task_easy_cert_expiry":
        expiry = rng.choice([3600, 14400, 28800, 43200, 82800])
        return {
            "status": "degraded",
            "http_server_error_rate": round(rng.uniform(0.0, 0.08), 4),
            "tls_certificate_expiry_seconds": expiry,
            "logs": {"payment-service": [f"TLS cert expires in {expiry}s — check rotation"]},
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
            "logs": {"payment-processor": [f"Liveness probe failing — restart #{restart_count} in last hour"]},
        }
    if tid == "task_easy_timeout_propagation":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.30, 0.45),
            "http_server_request_duration_p99": rng.choice([5.2, 6.4, 7.8, 8.4, 9.1, 10.2]),
            "logs": {"order-service": ["order-service timeout: inventory-service not responding within SLA"]},
        }
    if tid == "task_easy_thread_deadlock":
        blocked = rng.choice([42, 50, 55, 58, 60])
        return {
            "status": "critical",
            "http_server_error_rate": rng.choice([0.88, 0.92, 0.97, 0.99, 1.00]),
            "runtime_blocked_thread_count": blocked,
            "wait_ratio": round(rng.uniform(0.88, 1.00), 2),
            "logs": {"order-service": [f"Thread deadlock detected: {blocked} threads blocked on owned mutex"]},
        }
    return {}


def _build_medium_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_medium_replica_lag":
        db_lag = rng.randint(22, 80)
        write_err = round(rng.uniform(0.0, 0.02), 4)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.28, 0.58),
            "db_replication_lag_seconds": db_lag,
            "http_server_write_path_error_rate": write_err,
            "logs": {"user-service": [f"DB replication lag: {db_lag}s — stale reads occurring"]},
        }
    if tid == "task_medium_retry_storm":
        retry_count = rng.randint(4, 8)
        rps_mult = round(retry_count * 0.85, 2)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.60, 0.90),
            "http_client_retry_count": retry_count,
            "effective_rps_multiplier": rps_mult,
            "logs": {"api-gateway": [f"Retry storm: {retry_count} retries/request, multiplier {rps_mult:.2f}×"]},
        }
    if tid == "task_medium_mtls_rotation":
        failure_rate = rng.choice([0.55, 0.68, 0.78, 0.88, 0.95])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.55, 0.95),
            "sidecar_cert_rotation_status": "stale",
            "mtls_handshake_failure_rate": failure_rate,
            "logs": {"payment-service": [f"sidecar cert stale — mTLS handshake failure rate {failure_rate:.0%}"]},
        }
    if tid == "task_medium_single_az_partition":
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.75, 0.95),
            "az_b_error_rate": rng.uniform(0.75, 0.95),
            "az_a_error_rate": round(rng.uniform(0.0, 0.03), 4),
            "logs": {"api-gateway-az-b": ["AZ-B partition: all requests timing out"]},
        }
    return {}


def _build_hard_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_hard_consensus_degradation":
        data_age = rng.choice([420, 480, 540, 600, 720])
        elections = rng.choice([3, 4, 5, 6, 8])
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.45, 0.75),
            "config_data_age_seconds": data_age,
            "consensus_leader_election_count": elections,
            "logs": {"config-service": [f"Consensus: minority data age {data_age}s, elections {elections}/hour"]},
        }
    if tid == "task_hard_pipeline_freshness":
        return {
            "status": "degraded",
            "http_server_error_rate": 0.0,
            "freshness_lag_seconds": rng.randint(380, 650),
            "pipeline_throughput_ratio": rng.choice([0.72, 0.78, 0.82, 0.86, 0.90]),
            "logs": {"feature-pipeline": ["Feature pipeline bottleneck: memory pressure causing lag"]},
        }
    if tid == "task_hard_redis_split_brain":
        diverged_keys = rng.randint(1200, 4800)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.35, 0.65),
            "redis_diverged_key_count": diverged_keys,
            "cluster_sync_state": "diverged",
            "logs": {"redis-cluster": [f"Split brain: {diverged_keys} keys diverged between primary and replica"]},
        }
    return {}


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = [
        "task_easy_slow_db_query", "task_easy_cert_expiry", "task_easy_liveness_probe_flap",
        "task_easy_timeout_propagation", "task_easy_thread_deadlock",
    ]
    medium_ids = [
        "task_medium_replica_lag", "task_medium_retry_storm",
        "task_medium_mtls_rotation", "task_medium_single_az_partition",
    ]
    hard_ids = [
        "task_hard_consensus_degradation", "task_hard_pipeline_freshness",
        "task_hard_redis_split_brain",
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
                "variation_strategy": "suboptimal_path,metric_value",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("easy", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {task["fault_service"]: m},
                    "logs": m.get("logs", {task["fault_service"]: [f"fault: {tid}"]}),
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [],
                "expected_score_range": [0.70, 1.0],
                "suboptimal_paths": SUBOPTIMAL.get(tid, []).copy(),
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
                "variation_strategy": "suboptimal_path,metric_value",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("medium", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {task["fault_service"]: m},
                    "logs": m.get("logs", {task["fault_service"]: [f"fault: {tid}"]}),
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [],
                "expected_score_range": [0.50, 0.90],
                "suboptimal_paths": SUBOPTIMAL.get(tid, []).copy(),
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
                "variation_strategy": "suboptimal_path,metric_value,adversarial_content",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("hard", tick),
                    "alerts": [f"[WARNING] {task['fault_service']} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {task["fault_service"]: m},
                    "logs": m.get("logs", {task["fault_service"]: [f"fault: {tid}"]}),
                },
                "gold_action_sequence": GOLD.get(tid, ["declare_resolved"]).copy(),
                "gold_alternatives": [],
                "expected_score_range": [0.30, 0.80],
                "suboptimal_paths": SUBOPTIMAL.get(tid, []).copy(),
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
    exs = generate(task_list, rng_seed=17000)
    print(f"Generated {len(exs)} examples")
    for i, ex in enumerate(exs[:3]):
        print(f"\nEx {i}: {ex['task_seed_id']} tier={ex['tier']}")
        print(f"  gold: {ex['gold_action_sequence']}")
        print(f"  suboptimal_paths: {len(ex['suboptimal_paths'])}")