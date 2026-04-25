"""
gen_20_mixed_incremental_difficulty.py — Mixed Batch: Incremental Difficulty Progression

Script: gen_20_mixed_incremental_difficulty.py
Batch: 019 (script_num = 20, batch = 019)
Primary axes: variation_strategy (difficulty ramp) + metric_value
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-20
This script teaches the agent that difficulty escalates gradually.
It uses a "difficulty ladder" strategy: easy examples show mild
degradation; medium examples show worsening; hard examples show
severe degradation with compound faults.

Ticks sampled progressively wider per tier:
  Easy:    tick in {1, 2, 3, 4, 5}
  Medium:  tick in {4, 6, 8, 10, 12}
  Hard:    tick in {8, 10, 12, 15, 18, 20}
Budget floor: if budget < 0, recompute from the sampled tick.
"""

import random
import sys
from pathlib import Path
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


def _sample_tick(tier: str, rng: random.Random) -> int:
    """Sample a tick that reflects increasing difficulty."""
    if tier == "easy":
        return rng.choice([1, 2, 3, 4, 5])
    elif tier == "medium":
        return rng.choice([4, 6, 8, 10, 12])
    else:
        return rng.choice([8, 10, 12, 15, 18, 20])


def _escalation_label(tier: str, tick: int) -> str:
    """Human-readable escalation label for the observation."""
    if tier == "easy":
        if tick <= 2: return "mild"
        return "moderate"
    elif tier == "medium":
        if tick <= 6: return "worsening"
        return "severe"
    else:
        if tick <= 12: return "acute"
        return "critical"


# Difficulty anchors: how bad metrics are at each tier and tick
def _build_metrics_snapshot(tier: str, tick: int, fault_service: str,
                             rng: random.Random) -> dict[str, dict[str, Any]]:
    """Build a per-service metrics dict showing escalating fault severity."""
    base = {
        "api-gateway": {"status": "healthy", "http_server_error_rate": 0.01,
                         "requests_per_second": 142.0, "process_memory_utilization": 0.38},
        "user-service": {"status": "healthy", "http_server_error_rate": 0.01,
                         "requests_per_second": 89.0, "process_memory_utilization": 0.41},
        "order-service": {"status": "healthy", "http_server_error_rate": 0.01,
                          "requests_per_second": 55.0, "process_memory_utilization": 0.35},
        "payment-processor": {"status": "healthy", "http_server_error_rate": 0.01,
                              "requests_per_second": 28.0, "process_memory_utilization": 0.29},
        "inventory-service": {"status": "healthy", "http_server_error_rate": 0.01,
                             "requests_per_second": 41.0, "process_memory_utilization": 0.33},
        "notification-service": {"status": "healthy", "http_server_error_rate": 0.01,
                                "requests_per_second": 19.0, "process_memory_utilization": 0.22},
        "audit-log": {"status": "healthy", "http_server_error_rate": 0.01,
                      "requests_per_second": 9.0, "process_memory_utilization": 0.18},
        "auth-service": {"status": "healthy", "http_server_error_rate": 0.01,
                         "requests_per_second": 67.0, "process_memory_utilization": 0.44},
        "checkout-service": {"status": "healthy", "http_server_error_rate": 0.01,
                             "requests_per_second": 38.0, "process_memory_utilization": 0.31},
        "product-catalog": {"status": "healthy", "http_server_error_rate": 0.01,
                            "requests_per_second": 72.0, "process_memory_utilization": 0.27},
        "recommendation-engine": {"status": "healthy", "http_server_error_rate": 0.01,
                                  "requests_per_second": 31.0, "process_memory_utilization": 0.52},
        "redis-cluster": {"status": "healthy", "http_server_error_rate": 0.01,
                          "requests_per_second": 0.0, "process_memory_utilization": 0.61},
        "config-service": {"status": "healthy", "http_server_error_rate": 0.01,
                           "requests_per_second": 22.0, "process_memory_utilization": 0.19},
        "feature-pipeline": {"status": "healthy", "http_server_error_rate": 0.01,
                             "requests_per_second": 14.0, "process_memory_utilization": 0.24},
        "api-gateway-az-b": {"status": "healthy", "http_server_error_rate": 0.01,
                             "requests_per_second": 0.0, "process_memory_utilization": 0.00},
    }

    # Escalate based on tier and tick
    if tier == "easy":
        escalation = min(tick * 0.06, 0.28)
        mem_escalation = min(tick * 0.04, 0.18)
        rps_drop = min(tick * 8, 30)
    elif tier == "medium":
        escalation = min(0.20 + (tick - 4) * 0.09, 0.55)
        mem_escalation = min(0.15 + (tick - 4) * 0.05, 0.28)
        rps_drop = min(15 + (tick - 4) * 10, 65)
    else:
        escalation = min(0.30 + (tick - 8) * 0.11, 0.78)
        mem_escalation = min(0.20 + (tick - 8) * 0.06, 0.40)
        rps_drop = min(25 + (tick - 8) * 12, 90)

    # Inject fault into fault_service
    if fault_service in base:
        base[fault_service]["http_server_error_rate"] = round(escalation, 4)
        base[fault_service]["status"] = _derive_status(escalation)
        base[fault_service]["requests_per_second"] = max(
            10, base[fault_service]["requests_per_second"] - rps_drop
        )
        base[fault_service]["process_memory_utilization"] = round(
            min(base[fault_service]["process_memory_utilization"] + mem_escalation, 0.99), 4
        )

    return base


def _build_alerts(tier: str, tick: int, fault_service: str,
                   escalation: str, rng: random.Random) -> list[str]:
    """Build alert list reflecting escalating severity."""
    base_alerts = [
        f"ERROR: {fault_service} health check degraded",
        f"WARN: {fault_service} error rate above threshold",
    ]

    if tier == "easy":
        if tick >= 3:
            base_alerts.append(f"ERROR: {fault_service} responses timing out")
        if tick >= 5:
            base_alerts.append(f"CRITICAL: {fault_service} error rate {round(min(tick * 0.06, 0.28), 3)}")
    elif tier == "medium":
        base_alerts.append(f"ERROR: {fault_service} responses timing out")
        base_alerts.append(f"CRITICAL: {fault_service} error rate elevated")
        if tick >= 10:
            base_alerts.append(f"ALERT: cascading latency detected in {fault_service}")
        if tick >= 12:
            base_alerts.append(f"CRITICAL: {fault_service} partially unreachable")
    else:
        base_alerts.append(f"ERROR: {fault_service} responses timing out")
        base_alerts.append(f"CRITICAL: {fault_service} error rate elevated")
        base_alerts.append(f"ALERT: cascading latency detected in {fault_service}")
        base_alerts.append(f"CRITICAL: {fault_service} partially unreachable")
        if tick >= 15:
            base_alerts.append(f"EMERGENCY: SLO breach imminent for {fault_service}")
        if tick >= 18:
            base_alerts.append(f"EMERGENCY: multiple services degraded, {fault_service} critical")

    # Red herring for hard
    if tier == "hard" and rng.random() < 0.35:
        red_herring_svc = rng.choice(["auth-service", "notification-service", "audit-log"])
        base_alerts.append(f"ERROR: {red_herring_svc} health check degraded [RED HERRING]")

    return base_alerts


def _build_logs(tier: str, tick: int, fault_service: str,
                rng: random.Random) -> dict[str, list[str]]:
    """Build per-service logs reflecting escalating fault severity."""
    logs: dict[str, list[str]] = {}

    fault_logs = [
        f"ERROR {fault_service} request failed: connection timeout after 30s",
        f"WARN  {fault_service} request rate: {max(5, 80 - tick * 5)} rpm, error rate {round(min(tick * 0.06 if tier == 'easy' else (0.20 + (tick-4)*0.09 if tier == 'medium' else 0.30 + (tick-8)*0.11), 0.78), 3)}",
        f"ERROR {fault_service} upstream connection reset by peer",
    ]
    if tier != "easy":
        fault_logs.append(f"FATAL {fault_service} cannot establish connection to backend")
    if tier == "hard" and tick >= 12:
        fault_logs.append(f"CRITICAL {fault_service} heap overflow detected, restarting")

    logs[fault_service] = fault_logs

    # Add a red herring log for medium/hard
    if tier in ("medium", "hard"):
        rh_svc = rng.choice(["auth-service", "notification-service"])
        logs[rh_svc] = [
            f"INFO {rh_svc} token validation slow: 120ms (threshold 100ms)",
            f"WARN {rh_svc} cache miss rate elevated: 18%",
        ]

    return logs


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


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    rng.seed(rng_seed)

    # 5 Easy tasks × 4 = 20
    easy_ids = []
    for tid in [
        "task_easy_slow_db_query",
        "task_easy_cert_expiry",
        "task_easy_liveness_probe_flap",
        "task_easy_timeout_propagation",
        "task_easy_thread_deadlock",
    ]:
        easy_ids.extend([tid] * 4)

    # 5 Medium tasks × 4 = 20 → trim to 18 (5+4+4+5 pattern)
    medium_ids = []
    for tid in [
        "task_medium_replica_lag",
        "task_medium_retry_storm",
        "task_medium_mtls_rotation",
        "task_medium_single_az_partition",
        "task_medium_ntp_clock_drift",
    ]:
        medium_ids.extend([tid] * 4)
    medium_ids = medium_ids[:18]  # 5+4+4+5 = 18

    # 3 Hard tasks × 4 = 12
    hard_ids = []
    for tid in [
        "task_hard_consensus_degradation",
        "task_hard_pipeline_freshness",
        "task_hard_redis_split_brain",
    ]:
        hard_ids.extend([tid] * 4)

    examples: list[dict] = []

    for task_id in easy_ids + medium_ids + hard_ids:
        task = next(t for t in tasks if t["task_id"] == task_id)
        tier = task["difficulty"]

        tick = _sample_tick(tier, rng)
        budget = _calculate_budget(tier, tick)
        if budget < 0:
            budget = _calculate_budget(tier, tick)

        escalation = _escalation_label(tier, tick)

        metrics = _build_metrics_snapshot(tier, tick, task["fault_service"], rng)
        alerts = _build_alerts(tier, tick, task["fault_service"], escalation, rng)
        logs_dict = _build_logs(tier, tick, task["fault_service"], rng)

        # Gold action sequence
        gold = GOLD.get(task_id, ["get_metrics_detail(" + task["fault_service"] + ")", "declare_resolved"])

        examples.append({
            "task_seed_id": task_id,
            "tier": tier,
            "fault_type": task["fault_type"],
            "variation_strategy": "incremental_difficulty",
            "observation": {
                "tick": tick,
                "budget": budget,
                "escalation": escalation,
                "alerts": alerts,
                "service_metrics": metrics,
                "logs": logs_dict,
            },
            "gold_action_sequence": gold,
            "gold_alternatives": [],
            "expected_score_range": {"min": 0.50, "max": 1.0},
            "suboptimal_paths": [],
        })

    rng.shuffle(examples)
    return examples


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "firewatch_env"))
    from config import TASKS

    task_list = [
        {
            "task_id": tc.task_id,
            "difficulty": tc.difficulty,
            "fault_type": tc.fault_type,
            "fault_service": tc.fault_service,
            "services": list(tc.services) if tc.services else [],
            "red_herrings": list(tc.red_herrings) if tc.red_herrings else [],
            "initial_state_overrides": tc.initial_state_overrides or {},
        }
        for tc in TASKS.values()
    ]

    examples = generate(task_list, rng_seed=20000)
    print(f"Generated {len(examples)} examples")

