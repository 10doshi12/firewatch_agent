"""
gen_29_mixed_suboptimal_subst.py — Mixed Batch: Suboptimal Paths + Substitution Combined

Script: gen_29_mixed_suboptimal_subst.py
Batch: 028 (script_num = 29, batch = 028)
Primary axes: suboptimal_path + service_pool_substitution + metric_value
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-29
Bootstrap: CONTEXT-BOOTSTRAP.md
Combines wrong-action teaching with service substitution.
"""

import random
from typing import Any

HEALTHY_ERROR_RATE = 0.03

SUBSTITUTION_POOL = {
    "api-gateway": ["ingress-controller", "edge-proxy"],
    "auth-service": ["identity-service", "sso-service"],
    "payment-service": ["billing-service", "transaction-service"],
    "user-service": ["profile-service", "account-service"],
    "db-proxy": ["data-proxy", "query-router"],
    "checkout-service": ["order-service", "cart-service"],
    "notification-service": ["alert-service", "messaging-service"],
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
    "task_easy_quota_runaway": ["trace_dependencies(user-service)", "get_metrics_detail(notification-service)", "rollback_deploy(notification-service)", "declare_resolved"],
    "task_easy_timeout_propagation": ["trace_dependencies(order-service)", "fetch_logs(inventory-service)", "optimize_query(inventory-service)", "declare_resolved"],
    "task_easy_rate_limiter_misconfig": ["fetch_logs(api-gateway)", "revert_config(api-gateway)", "declare_resolved"],
    "task_easy_liveness_probe_flap": ["get_metrics_detail(payment-processor)", "fetch_logs(payment-processor)", "adjust_probe_timing(payment-processor)", "declare_resolved"],
    "task_easy_pool_restart_cycle": ["fetch_logs(auth-service)", "revert_config(auth-service)", "declare_resolved"],
    "task_medium_asymmetric_blast": ["trace_dependencies(auth-service)", "trace_dependencies(payment-service)", "get_metrics_detail(db-proxy)", "restart_service(db-proxy)", "declare_resolved"],
    "task_medium_cache_eviction_storm": ["trace_dependencies(user-db)", "get_metrics_detail(cache-service)", "fetch_logs(cache-service)", "increase_cache_memory(cache-service)", "declare_resolved"],
    "task_medium_circuit_breaker_masking": ["trace_dependencies(product-catalog-service)", "get_metrics_detail(pricing-service)", "scale_replicas(pricing-service)", "declare_resolved"],
    "task_medium_rollout_quota_exhaustion": ["trace_dependencies(auth-service)", "get_metrics_detail(api-gateway)", "rollback_deploy(api-gateway)", "declare_resolved"],
    "task_hard_config_drift_noise": ["get_metrics_detail(api-gateway)", "fetch_logs(api-gateway)", "revert_config(api-gateway)", "declare_resolved"],
    "task_hard_multiteam_dual_fault": ["trace_dependencies(checkout-service)", "rollback_deploy(auth-service)", "scale_replicas(payment-service)", "declare_resolved"],
}


SUBOPTIMAL = {
    "task_easy_quota_runaway": [{"wrong_action_sequence": ["scale_replicas(user-service)", "declare_resolved"], "consequence": "Temporary relief 2 ticks, overcalling persists, errors return.", "score_penalty": 0.20}],
    "task_easy_timeout_propagation": [{"wrong_action_sequence": ["optimize_query(order-service)", "declare_resolved"], "consequence": "order-service has no query to optimise. inventory-service slow query persists.", "score_penalty": 0.30}],
    "task_easy_rate_limiter_misconfig": [{"wrong_action_sequence": ["scale_replicas(api-gateway)", "declare_resolved"], "consequence": "More instances, same wrong rate limit, 429 unchanged.", "score_penalty": 0.25}],
    "task_easy_liveness_probe_flap": [{"wrong_action_sequence": ["restart_service(payment-processor)", "declare_resolved"], "consequence": "Re-enters same probe fail loop immediately.", "score_penalty": 0.25}],
    "task_easy_pool_restart_cycle": [{"wrong_action_sequence": ["restart_service(auth-service)", "declare_resolved"], "consequence": "Pool re-exhausts within 2 ticks, restart cycle resumes.", "score_penalty": 0.20}],
    "task_medium_asymmetric_blast": [{"wrong_action_sequence": ["restart_service(auth-service)", "declare_resolved"], "consequence": "Errors clear momentarily, return in 2 ticks.", "score_penalty": 0.30}],
    "task_medium_cache_eviction_storm": [{"wrong_action_sequence": ["revert_config(user-db)", "declare_resolved"], "consequence": "No-op, db config is fine, cache evictions continue.", "score_penalty": 0.25}],
    "task_medium_circuit_breaker_masking": [{"wrong_action_sequence": ["circuit_break(product-catalog-service)", "declare_resolved"], "consequence": "No-op — product-catalog already CB-protected.", "score_penalty": 0.30}],
    "task_medium_rollout_quota_exhaustion": [{"wrong_action_sequence": ["scale_replicas(auth-service)", "declare_resolved"], "consequence": "3-tick relief, retry bug persists, fills again.", "score_penalty": 0.20}],
    "task_hard_config_drift_noise": [{"wrong_action_sequence": ["restart_service(auth-service)", "declare_resolved"], "consequence": "Wrong action from adversarial injection.", "score_penalty": 0.35}],
    "task_hard_multiteam_dual_fault": [{"wrong_action_sequence": ["rollback_deploy(auth-service)", "declare_resolved"], "consequence": "checkout error drops 0.65 to 0.35, payment leak persists.", "score_penalty": 0.25}],
}


def _build_easy_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_easy_quota_runaway":
        return {"status": "critical", "http_server_error_rate": rng.uniform(0.45, 0.70), "last_deployment_age_seconds": rng.choice([60, 90, 120, 180])}
    if tid == "task_easy_timeout_propagation":
        return {"status": "critical", "http_server_error_rate": rng.uniform(0.30, 0.45), "http_server_request_duration_p99": rng.choice([5.2, 6.4, 7.8, 8.4])}
    if tid == "task_easy_rate_limiter_misconfig":
        return {"status": "critical", "http_server_error_rate": rng.uniform(0.55, 0.88), "rate_limit_rpm": rng.choice([50, 75, 100])}
    if tid == "task_easy_liveness_probe_flap":
        return {"status": "critical", "http_server_error_rate": rng.choice([0.65, 0.72, 0.80, 0.88]), "restart_count": rng.randint(4, 10)}
    if tid == "task_easy_pool_restart_cycle":
        return {"status": "critical", "http_server_error_rate": rng.choice([0.52, 0.58, 0.61, 0.66]), "process_open_file_descriptors": rng.randint(2, 6)}
    return {}


def _build_medium_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_medium_asymmetric_blast":
        return {"status": "critical", "http_server_error_rate": rng.uniform(0.72, 0.92), "dbproxy_error_rate": rng.uniform(0.82, 0.98)}
    if tid == "task_medium_cache_eviction_storm":
        return {"status": "degraded", "http_server_error_rate": rng.uniform(0.18, 0.42), "cache_memory_utilization": rng.choice([0.95, 0.97, 0.98])}
    if tid == "task_medium_circuit_breaker_masking":
        return {"status": "critical", "http_server_error_rate": rng.uniform(0.00, 0.01), "process_memory_utilization": rng.choice([0.78, 0.82, 0.86]), "circuit_breaker_state": "open"}
    if tid == "task_medium_rollout_quota_exhaustion":
        return {"status": "critical", "http_server_error_rate": rng.uniform(0.55, 0.85), "http_client_retry_count": 10}
    return {}


def _build_hard_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_hard_config_drift_noise":
        return {"status": "critical", "http_server_error_rate": rng.uniform(0.52, 0.78), "process_open_file_descriptors": rng.randint(2, 6)}
    if tid == "task_hard_multiteam_dual_fault":
        auth_err = rng.choice([0.38, 0.50, 0.55, 0.60, 0.65])
        mem_trend = [round(rng.uniform(0.65, 0.72), 4), round(rng.uniform(0.72, 0.80), 4), round(rng.uniform(0.80, 0.86), 4)]
        return {"status": "critical", "http_server_error_rate": rng.uniform(0.30, 0.55), "auth_error_rate": auth_err, "process_memory_utilization": mem_trend[-1], "memory_trend": mem_trend}
    return {}


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = ["task_easy_quota_runaway", "task_easy_timeout_propagation", "task_easy_rate_limiter_misconfig",
                "task_easy_liveness_probe_flap", "task_easy_pool_restart_cycle"]
    medium_ids = ["task_medium_asymmetric_blast", "task_medium_cache_eviction_storm",
                   "task_medium_circuit_breaker_masking", "task_medium_rollout_quota_exhaustion"]
    hard_ids = ["task_hard_config_drift_noise", "task_hard_multiteam_dual_fault"]

    examples = []

    # Easy: 5 tasks × 4 = 20
    for tid in easy_ids:
        task = task_map[tid]
        sub_svcs = [task["fault_service"]]
        for i in range(4):
            tick = [0, 2, 4, 6][i]
            sub_map = {svc: _substitute(svc, rng) for svc in sub_svcs}
            m = _build_easy_metrics(tid, rng, tick)
            slo_burn = rng.choice([1.8, 2.5, 3.2, 4.0, 5.1])
            fault_svc_sub = _apply_subst(task["fault_service"], sub_map)
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "easy",
                "fault_type": task["fault_type"],
                "variation_strategy": "suboptimal_path,service_pool_substitution,metric_value",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("easy", tick),
                    "alerts": [f"[WARNING] {fault_svc_sub} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {fault_svc_sub: m},
                    "logs": {fault_svc_sub: [f"fault: {tid}"]},
                },
                "gold_action_sequence": [(_apply_subst(a, sub_map) if '(' in a else a) for a in GOLD.get(tid, ["declare_resolved"])],
                "gold_alternatives": [], "expected_score_range": [0.70, 1.0],
                "suboptimal_paths": SUBOPTIMAL.get(tid, []).copy(),
            })

    # Medium: 4 tasks × ~4-5 = 18
    medium_counts = [5, 4, 4, 5]
    for idx, tid in enumerate(medium_ids):
        task = task_map[tid]
        sub_svcs = [task["fault_service"]]
        count = medium_counts[idx]
        for i in range(count):
            tick = [0, 1, 3, 5][i % 4]
            sub_map = {svc: _substitute(svc, rng) for svc in sub_svcs}
            m = _build_medium_metrics(tid, rng, tick)
            slo_burn = rng.choice([2.2, 3.0, 4.5, 5.8, 7.0])
            fault_svc_sub = _apply_subst(task["fault_service"], sub_map)
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "medium",
                "fault_type": task["fault_type"],
                "variation_strategy": "suboptimal_path,service_pool_substitution,metric_value",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("medium", tick),
                    "alerts": [f"[WARNING] {fault_svc_sub} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {fault_svc_sub: m},
                    "logs": {fault_svc_sub: [f"fault: {tid}"]},
                },
                "gold_action_sequence": [(_apply_subst(a, sub_map) if '(' in a else a) for a in GOLD.get(tid, ["declare_resolved"])],
                "gold_alternatives": [], "expected_score_range": [0.50, 0.90],
                "suboptimal_paths": SUBOPTIMAL.get(tid, []).copy(),
            })

    # Hard: 3 tasks × 4 = 12
    hard_ids_full = ["task_hard_config_drift_noise", "task_hard_multiteam_dual_fault",
                     "task_hard_gray_failure"]
    GOLD["task_hard_gray_failure"] = ["get_metrics_detail(auth-service)", "fetch_logs(auth-service)", "inspect_network_policy(auth-service)", "revert_network_policy(auth-service)", "declare_resolved"]
    GOLD["task_hard_cache_corruption"] = ["get_metrics_detail(cache)", "evict_corrupted_keys(cache)", "declare_resolved"]
    SUBOPTIMAL["task_hard_gray_failure"] = [{"wrong_action_sequence": ["restart_service(auth-service)", "declare_resolved"], "consequence": "Gray failure requires network policy revert, not restart.", "score_penalty": 0.30}]
    SUBOPTIMAL["task_hard_cache_corruption"] = [{"wrong_action_sequence": ["scale_replicas(cache)", "declare_resolved"], "consequence": "Scaling doesn't fix corrupted data — must evict.", "score_penalty": 0.25}]
    for tid in hard_ids_full:
        task = task_map[tid]
        sub_svcs = [task["fault_service"]] if task["fault_service"] in SUBSTITUTION_POOL else []
        for i in range(4):
            tick = [0, 2, 5, 7][i]
            sub_map = {svc: _substitute(svc, rng) for svc in sub_svcs}
            if tid == "task_hard_gray_failure":
                p99 = rng.uniform(5.5, 9.0)
                m = {"status": "degraded", "http_server_error_rate": rng.uniform(0.12, 0.25), "network_packet_loss_rate_inbound": rng.uniform(0.12, 0.25), "http_server_request_duration_p99": p99}
            elif tid == "task_hard_cache_corruption":
                m = {"status": "critical", "http_server_error_rate": rng.uniform(0.40, 0.72), "cache_checksum_errors": rng.randint(80, 350), "cache_hit_rate": rng.uniform(0.08, 0.22)}
            else:
                m = _build_hard_metrics(tid, rng, tick)
            slo_burn = rng.choice([3.0, 4.5, 6.0, 8.0, 10.0])
            fault_svc_sub = _apply_subst(task["fault_service"], sub_map) if sub_svcs else task["fault_service"]
            examples.append({
                "example_id": "", "source_script": "", "task_seed_id": tid, "tier": "hard",
                "fault_type": task["fault_type"],
                "variation_strategy": "suboptimal_path,service_pool_substitution,metric_value,adversarial_content",
                "observation": {
                    "tick": tick, "budget": _calculate_budget("hard", tick),
                    "alerts": [f"[WARNING] {fault_svc_sub} SLO burn {slo_burn:.1f}×/hr"],
                    "service_metrics": {fault_svc_sub: m},
                    "logs": {fault_svc_sub: [f"fault: {tid}"]},
                },
                "gold_action_sequence": [(_apply_subst(a, sub_map) if '(' in a else a) for a in GOLD.get(tid, ["declare_resolved"])],
                "gold_alternatives": [], "expected_score_range": [0.30, 0.80],
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
    exs = generate(task_list, rng_seed=29000)
    print(f"Generated {len(exs)} examples")