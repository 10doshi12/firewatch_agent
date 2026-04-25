"""
gen_24_mixed_log_content.py — Mixed Batch: Log Content Variation

Script: gen_24_mixed_log_content.py
Batch: 023 (script_num = 24, batch = 023)
Primary axes: log_content + metric_value
Batch composition: 20 Easy + 18 Medium + 12 Hard = 50 examples
Shuffle: rng.shuffle(examples) before returning

SPEC: GEN-SPEC-24
Bootstrap: CONTEXT-BOOTSTRAP.md
Varies log surface form: exception names, hostnames, thread IDs, timestamps.
Diagnostic token must always be present.
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


# Variation pools for log content
HOSTNAMES = ["svc-a1", "svc-b2", "svc-c3", "pod-7f8d", "pod-2a9c"]
TIMESTAMPS = ["2026-04-22T14:32:18.432Z", "1745334738432", "t+32s"]
THREAD_IDS = ["[thread-14]", "[worker-3]", "[http-nio-8080-exec-7]", "[grpc-default-executor-2]"]
OOM_EXCEPTIONS = ["java.lang.OutOfMemoryError", "com.sun.jvm.HeapOverflowError", "io.netty.util.internal.OutOfDirectMemoryError"]
CONN_TIMEOUTS = ["java.net.SocketTimeoutException", "io.grpc.StatusRuntimeException: DEADLINE_EXCEEDED", "org.springframework.web.client.ResourceAccessException"]
POOL_EXHAUSTIONS = ["com.zaxxer.hikari.pool.HikariPool$PoolInitializationException", "HikariPool-1 timeout", "Connection pool exhausted"]


GOLD = {
    "task_easy_oom_baseline": ["fetch_logs(auth-service)", "scale_replicas(auth-service)", "declare_resolved"],
    "task_easy_thread_deadlock": ["thread_dump(order-service)", "restart_thread_pool(order-service)", "declare_resolved"],
    "task_easy_dns_nxdomain": ["fetch_logs(payment-service)", "update_service_endpoint(payment-service)", "declare_resolved"],
    "task_easy_http2_streams": ["get_metrics_detail(api-gateway)", "increase_max_streams(api-gateway)", "declare_resolved"],
    "task_easy_cert_expiry": ["fetch_logs(payment-service)", "rotate_tls_certificate(payment-service)", "declare_resolved"],
    "task_medium_cache_eviction_storm": ["trace_dependencies(user-db)", "get_metrics_detail(cache-service)", "fetch_logs(cache-service)", "increase_cache_memory(cache-service)", "declare_resolved"],
    "task_medium_corrupted_external_dep": ["fetch_logs(user-service)", "rollback_deploy(user-service)", "declare_resolved"],
    "task_medium_db_connection_herd": ["fetch_logs(db-proxy)", "stagger_connection_pool_reconnect(db-proxy)", "declare_resolved"],
    "task_medium_grpc_deadline": ["get_metrics_detail(payment-service)", "trace_dependencies(order-service)", "enable_deadline_propagation(order-service)", "declare_resolved"],
    "task_hard_config_drift_noise": ["get_metrics_detail(api-gateway)", "fetch_logs(api-gateway)", "revert_config(api-gateway)", "declare_resolved"],
    "task_hard_pipeline_freshness": ["inspect_pipeline_topology(feature-pipeline)", "get_metrics_detail(feature-pipeline)", "restart_pipeline_job(feature-pipeline)", "declare_resolved"],
    "task_hard_mesh_proxy_upgrade": ["inspect_mtls_status(payment-service)", "rollback_proxy_upgrade(payment-service)", "declare_resolved"],
}


def _vary_log(line: str, rng: random.Random) -> str:
    """Apply random surface-form variations to a log line."""
    result = line
    if "{hostname}" in result:
        result = result.replace("{hostname}", rng.choice(HOSTNAMES))
    if "{timestamp}" in result:
        result = result.replace("{timestamp}", rng.choice(TIMESTAMPS))
    if "{thread}" in result:
        result = result.replace("{thread}", rng.choice(THREAD_IDS))
    return result


def _build_easy_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_easy_oom_baseline":
        exc = rng.choice(OOM_EXCEPTIONS)
        hostname = rng.choice(HOSTNAMES)
        timestamp = rng.choice(TIMESTAMPS)
        thread = rng.choice(THREAD_IDS)
        return {
            "status": "critical",
            "http_server_error_rate": rng.choice([0.12, 0.15, 0.18, 0.22, 0.27, 0.31]),
            "process_memory_utilization": rng.choice([0.93, 0.95, 0.96, 0.97, 0.98, 0.99]),
            "restart_count": rng.randint(2, 8),
            "logs": {"auth-service": [
                f"Exit code 137 — OOMKilled",
                f"{timestamp} {hostname} {thread} {exc}: Heap space capacity exceeded",
            ]},
        }
    if tid == "task_easy_thread_deadlock":
        blocked = rng.choice([42, 50, 55, 58, 60])
        thread = rng.choice(THREAD_IDS)
        return {
            "status": "critical",
            "http_server_error_rate": rng.choice([0.88, 0.92, 0.97, 0.99, 1.00]),
            "runtime_blocked_thread_count": blocked,
            "wait_ratio": round(rng.uniform(0.88, 1.00), 2),
            "logs": {"order-service": [
                f"{thread} deadlock detected: {blocked} threads blocked on owned mutex",
                f"Thread state: BLOCKED — waiting on io.netty.util.concurrent.DefaultPromise",
            ]},
        }
    if tid == "task_easy_dns_nxdomain":
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.0, 0.08),
            "dns_lookup_failure_rate": rng.uniform(0.55, 0.88),
            "logs": {"payment-service": ["DNS lookup failed: NXDOMAIN for payment-service.prod.svc.cluster.local"]},
        }
    if tid == "task_easy_http2_streams":
        max_streams = rng.choice([80, 100, 128, 150])
        thread = rng.choice(THREAD_IDS)
        return {
            "status": "degraded",
            "http_server_error_rate": round(rng.uniform(0.0, 0.08), 4),
            "http2_max_concurrent_streams": max_streams,
            "http2_stream_utilization": round(rng.uniform(0.95, 1.00), 2),
            "logs": {"api-gateway": [f"stream limit reached: {max_streams} active", f"{thread} HTTP/2 stream exhaustion"]},
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
    if tid == "task_medium_cache_eviction_storm":
        cache_mem = rng.choice([0.95, 0.97, 0.98, 0.99, 1.00])
        hostname = rng.choice(HOSTNAMES)
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.18, 0.42),
            "cache_memory_utilization": cache_mem,
            "cache_evictions_per_second": rng.choice([180, 280, 380, 450, 550, 680]),
            "logs": {hostname: [f"Cache maxmemory reached — eviction storm"]},
        }
    if tid == "task_medium_corrupted_external_dep":
        hostname = rng.choice(HOSTNAMES)
        exc = rng.choice(CONN_TIMEOUTS)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.55, 0.88),
            "dependency_health_score": rng.uniform(0.25, 0.45),
            "logs": {hostname: [f"Dependency checksum mismatch: {exc}"]},
        }
    if tid == "task_medium_db_connection_herd":
        active = rng.choice([210, 228, 238, 248, 255, 270])
        hostname = rng.choice(HOSTNAMES)
        exc = rng.choice(POOL_EXHAUSTIONS)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.35, 0.68),
            "db_active_connections": active,
            "db_max_connections": 200,
            "logs": {hostname: [f"Too many connections — {active}/200", f"{exc}"]},
        }
    if tid == "task_medium_grpc_deadline":
        orphaned = rng.choice([0.55, 0.65, 0.72, 0.80, 0.88])
        thread = rng.choice(THREAD_IDS)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.50, 0.88),
            "grpc_orphaned_call_rate": orphaned,
            "logs": {"payment-service": [f"gRPC orphaned calls: deadline exceeded", f"{thread} grpc.orphan"]},
        }
    return {}


def _build_hard_metrics(tid: str, rng: random.Random, tick: int) -> dict[str, Any]:
    if tid == "task_hard_config_drift_noise":
        fds = rng.randint(2, 6)
        hostname = rng.choice(HOSTNAMES)
        return {
            "status": "critical",
            "http_server_error_rate": rng.uniform(0.52, 0.78),
            "process_open_file_descriptors": fds,
            "logs": {hostname: [f"HikariPool: pool_size={fds} (recommended: 20)"]},
        }
    if tid == "task_hard_pipeline_freshness":
        throughput = rng.choice([0.72, 0.78, 0.82, 0.86, 0.90])
        hostname = rng.choice(HOSTNAMES)
        return {
            "status": "degraded",
            "http_server_error_rate": 0.0,
            "freshness_lag_seconds": rng.randint(380, 650),
            "pipeline_throughput_ratio": throughput,
            "logs": {hostname: [f"throughput_ratio={throughput:.2f} — queue backlog growing"]},
        }
    if tid == "task_hard_mesh_proxy_upgrade":
        return {
            "status": "degraded",
            "http_server_error_rate": rng.uniform(0.20, 0.45),
            "proxy_upgrade_completion_pct": rng.choice([0.55, 0.60, 0.65, 0.70, 0.75]),
            "mtls_cipher_compatibility": False,
            "logs": {"payment-service": ["TLSV1_ALERT_PROTOCOL_VERSION — proxy handshake failure"]},
        }
    return {}


def generate(tasks: list[dict], rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    task_map = {t["task_id"]: t for t in tasks}

    easy_ids = [
        "task_easy_oom_baseline", "task_easy_thread_deadlock", "task_easy_dns_nxdomain",
        "task_easy_http2_streams", "task_easy_cert_expiry",
    ]
    medium_ids = [
        "task_medium_cache_eviction_storm", "task_medium_corrupted_external_dep",
        "task_medium_db_connection_herd", "task_medium_grpc_deadline",
    ]
    hard_ids = [
        "task_hard_config_drift_noise", "task_hard_pipeline_freshness",
        "task_hard_mesh_proxy_upgrade",
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
                "variation_strategy": "log_content,metric_value",
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
                "variation_strategy": "log_content,metric_value",
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
                "variation_strategy": "log_content,metric_value,adversarial_content",
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
    exs = generate(task_list, rng_seed=24000)
    print(f"Generated {len(exs)} examples")