"""
prompt.py — SFT prompt formatter (SPEC-T2 §10.3)

Combines observation + GNN blurb + gold action sequence into
system/user/assistant messages for completion-only SFT.

Loss computed only on assistant tokens.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# System message (fixed constant)
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = """You are an expert Site Reliability Engineer (SRE) managing a microservice-based production system. Your role is to diagnose incidents and execute precise remediation actions.

Available investigation actions:
- fetch_logs(service): Retrieve recent logs from a service
- get_metrics_detail(service): Get detailed metrics for a service
- trace_dependencies(service): Trace dependency graph from a service
- strace_process(service): Low-level syscall tracing
- profiler_dump(service): CPU/memory profiler output
- check_gc_pressure(service): GC pressure analysis
- trace_distributed_request(service): Distributed trace analysis
- inspect_thread_pool(service): Thread pool state inspection
- inspect_commit_diff(service): Recent deployment diff
- inspect_network_policy(service): Network policy inspection
- inspect_quota_usage(service): Resource quota status
- inspect_consensus_state(service): Consensus/cluster state
- inspect_cluster_topology(service): Cluster topology view
- thread_dump(service): Full thread dump
- inspect_mtls_status(service): mTLS certificate status
- inspect_pipeline_topology(service): Data pipeline topology

Available remediation actions:
- restart_service(service): Restart a service instance
- rollback_deploy(service): Rollback to previous deployment
- revert_config(service): Revert configuration to previous revision
- scale_replicas(service, params): Scale service replicas
- circuit_break(service): Enable circuit breaker on a service
- traffic_shift(service, params): Shift traffic between service instances
- enable_connection_throttle(service): Enable connection throttling
- extend_timeout(service): Extend request timeout
- optimize_query(service): Optimize database queries
- rebalance_load(service): Rebalance load across instances
- rotate_tls_certificate(service): Rotate TLS certificates
- restart_pipeline_job(service): Restart a pipeline job
- flush_pipeline_stage(service): Flush a pipeline processing stage
- And additional specialized remediation actions

Meta actions:
- declare_resolved: Declare the incident resolved (terminal)
- escalate: Escalate to a specialist team

Service status thresholds:
- healthy: error_rate < 0.10 AND latency_p99 < 0.50s
- degraded: error_rate >= 0.10 OR latency_p99 >= 0.50s
- critical: error_rate >= 0.50 OR latency_p99 >= 2.0s
- down: error_rate >= 0.90 OR memory_utilization >= 0.98

Respond with a sequence of actions to diagnose and resolve the incident. Each action should be a JSON object with 'action' and 'params' keys."""


def _serialize_observation(observation: dict) -> str:
    """Serialize an observation dict into a human-readable string for the prompt."""
    parts: list[str] = []

    # Alert summary
    alerts = observation.get("alerts", [])
    if alerts:
        parts.append("=== ACTIVE ALERTS ===")
        for alert in alerts:
            if isinstance(alert, dict):
                parts.append(
                    f"  [{alert.get('severity', 'unknown')}] "
                    f"{alert.get('alertname', 'Unknown')}: "
                    f"{alert.get('description', 'No description')}"
                )
            else:
                parts.append(f"  {alert}")

    # Service metrics
    service_metrics = observation.get("service_metrics", {})
    if service_metrics:
        parts.append("\n=== SERVICE METRICS ===")
        for svc_name, metrics in sorted(service_metrics.items()):
            if isinstance(metrics, dict):
                status = metrics.get("status", "unknown")
                error_rate = metrics.get("http_server_error_rate", 0.0)
                latency = metrics.get("http_server_request_duration_p99", 0.0)
                mem_util = metrics.get("process_memory_utilization", 0.0)
                cpu_util = metrics.get("process_cpu_utilization", 0.0)
                parts.append(
                    f"  {svc_name}: status={status} | "
                    f"error_rate={error_rate:.3f} | "
                    f"latency_p99={latency:.3f}s | "
                    f"mem_util={mem_util:.2f} | "
                    f"cpu_util={cpu_util:.2f}"
                )
            else:
                parts.append(f"  {svc_name}: {metrics}")

    # Logs (if populated)
    logs = observation.get("logs", {})
    if logs:
        parts.append("\n=== RECENT LOGS ===")
        for svc_name, log_lines in sorted(logs.items()):
            if log_lines:
                parts.append(f"  [{svc_name}]:")
                if isinstance(log_lines, list):
                    for line in log_lines[:10]:
                        parts.append(f"    {line}")
                else:
                    parts.append(f"    {log_lines}")

    # Budget and time
    budget = observation.get("budget", None)
    tick = observation.get("tick", None)
    if budget is not None:
        parts.append(f"\nSLO Budget Remaining: {budget:.1f}%")
    if tick is not None:
        parts.append(f"Current Tick: {tick}")

    return "\n".join(parts)


def _serialize_gold_actions(gold_actions: list[dict]) -> str:
    """Serialize gold action sequence as structured JSON for the assistant message."""
    action_strs: list[str] = []
    for i, action in enumerate(gold_actions, 1):
        action_str = json.dumps(action, separators=(",", ":"))
        action_strs.append(f"Step {i}: {action_str}")
    return "\n".join(action_strs)


def format_sft_prompt(
    example: dict,
    gnn_blurb: str | None = None,
) -> dict[str, str]:
    """
    Format a single training example into system/user/assistant messages.

    Args:
        example: One training example with observation and gold_action_sequence
        gnn_blurb: Optional GNN-generated text blurb to append to user message

    Returns:
        Dict with 'system', 'user', 'assistant' keys.
    """
    observation = example.get("observation", {})
    gold_actions = example.get("gold_action_sequence", [])

    # User message: observation serialization + GNN blurb
    user_parts: list[str] = []
    user_parts.append("An incident has been detected in the production system. Here is the current state:")
    user_parts.append("")
    user_parts.append(_serialize_observation(observation))

    if gnn_blurb:
        user_parts.append("")
        user_parts.append(gnn_blurb)

    user_parts.append("")
    user_parts.append("Analyze the situation and provide the optimal sequence of actions to diagnose and resolve this incident.")

    # Assistant message: gold action sequence
    assistant_msg = _serialize_gold_actions(gold_actions)

    return {
        "system": SYSTEM_MESSAGE,
        "user": "\n".join(user_parts),
        "assistant": assistant_msg,
    }


def format_chat_messages(prompt: dict[str, str]) -> list[dict[str, str]]:
    """Convert system/user/assistant dict to chat message format for tokenizer."""
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
        {"role": "assistant", "content": prompt["assistant"]},
    ]
