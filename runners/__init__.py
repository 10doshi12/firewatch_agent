"""
runners — Local production baseline runners for FirewatchEnv.

These runners are HTTP-based and API-LLM-based, distinct from
`eval/baseline.py` which loads LLM weights locally and connects to the
remote HF Space via WebSocket.

Use `runners` when:
  - You want to evaluate against a *local* sim (uv run server / docker).
  - You want to use a hosted LLM (OpenRouter, OpenAI, Together, Anyscale)
    or a small local LLM via Ollama/vLLM (OpenAI-compatible endpoint).
  - You want full per-step trajectory capture (prompt, response, action,
    reward, score) for offline analysis / SFT data prep.
  - You want to compare an *honest* (no leakage) baseline vs an
    *informed* (rewards in history) baseline as an ablation.

Components:
  http_sim_client.py — synchronous HTTP client to FirewatchEnv server
  llm_client.py      — pluggable OpenAI-compatible LLM client (openai/ollama)
  honest_prompt.py   — production prompt with no fault-cheat-sheet leakage
  gnn_baseline.py    — wraps gnn/ for production inference (untrained or ckpt)
  policy.py          — composes GNN + LLM into a step policy
  trajectory.py      — JSONL trajectory logger (per-step + per-episode)
  inference.py       — CLI entry point: [START]/[STEP]/[END] + trajectory JSONL
"""
