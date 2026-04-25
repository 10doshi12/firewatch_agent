# firewatch_agent/data_gen_scripts

This directory contains the 30 generator scripts (`gen_01_*.py` through `gen_30_*.py`).
Each script produces exactly 50 training examples when run.

**Entry point for all scripts:** `generate(tasks: list[dict], rng_seed: int) -> list[dict]`

**Convention:** One script = one batch = 50 examples = 1 reviewed JSONL on HuggingFace.

## Adding a new generator

1. Author the spec in `docs/firewatch_agent_docs/gen_specs/GEN-SPEC-XX.md`
2. Implement `gen_XX_<name>.py` following the CONTEXT-BOOTSTRAP.md rules
3. Ensure `generate()` returns exactly 50 examples, shuffled with `rng.shuffle()`
4. Test by running `python -m firewatch_agent.data_gen.run_generator --script XX`
5. Review via `python -m firewatch_agent.data_gen.review --batch <N>`
6. Upload via `python -m firewatch_agent.data_gen.upload --batch <N>`