# AGENTS — Guidance for AI coding agents

Purpose
 - Provide concise, actionable guidance so agents can be immediately productive in this repository.

Quick actions
 - Install deps: `pip install -r moe-engine/requirements.txt` ([requirements.txt](moe-engine/requirements.txt))
 - Run unit + integration tests: `pytest -m "not chaos" -v` (see [pyproject.toml](moe-engine/pyproject.toml) pytest options)
 - Smoke-run training (single-rank CPU): `python moe-engine/train.py --config moe-engine/configs/smoke.yaml --smoke` ([train.py](moe-engine/train.py))
 - Launch multi-node: `bash moe-engine/scripts/launch.sh` (wraps `torchrun`) ([scripts/launch.sh](moe-engine/scripts/launch.sh))

Entrypoints & important files
 - `moe-engine/train.py` — primary entrypoint; parses configs and builds topology.
 - `moe-engine/pyproject.toml` — project metadata and test config. Link: [pyproject.toml](moe-engine/pyproject.toml)
 - `moe-engine/configs/default.yaml` — source of truth for runtime config; use `configs/smoke.yaml` for small local runs.
 - `moe-engine/pkg/` — implementation: `distributed/`, `elastic/`, `kernels/`, `telemetry/`, `utils/`.
 - Tests: `tests/` (unit/integration) and `moe-engine/tests/` (repo contains both); test markers include `chaos` (opt-in).

Conventions and agent guidance
 - Link, don't duplicate: prefer linking to the README and docs instead of copying large sections.
 - Use smoke/dev workflows first: run `configs/smoke.yaml` and `pytest -m "not chaos"` to reproduce issues cheaply.
 - Respect `configs/default.yaml` as the canonical runtime configuration; change only smoke/dev configs for local testing.
 - When proposing code changes, run relevant tests (`pytest tests/test_*.py -q`) and keep edits minimal and well-scoped.
 - Chaos tests are expensive/special: run only when requested (`-m chaos`) and use `GLOO_SOCKET_IFNAME=lo` for local simulation.

Potential pitfalls
 - Multi-GPU logic depends on correct 4-D mesh product. Prefer running `tests/test_distributed.py::test_topology_4d_product` when modifying topology code.
 - Triton kernels require GPU and compatible Triton/CUDA; use CPU fallbacks for unit tests.

If you need more targeted instructions, suggest creating smaller customization files (e.g., `AGENTS-tests.md`, `AGENTS-elastic.md`) for complex subsystems.
