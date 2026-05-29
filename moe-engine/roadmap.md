# `moe-engine` — Engineering Roadmap & State Ledger

> **Protocol-1 ledger.** Every turn must read this file before mutating code,
> and append a markdown blockquote to the response summarising any updates
> made here. The schema below is **load-bearing**: do not rename sections.

Last update: turn-001 (current turn)

---

## Completed Tasks

Each entry: `<scope>` — `<file path(s)>` — `<verification reference>`

### Carried over from prior session (verified before this ledger existed)
| # | Scope | Files | Verification |
|---|-------|-------|--------------|
| C-01 | Custom Triton Top-K MoE routing kernel (forward only) with CPU autograd fallback | `pkg/kernels/moe_router.py` | `pytest tests/test_kernels.py` (carried over from session 0; backward kernel is **NOT** yet present — see TD-04) |
| C-02 | 2D `(dp, ep)` `DeviceMesh` + FSDP2 `fully_shard` + EP `all_to_all_single` on a dedicated CUDA stream | `pkg/distributed/parallel_mesh.py` | `pytest tests/test_distributed.py` |
| C-03 | Two-tier (Local NVMe → S3/MinIO) async checkpointer scaffold with `boto3` adapter, retention pruning, signal-driven flush | `pkg/elastic/fault_monitor.py` | `pytest tests/test_elastic.py` |
| C-04 | TorchElastic chaos test driver: baseline + Scenario B (storage stall) **passing** | `tests/test_chaos.py`, `tests/_chaos_worker.py` | `pytest -m chaos -k "baseline or scenario_b"` |
| C-05 | Telemetry, MFU accountant, structured JSON logger, training entrypoint | `pkg/telemetry/`, `pkg/utils/`, `train.py` | `python train.py --config configs/smoke.yaml --smoke` |

### This turn (turn-001)
| # | Scope | Files | Verification |
|---|-------|-------|--------------|
| T01-A | Initialise Protocol-1 state ledger | `roadmap.md` | file present at repo root |
| T01-B | Enterprise-grade README with ASCII architecture, HW reqs, local + cluster orchestration guide | `README.md` | file present, replaces prior 93-line stub |
| T01-C | **Item #1** — True 4D `(pp, dp, ep, tp)` topology via `init_device_mesh`; TP intra-node, PP inter-node; FSDP2 sharded along `dp`; EP collectives along `ep` | `pkg/distributed/parallel_mesh.py` | `pytest tests/test_distributed.py` (1-rank degenerate path preserved) |
| T01-D | **Item #3** — `dist.all_to_all_single(async_op=True)` + explicit `Work.wait()` overlap; 3-phase forward (launch → independent local compute → wait) in `DistributedMoELayer.forward` | `pkg/distributed/parallel_mesh.py` | shape & autograd test in `test_distributed.py` still green |
| T01-E | **Item #2** — Pinned-host-memory async checkpoint staging pipeline; intercept SHARDED_STATE_DICT on main stream → detach → clone → `pin_memory()` non-blocking D2H copy → enqueue **only** host-resident snapshot to writer thread | `pkg/elastic/fault_monitor.py` | `pytest tests/test_elastic.py`; new `test_async_ckpt_no_device_refs` regression |

---

## In-Progress / Active Focus

**Turn-001 (current):** Items #1, #2, #3 of the Technical Remediation
Framework + Protocol-2 README + Protocol-1 ledger initialisation. All three
remediations land in this turn. No partial deliveries.

---

## Outstanding Technical Debt & Backlog

Chronological — top of list = next turn's focus.

| # | Title | Rationale | Touch points |
|---|-------|-----------|--------------|
| TD-01 | **Item #4 — Triton kernel safety + autograd backward** | Forward kernel currently assumes static `BLOCK_M`; produces unaligned-coalesced loads on variable seq-len batches. Backward path runs in eager PyTorch → 4× slower than necessary. | `pkg/kernels/moe_router.py` |
| TD-02 | **Item #5 — Non-uniform expert resharding state-machine** | After a node drop, `experts_on_this_rank` re-divides modulo survivors; the *weights* of orphaned experts are not migrated atomically — load-state-dict shape mismatch when EP shrinks from 8→3. | `pkg/elastic/fault_monitor.py` (`reshard`, `load`), `pkg/distributed/parallel_mesh.py` |
| TD-03 | **Item #6 — NCCL communicator-poisoning watchdogs** | A hard `SIGKILL` on a NCCL peer leaves survivors stuck in un-abortable `ncclRecv`. Need `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `TORCH_NCCL_DESYNC_DEBUG=1`, `NCCL_BLOCKING_WAIT=0` injected at harness init. | `pkg/elastic/fault_monitor.py`, `scripts/launch.sh` |
| TD-04 | **Item #7 — Dynamic MoE FLOP / MFU equation** | Current MFU uses static FLOPs/token. Replace with `FLOPs_step = 2·T_active·P_dense + 2·T_routed·P_experts` recomputed per-step from the actual router profile. | `pkg/utils/mfu.py`, `train.py` |
| TD-05 | Resolve `test_chaos_scenario_a_sudden_node_failure` (carried over) | Gloo rendezvous on `world_size=4` flakes after `SIGKILL`; survivor restart-generation telemetry shows `AssertionError` during state reload — likely intersection of TD-02 (uneven reshard) and TD-03 (stuck communicator). May auto-resolve once TD-02 + TD-03 land. | `tests/test_chaos.py`, `tests/_chaos_worker.py`, `pkg/elastic/fault_monitor.py` |
| TD-06 | Regression: `pytest -m "not chaos"` after every TD-0x lands | Catch collateral damage from large refactors. | n/a |
| TD-07 | `--runchaos` opt-in CLI flag in `conftest.py` | Make `-m chaos` opt-in by default in CI; nightly matrix flips it on. | `tests/conftest.py` |
| TD-08 | Wire chaos suite into a nightly GH Actions matrix | Continuous fault-injection coverage. | `.github/workflows/nightly.yml` |

---

## Mathematical Invariants (CI gates)

These must hold and are codified in tests:

1. **4-D Mesh shape:**  `|TP| · |PP| · |FSDP2| · |EP| = world_size`
2. **Token conservation:**  `Σ_r dispatched_r = B · S · K`
3. **Checksum identity:**  `hash(state_dict_post_load) = hash(state_dict_pre_save)`
4. **Monotonic progression:**  `step_{n+1} > step_n` across every restart generation
5. **Dynamic MoE FLOP:** `FLOPs_step = 2·T_active·P_dense + 2·T_routed·P_experts`
