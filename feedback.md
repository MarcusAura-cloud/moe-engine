Here is a precise, strict prompt you can hand directly to the coding agent:

---

# ENGINEERING DIRECTIVE: moe-engine Remediation & Roadmap

**From:** Principal Review  
**Priority:** P0 (blocking) → P1 (core) → P2 (hardening)  
**Context:** A deep audit of the current `moe-engine` repo has found that the codebase does not meet the standards claimed in its README. The following is a prioritized, unambiguous list of what must be built, fixed, or removed. There are no suggestions here — these are requirements.

---

## IMMEDIATE ACTIONS (Do First, Before Any New Code)

### CLEANUP — Remove Agent Scaffolding Artifacts

Delete the following files from the repo root. They are internal agent artifacts that must never appear in a public engineering repository:

- `pid.txt`
- `verify_pid.txt`
- `test_result.md`
- `SIGNAL_WEEK1_P0_COMPLETION.md`
- `IMPLEMENTATION_CHECKLIST.md`
- `memory/` (entire directory)
- `.emergent/` (entire directory)

Add the following lines to `.gitignore`:
```
pid.txt
verify_pid.txt
*.pid
memory/
.emergent/
```

Commit this cleanup as a single standalone commit with message: `chore: remove agent scaffolding artifacts from public repo`

---

## P0 — CRITICAL CORRECTNESS (The repo is broken without these)

### P0-1: Implement the Triton Backward Kernel

**File:** `moe-engine/pkg/kernels/moe_router.py`

This is the single most important missing piece. The current code falls back to PyTorch autograd for the backward pass. The prompt required an explicit `@triton.jit` backward kernel. You must implement it.

**What to build:**

Write a `@triton.jit` kernel named `_router_bwd_kernel` that accepts:
- `grad_output`: incoming gradient from downstream, shape `[N, K]` (one gradient per token-expert assignment)
- `softmax_weights`: the saved softmax scores from the forward pass, shape `[N, E]`
- `expert_indices`: the saved Top-K indices from the forward pass, shape `[N, K]`
- `grad_gate_logits`: output buffer, shape `[N, E]`

The backward must compute the gradient through the softmax + Top-K selection. The mathematically correct approach:
1. Reconstruct the sparse softmax Jacobian: for each token, the gradient w.r.t. the pre-softmax logit `z_i` is `grad_w_i = w_i * (grad_output_i - sum_j(grad_output_j * w_j))` for the K selected experts, zero elsewhere.
2. Write the result into `grad_gate_logits` using coalesced stores, same memory layout and tiling strategy as the forward kernel.

The kernel must match the forward kernel's block size and memory access pattern for SRAM reuse consistency.

Wire it into a `torch.autograd.Function` subclass named `MoERouterFunction` with a `forward()` that calls `_router_fwd_kernel` and saves the required tensors for backward, and a `backward()` that calls `_router_bwd_kernel`.

**Acceptance test** (must pass in CI):
```python
# In tests/test_kernels_numerics.py
# Cast to float64, run reference softmax+topk in PyTorch
# Run MoERouterFunction on float32, cast grads to float64
# Assert: torch.allclose(grad_triton.double(), grad_ref, atol=1e-5, rtol=1e-5)
# Run across 50 random seeds, varying H in [64,128,256,512], E in [8,16,32], K in [1,2,4]
```

---

### P0-2: Remove the `if False` Dead Code in `parallel_mesh.py`

**File:** `moe-engine/pkg/distributed/parallel_mesh.py`

Search for any remaining `if False else` conditional or any commented-out `all_to_all_single` call path. This was partially addressed in Week 1 but must be fully verified. The token dispatch path must execute unconditionally. Run the following grep and fix every hit:

```bash
grep -n "if False" moe-engine/pkg/distributed/parallel_mesh.py
grep -n "placeholder" moe-engine/pkg/distributed/parallel_mesh.py
grep -n "# TODO" moe-engine/pkg/distributed/parallel_mesh.py
```

Every one of these must be resolved to real, executing code before proceeding.

---

### P0-3: Fix the Chaos Scenario A Test (Node Kill + Recovery)

**File:** `moe-engine/tests/test_chaos.py` and `moe-engine/pkg/elastic/fault_monitor.py`

Scenario A (sudden node failure mid-training) is currently quarantined. This is the headline feature of the elastic module. It must work. The two root causes identified are:

**Root cause 1 — Uneven expert reshard (TD-05):**  
When a rank is killed, the surviving ranks must redistribute experts evenly. Implement `_rebalance_experts(active_ranks: list[int], mesh: DeviceMesh) -> dict[int, list[int]]` in `fault_monitor.py`. This function must:
- Take the list of surviving rank IDs
- Compute the new even distribution of expert IDs across those ranks (round-robin assignment is acceptable)
- Return a mapping `{rank_id: [expert_ids]}` for the new topology
- Be called inside `ClusterStateMachine._on_rank_failure()` before triggering the checkpoint reload

**Root cause 2 — NCCL watchdog / timeout handling (TD-06):**  
When a rank dies mid-collective, NCCL will hang other ranks waiting for it unless `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` and `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` are set. Add the following to `fault_monitor.py`'s `ElasticTrainerHarness.__init__()`:

```python
import os
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "30")
os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "1048576")
```

**Acceptance test:** `pytest -m chaos -v -k "scenario_a"` must pass. The test must:
1. Spawn 4 Gloo workers
2. SIGKILL one worker at step 3
3. Verify surviving workers re-rendezvous via TorchElastic
4. Verify `step_after_recovery > step_at_kill` (monotonic progress)
5. Verify the model's output on a fixed input is numerically identical before and after recovery (deterministic checkpoint reload)

---

## P1 — CORE MISSING FEATURES (The repo claims these exist but they don't)

### P1-1: Implement Tensor Parallelism

**File:** `moe-engine/pkg/distributed/parallel_mesh.py`

The TP mesh axis exists but nothing runs on it. Implement column-parallel and row-parallel linear layers.

**What to build:**

```python
class ColumnParallelLinear(nn.Module):
    """
    Splits the output dimension across the TP group.
    Weight shape: [out_features // tp_size, in_features] per rank.
    Requires all-gather on output after compute.
    Uses DTensor with Shard(0) placement on the TP mesh dimension.
    """

class RowParallelLinear(nn.Module):
    """
    Splits the input dimension across the TP group.
    Weight shape: [out_features, in_features // tp_size] per rank.
    Requires reduce-scatter on output.
    Uses DTensor with Shard(1) placement on the TP mesh dimension.
    """
```

Both classes must use `torch.distributed.tensor.DTensor` with explicit placements on `mesh["tp"]`. The `ColumnParallelLinear` forward must issue a non-blocking all-gather on the `tp` process group. Wire these into `DistributedMoELayer` so the expert FFN layers use column-parallel for the up-projection and row-parallel for the down-projection.

---

### P1-2: Implement Pipeline Parallelism Schedule

**File:** `moe-engine/pkg/distributed/parallel_mesh.py` (new class `PipelineStage`)

The PP mesh axis is a stub. Implement a 1F1B (one-forward-one-backward) pipeline schedule.

**What to build:**

```python
class PipelineStage:
    def __init__(self, stage_id: int, num_stages: int, mesh: DeviceMesh): ...
    
    def forward_step(self, micro_batch: Tensor) -> Tensor:
        # If not first stage: recv activation from stage_id-1 via dist.recv
        # Run local layers
        # If not last stage: dist.send activation to stage_id+1
        ...
    
    def backward_step(self, grad: Tensor) -> Tensor:
        # If not last stage: recv grad from stage_id+1 via dist.recv
        # Run local backward
        # If not first stage: dist.send grad to stage_id-1
        ...
    
    def run_1f1b(self, micro_batches: list[Tensor]) -> list[Tensor]:
        # Implement the 1F1B interleave schedule
        # Warmup phase: fill the pipeline
        # Steady state: 1 forward + 1 backward per clock
        # Drain phase: flush remaining backwards
        ...
```

The schedule must minimize pipeline bubble fraction to `(p-1)/(m+p-1)` where `p` is pipeline stages and `m` is micro-batch count. Document this formula in an inline comment.

---

### P1-3: Implement Real MFU Calculation for MoE

**File:** `moe-engine/pkg/utils/mfu.py`

The current MFU calculation is flagged as `TD-04` because it does not account for MoE sparsity correctly. The correct formula for a sparse MoE model is:

```
FLOPs_per_step = 2 * T_total * P_dense_layers        # dense attention blocks
               + 2 * T_total * (K / E) * P_expert_layers  # sparse expert blocks (only K/E fraction active)

MFU = FLOPs_per_step / (world_size * hardware_peak_flops * step_time_seconds)
```

Where:
- `T_total` = total tokens in the batch = `batch_size * seq_len`
- `P_dense_layers` = parameter count of all non-expert layers (embeddings, attention, norms)
- `P_expert_layers` = parameter count of all expert layers
- `K` = top-k value
- `E` = total number of experts
- `hardware_peak_flops` = loaded from config (e.g., 989e12 for H100 SXM5 BF16)

Implement `compute_mfu(model_config, step_metrics) -> float` with full type annotations. Assert `0.0 < mfu < 1.0` and log a warning if `mfu < 0.30` (likely misconfiguration) or `mfu > 0.85` (likely measurement error).

---

### P1-4: Wire Telemetry to Real Measurements

**File:** `moe-engine/pkg/telemetry/logger.py`

The telemetry JSON schema is correct but all the `kernel.*` fields (SRAM occupancy, achieved bandwidth) are currently estimated/fabricated. Wire them to real sources:

- `kernel.sram_bytes_per_block`: compute from Triton kernel's `tl.constexpr` block sizes — `BLOCK_N * BLOCK_K * element_size_bytes`. This is deterministic and must be computed once at kernel compilation and cached.
- `kernel.achieved_bw_gbps`: measure via `torch.cuda.Event` timing around the kernel launch: `bytes_moved = 2 * N * E * element_size` (load logits + store results); `achieved_bw = bytes_moved / kernel_time_seconds / 1e9`. This requires wrapping the Triton kernel call in CUDA event timing.
- `collective.all_to_all_dispatch_ms` and `collective.all_to_all_combine_ms`: wrap each `dist.all_to_all_single` call with `torch.cuda.Event(enable_timing=True)` start/end events.
- `memory.peak_allocated_gb` and `memory.reserved_gb`: call `torch.cuda.memory_stats()` at the end of each step and read `"allocated_bytes.all.peak"` and `"reserved_bytes.all.peak"`.

No fabricated numbers. Every field in the telemetry JSON must be measured.

---

## P2 — HARDENING (Required before any "production" claim)

### P2-1: NVMe Streaming Checkpoint I/O

**File:** `moe-engine/pkg/elastic/fault_monitor.py`

The current `AsyncCheckpointer` stages to pinned host memory and then writes to disk in a background thread. This is correct in design but the background writer must stream in chunks rather than flushing the entire tensor at once, to avoid OOM when shards are large (e.g., 80GB expert shards on H100).

Implement chunked streaming in `_PinnedHostStager.flush_to_nvme()`:
```python
CHUNK_BYTES = 256 * 1024 * 1024  # 256 MB chunks
# Slice the pinned tensor into CHUNK_BYTES pieces
# Write each chunk to the NVMe file handle with O_DIRECT if available
# Update a progress counter for telemetry
```

### P2-2: Etcd Rendezvous for Scale > 100 Nodes

**File:** `moe-engine/pkg/elastic/fault_monitor.py`

The current `ClusterStateMachine.heartbeat()` uses a `dist.barrier()` as a liveness proxy. This does not work at scale: a barrier times out rather than identifying which rank died. Replace with proper TorchElastic rendezvous backed by etcd:

```python
# In ElasticTrainerHarness.__init__():
# Read RDZV_BACKEND from env (default: "c10d", production: "etcd")
# If "etcd": configure torch.distributed.elastic.rendezvous.etcd_rendezvous
# The agent should already handle this if launched via torchrun --rdzv_backend=etcd
# What's missing: the harness must read and log the current epoch/run_id
# from the rendezvous store to detect generation changes (restarts)
```

### P2-3: Sequence Parallelism for TP > 1

**File:** `moe-engine/pkg/distributed/parallel_mesh.py`

When `tp_size > 1`, the activation tensor `[B, S, H]` must be sharded along the sequence dimension to avoid duplicating the full sequence on every TP rank. Without this, TP does not save activation memory, defeating its purpose for long-context training. Implement `scatter_to_sequence_parallel` and `gather_from_sequence_parallel` utilities that shard/gather the sequence dimension across `mesh["tp"]`.

---

## ROADMAP.md — Required Entries

Replace or create `moe-engine/roadmap.md` with the following structure. Do not mark anything as complete that is not verified by a passing CI test.

```markdown
# moe-engine Roadmap

## Legend
✅ Complete + CI-verified  ⚠️ Partial  ❌ Not started  🔒 Blocked

---

## v0.1 — Correctness Foundation (current target)
- [❌] Triton backward kernel (`_router_bwd_kernel`) — P0-1
- [⚠️] Token conservation assertions — added, CI gate in place
- [❌] Chaos Scenario A: node kill + hot-resume — P0-3
- [⚠️] Chaos Scenario B: storage stall — passing
- [❌] Dead code removal (`if False` branches) — P0-2

## v0.2 — Complete 4D Parallelism
- [❌] Tensor Parallelism: ColumnParallel + RowParallel linear — P1-1
- [❌] Sequence Parallelism for TP > 1 — P2-3
- [❌] Pipeline Parallelism: PipelineStage + 1F1B schedule — P1-2

## v0.3 — Verified Performance
- [❌] Real MFU calculation (MoE-correct formula) — P1-3
- [❌] Telemetry wired to real CUDA measurements — P1-4
- [❌] BENCHMARKS.md with actual run data (kernel latency, overlap ratio, MFU)
- [❌] Async overlap ratio benchmark (target ≥ 0.60)

## v0.4 — Production Hardening
- [❌] NVMe chunked streaming checkpoint I/O — P2-1
- [❌] Etcd rendezvous for > 100 nodes — P2-2
- [❌] Nsight/CUPTI profiling integration
- [❌] Kubernetes / Kubeflow operator manifests

## Known Deficiencies (honest disclosure)
- No Triton backward kernel yet: backward uses PyTorch autograd (slower, not fused)
- TP and PP axes exist in DeviceMesh but have no compute mapped to them
- MFU numbers in README are illustrative, not measured
- Chaos node-kill test is quarantined pending expert reshard fix
- No etcd integration; c10d rendezvous only (suitable for < 100 nodes)
```

---

## RULES FOR THE AGENT

1. **Do not mark anything ✅ in the roadmap unless there is a passing CI test that verifies it.** A function existing in code is not the same as it being correct.

2. **Do not put fabricated numbers in the README or telemetry examples.** Every number that appears in documentation must either be clearly labeled `(illustrative)` or be pulled from an actual benchmark run with logs attached.

3. **Do not commit agent scaffolding files** (`pid.txt`, `test_result.md`, `SIGNAL_*.md`, `IMPLEMENTATION_CHECKLIST.md`, internal communication files). These belong in `.gitignore` or must never be created in the first place.

4. **Every new module must have a corresponding test** in `tests/` that runs under `pytest -m "not chaos"` on CPU with Gloo. No GPU required for unit tests.

5. **The order of work is strictly P0 → P1 → P2.** Do not start P1 until P0-1, P0-2, and P0-3 all have passing CI tests. Do not start P2 until all P1 items are CI-verified.

6. **When you are blocked**, write a `BLOCKERS.md` entry describing exactly what is blocking you (missing hardware, missing dependency, ambiguous spec) rather than silently scaffolding a placeholder and moving on.

---

**End of directive. Begin with the cleanup commit, then P0-1.**