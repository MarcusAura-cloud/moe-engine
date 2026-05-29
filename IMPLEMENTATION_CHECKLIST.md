# Implementation Checklist: Top 1% Profile Transformation

**Timeline:** 4 weeks → Top 1% signal (weeks 1–4 focus on correctness & positioning)

---

## WEEK 1: P0 Correctness Foundation

### Task 1.1: Numerics Test for Triton Backward
**File:** `moe-engine/tests/test_kernels_numerics.py` (new file)

```python
# Test: Triton backward vs FP64 reference
# - 100 random seeds, various H/E/K combinations
# - Assert atol < 1e-5, rtol < 1e-5 for gradients
# - Test on CPU (reference) and GPU (Triton) separately if available
```

**Acceptance:** PR passes CI with test results logged

---

### Task 1.2: Token Conservation Assertion
**File:** `moe-engine/pkg/kernels/moe_router.py`

**Change:** In `MoERouter.forward()`, after computing `dispatch_cnt`, add:
```python
assert dispatch_cnt.sum().item() == N * K, f"Token loss: {dispatch_cnt.sum()} != {N*K}"
```

**File:** `moe-engine/pkg/distributed/parallel_mesh.py`

**Change:** In `DistributedMoELayer.forward()`, after combine, add:
```python
# Verify token conservation across all-to-alls
assert received.shape[0] == total_recv, "Dispatch: shape mismatch"
assert combined.shape[0] == (B * S * K), "Combine: shape mismatch"
```

**Acceptance:** Tests run with assertions; code fails fast if violated

---

### Task 1.3: Fix Integer-ID Exchange
**File:** `moe-engine/pkg/distributed/parallel_mesh.py`

**Current Code Problem:**
```python
# Placeholder: re-derive expert id from ordering (FRAGILE)
ids_to_send = idx_sorted.to(torch.int64)
ids_recv = self._exchange_ids(ids_to_send, send_counts, recv_counts)
# Then use mask: ids_recv == gi
```

**Fix:**
```python
# Properly exchange expert ids via all_to_all_single
ids_recv = self._exchange_ids(ids_sorted, send_counts, recv_counts)
# Ensure it's packed correctly: each token carries its global expert id
```

Verify the `_exchange_ids` implementation is correct (it looks OK; just ensure no commented-out calls are replaced).

**Acceptance:** 4-rank test run with expert assignment logging; verify all tokens land on correct expert

---

### Task 1.4: CI Gate for Correctness
**File:** `.github/workflows/test.yml` (new or update existing)

```yaml
name: Correctness Gates
on: [push, pull_request]
jobs:
  numerics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r moe-engine/requirements.txt
      - run: pytest moe-engine/tests/test_kernels_numerics.py -v
  token-conservation:
    runs-on: ubuntu-latest
    steps:
      - run: pytest moe-engine/tests/test_smoke_e2e.py -v -k "token"
```

**Acceptance:** All commits require passing gates

---

## WEEK 2: Distributed Verification & Chaos Testing

### Task 2.1: Multi-Rank Gradient Parity Test
**File:** `moe-engine/tests/test_distributed_correctness.py` (new)

```python
# Simulate 4-rank distributed MoE on CPU/Gloo
# - Run forward/backward on each rank
# - Collect gradients
# - Verify: max_gradient_diff < atol
```

**Acceptance:** Test runs, logs gradient diffs per parameter

---

### Task 2.2: Chaos Test: Kill Rank & Recover
**File:** `moe-engine/tests/test_chaos_recovery.py` (new)

```python
# 1. Start 4-process training
# 2. At step 3, kill process rank 2
# 3. Call harness.recover()
# 4. Resume training
# 5. Verify loss < threshold (deterministic)
```

**Acceptance:** Test passes; loss matches reference run after recovery

---

### Task 2.3: Update pyproject.toml for Chaos Tests
**File:** `moe-engine/pyproject.toml`

```ini
[tool.pytest.ini_options]
markers = [
    "chaos: fault-injection tests (slow, requires multiprocessing)",
]
```

**Acceptance:** `pytest -m chaos` runs chaos tests

---

## WEEK 3: Benchmarks & Metrics

### Task 3.1: Kernel Latency Benchmark
**File:** `moe-engine/benchmarks/bench_kernel.py` (new)

```python
# Measure _router_fwd_kernel latency for H=1024, E=256, K=2
# - Time via torch.cuda.Event (not placeholder)
# - Measure bytes moved, compute GB/s
# - Run 10 iterations, report min/max/mean
# - Compare Triton vs reference
```

**Acceptance:** Benchmark runs, outputs JSON with latency in ms

---

### Task 3.2: Async Overlap Benchmark
**File:** `moe-engine/benchmarks/bench_async_overlap.py` (new)

```python
# On 4-GPU machine:
# - Measure all_to_all_single latency
# - Measure expert compute latency
# - Compute overlap ratio = min(comm, compute) / max(comm, compute)
# - Log per step
```

**Acceptance:** Benchmark outputs overlap_ratio; target >= 0.6

---

### Task 3.3: End-to-End MFU Benchmark
**File:** `moe-engine/benchmarks/bench_e2e_mfu.py` (new)

```python
# Run smoke config for 100 steps
# - Track MFU per step
# - Compute mean MFU
# - Output plot (or table) showing trend
```

**Acceptance:** Benchmark outputs MFU table; target >= 0.55

---

### Task 3.4: Write BENCHMARKS.md
**File:** `moe-engine/BENCHMARKS.md` (new)

```markdown
# Benchmark Results

## Hardware
- 4× A100 / H100 (or local machine)
- Date: [date]

## Kernel (Triton Router)
| Metric | Value | Notes |
|--------|-------|-------|
| Forward latency | 0.34 ms | H=1024, E=256, K=2 |
| Backward latency | 0.45 ms | Same config |
| Memory bandwidth | 1.8 TB/s | Achieved on H100 |
| Precision (vs FP64) | atol=2e-6 | See test_kernels_numerics |

## Async Overlap
| Config | Comm (ms) | Compute (ms) | Overlap Ratio |
|--------|-----------|--------------|---------------|
| 4-GPU, B=32, S=512 | 2.1 | 3.4 | 62% |

## End-to-End MFU
- Configuration: 4× H100, hidden=1024, experts=256, top_k=2
- Steps: 100
- Mean MFU: 0.58 (target: 0.55) ✓
- Throughput: 145k tokens/sec

## See Also
- Numerics validation: `tests/test_kernels_numerics.py`
- Distributed correctness: `tests/test_distributed_correctness.py`
```

**Acceptance:** File exists with real benchmark numbers

---

## WEEK 4: Reposition & Publish

### Task 4.1: Rewrite README.md
**File:** `moe-engine/README.md`

**New structure:**
```markdown
# moe-engine: A Validated Reference Implementation for Distributed MoE Training

## Overview (Pitch)
[Clarify: reference impl, not product; scope gaps documented]

## Design Rationale
[Why this architecture; trade-offs explained]

## Measured Performance (Benchmarks)
[Kernel latency, async overlap, MFU — actual numbers from BENCHMARKS.md]

## Correctness Verification
[Numerics tests, distributed invariants, chaos tests]

## Known Limitations & Gaps
[Honest assessment: no TP/PP, no NVMe streaming, etc.]

## Real-World Example
[Link to companion repo or run logs]

## Getting Started
[Install, smoke test, run benchmarks]
```

**Acceptance:** README is honest and evidence-driven

---

### Task 4.2: Write DESIGN.md
**File:** `moe-engine/DESIGN.md` (new)

```markdown
# Architecture & Design Rationale

## Problem Statement
At 1k+ GPUs, three challenges:
1. Hardware efficiency (sparse routing)
2. Communication bottleneck (all-to-all)
3. Fault resilience (node failures)

## Solution: 2D Mesh (Data × Expert Parallelism)
[Explain architecture, trade-offs, why these choices]

## Trade-off Analysis
| Design | Pro | Con |
|--------|-----|-----|
| Async all-to-all | Overlaps comm | Complexity |
| Background checkpointing | No stall | Pinned memory cost |
| Custom Triton kernel | 3.5x speedup | GPU-only, maintenance |

## Future Extensions
- Tensor Parallelism: Weight-splitting utilities exist; integration pending
- Pipeline Parallelism: Requires stage partitioning + bubble scheduling
- NVMe Streaming: Design exists; integration pending
- Etcd Heartbeat: Barrier works for <100 nodes; etcd recommended for scale

(See ROADMAP.md for implementation roadmap)
```

**Acceptance:** File exists with honest design discussion

---

### Task 4.3: Write ROADMAP.md
**File:** `moe-engine/ROADMAP.md` (new)

```markdown
# Roadmap: Future Extensions

## v1.1 (Next 4 weeks)
- [ ] Streaming checkpoint I/O (avoid OOM on large shards)
- [ ] Etcd-based heartbeat (scale > 100 nodes)
- [ ] Nsight/CUPTI kernel profiling integration

## v2.0 (Medium-term)
- [ ] Tensor Parallelism (TP): weight-splitting + collective coordination
- [ ] Pipeline Parallelism (PP): stage partitioning + microbatch scheduling
- [ ] NVMe streaming checkpoint I/O

## Research Directions
- [ ] Dynamic expert assignment (learnable routing instead of fixed)
- [ ] Sparse attention integration
- [ ] Distributed LoRA / adapters on top of MoE

## Contributing
See CONTRIBUTING.md for guidelines.
```

**Acceptance:** File exists with clear roadmap

---

### Task 4.4: Update GitHub Profile Bio
**Current location:** GitHub profile settings

**Change from:**
```
🚀 Building hyperscale ML systems | MoE specialist | Distributed training
```

**To:**
```
ML systems engineer: building validated, fault-tolerant distributed training 
engines. Creator of moe-engine (reference impl with measured benchmarks + 
chaos-tested resilience).
```

**Acceptance:** Bio updated

---

### Task 4.5: Update Project Description (on GitHub)
**Current:**
```
A production-grade hyperscale MoE training engine.
```

**New:**
```
Validated reference implementation for distributed MoE training with 
measured async overlap, chaos-tested fault recovery, and published benchmarks.
Implements Data×Expert parallelism; TP/PP/NVMe scaffolded.
```

**Acceptance:** Description updated

---

### Task 4.6: Tag Release v1.0-reference
**Command:**
```bash
git tag -a v1.0-reference -m "Correctness-verified, benchmark-validated reference implementation"
git push origin v1.0-reference
```

**Acceptance:** Tag pushed to GitHub

---

## WEEK 5+ (Optional Hardening)

### Task 5.1: Streaming Checkpoint I/O
**File:** `moe-engine/pkg/elastic/fault_monitor.py`

**Enhancement:** `AsyncCheckpointer.load()` should stream download from remote to NVMe instead of full load-to-memory.

---

### Task 5.2: Etcd Heartbeat Integration
**File:** `moe-engine/pkg/elastic/fault_monitor.py`

**Enhancement:** `ClusterStateMachine.heartbeat()` should use etcd-based rendezvous (via TorchElastic) instead of barrier.

---

## Verification Checklist (Before publishing v1.0)

- [ ] All correctness tests pass (numerics, token conservation, distributed invariants)
- [ ] All chaos tests pass (rank recovery, deterministic gradient matching)
- [ ] Benchmarks run without errors (kernel, overlap, e2e MFU)
- [ ] BENCHMARKS.md populated with real numbers
- [ ] README, DESIGN.md, ROADMAP.md written
- [ ] GitHub profile/bio updated
- [ ] v1.0-reference tagged and pushed
- [ ] CI gates in place for all correctness checks

---

## Success Metrics (End of Week 4)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Numerics tests pass | ✓ | 0/3 | ❌ |
| Distributed invariant tests pass | ✓ | 0/2 | ❌ |
| Chaos tests pass | ✓ | 0/1 | ❌ |
| Benchmarks exist + published | ✓ | 0/3 | ❌ |
| README honest + evidence-driven | ✓ | Partial | ⚠️ |
| GitHub profile updated | ✓ | No | ❌ |
| v1.0-reference tagged | ✓ | No | ❌ |

**Final Goal:** All metrics ✓ by end of Week 4

---

