# A Composed Mixture-of-Experts Engine (`moe-engine`)

Production-grade reference implementation of a hyperscale Mixture-of-Experts
training engine. Bridges three layers of the stack that typically live in
separate teams at FAANG/frontier-lab scale:

1. **Hardware-aware kernels** – custom Triton implementation of the sparse
   Top-K MoE router (forward + backward) with coalesced HBM access, SRAM
   reuse, and Tensor-Core friendly tile shapes. Validated against a
   double-precision PyTorch autograd reference at `atol = rtol = 1e-5`.
2. **3D distributed parallelism** – PyTorch 2.5+ native `init_device_mesh`,
   FSDP2 (`fully_shard`) with per-parameter DTensor sharding for Data
   Parallel, Expert Parallel (EP) `all_to_all_single` dispatch on a
   dedicated CUDA stream, and Tensor / Pipeline Parallel hooks.
3. **Fault-tolerant elastic infra** – TorchElastic-compatible harness,
   asynchronous sharded checkpointing (`SHARDED_STATE_DICT`) streamed via
   background I/O workers to a two-tier (local NVMe → S3 / MinIO) durable
   store, and a cluster state-machine that evicts dead ranks, recomputes
   the `DeviceMesh` slice, reshards experts, and hot-resumes training
   without operator intervention.

```
moe-engine/
├── pkg/
│   ├── kernels/
│   │   └── moe_router.py        # Triton top-k routing kernel + CPU fallback
│   ├── distributed/
│   │   └── parallel_mesh.py     # DeviceMesh, FSDP2 DTensor, EP all-to-all
│   ├── elastic/
│   │   └── fault_monitor.py     # Async ckpt, S3/MinIO, elastic resharding
│   ├── telemetry/
│   │   └── logger.py            # Structured JSON + TensorBoard exporter
│   └── utils/
│       └── mfu.py               # Model-FLOPs-Utilization accountant
├── tests/
│   ├── test_kernels.py
│   ├── test_distributed.py
│   └── test_elastic.py
├── scripts/
│   ├── launch.sh                # torchrun elastic launcher
│   └── simulate_node_failure.sh # chaos: drop 4 nodes mid-run
├── configs/default.yaml
└── train.py                     # Unified training loop
```

## Acceptance gates enforced in CI

| Gate                          | Target                                | Tested in              |
|-------------------------------|---------------------------------------|------------------------|
| Numerical tolerance           | `atol < 1e-5`, `rtol < 1e-5` (fp64)   | `tests/test_kernels.py`|
| Token conservation            | `Σ dispatched == B·S`                 | `tests/test_kernels.py`|
| MFU target                    | `>= 0.55` of theoretical peak         | `pkg/utils/mfu.py`     |
| Communication overlap         | `all_to_all` on dedicated CUDA stream | `pkg/distributed/`     |
| Zero-divergence resharding    | Continuous expert redistribution      | `tests/test_elastic.py`|

## Quick start

```bash
# Install (Python 3.10+, CUDA 12+ recommended)
pip install -r requirements.txt

# Single-node smoke test
python train.py --config configs/default.yaml --world-size 1

# Multi-node elastic (TorchElastic) launch
bash scripts/launch.sh

# Inject a 4-node failure mid-run (chaos test)
bash scripts/simulate_node_failure.sh
```

## Telemetry

Every training step emits a structured JSON envelope:

```json
{
  "step": 1024,
  "loss": 2.187,
  "mfu": 0.612,
  "tokens_per_sec": 184320,
  "kernel": {"sram_bytes_per_block": 49152, "achieved_bw_gbps": 1843.0},
  "collective": {"all_to_all_ms": 0.87, "all_gather_ms": 1.21, "reduce_scatter_ms": 1.05},
  "memory": {"peak_allocated_gb": 62.4, "reserved_gb": 70.0, "leak_delta_gb": 0.0},
  "infra": {"async_ckpt_commit_ms": 412.0, "active_nodes": 1250, "ep_world_size": 64}
}
```

These are also fanned out to TensorBoard via `pkg/telemetry/logger.py`.

## License
Apache-2.0
