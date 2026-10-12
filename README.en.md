# PyTorch backend for Intel NPU4000

<p align="center">
  <a href="./README.md"><kbd>简体中文</kbd></a>&nbsp;&nbsp;
  <a href="./README.en.md"><kbd><strong>English</strong></kbd></a>
</p>

Run PyTorch tensor operations, small-network training, and explicitly adapted model inference on the Intel NPU in Lunar Lake.
Built on [npunlock](https://github.com/hsfzxjy/npunlock), this project provides an `intel_npu` device backend,
custom SHAVE kernels, DPU graph execution for frozen models, and reuse of buffers, sessions, and compiled graphs.

**Current release: v0.1.0 Experimental Preview, dated 2026-10-11.**
Host components have been built from source in a separate local Conda environment and validated on real NPU hardware.
See [GitHub Releases](https://github.com/LCS-Official/torch-intel-npu/releases) for publication status.
All 45 offline checks passed in a local Git clone; GitHub-hosted CI had not executed at release time because of an account-level Actions restriction.
The project currently targets NPU operator development and model adaptation research; compatibility remains limited compared with general-purpose CUDA/XPU backends.

[Installation](INSTALL.md) · [Compatibility](COMPATIBILITY.md) · [Performance](#measured-performance) · [Roadmap](#roadmap) · [References](REFERENCES.md)

Release preparation and remaining validation are documented in [RELEASE_READINESS.md](RELEASE_READINESS.md),
with version details in [RELEASE_NOTES.md](RELEASE_NOTES.md). See [CONTRIBUTING.md](CONTRIBUTING.md) to report issues or contribute.
The detailed guides and research reports linked from this overview are currently primarily in Simplified Chinese.

## Two execution paths

| Path | Entry point | Current scope |
|---|---|---|
| PyTorch device backend | `tensor.to("intel_npu")`, `model.to("intel_npu")` | Supported FP16/FP32 operators, first-order gradients, and basic SGD; computation primarily uses custom SHAVE kernels |
| Frozen-model adapters | `QuantizedLinear`, `FrozenSmolLayer`, `FrozenSmolStack` | Explicitly constructed fused graphs with DPU matrix multiplication, for FP16 and selected low-bit inference experiments |

**Ordinary FP16 `torch.mm` currently computes through the FP32 SHAVE path and casts the result back to FP16. It does not automatically use the adapters' DPU FP16 matrix multiplication.**
Consequently, the model/DPU performance below does not apply to arbitrary models moved with `.to("intel_npu")`.
The frozen adapters do not provide LLM training.

The default runtime does not require the OpenVINO Python/runtime packages. It still requires the Intel NPU system driver and its graph compiler.
The device is registered through PyTorch PrivateUse1 as `intel_npu:0`; it is separate from `torch.xpu`.

## Quick start

The validated platform is **Windows x64 / Core Ultra 7 258V / NPU `8086:643e`**,
with driver `32.0.100.5540`, graph compiler `8.3`, Python 3.12.15, Torch 2.14.1+cpu, and NumPy 2.5.3.
Only this device/driver combination has been validated. With MSVC x64 and the Windows SDK available, run from the source root:

```powershell
conda create -p .\.conda\npu --no-default-packages --override-channels -c conda-forge python=3.12 pip -y
.\.conda\npu\python.exe -m pip install --no-cache-dir -r requirements-build.txt
# Build in an x64 Visual Studio Developer PowerShell
.\build_npu.ps1 -Python .\.conda\npu\python.exe
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\tools\smoke_install.py
```

A standalone MSVC toolchain can be selected with `-Toolchain`; see [INSTALL.md](INSTALL.md) for the full procedure.
The source build produces the host DLL, worker, session interface, and Torch PYD. By default, it uses the source-backed SHAVE ELF companions supplied with checksum validation.
Modifying SHAVE C requires an external MoviTools toolchain. There is currently no project wheel for direct `pip install`.

| Requirements file | Direct dependencies | Purpose |
|---|---|---|
| [requirements.txt](requirements.txt) | Torch CPU wheel, NumPy | Core operations and supported training |
| [requirements-models.txt](requirements-models.txt) | Core + safetensors, Transformers | Checkpoints, tokenizers, and model examples |
| [requirements-build.txt](requirements-build.txt) | Core + CMake, Ninja | Building from source; these two tools can be removed after the build |

Save the following as `my_test.py` and run it with the same entry point:
`run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\my_test.py`.

```python
import torch
import torch_intel_npu  # Import before the first backward call; registers intel_npu:0

x = torch.arange(17, dtype=torch.float32).to("intel_npu")
print((x + 70000).cpu())

model = torch.nn.Linear(4, 3).to("intel_npu")
inputs = torch.randn(2, 4).to("intel_npu")  # Random values are generated on the CPU
target = torch.zeros(2, 3).to("intel_npu")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(model(inputs), target)
loss.backward()
optimizer.step()
print(loss.detach().cpu())
```

## Current capabilities

| Feature | Implemented and validated scope |
|---|---|
| NPU tensors | FP16/FP32 storage, CPU/NPU copies, selected dense transposes and layout handling; Level Zero shared allocation |
| Operators and training | Add/subtract/multiply/divide, selected unary operations and reductions, 2D matrix multiplication, Linear, Conv2d, BatchNorm, ReLU, pooling, and MSE; first-order autograd and basic SGD for limited parameter combinations |
| True FP32 | FP32 arithmetic, compensated reductions, and matrix multiplication on SHAVE; this path is not replaced with FP16/TF32, but GEMM performance remains limited |
| FP16 model inference | The complete 30-layer decoder of SmolLM2-135M-Instruct, prefill, KV cache, and decode; decoder/KV run on the NPU, while embedding, final norm, lm_head, and sampling run on the CPU |
| Low-bit inference | Explicit `QuantizedLinear` experiments with INT8, INT4, NF4 weights and grouping; W8A16/W8A8 graph paths. INT4/NF4 support does not imply native 4-bit multiply-add or general low-bit device tensors |
| Runtime optimizations | Graph fusion, native session reuse, chained submission, buffer reuse, and optional disk graph caching; selected FP16 prefill cache hits can skip IR reconstruction |
| Installation baseline | Source build in a new Conda environment; the earlier installation validation passed 44 offline tests, two native session regression suites, and real-NPU operation/training/single-layer model smoke checks |

The [operator/parameter/gradient compatibility matrix](COMPATIBILITY.md) and its corresponding tests define the supported contracts.
A module name in the table does not imply support for every parameter combination.
See [INSTALLATION_REPORT.md](INSTALLATION_REPORT.md) for installation validation and [TORCH_NPU_README.md](TORCH_NPU_README.md) for historical operator/training details.

### Known limits

- Device tensors currently support ranks 0–4 and 1–1,048,576 elements, with additional intermediate-workspace limits. These are software limits, not the machine's physical-memory capacity.
- The SmolLM adapter has a fixed 64-position KV capacity and a maximum batch size of 8, subject to shape/workspace constraints. It does not directly support arbitrary Hugging Face models or long contexts.
- Deep training entirely in FP16 still has numerical limitations. AMP, second-order gradients, differentiable CPU/NPU copies, broad optimizer coverage, and arbitrary offset/overlapping views are not covered.
- General BF16/integer device tensors, device RNG, asynchronous multi-stream execution, and GPU+NPU cooperation are not implemented. Unsupported operators raise errors rather than silently falling back to CPU arithmetic.
- Full-model W8 calibration has not passed every existing quality threshold. INT4/NF4 remain primarily Linear/small-network experiments; full low-bit LLM adaptation is not established. Current model evaluations explicitly prioritize FP16.
- Shared physical memory across CPU/GPU/NPU does not imply zero transfers or zero synchronization. It does not eliminate driver-internal DMA or precision/layout conversions.

## Measured performance

These are historical measurements from saved reports, **not chip peaks or a general model ranking**.
Performance was not remeasured for this README update. Each table retains its own date, execution path, and timing scope.

### Model generation: warm requests

2026-10-10, Core Ultra 7 258V / Arc 140V / NPU4000, complete 30-layer SmolLM2-135M-Instruct.
A fixed 24-token prompt and greedy generation of 32 tokens were used, with 2 additional warmups and the median of 7 measured runs.
Values below are **aggregate tokens/s across the entire batch**. Timing includes the CPU stages, per-step token return to the host, and synchronization; loading and initial compilation are excluded.

| Path | Batch 1 | Batch 4 | Batch 8 |
|---|---:|---:|---:|
| CPU FP16, 8 threads | 30.668 | 69.064 | 108.509 |
| Arc 140V FP16, eager | 25.446 | 105.455 | 207.850 |
| Arc 140V FP16, SDPA | 32.325 | 122.545 | 196.090 |
| NPU FP16 decoder + CPU stages | 72.925 | 167.394 | 288.046 |

At batch 8, the NPU's aggregate 288.046 tokens/s corresponds to about 36.01 tokens/s per request, not single-request throughput.
Across these NPU configurations, the maximum logits relative L2 error against the CPU FP16 reference was about 0.74%–1.12%,
and generated tokens matched in the fixed tests. This does not replace a full language-model quality evaluation.

CPU/GPU use the Transformers model; the NPU uses a purpose-built fused adapter. Device placement and Torch environments also differ.
Desktop load and frequencies were not locked. Paired GPU batch-8 checks showed substantial variation, so these results do not establish a stable ranking or speed ratio.
See the [model comparison report](npunlock-probe/research/results/model-cross-device-followup/REPORT.md) for full conditions, measurement summaries, and quality thresholds.

### First token: a new process with an existing cache

2026-10-11, the same model, FP16, batch 1, and a 24-token prompt, using the same 30 compiled prefill graphs.
The comparison alternated 3 new processes per path. Both paths had an existing disk cache; the difference was whether an early hit skipped IR reconstruction.

| Timing scope | Full IR path | Early cache hit | Reduction in median time |
|---|---:|---:|---:|
| After model preparation to first token | 2.009 s | 0.967 s | 51.8% |
| Python script entry to first token | 6.795 s | 5.491 s | 19.2% |

The second row includes library imports, initialization, and model preparation, but excludes PowerShell/Python interpreter startup.
**The 0.967 s result requires an existing disk cache.** An uncached initial prefill compilation sample in the same experiment took about 28.56 s;
cache-hit improvements do not describe a first-install experience. This change did not demonstrate higher warm throughput.
See the [first-token optimization report](npunlock-probe/research/results/startup-ir-cache/REPORT.md).

Disk graph caching is disabled by default and must be enabled explicitly with `NPUNLOCK_GRAPH_CACHE`; the capacity limit is 512 MiB per directory.
Deleting the cache requires compilation again on the next run. See [installation](INSTALL.md) and [storage records](WORKSPACE_STORAGE.md) for commands and storage policy.

### Matrix multiplication: effective throughput at a specified shape

2026-10-05, logical operation count `2*M*K*N`, all shapes `1024×1024 @ 1024×1024`, frozen weights, FP16 output.
Inputs/outputs were preallocated. Timing excludes compilation and explicit uploads/downloads, but includes driver calls, in-graph DMA, and fences; median of 31 runs.

| Execution path | Median time | Effective throughput |
|---|---:|---:|
| DPU FP16 | 0.6733 ms | 3.18949 TFLOPS |
| DPU W8A16 | 0.6308 ms | 3.40438 TOp/s |
| DPU W8A8 | 0.5462 ms | 3.93168 TOPS |

W8A16 quantizes only the weights and must not be described as all-INT8 TOPS. Actual 8-bit DPU operands were checked for W8A8.
In the same experiment, ordinary FP32 `torch.mm` on 256-square matrices took 614.8185 ms, about 0.05458 GFLOPS,
through the SHAVE tiling/merge path. Its precision, shape, and timing scope differ from the DPU measurements above.
It shows that the current general FP32 GEMM remains slow; it does not measure the chip's FP32 peak.
See the [matrix-throughput report](npunlock-probe/research/results/throughput-20261005/REPORT.md).

## Roadmap

These directions are ordered by current evidence. They are not implemented features or delivery-date commitments.

| Priority | Direction | Next steps and validation |
|---|---|---|
| Near term | Broader release validation | Local Quick/non-Quick results are in [release validation](RELEASE_READINESS.md); add long-running stress regressions and installation validation on another clean Windows machine |
| Near term | First-token and runtime overhead | Profile repeated ELF/DPU parsing on cache hits, library imports, and repeated model preparation; preserve validation and use alternating A/B runs with output checks |
| Near term | Model coverage and quality | Expand prompt, batch, KV-boundary, and numerical tests before extending contexts/models; W8/INT4/NF4 must each pass quality thresholds |
| Medium term | Real-model hotspots | Remeasure RMSNorm/SHAVE instruction and register overhead, improve buffer reuse at larger batches, and verify each change's end-to-end benefit |
| Medium term | General operators and training | Add dispatch and gradients for operators missing from real models, improve general FP32 GEMM, and establish precision semantics before evaluating AMP and more optimizers |
| Long term | GPU+NPU cooperation and distribution | Investigate layer/pipeline scheduling and memory/synchronization costs before automatic graph partitioning; improve packaging and broaden device/driver validation |

A stable v1.0 requires explicit, reproducible compatibility, correctness, installation, and performance baselines, plus broader model/environment validation.
The current single-machine experiments do not establish general-purpose training or GPU+NPU acceleration.
Optimization candidates enter the default path only after passing numerical checks and demonstrating a measured benefit.
For example, the [four-row SIMD reduction candidate](npunlock-probe/research/results/shave-fp32-pipeline/REPORT.md) passed numerical checks but was slower, so it was not enabled.

## Source, documentation, and reproduction

| Path / document | Contents |
|---|---|
| `torch_intel_npu.py`, `torch_npu_*.py` | Device backend, runtime, quantization, model adapters, and examples |
| `npunlock/` | Modified base library; see [UPSTREAM.md](UPSTREAM.md) for the upstream baseline and changes |
| `npunlock/tests/fixtures/npu4000/` | Project SHAVE C, assembly, ELF companions, and manifests |
| `npunlock-probe/torch-device/backend.cpp` | PrivateUse1 native extension |
| `npunlock-probe/research/` | Diagnostic, build, and optimization experiment sources and Markdown reports |
| [INSTALL.md](INSTALL.md) / [INSTALLATION_REPORT.md](INSTALLATION_REPORT.md) | Installation commands, dependencies, and the scope of fresh-environment validation |
| [COMPATIBILITY.md](COMPATIBILITY.md) | Operator, precision, gradient, layout, and model support; release regression baseline |
| [TORCH_NPU_README.md](TORCH_NPU_README.md) | Detailed historical progress; old paths/experiment links may no longer work |
| [SOURCE_RELEASE_REPORT.md](SOURCE_RELEASE_REPORT.md) | Source-only packaging, provenance, snapshot validation, and export boundaries |

The source snapshot is about 6.1 MiB, with per-file SHA-256 records in `SOURCE_MANIFEST.json`.
Models, environments, SDKs/toolchains, host binaries, and driver-generated `.blob` files are excluded; full fixture regressions need separately prepared graph data.
Historical raw arrays, graph caches, and timing JSON were removed under the storage policy. Reports retain measurement summaries; rerunning experiments requires regenerating inputs/outputs.

<details>
<summary>Offline tests and source-snapshot verification</summary>

After installing the core dependencies, run the 45 basic regression checks without NPU hardware:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
python -B -m unittest test_torch_npu_geometry test_torch_npu_lowbit
python -B -m unittest discover -s npunlock/tests/python -p test_python_api.py
python -B tools/test_source_release.py
```

Before the first build, run `python -B tools/export_source_release.py --verify .` in the exported directory to check the file set and hashes.
Local build outputs will be reported as extra files by strict snapshot verification. From a development workspace retaining the upstream Git metadata, export again with:

```powershell
python -B tools/export_source_release.py --output source-release-next
python -B tools/export_source_release.py --verify source-release-next
```

The exporter refuses to overwrite an existing directory. Snapshots exclude nested `.git` metadata.
Historical research audits may depend on deleted data and are not included in these 45 regression checks.

</details>

## License and references

New project code is licensed under Apache-2.0, matching upstream npunlock. Third-party adaptations retain their respective license and copyright notices.
**npunlock is the direct foundation.** Selected public OpenVINO implementations and Intel NPU compiler/ELF/Level Zero sources are also explicitly attributed.
See [REFERENCES.md](REFERENCES.md), [references.bib](references.bib), [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [LICENSE](LICENSE).
When citing this project, record the actual version/commit and acknowledge the upstream work used. A bibliography does not replace license notices.
