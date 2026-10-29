# PyTorch backend for Intel NPU4000

<p align="center">
  <a href="./README.md"><kbd><strong>English</strong></kbd></a>&nbsp;&nbsp;
  <a href="./README.zh-CN.md"><kbd>简体中文</kbd></a>
</p>

Run PyTorch tensor operations, small-network training, and explicitly adapted model inference on the Intel NPU in Lunar Lake.
Built on [npunlock](https://github.com/hsfzxjy/npunlock), this project provides an `intel_npu` device backend,
custom SHAVE kernels, DPU graph execution for frozen models, and reuse of buffers, sessions, and compiled graphs.

**Version: v0.2.0 Experimental Preview, dated 2026-10-27.**
This update includes the compatibility, capacity, AMP, attention and model work through S45. Installation and hardware validation scope are recorded in [release validation](RELEASE_READINESS.md).
See [GitHub Releases](https://github.com/LCS-Official/torch-intel-npu/releases) for publication status.
Offline source tests and real-NPU checks are reported separately from GitHub-hosted CI; see [release validation](RELEASE_READINESS.md) for the current results.
The project currently targets NPU operator development and model adaptation research; compatibility remains limited compared with general-purpose CUDA/XPU backends.

[Installation](INSTALL.md) · [Compatibility](COMPATIBILITY.md) · [Performance](#measured-performance) · [Roadmap](#roadmap) · [References](REFERENCES.md)

Source-build and hardware validation are documented in [RELEASE_READINESS.md](RELEASE_READINESS.md),
with version details in [RELEASE_NOTES.md](RELEASE_NOTES.md). See [CONTRIBUTING.md](CONTRIBUTING.md) to report issues or contribute.
The detailed guides and research reports linked from this overview are currently primarily in Simplified Chinese.

**Development study after v0.2.0 — S48:** [W8 calibration on quantized prefixes](npunlock-probe/research/results/w8-output-quality/REPORT.md) evaluates eight FFN-down/output-residual candidates, including a fixed-input-basis constraint. The best internal candidate does not give a stable NPU quality gain: KV passing positions stay unchanged and device perplexity increases slightly. Production defaults and quality gates remain unchanged; the report includes CPU/NPU forward, KV, text and generation checks.

**Development study after v0.2.0 — S49:** [device numerical checks and internal selection](npunlock-probe/research/results/w8-device-quality/REPORT.md) isolate the FFN-down chain using actual NPU inputs and evaluate all existing candidates on eight internal trajectories. In five sampled layers the correction additions match, while larger differences occur at gated inputs and the main W8 projection. Device selection retains the S48 candidate, whose independent quality remains insufficient for promotion. Defaults and quality gates stay unchanged.

**Development study after v0.2.0 — S50:** [gate/up/SiLU isolation](npunlock-probe/research/results/w8-gate-quality/REPORT.md) separates projection, activation and product errors on actual forward/decode inputs. A SHAVE activation prototype improves all 20 sampled local FFN comparisons, but its internal maximum KL worsens and warmed submissions rise from 30 to 660. It does not meet the predefined promotion rule; production defaults stay unchanged.

**Development study after v0.2.0 — S51:** [bounded W8 scale/correction calibration](npunlock-probe/research/results/w8-scale-quality/REPORT.md) lowers calibration projection SSE by 0.889%, but internal NPU passes fall from 243 to 242/290. The full-FFN candidate is rejected by the existing local precision guard. A shared-prefix diagnostic traces one reproduced peak through the attention boundary; it does not establish a general root cause. Defaults stay unchanged.

**Development study after v0.2.0 — S52:** [attention stage diagnosis](npunlock-probe/research/results/w8-attention-quality/REPORT.md) reproduces the known peak and tests QK, softmax and PV replacements on the NPU. FP32 softmax improves the sampled local L2 errors, but full-decoder internal passes fall from 243 to 242/290 and warmed submissions rise from 30 to 240. The prototype is not promoted; the next target is calibration using actual NPU Q/K/V outputs against the original FP16 model.

**Development study after v0.2.0 — S53:** [device-observed Q/K/V calibration](npunlock-probe/research/results/w8-qkv-quality/REPORT.md) fits only the existing rank-32 output factors, retaining INT8 codes/scales, the input basis and 30 warmed submissions. The 0.50 blend lowers internal mean logits L2 from 2.041913% to 2.016721%, but passes fall from 243 to 241/290; all three fixed blends fail the original promotion rule. Defaults remain unchanged. Further work should target whole-model logits rather than local projection SSE.

**Development study after v0.2.0 — S54:** [whole-model logits/KL fitting](npunlock-probe/research/results/w8-logits-quality/REPORT.md) updates only existing Q/K/V rank-32 output factors, using CPU preparation and real NPU evaluation. The frozen second-epoch candidate raises passes from 243 to 244/290 internally and from 246 to 248/262 on eight new prompts. Maximum KL rises, and new-prompt maximum L2 worsens, so it does not meet the original promotion rule. Defaults remain unchanged.

**Development study after v0.2.0 — S55:** [upper-tail fitting and generalization audit](npunlock-probe/research/results/w8-tail-quality/REPORT.md) expands calibration to 48 prompts and updates all existing rank-32 output factors with full-gradient, score-checked steps. Calibration improves, but NPU internal passes stay at 243/290, the S54 regression falls from 246 to 245/262, and eight new prompts fall from 245 to 242/266. One earlier checkpoint is rejected by the unchanged local guard. Quantized CPU and NPU baselines both exceed the KL limit on a new prompt; defaults retain baseline.

**Development study after v0.2.0 — S56:** [real-NPU calibration selection](npunlock-probe/research/results/w8-native-selection/REPORT.md) compares six fixed correction-update blends using actual decoder outputs before freezing the candidate. The native calibration objective improves by 8.60%, but internal passes stay at 243/290 and viewed regression passes fall from 491 to 487/528. Both baseline and candidate hit the original final-layer guard on the first new prompt, so new-set NPU quality remains unscored. Defaults retain baseline; independent quality is still unresolved.

S31 improves [loss and operator capacity](npunlock-probe/research/results/operator-capacity/REPORT.md): hard-label CE/NLL tile complete class rows, and multi-packet training recomputes bounded masks instead of retaining a batch-wide dense mask. A million-row NLL uses 43 cached graphs and matches the complete CPU gradient. Ordinary Conv2d outputs, native Transformer dense workspaces, Half masked softmax and multi-output DMA concatenation use current storage checks; the report separates measured cases from implementation limits. Existing kernels are reused without new dependencies or host builds. Included in v0.2.0.

S30 improves [large tensors and graph reuse](npunlock-probe/research/results/memory-scale/REPORT.md): allocations up to 512 MiB, page-aligned range bindings, element-count-based pointwise graph keys, and streamed writes into final buffers. Hardware checks cover complete gradients beyond the old 256 MiB limit, 2,050 offsets, and 1,820 logical shapes. A 134.25M-parameter network completes two FP32 AdamW steps with 2.00 GiB of model states and a 2.25 GiB backend allocation peak; the report distinguishes full CPU references from sampled update checks. The 1,536-graph limit, operator workspaces and single-buffer storage remain. Training near 14 GiB has not been certified. Rebuild the session DLL and Torch PYD; no new dependencies or SHAVE ELF files. Included in v0.2.0.

S32 improves [capacity and graph lifetimes](npunlock-probe/research/results/graph-capacity/REPORT.md): up to 1 GiB per allocation (512 MiB per logical FP16/BF16 tensor), LRU recycling of ordinary operator graphs, and deferred release while a call or prepared chain retains a session. Custom frozen graphs stay pinned; a cache filled entirely with pinned graphs still rejects new entries. Rebuild the session DLL and Torch PYD. No new dependencies or SHAVE ELF files; measured coverage and remaining limits are recorded in the report. Included in v0.2.0.

S33 improves [frozen-model capacity and graph caching](npunlock-probe/research/results/frozen-capacity/REPORT.md): INT8/INT4/NF4 Linear, Transformer, SmolLM and rectangular GEMM graphs can be rebuilt during calls and participate in LRU recycling. Cached graph blobs are bounded by 1,536 entries and 512 MiB; a rebuildable graph above that budget executes transiently. Weak factory registration does not retain the model, and prepared decode chains retain their active sessions. A 268.50M-parameter network also completes two FP32 AdamW steps with 4.00 GiB of model states and a 4.25 GiB backend allocation peak. CPU checks cover sampled updates; the run takes about 35 minutes and does not establish a training speedup or certify 14 GiB training. Existing S32 host components need no rebuild or new dependencies. See the report for measured coverage and remaining limits. Included in v0.2.0.

## Two execution paths

S45 fixes [residual-factor cache identity and Smol graph-family retirement](npunlock-probe/research/results/smollm-cache-lifetime/REPORT.md). Two FP16 forward/KV cycles pass with the default 512 MiB budget. Cache pressure retires the family together; grouped plans reject insufficient budgets safely, with explicit resizing through torch.intel_npu.set_graph_cache_budget. The same 16 W8 correction failures were reproduced on the NPU; 14 include a substantial common logit offset. [Neither new calibration candidate won, and expanded model checks still show regressions](npunlock-probe/research/results/w8-guarded-quality/REPORT.md). Gates and INT4/NF4 whole-model certification are unchanged. No new dependencies; included in v0.2.0. The underlying native teardown cause remains unresolved.

S44 adds [optional low-rank W8 residual correction](npunlock-probe/research/results/w8-lowrank-quality/REPORT.md). INT8 projections plus small FP16 factors reduce mean logit error from 2.24% to 1.71% across the 30-layer SmolLM2-135M NPU regression, increasing passing positions from 118/142 to 126/142. Decode constants remain 38% below FP16. PPL improves by less than 0.01%; all eight generated sequences match W8, and 16 positions still fail the original gates. The option stays disabled by default. No new dependencies or kernel rebuilds; the FP16 control's cache-budget change and device error are documented separately. INT4/NF4 whole-model certification is unchanged. Included in v0.2.0.

S43 optimizes [SHAVE tiled attention and automatic block selection](npunlock-probe/research/results/flash-optimization/REPORT.md). For warmed forward plus backward at 128×256 with head dimension 16, FP32 decreases from 3.59 to 1.72 seconds and FP16 from 3.71 to 1.85 seconds, about 2× faster. Graph calls decrease by about 70%; Tensor allocation peaks remain 86–87% below math. The changes live in the C kernels and backend, preserving IEEE edge paths and packet budgets. This case remains about twice as slow as math; no whole-model or general training speedup is established. No new dependencies or DLL/PYD rebuilds. Included in v0.2.0.

S42 adds [experimental SHAVE tiled attention and strided NestedTensor inference](npunlock-probe/research/results/flash-nested/REPORT.md). Standard SDPA can explicitly select online softmax with recomputed Q/K/V gradients for bounded FP16/FP32 and GQA. Default Encoder nested inference, copying, MHA/LayerNorm and Nested SDPA are connected. A 128×256 case reduces backend Tensor allocation peaks by 86–88%, while the cold test including compilation is about 3× slower; no speedup is established. Two ELF kernels are added without dependency or DLL/PYD changes. Dropout, trainable masks, jagged/Nested training and steady-state tuning remain incomplete. Included in v0.2.0.

S41 expands [attention and original HF model migration](npunlock-probe/research/results/attention-migration/REPORT.md): position arange, integer cumsum/isin/label padding, empty KV concatenation and gradients, RoPE partial-view operations, and empty native MHA/Encoder inputs. Original two-layer, randomly initialized Llama/GPT-2 configurations pass FP16/FP32 inference, caching and greedy generate, plus FP32 top-k sampling replay and two training steps with checkpoint replay. Tied parameters require the explicit PyTorch swap-conversion recipe in the report. No new dependencies or native rebuilds; arbitrary pretrained models, FlashAttention, top-p and long-term training remain uncertified. Included in v0.2.0.

S40 adds the [standard GradScaler entry point](npunlock-probe/research/results/amp-standard/REPORT.md): after importing the backend, use `torch.amp.GradScaler("intel_npu")`, including aliases imported earlier. A backend-specific constructor adapter reuses CPU scaling controls and NPU gradient checks/unscaling. Iterable inputs, Tensor scale updates, overflow skipping, accumulation and state replay are covered. FP32 master parameters and explicit `foreach=False, fused=False` remain required. No new dependencies, native builds or ELFs; arbitrary-model AMP and long-term convergence remain uncertified. Included in v0.2.0.

S39 adds [attention boundaries and random APIs](npunlock-probe/research/results/attention-rng-boundaries/REPORT.md): SDPA handles empty batches/sequences/heads and zero QK/V features, including mask gradients and Dropout RNG consumption. Broadcast Tensor-probability Bernoulli and device-key-sort randperm support Generator/out and state replay. Exact dtype representation, workspace and existing sort bounds remain; large-scale sampling speed and general HF generation are not certified. No new dependencies, native builds or ELFs. Included in v0.2.0.

S36 expands [MaxPool and extrema capacity](npunlock-probe/research/results/pool-capacity/REPORT.md): ordinary MaxPool1d/2d use bounded packets beyond the old total sampled-cell limit, backward groups complete channel planes, and long-axis extrema restore global indices through device INT64 operations. Unused gradients for nondifferentiable index outputs are no longer materialized. Per-packet, per-plane and storage limits remained at that stage; the report separates measured cases from software bounds. No new dependencies or native rebuild. Included in v0.2.0.

S37 removes the [2²⁴ per-plane metadata limit in selection backward](npunlock-probe/research/results/selection-capacity/REPORT.md) using a new kernel with raw UINT32 headers. FP16/FP32 checks cover complete MaxPool forward/backward and extrema on a 16,777,219-element plane, plus accumulation of 16,798,727 gradients. The original 62 ELF kernels and small-input path are unchanged; the new ELF is 5,740 bytes. No dependency installation or DLL/PYD rebuild. Storage limits, other operators' forward limits and throughput concerns remain. Included in v0.2.0.

S38 improves [byte-based storage and dense planes](npunlock-probe/research/results/storage-shape/REPORT.md): all nine storage dtypes use the 1 GiB allocation budget by actual byte size. FP16/BF16/INT16 capacity doubles; BOOL/INT8 capacity quadruples. Dense MaxPool generates windows and indices on SHAVE and prunes backward scans by geometry. Complete CPU comparisons cover 4097×4097 forward/backward, plus every output of a 1 GiB Half negation. See the report for maximum-shape evidence and remaining limits. Rebuild the PYD and update kernel companions; no new runtime dependencies. Included in v0.2.0.

S35 improves [ordinary Conv2d workspaces](npunlock-probe/research/results/conv-capacity/REPORT.md): window/fold geometry selects bounded execution, and result tiles are written into one destination. A tested 3×3 convolution reduces backend tensor-allocation peaks from 7.00 to 1.84 MiB in FP32 and from 7.35 to 1.63 MiB in FP16; complete forward/backward and two training steps pass CPU comparisons. Small windows keep their existing path. Single-buffer and other convolution limits remain, with no new dependencies or native rebuild. Included in v0.2.0.

S34 improves [operator-specific capacity](npunlock-probe/research/results/index-capacity/REPORT.md): Embedding can partition feature columns and reuse existing NPU gather/gradient kernels beyond the old complete-index packet limits. Sort/topk input and output bounds increase to 16,777,216 elements while compute packets remain bounded. The report distinguishes tested vocabulary, index, sorting and training shapes from software limits. General indexed gradients, convolution and pooling retain separate constraints. Existing native components and kernels are reused without new dependencies. Included in v0.2.0.

| Path | Entry point | Current scope |
|---|---|---|
| PyTorch device backend | `tensor.to("intel_npu")`, `model.to("intel_npu")` | Supported FP16/FP32 operators, first-order gradients, and basic SGD; computation primarily uses custom SHAVE kernels |
| Frozen-model adapters | `QuantizedLinear`, `FrozenSmolLayer`, `FrozenSmolStack` | Explicitly constructed fused graphs with DPU matrix multiplication, for FP16 and selected low-bit inference experiments |

**Ordinary FP16 `torch.mm` currently computes through the FP32 SHAVE path and casts the result back to FP16. It does not automatically use the adapters' DPU FP16 matrix multiplication.**
Consequently, the model/DPU performance below does not apply to arbitrary models moved with `.to("intel_npu")`.
The frozen adapters do not provide LLM training.

S29 expands [loss compatibility](npunlock-probe/research/results/loss-extended/REPORT.md): weighted NLL/hard-label CE, differentiable soft-target and class-weight CE, broadcast and mixed-FP16/FP32 MSE, and native forward/backward/out interfaces. Independent CPU/NPU checks cover empty dimensions, special weights, gradients, and three-step combined training. Existing kernels are reused without new dependencies. Hard-label packet limits, exact preallocated outputs and first-order gradient boundaries remain. Included in v0.2.0.

S28 expands [SDPA and GQA support](npunlock-probe/research/results/sdpa-extended/REPORT.md): nonempty ranks 2–64, mixed-rank broadcasting, and GQA/MQA forward and shared K/V gradients. Attention-score validation now uses the current storage bound, with hardware checks beyond the previous one-million-element limit. Existing SHAVE/DMA kernels are reused without new dependencies. Limits remain: row width 65,536, causal query length 1,048,576, bounded allocations and graph cache. Included in v0.2.0.

S27 expands [matrix and model support](npunlock-probe/research/results/matmul-workspace/REPORT.md): long-K matrix multiplication uses rolling compensated accumulation instead of retaining every partial result; `matmul` accepts ranks 1–64 and bounded strided read views, and `bmm` handles empty batches and zero K. A model combining a 65,539-wide Linear layer with rank-5 matmul passes two independent CPU/NPU training steps in FP16 and FP32. Ordinary FP16 matmul still uses FP32 SHAVE computation. Storage, graph-cache, and operator-specific limits remain; these changes are included in v0.2.0.

S26 expands [large-tensor and model support](npunlock-probe/research/results/capacity-model/REPORT.md): up to 256 MiB per allocation, storage ranks 0–64, larger ordinary math/reduction/Linear workloads, and complete-row softmax tiling. The report covers independent CPU/NPU training of a 17.04M-parameter Linear model. Dtype element limits, operator workspaces, and the 1,536-graph cache remain bounded. Included in v0.2.0.

S25 adds [storage, indexing, and dtype compatibility](npunlock-probe/research/results/compat-foundation/REPORT.md): allocations up to 64 MiB, storage ranks 0–8, dynamic boolean/advanced indexing, writable slices, gather/scatter, common integer operations, selected BF16 operations, mixed-dtype concatenation, and additional math including sin/cos/pow. A 2.1M-parameter network has independent CPU/NPU forward, gradient, and SGD comparisons for two steps in FP16 and FP32. Operator workspace, graph-cache, and model limits remain; these changes are included in v0.2.0.

The default runtime does not require the OpenVINO Python/runtime packages. It still requires the Intel NPU system driver and its graph compiler.
The device is registered through PyTorch PrivateUse1 as `intel_npu:0`; it is separate from `torch.xpu`.

## Quick start

The validated platform is **Windows x64 / Core Ultra 7 258V / NPU `8086:643e`**,
with driver `32.0.100.5540`, graph compiler `8.3`, Python 3.12 (3.12.14 in this release check), Torch 2.14.1+cpu, and NumPy 2.5.3.
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

The local development version as of 2026-10-16 adds ordinary PyTorch migration checks:
FP16/FP32 training for an MLP, convolutional network and TinyResNet, SGD momentum/weight decay,
and explicit CPU checkpoint recovery. Use `foreach=False, fused=False` for the tested optimizer path.
Default BatchNorm counters, running statistics, recovery and the bounded classification losses below are now covered locally; see the bounded AdamW path below. Common Transformer layers still have known blockers.
Run `run_torch_npu.ps1 -Script .\torch_npu_migration_demo.py`; see the [migration validation report](npunlock-probe/research/results/torch-migration/REPORT.md).
These fixes are included in v0.2.0 and do not establish arbitrary model or CUDA runtime compatibility.

The subsequent [INT64 and default BatchNorm validation](npunlock-probe/research/results/int64-batchnorm/REPORT.md)
adds bounded INT64 storage (now rank 0–5 including empty tensors, up to 524,288 elements) and scalar counters.
SHAVE performs exact two-word integer addition with carry; the F32 carrier transports raw bits without numeric conversion.
Limited FP16/FP32 checks cover default BatchNorm running buffers, cumulative averaging, training, eval/reset and explicit CPU checkpoint recovery.

The subsequent [classification validation](npunlock-probe/research/results/classification/REPORT.md) adds FP16/FP32 forward and first-order backward for `log_softmax`, `NLLLoss` and `CrossEntropyLoss`.
Classification now accepts ranks 1–5, resident INT64 hard labels and optional class weights. CE additionally supports same-shape floating soft targets and gradients for soft targets and weights. Label smoothing in [0,1] and all reductions are supported; see S29 above for mean normalization and ignore-index rules.
Current `log_softmax` supports any valid axis for ranks 1–64, up to 67,108,864 total elements and 65,536 along the normalization axis. Hard-label NLL still requires `target_count*(C+4)<=65,536`.
Loss arithmetic and target validation run on the NPU; each nonempty hard-label forward synchronously reads a 4-byte error flag. There is no CPU tensor arithmetic fallback.
This also fixes the saved-statistics rounding boundary in pure FP16 BatchNorm. Six small-model/dtype combinations pass four-step independent CPU comparisons and CPU checkpoint recovery; arbitrary deep FP16 training remains unverified.
Standard usage (`logits` is the model's NPU output):

```python
labels = torch.tensor([0, 1], dtype=torch.int64, device="intel_npu")
loss = torch.nn.CrossEntropyLoss()(logits, labels)
loss.backward()
```

The local [AdamW validation](npunlock-probe/research/results/adamw/REPORT.md), dated 2026-10-18, adds ordinary single-tensor updates, moment states, AMSGrad, maximize and explicit CPU checkpoint recovery.
Use `foreach=False, fused=False, capturable=False, differentiable=False` and Python numeric hyperparameters. Parameters and moments compute on the NPU; step counters and bias correction retain PyTorch's CPU/Python control flow.
FP32 training is recommended for this path. Set `bias=False` explicitly for convolutions immediately followed by BatchNorm; the backend does not rewrite models.
Isolated FP16 updates pass, but some full-model independent CPU trajectories fail. Default `eps=1e-8` also rounds to zero in FP16, producing NaNs for zero gradients on both CPU and NPU.
FP16 tests explicitly use `eps=1e-4`; this still does not establish full-model trajectory parity. There is no automatic epsilon change or FP32 moment policy; these earlier tests did not use S17 AMP.

The local [LayerNorm and GELU validation](npunlock-probe/research/results/norm-activation/REPORT.md), dated 2026-10-18,
adds bounded FP16/FP32 forward and first-order backward for `nn.LayerNorm`/`F.layer_norm` and `nn.GELU`/`F.gelu`,
including both GELU formulas (`none` and `tanh`). LayerNorm reuses NPU FP32 reductions; GELU uses four custom SHAVE kernels.
FP16 inputs convert and compute on the device. A residual feed-forward network passes four-step independent CPU comparisons
with FP16/FP32 SGD and FP32 AdamW. The subsequent [mixed LayerNorm statistics fix](npunlock-probe/research/results/layernorm-mixed/REPORT.md)
passes the original failure replay and 22 expanded standalone checks for FP16 inputs with FP32 weight/optional bias, up to normalized width 65,536.
The subsequent [mixed-training update fix](npunlock-probe/research/results/mixed-training-rounding/REPORT.md)
corrects FP16 alpha conversion in Tensor `add_`. Both default and tanh GELU mixed networks with explicit FP32 CrossEntropy
pass four independent CPU-comparison steps with the original FP32 parameter-gradient thresholds.
The two FP16-loss trajectory checks still fail; generic mixed training and long-term convergence remain uncertified.
The subsequent [FP64 classification and 32-step checks](npunlock-probe/research/results/classification-precision/REPORT.md)
pass ten numerical checks. Both fixed four-sample networks learn, but strict independent CPU trajectories first differ
at steps 6 and 5; both 32-step checks remain strict failures. Four-step agreement does not establish long-term parity.
The subsequent [first-divergence and held-out checks](npunlock-probe/research/results/mixed-step-diagnosis/REPORT.md)
attribute these two failure points to fused versus separate SGD updates and half scalar-tail rounding. Local operators and
momentum match on identical inputs/state. On a synthetic 64-training/64-held-out task, both devices reach 75% held-out
accuracy with identical predictions; strict trajectory failures remain recorded.
This is included in v0.2.0; its reported results do not establish generic mixed training, full Transformer or AMP support.

The subsequent [softmax, bmm and tiny attention checks](npunlock-probe/research/results/attention-primitives/REPORT.md)
validate standard `torch.softmax`/`nn.Softmax` and three-dimensional `torch.bmm` forward/backward in FP16/FP32,
including no-gradient out variants and bounded tiling. S19 supports any valid positive/negative softmax axis for rank 1–4, up to 1,048,576 total elements and 65,536 along the normalization axis;
bmm requires equal batches and dtypes without broadcasting, within the global storage limit.
The composed `bmm → softmax → bmm` Q/K/V gradients and a tiny LayerNorm/Linear/residual attention network pass four
independent CPU/NPU SGD steps with explicit FP32 classification loss. Computation uses SHAVE FP32 and device casts.
This extends compatibility; batched GEMM performance remains future work. Subsequent matmul/SDPA coverage is described below.

The local [rank 2–4 matmul and SDPA checks](npunlock-probe/research/results/sdpa/REPORT.md) add batch-broadcasting `matmul`/`@`
with first-order gradients; `mm` remains two-dimensional. Standard `F.scaled_dot_product_attention` accepts same-rank 3D/4D
FP16/FP32 Q/K/V, zero dropout, and either causal or device-resident additive floating masks, including mask gradients and zero output for fully masked rows.
Differentiable clone/contiguous layout handling enables head merging. A two-head causal attention network with LayerNorm,
GELU and residuals passes four CPU-reference training steps with explicit FP32 loss.
S13 originally extended SDPA scores from 65,536 to 1,048,576 elements. S28 further expands storage validation and adds GQA; the S28 report linked above defines current rank, capacity, and training coverage.
Subsequent bounded Boolean-mask, Dropout and standard-module support are described below. Arbitrary full Transformers are not certified.
This is a SHAVE-based mathematical implementation using existing matrix operations; it is not FlashAttention or fused DPU attention and has not been performance-tuned.

The subsequent local [vector and scalar migration checks](npunlock-probe/research/results/vector-migration/REPORT.md)
extend `matmul`/`@` to ranks 1–4 and add bounded `dot`/`mv`, first-order gradients and exact-shape non-aliasing outputs.
FP16/FP32 `loss.item()` synchronously reads one scalar while preserving later backward calls.
The next local [bounded index and Embedding checks](npunlock-probe/research/results/embedding/REPORT.md) add device-resident INT64 indices, FP16/FP32 `index_select`/ordinary `nn.Embedding`, dense gradients, repeated-index accumulation and padding_idx through new SHAVE kernels.
An FP32 classifier containing Embedding passes four independent CPU-reference training steps. The initial packet is limited to 65,536 FP32 elements with a separate backward work limit; large vocabularies remain unsupported.
The subsequent [layout migration checks](npunlock-probe/research/results/layout-migration/REPORT.md) add bounded same-dtype FP16/FP32/INT64 `cat/stack`, read-only byte-DMA materialization of partial and zero-stride broadcast views, and first-order FP16/FP32 slice/select and split/unbind gradients.
Slices retain shared storage/version counters; use `clone()` for independent complete storage. Partial destination writes, empty tensors and mixed-dtype concatenation remain unsupported.
The subsequent [BOOL and mask checks](npunlock-probe/research/results/bool-mask/REPORT.md) add actual one-byte BOOL storage and casts, six comparisons, Boolean logic, three-argument `where`, `masked_fill` and first-order floating gradients. Boolean SDPA masks use True to keep an element.
An ordinary FP32 attention network with INT64 padding comparisons, Boolean causal/padding masks, SDPA and masked_fill passes four independent CPU-reference training steps. Comparisons preserve subnormal/NaN and INT64 word semantics; selection copies the chosen branch rather than multiplying by a mask.
The subsequent [RNG and Dropout checks](npunlock-probe/research/results/rng-dropout/REPORT.md) add on-SHAVE Philox random generation, FP16/FP32 `rand/randn`, seed/state replay, `fork_rng` and first-order ordinary `nn.Dropout`. A small FP32 network with Dropout passes CPU gradient/update comparisons with the same masks and bitwise replay of the next training step after checkpoint restore.
The subsequent [SDPA Dropout support](npunlock-probe/research/results/sdpa-dropout/REPORT.md) applies device-generated random masks to attention probabilities, with first-order FP16/FP32 Q/K/V and floating-mask gradients, preserving the existing broadcast, causal and fully masked row behavior. SDPA applies Dropout according to its probability argument; callers must explicitly pass `dropout_p=0` during evaluation.
Subsequent bounded Generator and discrete-sampling support is described under S16, and limited AMP under S17. Dynamic Boolean indexing, general AMP and asynchronous execution remain gaps. Bounded Boolean all/any support is described below. CUDA-specific APIs are not implemented; see the [migration priorities](COMPATIBILITY.md#cuda式模型迁移仍缺什么).

The local [standard MHA/EncoderLayer migration](npunlock-probe/research/results/standard-modules/REPORT.md) materializes read-only packed Q/K/V views with device DMA, adds baddbmm with first-order gradients, and decomposes the default inference entry points into existing NPU operators. Users need not modify standard modules or disable the global MHA switch. Ten hardware test groups pass, covering FP16/FP32 forward/backward, default inference and FP32 two-layer training with checkpoint replay. This extends bounded small-model compatibility; it does not add hardware-fused kernels or certify full Transformers.

The subsequent [dense TransformerEncoder checks](npunlock-probe/research/results/transformer-container/REPORT.md) pass two-layer FP16/FP32 forward/backward and dense inference, and add triu/tril, Boolean all/any and an FP16 native masked-softmax rounding kernel, connecting Embedding, learned positions, a two-layer encoder, padding-aware pooling, a classifier and CrossEntropy for FP32 training with SGD momentum and CPU-checkpoint replay in a fresh model/optimizer. Set `enable_nested_tensor=False` for padded inference. Automatic NestedTensor conversion, encoder-decoder Transformers and generic AMP remain unsupported. Performance and long-term convergence have not been measured.

The later [standard-model scaling checks](npunlock-probe/research/results/transformer-scale/REPORT.md) increase the same token classifier to 42,211 and 232,387 parameters. Both FP32 sizes and the larger FP16 model pass all parameter-gradient checks, SGD momentum updates and bitwise checkpoint replay in a fresh model/optimizer, with an explicitly FP32 classification loss. Embedding packets are now tiled by columns, reusing existing SHAVE kernels and device DMA while preserving repeated-token gradient accumulation order. A non-divisible 2048-by-35 case passes bitwise FP16/FP32 forward/backward comparisons; the larger model uses a 2048-by-64 embedding. General tensor and single-column index limits remain, with no additional dependencies or relaxed tolerances.

The new [complete-row attention checks](npunlock-probe/research/results/attention-rows/REPORT.md) extend softmax/log_softmax and SDPA score storage to 1,048,576 elements, with row/key width still at most 65,536. Causal packets carry the global query-row offset across head and tail boundaries; RNG semantics are preserved. A standard two-layer model exercises sequence96/batch4 and 147,456 scores; see the report for whole-model results by precision and measured memory. Compiled sessions have a bounded 1,536-entry limit and remain alive until shutdown; disk caching stays disabled by default. Full scores are still stored, so this is not FlashAttention or unrestricted context support.

The local [cast graph reuse change](npunlock-probe/research/results/cast-reuse/REPORT.md) shares FP16↔FP32 conversion graphs across equal element counts while preserving public shapes, IEEE conversion and device DMA compaction. This reduces cold compilation and cache pressure; it does not reduce submissions per conversion or add CUDA runtime or general DPU training support.

The subsequent [math and model migration work](npunlock-probe/research/results/unary-math/REPORT.md) exposes exp/log/tanh/sigmoid/SiLU/rsqrt, adds first-order sqrt gradients and supports standard RMSNorm. See the report for FP16/FP32 gated-network and bounded standard TransformerDecoder training, checkpoint and inference checks. The decoder reuses existing self/cross-attention operators; this does not add CUDA runtime or DPU training compatibility.

S16 adds [discrete RNG and sampling](npunlock-probe/research/results/discrete-rng/REPORT.md): INT64 randint/random_, FP16/FP32 multinomial and a native `torch.Generator(device="intel_npu")`, including independent streams, state/clone and sampled-token training/checkpoint checks. Values and sampling run on SHAVE; multinomial downloads only one 4-byte validation flag per packet. Output tensors must already have the exact shape, and sampling without replacement requires enough positive-weight categories. See the compatibility matrix for bounds. There are no new dependencies; updating an older local build requires rebuilding the Torch PYD.

S17 adds [limited AMP](npunlock-probe/research/results/amp/REPORT.md): standard `torch.autocast("intel_npu", dtype=torch.float16)` and the NPU-specific `torch.intel_npu.amp.GradScaler()`. Keep model parameters in FP32; matrix/conv operations use FP16 while losses and normalization use FP32. Gradient checks and unscaling run on NPU, downloading one Boolean flag per optimizer; the host maintains only scale/growth control state. Overflow skipping and checkpoints are covered. MLP passes strict training checks; convolution still has a strict FP32 gradient-comparison limitation documented in the report. S40 subsequently adds the standard `torch.amp.GradScaler("intel_npu")` entry point; BF16 AMP, arbitrary model migration and full-model low-bit quality remain incomplete; bounded dynamic indexing and randperm were subsequently added in S25/S39. There are no new dependencies or SHAVE ELFs; rebuild the PYD when updating. No training speedup is claimed.

S18 adds [AMP convolution diagnosis and broader checks](npunlock-probe/research/results/amp-conv/REPORT.md). With the original S17 initialization, inputs and labels, CPU Half weight-gradient differences are explained by per-batch rounding; existing NPU FP32 accumulation and the original strict failures are retained. Six local geometries and four-step SiLU/three-convolution residual training pass against an independent FP32 accumulation control, with bitwise checkpoint replay. Empty GradScaler checkpoints now raise the intended exception. CPU controls are used only for validation. This update needs no kernel/host rebuild, adds no dependencies and makes no performance claim.

S19 expands [common model migration support](npunlock-probe/research/results/portability/REPORT.md): FP16/FP32 grouped/depthwise Conv2d, Conv1d, softmax/log_softmax along any valid axis, spatial/sequence hard-label NLL/CE, label smoothing and rsub with alpha. Existing NPU kernels and DMA are reused; convolution groups execute serially. No OpenVINO dependency, new SHAVE ELF or host rebuild is added. The report documents bounded segmentation/standard Transformer training, AMP, checkpoint checks and numerical tolerances. These changes are included in v0.2.0 and do not provide CUDA runtime compatibility.

S24 expands [vision and sorting compatibility](npunlock-probe/research/results/vision-sort-extended/REPORT.md): rank-five and empty tensor foundations; stable INT64/BOOL sorting and k=0; kthvalue, median/nanmedian, mode and FP32 quantiles; Conv3d/ConvTranspose3d, volume interpolation/normalization, antialiasing, 3D/adaptive pooling and unpooling, padding, shuffles, Unfold/Fold, affine_grid and grid_sample with first-order derivatives. The report defines the tested scope and limits; this does not establish every visual/sorting configuration or general CUDA compatibility. Rebuild the device PYD when updating an old local installation. No new pip dependencies were added; computation remains on the NPU.

S23 adds [vision and floating sorting](npunlock-probe/research/results/vision-sort/REPORT.md): 1D/2D nearest, nearest-exact, linear, bilinear and bicubic interpolation; GroupNorm/InstanceNorm; grouped ConvTranspose1d/2d; stable sort/argsort; max/min values with indices; and larger-k topk. Bounded first-order gradients, in-place activation combinations and CPU comparisons are included. The source-backed sort ELF adds no dependencies or C++ rebuild. Nearly tied activations can reorder and amplify model-gradient differences: the report preserves strict-trajectory failures separately from fixed-route diagnostics. S24 expands these formerly missing families. Training rejects a few explicit-scale configurations where the installed CPU forward/backward shortcuts disagree; use size or recompute_scale_factor=True. This is not general CUDA support.

S22 adds [MaxPool, in-place ReLU and topk](npunlock-probe/research/results/pool-topk/REPORT.md): standard MaxPool1d/2d with spatial INT64 indices and overlapping gradients; `nn.ReLU(inplace=True)` training on non-leaf results without view aliases; bounded FP16/FP32 topk with value gradients, device indices, long-row chunks, largest/smallest and paired outputs. Three source-backed kernels require no new dependencies or C++ rebuild. S23 expands the original k<=128/work subset; S24 adds k=0/empty results; capacity limits remain. Equal-valued topk indices need not follow CPU ordering, and partial writes/differentiable views remain limited.

S21 adds [extreme reductions, vector norms and gradient clipping](npunlock-probe/research/results/reduction-norm/REPORT.md): argmax/argmin with device INT64 output, differentiable amax/amin, vector norms of order 0/1/2/+inf/-inf, and standard `torch.nn.utils.clip_grad_norm_(..., foreach=False)`. It also adds floating isnan/isinf/isfinite and same-dtype scalar device factors for in-place multiplication. Row and long-row chunks merge original indices on the NPU, preserving the first tie/NaN position. Existing capacity and FP16/FP32 limits remain; later topk support is covered by S22, while sort, arbitrary norm orders, matrix norms and general CUDA support remain gaps. No new dependencies are required.

S20 adds [bounded indexing, average pooling and clipping](npunlock-probe/research/results/index-pool/REPORT.md): gather/scatter_add/take_along_dim, AvgPool1d/2d, AdaptiveAvgPool1d/2d with up to 64 spatial bins, differentiable minimum/maximum/clamp, Hardtanh/ReLU6 and clip_grad_value_. SHAVE validates/normalizes INT64 indices and initializes scatter accumulation from self; each indexed row reads a four-byte status flag. Pooling reuses FP32 operations; variable divisor grids contain geometry metadata only. Source-backed ELF/manifest updates require no new Python dependencies or C++ rebuild. MaxPool, larger indexing and general CUDA ecosystem support remain gaps; no speedup is claimed.

The subsequent [sustained training and power measurements](npunlock-probe/research/results/sustained-training/REPORT.md)
scale this to four blocks and 793,604 parameters: 182.2 seconds of FP32 training, six updates and 91.7% average NPU engine activity.
A separate one-minute run reports 0.525 W for the Intel PMT NPU domain and 9.82 W for the package; Windows shared memory is about 43 MiB.
This fixed synthetic classification workload exercises SHAVE training; it does not establish peak DPU throughput, full language-model training or better energy efficiency than CPU/GPU.

A separate [matched-work DPU/CPU/GPU energy benchmark](npunlock-probe/research/results/gemm-energy/REPORT.md) reaches
10.39 TFLOPS FP16 and 14.38 TOPS INT8 with six-tile, four-output DPU graphs, below the 258V's 47 TOPS rating.
Across two rounds, package energy per 1024-cubed GEMM is 121.64 mJ for the lowest-energy tested CPU path,
2.58 mJ for GPU FP16 and 5.44 mJ for DPU FP16; the GPU is faster and uses less energy per GEMM here.
The FP16 DPU workload reports 15.16 W for the NPU domain. Background activity was not locked, and these operator results do not establish full-model/training efficiency or peak hardware throughput.

The local development version adds `FrozenSmolIO` for the explicit CPU endpoints of SmolLM NPU inference.
It loads only the embedding, final RMSNorm and tied output matrix, without constructing a CPU decoder.
Example (`ids` contains caller-supplied CPU int64 token IDs):

```python
from torch_npu_smollm import FrozenSmolStack
from torch_npu_smollm_io import FrozenSmolIO

io = FrozenSmolIO.from_checkpoint(model_directory)
stack = FrozenSmolStack.from_checkpoint(model_directory, num_layers=30, mode='fp16')
with torch.no_grad():
    hidden, kv = stack.prefill(io.embed_tokens(ids).to('intel_npu'))
    logits = io.logits(hidden.cpu())[:, -1].float()
```

Import `torch_intel_npu` first to register the device. This endpoint is limited to the verified tied SmolLM FP16 inference path;
it does not replace CPU reference models or support training. The module needs only Torch and safetensors;
tokenization is supplied separately. This interface is included in v0.2.0.

## Current capabilities

| Feature | Implemented and validated scope |
|---|---|
| NPU tensors | FP16/FP32 storage, CPU/NPU copies, selected dense transposes and layout handling; Level Zero shared allocation |
| Operators and training | Add/subtract/multiply/divide, selected unary operations and reductions, matrix multiplication, Linear, Conv2d, BatchNorm, ReLU, pooling and MSE; local bounded LayerNorm/GELU, softmax, bmm, higher-rank matmul, SDPA, clone layout gradients, log_softmax/NLL/CrossEntropy and AdamW; first-order autograd for limited parameter combinations |
| True FP32 | FP32 arithmetic, compensated reductions, and matrix multiplication on SHAVE; this path is not replaced with FP16/TF32, but GEMM performance remains limited |
| FP16 model inference | The complete 30-layer decoder of SmolLM2-135M-Instruct, prefill, KV cache, and decode; decoder/KV run on the NPU, while embedding, final norm, lm_head, and sampling run on the CPU |
| Low-bit inference | Explicit `QuantizedLinear` experiments with INT8, INT4, NF4 weights and grouping; W8A16/W8A8 graph paths. INT4/NF4 support does not imply native 4-bit multiply-add or general low-bit device tensors |
| Runtime optimizations | Graph fusion, native session reuse, chained submission, buffer reuse, and optional disk graph caching; early FP16/W8/mixed prefill hits and local lightweight CPU endpoints |
| Installation baseline | Source-built host components, offline tests and real-NPU operation/training/model checks; see [release validation](RELEASE_READINESS.md) for version-specific coverage |

The [operator/parameter/gradient compatibility matrix](COMPATIBILITY.md) and its corresponding tests define the supported contracts.
A module name in the table does not imply support for every parameter combination.
See [INSTALLATION_REPORT.md](INSTALLATION_REPORT.md) for installation validation and [TORCH_NPU_README.md](TORCH_NPU_README.md) for historical operator/training details.

Stage reports below retain their original experimental conditions; the current contract is [COMPATIBILITY.md](COMPATIBILITY.md).

### Known limits

- Current S38 storage supports ranks 0–64 and allocations up to 1 GiB. Limits use actual dtype bytes: up to 536,870,912 FP16/BF16 elements, 268,435,456 FP32/INT32 elements, 134,217,728 INT64 elements and 1,073,741,824 one-byte elements. These storage limits do not certify every operator or a 14 GiB training workload. Operator-specific workspaces remain bounded; Smol graph pressure follows the S45 family-retirement policy.
- The SmolLM adapter has a fixed 64-position KV capacity and a maximum batch size of 8, subject to shape/workspace constraints. It does not directly support arbitrary Hugging Face models or long contexts.
- Deep FP16 training retains numerical limits. AMP includes the standard torch.amp.GradScaler("intel_npu") entry point for tested configurations with FP32 master parameters; second-order gradients, differentiable CPU/NPU transfers, broad optimizer coverage, arbitrary differentiable in-place operations, and overlapping destination writes remain incomplete.
- Common INT8/UINT8/INT16/INT32/INT64 arithmetic and conversions are implemented. BF16 currently covers storage, conversion, and selected operations using FP32 intermediates. Complete dtype/operator coverage, the full RNG/sampling API, asynchronous streams, and GPU+NPU cooperation remain incomplete. Unsupported operations raise errors.
- Full-model W8 calibration has not passed every existing quality threshold. INT4/NF4 remain primarily Linear/small-network experiments; full low-bit LLM adaptation is not established. Current model evaluations explicitly prioritize FP16.
- Shared physical memory across CPU/GPU/NPU does not imply zero transfers or zero synchronization. It does not eliminate driver-internal DMA or precision/layout conversions.

The [full-model low-bit quality study](npunlock-probe/research/results/w8-sequential-calibration/REPORT.md)
dated 2026-10-15 compares sequential calibration and grouped INT8. Neither is ready for adoption;
the current compiler expands grouped graphs to FP16. Production defaults remain unchanged.

A subsequent [local mixed-precision quality experiment](npunlock-probe/research/results/w8-projection-mixed/REPORT.md)
adds an opt-in `fp16_projections` argument: keep MLP Up/Down in FP16 and use calibrated W8 for the other five projection types.
On the real NPU, passing positions improve from 118/142 to 129/142, and mean relative logits L2 drops from 2.238% to 1.471%.
Constants across 30 decode graphs use about 24% fewer bytes than full FP16. Quality still falls short of the thresholds at some positions;
subsequent speed measurements are listed below. This argument is included in v0.2.0.

The subsequent [logit diagnostics and layer-sensitivity study](npunlock-probe/research/results/w8-layer-sensitivity/REPORT.md)
finds that a common logit offset accounts for about 92.3% of squared error on average across the 13 failing positions.
Original quality thresholds are retained. A layer-specific candidate protecting 37.5% of projection elements passes 126/142 positions on the real NPU,
below the original policy's 129/142, and has not been promoted to a public option or default.

The [short-context PPL and free-generation pilot](npunlock-probe/research/results/w8-ppl-generation/REPORT.md) now evaluates 2,048 fixed WikiText-2 target tokens with 64 input positions.
Real-NPU PPL is 27.937/28.040/27.938/27.928 for FP16/full W8/the original 50% protection policy/the layer-specific 37.5% policy.
The two mixed policies generate identical completions on eight new prompts. This small sample does not certify general model quality; original logit thresholds and production defaults remain unchanged.

## Measured performance

These are measurements from the individual reports, **not chip peaks or a general model ranking**.
Each table retains its own date, execution path, and timing scope.

### CPU model and weight preparation: further startup reduction

2026-10-16: lightweight CPU endpoints and single-pass construction for the 50% mixed policy.
Identical cached graphs, all 30 layers, batch 1, a 24-token input, and 3 pairs of fresh processes per policy:

| Policy | Previous script-entry-to-first-token, s | New entry-to-first-token, s |
|---|---:|---:|
| FP16 | 5.707 | 5.326 |
| Calibrated W8 | 6.169 | 5.702 |
| Mixed, 50% FP16 | 5.917 | 5.280 |
| Layer-specific, 37.5% FP16 | 6.515 | 6.127 |

Startup-stage peak private commit decreases by about 169–202 MiB. Outputs and KV state are bit-identical
across 12 complete 32-token generation runs. This optimizes CPU preparation; it does not establish higher warm decode throughput.
Use the lightweight endpoint explicitly; full CPU reference scripts retain their reference models.
See the [startup preparation report](npunlock-probe/research/results/smollm-startup-prepare/REPORT.md) for stage timings,
memory scope and reproduction.

### W8 and mixed precision: startup with an existing cache

2026-10-16: FP16's early prefill-cache lookup now also supports calibrated W8 and explicit mixed precision.
All 30 layers, batch 1, a 24-token input, identical compiled blobs, and 3 alternating pairs of fresh processes per policy:

| Policy | Full-IR first token, s | Early lookup first token, s | Time reduction |
|---|---:|---:|---:|
| Calibrated W8 | 1.525 | 0.723 | 52.60% |
| Mixed, 50% FP16 | 1.701 | 0.824 | 51.55% |
| Layer-specific, 37.5% FP16 | 1.660 | 0.797 | 51.97% |

This table starts after model preparation. Script-entry-to-first-token latency changes from
6.215/6.641/6.581 seconds to 5.410/5.780/5.732 seconds respectively.
Outputs and KV state across all 30 layers remain bit-identical. These gains require an existing disk cache;
caching remains opt-in, and source-fingerprint changes require rebuilding old entries once.
This does not establish higher warm decode throughput or certify low-bit model quality.
See the [W8 early-cache report](npunlock-probe/research/results/w8-prefill-cache/REPORT.md) for implementation,
invalidation/corruption checks and boundary tests. This change has not been pushed to the public repository.

### Calibrated W8 and mixed precision: one request

2026-10-16, all 30 SmolLM2-135M decoder layers, batch 1, a fixed 24-token input and 32 prediction positions.
All four policies consume the same reference token trajectory, with 2 warmups and the median of 7 alternating rounds.
CPU stages and synchronization are included. This is not a free-generation quality test.

| Policy | Warm first token, ms | Decode tokens/s | Process lifetime peak private commit, GiB |
|---|---:|---:|---:|
| FP16 | 22.794 | 72.717 | 3.078 |
| Calibrated W8 | 22.651 | 74.987 | 2.188 |
| Mixed, 50% FP16 | 22.977 | 72.975 | 2.685 |
| Layer-specific, 37.5% FP16 | 22.944 | 73.859 | 2.561 |

W8 decode throughput is about 3.12% higher in this run; mixed-policy speed differences are small. The main benefit is lower memory usage.
Peak commit includes loading, compilation, execution and validation. The runner retains the full CPU model and original FP16 weights;
this is not dedicated NPU memory. With an existing graph cache, first-token latency after model preparation is
0.942/1.516/1.621/1.663 seconds respectively (median of 2 fresh processes per policy).
That run predates the cache optimization above, when low-bit policies still rebuilt prefill IR.
Disk caching remains opt-in, and temporary caches were removed.
See the [end-to-end efficiency report](npunlock-probe/research/results/w8-efficiency/REPORT.md) for stage timings, variability,
quality limitations and reproduction. Production defaults are unchanged.

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
| Near term | Broader release validation | Source-build and hardware checks are in [release validation](RELEASE_READINESS.md); add long-running stress regressions and installation validation on another clean Windows machine |
| Near term | First-token and runtime overhead | Profile repeated ELF/DPU parsing on cache hits, library imports, and repeated model preparation; preserve validation and use alternating A/B runs with output checks |
| Near term | Model coverage and quality | Expand prompt, batch, KV-boundary, and numerical tests before extending contexts/models; W8/INT4/NF4 must each pass quality thresholds |
| Medium term | Real-model hotspots | Remeasure RMSNorm/SHAVE instruction and register overhead, improve buffer reuse at larger batches, and verify each change's end-to-end benefit |
| Medium term | General operators and training | Add dispatch and gradients for operators missing from real models, improve general FP32 GEMM, and expand S17 AMP model/optimizer checks while retaining numerical boundaries |
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

The source snapshot is about 15.5 MiB, with per-file sizes and SHA-256 records in `SOURCE_MANIFEST.json`.
Models, environments, SDKs/toolchains, host binaries, and driver-generated `.blob` files are excluded; full fixture regressions need separately prepared graph data.
Historical raw arrays, graph caches, and timing JSON were removed under the storage policy. Reports retain measurement summaries; rerunning experiments requires regenerating inputs/outputs.

<details>
<summary>Offline tests and source-snapshot verification</summary>

After installing the core dependencies, run the offline regression suite without NPU hardware. Tests requiring absent optional model dependencies are explicitly skipped; see [release validation](RELEASE_READINESS.md) for counts:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
python -B -m unittest discover -s . -p 'test_torch_npu*.py'
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
Historical research audits may depend on deleted data and are not included in this offline suite.

</details>

## License and references

New project code is licensed under Apache-2.0, matching upstream npunlock. Third-party adaptations retain their respective license and copyright notices.
**npunlock is the direct foundation.** Selected public OpenVINO implementations and Intel NPU compiler/ELF/Level Zero sources are also explicitly attributed.
See [REFERENCES.md](REFERENCES.md), [references.bib](references.bib), [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [LICENSE](LICENSE).
When citing this project, record the actual version/commit and acknowledge the upstream work used. A bibliography does not replace license notices.
