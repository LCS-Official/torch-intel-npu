# v0.3.0-alpha.1 Experimental Preview

This preview combines the development work since v0.2.0 into one mainline source snapshot. It adds a classification-loss correctness fix and explicit four-bit SmolLM2 decoder execution, together with the numerical studies and reports. It remains an experimental Windows/Lunar Lake backend.

## Changes

- Classification-loss autograd now preserves the tested source-version behavior after layout copies. Mutating saved original labels or NLL inputs is rejected at backward where the validated CPU implementation rejects it. This does not provide general view-write or higher-order-gradient support.
- `FrozenSmolLayer` and `FrozenSmolStack` accept `mode="w4a16"` and `mode="nf4a16"`, with group-32 weight preparation or matching explicit packed weights. FP16 activations, outputs and KV remain unchanged.
- DPU evidence validation now follows the weight format. W8 still requires INT8 matrix evidence; four-bit decompression requires FP16 matrix evidence. NF4 single-row projections pad internally to 32 rows and trim to avoid a driver 8.3 compilation failure.
- S47–S59 retain layerwise, attention, activation, weight-decoding and calibration diagnostics. None of the new calibration candidates is promoted. The repeated last-layer fitting branch is closed in favor of actual model coverage and compatibility.
- The cumulative snapshot includes all previously separate development branches, refreshed source hashes, an English default README and Chinese language button. Reports remain; raw experiment arrays and logs are excluded.

## Execution and quality are separate

The 30-layer SmolLM2-135M decoder passes bounded forward, prefill and decode checks for each format. Packed four-bit weights are decoded for FP16 DPU computation; this is not native four-bit multiply-add or a general low-bit PyTorch Tensor backend.

The same eight-prompt S60 corpus scores 257 EOS-bounded positions against the original-weight FP16 CPU model:

| Mode | Original-model quality passes | Matching next token |
|---|---:|---:|
| FP16 | 257/257 | 257/257 |
| Plain per-row RTN W8 | 96/257 | 247/257 |
| Group-32 INT4 | 0/257 | 188/257 |
| Group-32 NF4 | 0/257 | 214/257 |

The gates remain raw-logit relative L2 < 0.03 and KL < 0.02. This W8 control has no optional calibrated factors and must not be compared directly with the older 142/290-position studies. Four-bit quality is not certified. Keep FP16 for reference-model quality.

NF4's full-stack warm check uses an explicit 1 GiB graph-cache budget; 512 MiB causes LRU eviction and recompilation. The production default remains 512 MiB. Packed source constants are smaller, but the compiler may expand them; no process-memory, speed, energy or chip-peak utilization advantage is claimed.

## Validation and upgrade

The reference platform remains Windows x64, Core Ultra 7 258V, NPU4000 `8086:643e`, driver 32.0.100.5540, compiler 8.3, Python 3.12, Torch 2.14.1+cpu. See [release validation](RELEASE_READINESS.md) for the cumulative checkout's offline and real-NPU checks and the reused-host-component boundary.

From v0.2.0, update the complete source snapshot and restart Python. Native sources, kernel files, requirements and host ABI are unchanged; matching v0.2.0 host components can be reused. First installation and upgrades from v0.1.0 still require the build procedure in [INSTALL.md](INSTALL.md). No wheel or model weights are distributed.

This release preserves the earlier tags and commits. The GitHub source archives contain source, licensed source-backed kernel companions, provenance and Markdown reports; they exclude environments, toolchains, host binaries and driver-generated graph data. Run `python -B tools/export_source_release.py --verify .` before creating local build outputs.

## Remaining limits

- This is an `intel_npu` device, not a CUDA runtime/PTX replacement or an addition to `torch.xpu`.
- Four-bit frozen inference does not provide four-bit training, QLoRA, arbitrary checkpoint import or general Hugging Face migration. New modes are validated only at the reported shapes and KV lengths; their free generation, PPL and grouped decode are not certified.
- Existing W8 calibration still has quality failures. Historical 16-position failures and the original model-quality limits are not removed by this release.
- General autograd/layout mutation, higher-order gradients, long-term AMP training, broad optimizer modes and serialization/compile integration remain incomplete.
- Smol KV capacity remains 64 positions. Blockwise attention/NestedTensor support is experimental.
- The native partial-teardown root cause remains unresolved; graph-family retirement and cache budgets retain their documented costs and limits.
- Asynchronous streams/events, distributed execution, GPU+NPU collaboration and arbitrary 14 GiB training remain uncertified or unimplemented.

Full contracts are in [COMPATIBILITY.md](COMPATIBILITY.md). Numerical studies: [S46 loss fix](npunlock-probe/research/results/loss-source-versions/REPORT.md), [S59 closeout](npunlock-probe/research/results/w8-native-down-check/REPORT.md), and [S60 four-bit execution](npunlock-probe/research/results/smollm-lowbit-coverage/REPORT.md). Attribution remains in [REFERENCES.md](REFERENCES.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), NOTICE and LICENSES.

---

本预览集中纳入v0.2.0之后的损失版本检查修复、四位解码器入口和研究报告，保留原提交历史。四位执行通过不等于整网质量通过；INT4/NF4本轮均0/257，FP16为257/257。已有匹配的v0.2.0宿主组件可复用，首次安装／v0.1升级仍须构建。尚不承诺通用CUDA兼容、低位训练、训练加速或GPU＋NPU协同。
