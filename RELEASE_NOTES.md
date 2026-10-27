# v0.2.0 Experimental Preview

This release brings the accumulated Intel NPU4000 backend work since v0.1.0 into one source release. It remains an experimental Windows/Lunar Lake backend, with explicit compatibility and numerical limits.

## Changes

- Broader PyTorch device compatibility: integer/Boolean arithmetic, indexing, reductions, sorting, losses, random sampling, standard modules and optimizers within the documented contracts.
- Rank 0–64 storage and allocations up to 1 GiB, with dtype-aware limits and bounded operator workspaces. These are storage capabilities, not certification of arbitrary 14 GiB model training.
- Standard `torch.amp.GradScaler("intel_npu")` integration for tested FP32-master-parameter training configurations.
- Additional attention and model migration support, including bounded GQA/empty-dimension behavior, experimental blockwise attention and strided NestedTensor inference. Default attention remains the existing math path.
- Complete 30-layer SmolLM2-135M frozen decoder inference on NPU, KV caching and startup/runtime improvements, plus optional calibrated W8/mixed-projection/low-rank experiments. Embeddings, final norm/head and sampling remain explicit CPU stages.
- Residual-factor prefill cache identity now includes factor values and rank. Smol cache pressure retires the graph family together; unsafe grouped-plan retirement is rejected before execution. Default graph cache remains 1536 entries/512 MiB, with explicit byte-budget control.
- Expanded source tests, automatic offline test discovery, release manifests and provenance. English is now the default README, with a Chinese overview and language-switching links. The previous English URL remains as a redirect page.

## Validation and upgrade

The supported reference platform is Windows x64, Core Ultra 7 258V, NPU4000 `8086:643e`, driver 32.0.100.5540, compiler 8.3, Python 3.12 and Torch 2.14.1+cpu. See [release validation](RELEASE_READINESS.md) for exact source-build, offline and real-NPU results, including skipped tests and CI status.

**Upgrading from v0.1.0 requires rebuilding host components with `build_npu.ps1`.** Do not mix the new Python source with old DLL/PYD files. SHAVE source/ELF/assembly provenance is included; model weights, environments, tools, host binaries and driver-generated graph blobs are excluded. There is no wheel release. See [INSTALL.md](INSTALL.md).

## Known limitations

- This provides an `intel_npu` PyTorch device; it is not a CUDA runtime/PTX replacement and is not merged into `torch.xpu`.
- Ordinary FP16 matmul still uses the generic SHAVE path; frozen DPU adapters are a separate execution route. Reported kernel/model timings are not chip peak throughput or general training speedups.
- FP16/AMP training remains bounded by tested models/configurations. Higher-order gradients, arbitrary in-place/view semantics, broad optimizer modes, serialization/compile integration and long-term training are incomplete.
- W8 correction retains 16 historical quality failures. New calibration candidates did not beat the existing internal selection. INT4/NF4 whole-model quality and generic low-bit training are not certified.
- The Smol adapter retains a 64-position KV capacity. Blockwise attention/NestedTensor support is experimental and does not cover arbitrary masks, dropout, layouts or training configurations.
- Graph-family retirement is a tested workaround with extra compilation cost. The underlying partial-teardown fault has not been conclusively attributed to the Windows driver or native session layer.
- Effective asynchronous streams/events, multi-device/distributed execution and GPU+NPU collaboration are not implemented.

Full contracts and per-stage evidence are in [COMPATIBILITY.md](COMPATIBILITY.md) and the linked reports. Sources and adapted algorithms are attributed in [REFERENCES.md](REFERENCES.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), NOTICE and LICENSES.

---

本版集中发布v0.1.0以来的兼容性、容量、训练接口、Attention、模型与缓存改进，仍为实验预览版。从旧版升级需重新构建宿主组件。低位16处质量失败、原生生命周期根因、异步与GPU＋NPU协同等问题仍保留，不宣称通用CUDA兼容、完整低位整网或训练加速。详细验收、云端CI状态和限制见上述链接。
