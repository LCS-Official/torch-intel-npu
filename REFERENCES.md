# References / 实现来源

这里同时记录直接代码来源、接口与设计参考、验证工具。需要正式 reference 时可从
[references.bib](references.bib) 选取实际采用的条目；许可声明仍需保留 `NOTICE` 与 `LICENSES/`。

## 核心来源

| 项目 | 使用关系及对应实现 | 固定来源 |
|---|---|---|
| npunlock | 直接修改的基础：Python symbolic 图、IR 编译、worker、ELF 操作和 Level Zero 会话；并扩展 NPU4000 和 Torch 集成 | [hsfzxjy/npunlock, bdf1c01](https://github.com/hsfzxjy/npunlock/tree/bdf1c01d2caa2e772315a9b2a529981c2810e037) |
| Intel NPU compiler | NPU40XX firmware / SHAVE ABI、任务描述、构建与编译配置；核对后实现本项目的校验、定位与替换 | [openvinotoolkit/npu_compiler, 0b38f7d](https://github.com/openvinotoolkit/npu_compiler/tree/0b38f7d42113ff329ac2bdd33583d123de4ccf2f) |
| Intel NPU compiler ELF | ELF section 类型、CMX workspace、reader 与 relocation 语义；用于本项目解析器边界检查 | [npu_compiler_elf, d325f45](https://github.com/openvinotoolkit/npu_compiler_elf/tree/d325f45f2cb405b5fa2ff17a30de9469f1641b73) |
| Intel Linux NPU driver | 图加载与扩展语义调研；不代表本后端已支持 Linux | [intel/linux-npu-driver, aea583d](https://github.com/intel/linux-npu-driver/tree/aea583dcd86a4723a8d3bb06ad94b6eb2f80c6f2) |
| Level Zero / Intel NPU extensions | Windows 原生会话使用的 loader/API/ABI，command list / fence / graph / shared allocation 的生命周期 | [Level Zero v1.17](https://github.com/oneapi-src/level-zero/tree/5604d3963dec3c98d4e56ccf6fef0671a47114a2)、[NPU graph ABI 1.20 文件](https://github.com/intel/level-zero-npu-extensions/blob/ddbf8d65e2e22150e23efd383839aa80f84d4dac/ze_graph_ext.h) |
| OpenVINO / bitsandbytes NF4 | 直接使用公开的 16 项 NF4 码表；OpenVINO 文件说明了 bitsandbytes 来源。不是移植完整 runtime 或 CUDA 内核 | [OpenVINO nf4.cpp](https://github.com/openvinotoolkit/openvino/blob/8fb839633478574cefd28d49e25747d7c830b5a6/src/core/src/type/nf4.cpp)、[bitsandbytes 原始内核](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/c82f51c0f784d8a43ebcb9cdefbf94e3f3b9c6c3/csrc/kernels.cu) |
| GPTQ | 直接改编 W8 激活感知校准/误差反馈，CPU FP64 与固定 FP16 scale，见 `calibrate_int8` | [IST-DASLab/gptq, 2d65066](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py) |
| FP16 | 直接改编 IEEE FP16 转换算法为 SHAVE SIMD，见 `rmsnorm.c` | [Maratyszcza/FP16, 782eea1](https://github.com/Maratyszcza/FP16/blob/782eea126dc5c755827be751a099eb01826175cf/include/fp16/fp16.h) |
| PyTorch | PrivateUse1、dispatcher、autograd、Tensor/allocator/C++ extension 接口；CPU 算术作为数值参照。本机 NPU 环境 Torch 2.14.1+cpu | [PyTorch 源码](https://github.com/pytorch/pytorch)、[PrivateUse1 文档](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) |

NPU compiler / ELF / driver 调研的 **37 个具体文件、URL 与 SHA-256** 保存在
`npunlock-probe/research/sources/manifest.json`。本次导出不带这些第三方完整源码副本或预编译 Intel kernel。

## 性能优化调研与验证参考

- [OpenVINO NPU plugin](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_npu)：提交开销与 DPU 路线对比，见 [OpenVINO 对照研究](npunlock-probe/research/INTEL_NPU_LIBRARY_COMPARISON.md) 及实验报告。普通执行不依赖 OpenVINO runtime。
- [llama.cpp SYCL norm](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-sycl/norm.cpp)、[PyTorch CPU moments_utils.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/moments_utils.h)：归约组织调研。有序/分组补偿归约候选未获性能收益，最终采用 FP16 SIMD 转换，见 [RMSNorm 报告](npunlock-probe/research/results/rmsnorm-reduce/REPORT.md)。不要写成照搬 GPU 并行内核。
- [llama.cpp ggml-quants.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)：`w8_scale_candidate.py` 中固定码字的最小二乘 scale 思路；研究候选，不是默认量化规则。
- [Intel NPU Acceleration Library](https://github.com/intel/intel-npu-acceleration-library)：现成方案能力对比；不是本项目调用的运行库。
- [Hugging Face Transformers](https://github.com/huggingface/transformers)、[SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)：真实模型、tokenizer、CPU/GPU 数值/性能参照；模型权重不随源码发布。

上述部分链接是未固定的 `master/main`：现存记录不足以恢复当时检查的准确提交，因此只作导航。
不把本轮查询到的 HEAD 冒充为历史设计来源。若后续直接移植其中具体代码，应补固定版本、文件哈希与相应许可。

## 对外描述建议

> An experimental PyTorch PrivateUse1 backend for Intel NPU4000, built on a modified npunlock.
> It uses Intel's driver graph compiler and custom SHAVE kernels, with ABI references from the
> open Intel NPU compiler/ELF ecosystem. Selected routines adapt GPTQ and FP16; the NF4 codebook
> follows OpenVINO/bitsandbytes. OpenVINO runtime is not required for the default execution path.

不能据此描述为“完全独立于 Intel 图编译器”“官方 PyTorch NPU 支持”或“已把 NPU 接入 torch.xpu”。
