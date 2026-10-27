# 第三方来源与许可说明

本说明保留2026-10-11源码整理快照的来源，并包含2026-10-18的本地增补。主许可证为 Apache-2.0，见 `LICENSE`。
主许可证不替换以下来源的版权和许可证。引用文献与代码再分发声明是两件事，均需保留。

## 直接保留或改编的内容

| 来源 | 本项目位置 / 修改 | 许可文件 |
|---|---|---|
| [npunlock](https://github.com/hsfzxjy/npunlock/tree/bdf1c01d2caa2e772315a9b2a529981c2810e037) | `npunlock/` 的完整源码基础；NPU4000、symbolic IR、共享分配、session 及测试的本地扩展见 `UPSTREAM.md` | `npunlock/LICENSE`，Apache-2.0 |
| [OpenVINO nf4.cpp](https://github.com/openvinotoolkit/openvino/blob/8fb839633478574cefd28d49e25747d7c830b5a6/src/core/src/type/nf4.cpp) | `torch_npu_lowbit.py` 中的 16 项 NF4 码表；Python 打包与量化逻辑在本项目实现。没有复制整个 OpenVINO runtime | `LICENSES/OpenVINO.txt`，Apache-2.0；Intel 版权见 `NOTICE` |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes/tree/c82f51c0f784d8a43ebcb9cdefbf94e3f3b9c6c3) | OpenVINO NF4 文件明确注明的上游来源；保留间接来源及版权，不宣称直接移植其 CUDA 内核 | `LICENSES/bitsandbytes.txt`，MIT |
| [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py) | `torch_npu_lowbit.calibrate_int8` 与 `research/w8_gptq_candidate.py`；改为 CPU FP64、固定 FP16 scale、对称 INT8、稳定排序、死列保留及逐行回退。相关研究脚本使用同一适配实现 | `LICENSES/GPTQ.txt`，Apache-2.0 |
| [Maratyszcza/FP16](https://github.com/Maratyszcza/FP16/blob/782eea126dc5c755827be751a099eb01826175cf/include/fp16/fp16.h) | `npunlock/tests/fixtures/npu4000/rmsnorm.c` 和 `research/build_rmsnorm_reduce.py` 中的 FP16→FP32 思路/函数改编为 SHAVE 四路 SIMD；生成的 RMSNorm ELF 也对应此源码 | `LICENSES/FP16.txt` 与内核旁 `FP16_LICENSE`，MIT；原版权见 `NOTICE` |
| [Level Zero v1.17](https://github.com/oneapi-src/level-zero/tree/5604d3963dec3c98d4e56ccf6fef0671a47114a2) / [Intel NPU extensions](https://github.com/intel/level-zero-npu-extensions/tree/ddbf8d65e2e22150e23efd383839aa80f84d4dac) | `npunlock/src/workers/level_zero_min.h` 中重新命名并缩减的 API/ABI 声明；`research/dpu_diagnostic_extra.c` 的 profiling 声明。并未打包 loader/driver 实现 | `LICENSES/Level-Zero.txt`、`LICENSES/Intel-NPU-Extensions.txt`，MIT |
| [Intel NPU40XX firmware headers](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/details/api/vpu_nnrt_shavert.h) | `npunlock/python/npunlock/npu4000.py` 与 `torch_npu_lowbit.py` 的结构尺寸、字段偏移、dtype 和任务语义参考；保守保留 ABI 定义来源的 MIT 声明 | `LICENSES/Intel-Firmware-Headers-MIT.txt`；这些文件的 SPDX 是 MIT，不能只套仓库根 Apache-2.0 |
| [PyTorch moments_utils.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/moments_utils.h) | 2026-10-18新增`resident_layernorm.c`和`layernorm_mixed_moments.py`改编其Welford分组/级联合并顺序为标量SHAVE与独立研究模拟；未移植AVX指令。整数融合辅助运算为本项目实现 | `LICENSES/PyTorch.txt`，BSD-3-Clause；本机2.14.1+cpu头文件哈希见`provenance/layernorm-moments.json` |

新增 ELF 内核为本项目 C 源码的编译产物；保留源码、汇编与构建 manifest 供核对，不把它们说成从 OpenVINO 复制的现成计算内核。
2026-10-11快照未重新编译原内核；后续新增内核的源码、汇编、构建脚本与哈希由配套manifest记录。

## 设计参考与外部依赖

`npu_compiler_elf` 的 ELF 扩展、reader/relocation 代码，以及 `intel/linux-npu-driver` 的加载语义用于
理解并验证二进制格式；本快照只保留来源 URL/提交/SHA 清单，不整体打包这两个仓库。
所查 `npu_compiler_elf` 文件头为 Apache-2.0；Linux 驱动根许可为 MIT，另存 `LICENSES/Intel-Linux-NPU-Driver.txt`。
`npu_compiler` 的构建设置、OpenVINO 的执行方式、PyTorch 的 CPU reduction 和 llama.cpp 的 SYCL norm/scale
用作设计或实验参考。不能把所有“看过”的代码描述为直接移植；具体采用位置见 `REFERENCES.md` 和对应实验报告。

PyTorch / NumPy 是外部依赖。Transformers / safetensors / tokenizer 与 SmolLM2 模型用于单独的模型验证，
OpenVINO runtime / Intel NPU Acceleration Library 用于比较和调研；本快照不包含它们的软件包或模型权重。
各软件包与模型下载后遵循其自身许可证，不由本项目许可证覆盖。

Intel NPU 驱动、图编译器、MoviTools、Windows SDK、MSVC/MinGW 和预编译 worker 不在源码快照内。
没有据此声称本项目能脱离 Intel 编译器，或这些外部二进制都适用 Apache/MIT 再分发。
驱动生成的图 `.blob` 也不纳入本次公开源码目录：本机已观察到载体图包含 Intel 预编译内核文本，
完整图夹具的再分发不能仅凭本项目的 C 源码许可证推定。

## 来源记录

- `provenance/reference-sources.json`：本轮网上核对的固定 URL、SHA-256、许可证文件及本地匹配结果。
- `npunlock-probe/research/sources/manifest.json`：原有 37 项 NPU compiler / ELF / driver 调研材料的固定提交和哈希。
- `provenance/npunlock-upstream.json` 与 `npunlock-changes.patch`：本地上游改动的可审阅记录。

本轮补核的版本与历史设计输入分开记录：例如当前固定的 OpenVINO NF4 文件与码表一致，
不据此声称历史实验当时就是检查了这一提交。已有 GPTQ / FP16 文件则和固定上游内容进行了直接比对。
