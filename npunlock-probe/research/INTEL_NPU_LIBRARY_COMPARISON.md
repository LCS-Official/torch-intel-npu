# 与 Intel NPU Acceleration Library 的对比

查阅日期：2026-10-03。对比官方归档 main 分支的实现；本次未在本机安装或运行该库，
不保证旧版本与当前 Torch 2.14.1/驱动 32.0.100.5540 兼容，也不对两者速度作排名。

Intel NPU Acceleration Library 确实提供 PyTorch 接入和实验性训练。其 Linear 的
`AutogradMatMul.backward` 调用两次 NPU `run_matmul`，因此不能称为仅支持推理。
[官方使用文档](https://intel.github.io/intel-npu-acceleration-library/usage.html)、
[Linear 自动求导源码](https://github.com/intel/intel-npu-acceleration-library/blob/main/intel_npu_acceleration_library/nn/autograd.py)

项目于 2025-04-24 归档，Intel 声明停止维护，并推荐转向 OpenVINO/OpenVINO GenAI。
[官方仓库](https://github.com/intel/intel-npu-acceleration-library)

| 方面 | Intel 库 main 分支中的相关路径 | 当前本地后端 |
|---|---|---|
| Torch 接入 | torch.compile 后端、替换模块、特殊 Module 的 to 方法 | 真正 PrivateUse1 设备，名称 intel_npu |
| Tensor/梯度 | run_matmul 接收普通 Torch Tensor，通过 NumPy 调用 NPU，输出转回普通 Torch Tensor | Level Zero 分配的设备 Tensor，梯度与 SGD 更新驻留 NPU |
| FP32 | run_matmul 明确先把输入转为 FP16，再按原 dtype 返回结果；此路径不能保留所有 FP32 输入信息 | 已验证不降低到 FP16 的 SHAVE FP32 算术与归约，包括范围/非规格化数边界 |
| 依赖 | 链接 OpenVINO runtime | 不安装或链接 OpenVINO runtime；仍依赖 Intel 驱动/编译器、Level Zero 及构建工具链 |
| 自定义计算 | 主要通过 OpenVINO 算子和编译流程扩展 | 可控制 SHAVE 内核、舍入、补偿累加和反向实现 |
| 覆盖与优化 | 已有量化、LLM 和编译优化接口/示例 | 目前小规模算子与模型，尚无完整量化/LLM 支持，未使用高效 DPU GEMM |

表中 Tensor/FP32 结论针对实际 PyTorch `run_matmul` 路径，不把 NNFactory 中所有
可构造的 dtype 都等同于该路径，也不把其图模式与逐层执行混为一谈。源码依据：

- [runtime.py](https://github.com/intel/intel-npu-acceleration-library/blob/main/intel_npu_acceleration_library/backend/runtime.py)：
  输入转换到 FP16、`.numpy()`、`torch.from_numpy()`、输出 `copy=True`。
- [module.py](https://github.com/intel/intel-npu-acceleration-library/blob/main/intel_npu_acceleration_library/nn/module.py)：
  特殊模块的 `.to("npu")` 设置推理模式，并把底层设备参数改成 CPU；这与真实 Torch 设备分配不同。
- [CMakeLists.txt](https://github.com/intel/intel-npu-acceleration-library/blob/main/CMakeLists.txt)：
  获取 OpenVINO，链接 `openvino::runtime`。
- [compiler.py](https://github.com/intel/intel-npu-acceleration-library/blob/main/intel_npu_acceleration_library/compiler.py)：
  模块转换及 torch.compile 注册；编译接入不自动意味着完整设备后端。

## 当前价值与边界

本地实现的价值是研究设备常驻的 eager Torch 训练、真正 FP32 和可修改的 SHAVE，
为后续更完整的设备/调度支持积累基础。没有证明速度、模型覆盖或总体可用性优于 Intel 库。
rank<=4、每 Tensor<=1,048,576 元素、编译布局、512 个缓存图等限制仍在；
单个自定义计算块最多 65,536 项，较大 Tensor 使用受控 DMA 分块。不支持完整 Adam、AMP、
大模型训练或自动 NPU+GPU 联合调度，共享物理内存也不等于已经实现跨设备零拷贝。

本次扩展的可微 `.float()`/`.half()` 复用 PyTorch 原生 ToCopyBackward，设备内采用
SHAVE/DMA。混合精度模型保留 FP16 卷积权重存储，但激活和卷积实际计算为 FP32；
归一化、残差、池化与输出头均为 FP32。新初始化 15 步与近收敛 5 步，同权重 CPU 梯度
对照的最大 relative L2 分别为 0.00779%、0.06359%，热步骤零主机载荷复制。
不含 autocast、GradScaler、FP32 主权重；不宣称 FP16 卷积吞吐或训练峰值内存优势。
后续矩阵实现了 K 轴分块，超限卷积使用流式窗口及输入/权重梯度，不再需要完整
im2col 或重叠折叠分配；这仍属于正确性扩展，没有证明相对于官方库的吞吐优势。
详情和例子位于根目录 TORCH_NPU_README.md、torch_npu_mixed_demo.py，
报告为 research/results/torch-npu-mixed-fp32-compute-1.json。

已有库值得作为数值、复制与速度的对照，也可借鉴其编译优化。面向当前目标，继续完成
设备后端有独立价值；如果目标改为直接部署现成推理模型，应优先评估维护中的官方推理栈。
