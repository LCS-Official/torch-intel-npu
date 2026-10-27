# 普通 PyTorch 模型迁移兼容性：第一轮

日期：2026-10-16。本地开发版，尚未同步至 GitHub v0.1.0。

本轮测试普通 ATen `intel_npu` 设备后端，不使用冻结 DPU/SmolLM 适配器；
这是 CUDA 风格 PyTorch 模型流程的迁移检查，不是 CUDA 运行时或二进制兼容认证。

## 环境与口径

- Windows x64 / Core Ultra 7 258V / NPU 8086:643e。
- NPU API driver_version=1005540，Torch 2.14.1+cpu，Python 3.12.14。
- 使用现有 `.conda/npunlock`；CPU 参考 1 线程、关闭 oneDNN 以运行 FP16 backward。
- CPU 初始化、固定随机种子 43、MSE；上传后独立训练；不重置 CPU 权重追随 NPU 轨迹。
- CPU/NPU 输出、loss、输入/参数梯度、参数与动量逐项对照；FP16 rtol=0.005/atol=0.0008，FP32 rtol=0.0001/atol=0.000008。
- 本轮未安装包、未下载模型、未重编译或修改 C++/SHAVE/驱动，不测吞吐/功耗。

## 发现与修复

1. 普通 SGD momentum 第二步执行 `buf.mul_`，原先被分解为 `mul.out` 并触发输出别名/布局拒绝。
   新增 PrivateUse1 `mul_.Scalar/Tensor`：设备内读到独立结果，再写回原 Tensor；支持完整密集转置并保留 version。
2. Nesterov 与 weight_decay 使用 `grad.add(..., alpha=...)`，原先只接受 alpha=1。
   现在加减法支持非 bool 实数 alpha 与对应一阶梯度；有限 alpha 必须在输出 dtype 范围内。
   非 1 alpha 使用 FP32 中间计算，alpha 按输出 dtype 舍入，最后返回原 dtype；alpha=1 保留旧路径。
3. `mul_` 显式限定 no_grad、同 shape/dtype/device 的 Tensor 或 Python 实数；拒绝广播、部分存储、
   不同 stride 的重叠别名与 bool。self-alias 安全写回，saved-tensor version guard 仍报错。

没有通过修改 torch.cuda、替换模型 forward 或 CPU 算术回退来取得通过结果。

## 普通模型训练与恢复

| 模型 | 精度 | 四步 NPU loss（前三步训练、第四步从 checkpoint 恢复） | 各步显式 host copies | 其中标量缓存首次上传 |
|---|---|---|---|---|
| mlp | torch.float16 | 0.04385376 → 0.042480469 → 0.040344238 → 0.037689209 | [3, 2, 0, 0] | [3, 2, 0, 0] |
| mlp | torch.float32 | 0.038916808 → 0.037463598 → 0.035147697 → 0.032362025 | [1, 0, 0, 0] | [1, 0, 0, 0] |
| conv | torch.float16 | 0.30249023 → 0.28710938 → 0.26220703 → 0.23168945 | [1, 0, 0, 0] | [1, 0, 0, 0] |
| conv | torch.float32 | 0.25801751 → 0.24385349 → 0.22098608 → 0.1931171 | [0, 0, 0, 0] | [0, 0, 0, 0] |
| tiny-resnet | torch.float16 | 0.073669434 → 0.070861816 → 0.06640625 → 0.061157227 | [0, 0, 0, 0] | [0, 0, 0, 0] |
| tiny-resnet | torch.float32 | 0.22808892 → 0.21737902 → 0.20008807 → 0.17886566 | [0, 0, 0, 0] | [0, 0, 0, 0] |

MLP：5→4→2；卷积网络：Conv2d(1→2)、ReLU、全局池化、Linear；TinyResNet：三层卷积、单个残差分支、全局池化和 Linear。
输入分别为 (2,5) 与 (2,1,4,4)；未加入 BatchNorm 或分类 loss。
普通 `.to("intel_npu")`、forward、MSE.backward、SGD.step、zero_grad(set_to_none=False)、eval/inference_mode 通过。

模型使用 lr=0.025、momentum=0.8、dampening=0.1、foreach=False、fused=False。
checkpoint 显式将 detached 参数及 SGD 动量转为 CPU，用内存 BytesIO 保存，`torch.load(..., map_location="cpu", weights_only=True)` 后
加载到新 NPU 模型/优化器并继续第四步；恢复后的输出、梯度、参数和动量仍匹配独立 CPU 轨迹。
复制统计排除输入准备、数值检查下载及 checkpoint 搬运；第一/二步可首次上传 Python 标量缓存。
第三步与恢复后的训练步显式 host payload copies 均为 0；不表示驱动内部没有 DMA。

## SGD 与算子契约

- basic、momentum、dampening、Nesterov、weight_decay、Nesterov+weight_decay、maximize：两种 dtype 共14组，每组4次更新。
- SGD 使用非连续完整密集转置参数；每一步检查参数、布局与动量缓冲区。Python 数值超参数，显式 foreach=False/fused=False。
- mul_：两种 dtype 的标量/Tensor、完整密集转置、self-alias、返回对象和版本计数。
- add/sub：两种 dtype 的广播、一阶梯度、out、Python 标量与 alpha=0/0.8/-2/60000/Inf/NaN。
- 12个非法更新/alpha/别名动作在没有新增 NPU 图提交的情况下拒绝；saved-tensor version 变更检查通过。

## 仍阻止通用模型迁移的项目

| 检查入口 | 本机实际首次失败位置 |
|---|---|
| batchnorm-default | `Requires FP16/FP32, rank 0-4, and 1-1048576 elements` |
| integer-target | `Requires FP16/FP32, rank 0-4, and 1-1048576 elements` |
| cross-entropy | `intel_npu does not implement aten._log_softmax.out; no CPU arithmetic fallback` |
| adamw | `intel_npu does not implement aten.lerp.Scalar_out; no CPU arithmetic fallback` |
| layernorm | `intel_npu does not implement aten.native_batch_norm.default; no CPU arithmetic fallback` |
| gelu | `intel_npu does not implement aten.gelu.out; no CPU arithmetic fallback` |
| softmax | `intel_npu does not implement aten._softmax.out; no CPU arithmetic fallback` |
| embedding | `intel_npu does not implement aten.index_select.default; no CPU arithmetic fallback` |
| batched-matmul | `NPU MatMul requires matching-dtype 2D matrices with compatible inner dimensions` |
| sdpa | `NPU MatMul requires matching-dtype 2D matrices with compatible inner dimensions` |
| native-checkpoint | `intel_npu requires FP16 or FP32` |

这些11项是已验证会明确拒绝的边界，不计为功能支持。默认 BatchNorm 的 num_batches_tracked 和分类 target 需要整数存储；
CrossEntropyLoss 先遇到 log_softmax 缺口，修复该算子后还需解决 target 与 NLL 路径。
LayerNorm 当前由 PyTorch 分解到 native_batch_norm；SDPA 当前先遇到非二维矩乘限制，并未验证其他注意力组成。
AdamW 在原地权重衰减后才遇到 lerp 缺口；未支持的组合可能已更新部分参数，捕获异常后应从原始 checkpoint 重建。
原生 torch.save(NPU Tensor) 走字节存储序列化，目前不支持 Uint8 设备存储；显式 CPU checkpoint 是本轮验证的入口。

## 验收计数与既有回归

迁移最终验收：25组计算/训练通过，11组明确未支持边界，共 6,501 次真实 NPU 调用。
四个独立进程均无 live buffer 增量、无 context failure；shutdown 后 live/cache/scalar/quarantine/session 全部归零。
基线33组中只有基本SGD两种 dtype 通过；其余20个计算/训练配置因 mul_/alpha 等限制失败，另11项为此次记录的未支持项目。
诊断开发期间修正过统计字段、标量缓存预期和首次未支持算子名称；这些不算作后端修复。

- 设备 Quick：33组 passed，4,160次 NPU 调用。
- 训练 Quick：11组 passed，2,732次 NPU 调用。
- 原有兼容性：5组 passed，236次 NPU 调用。
- 最终迁移与既有回归合计：13,629次 NPU 调用（不含开发中失败诊断/重跑）。

## 使用与复现

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_migration_demo.py
.\run_torch_npu.ps1 -Device -Quick
.\run_torch_npu.ps1 -Training -Quick
.\run_torch_npu.ps1 -Script .\torch_npu_compat_demo.py
.\.conda\npunlock\python.exe .\npunlock-probe\research\torch_migration_report.py
```

报告生成需当前 after 诊断及上述回归 JSON；四份 before 诊断可选，提供时额外审计修复前基线。
按用户要求验收后删除 JSON/日志，保留本报告与源码；清理后重新执行上述命令可审计当前实现。
重建修复前基线需恢复原始后端版本；不能直接从本报告恢复逐项原始数据。

## 后续优先级与范围

优先补默认 BatchNorm 的整数计数/模型状态迁移，再做分类训练所需的整数 target、稳定 log_softmax 与 NLL 前后向。
随后扩展 LayerNorm/Embedding/批量矩乘和注意力、AdamW；每项均需独立数值与拒绝契约测试。
本轮不认证任意 torchvision/HF 模型、CUDA扩展/运行时、torch.cuda API、多流、AMP、GPU+NPU协同、其他设备/驱动或长时间训练。

## 最终代码指纹（SHA-256）

```text
torch_npu_migration_demo.py
81ba78c075c6d9cfe19198f4c7b14c2a542ad748815410738363f16d222b2e10
torch_intel_npu.py
8a3b5542624947c96c490eda603520e0d0dae0a3c38b45d5301cbe353d277c9b
torch_npu_autograd.py
5ed64d0a31315df1b0d0f6210d71fcf5fbf65298c9ac5eb063ec4b5432789372
npunlock-probe/torch-device/backend.cpp
d6acf5d765b1024ef783e4370f30064e7c67aa4a26f7a10e9eba8c0864fe6d09
npunlock-probe/torch-device/_torch_intel_npu.pyd
3325f761ea55485271e26ecc3667b820f428d31f665e4c31fb916e8c30aefc8b
```

## 交付与清理

验收并生成本报告后，按用户约定清理31份临时JSON/日志，共163,849字节（约160.01KiB），
包括本轮修复前/后诊断与设备/训练/兼容性回归产物。保留实现、测试、研究脚本与Markdown报告。
未安装新依赖、下载模型、生成图片、启用持久图缓存或推送GitHub；原生内核与构建产物均保留。
