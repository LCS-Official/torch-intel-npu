# 激活感知 W8 校准与公开接口（2026-10-07）

已将本轮通过验证的 W8 校准接入 `torch_npu_lowbit.calibrate_int8`，并为
`FrozenSmolLayer` / `FrozenSmolStack` 增加可选 `packed_weights` 参数。
校准显式在 CPU 离线完成；推理继续使用压缩 INT8 权重、FP16 激活/输出和现有 DPU 图。
默认 `quantize_weight`、SHAVE 内核、扩展及 session DLL 均未改变，没有新增依赖。

## 方法和数据

参考 [GPTQ 官方实现](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py)
的协方差误差反馈方法，并作如下适配：

- 固定原 FP16 逐输出行 scale，使用对称 `[-127,127]` INT8 码，没有零点。
- CPU FP64 求解，稳定的激活能量降序，1% 对角阻尼，128 列分块；还原原列顺序后保存权重。
- 零激活列保留原权重。逐行比较校准输入上的投影重建误差，候选变差则保留原量化行。
- 目标使用未提前舍入到 FP16 的 `code * FP16_scale`，与已核对的 DPU 路线一致。

本地 SmolLM2-135M-Instruct：12 条校准文本，共 349 个 token 输入行；
采集前四层的 norm1、attention、norm2、gated 四类实际 Linear 输入，
准备 28 个矩阵、20,736 个输出行。本次所有行的校准重建误差均下降，
投影平方误差合计为原量化的 **0.889%**（下降 **99.111%**）。
这个数值是校准集投影目标，不能解读为整模型 logits 误差或困惑度下降 99%。

质量检查使用另外 16 条提示，与校准文本完全分离。其中 8 条是此前诊断使用的
回归提示，8 条是 GPTQ 检查前固定的新提示；参数和权重选择不使用这些提示的输出。
小数据集只能证明这次固定实验有效。

## 前缀和连续 decode 结果

前四层在 NPU，其余 26 层、embedding、最终 norm 和 logits 在 CPU。
参照是官方完整 CPU FP16 eager 模型；手动 CPU 分段管线与其 logits 逐位一致。
下表百分比是最后一个位置的完整 logits 向量相对 L2：

| 提示 | 原 W8A16 | 校准 W8A16 |
|---|---:|---:|
| bird（回归） | 4.882% | 1.979% |
| math（回归） | 0.472% | 0.198% |
| paris（回归） | 0.219% | 0.229% |
| garden（回归） | 5.088% | 1.837% |
| dog（回归） | 5.775% | 1.699% |
| cat（新增） | 5.313% | 0.510% |
| fox（新增） | 6.506% | 0.567% |
| rain（新增） | 4.688% | 1.167% |
| letter（新增） | 5.292% | 1.551% |

原门槛保持 **L2<3%、下一 token KL<0.02**。原 W8 有 7/16 条未通过，
校准后 **16/16 通过**，最大 L2 1.979%、最大 KL 0.000415，16 次 argmax 与 CPU 相同。
paris 的误差略有增加，校准并不保证每条文本的所有指标都改善。

在 bird、garden、cat 上分别进行 32 个位置的连续 decode，使用相同的 CPU greedy
token 流驱动三个模式，保持输入一致（teacher forcing，不在 EOS 提前停止）：

| 提示 | FP16 最大 L2 | 原 W8 最大 L2 | 校准 W8 最大 L2 | 原 W8 / 校准超标位置 |
|---|---:|---:|---:|---:|
| bird | 1.410% | 4.882% | 2.922% | 4 / 0 |
| garden | 0.778% | 5.088% | 1.837% | 4 / 0 |
| cat | 0.882% | 5.313% | 2.233% | 4 / 0 |

校准模式 **96/96 个位置通过**，最大 KL 0.001519，下一 token argmax 96/96 与 CPU 相同。
FP16、原 W8 的 argmax 在这些位置也全部相同，因此 token 相同本身不能说明 logits 误差已解决。
最差 2.922% 接近 3% 门槛；还需要更大的独立语料和困惑度评估。
这三组固定输入检查也不能替代自由生成、长上下文和完整模型评估。

## 执行与回归

- 128 个逐层 CPU/NPU 对照全部通过既定执行门槛：L2<0.3%、最大 token RMS 缩放误差<1%。
  实测最大为 0.1034% / 0.4094%。它们使用实际 NPU 前驱输入，与模型量化质量检查分开。
- 图中仍有 `amode=0, wmode=2` 的 INT8 权重 DPU 矩阵任务；校准前后压缩码、scale
  字节数和各图常量区大小相同。没有把权重展开成 FP16 常量来替代压缩路径。
- 暖前缀/最终 decode 重复输出逐位一致；每次四层提交 4 张图，无新编译或显式主机载荷复制。
  此计数不覆盖驱动内部 DMA。旧 KV 快照在检查点保持不变，最终缓存长度为 55。
- 5 项 CPU 单元测试通过；16 项接口/设备检查通过，包括非法预量化字典在设备执行前被拒绝、
  外部字典修改不影响内部快照、绝对 checkpoint 层号，以及 `calibrate_int8 + QuantizedLinear` 实际 DPU 执行。
- 原型迁移到公开 API 后，权重、校准输入、前缀与 decode 的 **811 组数组全部逐位一致**。
- 公开 API 阶段共 **1,606 次 NPU 调用**，三个测试进程关闭后 live/cached/quarantined
  buffer 字节均为 0。离线 audit 重算了 448 项数组指标并核对源文件、模型、权重及许可证指纹。

仅增加显式 CPU 校准和预量化构造入口，本轮没有性能 benchmark，也不声称吞吐提高。
权重压缩大小不变并不代表总进程内存不变：校准会建立 CPU FP64 的 K×K 协方差与其他工作数组。

## 使用方式

```python
import torch
import torch_intel_npu
from torch_npu_lowbit import calibrate_int8, QuantizedLinear

# cpu_linear 是原模型的 CPU Linear；inputs 来自代表性数据运行到该 Linear 的实际输入。
inputs = calibration_inputs.reshape(-1, cpu_linear.in_features).half()
packed = calibrate_int8(cpu_linear.weight.detach().half(), inputs)
linear = QuantizedLinear(packed, bias=cpu_linear.bias)
with torch.no_grad():
    output = linear(x.half().to("intel_npu"))
```

输入须为有限 CPU FP16/FP32 的 `(rows,K)`，1<=rows<=4096、K<=4096。
输入及校准输出各不超过 4,194,304 元素，权重受原 1,048,576 元素上限约束。
返回原有不可变 `PackedWeight`，无梯度更新、分组校准或自动运行时校准。

SmolLM 的权重名为七个投影（如 `self_attn.q_proj.weight`、`mlp.down_proj.weight`）。
为每层完整准备七项后，通过公开构造器传入：

```python
from torch_npu_smollm import FrozenSmolStack

# packed_by_layer = {绝对 checkpoint 层号: {七项投影名: PackedWeight}}
stack = FrozenSmolStack.from_checkpoint(
    model_dir, start_layer=0, num_layers=4, mode="w8a16",
    packed_weights=packed_by_layer)
with torch.no_grad():
    hidden, cache = stack.prefill(hidden_states.half().to("intel_npu"))
```

完整字典必须恰好覆盖请求层段，全部为匹配形状的逐行 INT8 权重。
`FrozenSmolLayer.from_checkpoint(..., packed_weights=projection_dict)` 使用单层字典。
不传参数时行为与此前相同。本地研究脚本 `w8_calibration.load_weights()` 可从带指纹
清单的 `weights.npz` 加载本次准备的字典；该加载器绑定本地模型和源码版本。
这里仍是 SmolLM2-135M 专用冻结推理段，保留已有 64-token KV 上限。
本轮没有扩大通用 Torch 训练、INT4/NF4 或其他 LLM 的支持范围。

## 失败候选、来源和复现

先尝试的 scale-only 方案虽将校准投影平方误差降低 47.8%，但 bird 的 logits L2
从 4.88% 变成 5.48%、garden 从 5.09% 变成 6.81%。已保存到 `scale-only/`，未接入默认路径。
`candidate-before-api/` 保存通过验证的 GPTQ 原型，`sources/` 保存当前接口与测试源码快照。

算法来自上述 GPTQ 官方提交 `2d65066eeb06a5c9ff5184d8cebdf33662c67faf`。
原源码、SHA256、URL 和原始 Apache-2.0 许可位于 `references/`，
仓库许可证副本位于 `E:\Pytorch_xfire_NGPU\LICENSES\GPTQ.txt`。
实现保留本地对称量化、固定 scale 和死列处理规则，并非直接使用官方 GPTQ checkpoint。

在工作目录用已有 `.conda\npunlock` 环境复现（脚本会覆盖当前结果，需先保存要保留的记录）：

```powershell
.\run_torch_npu.ps1 -Script .\test_torch_npu_lowbit.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_calibration.py -ScriptArguments '--method','gptq'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_calibration_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_calibration_decode.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_calibration_checks.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_calibration_audit.py
```

`prepare.json`、`holdout.json`、`decode.json`、`checks.json` 和 `audit.json` 保存完整记录；
四个 NPZ 保存校准输入、权重及两类输出数组。audit 使用本次保存的源码与原型快照；
修改代码后须更新对应快照并重新验证。历史 `w8-quality` audit 绑定当时源码指纹，
其失败不会自动表示当前版本数值回归，应使用本轮检查。
