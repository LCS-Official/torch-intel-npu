# W8A16 精度限制诊断（2026-10-06）

本轮复现了四层 NPU＋26 层 CPU 在鸟的故事提示下的 logits 误差。
主要证据指向权重量化与舍入扰动被后续层放大；没有发现超出已有执行容差的 NPU 错算。
3% 的质量门槛保持原值，默认量化器、模型、SHAVE 内核、Torch 扩展和 runtime 均未修改。
本轮没有性能优化接入，也没有新增 OpenVINO runtime 依赖。

## 同一提示、分层对照

固定本地 SmolLM2-135M-Instruct、CPU FP16 eager、前四层 NPU。
鸟的提示使用 system=`Tell stories.`，user=`Tell a story about a bird.`，24 个输入 token，
与上一轮失败提示完全相同。其他两个提示使用 system=`You are a helpful assistant.`，输入分别为 26、27 token。

| 提示 | NPU FP16 logits L2 | 原 W8A16 logits L2 | scale 优化候选 L2 |
|---|---:|---:|---:|
| Tell a story about a bird. | 1.410% | **4.882%** | **3.698%** |
| What is two plus two? | 0.080% | 0.472% | 0.407% |
| Complete: Paris is the capital of | 0.143% | 0.219% | 0.256% |

百分比都是相对完整官方 CPU FP16 模型最后一个位置的 logits 向量 L2。
加粗项没有通过原来的 3% 门槛；`status=passed` 只表示诊断流程与资源核对完成，
每项 `quality_gate_passed` 单独记录模型质量是否达标。

- 手动 CPU 四层接口、position、因果 mask 与剩余层管线，在三个提示下与官方整模型 logits 逐位一致。
- 原 W8A16 的 CPU 分段参考，在鸟的提示下也有 **4.402%** logits L2；
  NPU 对该量化参考的最终 logits L2 为 0.543%，不能把两种相对误差简单相减。
- 每层都用实际 NPU 前驱输出作固定输入，再比较 CPU/NPU。
  原方案的 24 个逐层对照最大执行 L2 为 **0.0801%**；
  候选的 24 个对照最大为 0.0876%，均通过原 L2<0.3%、最大 token RMS 缩放误差<1% 的执行门槛。
- 原权重量化在 28 个矩阵中的权重 L2 约为 0.77%～1.18%。
  这与最终 logits 的误差不是同一个量，后续 26 层会改变扰动的大小和方向。

这些结果支持“提示相关的量化/舍入敏感性”，不支持把 4.88% 全部归因于设备错误或 CPU/NPU 接口错误。
有限测试也不能证明驱动和模型支持中不存在其他 bug。

## 核对 DPU 的 scale 舍入语义

只把 INT8 解量化成 FP16 权重、再用官方 CPU Linear，鸟的 logits L2 为 2.884%。
它与现有 W8A16 DPU 路线有不同的舍入位置，不能据此判断原 CPU 参考实现写错。

对失败提示第一层的七种实际投影输入，单独运行：

1. 原压缩 INT8 图；
2. 同一权重明确解量化为 FP16 常量的 NPU MatMul；
3. CPU 以未提前舍入的 `code * FP16_scale` 做高精度矩乘；
4. CPU 以提前舍入为 FP16 的解量化权重做高精度矩乘。

七组压缩图均有 `amode=0, wmode=2` 的 DPU 矩阵任务证据。
对未提前舍入的 CPU 权重参考，执行 L2 为 0.0228%～0.0306%；
对提前 FP16 舍入的参考，为 0.0274%～0.0376%。七组压缩图与 FP16 常量图都没有逐位一致。
所有组都更接近现有未提前舍入的参考，因此保留现有 CPU 参考。
这是输出对照的证据，不能推断 DPU 内部每一步都严格采用某个浮点格式。

原始数组、矩阵模式与 FP16 对照 blob 指纹见 `linear-semantics.json`、`linear-semantics-outputs.npz`。

## 简单 scale 优化为何未接入

参考 [llama.cpp 的 make_qx_quants 源码](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)
中固定整数码的最小二乘 scale 思路，独立实现 `sum(weight*code) / sum(code*code)`。
候选最多迭代四次，scale 每次先序列化成 FP16，再重新生成整数码；
只有实际保存后的逐行权重平方误差降低才接受。INT8 码、scale 数量及图布局不变，无校准数据。

离线核对四层共 20,736 行，每行目标误差均没有增加；全体权重平方误差仅下降 **0.575%**。
鸟的 NPU logits L2 从 4.882% 降至 3.698%，仍未达标；
下一 token 的 KL 从 0.001549 升至 0.002498，巴黎提示的 L2 也略有增加。
减小权重重建误差不能保证减小最终模型误差，所以候选只保存在研究脚本中。
没有把只对一个提示有利的参数设成默认实现。

候选源码为 `w8_scale_candidate.py`，实验仅在独立进程中替换准备阶段的量化函数。
`baseline.json` / `refined.json` 保存三提示对照，默认 FP16 数值在两个进程中保持相同。

## 连续 decode 的实际影响

完整 CPU FP16 模型先决定下一个 token；两条 NPU 管线始终接收同一串 token，
用于隔离计算误差，避免不同生成文本造成不可比较的输入。
固定 32 次采样，不提前按 EOS 停止，最终 NPU/CPU tail cache 长度均为 55。

| 模式 | 32 个位置最大 logits L2 | 超过 3% 的采样步（从 0 开始） | 最大下一 token KL | 与 CPU 相同的 argmax |
|---|---:|---|---:|---:|
| FP16 | 1.410% | 无 | 0.0000438 | 32/32 |
| W8A16 | 4.882% | 0、1、4、15 | 0.0018675 | 32/32 |

最后一次 decode 的常驻重复执行，两种模式都逐位一致、各四次图提交，
无新的编译或显式主机载荷复制。
这条提示的贪心下一 token 未受超标影响，但不能因此豁免 3% 门槛，
也不能将少量提示当作完整模型质量或困惑度评估。
完整每步值、CPU 固定续写和原始 logits 在 `decode.json` / `decode-outputs.npz`。
本轮未计时，不据此作吞吐量结论。

## 核对与复现

四个正式报告进程共完成 454 次 NPU 调用，正常关闭后 live/cache/quarantine 均归零。
离线 audit 重新计算 133 项数组误差、KL 与下一 token 对照，检查 24 项正式源/二进制指纹，
核对两个进程的 57 个 FP16/CPU 对照数组逐位一致，并验证候选全部 20,736 行及零权重、常量权重的量化目标。
正式源与上一轮默认实现相同，快照保存在 `sources/`。
`audit.json` 还保存实际 safetensors SHA256。

最初诊断脚本遗漏最后一个模型循环引用的释放、投影脚本误用了参考接口、
audit 的预期行数漏算了两组投影；均已修正，初始日志与源码保留在 `initial-*`。
这些是研究脚本修正，没有改生产生命周期规则或数值门槛。

在 `E:/Pytorch_xfire_NGPU` 中运行：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_quality_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_quality_probe.py -ScriptArguments '--quantizer','refined'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_linear_semantics_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_decode_quality_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_quality_audit.py
```

重新运行会覆盖本目录同名结果，应先保存原批次。

下一项精度工作应考虑用独立校准数据评估激活感知的 W8 量化，
或在敏感投影保留较高精度，再用未参与校准的提示与连续 decode 检验收益。
若研究分组 INT8，还需独立确认驱动是否保留压缩 DPU 路径及性能，不能只看 CPU 重建误差。
本轮结果不支持为修复这个限制重写严格 FP32 GEMM。
