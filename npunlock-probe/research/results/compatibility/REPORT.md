# NPU 兼容性修复与首轮性能优化（2026-10-04）

已先修复可复现的正确性/布局问题，再优化归约索引。保持真正 FP32、补偿求和、既有精度容差与禁止 CPU 算术回退。

| 问题 | 修复/边界 |
|---|---|
| 极大/极小 FP32 除法梯度错误 | 分母导数改用 -(g/y)*(x/y)，不计算 y*y；含 0/Inf/NaN |
| mean(dtype=FP16) 的 FP32 输入梯度不符 | 先按输出 dtype 舍入 mean 的反向除法，再转回输入 dtype |
| 完整转置叶 Tensor 不能训练 | 支持密集 empty_strided、算术/归约、clone/copy_/zero_、梯度布局/累加与 SGD；DMA 逆排列写回 |
| neg/abs/sign/reciprocal 缺少反向 | 新增一阶梯度；reciprocal 保留 FP16 中间乘法的舍入/溢出 |
| 性能 resume 可能复用旧内核结果 | 新增原生扩展/build/内核 manifest/低层编译源哈希检查，另存本轮结果 |

五个独立兼容性进程通过；修改前失败复现位于 ../../compat-before/reproduced/。
原有设备回归 33 组/4,160 次提交、训练回归 11 组/2,732 次提交通过（Quick）。
另回归 K 超长矩阵/Linear 和密集流式卷积；各进程上下文健康、shutdown 分配归零。

## 晚导入 PyTorch 兼容办法

先 CPU backward，再注册 PrivateUse1，普通 NPU backward 仍可能触发内部队列断言。
PyTorch 只初始化一次设备队列：[引擎源码](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/engine.cpp)。
优先在首次任何 backward 前导入后端；已验证如下局部办法，退出后恢复调度设置，数值算术仍在 NPU 执行。

```python
with torch.autograd.set_multithreading_enabled(False):
    loss.backward()
```

## 归约优化的独立 A/B

内层循环从每项动态除法/取模换为单轴步进或多轴进位索引；遍历顺序、Neumaier 补偿和非规格化数处理不变。
84 个 shape/axes/kind 对照，每项随机有限值与 IEEE 特殊值两种模式，新旧结果逐位相同。
同一进程驻留输入/输出，3 次预热、9 次新旧交错计时，包含同步执行，排除编译/上传/校验下载。

| 形状 / 轴 / 类型 | 原内核 ms | 新内核 ms | 加速 |
|---|---:|---:|---:|
| [1, 256, 256] / [1] / sum | 20.632 | 13.689 | 1.51× |
| [64, 64, 16] / [1] / sum | 9.292 | 6.280 | 1.48× |
| [65536] / [0] / sum | 41.738 | 25.143 | 1.66× |
| [4, 4, 32, 32] / [0, 2, 3] / mean | 9.384 | 6.574 | 1.43× |

## 端到端代表工作负载

使用上轮相同保存输入/权重并校验 SHA256；Torch 2.14.1+cpu。前向采样 11 次、训练采样 7 次。
本轮 AC=1、电量 100%、Balanced；用户确认本轮在看视频，桌面未受控为空闲。其他负载/共享内存/功耗状态可能影响时间；
跨轮结果用作趋势参考，严格量化应在桌面空闲时另做对照。同进程交错 A/B 可减小干扰，仍不能消除全部波动。
训练包含 zero_grad、前向、FP32 MSE、反向、基本 SGD；首次调用检查前向/梯度/更新，热采样不逐步下载验证。
以下是两轮独立运行的中位数，兼容性修复也已进入本轮；不能把整个差值解释为严格的同进程内核 A/B。

| 工作负载 | 精度 | 上轮 ms | 本轮 ms | 加速 | 每次图提交（前→后） |
|---|---|---:|---:|---:|---:|
| conv_train | float16 | 3424.548 | 2600.744 | 1.32× | 2399→2399 |
| conv_train | float32 | 3329.425 | 2650.501 | 1.26× | 2326→2326 |
| mm_256 | float16 | 5923.773 | 4316.626 | 1.37× | 1034→1034 |
| mm_256 | float32 | 5920.319 | 4316.358 | 1.37× | 1030→1030 |
| mm_64 | float32 | 31.633 | 25.349 | 1.25× | 21→21 |
| resnet_train | float32 | 559.201 | 493.826 | 1.13× | 808→808 |

本轮所有热采样零测得的主机载荷复制/新增图编译；提交数未减少，改的是内核索引开销。
这仍明显慢于此前 CPU/GPU 基线，尚未实现专用 GEMM、DPU 训练或异步提交。

## 残差网络回退诊断

第一次整轮测量 SmallResNet 中位数 868.637 ms、范围 721.985–1175.639 ms，明显慢于上轮 559.201 ms；
原始结果保留在 ../performance-reduce-optimized/first-pass/，没有丢弃。剖析中归约略快，其他 DMA/原生调用及 Python 开销变大，
这组跨轮结果不能证明内核退化。随后移除连续 Tensor 的重复验证/stride 排序，完整边界检查仍在；上表为最终代码重测。
另用最终后端在同一进程交错运行两个相同网络，只切换新旧归约 ELF，保留两套图，不销毁/reimport。
3 对预热、9 对测量：旧 610.863 ms、新 541.788 ms，1.13×。
所有 12 对步骤的输入梯度、参数梯度及 SGD 更新按原始 FP32 位模式逐位检查一致；热循环零显式复制/新增编译。
这是同进程内核对照，不能量化其他阶段/系统波动的各自贡献。[原始 A/B 数据](resnet-interleaved.json)。

## 仍保留的限制

深层纯 FP16 训练的既有数值限制仍在；二阶导数、跨 CPU/NPU 可微传输、BF16/整数/RNG、Adam/动量 SGD、AMP、
任意部分/重叠/偏移视图、GPU+NPU 混合 Tensor、完整 torchvision 模型尚不保证支持。
没有声称已消除所有潜在 bug 或达到完整 PyTorch 后端兼容性。

原始数据：[汇总 JSON](summary.json)、[兼容性](compatibility.json)、[内核逐位/A-B](reduce-optimization.json)，
端到端原始样本、首次调用与 profiler 位于 ../performance-reduce-optimized/。

复现：`run_torch_npu.ps1 -Script torch_npu_compat_demo.py`；`run_torch_npu.ps1 -Script torch_npu_reduce_check.py`。
