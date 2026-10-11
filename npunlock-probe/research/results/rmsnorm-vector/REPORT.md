# RMSNorm 完整行并行与向量原型实测

2026-10-05，Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，驱动
32.0.100.5540，编译器 8.3。结果：`passed_with_limitations`。

默认 RMSNorm 已采用编译器的多 tile 完整行调度，复用上一阶段的 FP32 标量内核。
同进程 W8A16 单层比较，B1/S32 从串行融合的 10.13 ms 降到 3.17 ms，
B4/S64 从 57.41 ms 降到 9.95 ms。CPU 参考仍更快。
行内向量归约经过隔离原型测试，修正转换错误后数值通过，但无稳定性能收益，
所以没有替换默认内核。没有新增依赖或 OpenVINO runtime。

## 实现

上一阶段的 ReferenceSW 编译模式只用一个 tile。512 行时会把宽度 576 拆为
288+288，不能执行完整 RMS 归约，适配器因此只能按 <=256 行切片、分别执行、合并。

本轮实机编译检查发现：DefaultHW 对 8..512 行会把完整的 576 元素行分给
4 个 tile，每个 tile 2 个 SHAVE 任务；259 等非整除行数也保留完整行。
相应编译器源码中已有 SW task 维度选择和多 tile 分配逻辑，本轮直接复用，
不修改其 DMA、barrier、任务参数或调度字节。

少于 8 行仍使用 ReferenceSW。DefaultHW 的单行检查显示宽度拆成 80/64 元素任务，
3 行则会跨 tile 复制完整输入，不适合直接作为此处的完整行并行契约。
这些布局只做编译检查，没有执行自定义 norm。

`npunlock/python/npunlock/npu4000.py` 的 `compile_rmsnorm` 现在：

- 默认接受 1..512 行；`parallel=False` 保留 <=256 行的串行融合对照。
- 验证预期任务数量、没有 DPU、每个任务三个 memref 均为完整的 576 列、连续步长，
  每个并行 shard 最多 64 行；逐 tile 输出区间不重叠，shard 行数之和等于输入行数。
- 继续复用原 Power fingerprint、设备/编译器 profile、CMX、代码窗口检查；
  最终只替换 kernel text，其余 blob 字节保持不变。

`torch_npu_smollm.py` 的默认路径能一次提交全部 B*S 行，gamma 仍在设备上广播并缓存。
会话 key 区分串行/并行模式。原切片路径保留给串行性能对照。
最大 B8/S64 的两个 norm 各一张图，加 attention、FFN，整层从 24 次提交降到 4 次；
常用尺寸原来已是 4 次，本轮收益来自图内部的行并行。
所有暖运行均无新增显式主机上传、下载和会话创建；统计不包含驱动内部 DMA。

没有放宽通用算子的 65,536 元素计算分块或 1,048,576 元素存储上限。
`npu4000.py` 中除 `compile_rmsnorm` 外的代码与 `before/` 快照相同；
`torch_intel_npu.py`、`torch_npu_lowbit.py`、`torch_npu_transformer.py` 的 hash 未改变。

默认 C、ELF、汇编和 build manifest 与上一阶段一致：

- C SHA256：`f691e106a6fc9b472d30d81687f2a1785d040aa34c933bf299ad35aef1aa7066`。
- ELF SHA256：`1c19b63d8e083e59986304f5d40467746950d5d585399591bb2942db3e3e6a47`。
- `.text` 为 3,952 字节。FP16 IO、真正 FP32 平方/补偿均值/开方/倒数/归一化，
  以及 gamma 前的 FP16 舍入顺序都没有变化。

## 正确性与边界

默认路径完成 8 个 case，共 61 项记录，另有 5 项 carrier 行数拒绝检查。

| 检查 | 结果 |
|---|---|
| 11 种输入 × 两个真实 gamma | 22 组 norm 与旧逐算子 NPU 实现的 FP16 位模式相同 |
| 7/8/9 行切换、17/259 行尾部、B4/S64、B8/S64 | 完整行调度；每次 norm 仅一次图提交 |
| 零、±65504 的平方、FP16 subnormal/signed zero | 与旧 NPU 位模式相同；CPU FP32 参考容差通过 |
| Carrier 行数 0、513、1024；串行模式 257、512 | 编译前拒绝，无新增会话 |
| 层 0、10、20、29 的数值、最大尺寸、性能/profile 前置检查 | 共 17 项，原有门槛全部通过，所有整层暖调用为 4 次 |
| Guards | 16 项通过 |
| 因果性 | FP16/W8A16 两项通过 |
| 短文本混合生成 | 两个提示 × 两种模式，与 CPU 参考的 8 个生成 token 相同 |

Norm 对 CPU 参考仍采用 `rtol=.002, atol=.001`。全部样本合计 17 个 FP16 编码
与 CPU 参考不同，最大绝对误差 0.00012207；这并非本轮新增，默认内核代码没有变化。
跳过 gamma 前的中间 half 舍入会产生大量不同编码，测试仍明确检查这一语义。

整层门槛仍为对相同量化 CPU 参考的 L2<0.3%、每项
`abs(error)/(abs(reference)+token_RMS)<1%`；固定输入分段 L2<0.1%，量化质量
对未量化官方 CPU 层的 L2<3%。最大整层执行 L2 约 0.06784%，scaled error
约 0.42664%，最大量化质量 L2 约 1.134%。原先固定 `rtol=.01, atol=.005`
的逐元素 mismatch 仍保留，不能将回归通过解释为消除了所有 CPU/NPU 数值差异。

8 个最终 case 共 6,288 次 NPU 提交。每个结束后 live buffer、隔离 buffer、
缓存会话均为零，context 未失败。原始 JSON 记录执行时源码 hash、kernel manifest、
包版本及模型 manifest，最终源码快照在 `after/`，汇总在 `summary.json`。
模型固定 revision 仍为 `12fd25f77366fa6b3b4b768ec3050bf629380bac`，全部测试离线。

## 同进程单层性能

同时比较并行 FP16、并行 W8A16、旧逐算子 W8A16、串行融合 W8A16、
CPU FP16、CPU FP32。先检查新/旧/串行 W8A16 整层输出位模式相同，再各预热
3 次，进行 21 轮轮换顺序计时，取中位数。CPU 8 线程，RoPE 和 mask 已准备；
NPU 图、输入、权重、gamma 已准备。计时排除加载、初次编译和显式传输。

单位毫秒，B=batch，S=序列长度，hidden=576。

| 输入 | 旧逐算子 W8 | 串行融合 W8 | 并行融合 W8 | 串行/并行 | 并行 FP16 | CPU FP16 | CPU FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1/S1 | 13.207 | 2.707 | 2.803 | 0.97× | 3.172 | 1.041 | 0.896 |
| B1/S32 | 18.391 | 10.135 | 3.173 | 3.19× | 3.571 | 2.448 | 1.893 |
| B4/S64 | 130.711 | 57.411 | 9.950 | 5.77× | 10.257 | 7.180 | 6.635 |

B1/S1 仍采用相同的单 tile 内核，不预期有并行收益，表中小差异属于本轮计时表现。
B4/S64 对最早逐算子实现的同进程加速约 13.14 倍，但对 CPU FP32 仍慢约 1.50 倍。
本机桌面负载未受控，结论只适用于这些单层测量；没有测整网 token/s 或功耗。

另一次 B4/S64 W8A16 分段 profile（每段 9 次中位数）：

| 分段 | 延迟 ms | 图提交 |
|---|---:|---:|
| norm1 | 5.989 | 1 |
| attention | 0.969 | 1 |
| norm2 | 5.787 | 1 |
| FFN | 0.973 | 1 |

两次 norm 占分段中位数之和约 85.8%。profile 与整层 benchmark 来自不同进程和
运行时段，分段中位数不能相加当作整层延迟。本轮减少了 norm 的串行计算，但
FP16 转换、舍入和串行输出循环仍值得后续定位。

## 向量原型：没有接入默认路径

`build_vector_probe.py` 在本目录构建隔离的四通道 FP32 Neumaier 归约，
产生 VAU.MUL/ADD/SUB.f32 指令，默认 C/ELF 从始至终没有覆盖。

首次原型使用 `__builtin_convertvector(Half4,Float4)`，工具链
`moviCompile v00.114.11 Build 4207` 在对应位置生成
`CMU.CP.128.u16.f32`，将 half 位编码按整数转换。首个 576 元素输入中，
517 项超过 CPU 参考容差，全部 576 个编码与原内核不同，立即停止该原型测试。
源码、汇编、ELF 和失败记录保存在 `vector-cast-failure/`。

改用原先验证过的标量 `load_half`，只向量化 FP32 算术，得到正确结果：
11 个行数（1、3、7、8、9、17、32、64、256、259、512）×两个真实 gamma，
共 22 项与原标量内核逐位一致。该原型的检查使用随机有限输入；默认路径的
特殊值检查在前述 norm case 中完成。

同进程 norm 微基准（21 轮、3 次预热，两个版本都用新的多 tile 调度）：

| 行数 | 原标量 norm ms | 向量原型 norm ms |
|---|---:|---:|
| 1 | 0.7325 | 0.7130 |
| 32 | 1.4062 | 1.4273 |
| 256 | 4.9121 | 4.9083 |
| 512 | 7.6658 | 7.8068 |

没有稳定加速，512 行略慢，因此保留已验证的标量 kernel 配合多 tile 调度。
正确原型另执行 236 次 NPU 提交，清理后 buffer、隔离 buffer、缓存会话均为零，
context 未失败。原型失败与当前默认路径的通过结果分开记录。

## 复现

在 `E:\Pytorch_xfire_NGPU` 下串行执行：

```powershell
foreach ($rmsCase in 'norm','numeric','boundary','guards','causality','hybrid','performance','profile') {
    .\run_torch_npu.ps1 -Script torch_npu_smollm_demo.py -ScriptArguments '--case',$rmsCase
    if ($LASTEXITCODE -ne 0) { throw "Failed: $rmsCase" }
}
.\run_torch_npu.ps1 -Script npunlock-probe/research/results/rmsnorm-vector/inspect_carrier.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/results/rmsnorm-vector/build_vector_probe.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/results/rmsnorm-vector/vector_probe.py
```

`inspect_carrier.py` 的最新运行列表为小尺寸/非整除尺寸；本目录也保留先前对
32/256/512 行和四维布局的原始编译证据。默认 build 仍为原 `build_rmsnorm.py`。
`parallel-scalar/` 保留第一次并行测试；正式速度结论使用最终 `performance.json`
中的同进程对照，不混用上一天的计时。

支持范围保持不变：显式冻结 SmolLM2 单层、FP16 IO、hidden576、B<=8、S<=64。
混合生成仍为 1 个 NPU 层加 29 个 CPU 层，embedding/最终 norm/logits 在 CPU，
`use_cache=False`。未增加训练、完整 30 层 NPU、INT4/NF4 LLM 或通用模型后端。
后续可优先定位 norm 的输出转换/舍入成本，再推进 KV cache 和多层支持；
通用向量 half 转换须先建立正确性检查。
