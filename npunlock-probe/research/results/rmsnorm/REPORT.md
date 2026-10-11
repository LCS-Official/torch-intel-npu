# SmolLM2 融合 FP32 RMSNorm：实现与实机验证

2026-10-04，Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，驱动
32.0.100.5540，图编译器 8.3。结果：`passed_with_limitations`。

此文及本目录 case JSON 保留串行融合阶段。2026-10-05 的多 tile 完整行并行结果见
`../rmsnorm-vector/REPORT.md`；当前工作区默认实现已采用并行调度。

`FrozenSmolLayer` 已默认使用自定义 SHAVE RMSNorm。FP16 存储和输入输出不变，
平方、补偿求和、均值、开方、倒数与归一化使用真正 FP32。常用尺寸整层只提交
4 个图；W8A16 在同进程对比中加速约 1.80～4.57 倍，仍慢于 CPU。
本轮没有新增依赖或 OpenVINO runtime。

## 实现与边界

- `npunlock/tests/fixtures/npu4000/rmsnorm.c`：固定宽度 576、eps=1e-5，
  Neumaier 补偿累加、已有 FP32 开方和 FP16 边界转换。正常范围使用硬件转换，
  subnormal、signed zero、溢出等边界保留软件转换。编译使用 `-ffp-contract=off`，
  不使用 fast-math。先将归一化结果舍入 FP16，再与 FP16 gamma 相乘并舍入，
  保留现有模型的两个舍入步骤。
- `npunlock-probe/research/build_rmsnorm.py`：复用已有隔离式编译/汇编/链接工具和
  `arithmetic.c`、`resident_activation.c` 的辅助函数；两份辅助源码没有修改。
  最终 `.text` 为 3,952 字节，ELF 为 4,588 字节。
- `npunlock/python/npunlock/npu4000.py`：只新增 `compile_rmsnorm`。
  复用 FP16 Power carrier、已有设备/编译器 fingerprint、CMX 与代码窗口检查，
  只替换代码段。要求一个 SHAVE invocation、没有 DPU、三个 memref 都绑定完整
  连续的 576 元素行。其余已有编译函数与优化前快照逐字一致。
- `torch_npu_smollm.py`：默认融合 norm；保留旧实现供数值和性能对照。
  内核和源码 hash 校验、会话缓存、设备 gamma 广播缓存沿用现有结构。
  暖运行没有显式主机上传、下载或新建会话；统计不包含驱动内部 DMA。

在当前编译器上，512 行 carrier 会生成两个分别覆盖 288 列的 invocation，
不能直接执行跨 576 列的归约。因此每个融合 carrier 限制为 1～256 个完整行；
超过 256 行时在设备侧切片、执行、拼接，不能放宽检查后直接覆盖代码。
单行的编译器规范化 memref 布局也单独检查。

`B*S <= 256` 时两个 norm 各 1 次提交，加 attention、FFN，共 4 次。
最大支持输入 `B8/S64/hidden576` 仍受支持，包含切片/拼接共 24 次。
专用 carrier 的元素数可超过通用算子的 65,536 元素分块限制，但通用后端的限制
没有改变：tensor 上限仍为 1,048,576 元素。

## 数值与回归

全部检查使用最终内核
`1c19b63d8e083e59986304f5d40467746950d5d585399591bb2942db3e3e6a47`。
源码、ELF、汇编、包版本、模型 revision 与逐项结果见 `summary.json`、各 case JSON
和 `after/`。模型仍为原始 SmolLM2-135M-Instruct safetensors，固定 revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`；本轮测试离线运行。

| 检查 | 结果 |
|---|---|
| 单行、17 token 尾部、B4/S64、B8/S64、259 行分块尾部、零、±65504、半精度 subnormal/signed zero，分别使用两个真实 gamma | 16 组全部与旧 NPU norm 的 FP16 位模式一致 |
| CPU FP32 norm 参考 | 原 `rtol=0.002, atol=0.001` 全通过；全部样本合计 17 个 FP16 编码不同，最大绝对误差 0.00012207 |
| 中间 FP16 舍入语义 | 测试中跳过这一步会产生 312,872 个编码差异，融合版保留这一步 |
| 真实层数值、最大尺寸、性能前置数值、profile 前置数值 | 共 17 项，通过原有误差门槛；覆盖层 0、10、20、29 |
| 输入/缓存/快照等 guards | 16 项通过；另验证 carrier 行数 0、257、512 在编译前拒绝 |
| 因果性 | FP16/W8A16 两项通过 |
| 短文本混合生成 | 两个提示 × FP16/W8A16，均与 CPU 参考生成相同的 8 个 token |
| 既有 FrozenTransformer probe | FP16、W8A16、mixed INT8 三项通过 |

±65504 的平方超过 FP16 范围，测试仍得到有限的正确归一化结果，验证该路径没有
把平方或累加降为 FP16。正常 FP16 输入的平方及此处中间量在 FP32 中为 normal 或零，
因此可以使用原生 FP32 运算，避开通用内核为 FP32 subnormal 等边界准备的较慢辅助路径。

整层精度门槛没有放宽：NPU 对相同量化 CPU 参考的 relative L2 < 0.003，
每元素 `abs(error)/(abs(reference)+token_RMS) < 0.01`；固定输入分段 L2 < 0.001。
量化质量对未量化官方参考的 L2 < 0.03。此次最大整层执行 L2 约 0.06784%，
最大 scaled error 约 0.42664%，最大 W8A16 量化质量 L2 约 1.134%。
原先较严格的逐元素 `rtol=0.01, atol=0.005` 在部分深层/近零元素上仍有失败，
对应 mismatch 数保留在结果中。与旧 NPU 位模式相同只说明此次融合保留了行为，
不能推导为整个 NPU 模型与 CPU 逐位相同。

8 个最终 Smol case 合计 6,066 次 NPU 提交，另有旧 Transformer 回归 88 次。
每个 case 结束后 live buffer、隔离 buffer 和缓存会话都为零，context 未失败。
`norm.json` 保留执行时的 demo hash；随后 demo 只增加了源码追溯字段，
其内核 manifest 与其余最终运行一致。没有回填原记录中的源码 hash。

## 同进程性能

每个尺寸同时测融合 FP16、融合 W8A16、旧 W8A16、CPU FP16、CPU FP32。
先检查融合与旧 W8A16 的整层 FP16 位模式相同，再分别预热 3 次，进行 21 轮
轮换顺序计时，取中位数。CPU 为 8 线程；CPU 的 RoPE 和 mask 已准备；
NPU 输入、权重、图和 gamma 已准备。计时不含初次编译、显式传输和加载。

单位为毫秒；B 为 batch，S 为序列长度，hidden 固定 576。

| 输入 | 旧 W8A16 | 融合 W8A16 | 旧/融合 | 融合 FP16 | CPU FP16 | CPU FP32 |
|---|---:|---:|---:|---:|---:|---:|
| B1/S1 | 13.315 | 2.915 | 4.57× | 3.172 | 0.804 | 0.885 |
| B1/S32 | 16.547 | 9.210 | 1.80× | 9.531 | 2.481 | 1.725 |
| B4/S64 | 132.385 | 58.267 | 2.27× | 58.530 | 9.360 | 6.903 |

这三个尺寸的旧实现分别提交 22、26、120 个图，融合版都为 4 个。
表中是本机当前桌面负载下的单层延迟；不能据此推导整网 token/s、能耗优势或
所有尺寸上的性能。CPU 仍然明显更快。

B4/S64 W8A16 分段 profile（另一次运行，每段 9 次中位数）：

| 分段 | 延迟 ms | 图提交 |
|---|---:|---:|
| norm1 | 27.887 | 1 |
| attention | 1.217 | 1 |
| norm2 | 27.890 | 1 |
| FFN | 1.090 | 1 |

两个 norm 占上述分段中位数之和约 96%；这不是整层运行的严格时间分解。
当前 reduction 在单个 SHAVE 上按行串行计算。减少调度开销已经有效，后续主要
收益需要来自按行并行及向量化。

## 保留的失败与原型

- `whole-invocation-boundary-failure.json`、`carrier-*.json`：512 行产生宽度拆分，
  被完整行检查拒绝；没有运行错误布局的 RMSNorm。
- `slow-prototype/`：直接融合通用 FP32 边界辅助函数，精度通过但性能倒退，
  W8A16 B1/S32 约 35.44 ms、B4/S64 约 261.72 ms，未作为默认实现。
- `scalar-fp32-prototype/`：原生 FP32 算术、软件 FP16 转换，B4/S64 约 79.47 ms。
  正常范围采用硬件 FP16 转换后，最终版本降到约 58.27 ms。
- `before/`：优化前源码与原阶段性能/profile/汇总；`after/`：最终源码与内核。

原型计时来自不同运行，只用于说明迭代方向；正式加速倍数仅使用本轮
`performance.json` 中同进程的旧/新对比。

## 复现与下一步

在 `E:\Pytorch_xfire_NGPU` 下串行执行：

```powershell
.\run_torch_npu.ps1 -Script npunlock-probe/research/build_rmsnorm.py
foreach ($rmsCase in 'norm','numeric','boundary','guards','causality','hybrid','performance','profile') {
    .\run_torch_npu.ps1 -Script torch_npu_smollm_demo.py -ScriptArguments '--case',$rmsCase
    if ($LASTEXITCODE -ne 0) { throw "Failed: $rmsCase" }
}
.\run_torch_npu.ps1 -Script torch_npu_transformer_demo.py -ScriptArguments '--case','probe'
```

下一步优先研究保持完整行语义的多 SHAVE 分配及行内向量化，并继续保留旧内核
数值对照。之后再评估多层常驻、KV cache 和更长序列。

当前适配器仍是显式冻结单层、FP16 输入输出、hidden576、B≤8、S≤64。
混合生成是第 0 层 NPU、其余 29 层及 embedding/最终 norm/logits 在 CPU，
`use_cache=False`。未增加整网 NPU、训练、INT4/NF4 LLM、padding、自定义位置
或通用 Hugging Face 模型 `.to('intel_npu')` 支持；有限值 norm 的测试也不构成
通用 NaN/Inf 输入的契约。
