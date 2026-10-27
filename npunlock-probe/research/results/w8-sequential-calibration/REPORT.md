# 完整模型低位质量：顺序校准与分组 INT8 对照

2026-10-15，Windows / Core Ultra 7 258V / Lunar Lake NPU4000 `8086:643e`，
驱动 `1005540`、图编译器8.3、Torch 2.14.1+cpu。使用已有本地 SmolLM2-135M-Instruct。

**本轮没有找到可以启用的低位配置，没有改变生产内核、公开量化器、模型适配器或质量门槛。**
顺序校准没有改善预设的完整模型选择指标；较小分组改善了部分CPU指标，但仍有误差尖峰，
而且当前编译器把分组权重展开成FP16 DPU常量。不能把这种路线称为完整W8压缩执行。
公开发布版保持原状；新增内容为研究脚本和本报告。

## 数据与方法

沿用之前40条提示，用CPU FP16官方模型重新生成每条64个消费token的固定轨迹，不在EOS停止。
按轨迹编号 `i % 5 == 4` 留出8条，其余32条的2,048个token用于拟合。
内部验证为8条轨迹从提示末位置到第64个位置，共290个预测位置。
独立集仍是历史16条回归提示，与40条校准提示不重合；其中bird/garden/cat各检查32个位置，
另有重复前缀33个位置，合计17个案例、142个预测位置，覆盖64位置KV边界。
这些是固定回归案例，不是新的大规模独立语言模型基准。

本轮拟合激活按完整64-token前缀、batch8重新计算，不是复用旧的cached激活文件。
因此不能把本轮统计与旧349/2,560-token实验当成同一份权重的前后比较。
两种新校准方法共享相同token、拟合/内部验证划分、FP16行scale及公开 `calibrate_int8`：

- 原方式：210个投影均使用FP16官方前驱产生的激活。
- 顺序方式：使用前面已经量化的层输出，并按Q/K/V → O → Gate/Up → Down顺序重新采集层内激活。
- 两者均保持CPU FP64 GPTQ求解、1%阻尼、128列块、激活排序、对称 `[-127,127]` 和逐行RTN回退。
- 候选按“失败位置数、最大相对L2、平均相对L2”依次排序；独立集不参与选权重、选group。

质量门槛保持 **logits相对L2 < 3%、next-token KL < 0.02**，参照为完整官方CPU FP16模型。
独立连续测试使用同一串CPU greedy token进行teacher forcing，不能代替自由生成或困惑度评测。
全30层decoder在真实NPU对照中执行；embedding、final norm、lm_head显式在CPU。

## 顺序校准没有胜出

两种方法各准备30层/210个矩阵，拟合合计约186.32秒。
全部矩阵逐行拟合投影SSE不超过普通INT8舍入；累计SSE/RTN分别为1.3867%和1.3851%。
拟合误差相近且很小，不代表完整模型误差很小。

| 内部验证，CPU staged | 通过 / 290 | 平均logits L2 | 最大logits L2 | 最大KL | argmax一致 |
|---|---:|---:|---:|---:|---:|
| 原方式校准W8 | 189 | 2.8033% | 11.6684% | 0.008204 | 285 |
| 顺序校准W8 | 188 | 2.7894% | 10.5096% | 0.011284 | 284 |
| FP16 staged控制 | 290 | 0.3131% | 1.1665% | 0.000137 | 290 |

按预设规则选择原方式。顺序方案的均值/最大值略有改善，但失败位置和argmax差异更多。
冻结选择后再看独立回归：

| 独立验证，CPU staged | 通过 / 142 | 平均logits L2 | 最大logits L2 | 最大KL | argmax一致 |
|---|---:|---:|---:|---:|---:|
| 原方式校准W8 | 117 | 2.2362% | 10.1336% | 0.011365 | 140 |
| 顺序校准W8 | 116 | 2.2068% | 9.0248% | **0.026175** | 141 |
| FP16 staged控制 | 142 | 0.3233% | 1.3854% | 0.000201 | 142 |

顺序方案还出现KL超标，不能以平均L2略降或argmax更接近为由启用。
本轮未对已淘汰的顺序方案重复做完整NPU质量检查。

## 分组INT8：部分质量收益与执行代价

复用原 `quantize_weight`，不做新的激活拟合，只在同一8条内部轨迹选择group。
研究子类没有放宽公开 `FrozenSmolLayer` 的逐通道W8参数契约。
分组CPU模拟使用未提前舍入的 `code × FP16 scale` 权重和FP32 GEMM；乘积本身已与FP64解码逐项核对一致。
它不是编译器展开后FP16常量路径的逐位参考，也与校准W8的FP64 staged GEMM有舍入差异。

| Group size | 内部通过 / 290 | 平均logits L2 | 最大logits L2 | 最大KL |
|---:|---:|---:|---:|---:|
| 64 | 165 | 3.2090% | 11.4549% | 0.019910 |
| 32 | 189 | 2.8541% | 11.6376% | 0.019462 |
| 16 | 221 | 2.5144% | 13.3319% | 0.018928 |
| 8 | **255** | **1.8437%** | 9.5788% | 0.004895 |
| 4 | 189 | 2.4833% | 8.4276% | 0.004632 |

选中的group8在独立CPU集为 **122/142通过**，平均L2 **1.9658%**、最大 **11.6115%**、
最大KL **0.004138**、argmax一致 **140/142**。group变小不保证完整模型质量单调改善。
其210个投影的packed codes+scales为132,710,400字节，约为原FP16矩阵存储的62.5%；
这是准备阶段的存储数字，不是NPU编译图或运行时峰值内存。

真实NPU先检查第0层Q和Down投影，group32/16/8/4共8组：

- 相对独立FP64解码矩乘参考，执行L2最大0.03481%，最大缩放误差0.11492%，通过既有局部0.3% / 1%门槛。
- 全部非逐元素DPU任务为 `amode=0,wmode=0`：FP16激活、FP16权重，没有INT8矩阵证据。
- Q图常量区675,840字节，Down图1,781,760字节；缩小group没有保留压缩常量收益。
- 完整group8模型在首次图检查时，被现有 `Missing INT8 weight DPU tasks` 保护正确拒绝，模型NPU调用为0。
  失败路径shutdown后live/cache/quarantine归零、context健康。本轮保留保护，不伪装W8成功。

所以本轮没有完成“分组低位完整模型NPU质量认证”。如需保留分组压缩，必须研究单轴W8子矩乘
及部分和累加精度/范围/开销，而不是仅绕过检查；现有 `compact_groups` 还限制组数并有FP16部分和风险。

## 真正W8与FP16的完整NPU基线

两个独立进程，各142个预测位置，输入token完全相同。提示逐token送入以复用静态decode图。
FP16控制不含INT8矩阵任务；W8的30份decode图均含 `amode=0,wmode=2` 权重矩阵任务。
额外FP16布局任务不应误计为量化失败。

| 实际NPU执行 | 通过 / 142 | 平均logits L2 | 最大logits L2 | 最大KL | argmax一致 |
|---|---:|---:|---:|---:|---:|
| FP16 | **142** | 0.3481% | 1.2190% | 0.000182 | 142 |
| 原方式校准W8 | 118 | 2.2377% | **10.9058%** | 0.011397 | 140 |

W8数值流程完成，但质量未全部过关；不将脚本 `status=passed` 等同于质量通过。
每进程18,330次NPU调用，其中18,270次模型/热重复图、60次归一化gamma初始化broadcast。
17个案例各做热重复，30层图调用、零新增编译和显式payload复制；输出/KV/position共3,094组SHA256匹配，
旧seed共3,060组SHA256匹配。这里的复制计数不覆盖驱动内部DMA。
所有正式进程最终live、cached、quarantined、session归零，context健康。
本轮不把诊断elapsed当性能benchmark，不声称吞吐、首token或TOPS提高。

## 检查、来源与研究脚本修正

正式有效NPU调用 **36,668**，包括两个完整模型进程36,660次及8次矩阵探针。
另有一次初始W8数值运行18,330次，最后因新脚本误把FP16布局任务也要求为INT8而退出；
修正证据分类后完整重跑，142项数值、17项warm记录和最终资源统计全部相同。
首次运行不重复计入正式有效调用。所有本轮NPU执行合计54,998次，分组模型拒绝测试为0次。

研究脚本另修正FP32/FP64 KL交叉自检：差异样本约2.75e-6，检查容差由2e-6改为1e-5，
拟合函数及辅助代码AST核对不变、已保存权重未重算；模型KL门槛始终为0.02。
已有5项低位CPU单元测试、投影顺序/候选排序自检、7个实际投影解码检查通过。
清理前审计3,172条标量位置记录、权重/源码指纹、候选排序、60份decode图与资源回收，均通过。
不保存完整logits数组，因此清理后不能仅凭报告重算词表级误差，需要重新执行实验。

直接复用已有GPTQ适配器；逐层重放思想参考
[GPTQ opt.py，固定提交2d65066](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/opt.py)。
没有复制该文件代码、没有新增OpenVINO或其他依赖；既有Apache-2.0归属见 [LICENSES/GPTQ.txt](../../../../LICENSES/GPTQ.txt)。

| 指纹 | SHA256 |
|---|---|
| 模型safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| w8_sequential_calibration.py | `d652222c946321f5ddbfb86b8745cebcf306a6d2954c586303e22d6d3b167baa` |
| w8_grouped_quality.py | `8ed4e527e08b540fea353a7abdfe2169f13b63fe9151b9f059c4ece21829dc1f` |
| w8_sequential_npu.py | `6d0cf75d3bb7410d6ef52907400dfc1d4f1221c4376a7ebeecc99ffa60c97338` |

## 下一步

优先研究**按敏感投影保留FP16/异常值的混合精度**，而不是继续只增加相似校准文本、改变阻尼或只保留末尾层。
用内部校准轨迹确定保护的投影和预算，独立数据只验收；NPU图要同时记录INT8与FP16矩阵比例、常量大小和延迟。
这可以在多数权重仍保持逐通道W8压缩的情况下，测试是否能消除少数高敏感路径的尖峰；目前只是待验证方向。
同步补充固定独立文本的NLL/困惑度与自由生成质量；现有logits/KL/argmax回归不能替代这些指标。
INT4/NF4全模型适配放在这项质量与执行证据完善之后。

## 复现与空间

已有 `.conda/npunlock` 环境，仓库根目录，硬件串行运行，`NPUNLOCK_GRAPH_CACHE`保持未设置。
命令会重新生成本轮临时数据；先做fit，后续读取其权重与token清单。

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_sequential_calibration.py -ScriptArguments '--case','fit'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_sequential_calibration.py -ScriptArguments '--case','selection'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_sequential_calibration.py -ScriptArguments '--case','independent-cpu'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_grouped_quality.py -ScriptArguments '--case','cpu'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_grouped_quality.py -ScriptArguments '--case','probe'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_sequential_npu.py -ScriptArguments '--variant','grouped-cpu'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_sequential_npu.py -ScriptArguments '--variant','simultaneous'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_sequential_npu.py -ScriptArguments '--variant','fp16'
# 负例：以下预期报 Missing INT8 weight DPU tasks，不能把非零退出认作模型计算通过。
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_sequential_npu.py -ScriptArguments '--variant','grouped'
```

本轮未下载模型、未安装依赖、未保存图blob/激活/logits/图片或启用磁盘图缓存。
报告验收后已删除12份临时权重/JSON，共197,329,937字节（约188.19MiB），
其中两份压缩权重共193,801,372字节。
保留上述三个研究脚本、当前运行内核与本报告；公开发布快照及GitHub main未在本轮修改。
