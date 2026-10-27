# Torch → Intel NPU：FP16 / FP32 与最小设备后端

S42：新增原生SHAVE在线分块Attention与重计算Q/K/V梯度，可通过标准sdpa_kernel(FLASH_ATTENTION)显式选择，默认仍math；接入strided NestedTensor推理、默认Encoder、MHA/LayerNorm/复制和Nested SDPA。新增两枚有源码ELF，原64组及DLL/PYD不变。128×256测例省86%–88% Tensor峰值，但含编译冷测试约慢3倍；不宣称加速、jagged/Nested训练或通用FlashAttention支持。接口、实际范围与证据见[S42报告](npunlock-probe/research/results/flash-nested/REPORT.md)。

S41：扩展原版HF Llama/GPT-2的有限模型迁移入口，补位置arange、整数cumsum/isin/标签padding、空KV拼接/梯度、RoPE部分视图取负及native MHA/Encoder空输入。共享参数按报告使用PyTorch交换转换选项；核心依赖、原生组件及ELF不变。具体推理/生成/训练对照和未认证范围见[S41报告](npunlock-probe/research/results/attention-migration/REPORT.md)，不表示任意HF模型或FlashAttention已支持。

S40：可直接使用torch.amp.GradScaler("intel_npu")，通过仅针对该后端的构造适配保留提前导入别名和原CPU构造；梯度计算仍在NPU，缩放控制在CPU。支持iterable/dtype语义、Tensor新scale及状态恢复，旧专用入口保留。无新依赖或原生构建；单NPU/FP32主参数/普通优化器等边界见[S40报告](npunlock-probe/research/results/amp-standard/REPORT.md)。

S39：补有界空batch/序列/头和零QK/V特征SDPA，保留mask零梯度/None及Dropout RNG语义；新增Tensor概率Bernoulli与NPU随机键排序randperm、Generator/out及重放。无需新依赖、原生构建或ELF；实际dtype/容量范围、验证和剩余限制见[S39报告](npunlock-probe/research/results/attention-rng-boundaries/REPORT.md)。

S38：九dtype按1GiB字节预算分配；FP16/BF16/INT16最大536,870,912元素，BOOL/INT8/UINT8最大1,073,741,824。密集普通MaxPool改用内联SHAVE窗口和有序几何反向；配对输出直接按dtype写回。需要更新PYD和内核伴随文件，其他算子的专用工作区、存储1GiB及同步提交限制保留，具体最大形状与冷热吞吐见[S38报告](npunlock-probe/research/results/storage-shape/REPORT.md)。

S37：大平面选择反向改用原始UINT32元数据，普通MaxPool与浮点极值不再受旧2²⁴单平面保护限制，改用当前dtype存储检查。新增selection-backward-u32 ELF，小尺寸路径及原62枚ELF保持不变；无需新依赖或DLL/PYD重建。实际大平面完整梯度及跨包累加见[S37报告](npunlock-probe/research/results/selection-capacity/REPORT.md)；单包、其他算子前向和吞吐边界仍需分别检查。

S36：普通MaxPool1d/2d的空间输出保护扩大到16,777,216，解除旧总采样点限制；跨通道包与按平面组装的反向保留C内核单包/元数据保护。极值长轴索引由设备INT64恢复，不可微索引不再分配无用零梯度，窗口读取优先页对齐绑定。软件边界、真实数值与训练证据见[S36报告](npunlock-probe/research/results/pool-capacity/REPORT.md)；无新依赖或C/ELF/DLL/PYD构建。

2026-10-24 S35：普通Conv2d以262,144个窗口/fold元素为路径选择预算，更早使用已有流式算法；FP32结果按对齐范围逐块写回，避免所有结果块长期堆积。实测3×3卷积后端分配峰值FP32为7.00→1.84MiB、FP16为7.35→1.63MiB，小窗口保持原路径。完整梯度、特殊几何与两步训练及剩余界限见[S35报告](npunlock-probe/research/results/conv-capacity/REPORT.md)。无新依赖、SHAVE ELF或DLL/PYD重建。

2026-10-24 S34：Embedding超出原完整索引包时复用byte gather/selection backward，并按特征列拆分；sort/topk输入输出总量保护扩到16,777,216元素，原计算包和一般索引反向保护不变。无需新增依赖或重编已有S32组件；软件界限与实际验收分开记录，见[S34报告](npunlock-probe/research/results/index-capacity/REPORT.md)。卷积/池化及次序统计等专用界限仍需单独处理。

2026-10-24 S33：冻结Linear/Transformer/SmolLM与矩形GEMM登记调用期弱引用重建工厂，可参与LRU；缓存增加512MiB图blob预算，超预算可重建图临时执行。prepared chain继续保留图对象，已覆盖四层decode和多路径重建。2.69亿参数网络通过两步FP32 AdamW，状态4.00GiB/后端峰值4.25GiB，抽样更新对照通过；耗时约35分钟，未认证训练加速或14GiB整网。已有S32组件无需构建或新依赖；下面S32“冻结图固定保留”由[S33报告](npunlock-probe/research/results/frozen-capacity/REPORT.md)与当前矩阵替代。

2026-10-24 S32：单allocation上限1GiB，FP16/BF16逻辑Tensor最多512MiB；普通算子图采用1536项LRU，已准备的执行计划持有图直到最后使用者释放。自定义冻结图目前仍pinned，满缓存会明确拒绝。重建session DLL与Torch PYD，无新SHAVE ELF/依赖；以下旧容量与图保留策略以[S32报告](npunlock-probe/research/results/graph-capacity/REPORT.md)及当前兼容性矩阵为准。

2026-10-24 S31：硬标签CE/NLL按完整行分块并重建反向mask，支持超过旧单packet的batch；常见稠密算子容量检查扩大，百万行NLL、超旧输出界限Conv2d及encoder FFN等验收见[报告](npunlock-probe/research/results/operator-capacity/REPORT.md)。保留类别/softmax行宽、浮点索引元数据和图缓存保护；复用原62枚ELF，无新增依赖或构建。

2026-10-23 S30：按本机实用范围扩展容量。单allocation为512MiB，rank64不变；显式page-aligned范围绑定复用不同偏移，逐元素图按元素数复用，分块运算直接绑定输入/最终输出或逐块写回结果。旧完整缓冲区入口保持保护，图缓存仍为1536。测试与实际模型规模见[报告](npunlock-probe/research/results/memory-scale/REPORT.md)；约14GiB训练尚未实测认证，下面S26–S29的历史容量数字以S30为准。须重建session DLL及Torch PYD，不新增SHAVE内核或依赖。

2026-10-23 S29：接通加权NLL/硬标签CE、可微软标签与权重CE、混合FP16/FP32及广播MSE、原生forward/backward/out接口，含空维度、版本保护和三步组合训练验收，见[报告](npunlock-probe/research/results/loss-extended/REPORT.md)。复用62枚resident ELF，不新增依赖或重建宿主组件；硬标签packet和一阶梯度等边界仍保留。下文历史的无权重/无软标签/同shape MSE限制以S29及当前兼容性矩阵为准。

2026-10-23 S28：SDPA扩至非空rank2–64及不同rank广播，接通GQA/MQA与共享K/V梯度；attention矩阵改用当前存储上限检查，旧百万元素限制的实际跨越及训练验收见[报告](npunlock-probe/research/results/sdpa-extended/REPORT.md)。复用62枚既有resident ELF，不新增依赖或重建DLL/PYD；单行宽、因果query长度及图缓存仍有边界。

2026-10-23 S27：长K矩乘用新SHAVE滚动补偿累加，避免完整partial堆叠；matmul扩至rank1–64及有界切片/广播只读视图，修复bmm空维度/零K与完整密集梯度目的的DMA复制。65539维Linear和5维matmul组合训练的FP16/FP32独立对照见[报告](npunlock-probe/research/results/matmul-workspace/REPORT.md)。新增一枚ELF，总数62；不重建宿主DLL/PYD、不新增依赖，S26容量及图缓存边界保留。

2026-10-23 S26：存储上限256MiB、rank0–64；普通数学/归约/Linear与softmax行分块扩展，1704万参数Linear独立训练验收见[报告](npunlock-probe/research/results/capacity-model/REPORT.md)。旧环境需先重建session DLL再重建PYD；不新增SHAVE ELF或pip依赖，旧算子工作区和图缓存限制保留。

2026-10-23 S25：普通Torch存储扩到64MiB/rank8；新增整数/BF16转换与部分运算、动态布尔/高级索引、非重叠NPU目的写入、公开gather/scatter与一阶梯度、混合dtype拼接及更多数学函数。210万参数网络完成FP16/FP32各两步独立CPU/NPU训练对照。见[S25报告](npunlock-probe/research/results/compat-foundation/REPORT.md)和[当前矩阵](COMPATIBILITY.md)。旧4MiB/百万元素描述属于历史阶段；冻结DPU、排序/统计等独立边界仍保留。需重建设备PYD；新增SHAVE ELF随源码提供，无新pip依赖。

2026-10-22 S24：继续扩展五维/空Tensor、整数/BOOL排序和次序统计，以及3D卷积/插值/归一化/池化、抗锯齿、Unpool、padding/shuffle、Unfold/Fold及动态网格采样。见[完整范围与回归](npunlock-probe/research/results/vision-sort-extended/REPORT.md)。旧本地版本须重编设备PYD，新增SHAVE ELF随源码提供；无新pip依赖。只有验收矩阵内的有限支持，不能宣称全部视觉/排序配置或通用CUDA已经完成。

2026-10-22 S23：本地新增有界1D/2D插值、GroupNorm/InstanceNorm、ConvTranspose1d/2d、浮点稳定sort/argsort、max/min值与索引和更大k路径，含一阶反向及常见原地激活组合。见[报告](npunlock-probe/research/results/vision-sort/REPORT.md)，其中保留含离散排序的整网严格轨迹边界及旧CPU显式scale前后向不一致配置的拒绝策略。这些旧缺口现由S24扩展；不是通用CUDA完成声明，无新依赖或C++重建。

2026-10-21 S22：本地新增MaxPool1d/2d及空间INT64索引/一阶梯度、非view非叶张量的原地ReLU、k<=128的有界topk及长行候选合并。标准Conv2d/BN2d/Linear后的原地ReLU保留版本保护，见[报告](npunlock-probe/research/results/pool-topk/REPORT.md)；下方历史“无MaxPool/原地ReLU/topk”以[当前清单](COMPATIBILITY.md)为准。未增加OpenVINO依赖、C++重建或性能加速声明。

2026-10-21 S21：本地新增argmax/argmin、可微amax/amin、有界0/1/2/正负无穷向量范数及标准非foreach梯度范数裁剪、浮点isnan/isinf/isfinite；见[报告](npunlock-probe/research/results/reduction-norm/REPORT.md)。两枚有源码的SHAVE行极值ELF处理首个tie/NaN与原始INT64索引，长行自动分块。通用容量、dtype、异步和CUDA生态边界保持不变；下方历史缺口以[当前清单](COMPATIBILITY.md)为准。

2026-10-21 S20：本地新增有界gather/scatter_add/take_along_dim、平均/自适应平均池化、可微极值/限幅与Hardtanh/ReLU6，见[索引/池化报告](npunlock-probe/research/results/index-pool/REPORT.md)。索引保持设备INT64，新增SHAVE规范化及初始值累加语义；后文“不支持非全局池化/maximum无梯度”等历史限制以当前[兼容性矩阵](COMPATIBILITY.md)为准。MaxPool和无限制索引仍未实现。

2026-10-21 S19：本地新增分组/深度Conv2d、Conv1d、任意合法轴softmax/log_softmax、空间/序列硬标签损失、label smoothing与rsub梯度，见[迁移接口报告](npunlock-probe/research/results/portability/REPORT.md)。本页按时间保留历史记录，下面旧的groups=1、最后一轴、无AMP等限制以当前[兼容性矩阵](COMPATIBILITY.md)为准。分组仍串行复用现有内核，无新增依赖或宿主重建。

2026-10-20 S18：AMP卷积的CPU Half差异已完成分批舍入归因；六种局部几何、原SiLU及三卷积残差网络的四步独立FP32累加控制和checkpoint通过，见[报告](npunlock-probe/research/results/amp-conv/REPORT.md)。原生CPU Half严格差异仍保留，不能记为标准CPU轨迹全部通过；仅修复空scaler checkpoint的异常处理，现有计算内核不变。

2026-10-20 S17：原生FP16 autocast及NPU专用torch.intel_npu.amp.GradScaler已接通，FP32主参数、设备梯度检查/反缩放、溢出跳步和MLP恢复见[AMP报告](npunlock-probe/research/results/amp/REPORT.md)。卷积严格FP32梯度仍有数值边界；标准torch.amp.GradScaler入口后由S40接通，BF16或任意模型AMP仍未认证。CPU只保存缩放/增长控制状态，更新需重编PYD；下方按时间保留的“无AMP支持”描述以本轮有限范围为准。

2026-10-20 S16：新增原生C++ torch.Generator、SHAVE INT64 randint/random_及FP16/FP32 multinomial；支持有界随机token训练、采样与状态恢复，详见[离散RNG报告](npunlock-probe/research/results/discrete-rng/REPORT.md)。更新旧本地构建需运行build_torch_npu_device.ps1重编PYD；旧24字节RNG状态继续兼容。randperm及有限AMP后由S39/S17补子集；通用AMP及低位整网质量仍未补齐。

2026-10-20本地兼容性更新：公开exp/log/tanh/sigmoid/SiLU/sqrt/rsqrt的一阶梯度与标准RMSNorm已接通，新增门控网络和有限标准Decoder迁移验收，见[数学/模型报告](npunlock-probe/research/results/unary-math/REPORT.md)。下方“sqrt无梯度/exp等未公开”等历史状态以当前[兼容性矩阵](COMPATIBILITY.md)为准；通用CUDA运行时、AMP和低位整网质量尚未解决。

2026-10-19 本地RNG更新：已有SHAVE Philox、FP16/FP32 rand/randn、种子/状态与普通Dropout一阶梯度，见[RNG/Dropout报告](npunlock-probe/research/results/rng-dropout/REPORT.md)。本页下方按时间保留历史实现说明；其中“随机数未实现/无RNG状态”等旧记录不代表当前状态，当前支持边界以[兼容性矩阵](COMPATIBILITY.md)为准。

2026-10-10 保留策略更新：按用户要求保留源码、内核运行/构建所需文件和文字报告，清理历史实验数组、编译图、日志与原始计时文件。
历史报告中的原始文件链接可能已失效；其中结论是清理前的验收记录，重新复算需要重跑对应实验。
最新结果见 [CPU/GPU快路径与NPU图缓存报告](npunlock-probe/research/results/model-cross-device-followup/REPORT.md)，空间明细见 [工作区空间记录](WORKSPACE_STORAGE.md)。

2026-10-11 优化方向确定为新进程冷首token的准备链路，已将可选图缓存写入改为免解压ZIP存储。
相同blob的三次新进程格式对照，冷首token中位2.637→1.841秒；正式入口另测两次为2.079–2.351秒。
30份prefill图由160.75增至216.88MiB，仍受512MiB总上限约束；缓存默认关闭，保留缓存才有下次启动的命中收益。
19项缓存检查及11进程逐位输出对照通过，损坏条目恢复正常；本轮临时图/数组已按约定清理。
只测B1/24-token提示词的首token，未重新测试完整32-token生成；以下后续已接入缓存命中时避免完整IR重建。
详情见 [冷首token诊断与优化报告](npunlock-probe/research/results/startup-first-token/REPORT.md)。

2026-10-11 后续已接入FP16整层prefill的提前缓存命中，省去完整IR及大权重序列化。
相同30份编译blob的交替新进程对照，首token中位2.009→0.967秒，降低51.8%；脚本入口到首token中位6.795→5.491秒。
18个完整30层进程hidden/logits逐位一致，38项缓存检查、21项既有保护检查和8项边界硬件对照通过。
权重/RoPE/形状/源码/内核/profile继续决定缓存键，I/O契约与ELF/DPU证明继续校验；损坏1张图只重编译该图。
仅缓存命中的FP16整层prefill使用捷径，首次编译仍约28.6秒；默认关闭磁盘缓存，decode、W8A16和研究替换图沿用原路径。
本轮缓存/数组/原始计时按约定验收后清理；完整生成和训练没有复测。详情见 [跳过IR重建报告](npunlock-probe/research/results/startup-ir-cache/REPORT.md)。

2026-10-03，Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，
驱动 32.0.100.5540，图编译器 8.3。

本地修改版 npunlock 已支持 FP16 / FP32 自定义 SHAVE 算术、精度转换，
以及基于这些内核的归约和二维矩阵乘法。普通 Torch 多算子图可通过
torch.compile 的 npunlock_npu 后端执行；不支持的操作明确报错。
torch_npu_minimal 的 Tensor 存储仍在 CPU，并未实现 NPU Tensor 设备。
新增 torch_npu_runtime 已实现同一 Level Zero 上下文的图会话和共享缓冲区复用，
可将上一图的输出直接绑定为下一图的输入，作为设备后端的执行层基础。
torch_intel_npu 进一步实现真正的 PrivateUse1 Tensor 设备、C++ allocator、
设备钩子、CPU/NPU 复制与普通 ATen 加减乘除、取负、绝对值和符号函数，
并支持设备 FP16/FP32 转换、广播与标量算术、sum/mean 归约及二维矩阵乘法。
Tensor 与 Storage 均标记为 intel_npu:0，数据直接存放在 Level Zero 共享分配中。
目前这是支持有限一阶训练与 SGD 的同步设备后端，完整模型覆盖与 GPU 协同尚未实现。

2026-10-04 新增直接 FP32 SHAVE GEMM 实验。串行版本支持边长 2..64 的完整方阵；
并行版本用单 tile 两个 SHAVE 行任务，或四个 tile 各一个 SHAVE 行任务。
四 tile 版本在 NPU 上复制完整 A/B、输出互不重叠的行，再用 NPU 归约合并，
无需展开 M*K*N 乘积张量。普通 FP32 运算内联，特殊值和次正规数保留原精确路径。
数值及 FP32/FP16 各 3 步 Linear 训练检查与旧路径逐位一致。
小尺寸单 tile 更快，64x64 四 tile 已测得加速；32x32 暂未跑赢旧路径。
上述小方阵版本保留为实验；原生 FP16 输入内核尚未实现。
运行 `run_torch_npu.ps1 -Script torch_npu_gemm_demo.py -ScriptArguments '--parallel'`。
本轮结果见 `npunlock-probe/research/results/gemm-parallel/REPORT.md`，
原串行基线见 `npunlock-probe/research/results/gemm/REPORT.md`。

随后新增有界矩形 GEMM，M/N/K 均可在 1..256 内，M/N 分块并保留完整 K，
支持尾块和完整密集转置输入。38 组数值/普通 API 检查，以及两种非方阵 MLP、
128x128 Linear 的 FP32/FP16 各 3 步训练通过；输出、梯度和 SGD 更新与旧路径逐位一致。
同轮 FP32 128x128 从约 510 ms 降为 193 ms，256x256 从约 3963 ms 降为 1537 ms。
这两种完整方阵已接入普通 `a @ b` / `torch.mm` / `out=` 的默认设备路径，
FP16 仍先在 NPU 上转换为 FP32 计算，再一次转回 FP16。其他形状沿用原路径，
一般矩形实验可显式调用 `torch_npu_rect_gemm.matrix_fp32`（返回 FP32）。
运行 `run_torch_npu.ps1 -Script torch_npu_rect_gemm_demo.py`；详细结果及未控制的桌面负载限制
见 `npunlock-probe/research/results/gemm-rect/REPORT.md`。

继续优化后，矩形内核已支持四个 tile 各两个 SHAVE，共八个完整输入副本。
单次矩阵调用内复用补零和复制后的 A/B，以 DMA 选择每行的唯一结果，省去完整容量归约。
同进程交错测试中，128x128 从上一版四路约 191 ms 降到 82 ms，
256x256 从约 1534 ms 降到 599 ms；普通 API 的这两个尺寸默认使用八路。
FP16/FP32 数值、转置、尾块和三种各 3 步训练配置继续与旧 NPU 路径逐位一致。
完整八路缓冲区先通过已有 DMA 转置/截取，再使用有界 Gather，未扩大 Gather 源限制。
显式实验可用 `matrix_fp32(a,b,replicas=4)` 对比四路；默认八路，返回 FP32。
这仍是 SHAVE FP32 路线，CPU 在这些尺寸上仍快得多。新记录见
`npunlock-probe/research/results/gemm-eight/REPORT.md`；桌面负载未控制。

2026-10-05 增加可复现的矩乘吞吐测试：
`run_torch_npu.ps1 -Script torch_npu_throughput_benchmark.py`。
分别测量当前 SHAVE FP32、DPU FP16、W8A16 和真正的 W8A8，保存原始样本、
数值验证、编译图精度字段及源码指纹。1024 方阵的单图预分配绑定本轮测得
FP16 3.19 TFLOPS、W8A8 3.93 TOPS；包含输出分配的图接口约为 2.18 TFLOPS、
2.51 TOPS。当前 FP32 SHAVE 单图最大约 0.064 GFLOPS。均是包含图 DMA 与同步的
有效吞吐，不是芯片峰值；完整方法及所有形状见
`npunlock-probe/research/results/throughput-20261005/REPORT.md`。

2026-10-06 增加 DPU 主机分段、device profiling、固定地址 command list/fence
复用实验，以及独立环境中的 OpenVINO 2026.4.1 对照。对相同编译 blob 交替执行，
OpenVINO 比当前标准提交路径减少约 19%～37% 延迟；诊断 replay 路径又比
OpenVINO 减少约 7%～21%。本轮 1024 方阵 U8 输入、驱动 8.3 编译器的同轮结果
为 standard 3.76 TOPS、OpenVINO 4.80 TOPS、replay 5.31 TOPS，均为真实 W8A8。
单独的 4/6 tile 对照中，6 tile 对大矩乘约减少 9%～10% 延迟，小矩乘收益不一致。
在上述诊断阶段，正式 runtime 和普通 Torch API 尚未接入复用逻辑；未新增
OpenVINO 运行时依赖。原始证据、8716 次成功执行、方法与复现命令见
`npunlock-probe/research/results/dpu-diagnostic/REPORT.md`。

2026-10-06 后续已将 command list/fence 对象复用接入正式 session DLL，普通
Torch API 和已有模型适配器自动使用；每次重新记录当前 Tensor 绑定，完成后
清空命令并重置 fence，避免持有已释放输出地址。相同旧 blob、驻留输入的
8 组交替对照延迟降低 23%～33%；1024 方阵预量化 U8 从 0.6248 ms 降到
0.4348 ms，约 4.94 有效 INT8 TOPS。四层 SmolLM2-135M 的独立进程 ABBA
对照中，batch 4 的 prefill/decode 降低约 7%～22%，batch 1 收益较小且波动明显。
104 项输出/KV/位置数组跨进程逐位一致，旧输出与 seed KV 未被后续调用覆盖。
23 项会话、33 项设备、11 项 FP16/FP32 训练检查及原生故障模拟通过。
本次没有改动矩阵内核或模型适配器，也未新增 OpenVINO runtime 依赖。
完整范围、原始样本、构建指纹与复现命令见
`npunlock-probe/research/results/session-reuse/REPORT.md`。

2026-10-06 后续主机分段实测发现，四层暖 decode 的驱动共享内存分配/释放
约占调用及结果清理总时间的 25%～30%。C++ Tensor 分配器现已默认缓存
最后一个 Storage/view 引用释放后的相同字节大小内存块，初始上限 16 MiB / 64 块；
2026-10-08 已调整为 16 MiB / 128 块，详见下方 30 层缓存对照。
暖循环每组 1008 次逻辑 Tensor 分配对应的驱动分配从 1008 次降为 0～2 次；
逐次输出与 KV 仍独立所有，104 项跨版本数组逐位一致。
可调用 `torch.intel_npu.empty_cache()` 或 `torch_intel_npu.empty_cache()` 释放闲置块，
shutdown 自动清空缓存；执行失败时缓存也遵守 native 上下文隔离规则。
`buffer_allocations` 继续计逻辑分配，新增 `driver_allocations`、`allocation_reuses`、
`cached_buffer_blocks`、`cached_buffer_bytes`；`live_buffer_bytes` 不含闲置缓存。
本轮相同四层模型的 ABBA 样本、分段数据和检查见
`npunlock-probe/research/results/warm-allocation/REPORT.md`。

2026-10-06 后续完成四层 SmolLM 融合图的隔离 DPU / SHAVE task profiling。
FP16 / W8A16、batch 1 / 4 的 32 张图共取得 96 份独立任务采样；
RMSNorm 占已记录设备时间跨度约 82%～87%（prefill）、35%～42%（decode），
DPU 任务约占 4%～5%。这些区间统计已扣除并行任务重叠；profiling 本身会改变图调度。
本机 NPU40xx 驱动栈不支持直接 DMA profiling，未记录区段不能当作纯 DMA 时间。
104 项正常图 / profiling 图输出与 KV 数组逐位一致，320 次调用完成后资源归零。
正式 runtime、Torch 扩展、内核和模型实现指纹保持原值，未新增 OpenVINO runtime 依赖。
下一项优先优化目标为 SHAVE RMSNorm，decode 同时关注 KV 更新相关 Select。
完整结果、离线核对、时间线及复现命令见
`npunlock-probe/research/results/smollm-device-profile/REPORT.md`。

2026-10-06 后续已优化并接入默认 SHAVE RMSNorm：参考 MIT 许可的
Maratyszcza/FP16 实现四路 SIMD 转换，保留原有 FP32 补偿归约顺序及两次 FP16 舍入。
256 / 512 行 RMSNorm 的普通执行耗时下降约 12% / 13%；
相同四层 FP16 / W8A16 模型在两个进程中测得 prefill 耗时下降约 5%～9%。
decode 收益较小且有波动。104 项候选数值检查、四层输出/KV 逐位对照及
74 项正式接入回归通过。优化写在内核中，现有模型调用自动使用，未新增 runtime 依赖。
源码来源、许可、未采用的归约候选、完整样本和复现见
`npunlock-probe/research/results/rmsnorm-reduce/REPORT.md`。

2026-10-06 后续已修复分配缓存的形状切换问题：空间不足时淘汰最早归还的闲置块，
仍保持 16 MiB、64 块上限及 Storage/view 生命周期规则。构造的缓存饱和场景中，
每 31 次四层 decode 的驱动分配从 403 次降为 0；四个进程的输出/KV 全部逐位一致。
普通路径和固定 32 token 的四层 NPU＋26 层 CPU 混合生成波动较大，本轮未证明稳定加速。
设备、训练和 KV 回归通过。ScatterUpdate 替代 KV Select 的候选跑通，
但一组变慢约 10%，尚未接入默认模型。完整样本、精度限制和复现见
`npunlock-probe/research/results/decode-cache/REPORT.md`。

2026-10-06 后续完成 W8A16 精度诊断：鸟的故事提示在四层 NPU＋26 层 CPU 下的
logits L2 为 4.88%，CPU 量化分段参考也达到 4.40%；逐层 NPU 执行误差仍在原容差内。
连续 32 个相同输入 token 的 decode 中，W8A16 有四个采样位置超过原 3% 门槛，
FP16 全部通过，两种模式的 32 次下一 token argmax 均与 CPU 相同。
简单最小二乘 scale 候选仍未达标且部分指标变差，因此默认量化器和内核保持原样。
原始数组、DPU 舍入语义对照、候选与复现见
`npunlock-probe/research/results/w8-quality/REPORT.md`。

2026-10-07 增加显式 CPU 激活感知 W8 校准 `calibrate_int8`，并允许冻结 SmolLM
构造器接收预量化权重。参考 Apache-2.0 许可的 GPTQ 误差反馈方法，保留原 FP16
逐行 scale、INT8 存储大小和 DPU 执行路径；默认量化方式保持原样。
12 条校准文本之外的 16 条提示全部通过原质量门槛，鸟提示 logits L2 从
4.88% 降至 1.98%。三组各 32 个相同输入 token 的 decode 也全部通过，
最差 L2 为 2.92%；原 W8 有 12 个位置超标。仅验证前四层 NPU＋26 层 CPU
的 SmolLM2-135M 冻结推理，本轮没有吞吐测试。公开 API、限制和完整证据见
`npunlock-probe/research/results/w8-calibration/REPORT.md`。

2026-10-07 后续完成当前版本的性能复测和热点核对。原 W8、校准 W8、FP16 在
两个进程中做同进程交错计时，校准并未带来稳定加速，B4 decode 还测得约 2% 的
小幅变慢。预热计时内驱动分配为零；校准 W8 的设备 profiling 中，RMSNorm 占
已记录跨度约 83%～86%（prefill）、34%～39%（decode），DPU 约占 4%～6%。
decode 的 Select 约占 13%～23%。隔离的 O3 RMSNorm 候选数值通过，但独立内核
及八组模型计时均更慢，因此未采用；正式内核与接口保持原样。
15,242 次调用、876 项数组逐位核对通过，数据和复现见
`npunlock-probe/research/results/w8-performance/REPORT.md`。

2026-10-07 后续已将 SHAVE RMSNorm 的输出转换范围判断改为四路 SIMD 判断，
并接入默认 C / ELF。FP32 补偿归约、两次 FP16 舍入和特殊值回退规则保持原样。
256 / 512 行的独立 RMSNorm 耗时下降约 18% / 15%；相同四层 FP16 / 校准 W8A16
模块在两个独立进程中测得 prefill 耗时下降约 4%～12%，decode 收益较小且有波动。
208 项候选数值检查、74 项接入回归、16 条独立提示和 96 个连续 decode 位置通过；
接入前后的 811 项权重、输入与推理数组逐位一致。现有调用在新进程中自动使用。
未新增 runtime 依赖，完整样本、指纹和复现见
`npunlock-probe/research/results/rmsnorm-encode/REPORT.md`。

2026-10-07 后续完成 decode KV 更新的隔离对照：INT64 / INT32 Gather、
扁平 ScatterUpdate 均通过四层连续 decode 逐位检查；ScatterND 候选被当前编译器拒绝。
INT64 Gather 在 B4 校准 W8 下两轮测得约 8% / 12% 的中位耗时下降，但 B1 回退，
B4 FP16 的逐轮收益不稳定，因此未接入默认模型。设备记录显示 Select 工作减少，
同时每层 SHAVE 任务从 106 增至 120，后续继续检查布局搬运和调度成本。
8,944 项候选数组、6,708 项跨进程 baseline 数组、102 项 profiling 数组逐位一致；
14,496 次成功调用、原始任务记录和复现见
`npunlock-probe/research/results/kv-gather/REPORT.md`。

2026-10-07 后续跑通了专用 SHAVE FP16 Select：直接向量复制 KV 行、选择注意力遮罩，
不改变模型计算精度或增加任务数量。两轮四层 decode 的四组中位耗时均下降；
101 轮确认中 B1 / B4 的 FP16 与校准 W8A16 耗时下降约 3%～8%。
B4 校准 W8A16 的设备 Select 累计时间从约 0.572 ms 降至 0.309 ms。
11,180 项候选模型数组逐位一致；独立测试验证全部 FP16 位模式和 KV 写入特殊值的位保真性。
此前 grouped GQA、K/V 合并 Select 和通用复制内核的收益不足，均未采用。
专用版本目前仍为隔离研究候选，默认模型尚未接入；需进一步完成其他 batch 和完整模型回归。
22,720 次成功调用、原始任务记录和复现见
`npunlock-probe/research/results/kv-group/REPORT.md`。

2026-10-07 后续已将专用 Select 接入默认 `FrozenSmolLayer` / `FrozenSmolStack` decode 路径，
现有调用在新 Python 进程中自动使用。正式 loader 校验 C / ELF，graph cache 纳入 Select 指纹，
布局检查在 Python `-O` 下仍有效；设备审计同时支持 RMSNorm 与 Select 的复合补丁。
B1～B8、四个 KV 边界及全 30 层 FP16 / W8A16 的 32 步连续 decode 均与旧实现逐位一致，
共 6,710 项默认路径数组；74 项模型回归、16 项既有离线测试及 70 项 Select 保护检查通过。
重新生成的四层校准 profile 保持原有质量门槛，811 项历史权重/输入/推理数组逐位相同。
全层测试的 30 个 decoder layers 使用 NPU，embedding / final norm / LM head 使用显式 CPU 参考。
未经全层校准的 W8 logits 最大相对 L2 约 10.34%，后续需单独扩展全层校准；新旧实现的该误差相同。
8,043 次成功调用、完整模型回归范围和复现见
`npunlock-probe/research/results/kv-select-integration/REPORT.md`。

2026-10-07 后续将 W8 激活校准扩展到全部 30 层、210 个矩阵，复用公开 `calibrate_int8`。
12 条提示共 349 个 token；校准样本逐行重建 SSE 均不增加，前四层权重保持逐位一致。
16 条独立提示加 2 项 prefill 对照中，校准 W8 最大 logits 相对 L2 从 10.28% 降至 3.27%，17/18 通过；
129 个连续生成位置中，平均误差由 6.10% 降至 3.45%，但最差仍为 13.46%，仅 76/129 通过原 3% 门槛。
CPU 使用同一份校准权重也复现大误差；90 项局部 NPU 执行检查通过，主要问题来自量化后的模型。
完整 map 已可通过现有 `packed_weights` 接口使用，但候选尚不推荐默认，下一步需增加包含回答/续写的代表性校准样本。
52,830 次成功 NPU 调用、FP16 对照、质量复算和 CPU 诊断见
`npunlock-probe/research/results/w8-full-calibration/REPORT.md`。

2026-10-07 后续将全层 W8 校准扩充为 40 条、2,560 个 token 的 CPU 提示/续写轨迹。
连续测试平均 logits 相对 L2 从上一轮 3.45% 降为 2.51%，通过数从 76/129 增至 100/129，next-token 一致 129/129。
最差误差仍为 11.84%，letter 独立提示退化至 5.08%，因此候选仍不推荐默认启用，原质量门槛保持不变。
CPU 同权重复现大误差；末尾 1/2/4/8 层保留 FP16 的 CPU 诊断也未稳定解决，后续需检查逐层漂移与校准求解约束。
52,830 次 NPU 调用、90 项局部执行检查正常，461 项历史 CPU/FP16/原始 W8 控制数组逐位一致。
代码、完整失败范围、180 份图证据和复现见 `npunlock-probe/research/results/w8-continuation-calibration/REPORT.md`。

2026-10-07 后续检查全部 210 个矩阵的权重偏移：2,560-token 校准的权重平方误差为普通 INT8 舍入的 2.2868 倍。
在原 40 条校准轨迹内部按 32/8 划分，比较 GPTQ 阻尼 1% / 10% / 30% / 100%。
较强阻尼降低权重误差，但 290 个验证位置的通过数从 197 降至 171 / 156 / 145，平均 logits 误差增加。
10% 改善最差值，但按预先规定的失败位置数优先排序仍选原 1%，本轮未修改公开校准器或新增 NPU 推理。
1,160 项指标、1,680 项权重数组及 26 份来源快照审计通过；现有全层 W8 候选仍不推荐默认。
复现见 `npunlock-probe/research/results/w8-regularized-calibration/REPORT.md`。

2026-10-08 后续刷新当前 30 层 FP16 decoder 的普通基线、主机计时及真实任务剖析。
RMSNorm 占 prefill 任务时间跨度约 74%–80%，decode 约 31%–33%；这些比例不是硬件利用率。
四行交错的 FP32 补偿累加候选通过 112 项数值检查，但 4–512 行独立调用慢约 16%–25%，因此未接入默认内核。
30 层 decode 每步还记录到 28 次驱动分配，分配约 0.7–0.8 ms；下一步适合单独验证 64 块缓存上限的影响。
57,960 条任务记录、1,272 次数组逐位对照及当前 Power / Select 补丁审计通过；完整结果见
`npunlock-probe/research/results/shave-fp32-pipeline/REPORT.md`。

2026-10-08 后续确认 30 层 decode 一轮需要缓存 92 块，旧 64 块上限导致每步 28 次驱动分配/释放。
默认 allocator 现已改为 128 块，16 MiB 闲置字节上限不变，模型无需调整调用方式。
两次独立的同进程容量 A/B 中，B1/B4 decode 的调用加输出释放耗时降低约 12%–14%，预热后驱动分配/释放均为零。
prefill 仍仅需 62 块，未测得稳定加速。33 项设备检查、11 项 FP16/FP32 训练检查及缓存寿命/边界/失败隔离通过。
当前实现、原始样本和复现见 `npunlock-probe/research/results/cache-blocks/REPORT.md`。

2026-10-08 新增独立的连续 2 / 4 层同步合并提交实验，尚未接入正式默认路径。
30 层 SmolLM2-135M、FP16、B1/S32 decode 的不带计数器同进程交错对照中，
四层分组把每步提交 / 等待从 30 次降为 8 次，调用加输出释放中位数从 18.718 ms
降为 14.584 ms，减少 22.1%，21 轮配对全部更快。图之间使用依赖屏障，仍按序计算。
输出和 KV 逐位一致，8 步连续缓存推进、真实设备拒绝检查及模拟驱动失败隔离通过。
计时不含 CPU embedding、最终 norm / lm_head 或采样，未测通用训练和其他 batch / 精度。
原始样本、实现范围与复现见 [decode 分组提交实验](npunlock-probe/research/results/decode-chain/REPORT.md)。

2026-10-09 已接入正式同步链式执行接口 `graphinfer_session_infer_chain` 和 C++ `run_chain`，
不再临时替换全局执行函数。`FrozenSmolStack.decode(token, caches)` 在 batch 1 / 4 / 8、
FP16 / W8A16、多于一层且全层融合 RMSNorm 配置下默认每四层合并提交；
`group_size=1` 保留逐层执行，可显式选择 `group_size=2` 或 `4`。
其他 batch、单层或未融合配置默认使用逐层执行。
首次调用自动按顺序完成冷图初始化，预热后每组一次提交和最终等待。
`stats()` 新增 `chain_groups` / `chain_graphs`，统计成功的原生链式调用组 / 图数；
冷组可能顺序提交多个图，因此 `chain_groups` 不是冷启动的实际队列提交次数。
30 层 B1 FP16 的同进程交错对照从 18.200 ms 降至 13.580 ms，耗时减少 25.4%；
七层 B1/B4/B8、FP16/W8A16 的六组显式分组对照均通过，耗时减少约 18.7%～28.9%。
这些时间只覆盖驻留 decoder。算子及计算精度沿用原实现；此项属于运行时调度优化。
完整验证、限制与复现见 [正式分组接口报告](npunlock-probe/research/results/decode-chain-api/REPORT.md)。

2026-10-09 后续补齐完整 30 层、前缀 32 的 FP16 / W8A16、batch 1 / 4 / 8 六组对照，
各 21 轮全部更快，耗时减少约 18.2%～22.2%，据此扩展上述默认策略。
每组连续 decode 32 步至位置 64，输出 / KV / position 与逐层路径逐位一致，旧 seed 保持不变；
另测真实 63-token prefill → decode64 → 拒绝65。默认选择与回退也在真实设备复查。
本轮仅改变 decode 默认分组条件；native / SHAVE 内核及计算精度未改变。
六组原始计时、缓存检查和复现见 [完整模型分组报告](npunlock-probe/research/results/decode-chain-full/REPORT.md)。

2026-10-09 后续测试全部 30 层 decoder 在 NPU、embedding / 最终 norm / lm_head 在 CPU 的
实际贪心生成：B1、24-token 提示、固定生成 32 token、不提前停止 EOS，预热后计时含输入上传、
输出下载和 CPU 输出头。原分组策略相对逐层执行，FP16 40.95→54.38 tokens/s，W8A16 46.39→58.55 tokens/s。
剖析后减少了新分配多输出的重复别名检查；显式 `outs` 与形状 / 精度 / 布局检查保持完整。
另一轮同进程交错对照中，FP16 56.74→59.60 tokens/s，W8A16 58.96→61.51 tokens/s，
各 7/7 轮更快，全部 32 步 logits / token 逐位一致。改动自动作用于共享运行时，无需用户开关。
这些是两轮各自的配对数据，不能跨运行时段计算累积增益；未锁定后台负载 / 频率。
W8A16 的首步 logits 相对 CPU FP16 的 L2 误差仍为 5.69%，超过原 3% 门槛；性能 / 新旧一致性通过不代表量化质量通过。
FP16 同项误差为 0.48%。未采用出现舍入差异的最后位置输出头候选，精度策略及 native / SHAVE 未改变。
原始分阶段数据、回归与复现见 [完整生成剖析报告](npunlock-probe/research/results/smollm-generation/REPORT.md)。

2026-10-09 后续完成 30 层 B1、前缀 24 的原生阶段 / 设备任务分解。
不启用设备任务 profiling 的 FP16 decode 中位数 12.48 ms，原生等待约 10.07 ms、
提交约 0.69 ms、绑定约 0.15 ms，暖路径驱动内存分配为零。
启用 profiling 后，逐图合并重叠任务、再跨层求和，SHAVE 约 9.65 ms，其中两处
自定义 RMSNorm 合计约 4.91 ms；DPU 约 0.79 ms。这些设备采样与正常计时不能混成
同一耗时分解，驱动没有给出 DMA 时长，也不能据此排除 IO。
隔离测试了重复行复用和单行 FP32 位序比较两个候选；数值检查通过，但完整模型
没有 decode 收益，均未采用。B1 两行 carrier 实际拆成每次一行的 SHAVE 调用，
不能假定单次内核能访问另一行。生产 runtime / 内核保持本轮之前的实现。
完整样本、FP16 / W8A16 任务与候选对照见 [原生与设备热点报告](npunlock-probe/research/results/decoder-native/REPORT.md)。

2026-10-09 后续已将 RMSNorm 的 FP16 编码慢分支改为精确 SIMD 整数实现，
保留原 FP32 Neumaier 顺序、sqrt32 与两次半精度舍入。编译器将 encode4 内联，
消除了输出循环中每四元素的两次辅助调用。319,494 个 FP32 转换值新旧逐位一致，
158 项 RMSNorm 数值检查及正式内核 74 项归一化 / 模型回归通过。
三组完整 30 层配对对照中，prefill 延迟下降约 6.3%～12.9%，decode 约 1.5%～3.0%。
另一组真实 32-token 生成计时中，FP16 60.99→64.43 tokens/s，W8A16 66.85→70.69 tokens/s，
均 7/7 轮更快，全部 logits / token 逐位一致；FP16 首 token 中位数略增，后台负载未锁定。
已写入默认 RMSNorm C / ELF，无需用户开关，无新增依赖；通用训练与量化质量策略未改变。
W8A16 既有 5.69% 首步 logits 误差仍超过 3% 门槛。
消融、全部样本、精度边界与接入证据见 [单行 RMSNorm 优化报告](npunlock-probe/research/results/rmsnorm-single-row/REPORT.md)。

2026-10-09 后续将 SmolLM decode 的 RoPE cos / sin 与因果掩码合为一次 Gather，
已接入默认固定 64-token 路径，无需用户开关。四层 profiling 每层 SHAVE 任务 106→86，
DPU 任务数不变；完整 30 层 B1 配对 decode 13.081→12.790 ms，24/31 轮更快。
真实 32-token 生成 FP16 61.00→63.65 tokens/s（5/7 轮），W8A16 69.38→71.78（7/7 轮），
全部 logits / token 逐位一致；B4 长前缀几乎持平，后台负载未锁定，不能保证各阶段均提速。
B1/B4/B8 全部有效 decode 位置对照、正式 74 项回归与离线审计通过。
本次只优化模型图，内核 / runtime / 精度和 W8A16 量化质量策略不变。
完整生成两版仍各有 86 次驱动分配，与此前记录一致；驻留 decode 的零分配结论不适用于整次生成。
原始数据、失败断言记录与最新复现入口见 [decode 查表融合报告](npunlock-probe/research/results/decode-mask-table/REPORT.md)。

2026-10-10 后续定位并消除了已测完整生成中的 86 次重复驱动分配：
原空闲缓存 128 块上限容不下该流程所需的 214 块，每轮淘汰后又重新申请；
已将默认块数上限改为 256，保留 16 MiB 字节上限及原有释放 / 失败隔离保护。
30 层 B1、24-token 提示词、生成 32 token 的预热配对测试，FP16 中位耗时
490.749→484.910 ms（7/7 轮），W8A16 452.282→446.944 ms（5/7 轮），均约降低 1.2%。
全部 32 步 logits / token 和最终 KV 状态逐位一致；正式扩展各精度 8 次确认均为零新增驱动分配，
5 项缓存保护、3 项顺序、合成失败隔离、33 项设备和 11 项快速训练检查通过。
此工作负载多保留约 0.955 MiB 空闲缓存；冷启动 / 其他尺寸仍可能申请，后台负载未锁定。
W8A16 既有量化质量门槛和长期训练限制没有改变，最终状态清理在性能计时之外。
历史审计绑定修改前指纹；该阶段的离线复核与完整证据见
[生成缓存修复报告](npunlock-probe/research/results/generation-cache/REPORT.md)。

2026-10-10 后续完成当前 decoder 的原生阶段与 FP16 / W8A16 设备热点复测。
普通 30 层 decode 约 12.53 ms，其中原生等待约 9.38 ms；独立设备记录仍以 RMSNorm 等 SHAVE 工作为主要已记录热点。
已把 KV 更新的选中标记并入已有 RoPE / 因果掩码位置 Gather，进入默认固定 64-token decode 图，无需开关。
FP16 每层省去 12 个 Equal 任务，SHAVE 总数 86→74，DPU 任务数保持 96。
30 层 B1 配对 decode，FP16 11.687→10.578 ms（31/31 轮），W8A16 10.807→10.431（24/31 轮）。
真实 32-token 生成 FP16 66.16→71.56 token/s，W8A16 70.87→73.01，均 7/7 轮更快，全部 logits / token 逐位一致。
B1/B4/B8 的四层有效位置对照、正式各精度 8 次生成确认与 74 项模型回归通过，预热分配保持为零。
本次只改模型图，runtime / 内核 / 精度和量化策略不变；W8A16 5.69% 首步 logits 误差仍超过 3% 门槛。
独立 profiling 不作为普通图耗时分解，后台负载未锁定，W8A16 首 token 本轮略慢，通用训练全集未重跑。
原始样本、剩余热点、正式接入指纹及当前最新离线复核见
[decoder 热点复测与 KV 标记融合报告](npunlock-probe/research/results/decoder-refresh/REPORT.md)。

2026-10-10 增加同模型 CPU / NPU / Arc GPU 端到端推理基准。
完整 30 层 SmolLM2-135M、FP16、batch1、24-token 提示词、生成32 token，七轮热中位速度：
CPU 8线程约 31.82–33.50 token/s，当前 NPU decoder + CPU 两端约 68.55，GPU eager 约 24.83。
热首 token 分别约 61.46–62.62 / 23.71 / 42.34 ms；NPU 首次完整调用含编译约 68.12 秒，另计模型加载。
四种 CPU 环境/线程对照和 NPU/GPU 均生成相同的32 token，固定 token 回放最大 logits 相对 L2
分别为 0 / 1.12% / 0.84%，全部通过现有3%门槛。NPU 热运行没有新编译或驱动分配，正常清理归零。
这是当前融合 NPU 图与 CPU/GPU eager 实现的对照，Torch版本不同；CPU FP32/BF16、GPU SDPA/compile、
batch4/8、长上下文和训练未在本轮测量，不能解释为硬件最终排名或能效结论。
复现 `npunlock-probe/research/run_model_cross_device.ps1`，原始样本、冷启动、准确性证据与离线核验见
[CPU / NPU / GPU 模型基准报告](npunlock-probe/research/results/model-cross-device/REPORT.md)。

2026-10-10 后续补齐 CPU FP32/BF16、GPU SDPA 和 batch4/8。FP16 8线程 CPU 的 B1/B4/B8 合计速度约30.67/69.06/108.51 token/s，
GPU SDPA约32.32/122.55/196.09，NPU decoder + CPU两端约72.93/167.39/288.05。
batch增大提高合计吞吐，单请求延迟也增加；GPU B8复核波动较大，没有统一自动切换SDPA。CPU BF16未通过3%误差门槛。
新增可选 `NPUNLOCK_GRAPH_CACHE`，相同模型/形状/驱动下新进程复用60份图，首次完整生成由含编译写缓存的76.41秒降至6.23秒。
缓存损坏自动重编译，三种缓存状态的全部logits/token逐位一致；17项缓存检查及21项冻结模型保护检查通过，图计算和默认精度不变。
缓存默认关闭，最多512MiB并保留256MiB磁盘余量；本次322.15MiB缓存按用户要求验收后清理，首次启用需重新生成。
详细数值、环境限制和复现见 [后续优化报告](npunlock-probe/research/results/model-cross-device-followup/REPORT.md)。

现已增加显式 `torch_npu_lowbit.QuantizedLinear` 实验：INT4/NF4 压缩冻结权重，
FP16 激活和输出，直接通过驱动编译器生成 DPU 图。首次调用按权重指纹和 batch 编译，
预热后不重新序列化完整权重，无测得的显式主机载荷复制或新编译。
20 组编码/缩放/边界检查、两种低位 MLP 推理及 8 组常驻性能对比通过。
分组 INT4 小 batch 的形状相关错算采用 NPU 补零到至少 32 行、结果 DMA 截取绕过；
现已将两步与 MatMul 合入单图，连续输入每次从 4 次提交降为 1 次。
分组编译图不一定保留压缩常量。默认逐输出通道缩放的 1024x1024 权重图，
常量区约 528 KiB，对照解码权重 FP16 图约 1.79–1.94 MiB。
这不是通用 INT4/NF4 Tensor 设备或低位训练；当前只做冻结权重推理，未加入常规 ATen 分流。
详细误差、速度、失败记录与复现命令见 `npunlock-probe/research/results/lowbit/REPORT.md`。
本阶段同进程对照中，分组 INT4 1024x1024、batch1 从约 1.415 ms 降至 0.369 ms。
新增显式 `compact_groups=True`：每组独立压缩 MatMul、FP16 平衡树合并，最多 32 组。
1024x1024、group128 编译常量区可降至 640 KiB；不会自动启用此选项。
各组部分和有额外 FP16 舍入，超范围还可能被 DPU 截断，使抵消结果错误；
它仅适用于确认部分和范围及模型质量的实验，不提供单次完整 MatMul 的等价保证。
对照和边界结果见 `npunlock-probe/research/results/lowbit-group/REPORT.md`。

```python
import torch
import torch_intel_npu
from torch_npu_lowbit import QuantizedLinear

# CPU 准备冻结权重；默认逐输出通道缩放，也可指定整除 K 的 group_size。
source = torch.nn.Linear(256, 256).eval()
layer = QuantizedLinear.from_float(source, format="nf4")  # 或 "int4"
x = torch.randn(16, 256).half().to("intel_npu")
with torch.no_grad():
    y = layer(x)  # FP16 intel_npu Tensor

# 实验性节省分组常量空间：需检查部分和范围及模型质量，默认关闭。
compact = QuantizedLinear.from_float(source, format="nf4", group_size=128,
                                     compact_groups=True)
```

INT4 为二进制补码；NF4 是 16 项码表。两者每字节两个码，低 nibble 在前，
缩放值为 FP16；没有零点、双重量化或现成 checkpoint 格式适配。
此模块的权重是不可变字节描述，不是可求梯度 Parameter；普通 state_dict 目前不保存它，
从原 CPU Linear 重新准备。设备常规 FP16/FP32 训练仍使用原接口。

2026-10-04 增加 `format="int8"`。默认 W8A16 使用 FP16 激活、INT8 权重的
DPU 矩阵任务；显式指定 `activation_scale` 则使用 W8A8。两者的 Torch 输入、
输出仍为 FP16，矩阵与激活量化都在 NPU 图内完成。W8A8 编译后检查所有非逐元素
DPU 任务的格式字段，要求激活和权重都是 8-bit，否则拒绝加载。
本机 W8A8 实际为 U8×U8，零点 128，表示有符号量化码；不是只存成 INT8 再
执行 FP16 MatMul。此路径复用驱动编译器，未增加 OpenVINO runtime 依赖。

```python
# source 仍为 CPU nn.Linear，冻结权重推理。
w8a16 = QuantizedLinear.from_float(source, format="int8")
w8a8 = QuantizedLinear.from_float(source, format="int8", activation_scale=1/128)
with torch.no_grad():
    y = w8a8(x)  # 输入/输出都是 FP16 intel_npu Tensor
```

可选 `calibrate_int8(weight, inputs)` 使用有代表性的实际 Linear 输入，在 CPU
准备逐输出通道 W8 权重，然后复用同一个 `QuantizedLinear`：

```python
from torch_npu_lowbit import calibrate_int8

# calibration_inputs 是来自原模型该 Linear 的 CPU 输入，展平为 (rows,K)。
packed = calibrate_int8(source.weight.detach().half(), calibration_inputs.half())
calibrated = QuantizedLinear(packed, bias=source.bias)
with torch.no_grad():
    y = calibrated(x)  # W8A16，运行时不重新校准
```

权重和校准输入须为有限 CPU FP16/FP32；1<=rows<=4096、K<=4096，
输入及校准输出各不超过 4M 元素，权重仍受原 1M 元素上限约束。
CPU FP64 协方差求解会使用额外内存；只适用于冻结权重，当前不支持分组校准。
校准集逐行投影误差不会变差，但不能保证任意文本的最终模型质量。
算法来源与许可保存在 `LICENSES/GPTQ.txt` 及本轮报告。

W8A8 暂限逐输出通道权重缩放，不能与 group_size<K 或 compact_groups 一起使用。
activation_scale 是用户按校准样本选择的静态步长，必须为 2 的幂，范围 2^-24..256；
输入量化至 [-128,127]，超出 [-128*scale,127*scale] 的值会饱和。
例如 1/128 的输入范围是 [-1,0.9921875]，它不是所有模型通用的设置。
任意激活 scale 在驱动 8.3 中已复现阈值漂移，暂时明确拒绝。
权重范围端点使用 FP16，因此编译后的有效权重 scale 可与原 scale 有细微差异；
W8A8 的验证使用独立 INT64 点积与对齐整数零点的有效 scale，而非假定浮点
FakeQuantize 的零点一定落在整数量化码上。输入应为有限值，未认证 NaN 行为。

25 组 INT8 数值检查、3 组激活 scale 边界、两种 MLP、输入和缓存边界通过；
原 INT4/NF4 的 20 组数值和两种 MLP 回归通过。常驻 1024x1024 权重、batch512
的同进程 21 轮交错测试：W8A8 0.824 ms，NPU FP16 0.962 ms，8 线程 CPU FP16
2.135 ms。batch1 仍是 CPU 更快。INT8 编译常量区约 1040 KiB，对照 FP16 2064 KiB；
不代表整个运行进程内存减半。测试排除首次编译与上传/下载，桌面负载未控制。
仍未支持普通 int8 设备 Tensor、低位训练、动态激活校准、QLoRA 或 LLM checkpoint。
复现：`run_torch_npu.ps1 -Script torch_npu_int8_demo.py -ScriptArguments '--case','numeric'`
（case 还可为 edges/mlp/guards/performance）。详细记录见
`npunlock-probe/research/results/int8/REPORT.md`。

2026-10-04 增加显式 `torch_npu_transformer.FrozenTransformer`：已验证 1–2 层、
D128/256、head_dim=32、FF=4D 的 causal pre-norm GELU Transformer 冻结推理，
S<=64，输入为 FP16 `(B,S,D)` NPU Tensor。attention、softmax、GELU、残差和
归一化都在 NPU 上执行；没有新增 OpenVINO runtime 依赖或 CPU 运算回退。
这是专用转换器，不代表普通 `nn.Transformer.to('intel_npu')` 已能训练或执行。

```python
import torch
import torch_intel_npu
from torch_npu_transformer import FrozenTransformer

block = torch.nn.TransformerEncoderLayer(
    256, 8, dim_feedforward=1024, dropout=0, activation='gelu',
    batch_first=True, norm_first=True).eval()  # CPU 准备
model = FrozenTransformer.from_float(block)  # 默认 W8A16；或 mode='fp16'
x = torch.randn(1, 64, 256).half().to('intel_npu')
with torch.no_grad():
    y = model(x)  # 强制 causal，FP16 intel_npu 输出

# 实验性混合 INT8：QKV/FFN 第一层 W8A8，其余投影 W8A16。
# 使用与实际任务相符的 CPU 校准样本；静态范围以外会饱和。
mixed = FrozenTransformer.from_float(
    block, mode='mixed_int8', calibration_samples=[torch.randn(2, 64, 256)])
```

默认 W8A16 和 FP16 将整网编入一次图提交；mixed_int8 使用已有真正 FP32
SHAVE 归一化，再执行 attention/FFN 两段，保持量化边界的精度，代价是更多提交。
完整 W8A8 Transformer 尚未通过验证，构造时拒绝。1–2 层的 FP16 逐项验证使用
rtol=.004/atol=.002×层数；mixed 整网量化参考 L2 门槛为 .5%，每个相同输入
分段仍要求逐项近似且 L2<.1%。历史较严格门槛的失败和修改原因保存在报告中。
随机模型对照通过不代表真实语言任务质量、长序列或文本生成已验证。

15 组数值、分段、因果性、输入/训练拒绝与缓存/权重快照检查通过。两层 D256，
B1/S64：W8A16 .803 ms，8 线程 CPU FP32 1.787 ms；B8/S64：1.296 vs 5.887 ms。
预热后零显式主机张量复制/新增编译；排除初始编译和传输，桌面负载未控制。
混合 INT8 为约 40/223 ms，当前更适合调试量化，默认仍为 W8A16。
尚无 KV cache、embedding、logits、mask 参数、通用 softmax/LayerNorm ATen、
INT4/NF4 Transformer、低位训练、QLoRA 或普通 state_dict 权重持久化。
保存原 CPU 模型，再通过 from_float 重建冻结快照。
复现：`run_torch_npu.ps1 -Script torch_npu_transformer_demo.py -ScriptArguments '--case','numeric'`
（另有 probe/segments/causality/guards/performance/norm-fp32）。完整记录见
`npunlock-probe/research/results/transformer/REPORT.md`。

## 真实 SmolLM2 权重的单层与连续层推理

2026-10-04 新增 `torch_npu_smollm.FrozenSmolLayer.from_checkpoint`。
已下载并校验 `HuggingFaceTB/SmolLM2-135M-Instruct` 的原始 safetensors，固定 revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`；文件在 `models/SmolLM2-135M-Instruct`。
真实层支持 RoPE、9 个 query head / 3 个 KV head 的 GQA、causal attention、
SwiGLU、残差与 RMSNorm，提供 FP16 和默认 W8A16。RMSNorm 现默认使用融合的
真正 FP32 SHAVE 内核，再按官方层的顺序舍入为 FP16、乘 gamma。
forward 的算术全部在 NPU，权重读取、INT8 编码和 RoPE 常量是显式 CPU 准备。
复用既有权重图、缓存和执行器，未改动基础设备后端或低位 Linear。

```python
import torch
import torch_intel_npu
from torch_npu_smollm import FrozenSmolLayer

layer = FrozenSmolLayer.from_checkpoint(
    r"E:\Pytorch_xfire_NGPU\models\SmolLM2-135M-Instruct",
    layer_index=0, mode="w8a16")
# hidden_states 为相应层的 CPU FP16 隐藏状态，不是整数 token ID。
with torch.no_grad():
    output = layer(hidden_states.to("intel_npu"))
```

输入仅限 FP16 `(B,S,576)`，B<=8、S<=64，位置为从零开始的完整前缀；
`forward` 仍计算完整前缀；有界 KV cache 使用下述显式 `prefill/decode` API。
没有 padding mask、训练或普通模型 `.to('intel_npu')` 接入。
demo 的 `hybrid` 将第 0 层放到 NPU，其余 29 层、embedding、final norm 和 logits
显式留在 CPU；两条提示 × 两种模式的短生成 token 与 CPU 参考一致。
这不能证明完整 30 层 NPU 生成、通用模型质量或 INT4/NF4 LLM 已可用。

本轮检查采用全层 L2<0.3%、每项 `abs(error)/(abs(reference)+token_RMS)<1%`，
固定实际输入的每段 L2<0.1%，相对官方浮点层 L2<3%。原来的固定绝对容差
`rtol=.01, atol=.005` 在部分输出上失败，失败与计数均保留；未宣称这些输出
通过了原门槛或逐位相同。完整误差、依赖版本和复现命令见
`npunlock-probe/research/results/smollm/REPORT.md`。

2026-10-05，融合 norm 进一步复用编译器的多 tile 调度：B*S>=8 时按完整行
分到 4 个 tile、8 个 SHAVE 任务；更小输入保留 ReferenceSW，避免宽度拆分。
该调度阶段保持核代码、FP32 算术及 gamma 前的 half 舍入。`compile_rmsnorm` 逐个验证
完整行、连续步长、CMX 输出不重叠以及总行数；只替换既有代码窗口，不改 DMA 或调度。
最大 B8/S64 的 norm 也只需一张图，整层热调用从 24 次降为 4 次。
普通算子的 65536 计算分块和 1048576 存储元素上限均未放宽。

22 组 norm 专项覆盖 7/8/9 行切换、非整除行数、最大尺寸、FP16 次正规数、零及
大数平方，与旧 NPU 路径逐位一致。真实层、最大尺寸、guards、因果性和生成
使用原有误差门槛回归。同进程 benchmark 同时保留旧逐算子版本、串行融合版本、
并行融合版本与 CPU 参考，具体延迟见最新报告。

行内四通道向量归约也已实机研究：工具链的 half 向量转换出现错误，改用已验证的
标量转换后结果正确，但实测无稳定加速，该归约原型未接入默认路径。
源码、原始计时、失败原型及复现命令见
`npunlock-probe/research/results/rmsnorm-vector/REPORT.md`；`rmsnorm/REPORT.md`
保留上一阶段串行融合记录。未新增运行依赖或 OpenVINO runtime，基础 FP32 设备保证未改变。

同日进一步优化输出循环，默认内核现在每组处理四个元素。正常范围的 FP16 解码和
RNE 舍入使用向量整数位运算，FP32 乘法使用 VAU；零、次正规数和溢出保留原处理。
FP32 补偿归约及开方顺序、gamma 前的中间 FP16 舍入和载体调度均保持原语义。
隔离检查覆盖全部 63488 种有限 FP16 编码，输出与旧 NPU 内核逐位对照。
同进程复测，真实 embedding 的 256/512 行 norm 约快 1.59/1.64 倍；
B4/S64 W8A16 单层从 9.620 ms 降到 6.994 ms。小输入收益受固定开销影响更大。
这些是暖运行的单层结果，计时排除准备和显式传输。源码、回归与原始计时见
`npunlock-probe/research/results/rmsnorm-output/REPORT.md`；诊断基线保留在
`npunlock-probe/research/results/rmsnorm-diagnostic/REPORT.md`。

2026-10-08，SmolLM2 的默认 SHAVE RMSNorm 新增跨 16 行的 FP32 SIMD 补偿归约。
每个 SHAVE 分片至少 32 行时启用，小分片及尾行继续使用原累加代码；逐行列顺序、
FP32 Neumaier 补偿、开方及 gamma 前后的 FP16 舍入均保留。
门限按实际 SHAVE 分片判断：当前 256 行 carrier 分成 8 个 32 行分片，不能把完整
张量行数直接当作内核的行数。整数位模式比较避免了工具链将向量浮点比较展开成
大量标量指令，NaN/Inf 仍保留原处理。编译仍为 `-O2 -ffp-contract=off`。

同进程交错 A/B，256/512 行 norm 约快 8.8%；30 层 FP16、B8/S32 预填充
95.884→88.607 ms（31 轮中 30 轮胜出）。B1/S32、B4/S16 及 decode 基本持平。
模型计时排除 CPU embedding、最终 norm、lm_head 与首次准备，不代表完整生成速度。
实际 C/ELF 已接入默认路径，无需用户按模型手工启用。数值、完整模型对照和正式
回归见 `npunlock-probe/research/results/rmsnorm-row16-gated32/REPORT.md`。

2026-10-05 新增连续层 `FrozenSmolStack`。它按顺序调用相邻 decoder 层，
每层输出直接作为下一层的 NPU 输入；热调用没有层间主机复制或新图编译。
已验证 2／4 层、FP16／W8A16，从第 0、10、26 层开始的层段，以及最大 B8/S64。
其形状、完整前缀与精度限制和单层相同；不会让同一序列的依赖层同时计算。

```python
from torch_npu_smollm import FrozenSmolStack

stack = FrozenSmolStack.from_checkpoint(
    r"E:\Pytorch_xfire_NGPU\models\SmolLM2-135M-Instruct",
    start_layer=0, num_layers=4, mode="w8a16")
with torch.no_grad():
    output = stack(hidden_states.to("intel_npu"))
```

2026-10-07 可显式传入 `packed_weights`。单层构造器要求完整的七项投影权重字典；
连续层要求 `{checkpoint_layer_index: projection_dict}`，索引使用 checkpoint
中的绝对层号，恰好覆盖请求的连续层段。每项必须是匹配形状、逐输出通道的
INT8 `PackedWeight`，仅用于 `mode='w8a16'`。构造时保存不可变权重快照，
不能用它替换已经编译的图。默认不传此参数时继续采用原量化方式。

输入为起始层对应的隐藏状态；返回最后一层的隐藏状态，不是 logits。
2／4 层热调用分别提交 8／16 张图，仍逐图同步；未融合跨层图或启用异步流水线。
多层与单层使用相同的执行误差及量化质量门槛，原固定绝对容差的失败计数继续保留。
新 `torch_npu_smollm_stack_demo.py --case hybrid` 将前 2／4 层作为一个常驻段，
段外显式上传／下载，其余 28／26 层与 embedding／final norm／logits 留在 CPU。
两条提示、两种模式、两种层数的 8 组短生成均复现 CPU token；这只是一组 smoke test。
尚未验证 30 层全 NPU 生成，训练仍不受这个冻结适配器支持。
数值、拒绝条件、内存范围、包含传输的公平对照和热运行计时见
`npunlock-probe/research/results/smollm-stack/REPORT.md`。

2026-10-05，单层与连续层新增显式 KV cache：

```python
with torch.inference_mode():
    output, cache = stack.prefill(prefix_hidden_states.to("intel_npu"))
    # 下一 token 在该层段入口的 FP16 隐藏状态，形状 (B,1,576)。
    output, cache = stack.decode(next_hidden_states.to("intel_npu"), cache)
```

K/V 按 3 个 GQA head 以 FP16 保留在设备上，每层固定 64 槽位；首次 prefill
从位置 0 开始，decode 每次追加一个 token，总长度最多 64。位置作为图的设备输入，
RoPE、缓存槽位更新和 causal mask 在 NPU 计算；换位置复用同一张 decode 图。
每次返回新缓存快照，旧快照可重用或分支，调用方不得编辑其中的 Tensor。
多层拒绝其他模型的缓存、乱序层、混合前缀／分支、长度不一致或满缓存，
在所有层完成检查前不执行任何层。FP16／W8A16 模式、形状上限及基础 FP32 保证不变。

驱动 8.3 会拒绝原 batch 布局及仅在动态矩乘前折叠 heads 的布局。当前单 token
注意力从投影／残差开始保持内部 N=1，将样本编码到 head 维度，并保留每个样本独立的
GQA 索引和 Softmax。B>1 decode 整批用一次注意力图提交，不再逐样本选取和拼接。
同样修复了批量单 token prefill／普通 forward 的编译失败；多 token prefill 路径保持原实现。
单图内部由编译器调度，不代表已经测得所有样本／多层在硬件上同时运行。
原生 PyTorch 绑定现支持多个独立输出，注意力输出和 K/V 直接写入各自的设备缓冲区，
省掉旧版打包与后续 DMA 拆分。位置更新也作为注意力图的第四个输出，取消独立加法；
有界整数位置 1..64 用 FP16 加一再转回 FP32 可精确表示，模型和基础 FP32 算术精度不变。
前轮三个独立输出的 DMA 拼接合并后，B1/B2/B4/B8 每层暖提交为 4/6/8/12；
当前整批布局进一步降为 4/4/4/4，仍逐图同步。
前轮 B8=49 包含首次 decode 的两次 gamma 广播准备，
重复暖调用已确认应为 47。当前不再上传 sample 常量，各位置复用整批 decode 图。
解码区间不需要显式主机上传／下载
或按位置重新编译。编译、初始权重／gamma 准备及段外输入输出传输仍有成本。

`torch_npu_smollm_kv_demo.py --case hybrid` 显式组合前 2／4 层 NPU 缓存与
其余 CPU 层的 Transformers `DynamicCache`；embedding、final norm、logits 留在 CPU。
这还不是普通 Transformers 模型的 `generate()`／`.to('intel_npu')` 自动接入。
本轮只完成 FP16／W8A16 KV cache，尚未接入 W4A16 或扩展 64-token 上限。
8 组缓存版混合生成复现 CPU token。首轮单输出版本的 6 组暖运行中 W8A16 decode 均比重算
前缀慢约 1.26–1.97 倍：B1／4 层为 19.54 对 15.52 ms，B4／4 层为 77.07 对
40.67 ms。这些历史结果保留在 `npunlock-probe/research/results/smollm-kv/REPORT.md`。
多输出版的数值、缓存完整性、连续解码和同进程旧版／新版性能对照见
`npunlock-probe/research/results/smollm-kv-multi/REPORT.md`。
同进程 21 轮交错计时中，W8A16 六组解码延迟降低 15.6–22.1%；4 层 B1 从
18.42 降到 14.62 ms，B4 从 74.89 降到 59.91 ms。B1 已接近重算短前缀，
B4 仍慢于重算，所有缓存路径仍慢于对应 CPU FP16 decode。计时排除初始编译与
段外传输，包含提交、设备 DMA 和同步；桌面负载与频率未控制。
接入层需运行 `build_torch_npu_device.ps1` 更新；session DLL 与 SHAVE 内核不需要重建。
`torch_npu_multi_output_demo.py` 验证混合形状／dtype 输出、缓冲区复用、别名拒绝、
错误绑定恢复及旧单输出真正 FP32 运算。原生多输出同样要求完整、独立且不重叠的分配。
本轮位置／样本选取合并复用该扩展，无需重新构建。102 条数值记录、336 项固定输入 K/V、
252 步连续解码、24 组旧／新缓存输出逐位一致检查及普通 forward 回归通过；
当前报告和同进程性能对照见 `npunlock-probe/research/results/smollm-kv-fused/REPORT.md`。
本轮 21 轮交错计时中，W8A16 相比上一轮延迟降低 16.9–42.3%；4 层 B1 为
17.09→14.20 ms，B4 为 63.01→37.99 ms。六组 W8A16 缓存 decode 比重算短前缀
快 7.4–16.3%，仍慢于对应 CPU FP16 decode；计时范围与桌面负载限制同上。
后续分段诊断与 DMA 拼接合并见 `npunlock-probe/research/results/smollm-kv-profile/REPORT.md`。
分段计时中单样本两次 RMSNorm 合计约 39%，批量注意力约 57–75%；原生执行包络
包含提交、设备工作和 fence，不能分解为纯内核／IO 时间。批量注意力布局与 QKV／缓存
拆分的两次试验在该阶段被编译器拒绝，失败记录保留。
三个 Concat 合并后追加同模型、同缓存的 41 轮 ABBA／BAAB 对照：4 层 B4 的
FP16 为 26.84→24.85 ms（延迟减少 7.4%），W8A16 为 27.29→25.68 ms（减少 5.9%）；
单层 B8 W8A16 减少 3.2%。B1 路径未改变，对照中位差异小于 1%。早期不同实例的
计时出现较大波动，保留原始样本，以追加对照作为主要依据；后台负载／频率仍未控制。
24 组缓存输出及追加五组对照逐位一致，
通用 F16/F32 位模式、输出复用、边界与错误恢复检查通过。无需 native／SHAVE 重建。
后续整批布局与单 token prefill 修复见 `npunlock-probe/research/results/smollm-kv-batch/REPORT.md`。
同模型、同缓存的 41 轮 ABBA/BAAB 对照：四层 B4 FP16 24.97→11.45 ms，W8A16
25.99→11.95 ms，延迟约减少 54%；单层 B8 两种精度约减少 74%，所有批量组均 41/41 轮更快。
四层 B4 W8A16 阶段计时中，两次 RMSNorm 合计约 44%，注意力约 28%，独立拼接为零。
这是本机暖片段计时，排除准备／编译／显式传输；未锁定桌面负载／频率，不代表全 LLM 端到端速度。
附加检查涵盖 B2..B8、单样本扰动和独立 B1 对照；保留数值门槛与逐位结果，
不保证不同 batch 形状的全部输出逐位相同。

小行数 RMSNorm 更新见 `npunlock-probe/research/results/rmsnorm-small/REPORT.md`。
FP16 解码/FP32 平方按四元素向量化，FP32 补偿求和顺序和舍入保持不变；内部同步
forward/prefill/decode 复用归一化输出工作区。每层暖 decode 分配从 7 次降到 5 次，
仍提交 4 图；公开输出和旧 KV 保持独立。私有归一化辅助函数默认返回新输出，
串行分块仍独立分配。生产内核已重建，native DLL/PYD 和严格 FP32 算子无需重建。
14 组同模型/输入/缓存的 41 轮三策略对照，中位延迟下降 5.3%–11.4%，
归一化与新旧模型输出/KV 逐位一致，原数值门槛保留。独立模型实例的 FP16 对照
仍有变慢的组，CPU 在这些小片段中仍更快；上述收益不构成普遍或端到端加速承诺。
复现模型检查时使用 `-ScriptArguments '--phase','rmsnorm-small','--case','numeric'`，
完整对照脚本为 `torch_npu_rmsnorm_benchmark.py`。

图内 RMSNorm→投影融合更新见 `npunlock-probe/research/results/rmsnorm-fusion/REPORT.md`。
冻结 SmolLM2 层的单 token B2..B8 forward/prefill/decode 已默认采用两张图：
FP32 SHAVE 归一化与 attention 的 DPU 投影合为一图，第二次归一化与 FFN 合为另一图。
每层暖 decode 提交从 4 次降到 2 次，分配仍为 5 次，公开输出与 KV 快照独立。
保留同一内核的 FP32 补偿求和及两次 FP16 舍入；FP16/W8A16 两种权重路径均已验证。
B1 载体会拆分行，多 token 前缀也继续使用原路径。只改经过指纹校验的 Power 代码窗口，
不重写 DMA/barrier/任务布局；首次 decode 的新 gamma 行数在 NPU 上从驻留参数广播。
42 个融合图与分图基线逐位相同，另有奇数 batch、样本扰动、输出/KV 保留及多线程检查。
8 组同实例 41 轮 ABBA/BAAB 对照中，B2/B4/B8 延迟下降 26.0%–43.0%；
四层 B4 FP16 10.79→6.15 ms、W8A16 10.56→6.22 ms。B1 执行路径不变，仅有计时波动。
独立实例对照保留 B1 变慢的结果；单层 B4 W8A16 略快于该轮 CPU，但四层 B4 仍更慢。
这些是本机暖片段时间，未锁定桌面负载/频率，不代表完整 LLM 或通用加速。
未增加 OpenVINO Python/runtime 依赖，仍依赖 Intel 驱动图编译器；未增加训练或模型范围。
复现使用 `-ScriptArguments '--phase','rmsnorm-fusion','--case','numeric'`，
同实例性能对照脚本为 `torch_npu_norm_fusion_benchmark.py`。

B1 单序列随后也已接入，见 `npunlock-probe/research/results/rmsnorm-b1-fusion/REPORT.md`。
归一化图内临时复制为两行，再切回第一行送入投影；模型 batch 与 KV 保持 B1，
不增加第二份 attention/FFN 或公开输出。继续保留 FP32 SHAVE 补偿归约及原舍入。
所有 B1..B8 单 token 路径现为每层 2 图、5 次暖分配；多 token 前缀仍使用原路径。
8 组 B1 同实例 41 轮对照下降 29.9%–33.8%，全部 41 轮更快；
四层 B1 FP16 9.31→6.36 ms、W8A16 10.21→6.92 ms。CPU 在本轮 B1 小片段中仍更快。
真实图与新旧输出/KV 逐位一致；56 组 B2..B8 图及权重字节完全未变。
覆盖总位置 64、单实例 batch 切换、驻留 gamma 复用及多线程独立输出/KV 检查。
未重建内核/native、未增加依赖或模型/训练范围。复现使用
`-ScriptArguments '--phase','rmsnorm-b1-fusion','--case','numeric'`，
同实例对照脚本为 `torch_npu_b1_fusion_benchmark.py`。上述为本机暖片段结果，
桌面负载和频率未锁定，不代表完整 LLM 的端到端性能。

整层单图更新见 `npunlock-probe/research/results/smollm-whole-layer/REPORT.md`。
当前 B1..B8 单 token forward/prefill/decode 已把两次 RMSNorm、attention、FFN
与 KV 更新放入同一张图，每层暖 decode 从 2 次提交／5 次分配降到 1 次／4 次。
两次归一化继续使用原 FP32 SHAVE 内核，矩乘继续使用原 FP16/W8A16 DPU 路径。
验证后只替换共享 Power 代码窗口；第二次 norm 支持经过精确步长校验的前导单例维度完整行布局。
48 组整层图与两图基线的输出/K/V/位置逐位一致。12 组同实例 41 轮 ABBA/BAAB
对照延迟下降 20.4%–30.2%，暖显式上传／下载／编译为零。
多 token 前缀继续使用原路径，范围仍为冻结 SmolLM2-135M、B≤8、总位置≤64；
未增加训练、padding mask、任意模型或跨层并行支持。内核、native DLL/PYD 与依赖未重建。
复现使用 `-ScriptArguments '--phase','smollm-whole-layer','--case','numeric'`，
同实例对照脚本为 `torch_npu_whole_layer_benchmark.py`。这些仍是未锁定桌面负载的暖片段测量。

多 token 前缀随后也已接入整层单图，见
`npunlock-probe/research/results/smollm-prefill-fusion/REPORT.md`。
当前 B≤8、S≤64 的冻结 SmolLM2-135M forward/prefill 与单 token decode
均为每层一张图；暖 prefill 提交 4→1 次，输出分配 4→3 次，decode 保持原样。
继续使用同一 FP32 SHAVE RMSNorm、FP16/W8A16 DPU 矩乘与完整行载体校验。
256 组多 token 图覆盖所有 B1..B8 和 8 种前缀长度，y/K/V 与四图基线逐位一致；
48 组单 token 图的 IR/权重完全未变。14 组同实例 41 轮对照下降 10.3%–65.4%，
四层 B1、32 token 的 FP16 8.47→4.67 ms，W8A16 8.53→4.71 ms。
暖显式上传／下载／编译仍为零；新前缀长度首次初始化 FP32 位置元数据仍会上传一个标量。
未改内核、native、精度策略、依赖或模型／训练范围。复现使用
`-ScriptArguments '--phase','smollm-prefill-fusion','--case','numeric'`，
同实例对照脚本为 `torch_npu_prefill_benchmark.py`。数据仍为未锁定桌面负载的暖层片段时间。

## 复现

```powershell
Set-Location E:\Pytorch_xfire_NGPU
.\run_torch_npu.ps1
.\run_torch_npu.ps1 -Quick

# 编译本地执行层 DLL，同时运行不依赖硬件的失败隔离检查
.\build_npu_session.ps1
# FP16 / FP32 各 1000 轮常驻链式计算，以及生命周期、并发提交和边界检查
.\run_torch_npu.ps1 -Session

# 编译最小 PyTorch C++ 设备接入层；使用项目内便携 MSVC/SDK
.\build_torch_npu_device.ps1
# 验证真正的 NPU Tensor、普通算子链、复制、视图和内存释放
.\run_torch_npu.ps1 -Device
.\run_torch_npu.ps1 -Device -Quick
# 重建新归约内核（ELF 已随源码保存，普通执行无需重建）
.\run_torch_npu.ps1 -Script npunlock-probe\research\build_resident_kernels.py
```

完整检查有 240 秒外层期限，每次编译 20 秒、执行 10 秒。
Quick 跳过全位模式、大型随机位模式和 opcheck。
设备检查另有 300 秒外层期限；其 Quick 只缩短迭代，保留全位模式和归约边界。
环境为 .conda/npunlock：Python 3.12.14、Torch 2.14.1+cpu、NumPy 2.5.3。
运行器配置 conda DLL 路径、源码 PYTHONPATH 和原生 wheel DLL 目录。

将下面示例保存为 my_script.py，再运行：

```powershell
.\run_torch_npu.ps1 -Script .\my_script.py
```

```python
import torch
import torch_npu_minimal as npu

x = torch.tensor([70000., 1.000244140625, 2**-140], dtype=torch.float32)
y = npu.add1(x)
f = torch.compile(lambda a, b: ((a + b) * b).sum(),
                  backend="npunlock_npu", fullgraph=True, dynamic=False)
z = f(x, torch.ones_like(x))
m = npu.matmul(x.reshape(1, 3), torch.eye(3))
print(y, z, m)
print(npu.execution_info())
```

## 支持范围

| 接口 | FP16 / FP32 行为 |
|---|---|
| add1、add、sub、mul、div | SHAVE 自定义内核；Tensor/标量、广播和混合 dtype |
| neg、abs、sign | SHAVE 自定义内核 |
| cast(x, torch.float16/float32) | NPU 执行转换和最近偶数舍入，保留非规格化小数 |
| sum、mean | 全部或指定维度、keepdim、dtype；FP32 树状累加 |
| matmul | 同 dtype 的二维矩阵；FP32 乘法和累加 |
| torch.compile | 上述函数/方法、half/float/to(dtype)、多算子图 |
| autograd | 上述算子的基础导数；所需算术复用 NPU 内核 |

输入为 CPU strided Tensor，0–4 维，每个 Tensor 1–65,536 个元素。
非连续输入会复制到连续 staging 缓冲区。矩阵乘法另有限制：
M*K*N <= 65,536，因为它展开逐项乘积。新形状须通过实际 CMX 布局校验。
尚未实现矩阵分块、批量 MatMul、动态形状、空张量、FP64/BF16、
模型级完整训练或 GPU+NPU 调度。

PyTorch 管理 CPU Tensor、视图、广播和梯度累积；广播梯度归并在 CPU。
基础 autograd 检查通过不代表整个训练过程都在 NPU。
这一节的 torch_npu_minimal 接口使用 CPU Tensor 存储。真实设备接口见下一节；
两种接口的覆盖范围不同。尚未将 NPU 合入 torch.xpu。

## 真正的 intel_npu Tensor

保存为 my_device.py，用 .\run_torch_npu.ps1 -Script .\my_device.py 运行：

```python
import torch
import torch_intel_npu as npu

x = torch.tensor([1., 2., 70000.], dtype=torch.float32).to("intel_npu")
y = torch.tensor([2., 3., .5], dtype=torch.float32).to("intel_npu")
z = (x + y) * y
print(z.device, z.untyped_storage().device)  # intel_npu:0 intel_npu:0
print(z.cpu())  # tensor([6., 15., 35000.])
print(npu.stats())
```

精度转换、算术和归约也可以直接连用：

```python
x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("intel_npu")
y = x.half().float()
z = (x + y).sum(dim=1, keepdim=True)
m = z.mean()
print(z.device, m.cpu())  # intel_npu:0 tensor(44.)
```

除初始 .to("intel_npu") 和最终 .cpu()，这条链路不复制主机载荷。
首次使用仍需编译静态图，之后复用同一上下文的图会话。

2026-10-20转换图复用：完整连续绑定通过原有验证后，FP16↔FP32内部按元素数编译一维图，并将结果恢复为调用方形状。scalar、out及大张量分块保留原契约；SHAVE转换和half位压紧公式不变。这减少不同shape重复编译，单次转换的提交数不变，见[报告](npunlock-probe/research/results/cast-reuse/REPORT.md)。

导入 torch_intel_npu 注册保留的 PrivateUse1 设备槽，已有其他后端占用时拒绝。
它没有替换 Tensor 类、伪造 device 属性或将 CPU Storage 重新标记为 NPU。
empty/empty_strided 通过 C++ c10::Allocator 调用 zeMemAllocShared；
Storage 的 DataPtr 释放回调管理原生缓冲区寿命。最后一个视图释放后才释放分配。
view、t/transpose/permute、detach 与安全范围内的 as_strided 共享原有 Storage。
矩阵、Linear、复制与普通逐元素/归约算子均会自动通过 NPU DMA 物化完整密集的
transpose/permute 输入。非零 offset、裁切、扩展重复或不能覆盖整个
分配的视图仍拒绝执行/复制。

支持范围：

| 功能 | 目前实现 |
|---|---|
| dtype / shape | FP16、FP32；rank 0–4；1–1,048,576 元素；单个 SHAVE 计算块仍最多 65,536 项 |
| Tensor 创建 | empty、连续 empty_strided、CPU Tensor .to("intel_npu")、.intel_npu() |
| 数据复制 | .cpu()、CPU/NPU copy_、NPU/NPU copy_、clone；相同 shape，可转换 FP16/FP32 |
| 精度转换 | .half()、.float()、.to(dtype=...)；保留非规格化数、最近偶数舍入；same dtype 的 copy=True 创建独立存储 |
| 普通算术 | add/sub/mul/div 支持广播、标量及 FP16/FP32 类型提升；neg/abs/sign/reciprocal、ReLU；add/sub 要求 alpha=1，div 仅 rounding_mode=None |
| 归约 | sum/mean；全部、单轴或多轴，负轴、keepdim、FP16/FP32 dtype；FP32 累加 |
| 视图与转置 | t、transpose、permute、detach 保留共享 Storage；完整密集排列可自动物化；empty_strided/clone/copy_/梯度累加/SGD 支持密集排列，DMA 保留原始位模式 |
| 矩阵乘法 | 同 dtype 二维矩阵的 @、torch.mm、torch.matmul，允许密集转置；FP32 计算，按 M/N/K 分块；输入和输出各 <=1,048,576 项 |
| 带偏置矩阵 / Linear | addmm 支持实数 alpha/beta 和偏置广播；nn.Linear/F.linear 支持 rank 1–4 输入，FP32 累加及加偏置后再转换输出 |
| 卷积 | NCHW Conv2d/NCL Conv1d；S19支持合法分组与depthwise，支持stride/padding/dilation和bias；输入/权重/输出各最多1,048,576项；窗口或折叠超过上限时采用流式前向/反向，受编译布局和缓存图上限约束；转置卷积/Conv3d未支持 |
| BatchNorm / 池化 | NCHW BatchNorm2d，FP32 统计/梯度；标准模块设置 track_running_stats=False；AdaptiveAvgPool2d(1) |
| 一阶训练 | 算术、sum/mean、二维矩阵、addmm、Linear、Conv2d、ReLU、BatchNorm、全局平均池化、同 shape/dtype MSELoss；标准 SGD 基础更新 |
| 输出复用 | 算术、归约、矩阵乘法 out=，须精确匹配结果 shape/dtype，禁止输出别名输入；可双缓冲交替迭代 |
| 提交 | S13至多1,536个编译图，保留至shutdown；新形状超限明确拒绝，共享同一上下文；同步fence、串行提交 |

不支持向量/批量矩阵乘法、任意步长/裁切 NPU 输入、任意 in-place 算术、
随机数、二阶 autograd、torch.compile 设备路径、未实现算子的完整模型、异步 stream/event、
Storage resize/序列化、直接 GPU/NPU 传输。
这些操作会明确失败，不启用 CPU 算术回退。torch_npu_minimal 中已有部分功能，
还需要移植成保留设备 Storage 的实现，不能把旧接口的覆盖范围等同于新设备后端。
非连续 CPU 输入可 .to("intel_npu", memory_format=torch.contiguous_format)，
packing 与上传仍是显式主机复制。普通 Tensor 默认保持布局的 .to 可能被拒绝。

设备转换复用现有精确 SHAVE 内核。FP32→FP16 先把半精度编码写到 FP32
分配的低 16 位，再以 metadata view 加 NPU DMA StridedSlice 压紧为真正的
FP16 分配；没有主机取位。DMA-only 图布局受到检查。内部传输视图最多有
131,072 个 FP16 槽，但单块仍只使用 256 KiB。较大转换按块执行，再由 DMA 拼接；
公共设备 Tensor 最多 1,048,576 个数值元素。

设备归约替换 ReduceSum/ReduceMean 的 SHAVE 代码，保留驱动生成的调度/DMA。
实测驱动原始归约会清零 FP32 非规格化数，自定义内核复用 add32/div32 的整数
边界路径。校验每个调用的输入/输出元素比例，拒绝切断归约轴的部分和布局。
singleton 归约使用已验证的 Abs carrier ABI，仍从 +0 开始累加。
当前使用顺序 FP32 Neumaier 补偿累加，减少相消与 FP16 中点舍入误差；
未做并行树优化，不承诺与 CPU 的求和顺序逐位一致。
FP16 大向量 CPU 求和可能对部分和作 FP16 舍入；设备结果另对照 FP32 累加后
转 FP16 的参考值。sum 的 dtype 会先转换输入，mean 的 FP32 输入则保持 FP32
累加后转换输出，与本机 Torch 的已测行为一致。

广播在 NPU 上物化连续输出，FP16 使用 DMA-only Broadcast 图；FP32 则按
每个数值两个 16 位存储片段广播，最后一维保留片段对，保证两个片段来自同一
FP32 数值。这不是 FP16 数值转换，原始位模式保持不变。公共 shape 仍最多
四维，内部 FP32 传输图可有五维。计算图不允许 DPU/SHAVE 计算任务混入这一
搬运路径，全部 FP16 位模式及随机 FP32 编码有逐位搬运对照。

Python int/float/bool 和实数 CPU 零维 Tensor 可作为标量参数；CPU 非标量
Tensor 仍拒绝。标量首次转换/上传到独立共享分配，按 dtype/位模式缓存至多
128 项；重复使用不上传，淘汰和 shutdown 会释放缓存。stats 的 live_buffers
因此包含缓存拥有的标量缓冲区，cached_scalar_buffers 单独列出。
FP16 标量乘除遵循本机 Torch 的右侧标量 FP32 opmath 路径；加减会先舍入标量。
例如 FP16 的 0.25 * 70000 得到有限的 17504，而加 70000 会溢出为 Inf。
左右输入顺序的边界行为也按实际对照验证，见
[PyTorch CPU BinaryOpsKernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp)。
bool 减法、非 1 alpha、非 None 除法 rounding_mode 明确拒绝。

设备矩阵乘法把 A(M,K)、B(K,N) 在 NPU 上广播为 (M,K,N)，进行 SHAVE
FP32 逐项乘法后归约 K 轴。FP16 输入也先升为 FP32，最后再转换输出；不会
让中间乘积在 FP16 中先溢出。K<=65,536 且超过 65,536 项乘积时，按 M/N 分块：以连续 DMA
StridedSlice 抽取输入行，列分块采用 DMA 转置后切片。每块独立计算完整 K 归约，
再以 U8 DMA Concat 拼接输出。超过 63 个
块时分层拼接，遵守原生图至多 64 个参数（含输出）的限制。每个乘积块最多
65,536 项，输入/输出各最多 1,048,576 项。K>65,536 时，对转置后的左矩阵与右矩阵
按连续 K 区间切片，计算 FP32 部分结果，再以补偿归约合并，最后转换输出。
这也覆盖大 batch 的 Linear 权重梯度，不再单独要求 K<=65,536。
没有 DPU MatMul 或融合乘加，尚未证明
优于 CPU/GPU 的速度。窗口展开与归约的其他编译器/CMX 布局校验仍会明确拒绝不支持的形状。
原始 x/w/bias 只在初始上传时复制，算子链内的广播/乘法/归约无主机载荷搬运。
out= 可复用最终分配，广播与展开乘积仍会分配中间缓冲区。

### 大 Tensor 的存储与计算分块

65,536 是已验证自定义计算块的保护上限，并非硬件的 Tensor 容量上限。
真实 intel_npu 后端现将存储上限独立为 1,048,576 个元素（FP32 4 MiB、FP16 2 MiB），
将大规模逐元素算术、ReLU/反向、精度转换拆成 <=65,536 项的小块，由 NPU DMA
切片与拼接。广播、完整密集转置和设备复制保持 DMA-only；不读取载荷到 CPU 计算。
torch_npu_minimal 的 CPU-staging 接口与旧 torch_npu_runtime 固定形状接口仍限 65,536 项。

大规模 sum/mean 先把非归约轴/归约轴排列成二维矩阵。完整归约宽度不超过 65,536
时，只分块输出行，保留每行完整归约；更宽的归约使用 FP32 补偿部分和再归约。
mean 最后按完整元素数除一次，正确处理不等长尾块。多级归约改变求和分组，
不承诺与 CPU 或旧顺序逐位一致。K 轴分块同样改变矩阵求和分组；部分结果始终
为真正 FP32，不能将“FP32 计算”理解成与单次完整补偿归约逐位相同。

卷积窗口/输入梯度折叠按静态索引分块：每次 Gather 的源区间不超过 65,536 项，
输出最多 8,192 项。源区间由 DMA 连续切片提取，索引只在 CPU 上作为几何元数据生成。
大型整体 Gather 曾触发 device-lost，或被编译成计算任务；后者被 DMA-only 校验拒绝。
新路径保持小块 Gather，失败报告保留在 research/results，未放松精度或搬运校验。

```powershell
.\run_torch_npu.ps1 -Large
# 分别在独立进程运行，避免跨案例累积静态图
.\run_torch_npu.ps1 -Large -LargeCase pointwise
.\run_torch_npu.ps1 -Large -LargeCase training
.\run_torch_npu.ps1 -Large -LargeCase conv
```

硬件检查覆盖 65,537 项尾块、257x257、1024x1024、较大位模式复制/转置、可微转换
与广播；FP16/FP32 Linear 分别验证 131,072 项权重和 65,552 项激活的三步训练。
卷积网络输入 2x3x65x65，含 Conv2d、BatchNorm、ReLU、全局池化和 Linear；
卷积输出 67,600 项，窗口/重叠折叠 228,150 项。三步损失 0.189275→0.147612，
最大梯度 relative L2 为 0.000154%。全部梯度、输入梯度和 SGD 参数更新按原有
CPU 容差检查；每步 CPU 权重重设为当前 NPU 权重，不认证独立训练轨迹。
预热步骤零测得的主机载荷复制/新图编译；验证下载另计，shutdown 分配归零。
报告：npunlock-probe/research/results/torch-npu-large.json，以及各 large-*.json。
完整大 Tensor 检查通过 10 组、7,263 次提交；原有 33 组设备检查、11 组训练检查
也通过，分别为 19,476、10,548 次提交。混合精度的 15+5 步检查通过；其 FP16
基线已有精度限制仍保留，因此混合精度总报告仍为 passed_with_limitations。
低层离线校验共 43 项通过。各回归报告保存为 torch-npu-*-large-regression-pass.json。

这仍是分块正确性路径，不代表性能加速或可使用整个约 17.9 GiB 预算。
rank<=4、每 Tensor 1,048,576 项、编译布局和有界缓存图的限制仍明确报错；缓存图数量从历史512项在S13调整到1,536项，运行中不淘汰。
较大形状仍需实际验证，不能只按输入分配大小判断可训练。

### K 轴分块与流式卷积训练

```powershell
.\run_torch_npu.ps1 -Stream
# 独立检查，自动创建新进程并设置 600 秒外层期限
.\run_torch_npu.ps1 -Stream -StreamCase k-f32
.\run_torch_npu.ps1 -Stream -StreamCase conv-f16
.\run_torch_npu.ps1 -Stream -StreamCase conv-geometry
```

当完整卷积窗口 M*K，或输入像素数乘 kernel 面积的折叠上界超过 1,048,576 时，
自动改用流式路径。NPU DMA 将输入转成 NHWC，窗口尽量按图像行对齐；单个 Gather
仍限制源区间 <=65,536、输出 <=8,192，保持原有 DMA-only 校验。普通窗口缓冲区
最多 65,536 项；单行 K 更宽时最多保留一行，仍受公共分配上限约束。
FP16 输入先在 NPU 上转换为 FP32，窗口与矩阵计算保持 FP32，输出最后转换回原 dtype。

流式 autograd 只保存原始 x/weight，不保存完整 im2col。权重梯度重新生成各窗口，
以 FP32 二叉归约树合并部分矩阵；输入梯度按输入像素收集相应输出梯度，
与排列为 (ky,kx,out_channel,in_channel) 的权重做矩阵计算，避免完整重叠折叠分配。
stride 的整除检查及 padding/dilation 索引在 CPU 上只处理几何元数据。
窗口搬运、乘法、求和、精度转换和 SGD 更新均在 NPU 上执行。
部分梯度的 FP32 树形合并改变求和顺序；没有声称等同于全局 Neumaier 累加或 CPU 逐位结果。

这条路径减少的是完整窗口/折叠临时存储与重复静态图，没有提供 DPU 卷积或专用 GEMM。
每步仍有大量同步图提交，尚未证明比 CPU/GPU 快。首次任意窗口分块触及 512 图上限的
失败报告保留为 torch-npu-stream-conv-cache-failure.json；后续采用按图像行对齐与
FP32 搬运图复用，保留缓存容量限制，不进行运行时销毁/重新导入。
硬件报告为 npunlock-probe/research/results/torch-npu-stream.json 及各 stream-*.json，
包含源代码 SHA256、逐步 CPU 对照、复制/编译计数、上下文和退出分配状态。
CPU 几何对照可运行 `.\.conda\npunlock\python.exe -m unittest test_torch_npu_geometry`。

本机新增硬件检查全部通过：6 个独立进程、10 组、22,589 次 NPU 提交。
FP16/FP32 矩阵检查覆盖 K=65,537 与 K=1,048,576、out=、相消/尾块与 beta=0；
65,537 项内部连续切片另对照原始位模式，包括 NaN/Inf/非规格化数。

| 三步训练案例 | 损失首步→末步 | 参数梯度 relative L2 最大值 | 缓存图数 |
|---|---|---|---|
| FP16 Linear，batch=65,537 | 0.197144→0.165527 | 0 | 152 |
| FP32 Linear，batch=65,537 | 0.197128→0.165562 | 8.22e-8 | 106 |
| FP16 Conv2d，1x4x176x176 | 0.047058→0.045959 | 3.69e-5 | 297 |
| FP32 Conv2d，1x4x176x176 | 0.047047→0.045969 | 1.41e-7 | 277 |
| FP32 Conv2d，2x2x176x176，stride=2/dilation=2，无 bias | 0.045576→0.044910 | 1.49e-7 | 295 |

前两种卷积的完整窗口/重叠折叠各需 1,115,136 项，现在按窗口块处理。
另检查密集转置的图像、权重和上游梯度，并对照回传到原始连续叶 Tensor 的梯度。
每个训练步骤都按原有 FP16/FP32 容差检查前向、输入/参数梯度及 SGD 更新，
CPU 参数每步重设为当前 NPU 参数；不认证独立训练轨迹。
所有预热后的步骤零测得的显式主机载荷复制/新增图编译；验证下载单独计数。
FP16 卷积预热步骤约 3.4 秒、FP32 约 3.3 秒，仍分别需要 2,405/2,319 次同步提交，
这些时间不能作为 CPU/GPU 对比结论。各进程退出分配归零，未发生 device-lost，
未安装 OpenVINO Python/runtime。离线检查为 43 个原有低层检查和 3 个 CPU 几何检查。

本次原有设备回归 33 组 / 19,476 次提交、训练回归 11 组 / 10,548 次提交、
大 Tensor 回归 10 组 / 7,263 次提交全部通过。混合精度 8 组 / 55,014 次提交
保持 passed_with_limitations：混合模式的 15+5 步对照通过，既有 FP16 基线限制仍在。
没有因为本次扩容放宽精度容差。回归报告保存为 research/results 下的
torch-npu-device-stream-regression-pass.json、torch-npu-training-stream-regression-pass.json、
torch-npu-large-stream-regression-pass.json 和 torch-npu-mixed-stream-regression.json。

### CPU / NPU / Arc GPU 性能实测

```powershell
.\run_torch_npu.ps1 -Benchmark -XpuPython C:\Users\LC\miniconda3\envs\blood\python.exe
# 也可单独测某个工作负载/精度
.\run_torch_npu.ps1 -Benchmark -BenchmarkWorkload conv_train -BenchmarkPrecision float32
```

本机 AC 供电、Balanced 电源方案，15 个工作负载/精度组合、30 个独立进程通过
首次调用的正确性对照。CPU 1/8 线程、NPU、GPU 及 GPU 环境的配套 CPU 基线共
75 组计时；全部输入/权重从公共 .pt 文件加载并核对 SHA256。
NPU 与主 CPU 使用 Torch 2.14.1+cpu，已有 blood 环境的 Arc 140V 使用
Torch 2.12.0+xpu，版本差异保留在报告中，没有修改环境或 NPU 内核。

FP32 热中位耗时：256x256 矩阵 NPU 5,920 ms，CPU 8 线程 0.0658 ms，GPU 0.1314 ms；
1x4x176x176 Conv2d 完整训练步 NPU 3,329 ms，CPU 1 线程 1.4426 ms，GPU 1.5151 ms；
小型四残差块网络训练步 NPU 559 ms，CPU 1 线程 2.0966 ms，GPU 6.5603 ms。
小任务中 CPU 线程多或使用 GPU 不保证更快。以上是驻留 eager 操作的 wall time，
包含每次操作末尾的 GPU 同步和每张 NPU 图自身的同步，排除初始传输、校验下载和最终输出对象释放。
FP16 Conv CPU 训练因本机 oneDNN 不支持半精度反向而使用原生回退；深层 FP16
SmallResNet 的既有精度限制未在性能测试中重新认证，残差网络只测 FP32。

所有 NPU 热计时零测得的主机 Tensor 复制/新增图编译。15 个 NPU 进程共
148,932 次提交，正常退出、分配归零。冷调用另记：上述矩阵/卷积训练分别约
44.2/40.6 秒，包含新图编译及执行，后端初始化独立记录。
256x256 矩阵每次仍需 1,030 张图，调用剖析约 94% 时间位于原生执行/等待，
归约路径约占 88%；卷积训练需 2,326 张图，约 84% 时间位于原生执行/等待。
原生计时包含驱动、DMA、SHAVE 与 fence，不能解释成纯硬件计算耗时。
这些结果表明当前展开乘积/通用补偿归约和同步提交路径需要优化，没有测量
硬件峰值、连续吞吐或能效，不能据此评价官方 DPU 推理路径。

完整方法、原始样本/IQR、传输计时、图提交数和版本控制基线位于
[性能报告](npunlock-probe/research/results/performance/BENCHMARK.md)，
以及同目录 benchmark.json、benchmark.csv、benchmark-fp32.png/.svg 和 .pstats。
测试脚本在首次 CPU backward 前注册 NPU，避免 Torch autograd 队列按未注册设备初始化；
早期测试流程失败报告保存在 performance/pilot，正式结果已修正流程并保持原有数值容差。

```python
x = torch.arange(15, dtype=torch.float32).reshape(3, 5).to("intel_npu")
w = torch.arange(20, dtype=torch.float32).reshape(5, 4).to("intel_npu")
bias = torch.arange(4, dtype=torch.float32).to("intel_npu")
y = ((x @ w) + bias) * 0.5
print(y.device, y.mean().cpu())
```

普通 nn.Linear 模块可以直接迁移。参数的 requires_grad 标记保留；推理使用
no_grad()/inference_mode()，训练使用下面的一阶 autograd 接口。eval() 本身不会关闭 autograd。
创建/初始化模块先在 CPU 上进行，NPU 随机初始化尚未实现。

```python
model = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Linear(4, 3)).eval()
model.to("intel_npu")
x = torch.randn(2, 5).to("intel_npu")
with torch.inference_mode():
    y = model(x)
    print(y.device, y.cpu())
```

Linear 将输入的前导维展平为 M，权重转置为 (K,N)，复用分块矩阵路径。
FP16 addmm/Linear 保持乘法、累加、alpha/beta 和偏置处理为 FP32，最后才舍入
为 FP16。例如 300*300-60000 得到 30000，避免先得到 FP16 Inf。beta=0 会忽略
偏置中的 NaN/Inf。采用分离乘法与顺序求和，没有保证与 CPU 的融合/求和次序逐位相同。
权重转置每次在 NPU 上物化；只缓存编译图，不缓存权重载荷，copy_/load_state_dict
更新后的参数会在下次前向生效。重复两层 Linear 前向也不需要主机中间载荷复制。

转置排列使用内部 U8 图，将每个 FP16/FP32 的 2/4 个字节作为不可拆开的末维。
图经检查只含 DMA 搬运任务，不允许 SHAVE/DPU 计算；对外 Tensor 仍为 FP16/FP32，
并未添加 Torch uint8 算术。共享会话执行层新增有界 U8 绑定支持，精度枚举依据
[Intel Level Zero graph 扩展头文件](https://github.com/intel/level-zero-npu-extensions/blob/main/ze_graph_ext.h)。
全部 65,536 个 FP16 编码和 65,536 个随机 FP32 编码的转置已做逐位对照。

可直接运行 .\run_torch_npu.ps1 -Script .\torch_npu_linear_demo.py，验证两层网络。
此前 Linear 里程碑检查通过 33 组、19,469 次 NPU 提交：86 项 addmm 对照、42 项
Linear 对照、转置的原始位模式/别名/寿命、44 项非法输入拒绝，以及线程和
shutdown 检查。FP16/FP32 两层 Linear 各运行 100 次，计 3,800 次提交，热循环
没有主机载荷复制和新图编译。结束后 live_buffers/live_buffer_bytes 均为 0，
context_failed 为 false。报告：npunlock-probe/research/results/torch-npu-device-linear-pass-1.json。

## 真实设备训练：Linear、Conv2d 与微型残差网络

### 兼容性修复（2026-10-04）

`torch_npu_compat_demo.py` 在五个独立进程中对照 CPU。极大/极小 FP32 除法的
分母梯度改用 `-(g/y)*(x/y)`，避免 `y*y` 先溢出或下溢；包含 0/Inf/NaN 对照。
mean 显式指定输出 dtype 时，反向先在输出 dtype 舍入除法，再转回输入 dtype，
修正 FP32 输入/FP16 输出的梯度差异。sum/mean 的两种跨精度组合均检查。
neg/abs/sign/reciprocal 新增一阶反向，含半精度 reciprocal 中间乘法的舍入/溢出规则。

完整密集的转置叶 Tensor 现在能执行算术、归约、clone、copy_、zero_、梯度累加
和基本 SGD；反向梯度保留叶 Tensor 的 strides。写回通过完整连续物理视图和逆排列
DMA 完成，不把偏移指针交给驱动。带孔/重叠布局、部分 Storage、非零 offset 仍拒绝。

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_compat_demo.py
```

优先在进程第一次任何设备的 `backward()` 之前 `import torch_intel_npu`。
PyTorch 引擎只初始化一次设备队列，先 CPU backward 再加载新 PrivateUse1 后端可能
触发内部 `device_ready_queues_` 断言；见 [PyTorch 引擎源码](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/engine.cpp)。
已验证以下局部调度办法，Tensor 算术/梯度仍执行 NPU SHAVE/DMA，退出后恢复原设置：

```python
# 已经先运行过 CPU backward、随后才导入 NPU 的进程
with torch.autograd.set_multithreading_enabled(False):
    loss.backward()
```

这是上游队列生命周期限制的兼容办法，普通晚导入 backward 尚未修复；没有重置
引擎或全局修改其他设备的调度。二阶导数、跨 CPU/NPU 可微传输、BF16/整数 Tensor、
RNG、Adam/动量 SGD、AMP、GPU/NPU 混合 Tensor 和完整 torchvision 模型仍不保证支持。
失败复现保存在 `npunlock-probe/research/compat-before/reproduced/`，修复后报告在
`npunlock-probe/research/results/compatibility/`。此前深层 FP16 训练的数值限制仍保留。

### 首轮性能优化

`resident_reduce.c` 保持原来的物理遍历顺序、Neumaier 补偿和非规格化数处理，
把每项动态除法/取模索引改为单轴步进或多轴进位。84 项 shape/axes/kind 的新旧
内核逐位对照通过，代表归约同进程交错计时加速约 1.43–1.66 倍。普通连续 Tensor
跳过密集 stride 排序和重复物化验证，但保留 dtype/device/Storage/offset/边界检查。

完整网络另有同进程对照：两个相同 FP32 SmallResNet 只切换新旧归约 ELF，
3 对预热、9 对测量，最终对照旧约 611 ms、新约 542 ms；12 对步骤的输入/参数梯度和
SGD 更新按原始 FP32 位模式对照通过，热循环没有显式主机载荷复制/新增图编译。
第一次跨轮网络测量存在较大波动，原数据保留；最终测量与局限见
[兼容性与优化报告](npunlock-probe/research/results/compatibility/REPORT.md)。仍明显慢于 CPU/GPU，
提交数没有减少，也尚未实现专用 GEMM/DPU 训练或异步提交。
用户在本轮看视频，桌面没有受控为空闲；跨轮耗时只作参考，同进程交错 A/B 仍可能有波动。

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_reduce_check.py
.\run_torch_npu.ps1 -Script .\torch_npu_resnet_ab.py
.\run_torch_npu.ps1 -Benchmark -BenchmarkWorkload mm_256 -BenchmarkPrecision float32 `
    -BenchmarkOutputDirectory .\npunlock-probe\research\results\performance-new
```

性能测试可另存目录、复用上轮固定输入/权重；resume 现在核对低层编译源、原生
扩展 build 与精度/驻留内核 manifest，避免更新内核后复用旧结果。

```powershell
.\run_torch_npu.ps1 -Training -Quick  # 每种网络/精度 3 步
.\run_torch_npu.ps1 -Training         # 每种网络/精度 15 步与梯度/边界对照
```

torch_npu_training_demo.py 使用合成数据测试 FP16/FP32 的两层 MLP、小卷积网络、
TinyResNet。TinyResNet 含 stem 卷积、两个卷积的残差分支、ReLU、空间 mean
和 Linear 输出；未含 BatchNorm，它不是 torchvision ResNet18/50 的完整适配。
测试逐步对照 CPU 前向、输入/权重/偏置梯度和 SGD 更新，并验证损失下降。
CPU FP16 参考关闭 oneDNN，使用本机支持半精度反向的基础实现；NPU 执行路径不受影响。

```python
import torch
import torch_intel_npu as npu

model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.ReLU()).to("intel_npu")
x = torch.randn(2, 1, 5, 5).to("intel_npu")
target = torch.zeros(2, 2, 5, 5).to("intel_npu")
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, foreach=False, fused=False)
for _ in range(15):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
print(loss.detach().cpu())  # 监测数值时显式下载
```

torch_npu_autograd.py 在 AutogradPrivateUse1 注册明确的一阶导数，用普通
torch.autograd.Function 保存设备 Tensor，前向/反向调用既有 NPU 内核。
没有把 Tensor payload 下载到 CPU 计算梯度。requires_grad、grad_fn、.grad 和
SGD 参数均为真实 PyTorch 对象；梯度分配属于 intel_npu。已验证分支梯度累加、
跨 backward 的 .grad 累加，以及保存参数被修改时的版本检查。二阶导数明确拒绝。
需要输入梯度时，可在上传后调用 .requires_grad_()；CPU requires_grad Tensor
直接上传的跨设备梯度路径尚未实现。训练标量参数使用 Python 数值或 NPU Tensor，
CPU scalar Tensor 参与可微算术暂不支持。

Conv2d 将窗口索引作为 CPU 静态 shape 元数据生成，U8 Gather/Concat 的 DMA-only
图搬运原始字节并补零。矩阵路径以 FP32 计算，最后转换输出。反向用矩阵乘法得到
窗口梯度和权重梯度，再以反向索引 Gather 加 FP32 ReduceSum 归并重叠输入梯度。
索引计算不读取图像、权重或梯度载荷。M*K*out_channels 的原有限制已经由矩阵
分块解除；超限窗口/输入梯度折叠采用上面的流式路径，公共输入/权重/输出仍须
<=1,048,576。小形状保留已验证的完整窗口/折叠路径，没有 DPU 卷积优化。

ReLU 及其反向使用新增的位操作 SHAVE 内核，保留零的符号、非规格化数与 NaN；
避免用 x*mask 导致 0*Inf/NaN 的错误。SGD 支持 momentum=0、weight_decay=0、
nesterov=False、foreach=False、fused=False、differentiable=False 的基本更新。
更新先在 NPU 计算独立结果，再由 DMA 写回参数；NPU/NPU copy_ 也改用 DMA。
预热后的训练步骤不需要主机载荷搬运；首次标量上传、输入上传和显式结果下载另计。

完整标准 ResNet、任意池化、交叉熵、Adam/动量 SGD、分组/转置卷积、AMP、
动态 shape 和训练加速尚未验证或实现。当前成果是有规模限制的正确性训练闭环，
没有证明它比本机 CPU/GPU 更快。报告保存在 npunlock-probe/research/results/torch-npu-training.json。

本次完整训练检查通过 11 组、10,548 次 NPU 提交；包含 102 项算术/矩阵/卷积
梯度对照和 65,546 个 ReLU 位模式/梯度边界。每种网络、每种精度训练 15 步，
损失均下降。TinyResNet 的 FP16 损失 0.160034→0.016724，FP32 为
0.296608→0.056952；每步参数和梯度均与 CPU 对照。预热后训练步骤的主机载荷
复制、新图编译都为 0；显式验证下载单独发生在测量步骤之后。shutdown 后
缓冲区/字节数均为 0，context_failed 为 false。完整推理回归另通过 33 组、
19,476 次提交；离线 Python 检查共 41 项通过。保存的完整报告为
npunlock-probe/research/results/torch-npu-training-pass-1.json 和
torch-npu-device-autograd-pass-1.json。

### 扩展训练：分块矩阵、BatchNorm 与多残差块

```powershell
.\run_torch_npu.ps1 -Models -Quick  # 3 步与新增算子对照
.\run_torch_npu.ps1 -Models         # 每个网络/精度 15 步
```

新增模型仍使用普通 PyTorch 模块和 `.to("intel_npu")`、`loss.backward()`、SGD：
宽 MLP 为 32→64→16、batch=64，共 3,152 个参数；SmallResNet 包含四个残差块、
十个卷积（含 stem 与投影分支）、8→16 通道变化、stride=2 下采样、BatchNorm、
ReLU、AdaptiveAvgPool2d(1) 与 Linear，共 11,020 个参数，输入为 2×3×8×8。
SmallResNet 的 FP16 路径使用 FP16 Conv/Linear 与 FP32 BatchNorm 参数；不是 AMP。
模型类和完整可运行测试位于 torch_npu_training_demo.py。

### 显式混合精度：FP16 权重存储，FP32 激活与计算

设备内 `.float()` / `.half()` / `.to(dtype=...)` 已支持可微转换，复用 Torch 原生
`ToCopyBackward`，正反向均走 SHAVE/DMA，不经过 CPU 数学或 NumPy。完整密集排列
也可转换；需要梯度的 CPU/NPU 上传/下载和 out= 转换仍明确拒绝。

```python
import torch
import torch_intel_npu
from torch_npu_training_demo import make_small_resnet

model = make_small_resnet(torch.float16, mixed_precision=True).to("intel_npu")
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, foreach=False)
# NPU 随机数尚未实现：初始数据从 CPU 上传。
x = torch.randn(2, 3, 8, 8).to("intel_npu")
target = torch.full((2, 4), 0.2).to("intel_npu")
optimizer.zero_grad(set_to_none=True)
loss = torch.nn.functional.mse_loss(model(x), target)
loss.backward()
optimizer.step()
```

此模式保持 10 个卷积的 10,712 个权重为 FP16，设备内可微地转成 FP32 工作权重。
卷积输入/输出、BatchNorm、ReLU、残差、池化、Linear 输出头和 MSE 均为 FP32；
另外 308 个参数为 FP32。卷积参数梯度回到 FP16，SGD 更新在设备内进行。
这是半精度参数存储方案，卷积实际调用 FP32 算子，不宣称 FP16 卷积吞吐优势；
转换产生 FP32 工作副本，也不保证训练峰值内存减少。按工厂设置精度后只迁移设备，
示例输入与标签使用 FP32。不包含 autocast、GradScaler 或 FP32 主权重。

仅保持归一化/残差/输出头为 FP32、卷积输入/输出仍为 FP16 的初次试验，不能消除
后期偏差；保存于 torch-npu-mixed-fp16-conv-quick.json。这是采用 FP32 卷积计算的原因。

```powershell
.\run_torch_npu.ps1 -Mixed -Quick  # 每个模式与场景 3 步
.\run_torch_npu.ps1 -Mixed         # 三种模式各 15 步，再从已有近收敛检查点各运行 5 步
```

完整对照采用相同初始数值权重/输入/目标；每种模式在自己的 NPU 权重上逐步重设 CPU
参考，因此比较的是每一步的设备数值一致性，不认证独立训练轨迹一致。三种模式的更新
轨迹会分叉，结果只适用于这些有限场景。混合精度与含 FP16 的路径使用原 FP16 逐项
容差 rtol=0.004、atol=0.0005，另外对混合精度要求整体梯度 relative L2 <=1%。

| 模式 | 15 步训练最大梯度 relative L2 | 近收敛 5 步最大梯度 relative L2 |
|---|---:|---:|
| FP16 Conv/Linear，FP32 BatchNorm 参数 | 8.374% | 36.009% |
| FP16 卷积权重存储，FP32 激活/计算 | 0.00779% | 0.06359% |
| FP32 | 0.00313% | 0.16185% |

混合精度的损失分别为 0.377763→4.35094e-5、1.01672e-6→3.67884e-7；全部梯度/更新
逐项检查及整体相对误差检查通过。FP32 后期损失更低，为 3.14633e-8，其梯度参考的范数
也更小，不能按此表断言混合精度比 FP32 更准确。FP16 初始场景存在逐项超差，作为测量
基线记录，不认证通过。总报告据此标为 passed_with_limitations。
8 组检查、55,014 次提交，热训练步的主机载荷复制和新图编译均为零；验证下载另计。
shutdown 后分配归零、上下文健康、OpenVINO runtime 未安装。
报告：npunlock-probe/research/results/torch-npu-mixed-fp32-compute-1.json。
与 Intel NPU Acceleration Library 的源码对比见
npunlock-probe/research/INTEL_NPU_LIBRARY_COMPARISON.md。

BatchNorm 的均值、有偏方差、归一化和导数在 NPU 上用 FP32 计算。新增 rsqrt
SHAVE 内核使用整数平方根与 FP32 倒数，覆盖非规格化数与特殊值；不调用 CPU/OpenVINO
数学运算。标准模块需使用 `torch.nn.BatchNorm2d(C, track_running_stats=False)`。
训练和 eval 均使用当前批次统计，遵循 [PyTorch BatchNorm2d 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)。
功能接口 F.batch_norm 已验证运行均值/无偏运行方差更新和 eval 梯度，但默认模块的
INT64 num_batches_tracked 尚不能随 `.to("intel_npu")` 迁移。BatchNorm 暂限 rank-4 NCHW；
池化暂限 output_size=(1,1)。未含交叉熵，因此测试使用合成数据与 MSE，不是分类基准。

FP32 网络与 CPU 独立训练对照。FP16 SmallResNet 每步将当前 NPU 权重下载给
CPU 参考，验证同一权重下的前向、全部梯度和更新。已定位原第 6 步的失败：
卷积归约的少量 FP16 舍入差异，经残差相加使一个 ReLU 从零变为正数，放大反向差异。
设备 sum/mean 改为 FP32 Neumaier 补偿累加后，该检查点的 ReLU 开关差异消失，
首层梯度 relative L2 从 6.25% 降为 0.174%。完整 15 步均通过原有逐项梯度/更新
容差（FP16 rtol=0.004、atol=0.0005；FP32 rtol=0.00007、atol=0.000006），没有放宽。
接近收敛时整体梯度 relative L2 仍可达 22.6%；逐项绝对容差通过不代表相对误差很小。
1024 倍静态损失缩放的额外实验也不能消除所有后期偏差。保留 `passed_with_limitations`
标记，不认证独立 FP16 训练轨迹与 CPU 一致；复杂模型的严格对照仍建议使用 FP32。
这些下载只用于验证，不参与 NPU 前向/反向/更新；未引入 CPU 数学回退或 OpenVINO。
诊断过程与报告索引见 npunlock-probe/research/FP16_TRAINING_DIAGNOSIS.md。

旧的 128 项图 LRU 在本机扩大训练时可重复触发设备丢失；改成 U8 广播不能解除，
64 项缓存则更早失败。保留图至 shutdown 后同样的矩阵/卷积测试通过。当前作为
保守规避，历史上将图缓存设为至多512项并取消运行中淘汰；S13因标准较长序列模型触及数量上限改为1,536项，仍保留至shutdown。超限时要求新进程，不自动
重试失败上下文。驱动内部原因尚未确认。失败与通过报告均保留在 research/results；
该规避不表示已修复 Intel 驱动本身。

补偿累加后的扩展检查运行 9 组、31,938 次 NPU 提交，79 项矩阵/卷积/BatchNorm/池化数值
对照，8,203 个 rsqrt 位模式，四个网络/精度组合各 15 步。宽 MLP 的 FP32 损失
0.056888→0.042879、FP16 0.046783→0.034454，均通过每步 CPU 梯度/更新对照。
SmallResNet 的 FP32 为 0.239699→1.75163e-7，也通过严格对照；FP16 为
0.258057→2.74181e-6，15 步同权重逐项对照通过，但后期整体梯度 relative L2
最高 22.6%，最低 cosine 约 0.97406。预热后的 NPU 训练步骤没有
主机载荷复制和新图编译（验证下载另计）。shutdown 后所有张量缓冲区归零，
context_failed=false，OpenVINO Python/runtime 未安装。完整报告为
npunlock-probe/research/results/torch-npu-models-compensated-1.json；原失败与修复前报告保留。
旧训练回归另通过 11 组、10,548 次提交，完整推理回归通过 33 组、19,476 次提交；
两者 shutdown 后均无张量缓冲区残留、无失败上下文。离线 Python 检查共 42 项通过，
新增 DMA Concat 硬件探针通过 6 个配置，包含 63 个输入加 1 个输出的原生边界。
回归报告为 torch-npu-training-model-extension-pass.json 和
torch-npu-device-model-extension-pass.json。

Python 端保留编译图/会话，C++ 端管理 Tensor 存储与原生提交；payload 在算子之间
直接绑定。torch.manual_seed 可正常设置 CPU 种子；NPU 没有 RNG 状态，随机算子仍拒绝。
npu.shutdown() 禁止新操作并关闭图，已有 Tensor 保留原生分配至最后一个 Storage 引用
释放。失败上下文不再允许数据复制/执行，沿用原生隔离规则，需要重启进程。

C++ 扩展针对当前 Torch 2.14.1+cpu 构建，导入时检查 Torch 版本、扩展/源文件及
session DLL 哈希；版本或源码变化须重建。构建不改变系统驱动。
项目内工具链为 MSVC 14.44.35207 / Windows SDK 10.0.26100.0，来源为微软安装清单，
载荷校验清单 SHA256，使用经审阅并限制清理路径的
[portable-msvc.py](https://gist.github.com/mmozeiko/7f3162ec2988e81e56d5c4e22cde9977) 解包。
首次准备可运行 .conda\npunlock\python.exe npunlock-probe\research\portable-msvc.py
--vs 2022 --target x64 --accept-license；该参数接受工具链许可证。
工具保存在 npunlock-probe/toolchain/msvc，build_torch_npu_device.ps1 临时设置编译环境。
设备扩展位于 npunlock-probe/torch-device，build.json 记录构建参数和哈希。

## 常驻缓冲区执行层

将下面代码保存为 my_resident.py，用 run_torch_npu.ps1 -Script 运行：

```python
import torch
from torch_npu_runtime import NpuRuntime

with NpuRuntime((1, 17), torch.float32) as runtime:
    a = runtime.upload(torch.ones(1, 17))
    b = runtime.upload(torch.full((1, 17), 2.))
    tmp, out = runtime.empty(), runtime.empty()
    for _ in range(1000):
        runtime.run("add", a, b, out=tmp)
        runtime.run("mul", tmp, b, out=out)
    print(runtime.download(out))  # 每个元素为 6
    print(runtime.stats())
```

每个 runtime 固定 shape/dtype。支持 add/sub/mul/div、add1/neg/abs/sign；
输入输出使用 zeMemAllocShared 分配的持久缓冲区，两个独立图共享一个上下文。
显式 upload/download 复制 CPU Tensor 数据；算子之间不使用主机载荷复制。
非连续 CPU 输入需要 packing，其额外复制单独计数。stats 的复制计数不包含
驱动内部 DMA、编译器或命令提交；没有据此宣称片上驻留或 GPU/NPU 零复制互通。

close/release 有引用计数和上下文锁保护；关闭父图后，子图与未释放缓冲区
保持有效。每次图提交等待 fence，线程提交按上下文串行；没有异步并行执行。
同一上下文复用一条命令队列，图二进制保留至图销毁。开发期间曾在重复多线程
检查中复现 device-lost；单独复用队列或固定提交线程均未消除问题。
补齐 C/Python 两层图二进制生命周期后，去掉了线程绕行，并连续通过三轮完整检查。
这是该配置下的实测结果，尚未证明 Windows 驱动内部故障的完整因果链。
输出不能与同图输入共享缓冲区；可使用两个输出/输入缓冲区交替迭代。
这一显式缓冲区接口不提供广播、cast、归约、MatMul、autograd；
普通 Tensor 的 .to("intel_npu") 由单独的 torch_intel_npu 设备接口实现。
每个逻辑缓冲区仍限 65,536 元素，最大 FP32 缓冲区 256 KiB；未测约 17.9 GB
设备预算、修改预算，或跨 GPU/NPU 驱动的指针共享。

若提交/等待返回错误，原生层隔离上下文，保留可能仍在使用的驱动资源至进程退出；
Python 层拒绝继续读写缓冲区。这是失败后的保护措施，恢复需要重新启动进程。
测试使用独立子进程及 180 秒外层期限，不故意在硬件上制造 device-lost。

## 精度实现

FP32 普通算术的输入、MemRef、步长和输出均保留 32 位。
accuracy 模式及 DisablePrecisionConversion 请求保留精度，补丁器另外
核对 dtype/stride；纯 FP32 图出现 DPU 任务或 FP16 MemRef 会被拒绝。

本机普通驱动 FP32 MatMul 在 accuracy 模式下仍产生 DPU 任务，
70000 等输入出现降精度/NaN。当前改为自定义 SHAVE mul + 逐级 add；
CPU-staging 接口归约也使用树状 add；真实设备接口使用独立归约内核。
尚未使用 DPU 的矩阵吞吐。
FP16 归约与 MatMul 使用 FP32 中间计算，最后显式转换为 FP16。
归约顺序与 CPU/BLAS 可能不同，不承诺逐位一致。

实测 SHAVE FP16 转换、驱动 FP32→FP16 转换及 FP32 算术存在
非规格化数清零行为。arithmetic.c 对 FP16 使用 IEEE 位转换和最近偶数舍入，
对 FP32 小数边界使用 SHAVE 整数运算；普通浮点算术仍使用 SAU.f32。
FP32 abs 也使用整数清除符号位；实测 SAU.ABS.f32 会将非规格化数清零。
NaN、Inf、有符号零、溢出和下溢都有测试。

FP32→FP16 的舍入由 SHAVE 完成，输出在 uint32 槽的低 16 位传回；
旧 CPU-staging 接口的主机只取位并解释为 FP16 存储，不对浮点数值舍入；
真实设备接口改用 NPU DMA 压紧存储。
FP16→FP32 使用已验证的 Convert SHAVE carrier，输入输出跨度不同。
同 dtype 的 cast 为复制，无需提交 NPU。
标量和 alpha 的舍入遵循 PyTorch 的 dtype 规则。

## 依赖和代码

- torch_npu_minimal.py：Torch 算子、FakeTensor、导数、FX 后端和缓存。
- torch_npu_runtime.py：常驻共享缓冲区、图会话复用、同步与复制计数。
- build_npu_session.ps1：使用本机 MinGW 构建独立 npunlock_session.dll。
- torch_intel_npu.py：真实设备的算子分派、图缓存与运行时注册。
- npunlock-probe/torch-device/backend.cpp：c10 allocator、设备钩子、存储释放与原生绑定。
- build_torch_npu_device.ps1：针对本机 PyTorch 构建 C++ 设备扩展。
- torch_npu_device_demo.py：带外层期限的设备、数值、生命周期和多线程检查。
- npunlock/python/npunlock/npu4000.py：算术/转换/归约 carrier 验证与补丁、DMA 半精度压紧和按位广播。
- npunlock/tests/fixtures/npu4000/arithmetic.c：4000xx 内核；
  precision/ 保存 ELF、汇编及 manifest.json。
- npunlock-probe/research/build_precision_kernels.py：通过已有隔离 Movi worker 重建。
- npunlock/tests/fixtures/npu4000/resident_reduce.c 与 resident/：设备归约源代码、ELF、汇编及 manifest。
- npunlock-probe/research/build_resident_kernels.py：复用 arithmetic.c 边界算法构建两个归约内核。

运行不导入或安装 OpenVINO Python / runtime 包；仍依赖 npunlock 原生 DLL、
Intel NPU 驱动、Level Zero 和驱动图编译器生成 carrier。
MoviTools 仅用于重建内核，不参与每次 Torch 执行。
静态 carrier 缓存最多 128 项。torch_npu_minimal 仍采用复制执行和全局同步锁；
torch_npu_runtime 使用新 DLL 的常驻缓冲区和每上下文同步锁。
历史普通图共享会话曾出现 device-lost；本次自定义 SHAVE 常驻路径检查未复现，
不能据此宣称所有图、驱动或长时间工作负载都已稳定。

新 DLL 只包含执行层和公共诊断代码，保存在 npunlock-probe/session-runtime；
build.json 保存编译参数及源文件/DLL 哈希。编译 IR、Movi worker 仍用现有发布包。
常规 npu.compile(npu.custom(...)) 自定义 C API 及其目标发现仍是 NPU3720 路径，
本地 NPU4000 使用新的 Python 接口。

## 验证

完整结果：npunlock-probe/research/results/torch-npu-precision.json，
日志：同目录 torch-npu-precision.log。
2026-10-03 此前完整精度检查：111 项数值/梯度检查、529 次 NPU 调用，约 24.7 秒。
后续补充 mean 的 FP32 输入/FP16 输出及标量乘除边界回归；最新 116 项、547 次
NPU 调用通过，约 25.4 秒。基础导数和五组 opcheck 仍全部通过。

- 全部 65,536 种 FP16 位模式的升精度、回转、取负和绝对值；
  NaN 按 NaN 等价比较，零另查符号，不承诺 NaN payload 保持。
- 8,192 对随机 FP32 位模式的加减乘除及转换；本次有限值算术最大 ULP 差为 0。
  除法容差为 3e-7，不能由这些样本推断所有输入逐位一致。
- 超过 FP16 范围、小差值、FP32 非规格化数、舍入中点和 Inf/零交互；
  另有保留这些值的 FP32 矩阵乘法测试。
- 两种 dtype 的多算子编译、广播、非连续输入、归约、基础反向传播，
  及 M*K*N=65,536 的矩阵乘法。
- unary、binary、cast、reduce、matmul 的 opcheck 各四项均 SUCCESS。
- 11 类输入/算子/隐式精度降级拒绝检查；实际设备始终为 8086:643e。

常驻执行结果保存在 npunlock-probe/research/results/torch-npu-session.json，
包括两种精度各 1000 轮递归 (x+y)*y：每种精度 2000 次图提交，循环中
零上传/下载/新增载荷缓冲区；两次提交使用同一个中间分配，无主机往返。
另覆盖标量、多维、2048/65536 元素、输入更新、各基础算子、符号零/非规格化数、
父子会话关闭、20 次创建/运行/关闭、4 线程串行提交、非法绑定及失败保护。
每轮含 4 线程合计 2000 次图提交。三轮记录见 torch-npu-session-pass-1/2/3.json，
汇总见 torch-npu-session-stress.json；早期失败保存在
torch-npu-session-queue-only-failure.json 和 torch-npu-session-thread-only-failure.json。
原生失败保护测试模拟 fence 超时、device-lost 与队列提交失败，不触发真实设备故障。

计时只比较相同小型链路的 copied 与 resident 提交路径；后者排除初始上传/最终下载。
它包含主机提交和 fence 等待，不是纯内核时间。矩阵乘法与归约对照使用数值容差，
尚未证明本实现比 CPU/GPU 快。
此前仅 FP16 加一的 torch-npu-hardware.json 保留为历史记录。

真实设备检查：npunlock-probe/research/results/torch-npu-device.json。
连续三轮结果见 torch-npu-device-pass-1/2/3.json 与 torch-npu-device-stress.json；
最终构建每轮 18 项、8,554 次 NPU 调用，约 9.2–9.4 秒（不含子进程导入），
包括 20 类不支持行为的拒绝检查，CPU 随机种子设置仍正常。
FP16/FP32 各有普通表达式与 out= 复用的 1000 轮递归链，两种路径中间均零载荷
主机复制；out= 路径零新增分配。覆盖标量、三维、65,536 元素、CPU/NPU/NPU 复制、
非连续 CPU packing、7 种算术的特殊值、Storage 视图寿命、4 线程共 400 次提交、
拒绝不支持的操作及关闭后的存储寿命。所有存活分配在最终释放后归零。
早期 GIL/原生锁停顿记录在 torch-npu-device-gil-failure.json 与
npunlock-probe/research/torch-device-thread-diagnostic.log；修复原生入口的 GIL 持有后
上述三轮通过。这些有限压力检查不构成长期或跨驱动版本稳定性保证。

设备转换/归约更新：22 组检查、10,387 次 NPU 调用，第一轮约 31.2 秒。
覆盖全部 65,536 种 FP16 位模式、8,192 种随机 FP32 转换输入、224 种
shape/axis/dtype 归约对照和 72 种特殊值/dtype 对照。NaN 按 NaN 等价比较，
不保证 payload；FP32 非规格化数、符号零及 FP16 溢出另作检查。
100 轮转换/乘法/sum/mean 链合计 900 次提交，循环中零主机载荷复制、零图编译。
FP16 out= 归约的压紧直接写入输出分配，无主机搬运；4 线程包含转换与归约，
最终存活缓冲区和字节均为零。最新记录为 torch-npu-device.json；
新版独立记录保存为 torch-npu-device-reduce-pass-1/2.json。
两轮均通过，共 20,774 次提交，第二轮约 31.6 秒，并增加默认非连续上传拒绝检查。
离线 Python 检查共 38 项通过，涵盖归约比例/代码指纹与 DMA-only 路径拒绝校验。

广播/矩阵更新：完整设备检查 28 组、14,343 次 NPU 调用，第一轮约 53.4 秒。
包括 204 个标量/广播/类型提升对照、62 个矩阵对照，以及 FP32 超 FP16 范围、
非规格化数、NaN/Inf、FP16 中间乘积溢出的边界测试。原始广播全位模式检查
保持 NaN payload 和符号零，算术结果的 NaN 仍只承诺 NaN 等价。
FP16/FP32 各 100 轮 mm(out=)+偏置广播+标量缩放+mean 合计 2,800 次提交，
预热后零主机载荷复制、零图编译。覆盖 128 项标量缓存的淘汰、4 线程矩阵提交、
错误输入拒绝和关闭后的 Storage 寿命，shutdown 后所有缓冲区归零。
新记录保存为 torch-npu-device-matrix-pass-1/2.json，最新记录仍为 torch-npu-device.json。
两轮均通过，共 28,686 次提交，第二轮包含 bool 减法/alpha 拒绝检查。
离线检查新增 FP16/FP32 广播形状、按位描述和计算任务拒绝检查，共 39 项。
