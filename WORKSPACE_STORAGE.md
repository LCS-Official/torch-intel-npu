# 工作区空间记录

2026-10-25 S40：通过仅针对intel_npu的构造分派接通torch.amp.GradScaler标准入口及提前导入别名，原类对象保留；复用CPU缩放控制与NPU梯度检查/反缩放，新增iterable/dtype语义及单元素设备FP32新scale。9个标准入口专项/模型进程与4个旧入口回归全部passed，共7,763次NPU提交；180项离线检查通过。SGD/AdamW溢出/累积、MLP三步与卷积/残差四步及恢复通过既定参考；CPU Half卷积严格差异保留。64组ELF/汇编及26份C源核对一致，无新依赖、内核或原生构建。报告见[S40](npunlock-probe/research/results/amp-standard/REPORT.md)。已删除13份真机JSON共57,678字节，清理前后831份保留文件指纹一致，随后补台账/报告；测试进程全部退出，source-release未改，未推送GitHub。仍要求单NPU、FP32主参数、显式foreach=False/fused=False；其他Torch版本/第三方构造扩展、长期AMP、原始Tensor序列化及通用compile未认证。

2026-10-25 S39：补SDPA空batch/序列/头及零QK/V特征，含mask零梯度/None和Dropout RNG语义；新增Tensor概率Bernoulli、九dtype输出及NPU INT64随机键排序randperm，带碰撞重抽、Generator/out及重放。FP16/FP32共42个attention边界用例，8种randperm dtype、8193元素跨包，既有三步GQA训练、随机及Generator回归均通过；15个正式进程共14,623次NPU提交，177项离线检查通过。64组ELF/汇编与26份C源指纹核验一致，无新依赖、内核或原生构建。报告见[S39](npunlock-probe/research/results/attention-rng-boundaries/REPORT.md)。已删除15份真机JSON共113,347字节，清理前后828份保留文件指纹一致，随后补充本台账/报告；所有测试进程退出，source-release未改，未推送GitHub。大规模洗牌性能、FlashAttention/NestedTensor、通用GradScaler/原生checkpoint及HF生成仍未完成。

2026-10-25 S38：存储统一按dtype实际字节计量用满1GiB，FP16/BF16/INT16容量翻倍、BOOL/INT8增至四倍；新增4,428字节密集MaxPool内核，在NPU生成窗口/索引，自动反向按有序几何剪枝，配对输出按dtype流式组装。九dtype1GiB分配/边界DMA、完整1GiB Half取负、4097×4097密集3×3前后向、最大1GiB输入及1GiB索引的FP16/FP32完整前后向均通过。20个主专项/回归和2个对照进程共1,168,033次正式提交，174项离线检查及468组宿主C参考通过。报告见[S38](npunlock-probe/research/results/storage-shape/REPORT.md)。重建Torch PYD，原session DLL与62枚ELF不变，更新u32边界并新增pool ELF后共64枚；无新运行依赖、模型、图表或磁盘checkpoint。开发大包版本的长等待/失败保留在报告，未计入正式通过；用户选择等待，没有执行进程注入或强杀。已清理74个原始/可重建/临时文件共14,765,213字节，清理前后522份保留文件SHA-256一致，随后补充本台账及报告；所有测试进程退出，公开source-release未改，未推送GitHub。1GiB单buffer、其他算子专用限制及同步提交仍在，未实现跨buffer分页或认证任意14GiB整网训练。

2026-10-25 S37：新增原始UINT32 header的选择反向内核，解除旧2²⁴单平面元数据保护，宽路径继续按小包执行并保留Half顺序舍入、赋值负零、padding与初值。FP16/FP32均完成16,777,219元素单平面MaxPool完整前后向、同长度极值/完整梯度及16,798,727条梯度累加；8个专项和10个既有回归进程全部通过，累计80,901次NPU提交，167项离线检查通过。报告见[S37](npunlock-probe/research/results/selection-capacity/REPORT.md)。原62枚ELF不变，新增ELF仅5,740字节；保留新C/ELF/汇编、构建manifest/脚本、Python派发/检查、CI及Markdown报告，无新依赖、模型、DLL/PYD构建或图表。全部进程退出后删除18份正式真机JSON、15份构建JSON和15份worker响应，共48个文件、368,452字节，清理前后510份保留文件SHA-256相同，随后补充本台账及报告；公开source-release未改，未推送GitHub。存储/单包及其他算子前向限制仍在，最大形状、密集大输出平面训练及约14GiB整网训练尚未认证。

S36（验收完成于2026-10-25）：普通MaxPool1d/2d空间输出保护扩至16,777,216、解除总采样量拒绝，选择反向按完整平面分组，浮点极值输入保护扩至16,777,216；窗口读取使用受限页对齐范围，并关闭不可微索引的无用零梯度实体化。FP16/FP32共12个新增容量专项与8个既有回归进程全部通过，累计50,295次NPU提交；163项离线检查通过。实际完成16,779,264源元素池化的完整前后向、9,474,084采样点重叠池化及两步CPU对照训练，报告见[S36](npunlock-probe/research/results/pool-capacity/REPORT.md)。无新依赖、模型、图表、SHAVE ELF或DLL/PYD构建。进程全部退出后清理20份原始JSON共96,164字节，清理前后504份保留文件SHA-256一致，随后补充本台账及报告；源码、内核、必要构建组件和Markdown报告保留，公开source-release未改，未推送GitHub。单平面元数据、3D/自适应前向、平均池化及一般索引等独立限制仍在，最大形状与14GiB完整训练没有认证。

2026-10-24 S35：普通Conv2d按262,144个窗口/fold元素选择流式路径，并把前向/输入梯度结果逐块写回最终缓冲；保留小窗口路径及现有存储/内核限制。3×3卷积后端Tensor分配峰值FP32为7.00→1.84MiB、FP16为7.35→1.63MiB，完整输出/三类梯度与两步训练CPU对照通过；小窗口多通道案例调用数和峰值不变。14份基线/最终记录共4,998次NPU调用、155项离线检查通过，报告见[S35](npunlock-probe/research/results/conv-capacity/REPORT.md)。保留S34四个卷积参考函数、新实现/验收/离线测试及CI列表；无新依赖、模型、图表、SHAVE ELF、项目持久图缓存或DLL/PYD构建。已清理14份原始JSON共75,356字节，清理前后501份保留文件SHA-256一致；公开source-release未改，未推送GitHub。最终布局/分组拼接、宽行及其他视觉/池化/一般索引边界仍需分别处理。

2026-10-24 S34：扩展Embedding/内部index_select的行数、长索引与特征列工作区，逻辑容量沿用存储保护，每列块仍保留16,777,216元素元数据界限；sort/topk总量保护扩至同一元数据上限。实际通过131,073×128 FP32词表完整梯度、FP16九列块压力、扩大排序/topk及两步111万参数模型训练。11组专项和3组原接口回归共62,109次最终验收调用，150项离线检查通过；报告见[S34](npunlock-probe/research/results/index-capacity/REPORT.md)。仅修改Python编排/测试与文档，无新依赖、模型、SHAVE ELF、图表、持久图缓存或DLL/PYD构建。清理14份原始JSON，共68,277字节，清理前后497份保留文件SHA-256一致；保留源码、必要构建组件和Markdown报告，公开source-release未改，未推送GitHub。一般索引反向、卷积/池化及其他专用边界继续保留，不宣称最大形状或14GiB训练认证。

2026-10-24 S33：低位Linear、Transformer、SmolLM和矩形GEMM图接通调用期弱引用重建工厂，缓存同时限制1536项/512MiB blob；超预算可重建图临时执行，prepared chain持有至结束。9组冻结模型/缓存压力、磁盘缓存命中及普通图回归通过。2.69亿参数网络完成两步FP32 AdamW，完整参数/梯度/两份状态4.001GiB、后端分配峰值4.251GiB，CPU抽样更新通过；耗时约35分钟，14GiB完整模型训练仍未认证。13个NPU进程共238,792次调用，146项离线单测和19项prefill缓存契约检查通过。报告见[S33](npunlock-probe/research/results/frozen-capacity/REPORT.md)。仅修改Python宿主/测试及文档，无新依赖、模型、图表、SHAVE ELF或DLL/PYD构建。已清理14份原始JSON及2份临时磁盘图缓存，共15,351,313字节，空缓存目录一并移除；清理前后495份保留文件哈希一致。保留源码、当前构建组件和Markdown报告，公开source-release未改，未推送GitHub。

2026-10-24 S32：单allocation上限1GiB，普通图缓存采用1536项LRU，执行计划持有的图延后释放，自定义冻结图保留pinned。FP16/FP32各1600个元素数、32项缓存下各6步独立训练、超过512MiB的完整梯度、九种dtype边界及14GiB多buffer驻留/NPU页读写通过；14GiB完整模型训练未认证。142项离线检查、Device Quick 33项、低位缓存保护与含单层SmolLM prefill/decode的安装回归通过。报告见[S32](npunlock-probe/research/results/graph-capacity/REPORT.md)。重建session DLL和Torch PYD；无新增SHAVE ELF、依赖、模型、图表或磁盘图缓存。保留源码、构建组件与报告，已清理9份原始JSON/日志共49,051字节，清理前后386份保留文件哈希一致；公开source-release未改，未推送GitHub。

2026-10-18 高维matmul/SDPA后续：新增rank 2–4批次广播与反向、标准SDPA及浮点mask/因果/safe softmax，并修复合并head时contiguous复制的autograd缺口。
新增13组全通过，含双头因果attention、LayerNorm/GELU/残差网络的FP16/FP32四步独立CPU对照及clone反向锁的双线程检查；连同原算子、Device Quick与迁移边界，最终59通过/2明确未支持，共13,780次NPU调用，另12项离线回归通过。
报告见[高维matmul、SDPA与可微布局](npunlock-probe/research/results/sdpa/REPORT.md)。新增一个SHAVE ELF/汇编及C/构建脚本；原22份ELF/汇编、C++/DLL/PYD内容未变，manifest扩展至23份内核。
保留实现、验收/审计脚本与报告；没有新增依赖、模型、图片或持久图缓存，未进行性能或长期训练认证；公开source-release未改，未推送GitHub。
验收后删除18份原始JSON/日志/编译响应，共156,049字节；清理前后94份保留文件的SHA-256一致。

2026-10-18 softmax/bmm后续：新增两个自定义SHAVE softmax前后向内核与标准三维bmm/一阶autograd，复用已有设备内转换、广播、补偿归约和分块。
新增11组检查全通过，含FP16/FP32小型attention网络四步独立CPU训练对照；共享分类与迁移边界回归后，最终23通过/4明确未支持，共6,547次NPU调用，另12项离线回归通过。
报告见[softmax/bmm与attention验收](npunlock-probe/research/results/attention-primitives/REPORT.md)。保留新C、两份ELF/汇编、manifest、构建/验收/审计脚本及报告；原20份resident内核及C++/DLL/PYD未重编。
未测试吞吐或长期训练，没有新增依赖、模型、图表或持久图缓存；公开source-release未改，未推送GitHub。
验收后删除22份原始JSON/编译响应，共185,543字节；清理前后89份保留文件的SHA-256一致。

2026-10-18 第5/6步归因后续：96次局部层重放和144次相同状态SGD重放，确认原两个失败点可由融合/分步乘加及half尾部舍入解释。
只改变CPU参照更新规则后，两网络六步参数/梯度一致；标准CPU严格失败保留。64训练/64独立验证样本任务32步后两侧准确率均75%、预测全部相同，严格轨迹仍失败。
最终验收19,074次NPU调用；报告见[SGD舍入归因与独立验证集](npunlock-probe/research/results/mixed-step-diagnosis/REPORT.md)。
保留扩展诊断脚本、新验证/审计脚本和报告，生产Python后端/autograd、20份resident ELF/汇编/manifest、C++/DLL/PYD未改。
无新增依赖、模型、图表或持久图缓存；公开source-release未改，未推送GitHub。

2026-10-18 分类精度/稳定性后续：新增FP64独立参考与两个32步混合网络探针，最终验收18,453次NPU调用。
10组分类精度检查通过；两个网络均拟合固定四样本且数值有限，但第6/第5步起有严格CPU轨迹偏差，32步严格检查均未通过。
生产Python后端、autograd、20份resident ELF/汇编、manifest及C++/DLL/PYD保持不变，无新增依赖、模型、图片或持久图缓存。
报告见[分类精度与32步边界](npunlock-probe/research/results/classification-precision/REPORT.md)；保留研究/审计脚本与报告，未推送GitHub。

2026-10-18 混合训练更新后续：修复Tensor add_的FP16 alpha转换，原tanh/FP32损失网络第三步梯度失败消除。
none/tanh两组均通过四步独立CPU对照；最终验收与辅助诊断16,346次NPU调用，46组通过，2组FP16损失严格失败保留。
65,536种half编码×10种alpha的数值相等检查、SGD/AdamW状态与残差训练回归通过；CPU小张量尾部仍不保证逐位相同。
报告见[混合训练更新修复](npunlock-probe/research/results/mixed-training-rounding/REPORT.md)。仅修改Python设备更新分派，未重编内核或原生扩展。
不安装依赖、下载模型、生成图片或启用持久缓存；保留代码和报告，公开source-release与GitHub未改动。

2026-10-18 混合LayerNorm后续：新增SHAVE有序Welford统计和整数融合验证内核，原4096宽混合梯度失败重放已与CPU逐位一致，22组扩大算子检查通过。
修复后有效验收与附加诊断17,911次NPU调用：主检查45组通过、5组明确未支持、3组混合整网严格失败；既有LayerNorm/GELU的14组回归全部通过。
默认GELU/显式FP32损失的一组四步混合训练通过，其他失败不计作支持；保留逐层与损失舍入诊断，见[混合统计修复报告](npunlock-probe/research/results/layernorm-mixed/REPORT.md)。
清理32份原始JSON/编译响应，共249,157字节（243.32KiB），清理时81份保留文件指纹一致。
保留源码、构建/测试/审计脚本、两个新增ELF/汇编及manifest、PyTorch BSD许可/来源指纹与报告；原有18份resident ELF及C++/DLL/PYD未改。
无新增依赖、模型、图片或持久图缓存；未推送GitHub，公开source-release保持干净。

2026-10-18 LayerNorm/GELU：新增同dtype FP16/FP32前后向、四个GELU SHAVE内核和六组残差前馈训练对照。
最终验收39,503次真实NPU调用，120组通过、5组明确未支持、1组宽层混合affine严格梯度失败；失败不计入支持范围。
12项离线回归通过；报告见[LayerNorm/GELU验收](npunlock-probe/research/results/norm-activation/REPORT.md)。
删除78份原始JSON/日志/编译响应，共429,771字节（419.70KiB）；清理时68份保留文件的SHA-256一致。
保留源码、构建/测试/审计脚本、新ELF/汇编/manifest及报告；C++、DLL/PYD和原有ELF未重建。
没有新增依赖、模型、图片或持久图缓存；公开source-release快照未改动，未推送GitHub。

2026-10-18 AdamW：新增设备端lerp_/addcmul_/addcdiv_、浮点标量add_与sqrt/maximum SHAVE入口，支持普通单Tensor AdamW及AMSGrad状态/CPU恢复。
验收与显式诊断合计42,035次NPU调用：110组检查通过、7组明确未支持、7组整网独立CPU轨迹失败；失败未计作通过。
三个FP32模型在关闭BN前冗余卷积bias后通过四步训练及恢复；纯FP16整网仍有限。已有迁移测试为29通过/7未支持，12项离线回归通过。
报告见[AdamW验收](npunlock-probe/research/results/adamw/REPORT.md)，保留失败参数位置、误差及代码指纹。
清理45份原始JSON/日志/编译响应，共183,389字节（约179.09KiB）；保留源码、构建/测试脚本、两个新增ELF/汇编及manifest和报告。
C++扩展、session DLL和既有ELF保持不变；无新增依赖、模型、图片或持久图缓存，公开source-release快照未改动，未推送GitHub。

2026-10-16 分类训练：新增有界log_softmax、NLL/CrossEntropy前后向四个SHAVE内核，修复纯FP16 BatchNorm保存统计的舍入边界。
11组分类检查（含252项算子数值比较和六种整网训练组合）、28组迁移功能/8组未支持边界、55组既有回归验收通过，
合计27,535次真实NPU调用；12项离线测试通过。报告见[分类训练验收](npunlock-probe/research/results/classification/REPORT.md)。
验收后清理53份原始JSON/日志、编译响应与临时诊断脚本，共290,403字节（约283.60KiB）。
保留源码、构建/测试脚本、新增ELF/汇编/manifest与Markdown报告；C++扩展、session DLL和既有ELF保持不变。
没有新增依赖、模型、图片或持久图缓存，公开source-release快照保持不变，未推送GitHub。

2026-10-16 INT64与默认BatchNorm：新增有界INT64存储、SHAVE精确标量计数，以及默认running buffers的模型迁移。
9组新增硬件检查、27组迁移功能/9组未支持边界、49组既有回归通过，最终18,536次真实NPU调用；8项离线检查通过。
重建当前C++设备扩展与新增INT64 ELF，已有浮点SHAVE ELF/session DLL保留；报告见[INT64与BatchNorm](npunlock-probe/research/results/int64-batchnorm/REPORT.md)。
验收后清理32份临时JSON/日志、编译响应与一次性布局诊断脚本，共114,468字节（约111.79KiB）。
保留源码、构建/测试脚本、当前内核/扩展与Markdown报告；没有新增依赖、模型、图片或持久图缓存，未推送GitHub。

2026-10-16 普通PyTorch迁移兼容性：新增25组计算/训练检查、11组明确未支持边界，
修复设备端mul_与带alpha加减法，扩展SGD动量/衰减及显式CPU checkpoint恢复。
最终迁移及49组既有回归合计13,629次真实NPU调用通过；C++/SHAVE/DLL/PYD未变。
报告见[迁移兼容性验收](npunlock-probe/research/results/torch-migration/REPORT.md)。
验收后清理31份临时JSON/日志，共163,849字节（约160.01KiB）；保留实现、测试、研究脚本与报告。
没有安装新依赖、下载模型、生成图片或启用持久图缓存，未推送GitHub。

2026-10-16 CPU启动准备优化：新增FrozenSmolIO轻量CPU两端，并让混合50%仅构造一次。
四种精度各3对新进程、12次完整32-token生成与旧入口逐位一致，22,320次真实NPU调用、20项离线测试通过。
报告见[CPU启动准备](npunlock-probe/research/results/smollm-startup-prepare/REPORT.md)。
NPU层计算与转换检查、图缓存身份、C++/SHAVE/DLL/PYD和默认精度未变。
验收后清理39份临时文件，共99,073,331字节（94.48MiB）；保留CPU入口、测试、研究代码与报告。
系统临时目录中的240份图缓存、1,386,207,776字节（1,321.99MiB）已逐精度删除，无新增依赖、模型或图片。

2026-10-16 W8/混合精度prefill提前缓存：三种策略3对相同blob的新进程对照，首token中位耗时降低约51%–53%。
53项离线检查、48项硬件检查及9,600次真实NPU调用通过；生产改动为缓存身份扩展与一个原函数身份记录，
C++/SHAVE/DLL/PYD和默认精度未变。报告见[低位prefill缓存](npunlock-probe/research/results/w8-prefill-cache/REPORT.md)。
验收后清理68份临时文件，共273,302,816字节（约260.64MiB），保留代码、修改前缓存源码和报告。
系统临时目录的210份配对图缓存、1,138,717,874字节（约1,085.97MiB）已逐策略删除；
没有新增依赖、模型、图片或持久图缓存，未推送GitHub。

2026-10-16 低位端到端效率验收：四种既定精度、7轮交替热请求、隔离host计时及已有缓存的新进程对照，
共98,400次真实NPU调用；报告见[端到端效率](npunlock-probe/research/results/w8-efficiency/REPORT.md)。
生产代码、内核、DLL/PYD、默认与公开快照未变。验收后删除13份临时权重/参考/计时/诊断产物，
共148,861,211字节（141.97MiB）；保留3份新增研究脚本及文字报告。
系统临时目录中4份图缓存共240个文件、1,321.96MiB已逐份删除，单份最多438.36MiB；
没有新增依赖、模型或图片，磁盘图缓存默认仍关闭。

2026-10-15 新文本PPL与自由生成研究：固定WikiText-2测试切片2,048个目标、8条新生成提示，
四种配置共57,480次真实NPU调用，内置loss/因果性/资源审计通过；生产默认及公开快照未变。
报告见[PPL与自由生成](npunlock-probe/research/results/w8-ppl-generation/REPORT.md)。
验收后删除11份JSON/NPZ，共107,044,464字节（102.09MiB），含临时语料、token IDs和校准权重；
保留3份研究脚本、完整生成文本及指纹报告。数据通过标准库读取，没有新增依赖、模型、图blob或图片。

2026-10-15 logits偏移与逐层敏感度研究：60项投影消融、6档内部候选，四组真实NPU共73,320次调用。
新增诊断确认原13个失败位置的误差能量平均约92.3%来自整体偏移；37.5%候选未取得独立质量优势。
生产接口/内核、默认与公开快照未变，报告见[误差诊断与逐层敏感度](npunlock-probe/research/results/w8-layer-sensitivity/REPORT.md)。
验收后删除11份JSON/NPZ，共104,362,936字节（99.53MiB）；只新增3份研究脚本及文字报告，
未下载模型/语料、安装依赖、保存图blob/原始logits/激活/图片或启用磁盘图缓存。

2026-10-15 敏感投影混合精度研究：新增默认关闭的本地FP16投影保护入口，
真实NPU质量从全W8的118/142改善到129/142；三组控制及局部检查共55,080次NPU调用，50项离线回归通过。
默认图逐字节一致，公开发布快照未改变。报告见[混合精度完整模型质量](npunlock-probe/research/results/w8-projection-mixed/REPORT.md)。
验收后删除本轮10份JSON/NPZ，共101,772,466字节（约97.06MiB），仅保留源码与文字报告；
未安装新依赖、下载模型、保存图blob/激活/logits/图片或启用磁盘图缓存。

2026-10-15 完整模型低位质量研究：新增顺序校准、分组INT8和完整NPU对照脚本，
正式有效NPU验证36,668次，候选尚未通过全部质量/压缩执行门槛，生产默认和公开发布快照未改变。
报告见 [完整模型低位质量对照](npunlock-probe/research/results/w8-sequential-calibration/REPORT.md)。
验收后删除本轮12份压缩权重/JSON，共197,329,937字节（约188.19MiB）；保留代码与文字报告。
没有下载新模型、安装依赖、保存图blob/激活/logits/图片或启用磁盘图缓存。

2026-10-11 GitHub 首发验收：非 Quick 设备/训练/模型共53组、61,958次NPU调用，Git克隆内45项离线测试通过。
删除本轮6份JSON/日志（40,649字节）及两份临时Git验证副本（17,530,146字节）；
仅保留约6.1MiB的最终源码快照与其发布Git元数据，记录见 [RELEASE_READINESS.md](RELEASE_READINESS.md)。
未安装新依赖、下载模型或创建新图缓存，原运行环境与内核未替换。

2026-10-11 发布兼容性基线：设备/训练/独立兼容性/小模型检查完成，合计58组、14,154次NPU调用；
小模型FP16保留训练轨迹限制，结果与指纹见 [COMPATIBILITY.md](COMPATIBILITY.md)。
文字记录验收后清理本轮17个临时JSON/日志，共71,904字节，保留历史Markdown报告；
未新增依赖、模型或磁盘图缓存，继续保留原环境、工具链与当前内核/宿主二进制。

2026-10-11 源码安装验证：新建 Conda 环境，从导出源码构建宿主组件，完成真实 NPU 运算、训练与单层模型检查。
验收后删除 `.conda/release-validation`（逻辑约944.60MiB）和 `source-release-validation`（逻辑约14.95MiB）；
保留 requirements、构建/运行入口、[安装说明](INSTALL.md)、[验证报告](INSTALLATION_REPORT.md)与约6.1MiB的干净源码快照。
未替换原 `.conda/npunlock` 环境和当前 DLL/PYD，未新增模型下载；不清理与其他环境共享的 Conda 包缓存。

2026-10-10，按用户要求，保留内核与优化相关源码、运行/重建所需文件及文字报告，清理可重建实验输出。

2026-10-11 延续同一策略：首token优化产生的113个临时文件约596MiB，验收后清理；保留新增研究源码与文字报告。
新的免解压缓存以额外空间换取启动速度，仍默认关闭且上限512MiB。开发验证后删除；实际使用若要保留启动收益，需要跨进程保留缓存。
本轮没有在C盘新建图缓存或安装新依赖。最新结果见 [冷首token报告](npunlock-probe/research/results/startup-first-token/REPORT.md)。

2026-10-11 后续提前命中优化验收后，删除143个临时文件，共760,057,299字节（724.85MiB），无失败项。
包括旧/新及交替对照图缓存、边界检查图、hidden/logits数组和原始JSON；保留研究源码、before文字源码及[跳过IR重建报告](npunlock-probe/research/results/startup-ir-cache/REPORT.md)。
清理后原生文件校验、严格FP32算术、FP16矩乘和一层prefill/decode冒烟检查通过（14次NPU调用，资源回收正常）。

| 项目 | 清理前 GiB | 清理后 GiB |
|---|---:|---:|
| E盘项目逻辑大小 | 20.63 | 4.57 |
| E盘项目实际占用（硬链接去重） | 15.33 | 3.83 |
| E盘可用空间 | 0.48 | 11.99 |

本轮删除 **9938个文件**，没有失败项。E盘项目实际占用减少约 **11.50GiB**。
逻辑大小与实际占用不同，因为历史文件曾做NTFS透明压缩，Conda部分文件采用硬链接。

C盘删除本轮60份SmolLM图缓存（322.15MiB）及npunlock旧ETL跟踪（8MiB），合计约330.15MiB。
没有清理其他项目、用户Conda环境或通用系统缓存。

## 已清理

- 全部实验目录中的NPZ/NPY张量、logits、KV与校准输入输出。
- 生成的IR/XML、blob、编译图、设备profile、图片、运行日志和原始计时JSON。
- 历史实验复制的DLL/PYD/ELF与无扩展名源码指纹快照。
- 已安装工具对应的旧下载包（EXE/WHL/ZIP），不再保留安装包副本。
- 本轮移动到C盘的图缓存以及旧npunlock调试跟踪。

保留129份Markdown实验报告，保留研究脚本、C/C++/汇编内核源码、补丁和文字说明。
报告保存清理前已经完成验收的结论；历史原始数据链接可能失效，离线audit不能直接重算已删除的数据。
需要复测时重新执行对应实验，产生新的参考和测量值。

## 仍保留的必要内容

- 当前SHAVE ELF及其源码/manifest、npunlock原生库、session DLL、PyTorch设备PYD和构建清单。
- 当前运行代码、测试代码、研究脚本、参考开源编译器源码与构建工具链。
- 项目NPU/OpenVINO对照环境、GPU模型依赖和约0.25GiB的测试模型输入。
- Conda已解包包缓存中存在与环境共享的硬链接；不把其逻辑大小误当成可直接释放空间。

清理后实际运行 **严格FP32加法、FP16矩乘、单层SmolLM prefill/decode**，共14次NPU调用通过；
原生文件指纹验证正常，最终live/cache/quarantine/session全部归零，无context failure。

## 后续约定

每轮交付以Markdown报告为主，把主要数值及必要样本写进报告后清理大型实验产物。
图缓存默认关闭；若使用`NPUNLOCK_GRAPH_CACHE`，会以磁盘空间换取下次启动速度，删除后首次需要重编译。
用户主动启用的缓存目录需自行选择；项目提供的默认缓存目录已纳入以下清理入口。

所有本项目实验退出后可预览并清理：

```powershell
.\npunlock-probe\research\clean_experiment_artifacts.ps1
.\npunlock-probe\research\clean_experiment_artifacts.ps1 -Apply
```

脚本只处理明确的实验目录、项目下载件和两个固定NPU用户缓存目录，保留源码与Markdown报告；
拒绝目录联接/符号链接，不递归删除运行环境、当前内核或工具链。自定义缓存路径不自动清理。

[最新模型优化报告](npunlock-probe/research/results/model-cross-device-followup/REPORT.md)记录本轮CPU/GPU快路径、batch4/8和NPU缓存验收。

历史：2026-10-07曾清理下载件并压缩原始实验数据，当时约18.29→13.47GiB；本轮按新的保留策略直接清除了这些原始实验数据。

2026-10-18 混合训练更新验收后清理：10份原始JSON，共198902 字节；清理前后88份保留文件指纹一致，随后补充记录。


2026-10-18 分类精度/32步验收后清理：两份原始JSON，共108594 字节；清理前后79份保留文件指纹一致，随后补充记录。


2026-10-18 第5/6步归因验收后清理：4份原始JSON，共600843 字节；清理前后83份保留文件指纹一致，随后补充记录。


2026-10-19 持续训练/独立NPU功耗验收后清理：14份原始记录及Intel便携工具，共572个文件、59838403字节；清理前后313份保留文件哈希一致。临时监控驱动正常卸载，无新运行依赖。

2026-10-19 DPU/CPU/GPU两轮稳态能效验收后清理：本轮输入、原始记录及Intel便携工具/下载包，共583个文件、74097313字节（70.665MiB）；清理前后668份保留文件哈希一致，随后补充报告与本台账。仅保留[能效报告](npunlock-probe/research/results/gemm-energy/REPORT.md)及研究源码，临时监控驱动正常卸载，无新运行依赖。

2026-10-19 向量/标量迁移验收后清理：9份原始JSON/日志，共57743字节；清理前后670份保留文件哈希一致，随后补充报告与本台账。保留[向量/标量迁移报告](npunlock-probe/research/results/vector-migration/REPORT.md)与研究源码。


2026-10-19 有界索引/Embedding验收后清理：20份原始JSON/worker响应，共186799字节；清理前后678份保留文件指纹一致，随后补充报告与本台账。保留[Embedding报告](npunlock-probe/research/results/embedding/REPORT.md)以及两个新索引ELF/汇编、C源码、构建与验收脚本。

2026-10-19 拼接/切片布局迁移验收后清理：28份原始JSON/日志，共91322字节；清理前后675份保留文件指纹一致，随后补充报告与本台账。保留[布局迁移报告](npunlock-probe/research/results/layout-migration/REPORT.md)、设备派发/梯度代码与新验收脚本；没有新增模型、运行依赖或内核二进制。

2026-10-19 BOOL/mask验收后清理：35份原始JSON、日志及Movi worker响应，共146324字节；清理前后696份保留文件指纹一致，随后补充报告与本台账。保留[BOOL/mask报告](npunlock-probe/research/results/bool-mask/REPORT.md)、新SHAVE C/ELF/汇编及manifest、重建的BOOL存储PYD和源码/测试；未下载模型、安装依赖或生成图表。


2026-10-19 RNG/Dropout验收后清理：20份本轮原始JSON、日志及Movi worker响应，共80411字节；清理前后704份保留文件SHA-256一致，随后补充文字报告/台账。保留新增SHAVE C/ELF/汇编/manifest、设备RNG与autograd源码及测试；没有下载模型、安装依赖或生成图表。报告见 [RNG/Dropout验收](npunlock-probe/research/results/rng-dropout/REPORT.md)。

2026-10-19 SDPA内部Dropout验收后清理：11份本轮原始JSON，共31031字节；清理前后707份保留文件SHA-256一致，随后补充文字报告与台账。保留设备派发/梯度源码、新验收脚本/参考测试及报告；无新内核二进制、模型、安装依赖或图表。报告见 [SDPA Dropout验收](npunlock-probe/research/results/sdpa-dropout/REPORT.md)。

2026-10-19 标准MHA/EncoderLayer验收后清理：16份本轮原始JSON/日志，共47871字节；清理前后714份保留文件SHA-256一致，随后补充文字报告与台账。保留设备派发/autograd、重建PYD及C++/构建记录、新验收与参考测试、双语文档和报告；没有新增SHAVE ELF、模型、依赖或图表。报告见 [标准模块迁移验收](npunlock-probe/research/results/standard-modules/REPORT.md)。

2026-10-19 稠密TransformerEncoder及FP16 native masked-softmax验收后清理：26份本轮原始JSON/日志/Movi响应与临时隔离脚本，共177268字节；清理前后720份保留文件SHA-256一致，随后补充报告与台账。保留新增SHAVE C/ELF/汇编/manifest、构建/派发/autograd/几何源码、正式验收与CPU测试、双语文档和报告；未下载模型、安装依赖或生成图表。报告见 [稠密容器与token分类训练](npunlock-probe/research/results/transformer-container/REPORT.md)。

2026-10-19 标准Transformer扩大模型/索引列分块验收后清理：8份本轮原始JSON，共36436字节；清理前后748份保留源码/报告/内核及必要运行时文件SHA-256一致，随后补充清理记录。保留设备派发/几何规划、扩大模型与索引验收、CPU测试及双语文档和[扩大模型报告](npunlock-probe/research/results/transformer-scale/REPORT.md)。复用原有SHAVE ELF，无新模型、依赖、宿主构建、图表或磁盘checkpoint；source-release未改。

2026-10-20 完整行attention/较长序列标准模型验收后清理：19份本轮原始JSON及Movi响应，共147771字节；清理前后750份保留文件SHA-256一致，随后补充报告与台账。保留统一行派发/几何规划、因果偏移C及更新ELF/汇编/manifest、验收/CPU测试、双语说明和[attention分块报告](npunlock-probe/research/results/attention-rows/REPORT.md)。新增内核ELF为7680字节；无依赖安装、模型下载、宿主组件重建、图表或磁盘checkpoint，未启用图磁盘缓存，source-release未改。

2026-10-20 转换图按元素数复用验收后清理：6份本轮原始JSON/日志，共38573字节；清理前后752份保留源码/报告/内核及必要运行时文件SHA-256一致，随后补充清理记录。保留最小_cast调度修改、新转换复用专项、双语说明和[转换图复用报告](npunlock-probe/research/results/cast-reuse/REPORT.md)。完整FP16模型、33组设备quick、归约转换梯度及68项离线检查通过；没有修改/重建SHAVE内核、宿主PYD/DLL、安装依赖或下载模型，磁盘图缓存仍关闭，source-release未改，未推送GitHub。

2026-10-20 兼容性优先的公开数学/激活/RMSNorm/Decoder验收后清理：56份本轮原始JSON及Movi响应，共325231字节；清理前后767份保留文件SHA-256一致，随后补充报告/台账和一处旧兼容性表述。保留七项公开数学/一阶梯度接入、标准RMSNorm、五个新SHAVE ELF（合计21988字节）/汇编/源码/构建manifest、门控/标准Decoder验收、三项CPU控制与CI列表、双语说明和[兼容性报告](npunlock-probe/research/results/unary-math/REPORT.md)。14组新增真机、16组既有AdamW检查和71项离线测试通过；没有新增依赖、模型下载、宿主PYD/DLL重建、图表或磁盘checkpoint/图缓存；低位量化未改，source-release未改，未推送GitHub。

2026-10-20 离散随机数/原生Generator兼容性验收后清理：42份本轮原始JSON/日志、Movi worker响应及可重建backend.obj/导出表，共6113158字节（5.830MiB）；清理前后777份保留文件SHA-256一致，随后补充报告与本台账。保留INT64随机数/FP16与FP32 multinomial的SHAVE C、两个新ELF（合计10420字节）/汇编/manifest、C++ Generator与重建PYD/构建记录、Python派发/autograd/CPU与硬件验收代码及[离散RNG报告](npunlock-probe/research/results/discrete-rng/REPORT.md)。52组正式硬件检查共6514次提交、74项离线检查通过；旧24字节RNG状态可恢复。旧import lib及session DLL保留；没有新增依赖、模型下载、监控驱动、图表或磁盘checkpoint/图缓存，source-release未改，未推送GitHub。

2026-10-20 有限AMP兼容性验收后清理：14份本轮原始JSON/日志及可重建backend.obj/导出表，共7533636字节（7.185MiB）；清理前后781份保留文件SHA-256一致，随后补充报告/台账。保留原生autocast策略、NPU专用GradScaler、重建PYD/构建记录、硬件/CPU检查、CI列表、双语说明及[AMP报告](npunlock-probe/research/results/amp/REPORT.md)。正式硬件39组通过、1组卷积有限通过，共7417次图提交；79项离线检查通过。卷积严格FP32梯度的三步差异在报告中保留为未通过，局部FP32累加参考及checkpoint重放通过，未放宽原严格阈值。无新增SHAVE ELF、依赖、模型下载、监控驱动、图表或磁盘checkpoint/图缓存；session DLL及旧import lib保留，source-release未改，未推送GitHub。

2026-10-20 AMP卷积舍入归因/残差训练验收后清理：仅删除四份本轮原始JSON，共15282字节；清理前后783份保留文件SHA-256一致，随后补充报告和本台账。保留CPU FP32/FP64与逐batch Half参考、六种卷积局部/两个四步网络验收、scaler空checkpoint错误修复、CPU测试及双语说明和[卷积归因报告](npunlock-probe/research/results/amp-conv/REPORT.md)。四个正式进程3907次NPU图提交、82项离线检查通过；通过的是独立FP32累加参考，原生CPU Half严格差异仍保留。没有新增依赖、模型下载、SHAVE ELF、宿主重建、图表、磁盘checkpoint或图缓存；source-release未改，未推送GitHub。

2026-10-21 S19常用接口迁移验收后清理：删除29份本轮原始JSON/日志，共89342字节；清理前后786份保留文件SHA256一致，随后追加报告与台账。保留分组/深度Conv2d、Conv1d、任意轴softmax、多维NLL/CE、label smoothing和rsub派发/梯度、新验收脚本及双语文档和[报告](npunlock-probe/research/results/portability/REPORT.md)。19个新专项加7个既有回归进程，共36294次NPU提交，82项离线检查通过；AMP明确使用FP16比较容差，旧严格差异保留。无依赖安装、模型下载、新SHAVE ELF、C++/PYD/DLL构建、图表或磁盘checkpoint/图缓存；source-release与GitHub未改。


2026-10-21 S20索引/平均池化/限幅兼容性验收后清理：按用户保留策略，删除本轮68份原始JSON/日志/Movi响应，共345132字节；清理前后790份保留源码、报告、内核和必要运行时文件SHA-256一致，随后补充本段与存储台账。保留源C、三个索引ELF/汇编/manifest及构建脚本、设备派发/autograd/几何、验收与离线测试和双语文档；无依赖安装、模型下载、C++/PYD/DLL重建、图表或磁盘checkpoint/图缓存。source-release与GitHub未改。 22个新增进程、70组旧回归及84项离线检查通过，28个正式进程共63165次记录提交。报告见[索引/池化/限幅验收](npunlock-probe/research/results/index-pool/REPORT.md)。


2026-10-21 S21极值/范数/标准梯度裁剪兼容性验收后清理：按用户保留策略，删除本轮35份原始JSON/日志/Movi响应，共199672字节；清理前后798份保留源码、报告、内核与必要运行时文件SHA-256一致，随后追加本段与存储台账。保留新C、两枚ELF/汇编/manifest、构建与验收入口、几何/派发/autograd、离线测试和双语文档。未安装依赖、下载模型、重建C++/PYD/session DLL、生成图表或磁盘checkpoint/图缓存；source-release与GitHub未改。 15个新增专项和4个旧回归进程（50组）按报告口径通过，85项离线检查通过；共33394次记录提交。长向量L1原CPU FP32严格差异单列保留，独立FP64参考通过。详见[极值/范数报告](npunlock-probe/research/results/reduction-norm/REPORT.md)。


2026-10-21 S22 MaxPool/原地ReLU/topk验收后清理：按用户保留策略，删除本轮44份原始JSON/日志/Movi响应，共251799字节；清理前后808份保留源码、报告、内核与必要运行时文件SHA-256一致，随后追加本段与台账。保留新C、三枚ELF/汇编/manifest、构建/验收、设备派发/autograd/几何、离线检查与双语文档。无依赖安装、模型下载、C++/PYD/session DLL重建、图表或磁盘checkpoint/图缓存；source-release与GitHub未改。 15个最终新专项、8个旧回归进程（50组检查/训练）、87项离线检查通过；23个正式进程共20374次记录提交。topk负零梯度逐位、MaxPool末NaN/重叠Half累加及原地版本保护已检查；仍有明确容量/可微view/k等边界。详见[MaxPool/topk报告](npunlock-probe/research/results/pool-topk/REPORT.md)。


2026-10-22 S23视觉/浮点排序验收后清理：删除本轮65份原始JSON/日志/Movi响应，共271581字节；清理前后817份保留源码、报告、内核与必要运行时文件SHA-256一致，随后追加本段与台账。保留1D/2D插值、GroupNorm/InstanceNorm、ConvTranspose、稳定sort/argsort/max/min及更大k的源码/反向、新排序C和8636字节ELF/汇编/manifest、构建/验收、CPU几何检查与双语文档。22个主专项20通过/2排序轨迹边界、12个旧回归全部通过；另有1个AMP运行统计严格FP32探针未通过，原误差和固定路由诊断分开保留，共35个最终记录、39871次NPU提交，91项离线检查通过。无依赖安装、模型下载、宿主C++/PYD/DLL重建、图表或磁盘checkpoint/图缓存；source-release与GitHub未改。


2026-10-22 S24视觉/排序扩展验收后清理：删除103份本轮原始JSON/日志/Movi响应及可重建backend.obj/导出表，共8093635字节（7.719MiB）；清理前后834份保留文件SHA-256一致，随后补充报告与本台账。保留五维/空Tensor的C++与重建PYD、3D/AA/池化/Unpool/padding/shuffle/Unfold/Fold/grid与排序/统计源码、一阶梯度、新验收/几何测试、45个resident ELF/汇编/manifest及必要运行时。22个新增专项和14个旧回归进程全部通过，共47501次NPU提交，95项离线测试通过；开发期设备错误、FTZ修复和数值口径分别保留。无依赖安装、模型下载、session DLL重建、图表或磁盘checkpoint/图缓存；source-release与GitHub未改。详见[S24报告](npunlock-probe/research/results/vision-sort-extended/REPORT.md)。

2026-10-23 S25容量/索引/dtype兼容性验收后清理：删除本轮227份原始JSON/日志、Movi响应及可重建backend.obj/导出表，共9443183 字节（9.006 MiB）；清理前后954份保留源码、报告、内核与必要运行时文件SHA-256一致，随后追加本段与台账。保留64MiB/rank8 C++与重建PYD、精确整数/部分BF16及动态索引/数学源码、一阶梯度、构建/验收/离线检查、61枚resident ELF/汇编/manifest和双语文档。41个最终真机进程全部通过，共53492次图提交；101项离线检查通过。无依赖安装、模型下载、session DLL重建、图表或磁盘checkpoint/图缓存；source-release与GitHub未修改。 详见[S25报告](npunlock-probe/research/results/compat-foundation/REPORT.md)。


2026-10-23 S26大Tensor/模型支持验收后清理：删除本轮44份原始JSON/日志、可重建backend.obj/导出表和已改动模块的bytecode，共8158505 字节（7.781 MiB）；清理前后957份保留源码、报告、内核与必要运行时文件SHA-256一致，随后补充本段与台账。保留256MiB/rank64存储与DMA源码、重建session DLL/PYD及构建指纹、较大点算/softmax/1704万参数训练验收、4项新增离线保护、双语说明及原61个resident ELF/汇编/manifest。17个最终真机进程通过，共57484次提交；105项离线检查及原生故障/链回归通过。未新增依赖、模型下载、SHAVE构建、图表或磁盘图缓存；source-release和GitHub未修改。 详见[S26报告](npunlock-probe/research/results/capacity-model/REPORT.md)。

2026-10-23 S27矩乘工作区/高维兼容性验收后清理：删除本轮55份原始JSON/日志、临时诊断脚本与Movi worker响应，共281837字节（0.269MiB）；清理前后964份保留源码、报告、内核与必要运行时文件SHA-256一致，随后补充报告和本台账。保留长K滚动补偿SHAVE C、新12,316字节ELF/汇编/manifest（总62枚）、构建/派发/密集梯度复制修复、rank64/空矩乘/宽Linear训练验收、6项新增离线检查、CI清单及双语说明。21份最终真机记录/66组检查全部通过，共28056次提交；111项离线检查通过。256MiB单allocation及1,536图缓存保留；合成258.26MiB partial流压力与真实GEMM范围分开记录。不新增依赖、模型下载、宿主C++/PYD/DLL重建、图表或磁盘checkpoint/图缓存；source-release和GitHub未修改。详见[S27报告](npunlock-probe/research/results/matmul-workspace/REPORT.md)。

2026-10-23 S28 SDPA/GQA兼容性验收后清理：删除本轮52份原始JSON/日志，共177912字节（0.170MiB）；清理前后967份保留源码、报告、内核与必要运行时文件SHA-256一致，随后补充报告及本台账。保留非空rank2–64/不同rank SDPA、GQA/MQA头复制及共享KV梯度、扩大score/广播输入检查与因果header保护、正式验收/5项新增CPU检查、CI与旧边界fixture及双语说明。25份最终真机记录/76组顶层检查全部通过，共28521次提交；116项离线检查通过。三步259参数GQA网络的近零K bias诊断、逐参数原容差与整体梯度L2判据单列记录。复用原62枚resident ELF，无依赖安装、模型下载、SHAVE构建、宿主C++/PYD/DLL重建、图表或磁盘checkpoint/图缓存；source-release与GitHub未修改。详见[S28报告](npunlock-probe/research/results/sdpa-extended/REPORT.md)。


2026-10-23 S29损失函数兼容性验收后清理：删除本轮60份原始JSON/日志、可重建字节码及临时校验清单，共795512字节（0.759MiB）；清理前后986份保留源码、报告、内核与必要运行时文件SHA-256一致，随后补充报告和本台账。保留加权NLL/硬标签CE、可微软标签/权重CE、广播混合dtype MSE、原生forward/backward/out及Half边界修复、验收/7项新增CPU测试、CI与双语说明。最终25个真机进程全部通过，新增404个测试组合、共35407次记录提交，123项离线检查通过；CPU Half级联求和、分类packet及高阶梯度等边界明确保留。复用原62枚resident ELF，没有依赖安装、模型下载、SHAVE或宿主重建、图表或磁盘checkpoint/图缓存；source-release与GitHub未修改。 详见[S29报告](npunlock-probe/research/results/loss-extended/REPORT.md)。

2026-10-23 S30验收后清理：删除本轮89份原始JSON/日志、可重建backend.obj/导出表、旧DLL/PYD临时回退副本及相关bytecode，共8935169 字节（8.521 MiB）；清理前后989份保留源码、报告、内核与必要运行时SHA-256一致，随后追加报告及本台账。保留512MiB/rank64容量、受检范围绑定与图复用/流式写回源码，重建DLL/PYD和构建指纹、验收/5项新增CPU测试、CI及双语说明，原62枚ELF/汇编不变。无新增依赖、模型下载、图表、磁盘checkpoint或图缓存；source-release和GitHub未修改。详见S30 memory-scale/REPORT.md。

2026-10-24 S31验收后清理：删除本轮55份原始JSON/日志，共250573 字节（0.239 MiB）；清理前后992份保留源码、报告、内核及必要运行时SHA-256一致，随后追加报告及本台账。保留硬标签损失分块/反向重建/标签复用、稠密工作区容量检查、验收及8项新增离线测试、CI清单和双语说明。24份真机记录通过，共69738次提交；136项离线检查通过。原62枚ELF、DLL/PYD不变，无新增依赖、模型、图表、磁盘checkpoint或图缓存；source-release及GitHub未修改。边界与数值口径见operator-capacity/REPORT.md。

2026-10-26 S41：补功能式arange、整数cumsum/isin/标签padding、空KV拼接及梯度、RoPE部分视图算术和native MHA/Encoder空输入。原版HF Llama/GPT-2两层随机小配置通过FP16/FP32 cache/贪心推理、FP32 top-k采样重放及两步训练/CPU checkpoint恢复；共享参数使用显式PyTorch交换转换。17组最终进程检查全部通过，共43,958次NPU提交；186项离线检查通过。157项内核/原生指纹一致，无新依赖、模型下载、ELF或DLL/PYD构建。报告见[S41](npunlock-probe/research/results/attention-migration/REPORT.md)。已清理19份原始JSON共118464 字节，清理前后896份保留文件指纹一致；公开source-release未改，未推送GitHub。FlashAttention、NestedTensor、top-p/beam、长期训练和任意大模型仍未认证。


2026-10-26 S42：新增SHAVE在线分块Attention及重计算Q/K/V梯度，标准SDPA可显式选flash；补strided NestedTensor推理、默认Encoder、MHA/LayerNorm、CPU往返/copy及Nested SDPA，CPU Nested前后向代理通过。21组最终进程检查共39,108次NPU提交，193项离线检查和16组/288个宿主C packet通过；两类HF小模型FP32训练/恢复及Llama FP16 cache/generate通过。128×256测例Tensor峰值降低86%–88%，含编译冷测试约慢3倍，未证明速度优势。新增两组ELF/汇编及一个C源，原64组、26份旧C源及DLL/PYD未改，162项指纹核对通过。报告见[S42](npunlock-probe/research/results/flash-nested/REPORT.md)。已删除43份临时文件共530092 字节，清理前后905份保留文件指纹一致；无新依赖/权重/环境，公开source-release未改，未推送GitHub。jagged/Nested训练、flash Dropout/可微mask及稳态性能仍需补充。

2026-10-26 S43：优化SHAVE分块Attention普通FP32算术热路径与query行缩放复用，按完整反向packet预算自动选择32/128或更小块，保留IEEE边界与32,768-word保护。128×256/head16的预热前后向FP32从3.586降至1.718秒，FP16从3.713降至1.845秒；图调用减少约70%，Tensor峰值仍比math低86%–87%，但该测例仍比math约慢一倍。23份性能/阶段/功能及HF整网记录通过，共64,220次记录提交；194项离线检查、16组/288个实际C宿主packet通过。重建两枚flash ELF，旧64枚、26份旧C源及DLL/PYD不变，162项指纹通过。报告见[S43](npunlock-probe/research/results/flash-optimization/REPORT.md)。删除41份本轮原始JSON、Movi响应与临时宿主构建文件，共614734字节，清理前后1949份保留文件指纹一致，随后补充报告与本台账。无新依赖、模型权重、环境、图表、磁盘checkpoint或图缓存；source-release干净且未更新，未推送GitHub。

2026-10-27 S44：新增默认关闭的W8＋FP16低秩残差准备与FrozenSmol Layer/Stack图接入；rank8/16/32内部选择32，30层decoder真实NPU独立回归平均logits误差2.2377%→1.7145%，通过118/142→126/142，decode常量比全FP16少38.01%。PPL收益不足0.01%，8条生成与W8相同，仍有16处未过原L2门槛、CPU最大KL超限，不替换默认；INT4/NF4整网未扩展。4份正式低位真机记录97,929次提交，额外恢复检查221次，203项离线检查及162项旧内核/原生指纹通过。W8一次编译超时重跑成功；FP16控制默认512MiB图缓存出现设备错误，768MiB重跑完整通过，原因未定位，生产预算未改。报告见[S44](npunlock-probe/research/results/w8-lowrank-quality/REPORT.md)。删除16份临时权重/因子、语料/token、JSON与AST对照，共125634330字节（约119.8MiB），清理前后1955份保留文件指纹一致。保留源码/测试/复现与报告；无新依赖、环境、权重下载、图表或持久缓存，source-release干净且未改，未推送GitHub。


2026-10-27 S45：修正W8低秩因子的prefill磁盘缓存身份；Smol图改为承压时同族失效重建，分组计划/固定族/超预算Smol临时图提前保护，新增公开graph budget setter，默认仍1536项/512MiB。原始原生库下两轮FP16前向/KV共660次、216次淘汰通过；256MiB安全拒绝后同进程512MiB重试通过，四种因子缓存miss/hit逐位一致。两个新校准候选未胜出，原NPU 16个失败位置完整复现，14处去整体偏移后剩余L2低于门槛但不放宽原标准；新128位置前向仍有token退化，INT4/NF4整网未扩展。7项正式真机进程（含旧四层FP16/W8容量脚本同步回归）共10,486次提交正常关闭，240项离线检查、4222位置严格标量审计及170项内核/原生/构建来源指纹通过。原生诊断改动均撤回，C++源码不变、session DLL恢复原值，本地PYD从未修改源码重建；无新依赖、环境、模型下载或图表。两阶段清理共54份原始JSON/JSONL/NPZ、临时图缓存、重复诊断源码/库及本轮obj/exp，共156,547,698字节（149.3MiB）；第一阶段2533份保留文件SHA一致，后补测试脚本/状态说明，末阶段同样2533份保留文件SHA一致，随后补充报告清理记录。见[缓存报告](npunlock-probe/research/results/smollm-cache-lifetime/REPORT.md)及[质量报告](npunlock-probe/research/results/w8-guarded-quality/REPORT.md)。最终只保留实现、测试、运行组件和Markdown报告，source-release未改，未推送GitHub。

2026-10-27 S48：新增量化前缀下down局部校准、整块输出回归及固定输入基约束，共8种候选；最佳input-local未带来稳定设备收益，NPU KV通过数仍82/98，PPL28.037196→28.041923，八条生成与基线相同，故不采用，不改生产默认或门槛。7个真机进程44130次提交正常关闭，265项本地离线检查通过；无新依赖、环境、模型下载、内核或原生构建。本轮删除47 份临时NPZ、JSON和日志，共147829860 字节（140.982MiB）。清理前后核对1931份源码、内核、文档、公开副本与运行组件SHA-256，全部一致，随后补充本清理记录及空间台账。研究目录仅保留REPORT.md；原模型、环境和全局缓存保留，所有本轮测试进程已退出。 保留两份复现源码、9项新离线测试和[报告](npunlock-probe/research/results/w8-output-quality/REPORT.md)；独立research/w8-output-quality分支保存本轮研究，v0.2.0 tag和原PR保持原内容。
