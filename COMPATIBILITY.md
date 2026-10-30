# 算子、模型与运行时兼容性

状态日期：2026-10-30，v0.3.0-alpha.1 Experimental Preview，累计纳入截至S60的实现和研究记录。适用范围为 Windows x64、Core Ultra 7 258V、NPU `8086:643e`、
驱动 `32.0.100.5540`、图编译器 `8.3`、Torch `2.14.1+cpu`。
这是当前实现和有限测试覆盖的清单，不表示所有形状、参数和组合均已验证。

设备名为 `intel_npu:0`。普通 ATen 后端与冻结 DPU 适配器是两条独立入口；
后者的矩乘/模型支持不能自动推广到 `.to("intel_npu")` 或通用 `torch.compile`。
安装方式见 [INSTALL.md](INSTALL.md)，性能口径见[中文首页](README.zh-CN.md#性能实测)。
下文阶段记录中的“本地开发版”或“未发布”保留当时的验证语境；截至S45纳入v0.2.0，S46–S60纳入本预览；支持范围仍以各项检查和限制为准。

## 状态与证据

- **已覆盖**：有对应 CPU/独立参考数值测试或明确的拒绝检查，只承诺这些检查范围。
- **有实现，有限验证**：源码提供路径，但不能据此保证任意参数或完整模型。
- **未支持/未认证**：不要把它当作发布承诺；某个简单组合能运行也不构成全面支持。

下表用 D、T、C、M 指向现有检查入口，具体函数/检查名称可在对应源码中查找：

| 标记 | 入口 | 主要覆盖 |
|---|---|---|
| D | [torch_npu_device_demo.py](torch_npu_device_demo.py) | 存储、精度、算术、归约、矩乘、布局、拒绝与生命周期 |
| T | [torch_npu_training_demo.py](torch_npu_training_demo.py) | 一阶梯度、Linear/Conv2d、三个小网络和训练约束 |
| C | [torch_npu_compat_demo.py](torch_npu_compat_demo.py) | 极值除法梯度、密集转置叶、单目梯度、归约转换、晚注册 |
| M | `torch_npu_training_demo.py --models` | BatchNorm、池化、分块梯度、较宽 MLP 与 SmallResNet；FP16 有限定 |
| P | [torch_npu_migration_demo.py](torch_npu_migration_demo.py) | 普通模型训练迁移、SGD 动量/衰减、CPU checkpoint 恢复、原地乘法与明确的未支持项；本地开发版 |
| B | [torch_npu_batchnorm_demo.py](torch_npu_batchnorm_demo.py) | INT64存储/精确计数、默认BatchNorm、running statistics、train/eval/reset与CPU checkpoint恢复；本地开发版 |
| L | [torch_npu_classification_demo.py](torch_npu_classification_demo.py) | log_softmax、NLL/CrossEntropy前后向、resident INT64标签、分类模型训练与CPU checkpoint恢复；本地开发版 |
| A | [torch_npu_adamw_demo.py](torch_npu_adamw_demo.py)及L的`--optimizer adamw` | AdamW更新、AMSGrad、状态/恢复与有失败记录的整网轨迹对照；本地开发版 |
| N | [torch_npu_norm_activation_demo.py](torch_npu_norm_activation_demo.py) | LayerNorm/GELU前后向、完整half编码、残差前馈训练及混合affine严格失败记录；本地开发版 |
| N2 | [layernorm_mixed_acceptance.py](npunlock-probe/research/layernorm_mixed_acceptance.py)与[内核检查](npunlock-probe/research/layernorm_mixed_probe.py) | 混合LayerNorm有序统计修复、22组扩大对照、融合数值语义及混合整网的明确失败边界 |
| N3 | [更新规则回归](npunlock-probe/research/add_alpha_regression.py)与[逐层轨迹诊断](npunlock-probe/research/mixed_training_rounding_probe.py) | Tensor add_的FP16 alpha转换；显式FP32损失的none/tanh混合网络四步严格对照；FP16损失仍有限 |
| N4 | [FP64分类对照](npunlock-probe/research/classification_precision_probe.py)与[32步边界](npunlock-probe/research/mixed_training_stability_probe.py) | 分类精度、CPU half溢出边界、原混合网络延长训练；有限学习通过不替代严格CPU轨迹验收 |
| N5 | N3的`--steps 6`/`--unfused-reference`与[独立验证集](npunlock-probe/research/mixed_validation_probe.py) | 原第5/6步偏差的SGD融合/half尾部舍入归因、相同状态动量/参数重放；64/64合成验证，严格失败保留 |
| S | [torch_npu_attention_demo.py](torch_npu_attention_demo.py) | softmax/bmm前后向、out、特殊值/分块、拒绝与version guard、Q/K/V梯度、小型attention网络四步SGD |
| S2 | [torch_npu_sdpa_demo.py](torch_npu_sdpa_demo.py) | rank 2–4 matmul/广播梯度、SDPA/因果与浮点mask、全屏蔽行、可微clone/contiguous、双头训练 |
| S3 | [torch_npu_sustained_train.py](torch_npu_sustained_train.py) | 4层793,604参数FP32合成任务持续训练、两步CPU梯度/更新对照、DXCore NPU忙碌/内存；另做用户授权的外部PMT功耗采集 |
| S4 | [torch_npu_vector_demo.py](torch_npu_vector_demo.py) | 向量matmul/dot/mv前后向与out、FP16/FP32同步item、四步向量头训练、迁移缺口探针；本地开发版 |
| S5 | [torch_npu_embedding_demo.py](torch_npu_embedding_demo.py) | 设备INT64动态索引、index_select/Embedding前后向、重复索引累加、padding_idx、越界与版本保护、普通词嵌入模型训练；本地开发版 |
| S6 | [torch_npu_layout_demo.py](torch_npu_layout_demo.py) | cat/stack、部分view/广播只读整理、slice/select梯度、split/unbind、原始位模式及分支网络训练；本地开发版 |
| S7 | [torch_npu_mask_demo.py](torch_npu_mask_demo.py) | BOOL字节存储/转换、六种比较、BOOL逻辑、where/masked_fill梯度、布尔SDPA及带padding/因果mask的attention训练；本地开发版 |
| S8 | [torch_npu_rng_demo.py](torch_npu_rng_demo.py) | SHAVE Philox随机数、FP16/FP32 rand/randn、种子/状态/fork_rng、普通Dropout一阶梯度与相同mask的训练/断点重放；本地开发版 |
| S9 | [torch_npu_sdpa_dropout_demo.py](torch_npu_sdpa_dropout_demo.py) | SDPA内部Dropout、FP16/FP32 Q/K/V及浮点mask梯度、广播/因果/全屏蔽行、设备RNG重放与双头训练；本地开发版 |
| S10 | [torch_npu_standard_modules_demo.py](torch_npu_standard_modules_demo.py) | 标准MHA/EncoderLayer、只读切片计算、baddbmm及梯度、默认推理入口与两层训练/checkpoint；本地开发版，范围见[报告](npunlock-probe/research/results/standard-modules/REPORT.md) |
| S11 | [torch_npu_transformer_demo.py](torch_npu_transformer_demo.py) | triu/tril、BOOL all/any、标准TransformerEncoder稠密容器、token分类训练与新模型checkpoint重放；本地开发版，见[报告](npunlock-probe/research/results/transformer-container/REPORT.md) |
| S12 | [torch_npu_transformer_scale.py](torch_npu_transformer_scale.py)及S5的`--case tiled` | 扩大标准token模型、Embedding/index_select列分块、训练梯度/更新和断点重放；本地开发版，见[报告](npunlock-probe/research/results/transformer-scale/REPORT.md) |
| S13 | [torch_npu_attention_rows_demo.py](torch_npu_attention_rows_demo.py)及S12的`--profile long --batch 4` | 完整行softmax分块、更大SDPA、因果全局行位置及标准模型较长序列验收；本地开发版，见[报告](npunlock-probe/research/results/attention-rows/REPORT.md) |
| S14 | [torch_npu_cast_reuse_demo.py](torch_npu_cast_reuse_demo.py)及S12的FP16 long配置 | 等元素数转换图复用、scalar/布局/out/分块精度及完整模型回归；本地开发版，见[报告](npunlock-probe/research/results/cast-reuse/REPORT.md) |
| S15 | [torch_npu_unary_math_demo.py](torch_npu_unary_math_demo.py) | 七项公开数学/激活与一阶梯度、标准RMSNorm、门控网络和标准Decoder迁移；本地开发版，见[报告](npunlock-probe/research/results/unary-math/REPORT.md) |
| S16 | [torch_npu_discrete_demo.py](torch_npu_discrete_demo.py) | INT64 randint/random_、FP16/FP32 multinomial、原生torch.Generator与生成token的训练/恢复；本地开发版，见[报告](npunlock-probe/research/results/discrete-rng/REPORT.md) |
| S17 | [torch_npu_amp_demo.py](torch_npu_amp_demo.py) | 原生FP16 autocast策略、FP32主参数与NPU专用GradScaler；含溢出跳步、训练及恢复，卷积严格梯度对照有限制，见[报告](npunlock-probe/research/results/amp/REPORT.md) |
| S18 | [torch_npu_amp_conv_demo.py](torch_npu_amp_conv_demo.py) | 卷积分批舍入归因、六种局部FP32/FP64参考、四步SiLU/三卷积残差AMP训练与恢复；scaler空checkpoint错误修复，见[报告](npunlock-probe/research/results/amp-conv/REPORT.md) |
| S19 | [torch_npu_portability_demo.py](torch_npu_portability_demo.py) | 分组/深度Conv2d、Conv1d、任意合法轴softmax/log_softmax、多维硬标签损失、label smoothing、rsub及普通分割/Transformer AMP训练与恢复，见[报告](npunlock-probe/research/results/portability/REPORT.md) |
| S20 | [torch_npu_index_pool_demo.py](torch_npu_index_pool_demo.py) | 有界gather/scatter_add/take_along_dim、平均/自适应平均池化、可微minimum/maximum/clamp、Hardtanh/ReLU6及梯度限幅，见[报告](npunlock-probe/research/results/index-pool/REPORT.md) |
| S21 | [torch_npu_reduction_norm_demo.py](torch_npu_reduction_norm_demo.py) | argmax/argmin、可微amax/amin、0/1/2及正负无穷向量范数、标准梯度范数裁剪与浮点分类谓词，见[报告](npunlock-probe/research/results/reduction-norm/REPORT.md) |
| S22 | [torch_npu_pool_topk_demo.py](torch_npu_pool_topk_demo.py) | MaxPool1d/2d及INT64索引/一阶梯度、非view非叶的原地ReLU训练、可微有界topk，见[报告](npunlock-probe/research/results/pool-topk/REPORT.md) |
| S23 | [torch_npu_vision_sort_demo.py](torch_npu_vision_sort_demo.py) | 1D/2D插值、GroupNorm/InstanceNorm、ConvTranspose1d/2d、稳定sort/argsort、max/min值与索引、更大k；严格整网轨迹与固定路由诊断分开记录，见[报告](npunlock-probe/research/results/vision-sort/REPORT.md) |
| S24 | [torch_npu_vision_sort_extended_demo.py](torch_npu_vision_sort_extended_demo.py) | 五维/空Tensor、整数/BOOL排序和次序统计；3D、AA、池化/Unpool、padding/shuffle、Unfold/Fold、动态网格采样与3步训练，见[报告](npunlock-probe/research/results/vision-sort-extended/REPORT.md) |
| S25 | [torch_npu_compat_extended_demo.py](torch_npu_compat_extended_demo.py) | 64MiB存储/rank8、整数/BF16转换与常用运算、动态高级/布尔索引、可写切片、gather/scatter及梯度、混合dtype拼接和210万参数独立训练；见[报告](npunlock-probe/research/results/compat-foundation/REPORT.md) |
| S26 | [torch_npu_capacity_demo.py](torch_npu_capacity_demo.py) | 存储256MiB/rank64、超旧64MiB的点算/平方/归约、扩大softmax行分块与1704万参数Linear训练；具体验收和边界见[报告](npunlock-probe/research/results/capacity-model/REPORT.md) |
| S27 | [torch_npu_matmul_workspace_demo.py](torch_npu_matmul_workspace_demo.py) | 长K滚动补偿累加、rank1–64/部分视图矩乘、空bmm/零K、密集梯度DMA复制及宽Linear+rank5矩乘两步训练；见[报告](npunlock-probe/research/results/matmul-workspace/REPORT.md) |
| S28 | [torch_npu_sdpa_extended_demo.py](torch_npu_sdpa_extended_demo.py) | SDPA rank2–64/不同rank广播、GQA/MQA与共享K/V梯度、超过旧attention容量、Dropout/RNG及高维GQA网络训练；见[报告](npunlock-probe/research/results/sdpa-extended/REPORT.md) |
| S29 | [torch_npu_loss_extended_demo.py](torch_npu_loss_extended_demo.py) | 加权NLL/硬标签CE、可微软标签/权重CE、广播混合dtype MSE、原生forward/backward/out及组合训练；见[报告](npunlock-probe/research/results/loss-extended/REPORT.md) |

| S30 | [torch_npu_memory_scale_demo.py](torch_npu_memory_scale_demo.py) | 512MiB存储、检查所有权/对齐/边界的范围绑定、偏移及逐元素形状图复用、流式分块与适中模型AdamW训练；见[报告](npunlock-probe/research/results/memory-scale/REPORT.md) |

| S31 | [torch_npu_loss_extended_demo.py](torch_npu_loss_extended_demo.py)、[torch_npu_operator_capacity_demo.py](torch_npu_operator_capacity_demo.py) | 硬标签损失行分块、反向mask重建、高样本数图复用，以及稠密算子专用容量；见[报告](npunlock-probe/research/results/operator-capacity/REPORT.md) |

| S32 | [torch_npu_graph_capacity_demo.py](torch_npu_graph_capacity_demo.py)、[torch_npu_memory_scale_demo.py](torch_npu_memory_scale_demo.py) | 普通图LRU、保留执行计划/冻结图、1GiB单次分配及超过512MiB的完整梯度；见[报告](npunlock-probe/research/results/graph-capacity/REPORT.md) |

| S33 | [torch_npu_frozen_capacity_demo.py](torch_npu_frozen_capacity_demo.py)、[torch_npu_memory_scale_demo.py](torch_npu_memory_scale_demo.py) | 冻结图可重建LRU、缓存字节预算、超预算图临时执行、四层decode计划生命周期，以及2.69亿参数/4GiB状态的两步FP32 AdamW；见[报告](npunlock-probe/research/results/frozen-capacity/REPORT.md) |

| S34 | [torch_npu_index_capacity_demo.py](torch_npu_index_capacity_demo.py) | Embedding按列拆分专用工作区，大词表/长索引反向，以及扩大sort/topk总容量；见[报告](npunlock-probe/research/results/index-capacity/REPORT.md) |
| S35 | [torch_npu_conv_capacity_demo.py](torch_npu_conv_capacity_demo.py) | 普通Conv2d窗口预算、逐块组装、CPU索引缓存与峰值内存对照，含特殊几何及两步训练；见[报告](npunlock-probe/research/results/conv-capacity/REPORT.md) |
| S36 | [torch_npu_pool_capacity_demo.py](torch_npu_pool_capacity_demo.py) | 普通MaxPool空间/采样量扩容、通道组反向、长轴极值索引与不可微输出梯度缓冲优化；见[报告](npunlock-probe/research/results/pool-capacity/REPORT.md) |
| S37 | [torch_npu_selection_capacity_demo.py](torch_npu_selection_capacity_demo.py) | UINT32选择反向元数据、超2²⁴单平面池化/极值及跨包梯度累加；见[报告](npunlock-probe/research/results/selection-capacity/REPORT.md) |
| S38 | [torch_npu_storage_shape_demo.py](torch_npu_storage_shape_demo.py)、[torch_npu_pool_dense_demo.py](torch_npu_pool_dense_demo.py) | 九dtype按字节存储、密集MaxPool几何反向及最大形状验证；见[报告](npunlock-probe/research/results/storage-shape/REPORT.md) |
| S39 | [torch_npu_sdpa_extended_demo.py](torch_npu_sdpa_extended_demo.py)的boundaries及[torch_npu_rng_extended_demo.py](torch_npu_rng_extended_demo.py) | SDPA空维度/零特征、Tensor概率Bernoulli与基于设备整数随机键的randperm；见[报告](npunlock-probe/research/results/attention-rng-boundaries/REPORT.md) |
| S40 | [torch_npu_amp_standard_demo.py](torch_npu_amp_standard_demo.py)及AMP demo的--scaler-api standard | 标准GradScaler构造入口、提前导入别名、iterable/dtype语义、Tensor新scale、SGD/AdamW与AMP模型恢复；见[报告](npunlock-probe/research/results/amp-standard/REPORT.md) |
| S41 | [torch_npu_attention_migration_demo.py](torch_npu_attention_migration_demo.py) | 原版HF Llama/GPT-2的有限迁移、KV cache/generate/训练入口，整数序列算子及native attention空输入；机制与实测见[报告](npunlock-probe/research/results/attention-migration/REPORT.md) |
| S42 | [torch_npu_flash_nested_demo.py](torch_npu_flash_nested_demo.py)及HF demo的--backend flash | 可选SHAVE在线分块/重算Attention、strided Nested推理、CPU代理检查、内存对照及HF整网回归；见[报告](npunlock-probe/research/results/flash-nested/REPORT.md) |
| S43 | [torch_npu_flash_optimization_demo.py](torch_npu_flash_optimization_demo.py)及S42回归 | SHAVE常规算术热路径、FP32次正规数前反向、自动32/128或容量受限分块；预热/冷调用、math对照与边界见[报告](npunlock-probe/research/results/flash-optimization/REPORT.md) |
| S44 | [w8_lowrank_quality.py](npunlock-probe/research/w8_lowrank_quality.py)及[审计](npunlock-probe/research/w8_lowrank_audit.py) | 可选W8＋FP16低秩修正、30层整网、固定独立回归/NLL/PPL/自主生成；尚未通过全部质量门槛，见[报告](npunlock-probe/research/results/w8-lowrank-quality/REPORT.md) |
| S45 | [缓存生命周期](npunlock-probe/research/results/smollm-cache-lifetime/REPORT.md)及[低位诊断](npunlock-probe/research/results/w8-guarded-quality/REPORT.md) | 因子缓存身份、Smol同族失效重建、分组计划预算保护及公开budget setter；真实KV复现原16个失败，新增128位置前向及FP16控制；新校准候选未采用 |
| S46 | [损失版本检查](torch_npu_loss_extended_demo.py)的--case versions | 分类损失布局复制后的来源版本、不同rank的CPU复制策略、跨packet与训练回归；见[报告](npunlock-probe/research/results/loss-source-versions/REPORT.md) |
| S47 | [W8逐层诊断](npunlock-probe/research/w8_layerwise_quality.py) | 30层/210投影、内部单层/投影FP16保护及独立CPU/NPU/KV对照；候选退化不采用，见[报告](npunlock-probe/research/results/w8-layerwise-quality/REPORT.md) |
| S48 | [量化前缀与输出校准](npunlock-probe/research/w8_output_quality.py)及[文本评测](npunlock-probe/research/w8_output_language.py) | 8种候选、独立CPU/NPU前向/KV/PPL/生成；候选未稳定改善不采用，见[报告](npunlock-probe/research/results/w8-output-quality/REPORT.md) |
| S49 | [设备数值诊断与内筛](npunlock-probe/research/w8_device_quality.py) | 五层同输入down链、九配置/FP16的真实NPU内筛、冻结选择后独立前向/KV；结果及边界见[报告](npunlock-probe/research/results/w8-device-quality/REPORT.md) |
| S50 | [门控拆分](npunlock-probe/research/w8_gate_quality.py)及[原型](npunlock-probe/research/w8_gate_experiment.py) | forward/decode的gate/up/SiLU同输入隔离及SHAVE激活内部对照；局部改善但整网未晋级，见[报告](npunlock-probe/research/results/w8-gate-quality/REPORT.md) |
| S51 | [scale联合校准](npunlock-probe/research/w8_scale_quality.py)及[共享输入诊断](npunlock-probe/research/w8_scale_audit.py) | 固定INT8 codes／rank32，局部SSE下降但整网未晋级；输入敏感峰值及候选拒绝处理见[报告](npunlock-probe/research/results/w8-scale-quality/REPORT.md) |
| S52 | [Attention分段诊断](npunlock-probe/research/w8_attention_quality.py)及[原型](npunlock-probe/research/w8_attention_experiment.py) | 26输出／213项精确分段控制；FP32 softmax局部改善但整网未晋级，见[报告](npunlock-probe/research/results/w8-attention-quality/REPORT.md) |
| S53 | [真实设备Q/K/V校准](npunlock-probe/research/w8_qkv_quality.py) | 固定rank32输出因子、32条校准／8条内部筛选；部分整网指标改善但未晋级，见[报告](npunlock-probe/research/results/w8-qkv-quality/REPORT.md) |
| S54 | [整网logits／KL拟合](npunlock-probe/research/w8_logits_quality.py) | 两轮既有输出因子拟合、冻结后新提示NPU对照；通过数略增但最大误差退化，见[报告](npunlock-probe/research/results/w8-logits-quality/REPORT.md) |
| S55 | [高误差整批校准](npunlock-probe/research/w8_tail_quality.py)及[CPU对照](npunlock-probe/research/w8_tail_audit.py) | 校准改善但三组NPU未晋级；新增高KL反例、CPU／NPU分歧与精度拒绝记录，见[报告](npunlock-probe/research/results/w8-tail-quality/REPORT.md) |
| S56 | [真实NPU校准选择](npunlock-probe/research/w8_native_selection.py) | 六个固定混合与设备预算、冻结候选及独立诊断；校准改善但原晋级拒绝，新提示局部守卫待定位，见[报告](npunlock-probe/research/results/w8-native-selection/REPORT.md) |
| S57 | [末层分段诊断](npunlock-probe/research/w8_final_layer_audit.py)及[down尾图对照](npunlock-probe/research/w8_final_layer_down.py) | 复现SiLU峰值和候选的解压权重舍入边界，CPU上传激活对照排除残余差异；局部反事实未推广为整网修复，见[报告](npunlock-probe/research/results/w8-final-layer-audit/REPORT.md) |
| S58 | [FP16解压代理试点](npunlock-probe/research/w8_decode_proxy.py) | 实际NPU down输入、两个固定单层拟合与新提示；局部误差预测更接近设备，整网仍未晋级，见[报告](npunlock-probe/research/results/w8-decode-proxy/REPORT.md) |
| S59 | [native-main拟合收尾](npunlock-probe/research/w8_native_down_check.py) | 主代理逐项匹配仍未提升内部通过数，结束单层校准方向，见[报告](npunlock-probe/research/results/w8-native-down-check/REPORT.md) |
| S60 | [INT4/NF4解码器验收](npunlock-probe/research/smollm_lowbit_coverage.py) | 显式30层前向与有限KV，区分本格式执行正确性和原模型质量，见[报告](npunlock-probe/research/results/smollm-lowbit-coverage/REPORT.md) |

## S46 分类损失的原始Tensor版本保护（v0.2.0之后）

CE、NLL及原生nll_loss2d_forward补齐布局复制后的原始Tensor版本保护：4D非连续硬标签及其共享storage别名被修改时，反向按本机CPU行为拒绝；空间NLL还保留4D原始输入以及其他空间rank的连续原始输入版本。3D/5D复制非连续源后的接受行为与CPU一致，不一律增加拒绝。重复retain_graph反向继续检查版本，no_grad入口保持原行为。

共享检查只保存PyTorch原始Tensor引用，在相关反向计算前解包触发版本检查；不新增设备计算、CPU payload回退或SHAVE/宿主构建。仍仅承诺已覆盖的一阶路径，不提供任意view原地写入或高阶梯度。FP16/FP32真机、跨NLL packet与训练回归见[报告](npunlock-probe/research/results/loss-source-versions/REPORT.md)。

## S47 W8逐层质量诊断（v0.2.0之后的研究）

新增30层/210投影误差定位，区分相同输入的局部误差、前缀传播和非线性交互；内部8条/290位置筛选30种整层和7种单投影FP16保护。第25层整层保护虽在内部246→250/290，独立NPU前向历史126→125/142、新集合113→111/128，原失败轨迹KV82→81/98，故不采用。210个投影的rank32局部误差均降低，但跨层累积及整网门槛仍未解决；去均值仅作诊断。

五个最终真机进程15,390次提交均正常关闭，FP16前向270/270通过；没有生产后端、内核、默认参数、依赖或原生构建变化。复现、原16个失败位置及边界见[报告](npunlock-probe/research/results/w8-layerwise-quality/REPORT.md)。INT4/NF4整网、PPL与自由生成未新增认证。

## S60 INT4/NF4解码器入口与主线模型覆盖

`FrozenSmolLayer`／`FrozenSmolStack.from_checkpoint`新增`w4a16`、`nf4a16`，默认group32；也可提供格式／形状匹配的完整`PackedWeight`字典。保持FP16输入、输出和KV，压缩常量在图内解压后进入FP16 DPU，并非原生INT4/NF4乘加或低位训练。原FP16/W8默认与参考保持，FP16保护和rank32修正仍只用于W8。

NF4单行投影在图内补至32行后裁回，避开8.3编译器的`LowFpType or levels attribute should has value`失败；公开token、attention和KV形状不扩大。图检查按格式要求DPU证据，W8仍必须有INT8矩阵任务，四位解压路径必须有FP16矩阵任务。新增模式绕过仅适用于FP16/W8的磁盘快捷缓存，仍使用现有内存图缓存。

本机B1/S64的30层teacher-forced前向，以及8-token prefill和第9个token decode分别认证；旧KV快照不变。局部参考明确使用实际四位权重的FP16解压值，沿用L2<0.003、scaled max<0.01；原模型质量仍以独立原权重teacher的raw logits L2<0.03、KL<0.02判断。两类检查不可互换，当前简单四位量化未达整网质量门槛。

NF4默认512MiB缓存会发生LRU淘汰和热路径重编译；输出及90项局部检查通过，原记录中的缓存复用断言未通过。完整热路径验收用显式1GiB图预算，生产默认未改。不是吞吐／内存优势认证，尚未覆盖新模式的自由生成、PPL、grouped decode、其他batch／长度、通用Hugging Face迁移或训练。后续按真实模型缺口推进，不再无限追加单层校准分支。见[报告及用法](npunlock-probe/research/results/smollm-lowbit-coverage/REPORT.md)。

## S59 单层校准方向收尾

最后一次固定native-main末层down拟合使主代理与NPU输出逐项一致、局部SSE下降约4.96%，但三组内部仍243/290且新候选最大KL更差。原规则保留baseline，不做新提示晋级；结束这条单层拟合方向。4个进程2686次提交、120项局部检查及4项观测／绑定控制通过并健康关闭，无新生产默认；详见[报告](npunlock-probe/research/results/w8-native-down-check/REPORT.md)。

## S58 FP16解压拟合代理与末层down试点

在原48条校准轨迹的3072行真实NPU down输入上，对照精确code×scale与显式FP16解压代理。新代理主投影相对L2降低59.17%，基线局部SSE预测偏差22.18%→0.60%；但一致比例86.66%→56.22%，scaled max上升，不是DPU逐位模拟，不替换原独立参考。

两种固定alpha=0.5的rank32输出因子拟合仅改变末层down的18,432个系数，其他209因子、所有输入基和packed权重不变。实际局部校准SSE下降4.9790%／4.9321%，内部均243/290且平均L2变差，新代理最大KL也升高；原规则保留baseline。内部择优的exact-a50在八条新提示234→233/260，decoded-a50本轮未做新提示NPU验收，不将该结果归为解压收益。

6个正式进程3,346次提交及180项已执行局部检查通过；另一次重复记录字段错误在健康关闭后修复重跑，额外1,600次提交。七个实际进程均健康关闭，374项完整工作区离线检查通过，228份fixture及16份旧身份未变。无生产后端、内核、依赖、原生构建或精度门槛变化；INT4/NF4等整网质量范围未扩展。见[报告](npunlock-probe/research/results/w8-decode-proxy/REPORT.md)。

## S57 末层SiLU与权重解压舍入诊断

基线和S56固定旧候选在zipper的(0,61,17)坐标均原样复现0.5差异；固定顺序重放在SiLU阶段引入该差值。SHAVE激活／乘法将基线scaled max由0.0106282降至0.0047944，但候选仍0.0102929。对照提示保持通过，未放宽原0.003／0.01局部守卫。

上传CPU精确激活乘积后，候选仍失败：同输入down main的NPU／精确权重／FP16解压权重结果为-583.5／-584.0／-583.5，低秩delta相同，残差565.0抵消后成为-18.5／-19.0。这是特定输入／坐标上的证据，整个矩阵仍有差异；不能据此归因所有模型误差或宣称已修复候选。下一步优先让拟合代理模拟FP16解压，独立teacher、原守卫与默认维持。

4个正式诊断进程3,260次提交均健康关闭；120项原局部检查118通过、两项拒绝完整记录，54项切图／观测／参考控制逐项相同。364项完整工作区离线检查通过，228份fixture和12份旧身份不变。无生产后端、内核、依赖、原生构建或新整网质量分数，INT4/NF4范围未扩展。见[报告](npunlock-probe/research/results/w8-final-layer-audit/REPORT.md)。

## S56 真实NPU校准选择与新提示守卫

六个预先固定的既有rank32输出因子混合方案，在48条／1,657位置的真实NPU输出上选择；原INT8编码、scale、输入基、参数字节和推理图结构不变。用同一目标与逐位置NPU基线预算，最佳tail-e2-a100将目标降低8.60%，通过1604→1622/1657；选择及七份校准记录哈希在内部／回归／新提示前冻结。

原内部晋级仍拒绝：243/290不变、最大KL升高、token一致数下降，最终保留baseline。16条已查看回归491→487/528；新8条的首条zipper在基线／候选末层scaled max为0.0106282／0.0102929，均越原0.01守卫，因此没有有效的NPU 268位置分数。CPU新集238→232/268也退化，既有校准泛化问题，也有待定位的局部边界；不能只归因驱动。

13个正式NPU进程11,250次提交，10个完整完成、3个精度拒绝，全部健康关闭；390项局部检查387通过。354项完整工作区离线检查通过，228份fixture及9份旧身份未变。仅新增研究／测试与报告，无生产后端、内核、依赖或原生构建变化；INT4/NF4、KV、PPL及自由生成范围未扩展。见[报告](npunlock-probe/research/results/w8-native-selection/REPORT.md)。

## S55 高误差约束与校准泛化

使用48条校准提示，固定INT8编码／scale和rank32输入基；最终拟合全部210个既有输出矩阵（4,976,640参数），保持Half前向。目标增加top10% L2／KL及逐位置baseline超额惩罚；整批梯度、实际校准分数回溯使两次目标下降1.64%／9.26%，不作为NPU收益证明。

tail-e1在末层scaled max=0.010120984触发原0.01守卫，未继续给出整网质量；tail-e2完整执行但内部仍243/290、最大KL升高，原规则保留baseline。冻结后S54回归246→245/262，新八条提示245→242/266。新提示shadow/62的量化CPU／NPU基线KL约0.034，均越过0.02，且不位于生成EOS后段；存在模型／校准泛化缺口，不能只归因驱动。

7个正式NPU进程共2,070次提交，6个完整完成、1个按原精度规则拒绝，7个均健康关闭；210项局部检查中209通过。340项完整工作区离线检查、另6组CPU整网审计完成，228份fixture及7份旧身份未变。推理仍30次预热提交，无生产后端／内核／依赖变化；未扩展INT4/NF4、KV、PPL／生成或长训练范围。详见[报告](npunlock-probe/research/results/w8-tail-quality/REPORT.md)。

## S54 整网输出目标与新提示验证

仅在CPU可微参考中拟合90个Q/K/V rank32输出因子，共921,600个master参数；前向逐次Half舍入，INT8编码／scale、输入基及其余权重固定。校准目标包含原FP16整网raw logits L2、teacher KL及高L2位置约束；初始高步长仅依据校准结果淘汰，最终两轮目标下降3.60%／8.77%，不当作NPU质量结论。

内部NPU第二轮通过243→244/290，均值L2下降，但最大KL升高，原规则保留baseline；候选冻结后，新八条提示通过246→248/262、平均L2略降，最大L2和KL却升高，token一致数未变。新样本没有参与拟合或候选重排；只认证有限teacher-forced前向，不新增PPL／生成、KV或INT4/NF4范围。

5个正式NPU进程共1,650次提交均健康结束，150项原局部检查、326项完整工作区离线检查通过；预热整网仍30次提交，无新增推理算子。228份fixture与9份旧身份未变，生产默认、内核及依赖不变。后续需要约束少数高误差位置并另备独立样本；复现与边界见[报告](npunlock-probe/research/results/w8-logits-quality/REPORT.md)。

## S53 真实NPU Q/K/V输出校准

使用原32条校准轨迹的真实NPU前缀、主投影及低秩中间值，目标为同输入原FP16投影；只拟合90个Q/K/V矩阵的rank32输出因子，编码、scale、输入基、参数量与执行图不变。三种固定blend使校准CPU代理SSE下降0.162%／0.277%／0.364%，不等同于整网收益。

0.50强度的内部平均logits L2由2.041913%降至2.016721%，最大L2／KL下降，token一致286→287；但通过数243→241/290。1.00强度仍243/290，平均L2／最大KL略升，原规则冻结选择baseline。所有配置仍30次预热提交，默认及门槛不变。

5个正式NPU进程共6,180次提交均健康关闭；1,920项精确图控制、120项局部检查和315项完整工作区离线检查通过，228份fixture未改。没有新独立前向、KV、PPL／生成、INT4/NF4或训练认证。下一轮应直接研究整网输出目标，并用新的独立样本验证；范围与复现见[报告](npunlock-probe/research/results/w8-qkv-quality/REPORT.md)。

## S52 Attention分段诊断与FP32 softmax原型

已知末层坐标在固定顺序有符号重放中，Q/K/V投影和softmax各贡献-0.1875，合计-0.375；该归因只针对一个坐标、相同W8参数参照且依赖替换顺序，不代表整网量化误差比例。真实NPU FP32 softmax将focus整层scaled max由0.013540070降至0.007103981；8个交叉设置L2均下降，但scaled max有2处上升。

完整30层／8条内部轨迹的通过数243→242/290，平均L2略降而最大L2／KL升高；原晋级规则选择baseline。预热提交30→240，单次诊断计时不作为正式性能基准。4个正式NPU进程共9,073次提交均健康关闭，213项图／分段精确控制、305项完整工作区离线检查通过；228份fixture和旧运行组件指纹保持原值。

生产默认、参数、内核及预算未改；未新增独立/PPL/生成、KV原型或INT4/NF4认证。后续优先使用真实NPU Q/K/V输出进行校准，并以原FP16整网质量为最终目标。复现与局部／整网参照边界见[报告](npunlock-probe/research/results/w8-attention-quality/REPORT.md)。

## S51 W8 scale校准与输入敏感精度边界

固定INT8 codes与rank32，两个方案仅改gate/up或全部FFN的scale／补偿；90个投影校准SSE合计下降0.889%，但真实NPU内部通过243→242/290，全FFN方案在第30层被原局部门槛拒绝。两次拟合参数逐字节相同，内部结果和拒绝点复现。选择器现能排除已知的局部拒绝，未知错误与不健康设备仍中止研究。

三个末层共享同一实际候选前缀时，最坏坐标均为actual=-5.9375、reference=-5.5625；有符号边界重放将此处-0.375差值定位到attention残差经FFN的传播。单独FFN同输入scaled max约0.003527，整层约0.01354；SHAVE SiLU没有消除峰值。本例没有认定具体内核错误或放宽原门槛，下一步仍需区分attention内部阶段。

8个正式NPU进程共1,839次提交，全部健康关闭，其中2个按原精度门槛拒绝候选；293项本地离线检查通过。120组图结构／常量区控制与228份fixture身份通过，无生产后端、内核、依赖或原生构建变化；本轮未进行独立/PPL/生成认证，未新增INT4/NF4。详见[报告](npunlock-probe/research/results/w8-scale-quality/REPORT.md)。

## S50 gate/up/SiLU拆分与严格激活原型

在内部两条轨迹、五层、forward与第64步decode上分离gate/up投影、SiLU及乘法，共20个同输入取样。gate/up传播与SiLU局部差异量级相近，乘法更小；导出图/普通FFN/融合层及down控制尾段均在取样中逐项相等。不能将全部门控差异归因于SiLU或直接认定驱动错误。

已有SHAVE FP32 SiLU及设备乘法原型使20处局部FFN误差均下降；内部通过243→246/290，但最大KL0.0040497→0.0041673，未满足预设不劣化规则（两者仍低于原0.02门槛）。预热提交30→660，不采用为默认。4个正式进程7,130次提交正常关闭，281项本地离线检查通过；未新增生产代码、SHAVE内核、依赖或原生构建。候选未晋级，本轮未做独立/PPL/生成认证。方法与边界见[报告](npunlock-probe/research/results/w8-gate-quality/REPORT.md)。

## S49 设备数值诊断与NPU内部筛选

新增真实NPU同输入诊断，固定五层检查主W8、两次低秩GEMM及相加；两次低秩矩乘误差约0.00009%–0.00074%，相加和普通FFN/融合层输出在本取样中逐项相等。门控输入和主W8投影差异更大，尚未分离gate/up/SiLU来源，不能直接宣称其中某项有错误。

九种既有配置及FP16控制各在独立NPU进程评测8条/290个内部位置；设备选择仍为input-local（基线243→246/290），独立前向和KV复现S48有限收益与退化，不采用为默认。16个进程18,500次提交正常关闭，272项本地离线检查通过。没有生产代码、内核、默认参数、依赖或原生构建变化；文本证据仅引用相同哈希候选的S48结果，本轮未重跑PPL。协议、限制和复现见[报告](npunlock-probe/research/results/w8-device-quality/REPORT.md)。

## S48 W8量化前缀与输出校准研究

新增down实际量化输入校准、整块输出残差回归及固定低秩输入基回归，共8种候选，32条/2048行拟合、8条/290位置内部选择。整块方案在内部退化；最佳input-local在独立NPU前向多通过1处但平均/最大误差及KL变差，KV仍82/98，PPL28.037196→28.041923，八条生成与基线相同，故不采用。实际校准输入来自CPU精确W8参考；真实NPU用于独立验收，尚非设备激活校准。

7个正式真机进程44,130次提交正常关闭，265项本地离线检查通过；没有生产后端、内核、默认配置、依赖或原生构建变化。数学方法、固定协议、数值对齐边界及复现见[报告](npunlock-probe/research/results/w8-output-quality/REPORT.md)。INT4/NF4整网及低位训练本轮未扩展。

## S45 缓存保护及低位边界

默认1536项/512MiB不变。Smol融合图、prefill元数据图及其RMSNorm图在承压时同族失效重建，避免本机可复现的部分释放后旧图复用故障；其他图仍LRU。分组decode构建/执行期间、含固定Smol图的族或单个Smol图超预算时提前拒绝危险淘汰/临时执行。预算可用torch.intel_npu.set_graph_cache_budget(整数bytes)显式设置，范围1字节至1GiB，不能小于现缓存量或在计划内修改。此预算仅计缓存blob，不是全部驱动内存；同族失效有额外编译成本，原生故障的最终根因仍未确定。

校正的prefill磁盘身份现在包括模块源码、投影名、shape、rank和因子字节。普通/原校正/同rank半强度/rank16的真实cache miss/hit已分别核对，不再复用同一身份。W8校正原16个KV失败位置仍在，去均值仅作诊断，不放宽原门槛；两个新校准方案均未胜出。新128位置的NPU前向通过数提高但token一致数退1个，故不认定整网质量全面解决；INT4/NF4、PPL和自主生成没有本轮新增认证。

## S44 W8低秩残差修正

FrozenSmolLayer/Stack新增默认None的corrections参数，接受与显式逐行W8 PackedWeight配套的不可变FP16因子，Layer按投影全名、Stack按绝对checkpoint层编号的完整map组织。形状须匹配、rank≤32，仅用于未受FP16保护的w8a16投影；公开构造器检查表示和shape，权重/因子语义配对由调用者负责，研究加载器另核对SHA。离线准备为CPU FP64协方差/SVD，forward/prefill/decode补偿矩乘在设备图中执行，无新依赖或C++/ELF重建。

SmolLM2-135M、30层decoder、B1/容量64、rank32实测平均logits误差改善，通过126/142；仍有16处超过原L2门槛，CPU最大KL亦超限。PPL和自由生成未显示明显语言质量提升，不作为默认。INT4/NF4、低位训练/QLoRA、任意模型迁移和更长上下文未由本轮认证。FP16控制组默认512MiB图缓存时出现一次设备错误，768MiB重跑通过且无淘汰；生产缓存默认不变，原因尚未定位，详见报告。

## S42 分块Attention与Nested推理

默认SDPA仍math，显式sdpa_kernel(FLASH_ATTENTION)或torch.intel_npu.flash_attention选择实验分块路径。FP16/FP32、rank2–64、有界广播/GQA，QK/V头维1–128，query/key block为1–32/1–128，前后向最坏packet<=32,768 words；无完整score/概率保存，反向重新计算块概率并累加共享梯度。dropout必须0，可微mask、二阶梯度、BF16及任意维度未提供。math保留旧Dropout/mask梯度；全score容量和整行65,536列检查只对math保留，其他存储/输出/图/因果header约束不变。

S43省略的query/key block参数由后端按完整反向预算自动选择，优先32/128；大head dimension必要时缩小自动块（本轮128维、Q33/K129选择32/64）。显式参数保持原值，超预算仍拒绝，不会静默放宽32,768-word保护。内核仅局部内联普通FP32算术并复用query行缩放，保留原IEEE边界路径；性能改善针对旧flash，不能解释为超过默认math或整网训练加速。

strided NestedTensor的真实NPU payload支持构造、CPU往返/clone/copy、unbind/values/padded、LayerNorm stats、标准MHA/Encoder及Nested SDPA推理。默认Encoder可在上游eval/no_grad及对齐条件下启用nested fastpath；native模块仍逐样本复用稠密数学计算。Nested SDPA可选择新flash，无额外mask且dropout=0；CPU布局元数据kernel只处理metadata，CPU Nested的SDPA前后向代理已核对，其他GPU Nested未本轮真机认证。Nested训练、jagged、通用算子全集/compile/AMP和任意布局仍未补齐。

21组最终进程检查全部通过，共39,108次NPU提交；193项离线检查及16组/288个实际C body宿主packet通过。原版两层Llama/GPT-2 FP32训练/恢复与Llama FP16 cache/generate通过。128×256对照省86%–88%后端Tensor峰值，含编译冷测试约慢3倍；没有速度优势承诺。新增两组ELF/汇编，原64组、26份C源及DLL/PYD不变，详见报告。

## S41 Attention与HF模型迁移

新增功能式arange/start/end/step工厂、整数/BOOL输入cumsum、实数/整数isin及out、整数constant pad；修复空rank1 KV拼接及梯度、部分浮点视图取负、native MHA/Encoder空batch/序列。cumsum只支持整数输出；isin按块扫描且最多16,777,216对比较，拒绝BOOL和整数Tensor/浮点混合；out要求精确预分配且无别名。arange仍从标量元数据生成CPU范围再上传，其余有效载荷计算/传输使用既有NPU路径。

HF验收使用本机Transformers5.18.0原版LlamaForCausalLM/GPT2LMHeadModel，均为2层、hidden16、随机初始化小配置，无权重/tokenizer下载或库源码改写。FP16/FP32推理、DynamicCache、带padding/无cache贪心generate、FP32 top-k状态重放及两步训练/CPU checkpoint恢复均通过；共17组最终进程验收、43,958次NPU提交、186项离线检查，见S41报告，不扩展为任意HF模型认证。

共享Parameter的模型迁移须按报告在转换期间使用PyTorch的set_swap_module_params_on_conversion(True)，finally恢复原设置，避免默认PrivateUse1转换拆开GPT-2 embedding/lm_head。这个设置只用于显式迁移步骤，没有永久更改Torch全局默认或修补Module._apply；普通model.to本身仍不能保证所有共享参数关系。

SDPA继续保存完整score矩阵并复制GQA的KV，原有存储/工作区/行长边界保持。FlashAttention、NestedTensor、浮点cumsum/top-p、beam search、更广泛cache/generate组合、预训练大模型质量、FP16/AMP整网训练和长期收敛未由S41认证。native空MHA返回None权重，与本机CPU一致，仍为推理专用入口。

## S40 标准GradScaler入口

导入torch_intel_npu后，可直接使用`torch.amp.GradScaler("intel_npu")`、`torch.GradScaler("intel_npu")`或提前导入的同类别名。已安装PyTorch没有供本项目使用的自定义设备scaler工厂，因此在导入时对原类的`__new__`安装仅针对intel_npu的构造分派，返回项目GradScaler子类；原类对象及已有别名保持，CPU/CUDA/XPU构造走原有分支。不是修改site-packages或新增NPU FP64计算。

FP32/INT32 scale与growth、FP64倒数保留CPU控制元数据；梯度检查和反缩放仍在NPU，每优化器回读一个BOOL。支持溢出跳步、动态增长/回退、累积、多优化器、state_dict/恢复及缩放器自身pickle。新增嵌套list/tuple/惰性iterable、空iterable和Tensor new_scale；后者只接受无梯度、单元素设备FP32并复制到控制状态，回读一个控制标量。scale的标量/向量dtype提升及Half大scale下零值行为与本机CPU GradScaler一致。

仍要求单NPU、FP32主参数/完整密集FP32梯度、普通优化器显式`foreach=False,fused=False`，不支持稀疏、梯度/参数别名、fused/capturable/differentiable路径。旧torch.intel_npu.amp.GradScaler及已有CPU checkpoint格式继续可用。CPU行为、XPU禁用构造及当前Torch2.14.1别名已验证；其他版本、第三方严格实例type检查和实际GPU AMP没有在本轮认证。通用torch.compile、长期收敛、BF16 AMP和原始NPU Tensor序列化仍缺。

```python
import torch
import torch_intel_npu

model = torch.nn.Linear(4, 3).to("intel_npu")  # FP32 master parameters
optimizer = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
scaler = torch.amp.GradScaler("intel_npu")

# x and labels are already on intel_npu.
optimizer.zero_grad(set_to_none=True)
with torch.autocast("intel_npu", dtype=torch.float16):
    loss = torch.nn.functional.cross_entropy(model(x), labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## S39 Attention边界与随机接口

SDPA支持有界空batch/头、空Q/K/V序列和零QK/V特征，保留广播及GQA。空K/V输出零；零QK特征仍执行mask、softmax、V加权和梯度。空序列/零V特征不消耗Dropout RNG，纯空batch的浮点mask得到零梯度，空序列/零V特征的未使用mask梯度为None。p=0/1不推进RNG，反向不抽样；非法dtype/广播/mask/GQA及原有容量保护保留。

新增torch.bernoulli及ATen标量/Tensor/out、bernoulli_的Tensor概率路径。概率为FP16/FP32/BF16同设备Tensor，可广播及只读有界视图；输出支持九种存储dtype。概率在设备检查后回读一个BOOL标志，非法数据在RNG消耗/目标写入前失败；原地目标须完整密集，out须精确shape且不别名。每个非空输出消耗一个Philox counter，空输出不消耗；原生Bernoulli零梯度及24字节状态重放保留。FP32工作区仍受1GiB保护。

新增torch.randperm及generator/out入口：设备生成完整有符号INT64随机键并排序，发现重复键则整批重抽，最多8次，避免稳定平局带来的偏差。主机只回读碰撞标志。n=0/1不消耗RNG；其余每次尝试消耗n个counter。支持INT8/UINT8/INT16/INT32/INT64、FP16/BF16/FP32，限制n-1可精确表示并遵守16,777,216元素排序上限；FP16最多2049、BF16最多257、INT8最多128、UINT8最多256。实际跨包验证到8193元素，不认证最大规模性能。无新依赖、原生构建或ELF；异步、FlashAttention/NestedTensor、通用HF生成和高阶梯度仍缺。

## S38 当前字节存储与密集平面

九种存储dtype均按1GiB单allocation字节预算检查：FP16/BF16/INT16最多536,870,912元素，BOOL/INT8/UINT8最多1,073,741,824，FP32/INT32最多268,435,456，INT64最多134,217,728；rank仍<=64。已通过九dtype恰好1GiB分配及首/中/末页NPU读写，越界在提交前拒绝。完整1GiB Half取负输出逐元素核对通过；存储容量不等于所有dtype均具备完整NN/AMP运算。

普通密集MaxPool前向在SHAVE生成窗口/索引，使用32,768-word carrier、受限工作量与页对齐源跨度；比较内联且输出坐标通过计数器推进。自动反向只扫描几何上可能到达当前输入块的窗口，保留顺序及Half舍入；这个优化仅应用于保存并通过version检查的前向索引。手动任意合法索引的ATen backward维持通用扫描语义。数值和INT64索引直接按目标dtype流式写回，避免大型完整FP32中间量。

4097×4097的16,785,409元素密集3×3池化，FP16/FP32各三次完整前后向与CPU零容差对照通过。最大输入/配对输出的完整检查、冷热时间及边界见[S38报告](npunlock-probe/research/results/storage-shape/REPORT.md)；它不推导全部最大几何或任意14GiB模型训练认证。仍无跨buffer分页、异步提交或硬件峰值吞吐承诺。

S38需要重建Torch PYD并同步新增pool ELF/汇编和更新后的selection-u32/manifest，session DLL及原62枚内核不变；无新增运行依赖。stats逐dtype容量字段为dtype_storage_limit_elements，旧max_tensor_elements字段保留FP32基数含义。

## S37 历史单平面选择元数据（当前存储以S38为准）

共享selection backward不再把大平面的宽度、偏移和元素数转成FP32数值。超过16,777,216时使用新增`selection-backward-u32`内核，8个header字段按原始UINT32传递，INT64索引仍按原始双字读取，拒绝非法高字及越界值。旧小尺寸入口及原62枚ELF保持不变；新增ELF为5,740字节，不需要重建DLL/PYD或安装依赖。

单平面的源/选择元素数沿用dtype存储检查：FP16/FP32逻辑元素最多268,435,456，INT64索引最多134,217,728；实际配对梯度还受索引容量约束。普通MaxPool空间输出保护改为134,217,728，仍受总输出配对存储和原窗口限制；浮点极值输入检查改用dtype存储。它们是软件边界，未认证最大形状。宽路径每包最多21,504条梯度、64,512个目标值，保留Half逐次累加、赋值负零、padding和初值语义。

FP16/FP32均实测单平面16,777,219元素的MaxPool完整前后向，以及同长度max/min完整输入梯度和argmax/argmin的并列/NaN首位置；实际索引16,777,217没有舍入。另测16,798,727条梯度跨包累加。宽源与宽目标同时很大时仍可能产生大量重复扫描。sort/topk、Embedding、其他池化和一般索引前向的独立容量未因此统一放开；详见[S37报告](npunlock-probe/research/results/selection-capacity/REPORT.md)。

## S36 历史MaxPool与极值容量（当前以S37为准）

普通MaxPool1d/2d解除旧524,288空间输出和8,388,608总采样点限制：空间输出最多16,777,216，每个窗口仍<=32,765采样点，按原<=65,536-word包执行。总输出同时遵守数值/INT64存储（当前最多134,217,728个配对输出元素）。这些是软件保护边界，不是最大形状的实测认证；普通正通道的空batch支持前后向，零通道仍拒绝。

前向包可跨多个通道平面，索引仍相对于各自空间平面。反向、TopK及sort/max/min的规则行选择按完整平面分组，保留现有C内核16,777,216元数据限制；多平面的总Tensor可以超过该数，单平面的源/选择元素数仍不能超限。小工作量的一组情况继续调用原入口。重复项的Half累加和不累加赋值的负零规则保留。一般高级索引/扁平scatter反向没有统一扩到存储上限。

浮点极值归约输入保护由1,048,576增至16,777,216元素；长轴超过旧内核offset范围时在设备INT64中恢复全局索引，不修改ELF内的offset界限。索引型输出的autograd关闭未使用零梯度的实体化，保持保存索引的version检查。

窗口源区间向页边界扩展并继续受65,536元素约束，复用已有范围绑定，减少大Tensor不对齐Slice编译。3D/自适应池化只复用规则平面反向及该读取逻辑，前向8,388,608采样点等原限制保留。AvgPool、插值、一般索引反向和最大形状未在本节全面认证；具体真机证据及剩余边界见S36报告。

## S34 当前索引与排序容量契约

S35的普通Conv2d工作区策略见下一节：不扩大单buffer存储界限，改为更早按窗口/折叠工作量选择有界执行并逐块组装结果。

Embedding及其内部index_select保留原来的小packet路径；超出完整索引包限制时使用已有byte gather和selection backward，必要时按特征列拆分。Embedding路径的源/结果仍按原268,435,456元素专用检查；S38普通Half字节存储扩至1GiB不自动放宽此处，词表行数和展开后的索引数各<=16,777,216；每个列块的源/结果<=16,777,216，未提高C内核浮点元数据界限或65,536-word carrier。上述是软件保护边界，不是最大形状的真机认证。

重复索引保持原顺序；Embedding的Half每次更新舍入，内部index_select使用FP32累加。padding通过where排除梯度，避免未选NaN污染。原始索引version保护不变，动态索引校验只下载少量状态，数据计算不回退CPU。大路径仍需扫描源分块，容量支持不等于高效。

sort/argsort和topk的输入/输出总量扩大到16,777,216元素，保留FP16/FP32/INT64/BOOL、稳定排序和现有计算包。超过小k工作量或宽轴限制时topk走已有排序；S37已替代共享selection backward的浮点元数据保护；S25公开index_select/gather/scatter_add的其他检查保留，不自动获得Embedding专用列分块；原生out/backward变体和所有极限配置未全面认证。具体实际形状、精度和训练验证见S34报告。

实际通过FP32的131,073×128词表完整输出/梯度（16,777,344源元素、两列块），FP16的九列块缩小预算压力用例，以及两种精度的长索引、非首轴、528,513元素完整排序反向、1,056,897输入元素topk和两步111万参数Embedding模型训练。150项离线检查、11组专项与3组原接口回归通过，共62,109次NPU调用；浮点精确比较为零容差数值比较，不单列NaN payload或正负零编码认证。

本轮仅修改Python编排、边界与测试，沿用S32原生组件和62枚resident ELF，无新依赖或重建。卷积窗口/折叠、池化、take_along_dim、次序统计单行等独立保护不在本轮统一放宽。

## S35 当前普通Conv2d工作区

当完整窗口或输入梯度fold的上界超过262,144元素时，普通Conv2d自动使用已有流式路径，避免创建/保存整批im2col及完整CPU正反索引表。该数字是路径选择预算，不是卷积输入/输出上限，也不是总临时内存的保证；小窗口仍走原路径，输出大小本身不单独触发切换。

流式前向和输入梯度通过单个FP32目标逐块写回，只保留不足1024个FP32元素的拼接尾部，保持现有4096字节对齐范围绑定。窗口行块同时考虑输入窗口和矩乘输出宽度；单行很宽时仍需容纳完整一行。权重梯度继续使用原二叉归约树；FP16仍沿已有FP32计算/转换规则，不新增DPU训练或改变计算内核。

1×4×96×97、4个输出通道的3×3卷积完整前后向验收中，后端Tensor分配峰值FP32为7.00→1.84MiB、FP16为7.35→1.63MiB，均不再保留该形状的完整CPU索引表。小窗口多输出通道、强制流式的分组/步幅/膨胀及两步小卷积训练另外通过；14份基线/最终记录、4,998次NPU调用、155项离线检查通过。详细门槛、冷耗时及限制见[S35报告](npunlock-probe/research/results/conv-capacity/REPORT.md)。

本轮只修改Python编排/测试；既有1GiB单allocation、dtype、卷积合法几何、单行、分组拼接和最终布局复制边界保留。没有普遍吞吐提升、最大形状、3D/转置卷积或长期Half训练认证，无需新增依赖或重编S32组件。

## S33 当前图缓存与冻结模型契约

QuantizedLinear（INT8/INT4/NF4）、FrozenTransformer、FrozenSmolLayer/Stack及矩形SHAVE GEMM在调用期间登记弱引用重建工厂，冷图或已淘汰的图可由当前模型重新生成。缓存不持有模型/工厂；调用者持有factory至提交完成，prepared chain继续持有原生session。普通图和这些适配器共用缓存；S45起Smol图承压时同族失效重建，其他图继续LRU，原device/profile/ELF验证保持。没有重建入口的手工/研究图仍pinned，全pinned时仍可拒绝新缓存项。

默认缓存同时限制1536项和512MiB的serialized graph blob总字节数。`cached_graph_blob_bytes`和`graph_cache_limit_bytes`只描述缓存中的图blob，不代表物理RAM、驱动图对象或所有活跃图的总占用。其他可重建图超过字节预算时仍可临时加载，不放入缓存，最后使用者释放后关闭；S45起Smol超预算图提前拒绝，分组计划内禁止危险的同族淘汰。无重建入口的超预算图仍明确拒绝。Smol旧小缓存/临时图测试的历史结果以S33报告为准，当前安全保护及重试验收以S45为准。

新增`graph_factory_calls`（包括首次加载的工厂调用）、`live_graph_factories`与`transient_graph_imports`；沿用淘汰、pinned与deferred统计。已覆盖小缓存下的低位Linear/Transformer/SmolLM whole/split/plain重建和四层链式decode，输出与缓存充足基线逐位一致，CPU独立参考范围见S33报告。缓存预算不改变单allocation1GiB/rank64、算子专用工作区或dtype边界。

另验证16层4096宽、268,500,992参数的残差网络两步FP32 AdamW：完整参数/梯度/两份状态4.001GiB、后端分配峰值4.251GiB，损失下降、所有参数有梯度，131,072个位置的更新/状态与CPU抽样参考一致。没有同规模CPU整网梯度参考，耗时约35分钟，不表示吞吐优势或长期收敛通过。

本轮只修改Python宿主代码和检查，已有S32组件无需重建DLL/PYD或安装新依赖。S32的“所有冻结图pinned”描述由本节替代；跨buffer分页、任意模型和14GiB完整模型训练仍未认证。

## S32 历史存储基线（当前存储以S38、缓存以S33为准）

单allocation上限为1GiB，普通逻辑Tensor仍为rank0–64。FP16/FP32/BF16/INT8/INT16/INT32/BOOL最多268,435,456元素，INT64最多134,217,728，UINT8字节载体最多1,073,741,824；FP16/BF16逻辑Tensor最多512MiB，INT8/BOOL最多256MiB。这些是保护边界，仍需满足物理/commit内存和各算子独立工作区限制；没有跨allocation分页Tensor。

普通算子图的缓存最多1536项，满时淘汰最近最少使用的可重建图，后续用到时重建。缓存移除只放弃自身引用；同步执行调用、已准备好的chain持有图对象，最后一个使用者释放后才关闭原生session。shutdown也会关闭仍由外部plan持有的已淘汰图。stats新增graph_cache_evictions、pinned_graph_sessions及deferred_graph_sessions；1536是缓存项上限，外部plan仍持有的图另行计数。

S32阶段冻结DPU/低位模型和矩形GEMM仍pinned，后来由S33补入调用期重建工厂。执行失败的context不参与继续淘汰/重建，原隔离策略保留。历史设备丢失记录不删除，只承诺报告所列设备、驱动和工作负载验证范围。

普通softmax、SDPA、MHA/encoder稠密工作区等委托当前存储检查的入口随之使用268,435,456元素边界；完整attention矩阵、每行宽度、Embedding/索引/卷积窗口/排序的专用保护不变。下面S30/S31的具体容量数字与保留全部图策略属于历史状态，以本节为准。约14GiB训练、任意最大形状及长期FP16收敛仍未认证。更新需依次重建session DLL和Torch PYD，无新SHAVE ELF或依赖。

## S31 损失与专用工作区容量（存储/缓存以S32为准）

硬标签CE/NLL保持rank1–5及原dtype/权重/ignore语义，解除整批rows*(C+4)<=65536：每packet仍<=65536，非空C<=65532，总输入按134,217,728元素及实际dtype存储检查。多packet前向只下载一份4字节校验标志，反向由保存的标签重建mask；原生backward独立重新验证，autograd保留标签/权重version检查。百万行例子完整梯度与CPU一致，43张图，详见报告。

普通Conv2d输出、原生MHA的score/projection、encoder FFN和多输出纯DMA拼接改用当前134,217,728元素检查；原生Half掩码softmax沿用当前行分块存储上限。实际越过旧容量的输出/梯度、FFN、mask和DMA范围见S31报告；MHA大score只做元数据边界检查，不推导最大attention认证。原生encoder仍是已列推理入口。

连续空间标签现在直接复用contiguous视图，减少副本并保留原标签version；S46补齐4D复制标签与空间NLL输入的原始版本保护，其他维度保留CPU复制语义。保留卷积窗口/fold、索引/Embedding、排序与selection的浮点元数据及packet界限；每行softmax<=65536、1536张图和单buffer存储不变。许多宽类别行、不同形状或不对齐偏移仍可能触及缓存。没有新增BCE/KL/CTC家族、长期Half收敛或14GiB训练认证。本节替代下文旧分类整批packet及相应稠密工作区容量描述。

## S30 历史容量与图复用契约（当前以S32为准）

当前普通Torch存储为rank0–64、单allocation最多512MiB。FP16/FP32/BF16/INT8/INT16/INT32/BOOL最多134,217,728元素，INT64最多67,108,864，UINT8字节载体最多536,870,912；因此FP16/BF16单Tensor最多256MiB，BOOL/INT8最多128MiB。容量是软件保护边界，不表示每个算子都支持最大形状。S25/S26的容量数字为历史记录，以本节为准。

连续、非空且4096字节起始对齐的内部子范围可按长度复用图；Tensor持有原storage，不切换所有者。输入/输出边界、图参数大小、context与输出别名都在native提交前检查。原infer/chain继续要求完整buffer；不对齐切片仍用原Slice。逐元素运算按dtype/元素数建图，大块运算直接绑定范围，其他分块结果逐块写回最终allocation，避免保存全部中间tile。

真机覆盖2050个不同偏移只新增1图、FP16/FP32各1820种同元素数形状各新增1图，以及超过旧256MiB的完整CPU梯度对照。实际模型训练参数量、状态/峰值内存与CPU参照范围见S30报告；约14GiB训练尚未实测认证。

1,536缓存图仍保留至shutdown，未实施运行中图淘汰或跨buffer分页。不同算子、元素数及不对齐切片仍可触及缓存；旧FP32元数据16,777,216、分类/Embedding/卷积/排序等专用界限继续适用。普通softmax与SDPA沿用当前134,217,728元素存储检查，完整attention分数仍生成。更新须重建session DLL和Torch PYD；复用原62枚ELF，无新增依赖。

## S29 当前损失函数契约

验收入口：[torch_npu_loss_extended_demo.py](torch_npu_loss_extended_demo.py)，详见[损失函数报告](npunlock-probe/research/results/loss-extended/REPORT.md)。

NLL/硬标签CE接受rank1–5的FP16/FP32输入、同设备INT64标签、可选同dtype `(C,)` 类别权重、三种reduction和INT64 ignore_index；CE接通label_smoothing。mean按未忽略标签的所选类别权重之和归一化，不要求权重为正；零分母遵循CPU的NaN/Inf语义。NLL跳过未选中的NaN/Inf权重；硬标签权重不支持求梯度，原标签和权重有反向版本保护。空batch/空间维度支持，类别数必须大于零。S31改为完整行分块，非空C<=65,532、每packet<=65,536，总输入按当前存储检查；每次非空NLL前向只回读4字节校验标志。

CE也接受与输入同shape的FP16/FP32软标签，支持混合FP16/FP32标签/类别权重；输入、软标签和权重均可求一阶梯度。mean按样本/空间位置数归一化；不自动归一化标签或限制其值域。负ignore_index不参与软标签计算，非负值拒绝。soft CE没有NLL packet限制，但仍受普通存储、softmax行宽及图缓存边界约束。

MSE支持rank0–64可广播输入、FP16/FP32混合dtype和双方一阶梯度，输出遵循dtype提升。`F.mse_loss`先广播；直接ATen backward返回广播shape，mean因子按原self.numel()，与本机原生CPU接口保持一致。`mse_loss.out`、`mse_loss_backward.grad_input`、`nll_loss_forward.output`、`nll_loss_backward.grad_input`及对应nll_loss2d接口均已接通；out要求预先分配精确shape/dtype且不与输入或另一输出别名，不自动resize。

FP16 NLL补入加权乘积、归约结果及mean梯度除法的Half舍入边界，覆盖本轮大/小权重的溢出、下溢。SHAVE仍使用既有FP32分块归约，不复刻CPU逐项Half级联累加；极端抵消、接近舍入分界或更大组合仍可能不同，未承诺逐位一致。训练宜显式使用FP32损失。

高阶梯度、任意大分类packet、其他损失家族（如CTC、KL散度、BCE等）的完整接口仍未认证。训练验收与数值容差以S29报告为准，不能据此称所有损失函数或CUDA兼容性完成。

## S28 当前SDPA契约

公开`scaled_dot_product_attention`支持非空FP16/FP32 Q/K/V，rank2–64，可不同rank并广播批次。Q/K特征维度和K/V序列长度必须匹配。`enable_gqa=True`要求三者rank>=3、K/V头数相同、Q头数为K/V头数的正整数倍；按连续组复制K/V头，反向在FP32合并共享头贡献，再按原批次广播形状归约并转换为输入dtype。实际GQA/MQA、rank64、混合rank及部分只读视图的测试见S28报告。

attention总元素、每个广播Q/K/V和输出沿用普通存储检查（S30为134,217,728元素）；完整attention矩阵仍会生成。每行key长度<=65,536，因果query长度<=1,048,576（现有C内核header限制），单allocation512MiB、dtype和1,536图缓存边界保留。测试跨越旧百万分数边界不表示已经认证最大容量或任意长上下文；空序列/空批次/零特征仍明确拒绝。

GQA兼容既有BOOL/浮点mask、浮点mask梯度、负scale、因果及Dropout；mask仍必须是可广播到既定score形状的设备Tensor，不同时使用显式mask和is_causal。Dropout沿用NPU Philox，反向不推进RNG。普通GQA会在设备上复制K/V，可能增大内存；没有FlashAttention/NestedTensor、DPU训练或异步执行。本轮不修改native MHA/Encoder专用入口的工作区保护。S27所述SDPA旧总attention容量限制由此替代，其他算子边界不变。

## S27 当前矩乘契约

FP16/FP32 `matmul`/`@`接受rank1–64，仍遵守向量提升、批次广播、内维匹配和每个广播操作数/输出的存储上限。`mm`仍要求2D、`mv`要求2D/1D、`dot`要求1D；输入允许有界、非负stride的偏移、步长和零stride重叠只读视图，执行前用DMA整理。前后向与`out=`均在已列用例中验证；不允许输出别名或重叠写入，native绑定仍必须覆盖完整连续allocation。

长K（>65,536）矩乘保留原分块及Neumaier顺序，改为每个输出两个FP32滚动状态，避免与K块数成正比的完整partial堆叠。新SHAVE任务最多32,768个输出状态；最终输出与操作数仍须满足S26容量。压力测试在NPU上流式累加逻辑258.26MiB的合成partial，不能视为同规模完整GEMM或全容量模型认证。

公开`bmm`处理空M/N/批次而不提交图，K=0产生正确零结果及一阶梯度；完整密集目的（如转置叶的梯度）复用DMA排列复制，不再经过动态scatter的索引上传/标志读回。真正部分目的写入仍保留动态索引验证同步。具体验收包括65539内维的矩乘、rank64与rank5广播梯度、部分/展开输入、空维度及两步独立训练。

没有扩大单allocation256MiB、1,536图缓存或其他算子专用工作区；普通FP16矩乘仍是FP32 SHAVE计算后转换，未改成DPU训练、异步stream或通用大模型后端。S26所列“长K完整partial堆叠”问题由本轮替代，其余容量说明继续适用。新ELF随源码提供；已使用S26宿主组件时无需重建DLL/PYD或安装新依赖。

## S26 历史容量契约（当前容量以S30为准）

普通Torch存储现为rank0–64，单allocation最多256MiB。FP16/FP32/BF16/INT8/INT16/INT32/BOOL最多67,108,864元素，INT64最多33,554,432，UINT8字节载体最多268,435,456；因此FP16/BF16单Tensor的逻辑上限为128MiB，BOOL/INT8为64MiB。256MiB是软件保护边界，不是NPU可用总内存。

FP16/FP32的复制、转换、广播、点算、sum/mean与Linear/矩乘的存储检查已扩展；平方与scalar pow(1)复用已有分块乘法/复制。普通softmax/log_softmax支持rank1–64和最多67,108,864总元素，仍要求每行<=65,536，保持完整行小块及原有数值顺序。实际验证范围见S26报告，不能据实现上限推断所有上限形状均已测试。

缓存仍最多1,536个图，运行期间不淘汰图；动态形状、很多不同偏移或4096元素的小packet可能先触及缓存。冻结DPU仍用原百万元素边界；旧工作区与FP32元数据算子保留16,777,216或其更小专用边界。SDPA总attention、Embedding、卷积、排序/统计与一般索引反向没有随存储上限自动扩大；selection backward现在显式拒绝超过16,777,216的元数据几何，避免浮点地址失真或内核静默不执行。

更新旧本地环境须依次运行`build_npu_session.ps1`和`build_torch_npu_device.ps1`。本轮不新增SHAVE ELF或pip依赖，不表示任意大模型已经适配。

## S25 基线扩展契约（历史容量，当前以S30为准）

S25将普通Torch存储与冻结DPU适配器的容量分开；后者仍保持既有模型、工作区与低位质量限制。
下列是S25的新契约；下文S5–S24描述的历史packet/模型范围不自动扩大。

| 领域 | 当前实现与验证范围 | 保留边界 |
|---|---|---|
| 存储/分块 | 单allocation最多64MiB；rank0–8；FP16/FP32/BF16/INT8/INT16/INT32/BOOL最多16,777,216元素，INT64最多8,388,608，UINT8字节载体最多67,108,864；FP16/FP32点算、归约、Linear沿用有界计算分块 | SHAVE carrier仍<=65,536 words；最多1,536缓存图；算子专用形状/工作区限制保留。不是任意大Tensor/模型认证 |
| dtype | 新增INT8/UINT8/INT16/INT32/BF16存储及设备转换；整数加减乘、位运算、整除/余数、整数幂、sum、逻辑及all/any；INT64原始字算术保留超过2^53的整数 | FP64/复数未实现；整数除法极值溢出、超范围浮点转整数不作跨平台承诺；BF16当前是FP32中间计算与舍入，不是完整BF16矩乘/卷积后端 |
| 布局 | 非负stride的偏移、广播和重叠视图可DMA整理；非重叠目的支持广播copy、fill、masked_fill和no_grad算术更新 | 重叠目的写入仍拒绝；带孔洞的empty_strided工厂不支持；一般可微算术inplace仍有限。部分copy/fill/masked_fill的原生一阶梯度另做验收 |
| 动态索引 | nonzero/单参数where、BOOL/INT32/INT64高级索引、负高级索引、连续/分离多轴广播；index_put覆盖/累加、scatter覆盖、gather、index_select、scatter_add | 结果仍受dtype容量；nonzero计数与索引校验会同步下载少量元数据。byte gather/scatter需要扫描源/目标分块，兼容性不代表高效率 |
| 索引反向 | 重复索引累加、原索引version保护、可微out-of-place index_put/scatter及原生inplace写入；gather/index_select/scatter_add的Half采用FP32累加 | 稀疏梯度、BF16累加写入、全部原生out/backward重载和极值组合未认证；当前Embedding由S34专用列分块扩容，一般索引反向仍有独立元数据限制 |
| 数学/拼接 | sin/cos/tan、pow、expm1/log1p、floor/ceil/trunc/round及一阶导、精确shape的out；新增无梯度原地版本；cat/stack支持已实现dtype的promotion及浮点梯度 | 超越函数不宣称正确舍入；round decimals、其他未注册数学函数、高阶梯度和任意可微inplace未实现 |

S25验证了2,102,276参数的Linear→sin→Linear网络，在FP16/FP32各进行两步独立CPU/NPU前向、梯度及SGD更新对照。
64MiB是存储软件边界；扩大它不能自动解除softmax/attention、卷积窗口、排序/统计、Embedding及冻结LLM适配器的独立限制。

## 普通设备 Tensor 与存储

| 接口 / 项目 | 当前契约 | 证据与边界 |
|---|---|---|
| `empty` / `empty_like` / `zeros` / `ones` / `full` | 9种存储dtype、rank0–64，容量见S38 | D/B/S7/S24/S25/S26；零元素不分配驱动buffer、不绑定空图；空Tensor不表示所有算子接受空输入 |
| `empty_strided` | 完整密集维度排列，无孔洞/重叠，非负 stride | D 的拒绝检查，C 的转置叶训练；不是任意 strided 存储 |
| `view` / `reshape` / `t` / `transpose` / `permute` / `detach` | 支持部分元数据变换；随后参与计算时须满足完整存储绑定契约 | D 的 `views`、`transpose_bits`；不能把能创建一个 view 当作能计算 |
| `.contiguous()` / `.clone()` | S25扩到9种dtype、非负stride与重叠只读view的DMA整理；clone独立存储 | contiguous可原样返回连续切片，需独立存储时用clone |
| `.to()` / `copy_` | 已实现dtype的同步CPU↔NPU/NPU↔NPU复制与转换；copy可广播并写入非重叠部分view | S25；非连续CPU上传仍需contiguous_format；跨设备可微链不支持 |
| 本地 INT64 `.to/copy_/clone/fill_/zero_` | 同步复制、整数/浮点转换、精确整数运算及非重叠目的写入 | B/S6/S25；原始字加减乘、位运算、整除/余数、sum，超过2^53对照；不是通过FP32保存整数 |
| 本地 BOOL存储与转换 | 真正1字节存储、非零判真、复制/布局、逻辑、sum/all/any、动态nonzero/布尔索引 | S7/S24/S25；结果仍须满足目标dtype容量 |
| 本地 `.item()` / Python标量读取 | FP16/FP32/INT64/BOOL，一个元素且覆盖完整连续存储，同步下载2/4/8/1字节；不可微 | B/S4/S7：精确整数、浮点特殊值及BOOL；保留loss.item后的反向，不是通用CPU算术回退 |
| 设备内 dtype 转换 | FP16↔FP32 的 SHAVE/DMA 转换；S14在完整连续绑定验证后按元素数共享图，恢复原始shape；设备内转换的一阶梯度有限覆盖 | D/S14/S38的全部half编码、FP32样本、scalar、out/布局和分块；显式整张cast仍须满足目标dtype字节容量，仍有静态图缓存上限 |
| CPU/NPU 传输与 autograd | 传输不提供跨设备可微链；使用 `detach()` 或 `no_grad()`，上传后再 `requires_grad_()` | D 的 `autograd_upload` 拒绝；不支持把 CPU 叶直接可微上传 |
| 输入绑定 | 原生图绑定仍要求`storage_offset=0`且覆盖整个allocation；公共入口先整理或通过设备分块更新目的 | S25允许非重叠部分目的写入；这不放开原生buffer绑定校验或重叠目的 |

元素上限是当前软件契约，不是物理内存容量。卷积、广播和矩乘还要检查输出与中间工作区。
共享物理内存/Level Zero shared allocation 不意味着没有 DMA、布局转换或同步。
S25只读布局检查非负stride及实际存储范围，允许重叠读取；写入另外检查目的不重叠。普通算术入口仍可能要求先`clone()`。
`view(dtype)` 有底层注册，主要用于内部位重解释，不把它认证为通用可微 dtype 转换 API。

## 算子与一阶梯度

表内“有”表示实现并有对应有限测试，不表示逐位等同 CPU 或可进行二阶求导。
普通设备算术支持 FP16/FP32；FP16 矩乘/卷积等路径可能使用 FP32 中间计算。

| 操作 | 一阶梯度 | 当前参数与形状限制 | 证据 |
|---|---|---|---|
| 本地 `triu/tril` | FP16/FP32有界一阶 | FP16/FP32/INT64/BOOL、rank 2–4非空，继承dtype容量；整数diagonal，部分只读输入先DMA整理；out精确且不别名，无inplace | S11：字节Gather与零哨兵、特殊值/大整数位保真、梯度及out |
| 本地 `all/any` | 不可微 | 已实现存储dtype的非零判真、rank0–8；None/int/维度列表、负轴、keepdim、空输入及空维度列表 | S11/S25；同步读标量只发生在用户请求bool/item时；精确无别名BOOL out |
| `add` / `sub` | 有 | 实数 alpha，不接受 bool；有限值须在输出 dtype 可表示范围内；实数标量或 FP16/FP32 Tensor；广播结果满足全局上限 | D/T 原有 alpha=1；P 新增非 1 alpha、广播、out 与一阶梯度检查 |
| `mul` / `div` | 有 | 有限广播与 dtype promotion；`div` 仅 `rounding_mode=None` | D、T；C 的大/小分母及 0/Inf/NaN 梯度 |
| 本地 `baddbmm` | 有界一阶 | 匹配dtype FP16/FP32，batch1/batch2为3D、batch相同且内维匹配；同dtype bias可广播到(B,M,N)，alpha/beta为FP32范围的Python实数；支持只读切片与精确无别名out | S10：部分输入/广播bias梯度、系数、beta=0忽略NaN bias、alpha=0保留NaN乘积；未认证所有极值/舍入组合 |
| 本地 `rsub` / 标量减 Tensor | 有界一阶 | 复用sub的广播、dtype与实数alpha契约；Tensor/Scalar及ATen out重载 | S19修复alpha的位置参数绑定并新增反向；公开torch.rsub没有out参数，使用ATen out检查 |
| `neg` / `abs` / `sign` / `reciprocal` | 有 | FP16/FP32；部分 `out=` 只用于不需要梯度的调用 | C 的 unary；保留已有零/特殊值约定 |
| `relu` / 本地 `relu_` | 有界一阶 | FP16/FP32，完整密集存储；relu_返回原对象/存储，递增version；可微原地仅非叶且非view输入，叶/可微view在提交前拒绝，可先clone；no_grad允许叶的完整存储更新 | T/S22：NaN/负零/subnormal、原地梯度、叶/view预检、版本保护、Linear/Conv2d/BN2d/ReLU标准组合；不是任意部分写入或原地ReLU6 |
| 本地 `layer_norm` / `nn.LayerNorm` / `native_layer_norm` | 有界 | FP16/FP32，rank 1–4，非空normalized_shape匹配尾部维度，eps为正有限FP32；可选weight/bias，完整密集排列；混合FP16输入+FP32 weight/可选bias的归一化宽度<=65,536；不提供native out重载或二阶梯度 | N/N2：同dtype与混合独立前后向、stats、单元素、多轴及分块；混合bias-only拒绝；独立算子修复不代表整网混合轨迹或AMP已认证 |
| 本地 `gelu` / `nn.GELU` / `gelu_backward` | 有界 | FP16/FP32，rank 0–4，全局容量界限；approximate仅none/tanh；无梯度out精确shape/dtype且不别名 | N：两种公式前后向、out、half全编码、FP32样本、特殊值及分块；SHAVE FP32计算后转回原dtype，erf/tanh是数值近似，不保证逐位等同CPU |
| `log_softmax` | 有界 | FP16/FP32，rank 1–64，任意合法正/负轴，最多67,108,864元素、归一化轴宽<=65,536；支持显式dtype=FP32，拒绝内部half_to_float=True；out仅无梯度且不别名 | L/S13/S19/S26：完整行分块及轴交换DMA整理，含部分正stride读/梯度；不改变单行归约内核 |
| 本地 `softmax` / `nn.Softmax` | 有界 | FP16/FP32，rank 1–64，任意合法正/负轴，最多67,108,864元素、归一化轴宽<=65,536；部分只读布局可整理、显式dtype=FP32；out仅无梯度、精确shape/dtype且不别名；内部half_to_float=True拒绝 | S/S13/S19/S26：前后向、out、NaN/Inf与跨块部分view梯度；SHAVE FP32计算后转回原dtype |
| 本地 `NLLLoss` / `CrossEntropyLoss` | 有界 | rank1–5；硬标签INT64、可选同dtype类别权重；CE可用同shape FP16/FP32软标签及混合浮点权重。三种reduction、label_smoothing、ignore语义见S29；硬标签packet<=65,536 | L/S19/S29：前后向、空间类轴、空维度、软标签/权重梯度、NLL原生及out、版本保护。非空硬标签前向读4字节校验标志；不支持硬标签权重梯度 |
| `sum` / `mean` | 有 | 轴可为 None/int/tuple/list，支持负轴、keepdim；轴不可重复；结果 dtype 仅 FP16/FP32 | D 的 axes/edges，T 的 reduction，C 的 dtype backward |
| 本地 `argmax/argmin` | 输出不可微 | FP16/FP32，原rank/轴/keepdim及精确out契约；S37浮点输入按dtype存储检查，输出遵守INT64存储 | S21/S36：位序比较、重复极值/NaN首位置；长轴超旧offset范围后在设备INT64恢复索引；整数输入走已有排序路径，不代表所有dtype或最大形状认证 |
| 本地 `topk` | 浮点值有界一阶；索引不可微 | FP16/FP32/INT64/BOOL，已验收rank0–5；0<=k<=轴长（scalar k=0按Torch返回scalar）；S34输入/输出<=16,777,216；精确双out、只读布局 | S22/S23/S24/S34；整数/BOOL、大k或超小k工作量/宽轴界限走已有排序。单包保护不变，平局索引不保证CPU相同顺序；不是最大形状认证 |
| 本地 `sort/argsort` | 浮点值有界一阶；索引不可微 | FP16/FP32/INT64/BOOL，已验收rank0–5，含空Tensor；任意轴、descending/stable、只读布局和精确out；S34输入/输出<=16,777,216 | S23/S24/S34；INT64直接比较高低原始字，保留超2^53与signed64极值；FP32 merge每批10,922值、INT64每批8,192；宽行bitonic O(n log²n)。图缓存采用S33双预算LRU |
| 本地 `kthvalue/median/nanmedian/mode/quantile/nanquantile` | 浮点一阶；indices不可微 | 排序行宽1–32,760；k/轴/keepdim，median/mode的pair out及全局median；quantile仅FP32、标量或同设备rank0/1 q（最多128个），五种interpolation；q不求导 | S24：排序及统计在NPU，INT64原始字不转FP32；每行读4字节校验标志。median全局按tie/NaN分摊梯度；mode平局索引允许平台差异。超长统计行、更大q、全特殊值组合和二阶导仍不承诺 |
| 本地 `max/min` | 浮点值有界一阶；dim索引不可微 | FP16/FP32/INT64/BOOL；浮点二Tensor极值重载保留；整数单轴/全局归约用稳定排序，INT64索引容量不变 | S23/S24；非空归约轴、空输出支持；浮点全局平分tie/NaN梯度。整数二Tensor算术仍不属于本轮 |
| 本地 `amax/amin` | 有界一阶 | FP16/FP32、rank0–4、全归约/多轴/keepdim及只读view；精确无别名out；S37行极值输入按dtype存储检查 | S21/S36：复用行极值内核，多处相等时平分梯度；NaN导数与本机CPU对照，不承诺signed-zero极值输出逐位一致；max/min二元接口另见S23 |
| 本地 `linalg.vector_norm` / 普通 `norm` | 有界一阶，p=0返回None梯度 | FP16/FP32、rank0–4、ord仅0/1/2/+inf/-inf，合法多轴及keepdim；dtype同输入或FP16升FP32，不允许缩窄；精确无别名out | S21：FP32 sum/平方和/sqrt或极值组合；不是任意p、矩阵谱范数、稳定缩放L2或通用pow；FP32平方和可溢出/下溢，单值归约按abs处理。长向量L1的原CPU FP32严格对照失败，独立FP64参考通过，差异在报告中保留 |
| 本地 `isnan/isinf/isfinite` | BOOL不可微 | FP16/FP32有界只读布局，已存在的ATen out重载要求精确BOOL输出且无别名 | S21：比较/逻辑在NPU执行；尚未认证整数/BOOL输入 |
| `mm` | 有 | **只接收二维**、匹配dtype的矩阵；完整密集转置可materialize | D 的 matrices/edges/transposed，T 的 matrix gradients；S2检查mm仍拒绝三维 |
| 本地 `matmul` / `@` | 有界 | FP16/FP32、S27 rank1–64；向量按PyTorch规则补维并在输出移除，两个向量返回标量；仅batch维广播，广播操作数/输出遵守S26存储上限；out精确shape/dtype且不别名 | S2/S4/S27：混合rank、广播/有界只读视图、单双侧梯度、out、M/N/K分块及长K滚动补偿；反向在FP32归约广播轴及共享向量梯度；空/零K范围见S27 |
| 本地 `dot` / `mv` | 有界一阶 | FP16/FP32，dot仅1D×1D、mv仅2D×1D，不广播；复用matmul计算、完整存储及精确无别名out契约 | S4：前后向、out、版本保护、非法rank/shape/dtype拒绝；普通mm仍仅二维，未新增专用DPU向量内核 |
| 本地 `index_select` | 浮点有界一阶 | S25公开入口：rank1–8、9种dtype的原始字读取、scalar/vector INT32/INT64索引、负axis；索引值须非负；out精确无别名 | S25通用分块；Half梯度FP32累加，原索引version保护；共享selection反向由S37使用整数元数据，其他索引检查保留。Embedding内部接口由S34按列扩容，两条入口的界限不能混用 |
| 本地 `gather/scatter/scatter_add` | FP16/FP32有界稠密一阶 | rank1–8、同rank INT64索引，非索引轴可取较小子集；gather原始字读取、scatter覆盖、整数scatter_add；形状/dtype/out及容量须合法 | S25；取消公开入口128行限制，已测130行；Half scatter_add初始化FP32 accumulator后累加；稀疏梯度及BF16累加写入不支持 |
| 本地 `take_along_dim` | 有界一阶 | 继承gather行/packet界限；dim=None时展开且严格检查索引范围；有dim时支持非索引轴广播，INT64正负及超范围索引按本机Torch2.14语义规范到合法余数 | S20：专用64位整数SHAVE路径，不把索引转为数值FP32；规范化每32765个索引为一包、额外读一次4字节标志。不是通用公开INT64 remainder算子 |
| 本地 `F.embedding` / `nn.Embedding` | 有界稠密一阶 | 二维FP16/FP32权重，连续同设备INT64索引rank 0–3；padding_idx；sparse=False、scale_grad_by_freq=False、max_norm=None；S34存储/列分块边界见上文 | S5/S12/S34：重复索引、padding/NaN隔离、冻结模块及普通模型CPU训练对照；支持大词表/长索引后端路径，不代表任意Hugging Face模型或最大形状认证 |
| 本地 `cat` / `stack` | 浮点有界一阶 | 同设备、支持dtype promotion；输出rank1–64及dtype字节容量；空Tensor基础与部分view，S41补rank1空KV占位且计入dtype提升 | S6/S24/S25/S41；混合Half/FP32梯度返回各自dtype，空输入保留空梯度；非浮点输入不可微，out须精确无别名 |
| 本地 `arange` / 整数`cumsum` / `isin` | 序列控制数据，不支持浮点scan梯度 | S41：arange先Meta预检后CPU标量范围上传；cumsum接受整数/BOOL输入、整数输出；isin实数/整数输入，比较对数<=16,777,216，拒绝BOOL和整数Tensor/浮点混合 | 空/标量/布局、INT64原始值及NaN语义见S41；out精确预分配无别名，cumsum/isin有效载荷在NPU，不支持通用浮点cumsum或高效大集合算法 |
| 本地 `eq/ne/lt/le/gt/ge` | 无梯度 | FP16/FP32/INT64/BOOL，可广播、支持同设备Tensor或无梯度CPU零维/Python实数常量；拒绝INT64/浮点混用；精确BOOL out须不别名 | S7：六种关系、完整half编码、NaN/Inf/±0/subnormal、大整数高低字；不下载Tensor数据进行比较 |
| 本地 BOOL `logical_*` / `bitwise_*` | 无梯度 | BOOL Tensor的not/and/or/xor，二元支持广播；out精确、不别名 | S7；不提供通用整数bitwise或Python标量bitwise重载 |
| 本地三参数 `where` | 浮点一阶 | BOOL条件，FP16/FP32/INT64/BOOL结果，可广播及标量分支；浮点可promotion，拒绝INT64/浮点混用；out仅无梯度且精确无别名 | S7：原始位选择、不让未选NaN/Inf污染输出/梯度、广播归约及saved mask version；单参数where/nonzero和动态布尔索引仍拒绝 |
| 本地 `masked_fill` / `masked_fill_` | out-of-place浮点一阶；inplace更新用途 | BOOL mask；实数常量或匹配输入dtype的零维Tensor value；值Tensor的NPU浮点梯度可用。out-of-place可广播；inplace仅完整可写存储、shape不变，不提供一般可微inplace | S7：输入/value梯度、溢出、version；ATen Scalar_out/Tensor_out已注册，公开torch.masked_fill没有out参数；CPU常量验证不读取NPU数据 |
| 本地切片/选择与`split/unbind` | 有界一阶 | 正step切片、整数select/narrow保留原生共享storage及version；clone/copy/cat/stack可整理；slice/select反向用DMA回填零梯度，split/unbind复用cat/stack | S6：部分/偏移/转置切片、scalar select、未用输出；slice_copy/select_copy沿用复合实现，narrow_copy仅验收无梯度前向；不是高级索引、通用scatter或部分目的写入 |
| 本地 `bmm` | 有界 | FP16/FP32、仅三维(B,M,K)×(B,K,N)，同dtype/批次，不广播；输出遵守S26存储上限；有界部分/零stride只读输入先DMA整理；out仅无梯度且不别名；无out_dtype扩展 | S/S10/S27：双侧/单侧梯度、投影视图、完整密集转置、批次/输出/工作区分块及K>65,536，空输出/零K；FP32展开乘积+补偿归约，未接入DPU batched GEMM |
| 本地默认math `F.scaled_dot_product_attention` | 有界 | S28/S39：rank2–64、同dtype FP16/FP32，可不同rank并广播，支持空维度/零特征；正K/V头时GQA要求Q头可整除同数K/V头；允许有界部分/零stride只读输入；dropout_p在[0,1]，有限FP32 scale；因果与mask不可同时；mask须同设备/广播，为BOOL或query dtype/FP32；分数/各广播操作数/输出各<=268,435,456，非空分数key序列<=65,536，因果query<=1,048,576 | QKV、共享K/V及浮点mask一阶梯度；BOOL True表示保留，全屏蔽行零输出/梯度；空序列/零V特征不抽样，未使用mask梯度为None；完整行因果偏移及设备RNG，backward不抽样、p=0/1不推进RNG；仍存完整分数及GQA重复K/V，S42另提供显式分块路径，二阶梯度未实现；eval须传dropout_p=0 |
| 本地 `clone` / `contiguous`布局梯度 | 有界 | FP16/FP32有界只读布局；默认/preserve_format保持密集stride或压紧空洞，contiguous_format转连续；INT64无梯度；无clone.out扩展 | S2/S6：密集转置、head合并、切片梯度；CPU/NPU传输原autograd拒绝策略不变，完整存储要求见上表 |
| `addmm` | 有 | bias 同设备/同 dtype 且可广播到 `(M,N)`；实数 alpha/beta，有限系数需在 FP32 可表示范围 | D 的 bias/alpha/beta/out 对照，T 的 addmm gradients |
| `linear` / `nn.Linear` | 有 | FP16/FP32、输入rank1–64、权重`(out,in)`、可选bias`(out,)`且dtype匹配；受存储/分块/缓存约束 | D/T/S25/S26；1704万参数Linear两步独立训练对照见S26报告 |
| 本地 `convolution` / `Conv1d/2d/3d` | 有界一阶 | FP16/FP32、NCL/NCHW/NCDHW、合法groups/stride/padding/dilation/bias；输入/输出沿用容量；3D每组核体积×最大通道<=65,536 | T/S19/S24；设备静态窗口DMA+FP32 GEMM，分组串行。新增3D输入/权重/bias梯度和三步小网络，不表示默认大型torchvision模型已经认证 |
| 本地 `ConvTranspose1d/2d/3d` | 有界一阶 | FP16/FP32、NCL/NCHW/NCDHW、groups/stride/padding/dilation/output_padding/bias；输出和每组window容量约束见报告 | S23/S24；静态反卷积窗口DMA+FP32 GEMM；3D一阶输入/权重/bias与小网络训练已检查；非DPU专用矩乘 |
| `mse_loss` | 有界 | rank0–64广播、FP16/FP32混合dtype，none/mean/sum、双方一阶梯度、native backward及out | T/S29：标量/行/列/rank64/步长/空维度、直接ATen语义与组合训练；out预分配且禁止别名 |
| `F.batch_norm` | 有界一阶 | rank2–5；训练每通道样本数>1；正eps、成对running buffers；输入dtype或FP32 affine | B/S24；非4D映射为(N,C,1,S)，新增3D模型train/eval与inference_mode检查；不是任意rank的native_batch_norm内核 |
| `nn.BatchNorm2d` | 有限 | 已覆盖 `track_running_stats=False`；本地新增默认True、momentum=0.1/None、affine=True/False的FP16/FP32模块迁移，S24新增BatchNorm1d/3d的rank映射，沿用原参数契约 | M/B：运行统计、INT64计数、四步独立CPU训练对照、CPU checkpoint恢复、eval/reset |
| 本地 `GroupNorm/native_group_norm` | 有界一阶 | rank2–5 FP16/FP32；groups整除C、正有限FP32 eps；同dtype或FP32 affine；原容量约束 | S23/S24；3D体数据前后向与小网络已检查；native out、mixed bias-only与二阶导仍不支持 |
| 本地 `InstanceNorm1d/2d/3d` | 有界一阶 | NCL/NCHW/NCDHW，FP16/FP32、可选affine与成对running stats、train/eval；空间统计数>1 | S23/S24；设备上变换为(1,N*C,1,S) BatchNorm，跨实例平均running统计；训练出口clone支持后续原地激活 |
| 本地 `interpolate/Upsample` | 有界一阶 | 1D nearest/exact/linear；2D nearest/exact/bilinear/bicubic含antialias；3D nearest/exact/trilinear；size/scales/align_corners，非空输出及原容量 | S23/S24；静态坐标、NPU加权与伴随反向；AA采用triangle/Keys(-0.5)滤波；少数旧CPU显式scale前后向不一致配置仍在训练前拒绝，使用size/recompute_scale_factor=True；NaN/Inf全组合未认证 |
| 本地 `AvgPool1d/AvgPool2d` | 有界一阶 | CL/NCL、CHW/NCHW，FP16/FP32；合法kernel/stride/padding、ceil_mode、count_include_pad；2D支持非零且绝对值<=2^24的divisor_override，继承输入/输出/工作区界限 | S20：FP32分组卷积求和后除数再转回dtype；分母只由几何决定，变化时上传一个元数据网格，无输入下载；不是专用高性能池化内核，MaxPool见S22，3D池化由S24补充 |
| 本地 `MaxPool1d/MaxPool2d` | 有界一阶 | FP16/FP32，CL/NCL或CHW/NCHW；原几何/indices/双out/backward契约；S37空间输出<=134,217,728、总配对输出按dtype存储、核面积<=32,765；不再按总采样格拒绝，每个窗口仍须有合法点 | S22/S36：跨平面前向包、规则平面分组反向，Half重复累加/空间INT64位置保留；空batch可用。大平面反向采用UINT32元数据，3D/自适应前向限制另列，不能推导任意大池化认证 |
| 本地 `AdaptiveAvgPool1d/AdaptiveAvgPool2d` | 有界一阶 | CL/NCL、CHW/NCHW，FP16/FP32，正输出；原64-bin限制由S24的通用窗口路径扩展；可上/下采样并允许重叠bin，继承总存储界限 | M/S20：保留4D到(1,1)原路径；其他配置按公开形状整理/FP32均值/拼接，可微且无payload下载；仍受容量/采样工作量限制；最大池化见S24 |
| 本地 `AvgPool3d/MaxPool3d/AdaptiveMaxPool2d/3d/AdaptiveAvgPool3d` | 有界一阶 | FP16/FP32，含无batch输入；3D kernel/stride/padding/ceil、max dilation、avg include_pad/divisor；自适应正输出；存储及最多8,388,608个采样点约束 | S24：静态窗口/DMA/FP32归约或SHAVE选择，一阶反向在NPU；Max每窗口须有有效采样点；更大原生out/backward变体、空batch/特殊值全组合未认证 |
| 本地 `MaxUnpool1d/2d/3d` | 有界一阶 | 同shape浮点输入与INT64空间indices，合法output_size及存储上限；1D由标准Torch降到2D | S24：设备索引覆盖写入，反向为分块动态读取；越界校验读4字节标志。重复indices采用确定的覆盖顺序，不保证CPU/CUDA的非确定性竞争结果 |
| 本地 `pad` / reflection/replication/constant padding | 浮点有界一阶；整数不可微 | FP16/FP32、最多末三空间轴（constant可更多）、正/负extent、constant/reflect/replicate/circular；S41另补整数/BOOL constant pad的原始位复制、合法裁剪与空结果 | S24的浮点空输入/输出与超一次循环wrap限制保留；S41整数标签padding不经过FP32转换 |
| 本地 `pixel_shuffle/pixel_unshuffle/channel_shuffle/Unfold/Fold` | 有界一阶 | FP16/FP32；合法factor/groups/2D kernel/dilation/padding/stride；输出与原存储边界 | S24：仅CPU地址元数据，payload DMA和重叠归约在NPU。Unfold前向逐位；重叠反向/ Fold按数值容差，非逐位累加顺序承诺；原生out变体未认证 |
| 本地 `affine_grid/grid_sample` | 有界一阶 | FP16/FP32、2D/3D；nearest或bilinear/trilinear，2D还含bicubic；zeros/border/reflection及align_corners；输出<=1,048,576、采样点<=8,388,608 | S24：grid模型数据在SHAVE生成动态索引/权重/坐标导数；NPU动态读/加权/scatter、theta/input/grid反向；坐标及展开像素坐标绝对值<=1,048,576，NaN grid按-1处理，Inf拒绝。新3步可学习仿射训练；全部原生out/backward变体未认证 |
| `fill_` / `zero_` / `add_` | 更新用途 | `add_` 要求 `no_grad`、输入同 shape/dtype/device；Tensor alpha先转换为输入dtype，再进行FP32中间运算，FP16有限alpha不得超过65504；不承诺一般可微 inplace 图 | C/T/N3的SGD、原样本、half全编码、密集布局与version guard；CPU小张量标量尾部有额外half中间舍入，不保证逐位相同 |
| 本地 `mul_` | 更新用途 | 要求 `no_grad`；Python实数、同shape/dtype/device的Tensor，或同dtype/device的零维Tensor系数；完整密集排列与self-alias，拒绝不同stride重叠别名 | P/S21：零维设备系数支持标准范数裁剪；其他广播、混合dtype及一般可微inplace仍拒绝 |
| 本地浮点标量 `add_` | 更新用途 | no_grad，FP16/FP32；Python实数或分派形成的无梯度CPU零维实数常量；沿用add的dtype/alpha边界 | A：epsilon舍入；INT64计数入口不变 |
| 本地 `lerp_.Scalar` / `addcmul_` / `addcdiv_` | 更新用途，无autograd | no_grad、FP16/FP32、同shape/dtype/device、完整密集排列；仅Python实数系数，拒绝广播/部分存储/不同stride的重叠别名 | A：FP32 opmath、最终舍入、self-alias/version及拒绝契约；未注册一般out-of-place/其他lerp重载 |
| 本地 `exp/log/tanh/sigmoid/silu/sqrt/rsqrt` | 前向及一阶梯度 | FP16/FP32、rank0–4、有界布局只读整理及分块；out须精确shape/dtype、完整连续、无输入别名且无梯度；不支持相应inplace/二阶梯度 | S15：全部half编码、随机FP32位模式、NaN/Inf/正负零、scalar、65,537元素、布局/out/version guard；sqrt沿用整数开方，其余超越函数不宣称libm正确舍入；rsqrt梯度保留result.pow(3)的dtype/溢出边界 |
| 本地 `rms_norm` / `nn.RMSNorm` | 前向及一阶梯度 | FP16/FP32输入与weight，非空末尾normalized_shape、rank1–4；eps为0或非负有限FP32可表示数；None按本机Torch2.14的FP32计算dtype epsilon；输出为输入dtype | S15：多个归一化轴、无/有仿射、FP16输入+FP32权重梯度；用已有NPU运算组合，不是冻结SmolLM专用融合RMSNorm；受通用存储/会话限制 |
| 本地 `minimum/maximum` | 有界一阶 | FP16/FP32、rank0–4，支持广播/提升/有界只读view；out精确，maximum原同shape/dtype完整密集路径仍允许既有精确别名，其他out不别名 | A/S20：保留旧AMSGrad路径；相等时平分梯度、NaN两侧梯度按本机Torch处理；不认证NaN payload或signed-zero ties逐位一致 |
| 本地 `clamp/clip/clamp_min/clamp_max` | 有界一阶 | FP16/FP32、实数标量或同设备浮点Tensor边界，Tensor边界可广播/求导；out精确且无别名；inplace仅标量边界、no_grad、完整可写且shape/dtype不变 | S20：按Torch2.14区分标量/Tensor边界、相等/倒置/NaN和半精度平分梯度舍入；支持foreach=False的clip_grad_value_，范数裁剪另见S21，一般可微inplace仍缺 |
| 本地 `Hardtanh/ReLU6` | 有界一阶 | FP16/FP32，非inplace；标量min/max，继承clamp布局/容量 | S20：边界/NaN梯度按本机Torch的Hardtanh语义；原地ReLU见S22，原地Hardtanh/ReLU6仍缺 |
| 本地 INT64 标量 `add_` | 无梯度整数计数 | rank-0 INT64、Python有符号64位整数（或分派形成的CPU零维INT64常量）、alpha=1；可在grad模式中更新 | B：SHAVE两字unsigned32加法与进位、2^53以上精确递增、符号回绕；raw F32载体不作浮点数值转换，单任务完整双字布局校验 |

`out=` 仅在已注册的重载上可用，要求精确 shape/dtype，通常不能与输入共用 storage。
S5初版使用单完整SHAVE packet：令R为被选轴长度、D为其余维度乘积、L为索引数量，要求`6+2*L+max(R,L)*D<=65,536`且`R*D*L<=1,048,576`；这是额外的索引工作区/反向工作量限制，不是大词表支持。
S12保留这些单内核界限，自动把D按列拆成宽度d：`d=min(D,floor((65,530-2*L)/max(R,L)),floor(1,048,576/(R*L)))`，要求d>=1、源和输出各<=1,048,576元素。所有索引仍在每个packet内完整校验，列分块不会改变重复索引梯度的累加顺序；每个块读取一个4字节状态。未拆索引数量或词表行数，因此单列都放不下时仍在提交前拒绝。已验证2048×35的三块非整除前后向，并用于2048×64词嵌入的扩大模型；不是任意大词表/长序列支持，仍可能有较高的R*D*L反向开销。
FP16索引梯度按本机CPU语义区分：index_select在FP32累加后转回half，Embedding每次half更新后舍入；128次重复索引及subnormal梯度做精确对照，未放宽容差掩盖二者差别。
它不是普通自动微分接口；不要把无梯度的 `out=` 成功推广到可微调用。
FP32 补偿归约/分块可能与 CPU 累加顺序不同，FP16 也可能存在不同融合/舍入；使用相应测试的数值容差。
冻结模型内的RoPE/RMSNorm/attention图不代表任意公开算子均已支持；新增sin/cos/pow等范围见S25，softmax仍限专用有界入口。

## 训练与运行方式

| 项目 | 状态与建议 |
|---|---|
| SGD | 发布基线覆盖基本更新；本地 P 扩展检查普通单 Tensor 路径的 momentum、dampening、Nesterov、weight_decay、maximize 和 CPU checkpoint 恢复；显式 `foreach=False, fused=False`、Python 数值超参数；不认证 fused/foreach、Adam或可微优化器，AdamW见A；本地AMP有限接入见S17 |
| 本地 AdamW | 单Tensor、foreach=False/fused=False/capturable=False/differentiable=False、Python数值超参数；FP16/FP32独立更新与moment状态、AMSGrad、maximize、zero/None梯度、CPU checkpoint恢复通过；整网优先FP32，纯FP16轨迹仍有限 | 
| 本地 `clip_grad_norm_` / `get_total_norm` | S21：直接调用标准torch.nn.utils接口，NPU浮点梯度同dtype、完整密集存储；foreach=None/False，支持已列向量范数阶数；返回设备标量并原地更新梯度。error_if_nonfinite=True读取BOOL标量用于标准异常控制；False路径不下载payload。尚未认证混合dtype梯度、foreach=True、稀疏梯度或任意p |
| 本地 checkpoint 迁移 | 已覆盖显式将detached参数及SGD/AdamW状态转为CPU Tensor后torch.save/load(weights_only=True)，再加载到NPU并继续训练；AdamW step仍在CPU；原始NPU Tensor序列化/deepcopy仍未支持 |
| 一阶 autograd | 支持表中已实现的导数、分支梯度累加与部分密集布局；多数自定义 backward 标记 `once_differentiable` |
| 二阶导数 | 不支持；不能把 `create_graph=True` 的第一步返回当作二阶求导成功 |
| 后端导入顺序 | 在进程首次任何 backward 前导入 `torch_intel_npu`；晚注册可能触发 PyTorch 队列限制，C 验证局部 `set_multithreading_enabled(False)` 办法 |
| FP16 小网络 | 有限数值测试通过；不能保证深层训练轨迹与 CPU 一致 |
| FP16 Linear/激活 + FP32 LayerNorm参数 | 显式FP32 CrossEntropy的none/tanh GELU两组四步独立CPU轨迹通过，FP32参数梯度保留rtol=3e-5/atol=3e-6；延长至32步仍可拟合固定四样本，但第6/第5步起有严格差异，两组均未通过严格轨迹检查；FP16损失也仍严格失败，不认证通用AMP或长期收敛，见[最新边界报告](npunlock-probe/research/results/classification-precision/REPORT.md) |
| 本地混合训练误差归因 | 原第5/6步偏差可由SGD融合/分步乘加和half尾部舍入解释；相同输入局部算子/动量一致。64训练/64独立验证样本任务两侧准确率均75%、预测相同，仍有严格轨迹失败；不推广为任意输入差异均属舍入，见[归因报告](npunlock-probe/research/results/mixed-step-diagnosis/REPORT.md) |
| SmallResNet FP16 | 既有测试每步用相同权重重新建立 CPU 参考；通过逐项容差仍不认证独立训练轨迹或接近零梯度的相对误差 |
| 本地随机初始化 / RNG | S8/S16的FP16/FP32 `rand/randn/rand_like/randn_like`、`uniform_/normal_`在SHAVE生成；历史专项覆盖rank0–4与有界容量，原地目标须完整密集。支持原生`torch.Generator(device="intel_npu")`及默认Generator、独立seed/offset/state/clone，沿用24字节CPU状态；不保证CPU/CUDA相同序列。CPU Generator拒绝；S39新增Bernoulli与randperm，范围见下行及上节 |
| 本地 Bernoulli / randperm | S39：Tensor概率Bernoulli的广播、FP16/FP32/BF16概率、九dtype输出、原地/函数/out和零梯度；空输出不消耗RNG。randperm复用NPU INT64随机键/排序，并检查碰撞重抽；有默认/独立Generator及精确预分配out，支持八种实数/整数dtype；dtype精确表示/排序容量和实际验收范围见S39，不承诺大规模吞吐 |
| 本地离散随机数 | S16：INT64 `randint/randint_like/random_`，rank0–4、非空、<=524,288元素；有符号64位范围用Philox与无偏拒绝采样，原始整数位模式经DMA存储。out须已有精确shape；resize_仅允许同shape连续存储的无操作，不支持真实扩容。INT32/浮点random_尚未接入 |
| 本地 `multinomial` | S16：rank1/2 FP16/FP32非负有限权重，replacement及不放回；NPU FP32 CDF/24位均匀数，输出INT64，不对权重求导。所有行先在设备校验，每包同步读取4字节错误标志，之后才消耗RNG/写out。不放回严格要求samples<=正权重类别数。输出<=524,288、rows*categories*samples<=1,048,576，完整类别行/输出/scratch须可容纳于65,536 word包；按行分包，不拆类别。输入只读布局先DMA整理，out须精确shape/连续且无输入别名；未认证通用top-k/top-p采样流程或HF generate |
| 本地 autocast | S17：标准`torch.autocast("intel_npu",dtype=torch.float16)`，原生AutocastPrivateUse1与PyTorch可微转换/权重缓存。linear、mm/matmul/mv、addmm/bmm/baddbmm、conv2d/convolution、SDPA使用FP16；损失、LayerNorm/RMSNorm、exp/log/reciprocal/rsqrt使用FP32；sum/mean/softmax/log_softmax未显式指定dtype时使用FP32。out/inplace不自动转换，其他算子按原dtype执行；BF16按PyTorch规则警告并禁用。矩乘仍是已有SHAVE路径，无新增DPU训练或性能收益声明 |
| 本地 GradScaler | S40：torch.amp.GradScaler("intel_npu")与原专用入口；后端构造适配保留原类对象/提前导入别名，返回后端子类。单NPU、FP32主参数/完整密集FP32梯度，显式foreach=False/fused=False；控制元数据在CPU，梯度检查/反缩放在NPU，每优化器回读一个BOOL。支持Inf/NaN跳步、增长/回退、累积、多优化器、state_dict/pickle、iterable及单元素设备FP32 new_scale。SGD/AdamW及MLP/卷积/残差恢复通过既定参考；CPU Half严格差异、长期收敛仍有限。不支持稀疏、FP16主梯度、别名梯度、fused/capturable/differentiable、BF16 AMP或任意模型 |
| 本地普通 Dropout | S8：FP16/FP32 `nn.Dropout/F.dropout`非原地前向、native_dropout BOOL mask及一阶反向；p在[0,1]。eval/p=0返回输入，不消耗RNG，native p=1也不消耗RNG。小型FP32 MLP相同mask的CPU梯度/更新及checkpoint下一步逐位重放通过；SDPA内部Dropout见S9，Dropout2d/3d或完整Transformer仍未认证 |
| 本地标准 attention 模块 | S10覆盖小尺寸FP16/FP32 MHA/EncoderLayer；S11扩展标准TransformerEncoder稠密容器，S12扩大token模型，S13验收两层width32/heads4/sequence96/batch4的FP16/FP32训练、恢复与推理。S42接通默认strided Nested padding推理，须满足eval/no_grad及上游对齐条件；其他路线继承算子/score界限。jagged/Nested训练、任意配置、完整encoder-decoder Transformer/LLM及通用AMP未认证 |
| 本地标准 `TransformerDecoder` | S15复用现有算子验收两层、width8/heads2、target3/memory4/batch2、GELU、Dropout0.25；含自注意力、交叉注意力、padding、所有参数梯度/SGD动量更新、checkpoint/RNG逐位重放及eval。只认证此稠密小配置，未认证完整encoder-decoder语言模型、任意配置或长上下文 |
| 多线程 / 异步 | 多线程生命周期有检查，提交仍序列化；不支持 `non_blocking=True`、CUDA 等价的异步 stream 或 GPU/NPU 混合 Tensor 算术 |
| 生命周期 | 默认1,536项/512MiB blob；普通及其他可重建冻结图LRU，Smol图按S45同族失效重建。Smol超预算图或计划中危险淘汰提前拒绝，其他超预算可重建图仍可临时执行。未提供工厂的手工图pinned。预算可显式设置至1GiB；`shutdown()`后不能重新初始化 |

S19额外验收了95参数分割网络和443参数标准两层Transformer的四步AMP/CPU独立轨迹与恢复。
采用报告明确列出的FP16容差；Transformer参数梯度最大绝对差4.8828125e-4，不记为严格FP32轨迹通过。
这扩大了S17的模型检查范围；S40随后接通标准GradScaler入口，长期收敛及CPU Half严格差异仍保留。

## 冻结 DPU / 低位 / 模型适配器

| 入口 | 精度与范围 | 不应推断的能力 |
|---|---|---|
| `QuantizedLinear` | 显式冻结 INT8/INT4/NF4 权重，FP16 激活与输出；部分逐通道/分组、W8A16/W8A8 图 | 不是普通低位设备 Tensor、低位训练、QLoRA 或任意 GGUF/HF 量化 checkpoint 导入 |
| INT4/NF4 分组 | 有编码/IR/数值验证；部分 INT4 小 batch 用补零/截取规避编译问题 | 不保证图内仍保持四倍压缩，也不把压缩常量宣称为原生 4-bit 乘加 |
| `FrozenSmolLayer/Stack` | SmolLM2-135M-Instruct，FP16 / W8A16；显式 prefill/KV/decode；固定 64-position 容量，batch<=8，受工作区约束 | 不支持任意 LLM；不是 Transformers 模型 `.to("intel_npu")` 即可运行，也不是自动 torch.compile 后端 |
| 本地 `FrozenSmolIO` | 显式CPU FP16 embedding、最终RMSNorm及共享输出矩阵；从BF16/FP16/FP32的canonical tied checkpoint准备权重快照，D576/eps1e-5 | CPU两端辅助入口，与S5有界NPU Embedding不同；不提供CPU fallback、训练或独立lm_head权重支持；尚未发布 |
| 完整 30 层 FP16 | 历史固定短上下文测试通过，decoder/KV 在 NPU；两端和采样在 CPU | 不代表长上下文、任务质量、训练、功耗或多设备协同已认证 |
| 完整模型 W8 | 有校准与执行路径，既有独立/连续质量检查未全部通过 | 当前不能作为通用默认精度；四位执行与质量边界见S60 |
| 完整模型 INT4/NF4 | S60显式SmolLM2-135M 30层前向及8→9-token KV，本格式局部数值检查通过 | 原模型质量未通过；新模式尚无自由生成、PPL、grouped decode或任意模型认证 |
| 本地实验 `fp16_projections` | 校准W8中显式保留Up/Down原始FP16，真实30层、batch1、64位置融合路径通过129/142个质量检查位置 | 默认关闭、尚未发布；质量仍有限，性能仅限下述实测，不代表AMP或任意保护组合的硬件认证 |

低位与完整模型的详细证据分别见 [INT8](npunlock-probe/research/results/int8/REPORT.md)、
[INT4/NF4](npunlock-probe/research/results/lowbit/REPORT.md)、[模型对照](npunlock-probe/research/results/model-cross-device-followup/REPORT.md)
和 [W8 质量边界](npunlock-probe/research/results/w8-full-calibration/REPORT.md)。
新增本地参数的契约、质量与常量空间代价见[混合精度质量报告](npunlock-probe/research/results/w8-projection-mixed/REPORT.md)。
后续[逐层敏感度与logits偏移诊断](npunlock-probe/research/results/w8-layer-sensitivity/REPORT.md)不改变原质量认证或API范围；
逐层37.5%候选尚无独立质量优势，未升级正式接口。
新增[真实文本PPL与自由生成小样本](npunlock-probe/research/results/w8-ppl-generation/REPORT.md)比较了四种精度路径，
不改变旧logits门槛、固定64位置容量、公开API或模型质量认证范围。
2026-10-16的[端到端效率验收](npunlock-probe/research/results/w8-efficiency/REPORT.md)补充了固定24-token输入、32个预测位置、
batch1的FP16/W8/混合50%/逐层37.5%测速、进程内存、缓存启动及诊断DLL耗时拆分。
它没有扩大算子/模型/精度的认证范围，两个混合候选和W8仍未通过全部原质量门槛，默认不变。
随后完成[W8/混合精度prefill提前缓存命中](npunlock-probe/research/results/w8-prefill-cache/REPORT.md)：
缓存身份包含实际codes/scales、布局及保护策略；替换建图/反量化函数的研究路径保持完整建图。
完整30层、batch1/24-token的新进程与旧路径输出/KV逐位一致；第0层FP16/W8/混合另做边界、decode和动态gamma检查。
这只优化已有磁盘缓存的启动，不扩大模型质量、上下文或通用算子范围，磁盘缓存仍默认关闭。
新增[CPU启动准备优化](npunlock-probe/research/results/smollm-startup-prepare/REPORT.md)复用相同NPU图，
四种精度共12次完整32-token生成、hidden/KV及CPU输出逐位一致；混合50%改用既有公开参数一次构建。
该入口需显式采用，校准/完整CPU参考脚本继续保留其参考模型，不改变低位质量门槛或默认精度。

S24另外接通rank5的任意合法轴softmax/log_softmax，以及(N,C,D,H,W)硬标签NLL/CE和label smoothing；类权重、soft targets和既有标签packet界限不变。两种精度的局部前向/梯度见S24 volume验收。

## 明确拒绝与未支持场景

D 的 `unsupported_contracts` 与 T 的 `training_contracts` 集中检查非法输入/未支持场景，
包括超出已实现dtype/rank/容量、重叠目的、异步复制、mm的非二维输入、非法广播/alpha/out、
不兼容的MSE广播形状和非法BatchNorm/卷积参数。已接通的sin、整数/BF16、部分写入等已移出旧拒绝组；新增正向验收见S25/S29。
元数据拒绝组核对这些动作没有新增NPU提交；动态索引越界则须在设备校验后返回错误。
保存的权重 version 变更、二阶梯度、缓存容量以及 shutdown 后访问另有独立检查。

`asin`等未注册算子显式报错，不自动用CPU完成；S16的randint/multinomial与S39的randperm/Bernoulli有界范围见上表。
但不是每一个 PyTorch API 都有统一的预检入口：元数据 view 可以先创建，直到计算/复制绑定时才拒绝。
未出现在支持表中的 API，应视为未认证；尤其是超出S34验收范围的Embedding/索引、超出S6契约的拼接/切片、超出S28/S39契约的SDPA、
未列明配置的 torchvision/Hugging Face 模型（S41仅验收两个随机小配置）、量化训练和其他 PrivateUse1 后端共存。

## CUDA式模型迁移仍缺什么

此处目标是普通PyTorch模型采用`intel_npu`设备后的算子/训练语义兼容。项目未提供CUDA运行时、PTX、自定义CUDA扩展或NCCL兼容；`torch.cuda`专属代码仍需移植。

| 顺序 | 缺口 | 当前证据与下一步 |
|---|---|---|
| 已补子集 | 容量/索引/布局与dtype | S32为1GiB/rank64；S33冻结图重建和字节预算；S34扩展Embedding列工作区及排序/topk总量；S35降低Conv2d窗口峰值；S36/S37扩展普通MaxPool并解除共享选择反向的2²⁴单平面元数据限制；S38按字节用满1GiB并对密集窗口分块/反向剪枝。仍缺跨buffer分页、更多dtype算子和重叠写入；一般索引的其他检查、宽行计算、其他池化及次序统计仍有独立界限，约14GiB训练未认证 |
| 已补子集 | RNG、Dropout与离散采样 | S8/S9的SHAVE随机数及Dropout，S16原生Generator、INT64 randint及有界multinomial；S39补randperm、Tensor概率Bernoulli和状态重放。更广泛随机dtype/配置、大规模高效采样和通用HF生成仍缺，异步复制与原始设备序列化仍拒绝 |
| 已补子集 | Attention与HF迁移 | S10–S15补MHA/Encoder/Decoder及有限训练；S41补native空输入、序列算子及原版Llama/GPT-2小配置的cache/generate/训练验收入口。共享参数须显式交换转换；S42补实验分块Attention/strided Nested推理。完整encoder-decoder、top-p/beam、jagged/Nested训练、任意预训练模型及长期FP16训练仍未认证 |
| 已补子集 | 常用数学、激活与RMSNorm | S25增加sin/cos/tan/pow/expm1/log1p和四种舍入；仍缺其他数学/激活、更多原地变体和二阶导 |
| 已补子集 | 卷积、类轴损失与softmax | S19/S23/S24补卷积、类轴softmax和硬标签分类；S29补类别权重、可微软标签及广播MSE和原生损失接口。分类packet、其他损失家族和未认证配置仍有边界 |
| 已补子集 | 动态索引、平均池化与限幅 | S25公开gather/scatter_add/index_select不再使用S20的128行packet；take_along_dim及Embedding仍保留专用界限。S24扩展自适应池化；S25兼容优先的扫描式索引仍需优化 |
| 已补子集 | 极值归约、排序与范数 | S24补INT64/BOOL排序、k=0/空结果及有界次序统计；仍不支持超容量/缓存的任意大排序，其他p/矩阵范数、混合dtype/foreach梯度裁剪仍缺。近似相等激活的排序可放大整网数值轨迹差异，详见S23报告 |
| 已补子集 | 最大池化、原地ReLU与topk | S37普通MaxPool空间输出按INT64存储保护；选择反向以整数元数据支持超2²⁴单平面，TopK/sort前向仍沿用S34上限。S24的3D/AdaptiveMaxPool/Unpool前向另有约束；任意可微view原地写入仍缺。S23部分公开1D/InstanceNorm reshape出口以真实clone补齐原地激活组合 |
| 已补验收范围 | 视觉迁移 | S24补3D、抗锯齿、动态grid_sample/affine_grid、padding/shuffle和Unfold/Fold；剩余是特殊配置、原生out/backward变体、空batch/极值数据组合、超容量及默认大网络认证。具体矩阵和限制见S24报告，不能宣称全部视觉API完成 |
| 已补子集 | AMP | S17/S18有原生FP16 autocast、专用GradScaler、FP32主参数训练/恢复；六种卷积几何与两个四步网络按明确FP32累加参考验收，CPU原生Half严格差异仍单列保留。S40新增标准torch.amp.GradScaler入口及状态恢复；仍未认证任意模型或长期收敛 |
| 随后 | 其余模型缺口 | 专用dtype/算子/特殊配置、rank>64、超1GiB单buffer存储、二阶导与广泛模型验证仍缺；算子空Tensor支持不统一。旧混合训练/排序数值轨迹和低位整网质量边界保留 |
| 运行时 | 异步stream/event、编译、原始设备checkpoint与多设备 | 现有提交同步串行；普通intel_npu Tensor尚无通用torch.compile、CUDA graph或跨设备异步契约；CPU存储桥接的有限compile入口不能代替完整设备后端 |

S4向量/标量、S5有界索引/Embedding、S6布局、S7布尔mask、S8 RNG/Dropout、S9 SDPA Dropout、S10标准attention模块和S11稠密TransformerEncoder已补齐子集；见[向量/标量报告](npunlock-probe/research/results/vector-migration/REPORT.md)、[Embedding报告](npunlock-probe/research/results/embedding/REPORT.md)、[布局报告](npunlock-probe/research/results/layout-migration/REPORT.md)、[BOOL/mask报告](npunlock-probe/research/results/bool-mask/REPORT.md)、[RNG/Dropout报告](npunlock-probe/research/results/rng-dropout/REPORT.md)、[SDPA Dropout报告](npunlock-probe/research/results/sdpa-dropout/REPORT.md)、[标准模块报告](npunlock-probe/research/results/standard-modules/REPORT.md)和[稠密容器报告](npunlock-probe/research/results/transformer-container/REPORT.md)。这张表不是通用兼容性完成百分比。

## 本地普通模型迁移验收（2026-10-16）

P 的4个新进程共完成25组计算/训练检查和11组明确的未支持边界，6,501次真实NPU调用；
MLP、卷积网络、TinyResNet的FP16/FP32四步独立CPU对照通过，第四步从显式CPU checkpoint恢复。
14组SGD配置/精度组合覆盖密集转置参数、动量、衰减、Nesterov及maximize；
新增原地乘法、带alpha加减法的梯度/别名/version/拒绝契约检查。
各检查live buffer无增量，shutdown后live/cache/scalar/quarantine/session归零，无context failure。

该第一轮历史记录中的默认BatchNorm与整数target存储已由下述B补齐，有界CrossEntropyLoss由L补齐，AdamW由A补齐，LayerNorm/GELU见N，softmax/bmm见S；
高维matmul与有界SDPA后来由S2补齐，S5新增有界Embedding。原始NPU Tensor序列化仍拒绝，不计为支持。
检查口径、具体错误与CPU checkpoint入口见[迁移验收报告](npunlock-probe/research/results/torch-migration/REPORT.md)。
这批本地修复尚未同步至公开v0.1.0，也没有认证CUDA运行时/自定义CUDA内核兼容。

## 本地INT64与默认BatchNorm验收（2026-10-16）

B 覆盖3组整数检查、6种默认BatchNorm训练组合，所有整数对照使用精确相等。
六种模型组合均完成三步训练、CPU checkpoint恢复后的第四步、eval及running stats重置；
检查输出、loss、输入/参数梯度、SGD动量、running mean/variance和INT64计数，CPU轨迹独立运行。
固定momentum的热步骤无显式host payload copies；momentum=None按标准PyTorch模块逻辑读取8字节计数，
并可首次上传随步数改变的平均系数，不能把它描述成零拷贝模式。
该轮P更新为27组功能检查通过、9组明确未支持；分类loss缺口已由下述L补齐，原始NPU Tensor序列化仍需要Uint8存储支持。
具体计数、SHAVE布局诊断与源码指纹见[INT64与BatchNorm报告](npunlock-probe/research/results/int64-batchnorm/REPORT.md)。

## 本地分类训练验收（2026-10-16）

L以自定义SHAVE完整行内核实现稳定log_softmax、NLL数据包及一阶反向；FP16输入经设备内FP32计算后返回原dtype。
exp/log为独立编写的近似实现，已做CPU容差对照，不声明通用libm正确舍入认证。
ReferenceSW载体加载前检查完整行、单次调用及已验证物理布局；标签按两个原始32位字通过DMA，未数值转换成FP32。
纯FP16 BatchNorm现在按PyTorch CPU约定舍入保存的mean/invstd后归一化与反向；FP32累加和running buffer更新仍保留。
MLP、Conv+默认BatchNorm、轻量残差网络两种dtype共六组，逐项检查四步输出/loss/输入及参数梯度/参数/SGD动量/BN buffers，
第三步显式CPU checkpoint恢复，第四步继续独立CPU轨迹，并额外对照相同logits上的loss梯度。
该分类轮P为28组功能检查通过、8组明确未支持；后续AdamW见A，LayerNorm/GELU见N，softmax/bmm见S，高维matmul/有界SDPA见S2，S5补齐有界Embedding；原始NPU序列化仍未支持。
具体结果、修复前失败记录、复制统计及源码指纹见[分类训练报告](npunlock-probe/research/results/classification/REPORT.md)。

## 本地AdamW验收（2026-10-18）

标准torch.optim.AdamW直接使用新增更新算子；没有替换优化器、修改torch.cuda或使用CPU参数/动量算术回退。
FP16状态仍按参数dtype存储，默认eps=1e-8会舍入为零；零梯度产生NaN的CPU/NPU一致性是边界检查，不算健康训练通过。
正常FP16检查使用eps=1e-4，但仍有整网独立轨迹失败；不自动调整epsilon或切换FP32状态。
原含BN前卷积bias的FP32网络也出现参数轨迹差异，相同参数/动量/梯度下的CPU AdamW更新通过；
此类冗余bias应由使用者在模型定义时显式关闭，后端不会修改模型或强制把微小梯度归零。
既有SGD/分类训练范围保持；AdamW新增范围、严格失败记录与受控bias=False验证见[AdamW报告](npunlock-probe/research/results/adamw/REPORT.md)。
该AdamW轮的迁移测试为29组功能通过、7组明确未支持；这不是对全部PyTorch API的完成度百分比。

## 本地LayerNorm/GELU验收（2026-10-18）

N补齐同dtype FP16/FP32的LayerNorm与两种GELU公式的一阶梯度，新增六组残差前馈网络四步独立CPU训练对照。
迁移探针包含非平凡前向、输入梯度和参数梯度；该轮P为31组功能通过、5组明确未支持，保留为历史记录。
该轮历史记录中的4096宽混合LayerNorm失败已由N2的SHAVE有序Welford统计修复，原样例梯度与CPU逐位一致，22组扩大检查通过。
N2的整网范围仍有限：默认GELU、显式FP32损失的一组四步SGD对照通过；FP16损失两组及FP32损失/tanh一组仍有严格梯度失败。
前后数据、CPU半精度损失舍入差异和契约见[混合统计修复报告](npunlock-probe/research/results/layernorm-mixed/REPORT.md)，旧[LayerNorm/GELU报告](npunlock-probe/research/results/norm-activation/REPORT.md)保留历史结果。
这不表示完整Transformer或AMP已支持。后续softmax/bmm与SDPA范围见下；S5后来补齐有界Embedding，原始NPU序列化仍未支持。

## 本地softmax/bmm验收（2026-10-18）

S新增11组FP16/FP32检查全部通过，含小型attention网络四步SGD的损失、全部参数与梯度独立CPU轨迹对照。
分类损失显式FP32；不将四步结果推广为长期收敛、通用AMP或任意Transformer支持。
该轮需要显式调用三维`torch.bmm`，当时三维`@`/`matmul`与SDPA仍拒绝；后续S2补齐了有界入口。
该轮迁移边界11项中7项通过、4项明确未支持；这是历史记录，不把旧31/5统计当作当前全套验收。
计算区间显式copy计数仅含新标量常量及分类错误标志读取；该计数不覆盖驱动DMA/编译器行为。
实现、复现与回归见[attention算子报告](npunlock-probe/research/results/attention-primitives/REPORT.md)。

## 本地高维matmul/SDPA验收（2026-10-18）

S2接入标准matmul与F.scaled_dot_product_attention，不修改Torch全局函数；普通mm及原复制/跨设备autograd拒绝策略保持。
FP16的Q/K/V、attention分数与概率在NPU内用FP32运算，最终结果与梯度转回对应dtype；因果与全屏蔽行处理新增一个SHAVE内核。
双头因果attention加LayerNorm/GELU/残差的小网络通过四步FP16/FP32 SGD独立CPU对照；分类损失显式FP32。
多头训练发现并修复了contiguous复制的autograd缺口；没有通过改模型避开head合并或开放任意copy_梯度。
全部具体形状、精度、回归与当前迁移边界见[SDPA报告](npunlock-probe/research/results/sdpa/REPORT.md)。
这仍不是nn.MultiheadAttention/nn.Transformer或完整语言模型训练认证，也未验证长期收敛或性能优势。

## 本地持续训练与功耗（2026-10-18至19）

S3复用既有算子，未扩展支持契约。4层、hidden128、4头、FFN512、batch4/sequence16的793,604参数FP32网络，
完成两步独立CPU loss/全部梯度/更新对照，随后连续182.227秒、6步SGD，loss降到0.460670，NPU引擎平均忙碌91.67%。
会话与live内存稳定；256块缓存上限仍导致每步162次驱动分配/回收周转，不能据此宣称零分配。
独立一分钟相同模型复测由Intel PMT读到NPU域平均0.5251W、整包9.8173W；仅是SHAVE路径与当前后台环境下的读数。
六项离线监控/能量换算检查通过。完整口径、短测脚本误判及清理记录见[持续训练报告](npunlock-probe/research/results/sustained-training/REPORT.md)。
未验证该较大网络的FP16持续训练、真实数据集收敛、DPU全负载、外置功率计或跨设备能效。

## 本地DPU/CPU/GPU稳态能效（2026-10-19）

本地另有[2026-10-19 DPU/CPU/GPU能效报告](npunlock-probe/research/results/gemm-energy/REPORT.md)：
17项DPU校准、两轮12个同输入测量窗口及等价大矩乘候选通过数值/资源审计，9项离线监控与计量检查通过。
六tile/多输出图只是研究基准；未改普通torch.mm的SHAVE路径，也不扩展训练算子契约。当前高负载DPU功耗不能称为芯片满吞吐功耗。

## v0.1.0 发布基线检查（历史记录）

下文保留 Quick 的独立记录；随后完成的 15 步非 Quick 发布回归及其更大的 FP16 梯度相对误差，
见 [RELEASE_READINESS.md](RELEASE_READINESS.md)。

2026-10-11，在上述设备与原 `.conda/npunlock` 环境串行复测：Python 3.12.14、
Torch 2.14.1+cpu、NumPy 2.5.3，环境内没有安装 OpenVINO。沿用当前原生 DLL/PYD，
没有重编译计算内核；这与新环境源码构建验收是两项独立检查。

| 检查 | 实际结果 | NPU 调用 | 覆盖摘要 |
|---|---|---:|---|
| D：设备 Quick | 33 组，`passed` | 4,160 | 算术、精度/布局/复制、矩乘、生命周期；43 个拒绝场景未新增图提交 |
| T：训练 Quick | 11 组，`passed` | 2,732 | 一阶梯度、MLP/Conv/TinyResNet，两种 dtype 各 3 步 SGD；11 个训练拒绝场景 |
| C：独立兼容性进程 | 5 组，`passed` | 236 | 极值除法、转置叶、8 个单目梯度案例、4 个归约转换案例、晚导入办法 |
| M：模型 Quick | 9 组，`passed_with_limitations` | 7,026 | 分块梯度、BatchNorm、1×1 池化、WideMLP/SmallResNet，两种 dtype 各 3 步 SGD |

合计 **58 组、14,154 次 NPU 调用**，四条命令均退出 0。各进程 shutdown 后 live/cached/
quarantine/session 计数归零，`context_failed=False`、`closed=True`。
T/M 的网络热训练步显式 host payload copy 与新增图编译均为 0；该拷贝计数不包括驱动 DMA，
参考数值下载也在训练计数区间外。日志中每 120 秒的栈转储来自 faulthandler 诊断，不是外层截止失败。

需要保留的限定：

- C 中普通晚注册 backward 仍复现 PyTorch 内部队列限制；局部关闭 autograd 多线程的办法通过，并恢复原设置。
- FP32 SmallResNet 三步 loss 为 `0.23969901 → 0.02654167`；逐项梯度与更新通过既有容差。
- FP16 SmallResNet 三步 loss 为 `0.25805664 → 0.02212524`，每步重建同权重 CPU 参考。
  三步梯度相对 L2 误差依次为 `0.103172% / 0.147237% / 0.162798%`，
  余弦相似度为 `0.99999905 / 0.99999833 / 0.99999797`。逐项容差通过仍不认证独立 FP16 训练轨迹。
- 修正了测试脚本把限制说明写死为“15 步”的报告文字；Quick 实际仅 3 步，未改变计算或容差。

复现时串行运行，脚本会生成临时 JSON/日志；保留文字摘要后可按项目存储策略清理：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NPUNLOCK_GRAPH_CACHE = ''
.\run_torch_npu.ps1 -Device -Quick
.\run_torch_npu.ps1 -Training -Quick
.\run_torch_npu.ps1 -Script .\torch_npu_compat_demo.py
.\run_torch_npu.ps1 -Models -Quick
```

`Quick` 缩短循环/训练步数，不是完整压力测试。本轮没有重新认证冻结低位/完整 LLM 的历史结果，
也没有新增性能、长时间运行、其他驱动/设备或另一台干净 Windows 的验证。
源码构建的原生 session mock 回归与新环境验证，另见 [INSTALLATION_REPORT.md](INSTALLATION_REPORT.md)。

<details>
<summary>本轮代码指纹（SHA-256）</summary>

```text
torch_intel_npu.py
bd147eb2adbf49d092084ae4f1de767b395424016e9f8862eb850ab571b55cce
torch_npu_autograd.py
2e8c10fe2a097ce4a9d3ff5de8790965545c35288c9520b5261efb1ec72bc757
npunlock-probe/torch-device/backend.cpp
d6acf5d765b1024ef783e4370f30064e7c67aa4a26f7a10e9eba8c0864fe6d09
npunlock-probe/session-runtime/npunlock_session.dll
0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9
npunlock-probe/torch-device/_torch_intel_npu.pyd
3325f761ea55485271e26ecc3667b820f428d31f665e4c31fb916e8c30aefc8b
torch_npu_device_demo.py
0cd79eb28baf6f5c59a61052818fae12de8799e3dabd4a390938e0df70dfa829
torch_npu_compat_demo.py
bd40cd690364f67429fc38f0a179245140030a3d7c29c8556f51790cd5e523b0
torch_npu_training_demo.py (T运行版本，报告文字修正前)
60bd764ed9c945ac49a91942db39a1572f4897cfeb6958348bcb17a9acd57f71
torch_npu_training_demo.py (M运行版本及本次导出)
56d9f01a7f3ba56f79b328c0a4bde05b369b6e8a8d241e2dfcc0cbd51fbdfeda
```

完整源码文件集的哈希由导出目录 `SOURCE_MANIFEST.json` 保存。DLL/PYD 指纹用于标识本轮沿用的本机产物；
从源码另行构建可产生不同二进制哈希，不要求安装者匹配这些字节。原始临时 JSON/日志在文字记录验收后清理。

</details>
