# References / 实现来源

这里同时记录直接代码来源、接口与设计参考、验证工具。需要正式 reference 时可从
[references.bib](references.bib) 选取实际采用的条目；许可声明仍需保留 `NOTICE` 与 `LICENSES/`。

## 核心来源

S45继续使用S44校准思想做内部数据筛选的反证实验，没有引入新的量化实现或第三方代码。缓存身份修复及有界同族失效回避基于本机复现；Level Zero错误码与图扩展用于解释API，Linux驱动仅作调研，不能代替本机Windows实现或证明最终根因。方法、撤回的原生候选及实际边界见[缓存报告](npunlock-probe/research/results/smollm-cache-lifetime/REPORT.md)和[质量报告](npunlock-probe/research/results/w8-guarded-quality/REPORT.md)。

S44参考[LQER（ICML 2024）](https://proceedings.mlr.press/v235/zhang24j.html)的低秩量化误差重建思想，自行实现带1% ridge的激活协方差白化、CPU FP64 SVD/三角求解和FP16因子存储；图内增加两次FP16矩乘与加法，主投影保留原W8。没有复制该项目代码，也没有新增运行依赖。实际质量改善、退化项、原门槛与NPU证据见[S44报告](npunlock-probe/research/results/w8-lowrank-quality/REPORT.md)，BibTeX为zhang2024lqer。

S29损失函数扩展使用本机PyTorch 2.14.1+cpu作为输出/梯度及原生ATen接口的独立oracle，包括加权均值、软标签/权重梯度和原生MSE backward的广播语义。FP16舍入边界另核对安装包的`torch.version.git_version=5c4886908584029761b579af026dcfb627c84070`所对应[LossNLL.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL.cpp)与[LossNLL2d.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL2d.cpp)；不复制其CPU级联累加实现，也不承诺逐位一致。设备实现组合本项目既有SHAVE log-softmax/NLL packet、数学归约和DMA；未引入新的第三方内核代码、OpenVINO runtime或依赖。详见[损失函数报告](npunlock-probe/research/results/loss-extended/REPORT.md)与`test_torch_npu_losses.py`。

| 项目 | 使用关系及对应实现 | 固定来源 |
|---|---|---|
| npunlock | 直接修改的基础：Python symbolic 图、IR 编译、worker、ELF 操作和 Level Zero 会话；并扩展 NPU4000 和 Torch 集成 | [hsfzxjy/npunlock, bdf1c01](https://github.com/hsfzxjy/npunlock/tree/bdf1c01d2caa2e772315a9b2a529981c2810e037) |
| Intel NPU compiler | NPU40XX firmware / SHAVE ABI、任务描述、构建与编译配置；核对后实现本项目的校验、定位与替换 | [openvinotoolkit/npu_compiler, 0b38f7d](https://github.com/openvinotoolkit/npu_compiler/tree/0b38f7d42113ff329ac2bdd33583d123de4ccf2f) |
| Intel NPU compiler ELF | ELF section 类型、CMX workspace、reader 与 relocation 语义；用于本项目解析器边界检查 | [npu_compiler_elf, d325f45](https://github.com/openvinotoolkit/npu_compiler_elf/tree/d325f45f2cb405b5fa2ff17a30de9469f1641b73) |
| Intel Linux NPU driver | 图加载与扩展语义调研；不代表本后端已支持 Linux | [intel/linux-npu-driver, aea583d](https://github.com/intel/linux-npu-driver/tree/aea583dcd86a4723a8d3bb06ad94b6eb2f80c6f2) |
| Level Zero / Intel NPU extensions | Windows 原生会话使用的 loader/API/ABI，command list / fence / graph / shared allocation 的生命周期 | [Level Zero v1.17](https://github.com/oneapi-src/level-zero/tree/5604d3963dec3c98d4e56ccf6fef0671a47114a2)、[NPU graph ABI 1.20 文件](https://github.com/intel/level-zero-npu-extensions/blob/ddbf8d65e2e22150e23efd383839aa80f84d4dac/ze_graph_ext.h) |
| OpenVINO / bitsandbytes NF4 | 直接使用公开的 16 项 NF4 码表；OpenVINO 文件说明了 bitsandbytes 来源。不是移植完整 runtime 或 CUDA 内核 | [OpenVINO nf4.cpp](https://github.com/openvinotoolkit/openvino/blob/8fb839633478574cefd28d49e25747d7c830b5a6/src/core/src/type/nf4.cpp)、[bitsandbytes 原始内核](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/c82f51c0f784d8a43ebcb9cdefbf94e3f3b9c6c3/csrc/kernels.cu) |
| GPTQ | 直接改编 W8 激活感知校准/误差反馈，CPU FP64 与固定 FP16 scale，见 `calibrate_int8` | [IST-DASLab/gptq, 2d65066](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py) |
| FP16 | 直接改编 IEEE FP16 转换算法为 SHAVE SIMD，见 `rmsnorm.c` | [Maratyszcza/FP16, 782eea1](https://github.com/Maratyszcza/FP16/blob/782eea126dc5c755827be751a099eb01826175cf/include/fp16/fp16.h) |
| PyTorch | PrivateUse1、dispatcher、autograd、Tensor/allocator/C++ extension 接口；CPU 算术作为数值参照。本机 NPU 环境 Torch 2.14.1+cpu | [PyTorch 源码](https://github.com/pytorch/pytorch)、[PrivateUse1 文档](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) |

NPU compiler / ELF / driver 调研的 **37 个具体文件、URL 与 SHA-256** 保存在
`npunlock-probe/research/sources/manifest.json`。本次导出不带这些第三方完整源码副本或预编译 Intel kernel。

## 性能优化调研与验证参考

- [PyTorch ufunc/add.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ufunc/add.h)、本机Loops.h/Half.h/vec256_half.h：核对Tensor add_的alpha转换、CPU向量乘加和half标量尾部中间舍入；本项目在设备分派中复用已有SHAVE算术和转换实现，没有复制CPU SIMD代码。数值样本、安装文件指纹与仍存在的尾部边界见[混合训练更新报告](npunlock-probe/research/results/mixed-training-rounding/REPORT.md)。
  后续[SGD舍入归因](npunlock-probe/research/results/mixed-step-diagnosis/REPORT.md)用本机vec256_float.h/vec256_half.h、UCRT fmaf核对融合结果，并通过只改变CPU参照更新顺序的实验解释原第5/6步分叉；[PyTorch数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)用于说明跨平台严格逐位一致性的边界，不替代实际验证。

- [PyTorch moments_utils.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/moments_utils.h)：混合LayerNorm直接改编其16步分组、8个FP32 lane及Welford级联合并顺序，保留BSD许可，见[来源指纹](provenance/layernorm-moments.json)。本机Torch 2.14.1+cpu安装头文件SHA-256为`fc7598344925a3cfb1e42afe81fc350cbbfe2f2973ba053c0765f601ff46efba`；移动的main URL只作导航。SHAVE整数融合乘加与设备数据传输为本项目实现，既有普通归约未改写为此顺序。

- [Abramowitz与Stegun，Handbook of Mathematical Functions，p.299，式7.1.26](https://personal.math.ubc.ca/~cbm/aands/page_299.htm)：`resident_gelu.c`的erf近似使用其公开数学系数，本项目自行实现SHAVE求值与边界处理，复用自有exp/IEEE辅助函数；没有复制第三方libm源码。默认GELU仍按erf公式计算，不以tanh替代。
- [PyTorch Activation.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/Activation.cpp)和[layer_norm_kernel.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/layer_norm_kernel.cpp)：核对GELU两种公式、解析一阶导数和LayerNorm保存统计的dtype、FP16参数梯度缓冲的舍入。后端用自有算子/内核实现，未移植CPU SIMD代码；宽层混合affine曾有严格失败，现已通过有序统计修复；更广混合训练仍有明确边界。

- [PyTorch LogSoftmaxKernelImpl.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h)与[SoftMaxKernel.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/SoftMaxKernel.cpp)：本地分类算子参考其稳定的先减最大值、再减log(sum(exp))的运算顺序与一阶导数公式，未复制CPU向量实现。`resident_classification.c`中的exp/log为本项目独立编写的区间缩减与Taylor/atanh展开，复用本项目已有FP32/subnormal辅助函数；没有引入第三方libm依赖。
  后续[FP64分类对照](npunlock-probe/research/results/classification-precision/REPORT.md)使用同一已量化logits核对数学误差，区分CPU half保存统计的溢出/舍入与NPU计算质量；CPU参考仅用于验证，不作为运行时数据计算回退。
- [Arm optimized-routines expf.c](https://github.com/ARM-software/optimized-routines/blob/master/math/expf.c)与[logf.c](https://github.com/ARM-software/optimized-routines/blob/master/math/logf.c)：调研区间缩减、近似与特殊值处理，未移植其源码、系数表或架构实现。
- [PyTorch Lerp.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Lerp.h)、[LerpKernel.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/LerpKernel.cpp)与[PointwiseOpsKernel.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp)：本地AdamW更新算子核对稳定lerp分支、FP16的FP32 opmath以及value*a*b/value*a/b的求值顺序；以本项目既有SHAVE算子组合实现，未移植CPU SIMD代码。sqrt复用本项目已有整数开方，maximum独立使用IEEE位排序；标准优化器控制流直接使用本地Torch的`torch.optim.adam._single_tensor_adam`，没有替换AdamW实现。
- [PyTorch性能指南：关闭BN前卷积bias](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide#disable-bias-for-convolutions-directly-followed-by-a-batch-norm)：本地AdamW整网实验显式比较保留/去掉冗余bias的模型配置，其他初始权重与RNG消耗保持不变；后端不会自动执行这种模型改写。
- [PyTorch Normalization.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Normalization.cpp)与[batch_norm_kernel.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/batch_norm_kernel.cpp)：核对纯FP16 BatchNorm保存均值/逆标准差的dtype边界，独立实现相同舍入边界；统计归约仍使用本项目FP32算子。[LossNLL.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossNLL.cpp)用于核对ignore_index、total_weight与reduction契约，没有复制CPU内核。
- [OpenVINO NPU plugin](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_npu)：提交开销与 DPU 路线对比，见 [OpenVINO 对照研究](npunlock-probe/research/INTEL_NPU_LIBRARY_COMPARISON.md) 及实验报告。普通执行不依赖 OpenVINO runtime。
- [llama.cpp SYCL norm](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-sycl/norm.cpp)、[PyTorch CPU moments_utils.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/moments_utils.h)：归约组织调研。有序/分组补偿归约候选未获性能收益，最终采用 FP16 SIMD 转换，见 [RMSNorm 报告](npunlock-probe/research/results/rmsnorm-reduce/REPORT.md)。不要写成照搬 GPU 并行内核。
- [llama.cpp ggml-quants.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)：`w8_scale_candidate.py` 中固定码字的最小二乘 scale 思路；研究候选，不是默认量化规则。
- [GPTQ opt.py，2d65066](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/opt.py)：逐层使用量化前缀重放校准的思路；本项目复用已有校准器自行实现研究调度，未复制该文件代码。顺序候选未胜出，见[完整模型低位质量对照](npunlock-probe/research/results/w8-sequential-calibration/REPORT.md)。
- [Intel NPU Acceleration Library](https://github.com/intel/intel-npu-acceleration-library)：现成方案能力对比；不是本项目调用的运行库。
- [Hugging Face Transformers](https://github.com/huggingface/transformers)、[SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)：真实模型、tokenizer、CPU/GPU 数值/性能参照；模型权重不随源码发布。

上述部分链接是未固定的 `master/main`：现存记录不足以恢复当时检查的准确提交，因此只作导航。
不把本轮查询到的 HEAD 冒充为历史设计来源。若后续直接移植其中具体代码，应补固定版本、文件哈希与相应许可。

## 硬件活动与能量计量

- [Intel Unified Telemetry](https://www.intel.com/content/www/us/en/developer/tools/intel-unified-telemetry.html)：2026-10-19使用v0.2.0-beta1.1的SoCWatch读取NPU、VCCGT与整包能量，按实际区间换算功率和每次矩乘能耗。仅外部实验工具，不加入运行依赖或源码分发；下载包及签名校验后使用，实验后清理。
- [Microsoft DXCoreAdapterState](https://learn.microsoft.com/en-us/windows/win32/api/dxcore_interface/ne-dxcore_interface-dxcoreadapterstate)：累计引擎运行时间与内存计量接口。tools/npu_dxcore.py通过设备ID/LUID排除核显；引擎忙碌率不当作DPU阵列利用率。
- [Intel 258V规格](https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html)：47TOPS INT8仅作硬件规格参照，不冒充实测值。
- [OpenVINO NPU属性定义](https://github.com/openvinotoolkit/openvino/blob/master/src/inference/include/openvino/runtime/intel_npu/properties.hpp)：核对NPU_TILES编译配置。能效基准通过已有驱动编译接口生成DPU图，没有引入OpenVINO runtime，也没有修改生产编译策略。

## 模型质量评测

本地完整模型质量评测另使用以下数据与方法参考，见[PPL与自由生成报告](npunlock-probe/research/results/w8-ppl-generation/REPORT.md)：

- [Salesforce WikiText，b08601e](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3)：固定 `wikitext-2-raw-v1/test` 的8份小样本。数据文本和token IDs不随源码发布；原数据许可见其数据卡。
- [Dataset viewer rows API](https://huggingface.co/docs/dataset-viewer/rows)：以标准库读取固定行范围并校验 `x-revision`，不需要 datasets/Parquet依赖。
- [Transformers固定上下文PPL说明](https://huggingface.co/docs/transformers/perplexity)：用于解释上下文/步长对PPL的影响。本项目自行实现按唯一目标token加权的损失，并与Transformers内置loss交叉核对，没有复制示例代码。

本地softmax/bmm扩展另核对[PyTorch softmax接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html)与[bmm接口](https://docs.pytorch.org/docs/2.14/generated/torch.bmm.html)，
以及本机Torch 2.14.1+cpu的ATen schema（softmax backward out的关键字为`grad_input`）。
softmax使用已有自编exp/IEEE辅助函数及补偿求和，解析梯度为`y*(g-sum(g*y))`；bmm复用本项目广播、乘法、补偿归约和矩阵分块。
未复制PyTorch CPU SIMD或新增OpenVINO runtime依赖；新C/ELF/构建与真机证据见[attention算子报告](npunlock-probe/research/results/attention-primitives/REPORT.md)。

高维matmul/SDPA扩展核对[PyTorch matmul广播契约](https://docs.pytorch.org/docs/2.14/generated/torch.matmul.html)和[SDPA接口/数学路径](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。
按Torch数学路径在FP32中预缩放Q/K，采用标准链式法则及广播轴梯度归约；CPU对照显式选MATH实现。
safe/causal softmax与clone布局梯度由本项目实现，复用已有IEEE/exp与设备DMA；不复制CPU SIMD、FlashAttention或新增OpenVINO依赖，见[SDPA验收](npunlock-probe/research/results/sdpa/REPORT.md)。

## 对外描述建议

> An experimental PyTorch PrivateUse1 backend for Intel NPU4000, built on a modified npunlock.
> It uses Intel's driver graph compiler and custom SHAVE kernels, with ABI references from the
> open Intel NPU compiler/ELF ecosystem. Selected routines adapt GPTQ and FP16; the NF4 codebook
> follows OpenVINO/bitsandbytes. OpenVINO runtime is not required for the default execution path.

不能据此描述为“完全独立于 Intel 图编译器”“官方 PyTorch NPU 支持”或“已把 NPU 接入 torch.xpu”。

## 设备随机数与Dropout

本地SHAVE RNG依据[Random123的Philox算法说明与原始实现](https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h)独立编写Philox4x32-10的32位分肢实现；未复制其头文件或引入运行依赖。每个输出元素占用一个计数器块，FP16/FP32均匀数分别取11/24个高位，正态数采用Box–Muller变换，自编log/sqrt/cos辅助函数。
接口与状态语义核对[PyTorch random/fork_rng](https://docs.pytorch.org/docs/2.14/random.html)及[Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html)，ATen重载以本机Torch 2.14.1的schema为准。相同seed不承诺与CUDA/CPU相同序列；数据在NPU生成，主机仅维护seed/counter元数据，见[RNG验收报告](npunlock-probe/research/results/rng-dropout/REPORT.md)。

SDPA内部Dropout扩展核对[PyTorch SDPA接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)与[v2.14.1数学实现](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/transformers/attention.cpp)：按概率应用Dropout，eval须显式传0；half输入保留FP32中间运算。独立CPU参考的输出/梯度用官方数学路径及其仅供测试的dropout_mask入口交叉校验；NPU运行时复用本项目既有SHAVE RNG与一阶链式求导，没有调用该CPU入口或引入新依赖，见[SDPA Dropout报告](npunlock-probe/research/results/sdpa-dropout/REPORT.md)。

标准模块迁移核对[MultiheadAttention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html)、[TransformerEncoderLayer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html)及本机Torch 2.14.1源码。native attention对合并mask转BOOL并使用普通masked-softmax的行为参考上述attention.cpp；本项目推理入口分解为现有NPU算子，不复制CPU数值内核。原生层仅注册PyTorch自带_reshape_alias元数据实现，仍依赖同版本PyTorch构建。

标准稠密容器扩展核对[TransformerEncoder](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html)、[triu](https://docs.pytorch.org/docs/2.14/generated/torch.triu.html)和[all](https://docs.pytorch.org/docs/2.14/generated/torch.all.html)，以及本机Torch 2.14.1源码/schema与CPU行为。三角数据移动复用现有byte-Gather零哨兵；BOOL all/any由本项目设备BOOL转换、FP32计数及比较组合，未复制CPU数值代码。padding推理显式选择官方容器参数enable_nested_tensor=False，不伪造NestedTensor支持。

FP16 native masked-attention的舍入序列另核对[PyTorch v2.14.1 SoftMax.cpp](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/SoftMax.cpp)及本机c10 Half运算重载，并以CPU原生API逐步交叉验证。新增SHAVE C独立实现FP16减法/exp/倒数/乘法的舍入边界，复用本项目IEEE转换与exp辅助函数；普通softmax/训练SDPA不采用该兼容序列。

S13的完整行分块与因果query行偏移为本项目自编调度/元数据扩展，复用上述SHAVE数值实现，不属于FlashAttention移植或在线softmax算法。大尺寸广播/因果/相同Dropout mask的CPU参考额外与本机ATen数学SDPA的输出及梯度交叉校验，设备端不调用该参考；见[完整行分块报告](npunlock-probe/research/results/attention-rows/REPORT.md)。

S14按元素数复用转换图为本项目缓存键/元数据视图调整，复用既有arithmetic.c的IEEE转换与DMA halfcode压紧；没有复制新的外部数值代码、修改SHAVE公式或新增依赖，见[转换图复用报告](npunlock-probe/research/results/cast-reuse/REPORT.md)。

S15的公开exp/log/tanh/sigmoid/SiLU复用本项目arithmetic/classification/GELU辅助函数并扩展数值域与特殊值分支；sqrt/rsqrt沿用既有内核。激活导数及RMSNorm的FP32 opmath/default epsilon/输出dtype顺序核对本机Torch2.14.1的`torch/_decomp/decompositions.py`和CPU行为，rsqrt另核对`-.5*grad*result.pow(3)`的dtype与溢出边界。标准Decoder直接使用本机PyTorch模块及既有独立同mask参考；没有新外部依赖，见[数学/模型报告](npunlock-probe/research/results/unary-math/REPORT.md)。

S16沿用上文Philox4x32-10，新增完整64位范围拒绝采样及FP32 CDF多项采样；复用本项目IEEE helper和字节DMA，不依赖NumPy/CPU生成随机payload。原生Generator按本机Torch2.14.1的`c10/core/GeneratorImpl.h`及`ATen/detail/PrivateUse1HooksInterface.h`实现getNewGenerator/getDefaultGenerator、seed/offset/state/clone接口；使用当前hook而非已弃用的REGISTER_GENERATOR_PRIVATEUSE1。沿用NPURNG01状态格式，未新增运行依赖或复制外部Generator实现。约束、数值检查及来源边界见[离散RNG报告](npunlock-probe/research/results/discrete-rng/REPORT.md)。

S17直接使用本机Torch2.14.1的`ATen/autocast_mode.h`中KERNEL_PRIVATEUSEONE/CastPolicy/cached_cast，在AutocastPrivateUse1注册有界策略和fallthrough，未复制PyTorch数值内核。NPU专用GradScaler继承`torch/amp/grad_scaler.py`的OptState、step/update、CPU控制状态及checkpoint，重写NPU梯度预检与反缩放；独立CPU标准GradScaler交叉核对跳步和增长序列。模型/梯度运算复用既有NPU算子，没有新增运行依赖/SHAVE ELF，见[AMP报告](npunlock-probe/research/results/amp/REPORT.md)。

S18核对[PyTorch v2.14.1 ConvolutionMM2d.cpp](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/ConvolutionMM2d.cpp)的逐batch累加实现，未复制源码。新CPU参考用原生FP32/FP64卷积及其导数独立组织精度边界，仅参与验收；scaler空checkpoint处理沿用已安装Torch2.14.1 GradScaler的RuntimeError。范围及保留的原生CPU差异见[卷积归因报告](npunlock-probe/research/results/amp-conv/REPORT.md)。

S19使用本机Torch2.14.1的普通Conv1d/Conv2d、softmax/log_softmax、NLL/CE及CPU autocast作为独立数值控制，未复制外部实现。分组/类轴整理、label smoothing和rsub派发基于本项目既有SHAVE/DMA/梯度原语组合；仍沿用原内核来源与许可，不新增OpenVINO依赖。范围、半精度容差及原有严格差异见[常用接口迁移报告](npunlock-probe/research/results/portability/REPORT.md)。

S20以已安装Torch2.14.1的CPU gather/scatter_add/take_along_dim、平均池化、极值/限幅及其autograd作为独立控制。64位索引规范化采用本项目自写整数逐位余数，沿用原SHAVE/IEEE helper；没有复制第三方实现。AvgPool复用本项目分组卷积和公开几何除数，自适应池化使用形状确定的窗口。标量/Tensor限幅边界和NaN导数按本机版本分别核对；这不建立跨PyTorch版本的数值承诺。新增来源/验证范围见[索引/池化报告](npunlock-probe/research/results/index-pool/REPORT.md)。

S21的SHAVE行极值和INT64候选合并由本项目实现，使用原始IEEE位序处理subnormal、首个tie/NaN；未复制外部归约内核。范数组合复用已有NPU算术；标准梯度裁剪直接执行已安装Torch2.14.1的torch/nn/utils/clip_grad.py，仅补它所调用的ATen接口。CPU arg/extreme/vector_norm及其autograd作为独立数值控制，源码来源与容量/数值边界见[极值/范数报告](npunlock-probe/research/results/reduction-norm/REPORT.md)。

S22以本机Torch2.14.1的MaxPool、topk、原地ReLU及其CPU梯度作为独立控制。池化窗口地址由公开形状/参数生成；SHAVE有序选择、INT64候选合并和线性索引梯度为本项目实现，复用IEEE helper及DMA，没有复制外部选择/池化内核。MaxPool的末个NaN与topk首个tie策略分别实现；原地ReLU使用PyTorch mark_dirty/increment_version与一阶Function接口。范围、平局索引差异及拒绝边界见[MaxPool/topk报告](npunlock-probe/research/results/pool-topk/REPORT.md)。

S23核对本机Torch2.14.1的`ATen/native/UpSample.h`坐标/尺度及cubic A=-0.75公式、ATen schema和CPU sort/极值、GroupNorm/InstanceNorm、ConvTranspose及其autograd。Python静态采样/转置卷积几何、归一化组合和SHAVE merge/bitonic排序为本项目实现，复用现有FP32算术、INT64原始字与DMA；未复制外部数值内核或新增OpenVINO依赖。CPU还用于辨别旧显式scale前后向不一致情形、Half BatchNorm中心化点积舍入顺序，以及近似平局排序的整网敏感性；这些参考不进入生产设备执行。范围、原失败记录和固定路由参考的区别见[视觉/排序报告](npunlock-probe/research/results/vision-sort/REPORT.md)。


S24核对本机Torch2.14.1的ATen schema、`ATen/native/GridSampler.h`的归一化坐标、边界反射及梯度约定和`UpSample.h`，以原生CPU前后向作为独立控制；抗锯齿triangle/Keys(-0.5)的边缘归一化同时通过CPU采样及伴随参考验证。ND卷积/池化/裁剪/shuffle/Unfold几何、动态grid stencil与读取、INT64逐字排序及排序行统计为本项目新增实现，复用已有IEEE FP32/DMA/索引累加；未复制外部数值内核或新增OpenVINO依赖。FP16缺少CPU原生API的三类算子单列FP32-opmath参考，不能写成原生CPU Half对照。构建拒绝新统计/grid内核的绝对跳转表与独立函数调用；开发期失败、界限和测试结果见[S24报告](npunlock-probe/research/results/vision-sort-extended/REPORT.md)。

S25以本机Torch2.14.1的ATen schema、CPU dtype promotion、整数溢出/整除、布尔/高级索引布局、原地version及一阶autograd作为独立控制。新增原始字类型转换/U64算术、动态索引packet、192位常量三角范围约减与多项式为本项目实现，复用既有IEEE helper和DMA；没有复制外部内核或增加OpenVINO依赖。Scalar/Tensor pow与CPU Half/FP32的不同快速路径分别测试；冻结DPU容量与普通Torch存储分开，未扩大既有低位质量结论。源码、验收口径和剩余边界见[S25报告](npunlock-probe/research/results/compat-foundation/REPORT.md)。

S26扩大既有Level Zero shared allocation与byte-DMA存储契约，保留原SHAVE任务和冻结DPU的边界；普通Torch CPU作为独立完整梯度/SGD参考，新增Scalar pow(x,1)/pow(x,2)复用既有copy/multiply，不引入外部内核代码或依赖。数值和学习门槛未因容量扩大而放宽，参见[S26报告](npunlock-probe/research/results/capacity-model/REPORT.md)。

S27的`resident_stream_sum.c`是项目内新增实现，沿用本仓库`resident_reduce.c`的Neumaier操作顺序和`arithmetic.c`的subnormal安全FP32 helper；通过既有ReferenceSW完整行carrier执行，不引入外部内核片段或OpenVINO运行依赖。高rank矩乘、视图整理和密集梯度复制复用现有byte-DMA及Torch接口；普通Torch CPU提供独立输出、梯度和SGD参考。完整partial流压力测试与真实GEMM/训练的范围分开记录，见[S27报告](npunlock-probe/research/results/matmul-workspace/REPORT.md)。

S28以本机PyTorch math SDPA作为GQA/MQA的独立CPU数值、梯度与训练参考，head排列另与原生`repeat_interleave`及其反向对照；同mask Dropout复用已经过Torch math核验证的项目CPU oracle和Philox控制。NPU实现仅复用已有DMA广播、FP32归约与softmax/矩乘，新增head布局和梯度合并代码，没有引入外部内核片段、OpenVINO运行依赖或新二进制。源码与验证范围见[S28报告](npunlock-probe/research/results/sdpa-extended/REPORT.md)。

S30的显式范围绑定、所有权/边界/别名保护、按元素数复用逐元素图和流式缓冲写回为本项目实现，沿用npunlock的Level Zero共享分配与现有SHAVE/DMA图。小模型使用独立PyTorch CPU前后向/AdamW全量参考；较大模型只用实际NPU梯度的采样CPU AdamW更新参考，两者分开记录。Windows GlobalMemoryStatusEx仅用于宿主物理/commit余量监测。未复制外部内核或引入OpenVINO/其他依赖；见[S30报告](npunlock-probe/research/results/memory-scale/REPORT.md)。

S31的完整行分类分块、校验标志设备归约、反向mask重建、行对齐及mean缩放复用为本项目Python实现，沿用已有classification/DMA内核。稠密工作区仅解除旧保护，未扩大浮点索引元数据范围；CPU PyTorch提供独立loss/梯度/SGD、Conv2d、原生encoder及Half masked-softmax参考。CPU不支持的oneDNN Half卷积反向通过关闭MKLDNN获得原生CPU参考，不进入NPU路径。无外部内核片段或新依赖，范围见[S31报告](npunlock-probe/research/results/operator-capacity/REPORT.md)。

S32的图LRU、Python对象所有权延后释放、冻结图固定保留及1GiB存储检查为项目内实现，复用既有npunlock/Level Zero会话和SHAVE/DMA。CPU PyTorch提供逐元素完整输出/梯度与动态batch训练的独立参考；未引入外部实现、OpenVINO依赖或新内核。历史运行中销毁导致设备丢失的记录保留，当前配置实际压力范围见[S32报告](npunlock-probe/research/results/graph-capacity/REPORT.md)。

S33的弱引用重建工厂、图blob字节预算和超预算临时session为项目Python实现，复用既有编译、profile/ELF检查与native会话生命周期；不复制外部缓存或内核代码，也不新增依赖。低位Linear与Transformer的CPU参考沿用本机PyTorch/项目已有显式量化参考；SmolLM的CPU staged prefill参考与无回收NPU的逐位输出/KV基线分开记录。四层prepared chain验证图所有权及KV快照，未据此扩大完整模型质量结论。2.69亿参数训练使用项目残差网络与原生CPU AdamW抽样更新/状态参考，没有同规模CPU全梯度镜像；见[S33报告](npunlock-probe/research/results/frozen-capacity/REPORT.md)。

S34复用项目已有的raw-byte dynamic gather、精确INT64地址运算、selection backward及sort/bitonic内核，通过Python特征列分块扩大Embedding工作区并调整排序总量保护；未移植新外部代码或修改C/ELF。CPU PyTorch的embedding/index_select/sort/topk及SGD提供独立前向、完整梯度和训练参考；浮点零容差数值比较与编码逐位认证区分说明，见[S34报告](npunlock-probe/research/results/index-capacity/REPORT.md)。

S35的窗口工作区选择、FP32逐块组装及尾部对齐均为项目Python实现，复用已有S30范围绑定/DMA、卷积窗口几何、SHAVE矩乘和权重梯度归约；没有复制外部实现或重编C/ELF。CPU PyTorch原生卷积（关闭MKLDNN）及SGD提供完整前向、输入/权重/偏置梯度和训练参考。保留S34四个卷积函数作基线，单次冷耗时不等同于隔离内核吞吐，见[S35报告](npunlock-probe/research/results/conv-capacity/REPORT.md)。

S36的跨通道窗口、完整平面分组、页对齐源范围与INT64索引恢复复用项目已有DMA和SHAVE内核，没有移植外部代码。`ctx.set_materialize_grads(False)`采用本机PyTorch自定义autograd接口，CPU原生MaxPool/max/min/argmax及标准训练提供独立参考；几何边界另用完整轴枚举对照。实现与验证见[S36报告](npunlock-probe/research/results/pool-capacity/REPORT.md)。

S37在项目resident_selection.c中新增原始UINT32 header解码，复用已有索引高低字、Half逐次累加和赋值实现；沿用项目Movi编译/汇编/链接工具流程，旧62枚ELF经重建指纹核对不变。Python分包和页对齐沿用已有实现，没有移植外部代码。CPU原生MaxPool/max/min/argmax/argmin提供完整参考，跨包Half用顺序舍入参考，原始字协议另有非法包和负零验证。见[S37报告](npunlock-probe/research/results/selection-capacity/REPORT.md)。

S38的字节容量、页步长、密集窗口C内核和有序反向范围为项目实现，复用已有selection_better位序、Half转换、UINT32 selection、范围DMA与Movi构建流程；没有复制新外部实现。CPU原生MaxPool验证真实输入/完整梯度，kernel1最大形状用CPU分块解析参考；宿主C oracle仅拓宽入口/指针以检查原C算法，不作为SHAVE性能证据。实现、开发异常和最终验证见[S38报告](npunlock-probe/research/results/storage-shape/REPORT.md)。

S39复用项目已有Philox SHAVE整数/浮点RNG、INT64排序、BOOL校验和DMA，实现Tensor概率Bernoulli与随机键排序randperm；碰撞时整批重抽，以免稳定平局引入索引偏差。Attention空维度的值、梯度和RNG语义以本机PyTorch math SDPA为参考；零QK特征及非零Dropout另用原独立CPU公式/Philox oracle验证。没有移植新外部代码、安装依赖或新增内核；范围及证据见[S39报告](npunlock-probe/research/results/attention-rng-boundaries/REPORT.md)。

S40依据本机已安装PyTorch的torch.amp.grad_scaler源码核对构造、iterable、dtype提升与状态机语义，继续继承其step/update/state_dict方法；构造分派与设备Tensor控制适配为本项目Python实现。CPU原生GradScaler提供独立数值/状态参考，卷积保留既有FP32-opmath及CPU Half双重参考。没有打包新的PyTorch源码或新增SHAVE/OpenVINO实现；许可与PyTorch来源沿用已有条目，机制及限制见[S40报告](npunlock-probe/research/results/amp-standard/REPORT.md)。

S41直接运行本机Transformers5.18.0的原版LlamaForCausalLM、GPT2LMHeadModel、DynamicCache和GenerationMixin，采用小型Config随机初始化并以独立CPU模型为oracle。没有复制/改写模型源码或下载权重。通用序列算子和布局修正为项目实现，复用既有SHAVE整数算术、比较及DMA；共享Parameter转换使用已安装PyTorch Module._apply公开支持的future swap选项，finally恢复全局设置。核心依赖仍为torch/numpy，HF仅使用现有可选依赖。安装源文件指纹、机制和有限模型范围见[S41报告](npunlock-probe/research/results/attention-migration/REPORT.md)；PyTorch/Transformers来源和许可沿用已有条目。

S42依据[FlashAttention论文v2](https://arxiv.org/abs/2205.14135v2)的在线softmax分块与反向重计算方法，自行实现NPU40XX SHAVE C/packet调度，复用本项目IEEE/exp和DMA helper，没有复制CUDA实现或引入flash-attn/OpenVINO运行依赖。Nested接口依据[PyTorch2.14 Nested文档](https://docs.pytorch.org/docs/2.14/nested.html)、本机2.14.1的ATen schema及NestedTensorImpl头文件核对；CPU metadata getter/view只读取布局或包装storage，AutogradNestedTensor SDPA对其他设备保留复合分派。CPU Nested前后向及独立FP64/C oracle用于验证，来源、指纹、实际范围和性能限制见[S42报告](npunlock-probe/research/results/flash-nested/REPORT.md)。对应论文BibTeX新增dao2022flashattention；许可声明沿用PyTorch来源及项目既有helper，未打包新的第三方源码。
