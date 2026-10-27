# 标准MultiheadAttention与TransformerEncoderLayer迁移

日期：2026-10-19。本地开发版，未推送GitHub，source-release发布快照未改。
平台：Windows x64、Core Ultra 7 258V、NPU 8086:643e、驱动32.0.100.5540、编译器8.3、Torch 2.14.1+cpu。
没有安装依赖、下载模型或增加SHAVE内核。本轮重建了注册元数据视图操作的Torch PYD，session DLL和SHAVE ELF沿用原版。

## 正式真机验收

正式10组全部通过，共8,045次真实NPU图提交；早期probe及定位/复核运行不计入此数。
所有正式组的运行时源码、C++、PYD和验收脚本指纹一致，默认MHA fastpath均保持True。

| 正式组 | NPU调用数 | 结果 |
|---|---:|---|
| primitives FP16 | 228 | 通过 |
| primitives FP32 | 157 | 通过 |
| MHA FP16 | 1,082 | 通过 |
| MHA FP32 | 672 | 通过 |
| EncoderLayer FP16 | 1,141 | 通过 |
| EncoderLayer FP32 | 626 | 通过 |
| 默认推理 FP16 | 519 | 通过 |
| 默认推理 FP32 | 333 | 通过 |
| 两层训练/checkpoint FP32 | 3,101 | 通过 |
| 元数据边界/直接NPU构造 | 186 | 通过 |

两层网络有359个参数，三步loss为0.181302622、0.152479276、0.231312618；每步梯度和更新均与独立同mask CPU参考在下述容差内一致。
这些值用于验证有Dropout时的更新语义，不是收敛或质量结论。保存第1步后的CPU权重/优化器/NPU RNG状态，重放第2步时loss、全部状态权重及RNG逐位一致；恢复后默认eval也通过。

10个正式进程均正常shutdown，context_failed=False；live_buffers、cached_buffer_blocks、cached_scalar_buffers、quarantined_buffer_bytes和cached_sessions均为0。
每组扣除新增标量缓存后的live buffer增量也为0。计算/反向/更新计数区间无payload下载；输出/梯度核对、日志和显式CPU checkpoint的下载另计，最终host_download_copies因此可以非零。
调用数是图提交计数，不代表算子数或速度/算力指标。

## 问题与修复

最初五项标准模块探测全部被阻塞：MHA的need_weights=False/True训练、默认eval，以及EncoderLayer的训练和默认eval。
训练路径的Q/K/V来自共享投影存储中的切片；原入口把“只读输入必须完整”与“原生绑定必须完整”混在一起。
默认eval/no_grad进入两个尚未注册的原生API：_native_multi_head_attention和_transformer_encoder_layer_fwd。

修复按现有机制实施：

- add/sub/mul/div、Linear、bmm/baddbmm、SDPA允许有界非重叠部分/零stride只读输入，复用byte-DMA Slice/Gather/Transpose先整理成完整设备allocation。
- 原生buffer_for仍要求offset=0、完整连续allocation；未把部分地址直接交给驱动，未开放部分/重叠目的写入。
- 新增baddbmm及一阶导数，FP32中间运算、最终转回输入dtype；bias可广播，beta=0不读取bias数值，out须精确shape/dtype且不与输入别名。
- 标准MHA的need_weights=True反向还需要_reshape_alias。在backend.cpp注册PyTorch自带的元数据实现，并重建PYD及构建指纹；没有复制或改写PyTorch的视图实现。
- 两个默认原生推理入口分解为现有NPU算子。标准模块源码及全局torch.backends.mha开关保持原状，没有把这些入口退到CPU运算。

推理入口的分解不等于实现了硬件融合内核，也不承诺PyTorch CPU/GPU融合路径的速度。
原生MHA入口只承接有界的3D同存储self-attention、匹配参数dtype和mask；训练/交叉注意力由标准模块的普通路径执行。
native入口在grad启用且任一输入/参数需梯度时明确拒绝，避免把仅用于推理的实现误作可微融合算子。

## 数值语义与参考

模块接口依据[PyTorch MHA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html)、[EncoderLayer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html)及本机Torch 2.14.1源码。
mask的一个细节来自[PyTorch v2.14.1 native attention](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/transformers/attention.cpp)：融合入口把合并mask转BOOL，True表示排除；普通SDPA的BOOL True表示保留，两者不能混用。
NPU推理桥接保持native排除语义，包括非零浮点合并mask的BOOL解释。普通MHA浮点加性mask路径仍沿用标准PyTorch模块。
native MHA使用普通masked-softmax，全屏蔽样本可产生NaN；这与SDPA安全softmax的零输出不同。本轮有专门CPU对照，不把SDPA的全屏蔽行结论推广到所有模块分支。

CPU参考使用同一标准模块及相同初始参数。训练时仅在CPU测试作用域替换F.dropout和F.scaled_dot_product_attention，以独立Philox预测的同一mask作数学对照。
该作用域在NPU forward之前退出，回调还断言只接收CPU Tensor，设备运行时没有这些替换。离线检查确认回调在异常后恢复、CPU RNG不推进、MHA和EncoderLayer的全部Dropout位置按预期推进测试counter。
单层训练检查输出、返回权重、全部参数及输入梯度，NPU前后向计数区间不下载payload，CPU对照/显式日志在区间外。
普通数值比较沿用FP32 rtol=3e-5/atol=3e-6、FP16 rtol=.003/atol=.0005；相同状态的checkpoint重放要求逐位一致。

首次Encoder FP16检查的CPU参考反向遇到oneDNN报错：`DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2`。
独立CPU FP16算子隔离确认LayerNorm、Linear、ReLU反向通过，GELU反向重现同一错误，阻塞来自CPU参考的GELU路径。
因此仅在CPU模块对照的forward/backward作用域使用`torch.backends.mkldnn.flags(enabled=False)`，退出后恢复；NPU路径不在该作用域。
未关闭或修改`torch.backends.mha`，也未放宽误差阈值。这个CPU参考平台限制不计为NPU算子错误。

## 验收入口

| 组 | 检查范围 |
|---|---|
| primitives FP16/FP32 | 四种只读切片算术及梯度；部分baddbmm与广播bias、三组系数、beta=0忽略NaN、alpha=0保留NaN；_reshape_alias共享存储与version |
| mha FP16/FP32 | packed self-attention的SDPA/权重路径、非零Dropout、BOOL/浮点mask与padding合并；packed cross、独立K/V维度+sequence-first、unbatched；输出/权重/参数/输入梯度 |
| encoder FP16/FP32 | batch-first标准EncoderLayer，post-norm ReLU与pre-norm GELU，attention/残差/FFN Dropout，BOOL因果+padding mask，全部梯度 |
| inference FP16/FP32 | MHA默认native入口、合并mask、全padding的NaN对照；EncoderLayer native入口、非零浮点mask、inference_mode；全局快路径开关不变 |
| training FP32 | 两层标准EncoderLayer及Linear头、三步相同mask CPU更新；序列化CPU参数/优化器/NPU RNG状态后重放；默认eval |
| contracts | 元数据拒绝、部分out/重叠只读/OOB view、dtype/系数边界无提交；直接NPU构造EncoderLayer并与同权重CPU对照 |

标准模块用embedding=4、heads=2、batch=2、query length=3；cross attention的key/value length=4，独立K/V维度为3/2；另有sequence-first和unbatched MHA。
EncoderLayer的FFN宽度为8。训练Dropout概率为.25；两层网络用ModuleList串接两个标准EncoderLayer，再接Linear(4,3)，不是完整TransformerEncoder容器认证。

这不是完整模型质量或任意形状认证。仍继承4MiB单allocation、rank/元素界限、score<=65,536、图缓存容量等限制。
未认证NestedTensor、完整torch.nn.Transformer/TransformerEncoder容器的嵌套快路径、其他attention配置/静态KV、通用AMP、二阶梯度、LLM训练或长期FP16轨迹。
本轮没有做速度、峰值算力或功耗基准。

## 复现

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\run_torch_npu.ps1 -Script .\torch_npu_standard_modules_demo.py
# 单项示例
.\run_torch_npu.ps1 -Script .\torch_npu_standard_modules_demo.py -ScriptArguments @('--case','mha','--precision','float32')
.\run_torch_npu.ps1 -Device -Quick
.\run_torch_npu.ps1 -Script .\torch_npu_layout_demo.py -ScriptArguments @('--case','views','--precision','all','--label','standard-modules-regression')
.\run_torch_npu.ps1 -Script .\torch_npu_sdpa_dropout_demo.py -ScriptArguments @('--case','operators')
$env:PYTHONPATH="$PWD;$PWD\npunlock\python"
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_geometry test_torch_npu_lowbit test_torch_npu_mask_packet test_torch_npu_rng test_torch_npu_sdpa_dropout test_torch_npu_standard_modules
```

--case probe是早期五项诊断模式，不能把它的probe-complete当作完整验收passed。默认命令运行正式分组，并在失败时停止后续组。
本轮离线共58项通过：基础/低位/mask/RNG/SDPA及标准模块CPU控制21项、Python API 32项、source exporter 5项。

## 既有路径回归

| 回归 | 检查量 | NPU调用数 | 结果 |
|---|---:|---:|---|
| 基础设备Quick | 33 | 4,162 | 通过 |
| 只读视图 FP16/FP32/INT64 | 3 | 630 | 通过 |
| SDPA Dropout FP16 | 1 | 781 | 通过 |
| SDPA Dropout FP32 | 1 | 641 | 通过 |

回归共6,214次真实NPU调用。设备33项含存储/复制、dtype转换、归约、矩阵/转置、模块与多线程生命周期及关闭后保护。
视图三组覆盖8种只读布局、复制/out、别名；仅浮点组验证切片/选择、广播clone、split/unbind反向。SDPA每种精度12个场景，含广播、负scale、BOOL/浮点/FP32 mask及梯度、因果矩形、p=0/1、单独Q/K/V/mask求导和全屏蔽行。
这4个回归进程全部closed且context_failed=False，live/cache/quarantine/session相关资源归零；回归源码指纹与当前文件一致。
基础设备运行约134.91秒正常完成；日志120秒的faulthandler栈快照不是300秒外层超时，不影响33项passed结论。

## 验收源码SHA-256

```text
torch_intel_npu.py
18c097d0abe44df547b5536e447868e1a74d0d9e31a78360c6e81cfad84fc96c
torch_npu_autograd.py
a91ef74ba1a308e7b8c124da86d0f34d6b6341f09e86c404043e6db99abd294b
torch_npu_standard_modules_demo.py
d00b4227395d1ff4b560a5ed37f5e673fda1209a14f41fdc259545e3d83961d9
npunlock-probe/torch-device/backend.cpp
b720aa11083b841e03ebb8d80e74d0368ff7ffefdc9892a761122875b514558b
npunlock-probe/torch-device/_torch_intel_npu.pyd
641087286922730429eceb6e4b9e8ba58792e9a05958933ce17b87e90e6bc6a5
```

正式验收后仅将脚本注释中的CPU LayerNorm更正为GELU，修正前后Python AST相同，执行语句未变。最终注释版脚本SHA-256：`843e711083d977f5d6e0a8fcb4f6a8638df902119f8d3870e9c83f149a54322f`。

native构建记录中的session DLL SHA-256：`0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9`，本轮未重建；RNG/其他SHAVE ELF与上轮相同。

## 临时结果清理

2026-10-19 标准MHA/EncoderLayer验收后清理：16份本轮原始JSON/日志，共47871字节；清理前后714份保留文件SHA-256一致，随后补充文字报告与台账。保留设备派发/autograd、重建PYD及C++/构建记录、新验收与参考测试、双语文档和报告；没有新增SHAVE ELF、模型、依赖或图表。
