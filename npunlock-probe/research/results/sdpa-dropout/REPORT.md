# SDPA内部Dropout与一阶训练迁移

日期：2026-10-19。本地开发版，未推送GitHub，发布快照source-release未修改。
平台：Windows x64 / Core Ultra 7 258V / NPU 8086:643e / 驱动32.0.100.5540 / 编译器8.3 / Torch 2.14.1+cpu。
本轮仅修改Python设备派发和一阶autograd，复用既有SHAVE随机数、BOOL、softmax和矩乘内核；没有安装依赖、下载模型、重建PYD或修改系统驱动。

最终相同源码指纹的7个独立设备进程全部通过，共2,801次NPU调用；两种精度各12个算子子场景通过。另有5组既有回归、1,129次NPU调用和55项离线检查通过。

## 目标和支持范围

把上一轮已验证的设备RNG/普通Dropout接到标准F.scaled_dot_product_attention，允许dropout_p在[0,1]，保留原有rank、精度、广播和mask边界。

| 项目 | 本轮范围 |
|---|---|
| Q/K/V | 同rank 3D/4D、同dtype FP16/FP32、原有batch/head广播、完整密集布局；Q/K最后维度相同、K/V长度匹配 |
| Dropout | 在softmax之后的注意力概率上使用既有SHAVE Philox生成的BOOL mask；概率必须在[0,1]，不是只对最终输出作Dropout |
| 中间精度 | Q/K预缩放、scores、softmax、Dropout及与V矩乘都在FP32中进行，最终输出转回Q的dtype |
| 一阶梯度 | Q/K/V及浮点attn_mask；包含广播归约、仅Q/K/V之一求梯度及仅浮点mask求梯度；二阶仍不支持 |
| 注意力mask | 同设备BOOL True表示保留；浮点mask为Q dtype或FP32，可广播；因果模式和显式mask仍不能同时设置 |
| 全屏蔽行 | 沿用safe-softmax的零概率语义，有限输入下输出和该行Q梯度为0 |
| RNG | 对0<p<1按完整广播后的score数量推进counter；backward复用保存mask，不抽取新随机数；p=0/1不推进counter |
| 容量 | score总元素<=65,536，各广播Q/K/V及输出<=1,048,576；没有扩大设备存储或工作区 |

SDPA没有独立training参数：即使处于no_grad或外层模块eval，传入非零概率仍按该概率执行。模块需自行按training状态选择概率，例如：

```python
output = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=mask,
    dropout_p=self.dropout_p if self.training else 0.0)
```

仍不提供GQA、fused/FlashAttention内核、通用AMP、显式Generator、任意部分Q/K/V view、异步stream/event或CUDA运行时兼容；没有因此认证标准nn.MultiheadAttention或完整Transformer。
继承既有非有限值行为，本轮新增数值覆盖主要使用有限Q/K/V和包含-inf的mask，不认证所有NaN/Inf/极端值组合。

## 实现及反向公式

令P=safe_softmax(S)，D是keep/(1-p)，A=P*D，O=A@V。p=1的有限数据采用全假mask与0缩放，避免除零；p=0直接复用P，不创建随机mask或增加Dropout图提交。
forward保存P、A及BOOL keep mask。backward中：

- dA=dO@V.T，dP=dA*D，再用原始P计算softmax反向dS。
- dV=A.T@dO，使用的是Dropout后的A。
- Q/K导数沿用预缩放矩乘的原有链式法则，浮点mask的导数由dS作广播归约。

保存原始概率和Dropout后概率是必要的；把二者混用会使Q/K或V梯度错误。保存的随机mask受PyTorch version保护，修改后反向明确报错。
随机数据及数值运算仍在NPU执行；主机只上传已有RNG的seed/counter控制header和字面常量。无“CPU生成随机Tensor再冒充NPU执行”的运行路径。
PrivateUse1公开入口仍为intel_npu，与torch.xpu分离。

## 独立参考与测试方法

语义参考[PyTorch 2.14 SDPA接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)与[v2.14.1数学实现](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/transformers/attention.cpp)，ATen签名以本机2.14.1为准。
CPU参考单独用PyTorch矩乘、softmax、mask和autograd计算；测试用Python大整数Philox根据seed/counter独立预测同一BOOL mask。测试辅助函数不在设备运行时调用，不下载NPU mask来建立期待值。
FP16参考同样保留FP32中间运算。比较只要求相同mask时的数学结果，不要求CPU/CUDA/NPU相同seed生成相同随机流。

先进行3项离线参考校验：

1. p=0的BOOL/浮点/因果mask输出及Q/K/V梯度，对照官方CPU MATH SDPA。
2. p=.25的显式相同mask输出及Q/K/V梯度，对照官方仅供测试的_scaled_dot_product_attention_math(dropout_mask=...)入口。
3. p=1的有限输出对照官方MATH SDPA，检查零输出/梯度及参考缺少mask时拒绝。

PyTorch提示dropout_mask仅用于测试是该CPU对照入口的预期警告；NPU执行没有使用这个CPU测试入口。
新增设备检查覆盖FP16/FP32、rank3、rank4广播和密集转置、负scale、BOOL全屏蔽行、同dtype/FP32浮点mask梯度、矩形因果mask、p=0/1、单独Q/K/V/mask梯度，以及8,193个score跨随机tile边界。
同时检查forward/backward计算区间无host_download_copies增量、CPU RNG不变、设备counter精确推进，数值对照及loss日志的显式下载位于计数区间外。

数值容差沿用原SDPA验收：FP32 rtol=3e-5/atol=3e-6，FP16 rtol=.003/atol=.0005，没有通过放宽旧容差使新功能过关。
RNG重放的输出及Q/K/V梯度要求逐位一致。非法概率不提交/不推进状态；保存mask改写后的反向也检查状态不变。
初版保护测试把上游sum的合法梯度准备计入“报错后无提交”，导致检查失败；改为直接给SDPA输出传入已准备好的梯度后复核，不改变实现或绕过version检查。

## 正式硬件结果

| 进程 | NPU调用 | 主要结果 |
|---|---:|---|
| smoke FP32 | 56 | 标准SDPA非零Dropout前后向，对照独立预测的相同mask |
| operators FP16 | 781 | 12个子场景，包含广播/转置/负scale、BOOL全屏蔽行、两种浮点mask、因果矩形、p=0/1、仅Q/K/V/mask之一求梯度 |
| operators FP32 | 641 | 相同12个子场景，沿用原有FP32容差 |
| state FP32 | 177 | 重放后输出及Q/K/V梯度逐位相同；backward不抽样，no_grad仍应用非零Dropout，p=0/1不推进RNG |
| tiled FP32 | 65 | 8,193个score跨随机tile边界，输出及Q/K/V梯度通过独立CPU参考 |
| contracts FP32 | 29 | 4个非法概率拒绝且无提交/状态变化；保存mask的version保护，直接输出反向报错前无新增提交/随机状态变化 |
| training FP32 | 1,052 | 95参数双头attention小网络，四步独立相同mask CPU梯度/更新、序列化checkpoint重放、eval切换 |

所有7组live_buffer_delta=0；shutdown后live_buffers、cached_buffer_blocks、cached_scalar_buffers、quarantined_buffer_bytes及cached_sessions为0，context_failed=False、closed=True。
上述调用包含梯度准备、数据整理和数值图，不等于单个算子调用数量，也不是吞吐/峰值算力指标。

训练模型为Q/K/V Linear(4,4)、两头attention、Linear(4,4)输出投影、残差、均值和Linear(4,3)头，使用BOOL因果mask、训练p=.25。
固定2个合成样本，每个3个token、4维输入；FP32 MSE、SGD lr=.03、无动量、foreach=False/fused=False。四步loss：
`0.2885267138 → 0.2593458295 → 0.2333354503 → 0.2079869658`。
每步CPU独立预测相同Philox mask，检查所有参数的梯度和更新；不是拿CPU相同seed的不同mask作直接比较。
第2步后将显式CPU参数/SGD state_dict及CPU ByteTensor表示的NPU RNG状态用torch.save序列化到BytesIO，weights_only=True读回并恢复到NPU。
重放第3步后loss、全部参数及RNG状态逐位相同。该验证不涉及原始NPU Tensor直接序列化、跨进程恢复或torch.utils.checkpoint的激活重计算。
恢复后的模型与相同权重CPU模型切到eval并显式令p=0，输出对照通过且RNG状态不变。训练及反向/更新计数区间无host payload下载；日志/对照/checkpoint明确下载在区间外。

## 离线验证

既有回归5组/1,129次NPU调用全部通过：原SDPA浮点mask的FP16/FP32两组共744次调用，原contracts一组34次调用，BOOL SDPA的FP16/FP32两组为195/156次调用。
全部采用dropout_p=0，覆盖浮点mask及其梯度、非有限mask、全屏蔽行、只对某个Q/K/V求梯度、BOOL/部分/标量mask；原contracts保留24个拒绝场景，拒绝区间无图提交，并检查保存输入version保护。
这4个回归进程关闭后相同live/cache/quarantine/session资源计数均归零，context_failed=False。旧零Dropout路径没有新增RNG提交；这里的调用数仍不是速度基准。

55项通过：geometry/lowbit/mask/RNG及新增SDPA参考共18项，npunlock Python API 32项，source exporter 5项。新增3项参考检查的内容见上文。
CI、双语首页及兼容性矩阵已同步；既有SDPA contracts中“非零概率不支持”改为概率超出范围的拒绝检查。

## 复现

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\run_torch_npu.ps1 -Script .\torch_npu_sdpa_dropout_demo.py
.\run_torch_npu.ps1 -Script .\torch_npu_sdpa_demo.py -ScriptArguments @('--case','sdpa-masks')
.\run_torch_npu.ps1 -Script .\torch_npu_sdpa_demo.py -ScriptArguments @('--case','contracts')
.\run_torch_npu.ps1 -Script .\torch_npu_mask_demo.py -ScriptArguments @('--case','sdpa','--label','sdpa-dropout-regression')
```

离线检查与CI命令新增test_torch_npu_sdpa_dropout；源码导出选择会包含新测试/脚本及本报告，不写入已有发布快照。
本轮目标是算子与一阶训练兼容，不是吞吐/功耗优化；没有测试长序列完整模型质量、长时间收敛、其他硬件/驱动或加速比。

## 正式源码SHA-256

```text
torch_intel_npu.py
eaa9f71b56569c98fbb6abc1921ac2e22a263059db730fd26f4d49aaff61063e
torch_npu_autograd.py
259c568782a8749563bd2ce0ec59182e7af078309569c0e733deb6b7ddb62a3e
torch_npu_sdpa_dropout_demo.py
7bee07836ff9dd37d7a1ba30b1f1a1b838989f8090be18fdcff19a83b2e85d62
torch_npu_rng_demo.py
df29788418378afeb7a3ce386eedb8bcc6e7e31a37b9963252dd2fa6ca3d80bf
torch_npu_rng.py
44530064b98c910d889a4b0dc91c2426a6c24eb77e70368212070b73f8d1b5e7
resident/rng-packet.elf (沿用)
3d5c8e0019281896d4abfcf516ab592c0a0e3ce26cbc56fa2df8a8f5709d7b52
```

## 临时结果清理

2026-10-19 SDPA内部Dropout验收后清理：11份本轮原始JSON，共31031字节；清理前后707份保留文件SHA-256一致，随后补充文字报告与台账。保留设备派发/梯度源码、新验收脚本/参考测试及报告；无新内核二进制、模型、安装依赖或图表。
