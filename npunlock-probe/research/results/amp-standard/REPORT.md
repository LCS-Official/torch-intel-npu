# S40：标准GradScaler入口与AMP接口兼容

日期：2026-10-25。本地开发版，尚未推送GitHub或修改公开source-release。平台为Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`、驱动32.0.100.5540、图编译器8.3、Torch2.14.1+cpu。

## 结果与实现机制

导入torch_intel_npu后，可直接使用torch.amp.GradScaler("intel_npu")、torch.GradScaler("intel_npu")及从torch.amp.grad_scaler导入的同类别名，包含后端导入前取得的别名。旧torch.intel_npu.amp.GradScaler继续可用。

本机PyTorch标准类的unscale_会对scale调用FP64倒数，scale/update还包含non_blocking复制；项目已有专用scaler以CPU控制元数据配合NPU梯度计算。当前已安装实现没有供本项目使用的自定义设备scaler工厂，因此本轮在后端导入时对原类的__new__安装仅针对intel_npu的构造分派，返回项目GradScaler子类。原torch.amp.GradScaler类对象、提前导入别名及isinstance关系保留；CPU/CUDA/XPU和其他类的构造委托原分支。安装幂等，并检查安装期间构造器是否被其他扩展修改。

这是一项本项目的后端适配，不表示上游PyTorch已内建此后端。没有修改site-packages文件、增加FP64设备算术、安装新依赖或重建DLL/PYD/ELF。FP32 scale、INT32增长计数、FP64倒数仍为CPU控制元数据；模型、梯度有限性检查和反缩放在NPU。常规unscale每优化器只回读一个BOOL标志，保持原状态机、Inf/NaN跳步、增长/回退、累积、多优化器和state_dict语义。

### 共同接口修正

- scale支持嵌套list/tuple、惰性iterable及空iterable；空/未消费iterable不初始化scale。无效标量输入明确拒绝。
- scale的结果dtype对齐本机CPU GradScaler：Half标量与FP32 scale相乘得到FP32，Half向量保留Half结果；FP32 opmath后再舍入，避免将65536提前转换为Half Inf而污染零值。此提升语义也适用于旧专用入口；旧版非标量Half输出原先统一提升至FP32。
- update可接受无梯度、单元素intel_npu:0 FP32 Tensor作为new_scale。回读该控制标量、检查正有限FP32值后复制到内部CPU控制状态；随后修改输入Tensor不会改变scaler状态。无效元数据在提交前拒绝，无效数值保持scaler状态不变。update必须已初始化scale。
- 原FP32主参数、完整密集FP32梯度、无别名及普通优化器预检保留。foreach=False、fused=False仍须显式指定，capturable/differentiable/稀疏等路径仍拒绝。

## 正式真机检查

共13个独立进程，全部passed；**7,763次NPU提交**。数量包含准备、参考读回及验证操作，不是性能基准。

| 入口 / case | scaler入口 | NPU提交 |
|---|---|---:|
| amp_standard / api | 标准、提前导入别名 | 12 |
| amp_standard / outputs | 标准 | 63 |
| amp_standard / updates | 标准 | 10 |
| amp_standard / adamw | 标准 | 309 |
| amp_demo / scaler | 标准 | 350 |
| amp_demo / contracts | 标准 | 32 |
| amp_demo / mlp | 标准 | 1,376 |
| amp_conv_demo / small | 标准 | 1,259 |
| amp_conv_demo / residual | 标准 | 2,338 |
| amp_demo / scaler | 旧专用 | 350 |
| amp_demo / contracts | 旧专用 | 32 |
| amp_demo / mlp | 旧专用 | 1,376 |
| amp_demo / policy | 旧专用/autocast | 256 |

api覆盖后端导入前的torch.amp、torch.amp.grad_scaler、torch顶层别名；检查原类对象不替换、实例仍为其子类、CPU控制dtype/device、state_dict及scaler自身pickle恢复。CPU GradScaler实例保留原类类型，CPU禁用workflow执行一步SGD正确；XPU只验证禁用构造，没有跑GPU硬件AMP。

outputs覆盖12种scale/dtype/shape组合：8与65536、FP16与FP32、标量/四元素向量/空Tensor；结果dtype、所有值、正负零符号及一阶梯度均与CPU GradScaler精确一致。另验证嵌套list/tuple、惰性generator和空iterable。计算区没有输出/梯度payload下载。

updates覆盖标量及单元素设备Tensor、新scale无输入别名、四项元数据拒绝、零/负/Inf/NaN四项数值拒绝、Python数值及初始化前拒绝。有效Tensor更新回读一个控制标量；元数据/阶段拒绝不新增NPU提交，拒绝保持状态不变。

SGD和AMSGrad AdamW（含weight decay）的有限/Inf/NaN七步控制历史与独立CPU scaler相同：8→16→8→8→4→4→8。参数及全部AdamW状态逐项对照通过，溢出不推进优化器状态。另通过两微批累积、多优化器标志合并、默认65536在真实Half backward中溢出后的跳步/回退，以及重复unscale/step、closure、别名梯度和非法优化器等拒绝检查。

### 模型与状态恢复

| 模型 | 步数 | loss首→末 | 参考与恢复 |
|---|---:|---|---|
| MLP + LayerNorm + SiLU | 3 | 1.3628246784→1.2819437981 | 独立CPU混合精度严格对照；模型、SGD动量与scaler下一步逐位恢复 |
| SiLU卷积网络 | 4 | 1.2885997295→1.2649385929 | 独立FP32-opmath严格对照；checkpoint逐位恢复 |
| 三卷积残差网络 | 4 | 1.1751620770→1.1649489403 | 独立FP32-opmath严格对照；checkpoint逐位恢复 |

MLP标准/旧专用入口的loss与scale轨迹相同，scale为128→256→256。卷积两模型的FP32-opmath strict_failures均为空；原生CPU Half梯度严格差异各保留4项，沿用S18的分batch舍入诊断，没有修改阈值、替换真实计算或将这些差异宣称为已解决。四步通过不认证长期收敛或任意模型AMP。

checkpoint中的模型/优化器Tensor仍显式转CPU，scaler本身是控制状态；本轮没有实现原始NPU Tensor直接序列化。

所有进程退出前shutdown均closed=True、context_failed=False，live_buffers、cached_buffer_blocks、cached_scalar_buffers、cached_sessions、quarantined_buffer_bytes均为0。

## 离线检查

根目录143项、npunlock Python API 32项、源码发布工具5项，**180项全部通过**。新增构造别名/CPU行为与pickle、空嵌套iterable、Half/FP32原生dtype提升检查，沿用原test_torch_npu_amp模块并自动包含在既有CI命令中。GitHub托管CI未在本轮执行。

64组ELF/汇编及26份C源逐一匹配manifest；C++/PYD/session DLL继续匹配原build.json，无原生构建或内核改动。

## 用法与复现

```python
import torch
import torch_intel_npu

model = torch.nn.Linear(4, 3).to("intel_npu")
optimizer = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
scaler = torch.amp.GradScaler("intel_npu")

# x/labels already reside on intel_npu; keep model master parameters in FP32.
optimizer.zero_grad(set_to_none=True)
with torch.autocast("intel_npu", dtype=torch.float16):
    loss = torch.nn.functional.cross_entropy(model(x), labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

已有S38/S39环境同步Python源码并重启Python会话即可。串行复现：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NPUNLOCK_GRAPH_CACHE = ''
foreach ($case in @('api','outputs','updates','adamw')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_amp_standard_demo.py -ScriptArguments @('--case',$case)
}
foreach ($case in @('scaler','contracts','mlp')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_amp_demo.py -ScriptArguments @('--case',$case,'--scaler-api','standard','--label','s40-standard')
}
foreach ($case in @('small','residual')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_amp_conv_demo.py -ScriptArguments @('--case',$case,'--scaler-api','standard')
}
```

旧专用入口回归使用amp_demo的--scaler-api legacy及--label s40-legacy，case为scaler/contracts/mlp/policy。离线完整命令见.github/workflows/offline.yml。

## 剩余边界

- 单NPU、FP32 master/完整密集FP32梯度、显式foreach=False/fused=False；其他优化器路径、稀疏及别名梯度仍缺。
- 构造分派针对当前PyTorch类接口，保留原类对象并返回后端子类；其他Torch版本、第三方严格type判断、自定义GradScaler子类或构造器扩展共存未全面认证。
- BF16 AMP、任意模型/长期FP16收敛、CPU Half卷积严格轨迹仍不完整；未新增FP64设备计算或GPU+NPU协同。
- 原始NPU Tensor序列化/deepcopy、通用torch.compile与异步stream/event继续是后续工作。
- 没有做性能、功耗、其他驱动/机器或实际CUDA/XPU硬件AMP验证。

## 源与组件指纹（SHA-256）

```text
torch_intel_npu.py
f63b59a00dd079f81a39979b8f201ca94aa21183ac7af80167e99cffd2576f09
torch_npu_amp.py
6537da6a7c9f5a428d30ca6d68b4796e1379dec853f0ed5892dc359a7648abde
torch_npu_amp_standard_demo.py
4a200748e83ecdc114c4123421a656a71f8e8e48221297e92248954e3260d30e
torch_npu_amp_demo.py
68fcf0e43d1e6cfe2f6a74595cf3556aea162f519824d9ad39f9f6c88956cbfe
torch_npu_amp_conv_demo.py
9d6df1dc3868bfec94f515a11c33b79c88412ef08b88e9c0cfb8217308d28193
test_torch_npu_amp.py
af887e224824c429e011ede51cef89e10654044a4c3758161182cf925a6f0947
torch_npu_autograd.py
454dc711240e48c121aaba84320ecf2bd39ccf64f897d21fde189ce709046115
npunlock-probe/torch-device/backend.cpp
9c3035281a9e1f6547c2d145a63cda495c0b2478ba2691e999d9548154970fba
npunlock-probe/torch-device/_torch_intel_npu.pyd
433757698bbc704a44da3ca9c3c3355d3fac49ab19b82d06f0d09790aa4c6ea0
npunlock-probe/session-runtime/npunlock_session.dll
1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90
npunlock/tests/fixtures/npu4000/resident/manifest.json
d7b35afdcd6270c435f1b644af2c3fec5d3e0e9a16cb44c8860a3316356eb6bb
.conda/npunlock/Lib/site-packages/torch/amp/grad_scaler.py
d13fbf446af0d6cd343ee0b4521cd152034a645e5236c2378e45122590405756
```

## 清理与交付

所有测试进程退出后，删除本轮13份正式真机JSON，共57,678字节，保留实现、测试、复现入口及本报告。删除前后831份受保护源码/内核/运行组件/报告及PyTorch grad_scaler.py指纹一致，随后补充本节与空间台账。无新增依赖、环境、模型、图表、磁盘checkpoint或构建临时文件。source-release未改，未推送GitHub。
