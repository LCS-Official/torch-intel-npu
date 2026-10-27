# AdamW设备更新与模型训练验收

日期：2026-10-18。本地开发版，未推送GitHub；此前报告保持为历史记录。

## 结论与明确限制

标准torch.optim.AdamW普通单Tensor路径已跑通；FP16/FP32独立更新、一二阶矩、AMSGrad、maximize、zero/None梯度与CPU checkpoint恢复通过。
三个FP32分类模型在关闭紧邻BatchNorm的冗余卷积bias后，完成四步独立CPU轨迹、显式CPU checkpoint恢复和eval检查。
纯FP16整网仍有严格数值检查失败，不能把独立更新通过推广到完整FP16模型训练兼容。
原含冗余bias的FP32卷积/残差网络也有参数轨迹失败；原测试及失败位置保留在下表，没有放宽容差或计作通过。

## 本轮实现与执行位置

- 新增lerp_.Scalar、addcmul_、addcdiv_、浮点标量add_、sqrt/maximum前向与受限out。
- 三个更新算子复用既有SHAVE算术，以FP32 opmath计算并在写回时转为原dtype；lerp采用稳定分支，
  系数先按FP32舍入再选择分支，覆盖0.5附近与±最大有限值的极端输入；
  addcmul/addcdiv保持value*a*b及value*a/b的顺序，没有替换为低精度乘法。
- sqrt新增SHAVE入口，直接复用resident_activation.c已有的整数开方；maximum新增独立IEEE位排序，保留subnormal比较与NaN传播。
- 参数、梯度消费和moments更新在NPU；标准Torch的step在CPU，偏差修正用Python数值，再作为标量常量参与NPU计算。
  这些是原优化器控制流，不是CPU参数/动量算术回退；不能宣传为全部控制逻辑在NPU或完全零拷贝。
- PyTorch把标量add_分派为Tensor重载时，允许无梯度CPU零维实数常量；没有引入CPU大Tensor混合算术。
- 没有改写torch.optim.AdamW、torch.cuda、用户模型；C++扩展、session DLL及既有ELF不变。
- 新增两个源码/ELF/汇编/manifest伙伴及构建入口；无新增依赖、模型下载、图片或持久图缓存。

支持边界：no_grad更新、FP16/FP32、完整密集排列、匹配shape/dtype/device，Python数值超参数；
foreach=False、fused=False、capturable=False、differentiable=False。不认证Adam、稀疏梯度、Tensor超参数、AMP或高阶autograd。
拒绝广播、部分存储、不同stride的重叠别名；精确self-alias先读到独立结果再写回。sqrt/maximum未实现autograd，NaN payload和signed-zero ties不作CPU逐位认证。

## CPU参照与测试口径

Windows x64 / Core Ultra 7 258V / NPU 8086:643e / API driver_version=1005540 / 编译器8.3。
Python 3.12.14 / Torch 2.14.1+cpu；现有.conda/npunlock，CPU1线程。更新测试种子61；整网种子53、关闭oneDNN。
普通assert_close仍为FP16 rtol=0.003/atol=0.0005，FP32 rtol=0.00003/atol=0.000003；没有为AdamW放宽。
实际使用的本地torch/optim/adam.py SHA-256：`e333d196bd24f6a44146839139ad0222bfaab0f71dcdb341585a4e3f9f4c2d87`。

修复前边界重放：AdamW在参数weight decay后遇到aten.lerp.Scalar_out缺口；
修复前torch_intel_npu.py SHA-256：`ef3666feb3dfe3ba426e031477d47c1d6b49c47e7cb604474ade5f99e09aeb82`。
不支持的优化器组合可能已部分更新参数/状态，异常后应从原始checkpoint重建。

## 更新算子与优化器状态

| 检查 | 数值比较/位模式 | NPU调用 |
|---|---|---:|
| primitives-torch.float16 | 78项pointwise；65,536种sqrt输入 | 817 |
| primitives-torch.float32 | 78项pointwise；8,192种sqrt输入 | 462 |

sqrt有限输出与FP64独立参考逐位一致，含全部65,536个half编码、8,192个FP32位模式及特殊值；另检查65,537元素分块。
本机Torch FP32 sqrt在三个样本上与独立FP64/math.sqrt参照相差1 ULP；NPU与独立参考一致，未为迎合CPU近似结果修改开方。
12组配置为两种dtype × 默认betas/no-decay/自定义betas/maximize/AMSGrad/组合；使用完整密集转置参数。
单参数配置lr=0.002；默认betas=(0.9,0.999)，自定义=(0.1,0.95)，weight_decay为默认0.01、0或组合0.1。
每组4次实际更新、中间None梯度不提交图且step不递增、零梯度正常递增，最后一次从CPU checkpoint恢复。
逐项对照参数/exp_avg/exp_avg_sq/可选max_exp_avg_sq/CPU step，检查布局及从未获得梯度的参数不分配state。
更新区间的显式host copies等于首次上传的标量常量数；逐步变化的偏差修正常量可继续上传，不能声称所有热步没有host copies。
19种非法输入/别名/grad模式调用在无新增图提交时拒绝；saved-tensor version guard通过。

FP16 epsilon边界单列：默认eps=1e-8在half中成为零，零梯度时CPU/NPU均产生NaN。该项只证明行为一致，不算健康训练通过。
普通FP16检查显式使用eps=1e-4，仍保持half参数/half moments，没有自动提高epsilon或使用FP32 master state。

## 整网训练与失败诊断

沿用MLP(5→8→3)、Conv+默认BN和三层卷积残差网络，batch4、CrossEntropyLoss。AdamW lr=0.001；
FP32 eps=1e-8，FP16 eps=1e-4；其余默认betas/weight_decay，foreach/fused关闭。三步训练后CPU checkpoint恢复，继续第四步。
CPU模型保持独立轨迹；没有重设CPU参数追随NPU。另设独立诊断shadow：把NPU更新前参数/moments复制到CPU，
以同一份NPU梯度调用标准CPU AdamW，严格检查本次参数与moments更新；它不替代独立模型轨迹。验证复制均在hot-copy统计区间之外。
关闭BN前bias的对照保持全部其他初始权重和RNG消耗不变；只在测试模型中显式去掉紧邻BN的bias，后端不自动改模型。

| BN前卷积bias | 模型/精度 | 独立轨迹状态 | 四步loss或首个失败位置 |
|---|---|---|---|
| True | mlp-torch.float16 | failed | mlp torch.float16, step=4, parameter=2.weight; identical-gradient AdamW update checked=True; close(p, q); max_abs=0.001220703125 |
| True | mlp-torch.float32 | passed | 1.0659971 → 1.0644666 → 1.0629447 → 1.0614313 |
| True | conv-torch.float16 | failed | conv torch.float16, step=2, parameter=0.bias; identical-gradient AdamW update checked=True; close(p, q); max_abs=0.00164794921875 |
| True | conv-torch.float32 | failed | conv torch.float32, step=1, parameter=0.bias; identical-gradient AdamW update checked=True; close(p, q); max_abs=0.0015741586685180664 |
| True | residual-torch.float16 | passed | 1.0234375 → 1.0195312 → 1.0175781 → 1.015625 |
| True | residual-torch.float32 | failed | residual torch.float32, step=1, parameter=stem.0.bias; identical-gradient AdamW update checked=True; close(p, q); max_abs=0.00016498565673828125 |
| False | mlp-torch.float16 | failed | mlp torch.float16, step=4, parameter=2.weight; identical-gradient AdamW update checked=True; close(p, q); max_abs=0.001220703125 |
| False | mlp-torch.float32 | passed | 1.0659971 → 1.0644666 → 1.0629447 → 1.0614313 |
| False | conv-torch.float16 | failed | conv torch.float16, step=2, parameter=1.bias; identical-gradient AdamW update checked=True; close(p, q); max_abs=0.0006256103515625 |
| False | conv-torch.float32 | passed | 1.3939296 → 1.3892826 → 1.3845434 → 1.3800585 |
| False | residual-torch.float16 | failed | residual torch.float16, step=1, parameter=branch.3.weight; identical-gradient AdamW update checked=True; close(p, q); max_abs=0.053955078125 |
| False | residual-torch.float32 | passed | 1.2597448 → 1.2532594 → 1.2468585 → 1.2404491 |

含bias的FP32 Conv与残差网络在第1步先偏离BN前卷积bias；相同梯度AdamW更新检查通过。
理论上此bias被BN抵消；微小反向归约残差经过小epsilon的AdamW归一化会被放大，这是与SGD不同的敏感处。
关闭冗余bias后三个FP32网络通过，支持这一判断；不能据此声称浮点反向逐位等同CPU或忽略任意微小梯度。
纯FP16参数/梯度量化与二阶矩下溢仍会造成明显轨迹偏差；本轮不保证纯FP16整网质量或长期收敛，优先使用FP32训练。

## 最终计数与资源

- 更新算子/状态/epsilon边界/拒绝契约16组通过；SGD分类11组通过；迁移29组功能通过、7组明确blocked；既有设备33/训练11/兼容性5组通过。
- AdamW两种模型配置共12组：5组严格通过，7组独立轨迹失败，失败组不计作通过。
- 上述最终验收与显式失败诊断共42,035次真实NPU调用，110组检查passed，另7组blocked、7组failed。
  不含开发期重复实验或修复前边界重放；passed中包含非有限epsilon行为一致性检查，不能全部解释为模型训练通过。
- 4项classification载体、3项INT64载体、5项源码导出离线回归，共12项通过。
- 49组既有回归未调用lerp；最后系数边界修复后重跑所有AdamW路径及分类/迁移套件，不重复无关旧算子回归。
- 全部进程无context failure，各检查live buffer无增量，shutdown后live/cache/scalar/quarantine/session归零。

## 使用、复现与后续

```python
import torch
import torch_intel_npu
model = model.float().to("intel_npu")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
    foreach=False, fused=False, capturable=False, differentiable=False)
```

卷积后紧接BatchNorm时，在模型定义中显式使用Conv2d(...,bias=False)。checkpoint仍先把参数/状态转到CPU，
不使用原始NPU Tensor序列化/deepcopy。Adam本体、LayerNorm/GELU/softmax/Embedding、bmm/SDPA、AMP和GPU+NPU协同未在本轮补齐。

```powershell
foreach ($case in @('primitives','optimizer','contracts')) {
  .\run_torch_npu.ps1 -Script .\torch_npu_adamw_demo.py -ScriptArguments @('--case',$case)
}
.\run_torch_npu.ps1 -Script .\torch_npu_classification_demo.py -ScriptArguments @('--case','training','--optimizer','adamw')
.\run_torch_npu.ps1 -Script .\torch_npu_classification_demo.py -ScriptArguments @('--case','training','--optimizer','adamw','--no-bn-bias')
```

两条AdamW整网命令保留严格失败，当前会返回非零；分别保存training.json与training-no-bn-bias.json。
完整审计还需重跑classification四个SGD用例、migration四个进程、Device/Training Quick及compatibility；
生成上述JSON后执行本脚本；修复前边界基线可选，提供时额外审计。旧基线需原始后端版本，不能用新版本生成blocked基线。
按用户约定审计后清理原始JSON/日志/编译响应，保留源码、内核与Markdown；清理后需重跑生成审计输入。

实现语义参考[PyTorch Lerp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Lerp.h)、
[CPU PointwiseOps](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp)。
[PyTorch性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide#disable-bias-for-convolutions-directly-followed-by-a-batch-norm)
也建议关闭紧邻BatchNorm的卷积bias；这里没有声称把这种使用建议自动写入后端。来源关系详见REFERENCES.md。

## 当前SHA-256

```text
torch_intel_npu.py
e7d65e84d2dd0ddde376a5d695d145444e044b5e7d017781bbada798c38e8253
torch_npu_autograd.py
9e8ec4aaf438728baa733e11fb52ef142e4aa7d7fe2fd832ddd66bfd650a1285
torch_npu_adamw_demo.py
4389d8c3518726152d753222068ee1853a76a3ce0ab4276aaebe25aaf8621543
npunlock/tests/fixtures/npu4000/resident_optimizer.c
7855328ade6a1144c319da94ff184c5046532fde5fe1cf58485a569cc10b4af4
npunlock/tests/fixtures/npu4000/resident/manifest.json
c150f8c05ab971ce0e5e8e9f48abaa24b6bfe6ceb0b01f132a447f715b160f6f
npunlock-probe/torch-device/_torch_intel_npu.pyd
b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e
torch_npu_classification_demo.py
d218125cac1048d9522a53d11bfc73e4f66628d44a4029f2eec49b6dc8952056
npunlock/python/npunlock/npu4000.py
96fd710b05d130b8290ce17d5bf76f1ec0ad3ecdc4176b1b61f18ae4e539e921
npunlock/tests/fixtures/npu4000/resident_classification.c
fcfb57bae8381b1875e11dec36505fdefd6fda3a858a5fbc7aca1c45ddfd607c
torch_npu_migration_demo.py
0f72a109fe35e40f314bbe75e0f7080f8d29f5ee6c598b4351775a8be2798ae1
npunlock-probe/torch-device/backend.cpp
006cad067bf8c16782be59b58b9b5d1dfd34244c99329a38854273dd39e6a924
npunlock-probe/research/build_optimizer_kernels.py
a87cd13c0cb391453253afbe31b68db46bc096cb5042cb957c219d6e2e766527
npunlock-probe/research/build_resident_kernels.py
273503ce89dfc0af8cdaa2bb77a0133aaf250d9504543c493c1f80805e77ce2e
npunlock-probe/session-runtime/npunlock_session.dll
0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9
npunlock/tests/fixtures/npu4000/resident/sqrt-f32.elf
5a3d4076250207ad1f32b75d605380f4fe77881b8769c0529f0662d788415827
npunlock/tests/fixtures/npu4000/resident/maximum-f32.elf
2269243696ef9d7d7578b09b90447e96656c46cbbedafa19dfb7c49d400d439b
```

## 验收后清理

已清理45份原始JSON/日志/编译响应，共183,389字节（约179.09KiB），包括严格失败的原始轨迹数据。
删除前逐项验证绝对路径位于工作区内且无reparse ancestor；保留源码、构建/测试脚本、ELF/汇编/manifest及Markdown报告。
清理前后核对两个新增ELF、manifest、PYD、session DLL及本轮/三个历史报告的哈希不变；随后仅为本报告附加此清理记录。
没有新增依赖、模型、图片或持久图缓存；公开source-release快照未改动，未推送GitHub。
