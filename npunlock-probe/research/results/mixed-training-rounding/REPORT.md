# 混合训练第三步偏差：Tensor add_ alpha修复

日期：2026-10-18。本地开发版；此前报告作为历史记录保留，公开source-release及GitHub未改动。

## 结果

- 原GELU(tanh)+显式FP32损失的第三步FP32 LayerNorm参数梯度失败已修复。
- none/tanh两组混合网络均通过四步独立CPU对照；原FP32梯度阈值rtol=3e-5/atol=3e-6不变。
- tanh样例四步的四层前向、上游梯度及所有参数梯度均逐位相同；更新权重仍有少量可接受的FP32舍入差异。
- FP16分类损失的两组严格失败仍然存在，不把它们计作支持；未认证通用AMP、长期收敛或任意Transformer。

## 原因与修改

网络为Linear(16,16)→LayerNorm(16)→GELU→Linear(16,3)，Linear/激活FP16，LayerNorm参数FP32。
固定seed127、batch4、CPU单线程且关闭oneDNN，SGD lr=.01/momentum=.8/foreach=False/fused=False。
保留none后tanh的原随机初始化/输入顺序，CPU/NPU各自连续更新，不将CPU轨迹重置为NPU权重。
修复前首次严格失败为step=2（第3步）的1.weight；逐层重放发现所有8步、4层在相同输入/参数/上游梯度下局部结果均相同。
该步FP32 LayerNorm weight梯度最大差由3.0517578125e-05降为0。
差异从首步SGD权重更新开始，第二步放大后改变第三步前向，最终跨过LayerNorm half累加边界。

原Tensor add_把alpha直接当FP32标量乘梯度；本机CPU先将alpha转换为Tensor dtype。
`-.01`转为FP16是`-0.01000213623046875`，不能用FP32的`-0.009999999776482582`替代。
修复在torch_intel_npu.py的_add_inplace：按输入dtype生成标量，再用设备转换/FP32乘加及最终dtype转换完成更新。
同时在提交前拒绝bool、超FP32范围和有限FP16 alpha超65504；仍要求no_grad、相同shape/dtype/device。
没有更改GELU/LayerNorm C源码、20份resident ELF/汇编、manifest、autograd、C++/DLL/PYD或torch.optim.SGD。
这项修改直接影响标准优化器所调用的设备更新算子，用户不需要改模型结构或手工重写更新公式。

## 可核对样本

| 场景 | CPU更新 | 原NPU更新 | 修复后NPU |
|---|---:|---:|---:|
| none首步0.weight[61]，旧值0.00015676021575927734，梯度0.1925048828125 | -0.0017690658569335938 | -0.0017681121826171875 | -0.0017690658569335938 |
| tanh首步0.weight[51]，旧值-0.01300811767578125，梯度0.09649658203125 | -0.01397705078125 | -0.01396942138671875 | -0.01397705078125 |

两条轨迹的初始输入+参数SHA-256：

```text
none 44663e71cfb8a16803eae45b8dd8ec2795fadfa5c50c66f391293764c85e9773
tanh d4886fce5cb84bd1105abe3f3fbbca626a1afdfea81aa230c8f3c9a5e51e2770
```

## 训练结果与保留边界

| 损失 | GELU | 修复后结果 |
|---|---|---|
| FP32 | none | passed；loss：1.10353124, 1.08670497, 1.0571959, 1.01870632 |
| FP32 | tanh | passed；loss：0.940976202, 0.910353839, 0.858752012, 0.795830905 |
| FP16 | none | failed；step=0, parameter=1.weight, fp32_loss=False: Tensor-likes are not close! |
| FP16 | tanh | failed；step=0, parameter=1.weight, fp32_loss=False: Tensor-likes are not close! |

FP16损失仍在第一步进入更新前失败，沿用[上一轮的log_softmax保存值舍入诊断](../layernorm-mixed/REPORT.md)。
显式FP32损失在CPU/NPU两侧同时使用，不是后端偷偷切换精度，也不是FP16损失已修复。
默认none配置第四步仍有少量half梯度差异，在原容差内；不能将本轮结果泛化为所有Tensor逐位一致。

## add_数值边界

- 原256元素参数向量的四个样本放在64元素向量中严格重放，通过；保留向量上下文，未拿CPU标量尾部作同一运算顺序。
- 65,536种half编码、与其滚动排列形成的另一操作数，10种alpha（含极小、极大、0/1、Inf/NaN），655,360个结果与CPU数值相等。
  NaN位置一致；这不是所有输入对的穷举，也不认证signed-zero或NaN payload逐位相同。
- 原版在-.01首项就有10,070/65,536数值不等；修复后10项均通过，未放宽这项相等判定。
- 密集转置、精确self-alias、version递增和保存Tensor版本保护通过。
- CPU小张量/标量尾部会额外把alpha*y中间量转回half；本NPU仍保留FP32中间量。
  明确的4元素边界有2个数值不等，最大差9.53674316406e-07，仅按原half容差通过。
  不为迎合CPU向量宽度而在NPU按32元素拆分算法；尚不承诺跨CPU线程/架构/形状的逐位轨迹相同。

## 回归与资源

| 检查 | 通过/失败 | NPU调用 |
|---|---:|---:|
| add_精度/契约 | 6/0 | 191 |
| 混合网络FP32损失 | 2/0 | 2136 |
| 混合网络FP16损失边界 | 0/2 | 384 |
| 既有更新接口 | 5/0 | 649 |
| SGD七种配置×两种dtype | 14/0 | 1202 |
| 残差前馈训练 | 6/0 | 6760 |
| AdamW状态/恢复/epsilon边界 | 13/0 | 2856 |
| 修复后逐层辅助诊断：两组×四步×四层 | 全部局部结果相同 | 2168 |

保留的最终修复后验收及辅助诊断共**16,346次NPU调用**，主检查**46组通过、2组已知严格失败**。
修复前辅助诊断2,194次另计；探索中的中间复测不并入最终验收统计。
逐组live buffer增量为零，所有进程shutdown后的live/cache/scalar/quarantine/session均归零，无context failure。
12项INT64/分类载体/源码导出离线检查通过。未重复完整设备/卷积/低位模型回归；本轮没有性能计时或长期训练。
修复会增加FP16标量到FP32的设备转换，不能据此宣称速度提升；没有启用持久图缓存、下载模型或安装依赖。

## 复现

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\add_alpha_regression.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\mixed_training_rounding_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\layernorm_mixed_acceptance.py -ScriptArguments @('--case','training','--fp32-loss')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\layernorm_mixed_acceptance.py -ScriptArguments @('--case','training')
.\run_torch_npu.ps1 -Script .\torch_npu_migration_demo.py -ScriptArguments @('--child','updates','--label','alpha-fix')
.\run_torch_npu.ps1 -Script .\torch_npu_migration_demo.py -ScriptArguments @('--child','sgd','--label','alpha-fix')
.\run_torch_npu.ps1 -Script .\torch_npu_norm_activation_demo.py -ScriptArguments @('--case','training')
.\run_torch_npu.ps1 -Script .\torch_npu_adamw_demo.py -ScriptArguments @('--case','optimizer')
```

FP16损失命令按既有严格失败返回非零。报告审计需要上述原始结果及修复前诊断；后者只能在历史后端上生成。
审计脚本通过精确撤销本次两个源码片段，在内存中核对旧后端指纹；不会写回旧代码或用当前结果伪造基线。
修复前诊断脚本有后续命令行/标量尾部检查调整，旧脚本指纹作为历史记录，不宣称其与当前文件相同。
按约定验收后清理原始JSON，保留代码、报告和运行所需内核；清理后不能直接重跑旧结果审计。

## 参考

[PyTorch ufunc/add.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ufunc/add.h)的标量T与向量接口用于核对alpha dtype和乘加顺序。
本机安装的Loops.h、Half.h、vec256_half.h用于核对CPU尾部与向量处理；本轮没有复制CPU SIMD实现。
链接main仅作导航，本机参考以Torch 2.14.1+cpu与下列安装文件指纹为准。

## SHA-256

```text
torch_intel_npu.py.before 29d81705ddc2d020008088133d6a333d7a52688ff9be0ee8b554fe6a3a8a3922
npunlock-probe/research/add_alpha_regression.py cbea69c2114f69319af537eab3fc9bdf6f5962f8c3b79e68ceedbe7d8a328eed
npunlock-probe/research/layernorm_mixed_acceptance.py e5318060a13b87351f34d58516849594be527837c13dee85c95416de3377a3b9
npunlock-probe/research/layernorm_mixed_diagnose.py 316bc3cb389f844ccc5b581053d6d86dd7f0307a3054c4afe7c1303bcf42f9fc
npunlock-probe/research/mixed_training_rounding_probe.py 111f32084bed3acd352cbb66a0de94627719347f05a5178028fca602272a38cd
npunlock-probe/torch-device/_torch_intel_npu.pyd b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e
npunlock-probe/torch-device/backend.cpp 006cad067bf8c16782be59b58b9b5d1dfd34244c99329a38854273dd39e6a924
npunlock/tests/fixtures/npu4000/resident/manifest.json f381add6d42cd58731f10d5832f20bd0737d06fa42b77f1655e9ee121837aee1
npunlock/tests/fixtures/npu4000/resident_gelu.c 8c97064a8e9a772ef4729b73a3d1d9e0aa0355a4a60e631368d0c13dd43e8938
npunlock/tests/fixtures/npu4000/resident_layernorm.c d581488a7c057a6c7f84339423757f36d07511ce805a22ec51ed150c74077b01
npunlock/tests/fixtures/npu4000/resident_optimizer.c 7855328ade6a1144c319da94ff184c5046532fde5fe1cf58485a569cc10b4af4
torch_intel_npu.py 0523715f8c418584baf28e75f239b8c3575a814a4f28cd32b03ba2f29067f55f
torch_npu_adamw_demo.py 4389d8c3518726152d753222068ee1853a76a3ce0ab4276aaebe25aaf8621543
torch_npu_autograd.py ab1513ac423b0468dc01399d44855a83efa24227e8c368bc2f55db7d0e31235e
torch_npu_migration_demo.py 69917aecd0fead2fc2b9dbcfb33b3ee8297d9b8c6c03866d95c3431234ee8f02
torch_npu_norm_activation_demo.py 43c2d241fff0bf3797810f9c60e93525e1d175c11f38b4a6f23302390f15c20a
npunlock-probe/research/mixed_training_rounding_report.py 2eb84176de8d41e1c1897cf242882ed92c096b75750f419d63f6d4c02b13b635
npunlock-probe/session-runtime/npunlock_session.dll 0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9
.conda/npunlock/Lib/site-packages/torch/optim/sgd.py 34476f526f3c48d40ebfa86eef556e5d6d740239edc0bb061ea4786d6ccda275
.conda/npunlock/Lib/site-packages/torch/include/ATen/native/cpu/Loops.h 6c83d56976eb9b7d8e92627cc24d7226f5a2a493efd3c17a76471351feb2ec7e
.conda/npunlock/Lib/site-packages/torch/include/torch/headeronly/util/Half.h b820c7f8267d3647edfd8666abbcd1cbe7e74c5ac37a3b93c9b120440b5cfe06
.conda/npunlock/Lib/site-packages/torch/include/ATen/cpu/vec/vec256/vec256_half.h 6092beff1546e33ec878e0eea589356ca305e5ec58aa016814679379d60a49d5
```

## 验收后清理

删除本轮10份原始JSON，共198902 字节。清理前后88份保留文件的SHA-256一致，随后追加本段与空间记录。
保留源码、20份resident ELF及汇编、manifest、诊断/验收/审计脚本、原生运行组件和Markdown报告；未生成图片或持久图缓存。

