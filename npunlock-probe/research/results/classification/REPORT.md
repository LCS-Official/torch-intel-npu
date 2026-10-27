# 有界分类训练与FP16 BatchNorm修复验收

日期：2026-10-16。本地开发版，尚未推送GitHub；此前迁移/INT64报告保留为历史记录。

## 结果与范围

- 新增自定义SHAVE log_softmax、log_softmax backward、NLL packet与NLL backward四个内核；
  PrivateUse1及AutogradPrivateUse1接入标准F.log_softmax、F.nll_loss/nn.NLLLoss、F.cross_entropy/nn.CrossEntropyLoss。
- FP16/FP32输入；FP16通过设备内FP32计算返回原dtype，不是DPU原生FP16分类内核。
- log_softmax：密集rank 1–4，仅最后一轴，总元素<=65,536；显式dtype=FP32与无梯度非别名out通过。
- NLL/CE：(C)配标量标签或(N,C)配(N)标签；连续、同设备INT64硬标签；weight=None、label_smoothing=0。
  支持none/mean/sum、signed INT64 ignore_index，且N*(C+4)<=65,536；不提供空输入/空间分类/soft targets/类别权重/NLL out重载。
- 计算及标签合法性判断在NPU；每次NLL/CE前向同步下载4字节错误标志，发现越界抛IndexError。
  标签始终按两个原始32位字通过byte DMA，不转成FP32数值；CPU仅构造形状索引和Python常量。
- C++设备扩展、session DLL、既有SHAVE ELF不变；无需新增Python依赖、OpenVINO runtime、模型下载或持久图缓存。

## SHAVE数值与布局

完整行先减最大值、补偿求sum(exp)，再按(x-max)-log(sum)顺序写回；反向为g-exp(logp)*sum(g)。
exp/log独立实现区间缩减与Taylor/atanh展开，复用既有FP32/subnormal辅助函数；未复制第三方libm源码/表。
这是CPU容差验证的近似实现，不声明通用libm正确舍入、全FP32位模式或所有长度认证。
ReferenceSW Power载体编译后要求一个range/一次invocation、三个完整MemRef；
只接受(width,rows,1,1)，以及单行的(1,1,width,1)物理布局。若编译器拆行或改变布局则拒绝，保留既有代码窗口指纹校验。
NLL选择目标位置直接取值；跳过未选中/被忽略的NaN，反向对被忽略位置直接写零，避免0*NaN。

## FP16整网失败与修复

初版六组分类训练中，FP16 Conv+BatchNorm的输入梯度最大绝对差为0.0118103，卷积权重梯度为0.00445557；
另五组通过。没有放宽容差或把失败改记为通过。相同NPU logits上的分类loss梯度对照通过。
隔离相同BN输入与incoming gradient发现CPU纯FP16保存的mean/invstd是half，而原NPU路径始终保留float。
例如CPU invstd=[8.5390625,17.71875]，原NPU为[8.5355768,17.7227974]。
修复共享_batch_norm：纯FP16训练时先把保存统计舍入到half再转FP32归一化/反向；
方差累加、原始均值以及running mean/variance更新仍使用FP32；有FP32参数/缓冲区的路径保留FP32保存统计。
修复后以原测试种子、数据、四步独立CPU轨迹和原容差重跑；六组全部通过，已有六种BN模型组合也通过。
这一结果覆盖所列小模型，仍不能保证任意纯FP16深层训练轨迹一致。

## 环境与验证口径

Windows x64 / Core Ultra 7 258V / NPU 8086:643e；API driver_version=1005540；编译器8.3。
Python 3.12.14、Torch 2.14.1+cpu，复用.conda/npunlock。CPU1线程、关闭oneDNN，分类测试随机种子53。
逐项assert_close：FP16 rtol=0.003/atol=0.0005；FP32 rtol=0.00003/atol=0.000003；NaN按equal_nan核对。
CPU仅为测试参照；模型上传后独立训练，不按NPU结果重设CPU参数。不测吞吐/TOPS/功耗或长时间收敛。

## 算子、拒绝契约与模型

| 检查组 | CPU比较次数 | NPU调用 |
|---|---:|---:|
| logsoftmax-torch.float16 | 46 | 256 |
| logsoftmax-torch.float32 | 46 | 79 |
| nll-torch.float16 | 80 | 956 |
| nll-torch.float32 | 80 | 534 |

log_softmax覆盖rank1–4、宽度1/17/257/4,096/65,536、大小共同偏置、NaN/+Inf/all−Inf/partial−Inf、
exp尾部−80/−90/−100/−103/−104，以及Inf/NaN incoming gradient。
NLL/CE覆盖三种reduction、普通/部分/全部忽略、正ignore_index、INT64_MIN、未选中NaN、
unbatched与batched total_weight，以及2,049行的FP16有效标签计数舍入边界。
12种结构/参数/别名拒绝没有新增图提交；四种越界标签（含2^32、2^53+1）在设备检查后安全拒绝；
backward前修改target触发saved-tensor version检查，context保持健康。

| 模型/精度 | 四步NPU loss | 各步显式host copies | 首次标量上传 |
|---|---|---|---|
| mlp-torch.float16 | 1.0332031 → 1.0332031 → 1.0322266 → 1.03125 | [5, 2, 1, 1] | [4, 1, 0, 0] |
| mlp-torch.float32 | 1.0659971 → 1.0638864 → 1.0601679 → 1.0552934 | [3, 1, 1, 1] | [2, 0, 0, 0] |
| conv-torch.float16 | 0.97021484 → 0.96582031 → 0.95898438 → 0.95019531 | [7, 1, 1, 1] | [6, 0, 0, 0] |
| conv-torch.float32 | 1.3939296 → 1.3676697 → 1.3677602 → 1.3376637 | [1, 1, 1, 1] | [0, 0, 0, 0] |
| residual-torch.float16 | 1.0234375 → 1.0214844 → 1.0175781 → 1.0136719 | [1, 1, 1, 1] | [0, 0, 0, 0] |
| residual-torch.float32 | 1.2597446 → 1.2443246 → 1.2174147 → 1.1878984 | [1, 1, 1, 1] | [0, 0, 0, 0] |

MLP：5→8→3；Conv：1→2卷积、默认BatchNorm、ReLU、全局池化、Linear；
残差网络：三层卷积、两个默认BatchNorm、残差ReLU、全局池化和Linear。batch4，输入(4,5)或(4,1,4,4)。
标准CrossEntropyLoss与SGD(lr=0.025,momentum=0.8,foreach=False,fused=False)，三步后显式CPU checkpoint恢复并继续第四步。
每步检查输出/loss/输入梯度/所有参数梯度/参数/动量/全部BN buffers；另核对相同logits上的分类梯度，末尾eval。
复制统计区间仅含NPU forward/backward/step，排除验证下载、输入准备和checkpoint；每步copies=首次标量上传+一次4B错误标志读取。
统计不包含驱动内部DMA，不能描述成完全零拷贝或异步训练。

## 最终计数与资源

- 分类11组全部passed：8,965次真实NPU调用。
- 迁移套件28组功能passed、8组blocked：6,550次调用；相对上一轮27/9新增分类入口。
- 既有默认BatchNorm模型6组、设备Quick33组、训练Quick11组、原兼容性5组passed：12,020次调用。
- 最终验收合计27,535次真实NPU调用，不含开发失败/诊断/重跑。
- 4项分类载体离线测试、3项INT64载体回归、5项源码导出回归，共12项通过。
- 每组live buffer无增量，各进程无context failure；shutdown后live/cache/scalar/quarantine/session全部归零。

仍blocked：AdamW、LayerNorm、GELU、softmax、Embedding、batched matmul、SDPA、原始NPU Tensor序列化。
未认证CUDA运行时/扩展、torch.xpu合并、AMP、二阶梯度、多流、GPU+NPU协同、其他硬件/驱动或任意torchvision/HF模型。

## 复现与参考

```powershell
foreach ($case in @('logsoftmax','nll','contracts','training')) {
  .\run_torch_npu.ps1 -Script .\torch_npu_classification_demo.py -ScriptArguments @('--case',$case)
}
.\run_torch_npu.ps1 -Script .\torch_npu_batchnorm_demo.py -ScriptArguments @('--case','models')
.\run_torch_npu.ps1 -Script .\torch_npu_migration_demo.py
.\run_torch_npu.ps1 -Device -Quick
.\run_torch_npu.ps1 -Training -Quick
.\run_torch_npu.ps1 -Script .\torch_npu_compat_demo.py
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python;$PWD\tools"
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_classification test_torch_npu_int64 test_source_release
.\.conda\npunlock\python.exe -B npunlock-probe/research/classification_report.py
```

修改分类C源码时用现有Movi环境执行build_classification_kernels.py；默认无需重建即可使用配套ELF。
按用户约定，报告审计后清理原始JSON/日志/编译响应；清理后需重新跑上述硬件检查生成审计输入。
数值算法与API契约参考[PyTorch LogSoftmax](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h)、
[SoftMaxKernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/SoftMaxKernel.cpp)、
[LossNLL](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossNLL.cpp)、
[Normalization](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Normalization.cpp)；调研与独立实现边界见根目录REFERENCES.md。

## 当前SHA-256

```text
torch_intel_npu.py
ef3666feb3dfe3ba426e031477d47c1d6b49c47e7cb604474ade5f99e09aeb82
torch_npu_autograd.py
9e8ec4aaf438728baa733e11fb52ef142e4aa7d7fe2fd832ddd66bfd650a1285
torch_npu_classification_demo.py
63a5e69fb33f4ab8afc67b8fa2b16ee4fe91d8338ea177a5afdcff7137e07f1a
npunlock/python/npunlock/npu4000.py
96fd710b05d130b8290ce17d5bf76f1ec0ad3ecdc4176b1b61f18ae4e539e921
npunlock/tests/fixtures/npu4000/resident_classification.c
fcfb57bae8381b1875e11dec36505fdefd6fda3a858a5fbc7aca1c45ddfd607c
npunlock-probe/torch-device/_torch_intel_npu.pyd
b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e
npunlock-probe/torch-device/backend.cpp
006cad067bf8c16782be59b58b9b5d1dfd34244c99329a38854273dd39e6a924
npunlock-probe/session-runtime/npunlock_session.dll
0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9
npunlock/tests/fixtures/npu4000/resident/log-softmax.elf
0f392dcd91c573c763c6d7fcd6f544bc02ebf84d3765f71fcd791fcc3fbbb91c
npunlock/tests/fixtures/npu4000/resident/log-softmax-backward.elf
e57c32d6f932fb2cd42fe0e59865b39233bd13dd6ac28bb4659fa82e8dbb6319
npunlock/tests/fixtures/npu4000/resident/nll-packet.elf
fdf1a8b05da3ca74039ea6023482d345b60618af4cd4f4dad941d6c039b08fe2
npunlock/tests/fixtures/npu4000/resident/nll-backward.elf
0b676d08eb13fa1506733aa6b7fe243b743a5842f9c683ffd84d01966ac3f12c
```

## 验收后清理

已删除53份原始JSON/日志、Movi编译响应和临时隔离诊断脚本，共290,403字节（约283.60KiB）。
仅删除逐项列出并验证位于工作区内、无reparse ancestor的文件；保留C/Python源码、构建/测试脚本、ELF/汇编/manifest与本报告。
清理前后核对四个新增ELF、PYD、session DLL及三个本轮/历史报告的哈希不变。之后仅为本报告附加此清理记录。
没有新增依赖、模型、图片或持久图缓存；未改动公开source-release快照或推送GitHub。
