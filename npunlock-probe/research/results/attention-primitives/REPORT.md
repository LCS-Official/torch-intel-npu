# softmax、bmm与小型attention训练验收

日期：2026-10-18。本地开发版；公开source-release与GitHub未改。

Windows x64 / Core Ultra 7 258V / NPU 8086:643e / API driver_version=1005540 / 编译器8.3 / Torch 2.14.1+cpu / Python 3.12.14。
CPU固定单线程、关闭MKLDNN、随机种子83；CPU/NPU使用相同初始输入和参数，独立执行训练轨迹。

## 结果

- 新增11组功能检查全部通过：softmax、bmm、小型attention的FP16/FP32前后向及四步SGD训练。
- 原分类logsoftmax/NLL/拒绝契约5组回归通过；迁移边界7通过/4明确未支持。
- 最终有效验收共23组通过、4组明确未支持，**6,547次NPU调用**；另12项离线回归通过。
- 容差未放宽：FP32 rtol=3e-5/atol=3e-6，FP16 rtol=.003/atol=.0005；NaN位置须一致。
- 所有检查live buffer无泄漏；所有进程shutdown后live/cache/scalar/quarantine/session归零，无context failure。

## 实现与使用边界

标准PrivateUse1与AutogradPrivateUse1注册，直接使用torch.softmax/F.softmax/nn.Softmax和torch.bmm。
softmax仅支持rank 1–4、最后一轴、非空且总元素<=65,536；显式dtype=FP32通过。
两个新SHAVE C内核分别计算稳定softmax与解析梯度y*(g-sum(g*y))，复用自编exp_negative/IEEE辅助函数。
FP16在设备内转换成FP32计算后转回；保留subnormal尾部。NaN、含+Inf或全-Inf行与CPU一致传播NaN。
前后向out均需无梯度、精确shape/dtype且不别名；内部half_to_float=True拒绝。
初轮out检查发现新softmax反向out的schema关键字为grad_input，现已修复注册并通过直接调用检查；log_softmax的out关键字仍为out。

bmm仅支持(B,M,K)×(B,K,N)、相同dtype与批次，不广播；FP16/FP32，输入与输出遵守全局1,048,576元素/4MiB存储限制。
输入可为覆盖完整存储的密集转置；不接受部分/重叠/zero-stride视图。
小形状按批次合并展开乘积，再进行FP32补偿归约；工作区大于65,536时拆批次并复用已有二维矩乘的M/N/K分块。
尚未接入DPU batched GEMM；本轮扩展兼容性，不提供吞吐或加速结论。

普通三维@/matmul仍拒绝；没有接入MultiheadAttention、fused SDPA、Embedding、mask/dropout专用接口、通用Transformer、AMP或二阶梯度。
小型四步训练不代表长期收敛；此前混合训练严格轨迹失败保持为独立已知边界。

```python
import torch
import torch_intel_npu

# q/k/v为同dtype、同批次的三维intel_npu Tensor。
scores = torch.bmm(q, k.transpose(1, 2)) * (q.shape[-1] ** -0.5)
output = torch.bmm(torch.softmax(scores, dim=-1), v)
```

## 覆盖范围

softmax每种dtype共52项数值比较：rank 1–4、宽度1/3/17/257/4096/65536、显式FP32输出、
完整密集排列、前后向out、NaN/Inf、极大logits、FP32 subnormal与非有限上游梯度。
bmm小形状每种dtype覆盖3种形状及双侧/左侧/右侧梯度；另检查5种分块形状(B,M,K,N)：
(3,16,64,32)、(2,16,257,17)、(5,128,1,128)、(65537,1,1,1)、(1,1,65537,1)。
包括超过单块的展开工作区、81,920元素输出、65,537批次和K维串流；每组均核对forward/out/双方梯度。
契约包含12项不新增图提交的拒绝检查，以及softmax输出与bmm输入被原地修改后的保存版本保护。
组合attention核对Q/K/V三路梯度；小网络含LayerNorm、Q/K/V Linear、attention、投影、残差、mean pooling及三分类head。
两种dtype各四步标准SGD，foreach=False/fused=False、lr=.025，CrossEntropy显式使用FP32 logits。

| 检查 | 状态 | NPU调用 |
|---|---|---:|
| softmax-torch.float16 | passed | 253 |
| softmax-torch.float32 | passed | 74 |
| bmm-torch.float16 | passed | 328 |
| bmm-torch.float32 | passed | 205 |
| bmm-large-torch.float16 | passed | 454 |
| bmm-large-torch.float32 | passed | 283 |
| contracts | passed | 19 |
| attention-torch.float16 | passed | 73 |
| attention-torch.float32 | passed | 37 |
| training-torch.float16 | passed | 1924 |
| training-torch.float32 | passed | 876 |
| logsoftmax-torch.float16 | passed | 256 |
| logsoftmax-torch.float32 | passed | 79 |
| nll-torch.float16 | passed | 956 |
| nll-torch.float32 | passed | 534 |
| classification-contracts | passed | 37 |
| batchnorm-default | passed | 30 |
| integer-target | passed | 1 |
| cross-entropy | passed | 18 |
| adamw | passed | 56 |
| layernorm | passed | 45 |
| gelu | passed | 2 |
| softmax | passed | 1 |
| embedding | blocked | 0 |
| batched-matmul | blocked | 1 |
| sdpa | blocked | 5 |
| native-checkpoint | blocked | 0 |

### 四步训练损失

| dtype | loss |
|---|---|
| training-torch.float16 | 1.19794869, 1.17833161, 1.15939689, 1.14106262 |
| training-torch.float32 | 1.05507481, 1.04495561, 1.03508449, 1.02544868 |

训练计数区间包住forward/backward/step，不包含显式CPU参考及下载比较。
首步只上传新增标量常量并读取一次分类错误标志；随后每步只有一次4字节错误标志读取。
独立softmax/bmm/attention计算区间仅允许新增标量常量上传；不使用CPU Tensor算术回退。
计数覆盖显式Tensor复制，不覆盖驱动图内DMA、布局搬运或编译器；共享内存不表示无数据移动。

## 构建、回归与复现

新增resident_softmax.c、两份ELF/汇编及构建脚本；原20份resident ELF/汇编未重编，22份与manifest逐一核对。
C++设备源码、PYD与session DLL保持原构建指纹，无新增依赖、模型、图表或持久图缓存。
未重复历史全套36项迁移、卷积大形状、低位LLM或性能测试；本轮采用共享分类与迁移边界回归。

```powershell
./.conda/npunlock/python.exe -B npunlock-probe/research/build_softmax_kernels.py
foreach ($case in @('softmax','bmm','bmm-large','contracts','attention','training')) {
    ./run_torch_npu.ps1 -Script ./torch_npu_attention_demo.py -ScriptArguments @('--case',$case)
}
foreach ($case in @('logsoftmax','nll','contracts')) {
    ./run_torch_npu.ps1 -Script ./torch_npu_classification_demo.py -ScriptArguments @('--case',$case)
}
./run_torch_npu.ps1 -Script ./torch_npu_migration_demo.py -ScriptArguments @('--child','boundaries','--label','attention')
./.conda/npunlock/python.exe -B npunlock-probe/research/attention_primitives_report.py
```

构建需既有外部MoviTools。审计需要上述原始JSON/编译状态，验收后按约定清理，仅保留源码、内核和本报告。

接口参考：[PyTorch softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html)、
[PyTorch bmm](https://docs.pytorch.org/docs/2.14/generated/torch.bmm.html)；实际out关键字按本机安装schema核对。

## 验收后清理

删除本轮22份原始JSON/编译响应，共185,543字节。清理前后89份保留文件的SHA-256一致，随后追加本段及空间记录。
保留源码、构建/验收/审计脚本、22份resident ELF/汇编/manifest和文字报告；公开source-release保持干净，未推送GitHub。

## SHA-256

```text
npunlock-probe/research/attention_primitives_report.py 6c7ad15593cbd6a8d2b109819db68554d4ca66f1225f4a8c5fc004f6d6859b48
npunlock-probe/research/build_resident_kernels.py 01b6f0a344b5477d2991f9700127706aa9a176d4d1e9a2245afb944f776ca4a6
npunlock-probe/research/build_softmax_kernels.py 96a6fe01984f4aebb41754805a4468e04e57708a8337436b6ae32f7aa7b294bf
npunlock-probe/session-runtime/npunlock_session.dll 0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9
npunlock-probe/torch-device/_torch_intel_npu.pyd b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e
npunlock-probe/torch-device/backend.cpp 006cad067bf8c16782be59b58b9b5d1dfd34244c99329a38854273dd39e6a924
npunlock/tests/fixtures/npu4000/resident/manifest.json 3390b7e77e62843e6d7ad939ea59548ae211ea61d1b7020a65a62503dcf40bc9
npunlock/tests/fixtures/npu4000/resident/softmax-backward.elf 67c8e41afd7d0be385e7c055925680c73ddf921875d1f9b79f82b58514c0094b
npunlock/tests/fixtures/npu4000/resident/softmax-backward.s 338868a903d7e8f338a8e39f4c3cf008d7792df13d9fd92e2200afb91956fd8f
npunlock/tests/fixtures/npu4000/resident/softmax.elf f2cf87ee0d09f05b4cf4f41c9c6cc7ea750b6eb2e4c892c1172aa2268e5cd3ab
npunlock/tests/fixtures/npu4000/resident/softmax.s 595a22f3cd624316a6113c1c5d909243cb778f925098a559bd61ecf151e59ace
npunlock/tests/fixtures/npu4000/resident_softmax.c 99e75fc654f6e273c537fb8dd7b4a7073f774011ce42e7dc938fae39d5a4e975
torch_intel_npu.py 234fee94b7d551a994e2da9556b4eb8eea27bc7daacca665ccc22282f17cd6b7
torch_npu_attention_demo.py b4fb932acddcd024bed14a08e616d4f5040f693469461c4951239d1eadfabebc
torch_npu_autograd.py 7352bb83303a4d8e942ba98a23d50d8331886aba3f55db5cb1c7ea7b4bc8dd8f
```
