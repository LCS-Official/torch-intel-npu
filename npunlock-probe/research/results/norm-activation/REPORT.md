# LayerNorm、GELU与残差前馈训练验收

日期：2026-10-18。本地开发版；公开source-release与GitHub未改动。历史报告保留原结果。

Windows x64 / Core Ultra 7 258V / NPU 8086:643e / API driver_version=1005540 / 编译器8.3 / Torch 2.14.1+cpu。
新增检查固定CPU单线程、关闭MKLDNN、随机种子73；独立CPU模型与NPU模型共享初始参数和输入。

## 结论与限制

- 同dtype FP16/FP32 LayerNorm前向、保存统计与一阶梯度通过有限CPU对照。
- GELU默认erf公式与tanh公式的FP16/FP32前后向及out通过；不保证逐位同CPU。
- 含LayerNorm、Linear、GELU、残差与CrossEntropy的前馈网络：FP16/FP32 SGD及FP32 AdamW，
  两种GELU各四步，损失、全部参数梯度与更新按独立CPU训练轨迹比较，六种组合通过。
- FP16输入配FP32 affine的宽LayerNorm仍有严格参数梯度失败，**暂不认证此混合组合**。
  没有放宽容差、把失败改成通过或重置CPU轨迹。当前完整验收命令因此返回非零。
- 本轮没有认证完整Transformer、通用bmm/SDPA/softmax/Embedding、AMP或二阶梯度。

## 执行路径与契约

设备仍为intel_npu PrivateUse1；标准nn/F入口直接调用本后端，无CPU张量算术回退。
LayerNorm复用FP32归约、平方、rsqrt与广播，前向使用完整统计，保存统计按输入/affine dtype舍入；
FP16参数梯度逐行经过FP16累加，匹配单线程CPU参考的缓冲语义。此路径提交次数较多，尚未优化性能。
rank 1–4、非空normalized_shape匹配尾部、eps为正有限FP32、可选weight/bias，
输入/affine须为完整密集存储，遵守全局1,048,576元素与4MiB allocation界限；不提供native out重载。
混合FP32 affine必须有FP32 weight；只有FP32 bias的半精度组合，CPU本身的反向也会类型报错，本后端明确拒绝。
高动态范围近常量行、不同CPU线程归约顺序、全部FP32编码和长时间训练未全面认证。

GELU使用四个新SHAVE FP32内核；FP16输入设备内转FP32，结果/梯度转回FP16。
erf采用A&S 7.1.26公开数学系数，自行实现求值；tanh采用exp及小值奇多项式；
复用本项目exp_negative与IEEE subnormal辅助函数，解析一阶导数按各自公式计算。
使用已有Power载体（前向第二只读绑定复用输入），不改代码窗口大小、MemRef布局、任务调度或驱动指纹校验。
仅GELU构建单元对IEEE辅助函数禁止展开并使用-Os，原辅助源码与已有内核不改；不引入libm/OpenVINO runtime。
GELU rank 0–4、最多1,048,576元素，内部65,536元素分块；out只用于无梯度、精确shape/dtype且不别名。

## 测试与严格失败

容差沿用FP32 rtol=3e-5/atol=3e-6，FP16 rtol=.003/atol=.0005，NaN位置须一致；
依据被比较Tensor的dtype选择阈值，FP32 affine梯度仍使用严格FP32门槛。
LayerNorm包含rank 1–4、多尾轴/全轴、affine四种选项、完整密集转置、单元素、4096宽层、
17×4096跨块、常量/近常量行及NaN/Inf；混合组在首个严格失败处停止，不能视为全部混合用例通过。
GELU每个dtype/公式24项比较：4097点网格、特殊/极大值、密集排列、65,537元素跨块、
全部65,536个FP16位模式或8192个随机FP32位模式，以及标量；包括直接out前向和反向。
另有13项无提交拒绝检查和两项保存Tensor版本保护。FP16全部编码覆盖不等于正确舍入认证。

| 检查组 | 状态 | 数量/说明 |
|---|---|---|
| layernorm-torch.float32-mixed-False | passed | 166项比较 |
| layernorm-torch.float16-mixed-False | passed | 166项比较 |
| layernorm-torch.float16-mixed-True | failed | (1, 4096)/(4096,)/both, gradient (4096,): Tensor-likes are not close!  Mismatched elements: 1 / 4096 (0.0%) Greatest absolute difference: 0.0001220703125 at index (1228,) (up to 3e-06 allowed) Greatest relative difference: 0.000892060634214431 at index (1228,) (up to 3e-05 allowed) |
| gelu-torch.float16-none | passed | 24项比较 |
| gelu-torch.float16-tanh | passed | 24项比较 |
| gelu-torch.float32-none | passed | 24项比较 |
| gelu-torch.float32-tanh | passed | 24项比较 |
| contracts | passed | 13项拒绝/2项版本保护 |
| feed-forward-torch.float16-sgd-none | passed | 4步独立训练 |
| feed-forward-torch.float16-sgd-tanh | passed | 4步独立训练 |
| feed-forward-torch.float32-sgd-none | passed | 4步独立训练 |
| feed-forward-torch.float32-sgd-tanh | passed | 4步独立训练 |
| feed-forward-torch.float32-adamw-none | passed | 4步独立训练 |
| feed-forward-torch.float32-adamw-tanh | passed | 4步独立训练 |

半精度CPU LayerNorm即使使用FP32 affine也会将参数梯度累加缓冲存成FP16；
FP32均值/逆标准差及乘加顺序的微小差异可跨过该FP16舍入边界，再体现为FP32梯度的一次half ULP跳变。
这是失败机制解释，不能据此认定本实现与CPU逐位一致；严格失败保留供后续修复。

### 六组训练损失

| 配置 | 四步loss |
|---|---|
| feed-forward-torch.float16-sgd-none | 1.4824219, 1.4667969, 1.4375, 1.4003906 |
| feed-forward-torch.float16-sgd-tanh | 1.453125, 1.4365234, 1.4072266, 1.3691406 |
| feed-forward-torch.float32-sgd-none | 1.2690489, 1.2484277, 1.2124395, 1.1659769 |
| feed-forward-torch.float32-sgd-tanh | 1.8789146, 1.8304569, 1.7458608, 1.6366901 |
| feed-forward-torch.float32-adamw-none | 1.6165218, 1.6062026, 1.5959914, 1.5858853 |
| feed-forward-torch.float32-adamw-tanh | 1.3000597, 1.2917702, 1.2835379, 1.2753602 |

## 回归与资源

- 最终同版验收合计**39,503次真实NPU调用**，120组passed、5组明确blocked、1组严格failed。
- 既有device Quick 33、training Quick 11、兼容性5、SGD分类11及AdamW更新/状态/边界16组通过。
- 迁移检查31组功能通过/5组明确未支持；新增LayerNorm/GELU迁移探针包含非平凡输入、CPU前向与反向对照。
- 剩余blocked：softmax, embedding, batched-matmul, sdpa, native-checkpoint。
- 新增及迁移/分类/AdamW的逐组live buffer增量为零；所有进程shutdown后live/cached/scalar/quarantine/session归零，context无失败。
- 12项离线测试通过（INT64载体3、分类载体4、源码导出5）；8份相关Python源码通过语法解析。
- AdamW的FP16默认epsilon零梯度NaN一致性仅为既有边界行为检查，不计作健康训练认证；旧整网FP16限制仍在。
- 本轮不复测历史AdamW卷积BN冗余bias诊断、低位LLM质量或DPU适配器性能；不声称性能提升。

## 复现

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_norm_activation_demo.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\norm_activation_regression.py
.\.conda\npunlock\python.exe -B npunlock-probe/research/norm_activation_report.py
```

前两个完整验收入口保留严格混合失败而返回1；第三步仅在除该已记录失败外的全部检查及指纹均符合时写报告。
最终验收JSON/日志和编译响应在报告审计后清理；复现需先重跑上述测试。

## 数学与语义参考

- [A&S p.299，7.1.26](https://personal.math.ubc.ca/~cbm/aands/page_299.htm)：公开数学公式和系数，非第三方源码移植。
- [PyTorch Activation.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/Activation.cpp)：两种公式与一阶导数。
- [PyTorch layer_norm_kernel.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/layer_norm_kernel.cpp)：保存统计及参数梯度缓冲的dtype。

## 源码及内核指纹（SHA-256）

```text
torch_npu_norm_activation_demo.py 43c2d241fff0bf3797810f9c60e93525e1d175c11f38b4a6f23302390f15c20a
torch_intel_npu.py acb1a43c29aa10d0cae36d8936db87f76c76886f07dc8678b9e3bd1764ba6632
torch_npu_autograd.py ab1513ac423b0468dc01399d44855a83efa24227e8c368bc2f55db7d0e31235e
npunlock/tests/fixtures/npu4000/resident_gelu.c 8c97064a8e9a772ef4729b73a3d1d9e0aa0355a4a60e631368d0c13dd43e8938
torch_npu_migration_demo.py 69917aecd0fead2fc2b9dbcfb33b3ee8297d9b8c6c03866d95c3431234ee8f02
npunlock-probe/research/build_gelu_kernels.py 6c3984a4a4650f0ca58c30d8c5f2b44c96915f49291564ca460cee6594a4ea2f
npunlock-probe/research/build_resident_kernels.py e24b184773d7e10b22528e120c62deea1eeb6eb98ef7bc5a8f9443cfe8e9dd0d
npunlock-probe/research/norm_activation_regression.py 2f603884bf056ba6eab5adfa44f9428a76c8a0d52f46e7d1e417f6b4afab89e9
npunlock-probe/research/norm_activation_report.py 05131f65e53f73366ac8a96069de569e6719f3030bb9755e38493e0cb2333385
npunlock-probe/torch-device/backend.cpp 006cad067bf8c16782be59b58b9b5d1dfd34244c99329a38854273dd39e6a924
npunlock-probe/torch-device/_torch_intel_npu.pyd b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e
npunlock-probe/session-runtime/npunlock_session.dll 0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9
npunlock/tests/fixtures/npu4000/resident/manifest.json 893e8b57255bbadabdb9a3585936ea554e70ed9cc7640ec6f36d8a26f96a3932
npunlock/tests/fixtures/npu4000/resident/gelu-none-f32.elf 4948dd85eda4bd5152fe687114d4c60a93881fb2411fb56197bf2a43a9f12c4b
npunlock/tests/fixtures/npu4000/resident/gelu-tanh-f32.elf 573c9ac4a5ef1477b2dd0f786da35a9e63dee20782563230f47655dddb6c2dd2
npunlock/tests/fixtures/npu4000/resident/gelu-backward-none-f32.elf 7d44cdbd29c0d09ce0d175fbdcd24a28f6cc1e883e1e0773a029213e0e0db25c
npunlock/tests/fixtures/npu4000/resident/gelu-backward-tanh-f32.elf fe1c062cb026188d9d57d359f021c99ad3325ca76a3ebd8b24eef93baecff02c
```

## 验收后清理

删除78份原始JSON、日志与Movi编译响应，共429,771字节（419.70KiB）。
删除时核对68份保留源码、内核、宿主组件、文档和报告的SHA-256完全一致，随后补充本段清理统计。
没有新增依赖、模型、图片或持久图缓存；保留全部新C源码、四份ELF/汇编、manifest、复现与报告脚本。
公开source-release的Git工作区保持干净，未推送GitHub。
