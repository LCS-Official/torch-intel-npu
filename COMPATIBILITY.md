# 算子、模型与运行时兼容性

状态日期：2026-10-11。适用范围为 Windows x64、Core Ultra 7 258V、NPU `8086:643e`、
驱动 `32.0.100.5540`、图编译器 `8.3`、Torch `2.14.1+cpu`。
这是当前实现和有限测试覆盖的清单，不表示所有形状、参数和组合均已验证。

设备名为 `intel_npu:0`。普通 ATen 后端与冻结 DPU 适配器是两条独立入口；
后者的矩乘/模型支持不能自动推广到 `.to("intel_npu")` 或通用 `torch.compile`。
安装方式见 [INSTALL.md](INSTALL.md)，性能口径见 [README.md](README.md#性能实测)。

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

## 普通设备 Tensor 与存储

| 接口 / 项目 | 当前契约 | 证据与边界 |
|---|---|---|
| `empty` / `empty_like` / `zeros` / `ones` / `full` | FP16/FP32；rank 0–4；1–1,048,576 元素 | D 的工厂、复制和形状检查；没有空 Tensor 或通用整数 Tensor |
| `empty_strided` | 完整密集维度排列，无孔洞/重叠，非负 stride | D 的拒绝检查，C 的转置叶训练；不是任意 strided 存储 |
| `view` / `reshape` / `t` / `transpose` / `permute` / `detach` | 支持部分元数据变换；随后参与计算时须满足完整存储绑定契约 | D 的 `views`、`transpose_bits`；不能把能创建一个 view 当作能计算 |
| `.contiguous()` / `.clone()` | 完整密集转置可通过设备 DMA materialize；受存储上限约束 | D、C 的布局与生命周期对照 |
| `.to()` / `copy_` | CPU↔NPU、NPU↔NPU，FP16/FP32，同步；`copy_` 不广播形状 | D 的复制和 cast；非连续 CPU `.to()` 上传需显式 `memory_format=torch.contiguous_format` |
| 设备内 dtype 转换 | FP16↔FP32 的 SHAVE/DMA 转换；设备内转换的一阶梯度有限覆盖 | D 的全部 half 编码/FP32 样本；训练代码中的显式转换 |
| CPU/NPU 传输与 autograd | 传输不提供跨设备可微链；使用 `detach()` 或 `no_grad()`，上传后再 `requires_grad_()` | D 的 `autograd_upload` 拒绝；不支持把 CPU 叶直接可微上传 |
| 输入绑定 | `storage_offset=0`，覆盖整个 allocation；密集排列由支持的入口 materialize | 部分切片、偏移、expand 的零 stride/重叠和负值视图不可作为通用计算输入 |

元素上限是当前软件契约，不是物理内存容量。卷积、广播和矩乘还要检查输出与中间工作区。
共享物理内存/Level Zero shared allocation 不意味着没有 DMA、布局转换或同步。
`view(dtype)` 有底层注册，主要用于内部位重解释，不把它认证为通用可微 dtype 转换 API。

## 算子与一阶梯度

表内“有”表示实现并有对应有限测试，不表示逐位等同 CPU 或可进行二阶求导。
普通设备算术支持 FP16/FP32；FP16 矩乘/卷积等路径可能使用 FP32 中间计算。

| 操作 | 一阶梯度 | 当前参数与形状限制 | 证据 |
|---|---|---|---|
| `add` / `sub` | 有 | `alpha=1`，不接受 bool alpha；实数标量或 FP16/FP32 Tensor；广播结果满足全局上限 | D 的 arithmetic/scalar，T 的 arithmetic gradients |
| `mul` / `div` | 有 | 有限广播与 dtype promotion；`div` 仅 `rounding_mode=None` | D、T；C 的大/小分母及 0/Inf/NaN 梯度 |
| `rsub` / 标量减 Tensor | 未单独认证 | 设备实现要求 `alpha=1`；无单独的本项目 autograd 注册/梯度基线 | 不能从 `sub` 的测试推断全部 `rsub` 重载 |
| `neg` / `abs` / `sign` / `reciprocal` | 有 | FP16/FP32；部分 `out=` 只用于不需要梯度的调用 | C 的 unary；保留已有零/特殊值约定 |
| `relu` | 有 | FP16/FP32；这里指非 inplace 入口 | T 的全部 half 编码和 FP32 特殊值 |
| `sum` / `mean` | 有 | 轴可为 None/int/tuple/list，支持负轴、keepdim；轴不可重复；结果 dtype 仅 FP16/FP32 | D 的 axes/edges，T 的 reduction，C 的 dtype backward |
| `mm` / `matmul` / `@` | 有 | **只接收二维**、匹配 dtype 的矩阵；完整密集转置可 materialize；不提供 vector/batched matmul | D 的 matrices/edges/transposed，T 的 matrix gradients |
| `addmm` | 有 | bias 同设备/同 dtype 且可广播到 `(M,N)`；实数 alpha/beta，有限系数需在 FP32 可表示范围 | D 的 bias/alpha/beta/out 对照，T 的 addmm gradients |
| `linear` / `nn.Linear` | 有 | 输入 rank 1–4、权重 `(out,in)`，dtype 匹配；可选 bias 形状 `(out,)` | D 的 1–4 维 forward，T 的 input/weight/bias gradients |
| `convolution` / `F.conv2d` / `nn.Conv2d` | 有 | NCHW、4D weight、groups=1、非转置；stride/padding/dilation 为合法二维值；bias 同 dtype、匹配输出通道 | T 的 3 种几何/两种 dtype；M 的分块梯度；stream demo 覆盖更大工作区 |
| `mse_loss` | 有；target 梯度有实现、未单独认证 | input/target 同 shape、同 dtype；不广播；reduction 为 none/mean/sum | T 的网络 mean-loss 训练、源码中的 MSE backward；未逐项认证所有 reduction/特殊值组合 |
| `F.batch_norm` | 有 | NCHW；训练每通道样本数>1；训练 eps>0；eval 需成对 running buffers；浮点参数为输入 dtype 或 FP32 | M 的 affine、混合 dtype、running buffers、train/eval gradients |
| `nn.BatchNorm2d` | 有限 | 已覆盖 `track_running_stats=False`；默认 True 的模块含整数计数 buffer，当前不认证为可直接 `.to()` 的用法 | M 的模块 train/eval；与 functional 的浮点 running buffers 支持区分 |
| `AdaptiveAvgPool2d` | 有 | **NCHW 且输出仅 `(1,1)`**；不是通用平均/最大池化 | M 的 global average pool gradients |
| `fill_` / `zero_` / `add_` | 更新用途 | `add_` 要求 `no_grad`、输入同 shape/dtype/device；支持实数 alpha；不承诺一般可微 inplace 图 | C 的 dense SGD、T 的优化器更新与 version guard |

`out=` 仅在已注册的重载上可用，要求精确 shape/dtype，通常不能与输入共用 storage。
它不是普通自动微分接口；不要把无梯度的 `out=` 成功推广到可微调用。
FP32 补偿归约/分块可能与 CPU 累加顺序不同，FP16 也可能存在不同融合/舍入；使用相应测试的数值容差。
内部 `_execute("rsqrt", ...)`、RoPE、RMSNorm、attention 图不等于公开 `torch.rsqrt/sin/softmax/...` 算子已支持。

## 训练与运行方式

| 项目 | 状态与建议 |
|---|---|
| 基本 SGD | 已覆盖 `momentum=0, weight_decay=0, foreach=False` 的常规更新与 zero_grad；不认证 fused/foreach、动量、Adam/AdamW、AMP/GradScaler 或可微优化器 |
| 一阶 autograd | 支持表中已实现的导数、分支梯度累加与部分密集布局；多数自定义 backward 标记 `once_differentiable` |
| 二阶导数 | 不支持；不能把 `create_graph=True` 的第一步返回当作二阶求导成功 |
| 后端导入顺序 | 在进程首次任何 backward 前导入 `torch_intel_npu`；晚注册可能触发 PyTorch 队列限制，C 验证局部 `set_multithreading_enabled(False)` 办法 |
| FP16 小网络 | 有限数值测试通过；不能保证深层训练轨迹与 CPU 一致 |
| SmallResNet FP16 | 既有测试每步用相同权重重新建立 CPU 参考；通过逐项容差仍不认证独立训练轨迹或接近零梯度的相对误差 |
| 随机初始化 / RNG | 在 CPU 生成随机数/初始化，再上传；NPU `rand/randn` 没有实现，`manual_seed_all` 不代表设备 RNG 已支持 |
| 多线程 / 异步 | 多线程生命周期有检查，提交仍序列化；不支持 `non_blocking=True`、CUDA 等价的异步 stream 或 GPU/NPU 混合 Tensor 算术 |
| 生命周期 | 图会话默认上限 512；存储缓存有独立容量；`shutdown()` 后不可计算/重新初始化，同进程继续使用需另起进程 |

## 冻结 DPU / 低位 / 模型适配器

| 入口 | 精度与范围 | 不应推断的能力 |
|---|---|---|
| `QuantizedLinear` | 显式冻结 INT8/INT4/NF4 权重，FP16 激活与输出；部分逐通道/分组、W8A16/W8A8 图 | 不是普通低位设备 Tensor、低位训练、QLoRA 或任意 GGUF/HF 量化 checkpoint 导入 |
| INT4/NF4 分组 | 有编码/IR/数值验证；部分 INT4 小 batch 用补零/截取规避编译问题 | 不保证图内仍保持四倍压缩，也不把压缩常量宣称为原生 4-bit 乘加 |
| `FrozenSmolLayer/Stack` | SmolLM2-135M-Instruct，FP16 / W8A16；显式 prefill/KV/decode；固定 64-position 容量，batch<=8，受工作区约束 | 不支持任意 LLM；不是 Transformers 模型 `.to("intel_npu")` 即可运行，也不是自动 torch.compile 后端 |
| 完整 30 层 FP16 | 历史固定短上下文测试通过，decoder/KV 在 NPU；两端和采样在 CPU | 不代表长上下文、任务质量、训练、功耗或多设备协同已认证 |
| 完整模型 W8 | 有校准与执行路径，既有独立/连续质量检查未全部通过 | 当前不能作为通用默认精度；INT4/NF4 全模型也未认证 |

低位与完整模型的详细证据分别见 [INT8](npunlock-probe/research/results/int8/REPORT.md)、
[INT4/NF4](npunlock-probe/research/results/lowbit/REPORT.md)、[模型对照](npunlock-probe/research/results/model-cross-device-followup/REPORT.md)
和 [W8 质量边界](npunlock-probe/research/results/w8-full-calibration/REPORT.md)。

## 明确拒绝与未支持场景

D 的 `unsupported_contracts` 与 T 的 `training_contracts` 集中检查非法输入/未支持场景，
包括 dtype/rank/容量、部分 view、异步复制、非二维矩乘、非法广播/alpha/out、Conv2d groups/转置、
MSE 广播、BatchNorm 参数、非 1×1 池化。拒绝组还核对这些动作没有新增 NPU 图提交。
保存的权重 version 变更、二阶梯度、缓存容量以及 shutdown 后访问另有独立检查。

`sin` 和设备 RNG 等未注册算子显式报错，不自动用 CPU 完成。
但不是每一个 PyTorch API 都有统一的预检入口：元数据 view 可以先创建，直到计算/复制绑定时才拒绝。
未出现在支持表中的 API，应视为未认证；尤其是 softmax、GELU、LayerNorm、Embedding、通用 bmm、
默认 torchvision/Hugging Face 模型、量化训练和其他 PrivateUse1 后端共存。

## 本轮发布基线检查

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
