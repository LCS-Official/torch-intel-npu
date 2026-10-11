# npunlock 在本机 Lunar Lake 上的实测

测试日期：2026-10-03。结论：**通过 Level Zero 和驱动编译器执行普通静态图可行；npunlock 的自定义 SHAVE 内核链路尚未兼容本机图格式。**

后续进展：在用户要求研究移植后，已用 `4000xx` 目标构建并在本机验证了自定义 FP16 加一内核，三组输入误差均为 0。这是固定 carrier 的独立研究原型，原始 npunlock 自动解析/补丁 API 仍需移植。详见 [NPU4000 移植研究](research/PORTING_NPU4000.md)。以下保留第一轮测试记录。

## 测试环境

- CPU：Intel Core Ultra 7 258V；GPU：Arc 140V；物理内存约 32 GB。
- NPU：Intel AI Boost，Level Zero 返回 vendor `0x8086`、device `0x643e`。
- 现有 NPU 驱动：`32.0.100.5540`，日期 2026-08-20；提取工具后再次核对版本一致。
- 驱动图编译器：`8.3`；graph extension：`1.18`。
- 独立 conda prefix：`E:\Pytorch_xfire_NGPU\.conda\npunlock`；Python 3.12.14、NumPy 2.5.3。
- 克隆源码：`E:\Pytorch_xfire_NGPU\npunlock`，提交 `bdf1c01d2caa2e772315a9b2a529981c2810e037`。
- 实际执行使用发布版 `2026-09-28.00` 的 Windows wheel，包版本 `0.1.0`。未从最新源码重新编译 C 库。本机没有安装 MSVC/Windows SDK。
- 没有安装 Python OpenVINO 或 PyTorch；图由 npunlock 序列化为 OpenVINO IR，再通过 Level Zero 调用现有 Intel 驱动编译器。这绕过了 OpenVINO Python/runtime，仍依赖 Intel 编译器及其 IR。

历史 hardware/downloads JSON 与完整环境冻结文件不随公开源码快照分发；原始实验数据已按存储策略清理。
当前公开安装入口为 [INSTALL.md](../INSTALL.md)，依赖与构建验收见 [INSTALLATION_REPORT.md](../INSTALLATION_REPORT.md)。

## 普通运算

所有输入/输出均为 FP16。固定随机种子，先由 NumPy 计算参考结果；矩阵乘参考用 FP32 累加后转 FP16。每个图以 copied API 执行 3 次，每次均比较结果。

| 运算 | 形状 | 三次正确性 | 最大绝对误差 | 编译耗时 ms | copied 调用中位数 ms |
|---|---|---|---:|---:|---:|
| Abs | 1×2048 | 通过 | 0 | 153.32 | 6.23 |
| Add | 1×2048，两输入 | 通过 | 0 | 146.68 | 5.62 |
| Subtract | 1×2048，两输入 | 通过 | 0 | 140.25 | 5.59 |
| Multiply | 1×2048，两输入 | 通过 | 0 | 145.28 | 4.81 |
| MatMul | 32×32 @ 32×32 | 通过 | 0 | 136.28 | 5.57 |

逐元素运算要求精确相等；MatMul 设 `rtol=0.02, atol=0.02`，本次实际误差也为 0。这是小规模兼容性测试，不能外推到任意形状、精度或模型。

copied 调用耗时包含建执行会话、加载图、分配/复制内存、提交、同步及 Python 返回。它不是纯 NPU 内核时间，也不是与 CPU/GPU 的吞吐比较。未测试大矩阵、功耗或长时间稳定性。

## 共享内存与会话复用

发布包和当前克隆源码的 `InferenceSession.__init__` 都向只有 4 个字段的 ctypes `_InferOptions` 传了 5 个参数，导致 `TypeError: too many initializers`。已删除多余的 `_View(None, 0)`，与公开 C 头文件匹配；修复同步应用于克隆源码和本地安装包，未提交。可审阅差异见 [shared-session-fix.patch](shared-session-fix.patch)。原始下载 wheel 保留不变。

修复后，复用会话/共享内存测试先热身 3 次，再测 20 次，并校验输出：

| 运算 | 成功一轮的调用中位数 ms | 已观察到的问题 |
|---|---:|---|
| Add | 0.70 | 本轮通过 |
| Subtract | 0.73 | 本轮通过 |
| MatMul | 0.80 | 本轮通过 |
| Multiply | 0.60 | 先前一轮曾失败，之后一轮通过 |
| Abs | — | 两轮均在共享会话执行时失败 |

失败诊断：`zeFenceHostSynchronize returned Level Zero result 0x70000001`。该返回值对应 device lost；错误来自执行/同步阶段，具体原因未定位。**共享会话路径还不能视为稳定可用。** 一次成功的 20 次计时不足以证明长期稳定。保存的记录在 [results/shared](results/shared)，初期诊断在 `results/initial`。

环境需要正确的 conda DLL 搜索路径。初次未配置时，NumPy 的矩阵乘参考步骤异常退出；添加环境的 `Library\bin` 等 PATH 后正常。复现脚本已处理这一点。

## 自定义 C 内核

试验源代码是 1×16 FP16 的 `out[i] = in[i] + 1`，使用项目提供的 NPU3720 头文件和默认目标。

1. **工具链通过**：从项目文档指定的 Lenovo 官方驱动包中，仅提取 MoviTools 到 `toolchain\Driver\MVC_DEPEND`。包 SHA-256 已验证，没有运行 `Silentinstall.bat` 或驱动安装步骤。编译得到 828 字节的 [add1.elf](results/add1.elf)。这只验证工具链能生成 NPU3720 ELF。
2. **carrier 图通过**：当前驱动编译了 1×16 FP16 Abs，并成功执行，与 CPU 精确一致。因此原始图可被本机驱动加载和执行。
3. **ACT 发现失败**：Python API 和发布版独立 `npunlock-inspect.exe` 都返回：

   ```text
   malformed_input / patchblob.parse_graph
   graph section extent is outside the input blob
   ```

4. **未执行补丁和自定义内核**：没有强制指定偏移、修改驱动或跳过边界校验。自定义内核的指令集兼容性、调用 ABI 和 firmware 接受情况均未验证。

离线检查找到与解析失败直接对应的节：`buffer.CMX_NN.0`，type `0x8aaaaaad`，offset 0，size 1,571,392，而文件仅 53,920 字节。旧解析器只对标准 `SHT_NOBITS=8` 豁免文件数据范围检查。另有三个同类型 CMX 节。完整记录见 [custom-elf-sections.json](results/custom-elf-sections.json)。

本机图使用 `shave.text`、`shave.params`、`task.shave.range.0.0` 至 `.3.0`、`task.shave.invocation.0.0` 至 `.3.0` 等名称；当前源码期待 `.text.KernelText`、`.text.KernelParams`、`.text.ActKernelInvocations`、`.text.ActKernelRanges`、`.rlt.text.ActKernelRanges`。因此，仅放宽 CMX 节检查仍不足以完成移植，还需识别新版 section、任务描述符、重定位以及跨 tile 分组，再验证内核 ABI。

错误记录见 [custom.json](results/custom.json)，原始图、XML 和 carrier 输入输出均保存在 `results`。这些现象证明现有解析器不适配该图格式，不能据此判定 Lunar Lake 无法执行自定义指令。

## 复现

在项目根目录的 PowerShell 中运行，不必事先激活环境：

```powershell
# 普通图，copied API，每项 3 次校验；应返回 exit 0
.\npunlock-probe\run.ps1

# 自定义链路阶段诊断：当前 ACT 发现失败，预期 exit 1
.\npunlock-probe\run.ps1 -Cases custom,toolchain

# 实验性共享会话测试；会覆盖相应 results/*.json
.\npunlock-probe\run.ps1 -Shared
```

每项测试在独立 Python 子进程中执行，有 60 秒外部期限；驱动编译/执行配置 20 秒期限。原始 XML、blob、NPZ 对照张量和 JSON/log 写入 `results`。默认共享测试以外的简单图不需要 MoviTools。

## 对 PyTorch NPU 后端的意义

本机实测支持先实现“PyTorch 静态子图 → IR → 驱动编译器 → Level Zero 执行”的小后端，复用现有编译器支持的算子。缓存编译结果、融合子图和可靠的会话复用将影响实际性能。npunlock 的自定义 SHAVE 路线需要 Lunar Lake 专门移植，可作为后续研究方向。

后续移植现已实现 NPU4000 自定义 SHAVE FP16/FP32 算术及真正的 intel_npu
PrivateUse1 设备 Tensor，支持常驻转换和 sum/mean。当前实现与复现步骤见
[工作区 Torch NPU 说明](../TORCH_NPU_README.md)；本文件中的早期探针结果仍作为历史记录保留。

这些测试没有验证 autograd、训练、GPU/NPU 跨设备内存共享、并行调度或加速比，也未实现 `torch.xpu` 的扩展。要研究 GPU+NPU 协作，应先建立可靠的 NPU 图执行路径，再测独立子图并行和传输/同步成本。

项目兼容边界说明：[npunlock PORTING.md](../npunlock/wiki/PORTING.md)。上游项目：[hsfzxjy/npunlock](https://github.com/hsfzxjy/npunlock)。
