# INT64存储与默认BatchNorm迁移验收

日期：2026-10-16。本地开发版，尚未推送GitHub。

## 本轮实现

- C++设备存储新增INT64，保留4MiB单allocation上限；INT64最多524,288元素、rank 0–4。FP16/FP32上限不变。
- INT64支持同dtype同步上传/下载、连续NPU复制/clone、fill/zero和单元素item。NPU内复制/填充使用raw-word DMA。
- fill接受范围内Python int或有限整数值float；不提供小数截断、整数/浮点cast、通用整数算术或NPU非连续复制。
- rank-0 INT64 counter.add_(Python int, alpha=1)由SHAVE unsigned32低字加法、高字加法和进位实现，模2^64回绕。
  F32载体仅传输两组原始位；未把INT64数值转成FP32。计数操作可在grad模式执行，整数不参与autograd。
- 默认BatchNorm2d的num_batches_tracked随模型迁移至NPU；使用标准nn模块，未替换forward、修改torch.cuda或使用CPU张量算术回退。

## SHAVE布局诊断

初版采用普通Power载体时，本机编译器生成8次调用，每次仅一个32位字，无法完成跨字进位；数值检查检测到失败。
修复为ReferenceSW单tile完整双字载体；加载前验证一组range/一次invocation、三个完整双字MemRef。
最终内核代码816字节，载体代码窗口21216字节；仅替换受指纹校验的代码窗口。
没有改写驱动任务指针、DMA地址或同步协议。编译器若再次拆分双字，离线/运行时契约会明确拒绝。

## 环境与验证口径

- Windows x64 / Core Ultra 7 258V / NPU 8086:643e / API driver_version=1005540。
- Python 3.12.14 / Torch 2.14.1+cpu / 图编译器8.3；现有.conda/npunlock，未安装依赖或下载模型。
- 重建PyTorch C++扩展与新增SHAVE计数内核；已有浮点SHAVE ELF和session DLL未改动。
- 整数用torch.equal/item做精确比较；没有用浮点相对容差接受大整数误差。
- 浮点CPU参考1线程、关闭oneDNN、随机种子47；FP16 rtol=0.005/atol=0.0008，FP32 rtol=0.0001/atol=0.00001。
  每个模型有相同初始权重的独立CPU训练轨迹，不按NPU结果重设CPU参数。

## 整数检查

- 存储往返：0、±1、2^24+1、2^32−1、±(2^53+1)、INT64_MIN/MAX，以及含FP32 NaN/Inf字模式的大整数。
- 五种rank、CPU非连续输入显式pack、工厂/复制/clone/zero、524,288元素最大容量通过逐位比较。
- 11对进位/负数/符号回绕/NaN字模式加法精确通过；从2^53起连续20次加1得到2^53+20。
- 每次计数更新为一次SHAVE加法加一次NPU DMA写回；14个非法形状/dtype/转换/参数/复制场景在没有新增图提交时拒绝。

## 默认BatchNorm模型训练、恢复与状态

Conv2d(1→2,3×3)、BatchNorm2d(2)、ReLU、全局池化、Linear(2→2)；输入(2,1,4,4)，MSE，SGD lr=0.01/momentum=0.8/foreach=False/fused=False。
六组均使用默认track_running_stats=True。每组3步训练后，显式CPU checkpoint保存参数、running mean/variance、INT64计数与优化器动量，
weights_only=True加载到新NPU模型，继续第4步；输出、loss、输入/参数梯度、参数、动量和全部buffers与CPU对照通过。
eval使用running buffers且计数不变；reset_running_stats恢复mean=0、variance=1、counter=0。

| 精度 | momentum | affine | 四步NPU loss | 各步显式host copies | 其中首次标量上传 | NPU调用 |
|---|---|---|---|---|---|---:|
| torch.float16 | 0.1 | True | 0.515625 → 0.4921875 → 0.45336914 → 0.40844727 | [9, 1, 0, 0] | [9, 1, 0, 0] | 1120 |
| torch.float16 | None | True | 0.17700195 → 0.17456055 → 0.16833496 → 0.15881348 | [3, 1, 3, 3] | [2, 0, 2, 2] | 1120 |
| torch.float16 | 0.1 | False | 0.21069336 → 0.20397949 → 0.19165039 → 0.17834473 | [0, 0, 0, 0] | [0, 0, 0, 0] | 860 |
| torch.float32 | 0.1 | True | 0.59385312 → 0.57675636 → 0.55108279 → 0.5167926 | [0, 0, 0, 0] | [0, 0, 0, 0] | 604 |
| torch.float32 | None | True | 0.43701142 → 0.41526076 → 0.38644147 → 0.35499245 | [1, 1, 1, 1] | [0, 0, 0, 0] | 604 |
| torch.float32 | 0.1 | False | 0.56851858 → 0.52251273 → 0.48031411 → 0.44213909 | [0, 0, 0, 0] | [0, 0, 0, 0] | 488 |

复制区间只包含NPU前向/反向/step，排除输入准备、验证下载和checkpoint搬运。
固定momentum=0.1第三/第四步的显式host payload copies为0。momentum=None按PyTorch Python控制流每次前向读取8字节计数，
且1/步数系数变化会首次上传新标量；它不是零拷贝模式。统计不包含驱动内部DMA，本轮不测吞吐或性能收益。

## 最终验收与回归

- 新增9组真实NPU检查通过：4,876次调用；新增3项离线单任务/双字载体检查与5项源码导出回归通过。
- 普通迁移套件：27组功能通过、9组明确未支持，共6,532次NPU调用（上一轮为25/11）。
- 原有设备33组、训练11组、兼容性5组通过：7,128次调用。
- 最终验收总计：18,536次真实NPU调用，不含开发诊断与失败重跑。
- 全部进程无context failure，各检查live buffer无增量；shutdown后live/cache/scalar/quarantine/session归零。

## 尚未支持与下一步

INT64 target可以存储和复制，但CrossEntropyLoss仍缺稳定log_softmax与NLL前后向；本轮没有打通分类训练。
INT64并非通用整数计算后端，INT8量化推理与该存储能力是不同接口。BatchNorm仍受原4D/通道/样本数/容量契约限制。
AdamW、LayerNorm/GELU/softmax/Embedding、batched matmul/SDPA、Uint8原始NPU序列化仍明确拒绝。
没有认证CUDA运行时/扩展、AMP、多流、GPU+NPU协同、其他设备/驱动或长时间训练。下一步优先实现分类loss。

## 复现

```powershell
.\build_torch_npu_device.ps1
.\run_torch_npu.ps1 -Script .\torch_npu_batchnorm_demo.py -ScriptArguments "--case","integer"
.\run_torch_npu.ps1 -Script .\torch_npu_batchnorm_demo.py -ScriptArguments "--case","models"
.\run_torch_npu.ps1 -Script .\torch_npu_migration_demo.py
.\run_torch_npu.ps1 -Device -Quick
.\run_torch_npu.ps1 -Training -Quick
.\run_torch_npu.ps1 -Script .\torch_npu_compat_demo.py
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
.\.conda\npunlock\python.exe -B test_torch_npu_int64.py
.\.conda\npunlock\python.exe -B tools/test_source_release.py
.\.conda\npunlock\python.exe -B npunlock-probe/research/int64_batchnorm_report.py
```

通常使用随源码提供的已校验ELF；修改计数C源码后，先用已有Movi环境执行build_integer_kernel.py，或完整build_resident_kernels.py。
按用户约定清理原始JSON/日志后，重算审计需先重新执行验收生成数据。

## 最终代码指纹（SHA-256）

```text
torch_intel_npu.py
96dfcd7983b4bbc1b62a19d7ef3da3cb7ec85038e770ca111fa9dedca0d88be6
torch_npu_batchnorm_demo.py
fb3e17f78d116c1f333f74de218909332aa611c5147caae14d207781c6efcbd2
npunlock-probe/torch-device/backend.cpp
006cad067bf8c16782be59b58b9b5d1dfd34244c99329a38854273dd39e6a924
npunlock-probe/torch-device/_torch_intel_npu.pyd
b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e
npunlock/python/npunlock/npu4000.py
cbd3cec9629e96e0e92fe4317c4fdf28a3c42a0b1fa9fa71985618fa0e8d5dcc
npunlock/tests/fixtures/npu4000/resident_integer.c
58759f1c6ebbe76dc7d1e27f2a908710b5f22da1aa0441ca97f55c0f38bc6d65
npunlock/tests/fixtures/npu4000/resident/int64-add.elf
df72c34aa17e13697c83fbf05adcf434ea5f3f895f378add28130dd7e465774d
npunlock-probe/session-runtime/npunlock_session.dll
0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9
```

## 交付与清理

验收后按用户约定清理32份临时JSON/日志、Movi编译响应和一次性布局诊断脚本，
共114,468字节（约111.79KiB）。保留实现、C/SHAVE源码、构建脚本、测试、当前内核/扩展与Markdown报告。
没有安装新依赖、下载模型、生成图片或启用持久图缓存；公开GitHub快照保持不变。
