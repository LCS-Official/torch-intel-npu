# S37：解除选择反向的2²⁴单平面元数据限制

本地开发版，2026-10-25，未同步GitHub。Windows x64、Core Ultra 7 258V / Lunar Lake NPU 8086:643e、驱动32.0.100.5540、图编译器8.3；Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3。

## 变更及软件边界

旧selection backward将count/start/per/width/base/span/flags/total这8个整数转换为FP32数值；为避免超过2²⁴后舍入而拒绝大单平面。现在保留小尺寸的旧selection-backward-f32路径，超过旧边界的包使用selection-backward-u32：header是原始UINT32字，梯度仍为FP32载体，索引仍是原始INT64双字。不是放宽FP32数值检查后继续使用不精确地址。

新增SELECTION_MODE=4复用原模式2的完整索引校验、顺序累加及赋值逻辑。字段各<=268,435,456，包长/跨度/flags/目标范围/行范围先检查；INT64索引高字非零或低字越界即设置错误状态，只有显式allow_padding接受-1。字段上界确保start+count和行×宽度不发生UINT32溢出。Python另检查几何整除、dtype存储、正尺寸和初值shape/dtype。非法包与恢复在真机验证。

共享选择反向的逻辑源/选择数改用dtype存储检查，FP16/FP32最多268,435,456元素；索引Tensor还须满足INT64的134,217,728元素上限。普通MaxPool1d/2d的单空间平面输出保护由16,777,216扩至134,217,728，且仍检查总配对输出存储及窗口几何。浮点极值输入改用dtype存储，长轴超过2²⁴时以64,512元素块读取，避免大连续源上的不对齐Slice编译。

宽选择路径每包最多21,504条梯度、64,512个目标值，保持每个目标的原梯度顺序。FP16仍在每次重复累加后舍入；非累加赋值保留负零，允许padding和非零初值。输出利用已有FP32目标写回；当前仍保留若干分块结果列表，完整梯度/索引/FP32转换不包含在单包预算内，不能当作总内存硬上限。

内核新增ELF 5,740字节及汇编，对选择家族重建后，原62枚ELF/汇编的manifest记录逐项核对相同。没有重建DLL/PYD、安装依赖、下载模型或引入OpenVINO runtime。更新必须同步Python派发、C、新ELF/汇编及manifest，源码/二进制指纹检查继续启用。

## 真机专项

每个用例FP32/FP16各一个独立进程，8个正式进程全部通过，共53,010次NPU提交。

| 用例 | 实际验证 | FP32 / FP16提交 | FP32 / FP16峰值设备buffer MiB | FP32 / FP16进程秒 |
|---|---|---:|---:|---:|
| packet | 原始奇数start/per/width/base均越过2²⁴，重复项Half舍入、赋值负零逐位；每精度11种非法包拒绝，padding及恢复 | 15 / 15 | 0.0 / 0.0 | 0.3 / 0.288 |
| pool | 单平面(1,1,1,16,777,219)，kernel(1,3)/stride(1,32768)，513输出；完整CPU前向/索引/输入梯度，末索引16,777,217 | 2908 / 4710 | 192.08 / 192.815 | 8.42 / 15.5 |
| extrema | 单平面16,777,219元素，max/min完整CPU输入梯度；重复极值及NaN的argmax/argmin首位置16,777,217 | 13464 / 18610 | 192.079 / 192.814 | 18.923 / 42.04 |
| dense | 16,798,727条梯度累加到3元素目标，含初值；跨包后保留逐步舍入和未选负零 | 6257 / 7031 | 193.723 / 225.764 | 14.371 / 20.645 |

池化及极值使用完整CPU原生PyTorch参考，rtol=0、atol=0、equal_nan=True；这是零容差数值相等，不是所有NaN payload的逐位认证。索引均为完整INT64相等。单平面池化仅513个稀疏窗口，不推导同规模密集重叠池化性能；源码接受大输出平面，不表示最大输出形状已经实测。

跨包dense不把CPU原生Half scatter_add的FP32累加策略当作参考，而按本后端既定的每次Half舍入规则独立顺序计算。预期FP32为[1.875,-0.0,5.0]、FP16为[0.75,-0.0,5.0]，保留原精度语义。未选目标的负零另逐位检查。该用例覆盖大per、总梯度数及跨源包的初值衔接；packet用例单独检验超过2²⁴的奇数start/base没有舍入。

峰值来自后端peak_buffer_bytes，不是进程RAM、任务管理器共享内存或整机峰值。时间包含编译、上传、CPU参考、完整回传和关闭，不作为热吞吐或设备能效结论。

## 既有路径回归

10个独立进程全部通过，每个精度5个；与8个专项合计18个正式进程、80,901次NPU提交。全部正常关闭且资源归零，未放宽原回归阈值。

| 用例 | 覆盖 | FP32 / FP16提交 |
|---|---|---:|
| pool_topk / edges | 原始位负零TopK梯度、末NaN池化、并列值、重叠Half累加、索引version | 188 / 275 |
| vision_sort_extended / pooling | 每精度21组3D/自适应池化、平均池化和Unpool | 347 / 491 |
| pool_capacity / planes | (8193,1,32,64)，完整4,194,816元素输出/索引及16,779,264元素输入梯度 | 10,623 / 13,742 |
| pool_capacity / training | 205通道bias+MaxPool+均值+Linear，两步SGD完整参数梯度及更新 | 320 / 480 |
| compat_extended / index | nonzero、BOOL索引、分离多轴广播及重复索引梯度 | 708 / 717 |

训练loss保持FP32 0.0176839251→0.0170685854、FP16 0.0176782925→0.0170133468；容差沿用原FP32 rtol=4e-5/atol=2e-6、FP16 rtol=.004/atol=.0005。它们是两步集成检查，不代表长期收敛。旧pool_capacity原始JSON本轮重新生成，结果归纳在此，不覆盖S36历史报告。

## 离线检查

最终130项根目录、32项Python API及5项发布工具检查通过，共167项。新增4项覆盖大平面原始位赋值与奇数索引、旧路径Half顺序/padding/初值、存储和几何错误在执行前拒绝，以及大平面规划/输出几何。更新旧容量测试及CI，避免旧的2²⁴拒绝契约继续错误地限制新入口。

## 复现

```powershell
.\run_torch_npu.ps1 -Script torch_npu_selection_capacity_demo.py -ScriptArguments @('--case','packet','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_selection_capacity_demo.py -ScriptArguments @('--case','pool','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_selection_capacity_demo.py -ScriptArguments @('--case','extrema','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_selection_capacity_demo.py -ScriptArguments @('--case','dense','--precision','float16')
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_selection_capacity
# 已有Movi工具链的开发工作区可重建，运行源码只需匹配manifest的ELF：
.\.conda\npunlock\python.exe npunlock-probe/research/build_selection_kernels.py
```

所有NPU进程串行。8个专项正常shutdown，context_failed=False，live/cached buffer、scalar、session、graph blob、factory及quarantine归零。

## 剩余限制

- 元数据由旧2²⁴限制改为当前存储边界，不提供跨buffer分页或超1GiB allocation。
- 源和目标同时很大时仍可能产生目标包×源包的重复扫描，兼容不代表高吞吐。
- sort/topk、Embedding、其他索引前向、3D/自适应前向、平均池化和插值仍有独立限制；共享反向可用不代表整条算子路径全面扩容。
- 未实测最大形状、完整大输出平面的密集池化训练、14GiB整网训练、长期FP16/AMP收敛或GPU/NPU协作。

## 最终指纹

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | `fb05ae861a31e55f7918efcb3e1dbffe84b5c7e9faa1eeaca128d145742b3849` |
| torch_npu_geometry.py | `d04a41604c33ad1d1ea61b14343c28ceb5fa7d8234a9458899a16392a027a14c` |
| torch_npu_capacity_demo.py | `4da008779b5211ae3d142c21081849a1a865469dd5065e90df7fa42a439118fe` |
| test_torch_npu_capacity.py | `978a769cd8efcbb63527231d0b48bb13a5366772e6d5d4acf6c94efa7c599484` |
| test_torch_npu_pool_capacity.py | `9150523e3ae10f555060a86375f55e4c312e1f525e2680013f9acda5500d64a6` |
| torch_npu_selection_capacity_demo.py | `07e7996e09849383084b5a360f17ac12c502c51b9ef0041af96b8ac943858de0` |
| test_torch_npu_selection_capacity.py | `d9a190a622d8ed6eb3e63a0a434cfa695101c29201ab6308dedee9992103f8a4` |
| npunlock/tests/fixtures/npu4000/resident_selection.c | `60a8922d9266c2c58e4a1892eedf8db1685f06e42719655e8edcfd76b4c1d2b7` |
| npunlock-probe/research/build_selection_kernels.py | `1230eaa1347305ff695d9b899b63c43c3b8ae73c5e9d6720797da33089c16dbd` |
| npunlock/tests/fixtures/npu4000/resident/selection-backward-u32.elf | `7e042cdb2b4855a5ab827ba702372d44e63392516ceb7d1de7c88f53a4d31de1` |
| npunlock/tests/fixtures/npu4000/resident/selection-backward-u32.s | `eada158ac7dc5105d34aef0c19494bef0a581411fe27822c4bb0d3a77401374f` |
| npunlock/tests/fixtures/npu4000/resident/manifest.json | `2eb24736db00ae8c1c923a7b019ccb46da6ba1c647453288ec47833659981b6f` |
| .github/workflows/offline.yml | `58ff0c1686c2130ceee5b1e1017d91341d7d97033e0754c69a27f8949808bc87` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `4fed86d100f9ca48ecda3e8d810ea2526400485a690465ba96e1cf92d7150e4a` |

## 清理及发布状态

全部进程退出后，清理18份正式真机原始JSON、15份编译/汇编/链接阶段JSON及15份worker响应，共48个文件、368,452字节。路径逐项解析为项目results内绝对路径，拒绝联接/符号链接、目录与保留文件；删除前检查正式通过状态及构建成功状态。清理前后510份选定源码、报告、内核和必要构建组件SHA-256相同，随后补充本段及空间台账。

63组ELF/汇编及25份resident源指纹已全部复核。新增C、5,740字节ELF、汇编、构建脚本/manifest、Python实现与检查、离线CI、双语说明和本报告保留。没有新增依赖、模型、图表、磁盘checkpoint或项目持久图缓存；DLL/PYD和C++/session C未修改，公开source-release未改，未推送GitHub。
