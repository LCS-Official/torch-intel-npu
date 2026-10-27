# v0.2.0 Experimental Preview 发布验收

日期：2026-10-27。结果来自本机独立导出的源码目录；本版仍为实验预览，不是stable v1.0。
源码范围与升级说明见[发布说明](RELEASE_NOTES.md)。v0.1.0的原验收记录保留在[历史tag](https://github.com/LCS-Official/torch-intel-npu/blob/v0.1.0/RELEASE_READINESS.md)。

## 环境与构建

- Windows x64、Core Ultra 7 258V、Lunar Lake NPU4000 `8086:643e`，驱动32.0.100.5540、图编译器8.3。
- Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3；可选模型依赖safetensors0.8.0、Transformers5.18.0。
- MSVC14.44.35207、Windows SDK10.0.26100.0、CMake4.4.4、Ninja1.13.2。
- 在独立导出的源码目录运行build_npu.ps1，重新构建完整npunlock DLL、worker、session API和Torch PYD，两个原生mock回归通过；没有复制开发目录的旧宿主二进制。
- 复用已有Python环境和外部本地SmolLM2-135M-Instruct checkpoint，没有重复复制模型；本轮不等于第二台机器或全新Conda环境的安装验证。旧新环境验证见[INSTALLATION_REPORT.md](INSTALLATION_REPORT.md)。

## 离线检查

| 模式 | unittest记录 | 通过 | 跳过 | 失败/错误 |
|---|---:|---:|---:|---:|
| 独立源码，包含可选模型依赖 | 243 | 242 | 1 | 0/0 |
| 核心依赖模式 | 230 | 219 | 11 | 0/0 |
| 开发目录对照旧公开源码，单独补验 | 1 | 1 | 0 | 0/0 |
| 原生session失败与chain的mock夹具 | 2 | 2 | 0 | 0/0 |

前两行均包含根目录自动发现、32项Python API检查及6项导出器检查。独立源码不含嵌套的旧source-release目录，因此旧版Smol图比较单项跳过；在覆盖发布checkout前已对照旧公开源码单独通过。该检查比较FP16/W8的默认序列化图和CPU参考，不是低位整网质量认证。

核心依赖模式通过导入隔离主动屏蔽safetensors、Transformers、torchvision、OpenVINO，模拟这些可选包不可用；相应11项以明确原因跳过，没有伪装为通过，也没有重新创建精简Conda环境。测试全部从独立源码导入npunlock和项目模块。

本轮另补充细粒度GitHub令牌导出拒绝测试、GCC/MSVC两种session来源记录解析测试，并让模型相关测试在可选依赖缺失时正确跳过。CI和贡献指南改用自动发现，避免后续新测试遗漏。

复现命令：

```powershell
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
python -B -m unittest discover -s . -p 'test_torch_npu*.py'
python -B -m unittest discover -s npunlock/tests/python -p test_python_api.py
python -B tools/test_source_release.py
```

## 真实NPU发布回归

以下6个进程逐个运行，均使用独立源码构建的宿主组件。每个进程结束均检查context未失败、live buffers/缓存会话/标量缓冲/隔离字节归零；没有用CPU回退代替设备数值执行。

| 检查 | 内容 | NPU调用 | 结果 |
|---|---|---:|---|
| 安装冒烟 | 严格FP32尾块、FP16矩乘、Linear/Conv2d前后向及SGD、本地checkpoint、单层FP16 prefill/decode及旧KV保留 | 77 | 通过；未导入OpenVINO |
| 标准AMP＋AdamW | FP32主参数、AMSGrad/weight decay、溢出跳步及完整优化器状态对照CPU | 309 | 通过 |
| 分块Attention FP32 | 多query/key tile前向和重算Q/K/V梯度对照 | 221 | 通过 |
| NestedAttention FP16 | ragged SDPA、GQA/causal、native MHA权重、空子序列及CPU原有行为保留 | 529 | 通过 |
| 软标签损失FP16 | 73组类别权重、reduction、label smoothing、空输入、部分view与梯度配置 | 2,585 | 通过 |
| 30层FP16缓存重建 | 默认512MiB预算，两轮prefill/decode、64位置forward、再decode及输出重放 | 660 | 通过；216次图淘汰，结束健康 |
| 合计 | 本轮发布回归，不包含以前阶段报告的调用数 | 4,381 | 全部通过 |

30层测试每个检查点均确认缓存字节不超过预算、未额外保留被淘汰图；两次64位置forward以及旧KV再次decode的重放输出一致。该流程验证S45图族回收的回避措施，不能据此认定原生释放的根因已经修复。

构建与真机复现示例（需准备本地模型）：

```powershell
.\build_npu.ps1 -Python python
.\run_torch_npu.ps1 -Python python -Script tools/smoke_install.py -ScriptArguments @('--model','D:\models\SmolLM2-135M-Instruct')
.\run_torch_npu.ps1 -Python python -Script torch_npu_amp_standard_demo.py -ScriptArguments @('--case','adamw')
.\run_torch_npu.ps1 -Python python -Script torch_npu_flash_nested_demo.py -ScriptArguments @('--case','flash','--precision','float32')
.\run_torch_npu.ps1 -Python python -Script torch_npu_flash_nested_demo.py -ScriptArguments @('--case','nested-attention','--precision','float16')
.\run_torch_npu.ps1 -Python python -Script torch_npu_loss_extended_demo.py -ScriptArguments @('--case','soft','--precision','float16')
.\run_torch_npu.ps1 -Python python -Script npunlock-probe/research/smollm_cache_lifetime.py -ScriptArguments @('--layers','30','--budget','512','--cycles','2','--label','release-check','--model','D:\models\SmolLM2-135M-Instruct')
```

## 来源与发布边界

核对66份resident ELF、66份汇编及27份C源码的manifest哈希；核对MSVC记录的38份原生源/头文件、session DLL和Torch PYD指纹。此次源码构建的标识如下，构建产物不随源码发布：

| 项目 | SHA-256 |
|---|---|
| backend.cpp | `9c3035281a9e1f6547c2d145a63cda495c0b2478ba2691e999d9548154970fba` |
| MSVC session DLL | `ff0d1b7905d687912c8d629756b0f6efa5946aed9f4b4ca2e637af9e3fbb5788` |
| Torch PYD | `d850158ee1081b4ccc5ed680c1a5ebe97949916f6ee3e7a026f01495eb509365` |

最终源文件集与哈希以SOURCE_MANIFEST.json为准。包内保留代码、来源/许可、C/汇编/ELF及Markdown报告，不包含模型权重、环境、编译工具、宿主DLL/PYD、驱动生成图、计时JSON或原始数组。首次构建前可运行：

```powershell
python -B tools/export_source_release.py --verify .
```

构建与测试后的额外文件会使严格快照检查报告文件集变化，这是预期行为。此轮原始测试输出在形成报告后清理；历史研究audit需要重新生成对应数据。

## 云端CI与剩余限制

发布前重新读取[旧CI run](https://github.com/LCS-Official/torch-intel-npu/actions/runs/38208763066)的检查注释，明确为“The job was not started because your account is locked due to a billing issue.”，不是执行测试后的数值/源码失败。本地离线及真机检查不等于GitHub托管CI通过；本版新提交的实际云端状态以[Actions](https://github.com/LCS-Official/torch-intel-npu/actions)为准。账单限制需账号方/GitHub处理，代码发布不能解除它。

本轮没有重跑所有历史大容量/长时间压力实验、全部算子配置或低位整网质量校准，也没有在另一台设备/驱动验证。S45保留的16个W8校正质量失败及INT4/NF4整网未认证，均不因此次发布变为已解决。

仍不承诺通用CUDA运行时/PTX兼容、任意模型迁移、14GiB训练、稳定异步stream/event、分布式或GPU＋NPU协同。完整边界见[兼容性矩阵](COMPATIBILITY.md)及各阶段报告。
