# v0.3.0-alpha.1 Experimental Preview 发布验收

日期：2026-10-30。累计纳入v0.2.0之后的S46–S60，定位仍为alpha预览。当前版本、升级与限制见[发布说明](RELEASE_NOTES.md)。v0.2.0的源码构建与4,381次发布回归保留在[历史验收](https://github.com/LCS-Official/torch-intel-npu/blob/v0.2.0/RELEASE_READINESS.md)。

## 当前验收范围

- 审查累计分支与v0.2.0的差异，纳入独立分类损失修复和全部研究分支；保留提交历史。
- 核对四个生产Python模块与已测开发源码一致；C/C++、原生构建脚本、requirements及SHAVE内核没有变化。
- 230份NPU4000 fixture与v0.2.0相同；原FP16/W8的六份默认图指纹不变。
- 从累计公开checkout导入后端，复用有哈希／Torch版本保护的匹配宿主组件；不把这次验收描述为重新源码构建或另一台机器验证。

环境仍为Windows x64／Core Ultra 7 258V／NPU8086:643e／驱动32.0.100.5540／编译器8.3，Python3.12、Torch2.14.1+cpu、NumPy2.5.3。没有新依赖、模型下载或环境复制。

## 离线与实际NPU结果

| 检查 | 结果 |
|---|---|
| 累计公开源码自动发现 | 349项，348通过、1项既有条件跳过 |
| Python API与导出器 | 32+6通过 |
| 离线合计 | 387项检查，386通过、1跳过、0失败 |
| 安装与FP16模型冒烟 | 77次NPU调用，严格FP32／FP16、Linear/Conv2d反向及SGD、单层KV通过 |
| 分类损失source versions | FP16/FP32各122项，无CPU/NPU接受／拒绝差异；14,539+10,240次NPU调用 |
| FP16/W8/INT4/NF4单层 | 各forward／prefill／decode，12项局部数值检查、44次NPU调用通过 |
| 本轮NPU合计 | 7个进程、24,900次调用，全部通过并健康关闭 |

四种格式每个进程结束均检查context_failed=False和缓存／buffer归零；冒烟确认未导入OpenVINO。没有重新运行所有历史算子、14GiB训练、长时间压力、另一台硬件或低位整网校准；本轮不能替代这些认证。

## 整网质量与升级

完整30层结果沿用[S60明确协议](npunlock-probe/research/results/smollm-lowbit-coverage/REPORT.md)：257个EOS边界内位置，FP16 257/257，普通RTN W8 96/257，INT4/NF4均0/257。局部执行通过与原模型质量通过分开；没有新默认、宽松门槛或低位质量修复声明。

NF4完整栈热路径验收显式使用1GiB图预算，生产默认仍512MiB。四位源常量压缩不等于进程／设备内存缩小或性能提升。当前仅承诺已测冻结SmolLM2入口；CUDA运行时、通用低位Tensor／训练、任意模型迁移、长上下文、稳定异步／分布式及GPU＋NPU协同仍有边界。

从v0.2.0升级时同步完整源码并重启Python，匹配的宿主DLL/PYD和内核可复用；首次安装或v0.1.0升级仍按[安装说明](INSTALL.md)构建。此前tag与分支不改写。

## 源包与云端检查

发布清单覆盖源码、许可／来源、配套内核及Markdown报告，排除临时宿主副本、模型、环境、工具链、原始JSON／日志及驱动图。发布前严格校验SOURCE_MANIFEST.json，并检查实际Git归档的文件集和SHA；首次构建前可运行：

```powershell
python -B tools/export_source_release.py --verify .
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
python -B -m unittest discover -s . -p 'test_torch_npu*.py'
python -B -m unittest discover -s npunlock/tests/python -p test_python_api.py
python -B tools/test_source_release.py
```

GitHub累计PR、main及tag的最终CI状态以[Actions](https://github.com/LCS-Official/torch-intel-npu/actions)和发布页对应提交为准。tag工作流执行严格源包清单检查；云端只安装核心依赖，模型测试会按可选依赖明确跳过，不具有NPU设备。

详细审查、逐进程记录身份及清理结果见[本轮累计发布报告](npunlock-probe/research/results/release-v0-3-alpha-1/REPORT.md)。
