# 参与开发与报告问题

这是面向 Windows NPU4000 的实验后端。提交前先核对 [安装说明](INSTALL.md) 和
[兼容性矩阵](COMPATIBILITY.md)，区分普通 `intel_npu` Tensor 与冻结 DPU 模型入口。

## 运行离线检查

安装核心 requirements 后，在源码根目录运行：

```powershell
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B -m unittest discover -s . -p 'test_torch_npu*.py'
python -B -m unittest discover -s npunlock/tests/python -p test_python_api.py
python -B tools/test_source_release.py
```

工作流自动发现所有根目录test_torch_npu*.py，再运行npunlock Python API和源码导出检查。
核心检查不需要NPU、模型、MoviTools或OpenVINO；需要safetensors的模型测试在依赖缺失时明确跳过。
安装requirements-models.txt后可扩展可选测试覆盖；通过与跳过数量见[本版验收](RELEASE_READINESS.md)。
GitHub的[离线工作流](.github/workflows/offline.yml)在Windows/Python3.12运行这些入口；云端通过不能代替真实NPU验证。

## 分支、提交与历史

按一个修复或实验方向创建分支；每个完整、可解释的改动单独commit，记录验证结果，并在每轮工作结束时push。
通过验收后合入main，版本里程碑使用tag和Release。历史依赖commit/tag，不为每个版本复制整个目录。
未合并但仍需回忆的实验应保留分支或实验tag；生产默认只纳入通过约定数值/兼容性门槛的方案。
公开仓库只保存精简源码、自有内核伴随文件、测试和报告；模型、环境、SDK、blob和原始数组继续排除。

## 改动如何验收

- 新算子或梯度：给出 shape/dtype/参数边界，与 CPU 或独立参考比较；保留原数值容差，检查未支持输入明确报错。
- 内核或运行时优化：用相同输入交替对照，区分编译、上传/下载、热执行；保存数值、资源回收与性能摘要。
- 硬件问题：报告 Windows、CPU/NPU ID、驱动、图编译器、Python/Torch 版本以及完整复现命令。
- 修改 SHAVE：同步对应源码、ELF/汇编、manifest 与来源声明；工具链要求见安装说明。
- 许可与来源：直接改编保留上游许可、固定版本及改编位置，更新 REFERENCES/NOTICE，不能仅写“参考”。

真实硬件检查使用兼容性矩阵中的 D/T/C/M 入口，串行执行。涉及的检查需要复测；
无需为纯文档改动重跑全部硬件。普通设备算术不应悄悄退到 CPU，也不应为了通过检查降低精度。

## 提交内容

提交源码、必要的自有内核伴随文件和文字报告。环境、模型、SDK、宿主 DLL/PYD、编译图 blob、
缓存、原始数组与日志不属于公开源码包。历史研究脚本可能依赖原开发目录或已清理的实验输入，
不是通用安装 API；清理脚本保留原工作区边界保护，换路径时需先审阅，不能直接删除保护条件。

`SOURCE_MANIFEST.json` 是特定发布快照的字节记录，普通 PR 无需为了修改代码而伪造旧快照。
CI 在 tag 或手动运行时执行严格校验，普通分支/PR 执行离线测试。
维护者发布前从保留上游 Git 元数据的开发工作区重新导出；导出器不是通用 Python 打包工具。
`.gitattributes` 保留原字节以避免 Windows 自动换行转换破坏 manifest，不手工更改已有文件换行。

Bug、支持请求和 PR 的模板位于 [.github](.github)。分享复现时仅提供必要数据，移除私人数据和凭据。
