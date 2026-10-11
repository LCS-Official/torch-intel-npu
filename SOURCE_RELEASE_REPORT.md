# 第一项发布准备：独立精简源码与来源整理

日期：2026-10-11。产物为工作区下的 `source-release/`；本次没有创建 GitHub 仓库、推送、打 tag 或发布 release。
当前定位仍是 v0.1.0 Experimental Preview 候选源码，不升级为 stable v1.0。

后续已完成新 Conda 环境的依赖安装、宿主组件源码构建与真实 NPU 冒烟验证，见
[INSTALLATION_REPORT.md](INSTALLATION_REPORT.md)。下文保留第一阶段当时的结果和待办状态。
当前发布准备与剩余操作以 [RELEASE_READINESS.md](RELEASE_READINESS.md) 为准。

## 本轮交付

- 保留原目录布局的独立源码快照：外层 Torch 后端、npunlock 本地扩展、PrivateUse1 C++、SHAVE C/汇编/自有 ELF、构建脚本、优化研究源码、58 份实验 Markdown 报告。
- Apache-2.0 主许可证、`NOTICE`、第三方许可全文和逐项改编说明；新增内容采用与 npunlock 一致的主许可证，原第三方许可不变。
- `REFERENCES.md` 与 12 项 `references.bib`：区分直接改编、ABI/算法参考、运行依赖、比较工具和验证模型。
- 可重复导出器与逐文件 SHA-256；明确拒绝覆盖现有目录，限制导出位置，排除环境、模型、SDK/工具链和本机原生扩展。
- 固定 npunlock 基准 `bdf1c01d2caa2e772315a9b2a529981c2810e037`；自动记录上游文件原始/当前哈希和文本修改 patch。

源码快照约 6.1 MiB，不是整个数 GiB 开发工作区的复制。原开发环境及模型保留供继续优化。
没有压缩包、图像、原始数组、模型权重、图磁盘缓存或成套外部编译器；小型自有 ELF 是与源码配套的内核文件。

## 验证

| 项目 | 结果与范围 |
|---|---|
| 独立源码目录中的离线回归 | 44 个测试通过：卷积几何 3、lowbit/校准 5、上游 symbolic/serialization/native mock 32、导出器 4；确认实际导入导出目录的源码 |
| Python 语法 | 270 个 Python 文件通过 `compile()`，未产生 pyc |
| 内核构建 provenance | 8 份现有 manifest，共 86 项源码/ELF/汇编 SHA-256 检查通过；没有重新编译或改动计算内核 |
| 上游文本 patch | `git apply --check` 与实际 apply 通过；70 个新增/修改文本文件与当前版本按 LF 归一化后逐项一致 |
| 参考资料 | 原 37 个固定 NPU compiler/ELF/driver 来源文件哈希通过；本轮另核对 12 个上游许可/源码 URL 并记录 SHA-256 |
| 复制一致性 | 逐文件校验原文件字节与导出副本；生成的 patch/上游状态单独计入快照 manifest |
| 基础材料检查 | README、来源说明、第三方说明的相对链接有效；核心模块未发现静态 `openvino` runtime 导入 |
| 导出边界 | 不含嵌套 `.git`、驱动图 `.blob`、模型权重、DLL/PYD/EXE、环境和原始实验结果；选定文本未发现高置信 token/私钥或明文口令赋值 |

首次独立测试发现一个真实整理问题：`test_python_api` 的 native blob 保存测试默认已有 `build` 目录。
已增加测试自身的目录创建；不改变 `Program.save()` 或运行时语义。
导出器还针对未以换行结束的文件补齐标准 patch 标记，并增加回归，避免生成不可应用的文本补丁。

最终目录的精确文件数、字节数与全部哈希以 `SOURCE_MANIFEST.json` 为准。
在尚未产生新文件的快照中可运行：

```powershell
python -B tools/export_source_release.py --verify .
```

## Reference 的实际关系

- **npunlock 是直接基础**，需要置于主要 acknowledgements/reference，不能只写“借鉴思路”。本地 NPU4000 和 Torch 层的边界记录在 `UPSTREAM.md`。
- **OpenVINO 并非运行时依赖**。但 NF4 码表直接来自其公开实现，且该文件标明 bitsandbytes 来源，两层版权已保留。
- **Intel NPU compiler / ELF / Level Zero / 扩展**是明确的 ABI 与格式参考。firmware 头文件的 MIT SPDX 和仓库根 Apache-2.0 不同，分别记录。
- **FP16 与 GPTQ 是直接改编来源**：已保留完整许可、固定提交和改编位置；不能只放网页链接而忽略许可。
- **llama.cpp、PyTorch CPU reduction、OpenVINO/Intel acceleration library 对照**属于设计或实验参考；未获收益的候选没有写成生产优化贡献。
- **SmolLM2 / Transformers**属于验证资源，不把模型权重作为本项目自有产物发布。

对于过去仅记录 main/master 的调研链接，明确承认无法从现存记录恢复准确历史提交。
本轮验证的固定版本另行标注，不冒充原先采用版本。

## 第一阶段当时未完成的项目

尚未做空白环境构建/安装验证、完整硬件发布回归或正式版本打包。
当前构建脚本仍有 `.conda/npunlock`、Python 3.12 和本地编译器路径假设，后续应集中处理这一项。
没有在本轮重新跑性能 benchmark，因此沿用报告中的硬件、shape、计时口径与质量限制。

驱动生成的图 blob 可能包含 Intel 预编译内核文本，未随公开源码目录复制。
完整 `test_npu4000.py` / `test_porting_probe.py` 夹具回归需要本地另备图数据；这 44 项测试不等同于完整设备测试。
模型和历史原始实验数据未打包，研究目录中的部分 before/after audit 需要这些外部材料。
核心优化 Python、C 与 ELF 没有为此次整理而重写；运行时性能结论不因源码目录整理而扩大。
