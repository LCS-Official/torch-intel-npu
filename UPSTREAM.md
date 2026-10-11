# npunlock 上游与本地改动

- 上游：[hsfzxjy/npunlock](https://github.com/hsfzxjy/npunlock)
- 本地基准提交：`bdf1c01d2caa2e772315a9b2a529981c2810e037`
- 许可证：Apache-2.0；原文保留在 `npunlock/LICENSE`。
- 本地修改尚未作为上游提交发布；本快照不声称是 npunlock 官方版本或已获上游合并。

主要改动：新增 `python/npunlock/npu4000.py` 的 NPU40XX ELF / SHAVE carrier 验证与替换、
FP32 传输、DMA 变换和 GEMM/RMSNorm/KV 内核接口；扩展 symbolic IR 和 packed 常量；
Windows session 的共享分配、command list/fence 复用、chain 提交及失败隔离；补充 ABI 声明、测试与移植文档。
外层 `torch_*`、PrivateUse1 `backend.cpp`、模型适配器、构建器和优化报告是本项目增加的集成层。

导出时自动生成：

- `provenance/npunlock-upstream.json`：基准提交、每个已导出上游文件的原始/当前 SHA-256、added / modified / unchanged / line_endings_only 状态，以及排除文件。
- `provenance/npunlock-changes.patch`：相对于上述提交的已导出文本改动，包含新文本文件；不含 ELF 二进制。
- `SOURCE_MANIFEST.json`：整个发布快照的逐文件哈希，覆盖自有 ELF 内核、汇编和报告。

文本 patch 不是整个工作区的补丁：外层 Torch 后端通过完整源码提供；图 blob、上游发布工作流和预编译本机扩展未导出。
原始工作区及其子仓库均未重置、提交或清理。
