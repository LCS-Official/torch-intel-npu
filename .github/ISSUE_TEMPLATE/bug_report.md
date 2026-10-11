---
name: Bug report / 问题报告
about: 可复现的安装、数值、崩溃或设备问题
title: ''
labels: ''
assignees: ''
---

请先核对 INSTALL.md 和 COMPATIBILITY.md。未支持的算子/参数请使用功能请求模板。

### 环境

- 项目版本或 commit：
- Windows 版本、CPU、NPU 硬件 ID：
- NPU 驱动 / 图编译器版本：
- Python / Torch / NumPy / 编译器版本：
- 普通 intel_npu Tensor 或冻结 DPU 适配器：

### 最小复现

提供完整命令、精度、shape、stride、随机种子和短代码；训练问题请包含 backward 与优化器设置。

### 预期与实际

数值问题请附 CPU 参考、容差及最大误差；速度问题请区分首次编译、缓存启动与热运行。

### 日志

附最早的异常与退出码，必要时提供 npu.stats()。分享前移除个人数据、令牌和不便公开的路径；无需上传模型或编译 blob。
