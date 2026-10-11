# GitHub 首发准备与验收

日期：2026-10-11。目标为 **v0.1.0 Experimental Preview 源码首发**，不是 stable v1.0。
本轮补齐仓库材料、验证真实 Git 往返与非 Quick 回归；最终结果在本报告后续章节记录。

## 发布内容

公开仓库根目录应是 `source-release/` **内部的内容**，不能上传整个开发工作区，也不要在 GitHub 再套一层该目录。
根 README、INSTALL、COMPATIBILITY、LICENSE/NOTICE、参考来源及 requirements 已具备。
本轮新增 [发布说明](RELEASE_NOTES.md)、[贡献说明](CONTRIBUTING.md)、问题/PR 模板与
[离线 CI](.github/workflows/offline.yml)。

CI 的发布快照校验仅忽略根 `.git` 元数据，仍拒绝其他额外文件、缺失文件和哈希变化。
`.gitattributes` 禁用自动换行转换以保持 manifest 字节；根 `.github` 纳入导出，上游旧发布 workflow 仍排除。
旧 `pip-freeze.txt` 含本机 `file://` 路径，连同旧完整 Conda 冻结清单只留在开发目录；当前安装使用 requirements。
GPU 对照脚本改为显式传入 `-XpuPython`，不再默认当前用户的环境。

## 本轮结果

在同一 Windows/NPU/驱动与原 Python 3.12.14 环境串行执行，未改动计算内核或容差：

| 非 Quick 检查 | 组数 / 状态 | NPU 调用 | 训练 / 循环 |
|---|---|---:|---|
| D：设备 | 33，`passed` | 19,476 | 主要链式组每组 1,000 轮，其他按正式入口；43 个明确拒绝场景 |
| T：训练 | 11，`passed` | 10,544 | FP16/FP32 的 MLP、Conv、TinyResNet 各 15 步 SGD |
| M：模型 | 9，`passed_with_limitations` | 31,938 | BatchNorm/池化/分块梯度，以及两种 dtype 的 WideMLP/SmallResNet 各 15 步 |

合计 **53 组、61,958 次调用**，命令均退出 0，全部组通过既有检查；各进程 shutdown 后资源缓存、
会话、活跃和隔离分配归零，context 健康。所有网络热训练步的显式 payload copy 与新增图编译均为 0；
计数不包含驱动 DMA 或训练区间之外的参考下载。各脚本约 120 秒的 faulthandler 栈转储为诊断，非外层超时。
本轮使用的后端/原生文件指纹与兼容性矩阵相同；训练脚本使用其“报告文字修正后”的指纹。

FP32 SmallResNet loss：`0.2396990061 → 1.751627963e-7`；FP16：`0.2580566406 → 2.741813660e-6`。
FP16 始终逐步重建同权重 CPU 参考；逐项原容差通过，不认证独立训练轨迹。
其每步梯度相对 L2 误差（百分比）如下，必须与 loss 接近零的背景一起阅读：

```text
step 0..14:
0.103172, 0.147237, 0.162798, 0.104490, 0.116797,
0.114282, 0.744951, 4.944776, 0.243184, 4.206466,
7.370763, 11.691692, 9.104773, 10.511289, 22.648105
```

最低梯度余弦相似度 `0.97405738`。只展示前 3 步的约 0.1%–0.16% 会低估后续差异，
因此 M 保留“有限制通过”，不升级为严格 FP16 训练认证。
C 的 5 个独立兼容性进程沿用上一轮已完成的记录，本轮未重复执行；完整 LLM/低位质量与性能也未重测。

离线检查已通过 **45 项**：8 项几何/低位、32 项 npunlock Python API、5 项源码导出检查。
这些命令在真实 Git 提交后重新克隆的源码内执行，导入位置也已核对；不是从开发目录借用模块。
验证提交使用临时身份，仅作本地测试，不会推送。最终公开提交使用 `LCS-Official` 及 GitHub noreply 地址。

Git 往返在 `core.autocrlf=true` 下完成，索引文件和克隆文件的 manifest 校验通过，
证明 `.gitattributes` 能保持内核及源码字节；根 `.git` 不再导致误拒绝。
高置信度凭据扫描未发现私钥或访问令牌；最终导出还将检查用户提供的实际令牌未进入源码。
少数历史报告/研究工具保留原开发路径；通用安装入口不依赖这些路径，开发清理脚本仍保留固定边界保护。

本轮 6 份硬件 JSON/日志共 40,649 字节及两份 Git 验证副本共 17,530,146 字节已删除。
保留本报告中的结果、代码和最终干净源码快照；未安装新依赖或下载新模型。

## 真正发布时的最后操作

1. 用户已指定公开账号 `LCS-Official`；目标仓库 `torch-intel-npu`，令牌所属账号与名称可用性已核对。
2. 把已验收快照内容初始化为仓库首个提交，检查 staged 文件包含 `.github`、许可证、内核与 manifest。
3. 推送后运行离线 CI，确认云端首次实际执行通过；本机执行同样命令不等于云端已经通过。
4. 创建版本 tag 和 **pre-release**，正文使用 RELEASE_NOTES；确认发布 commit 与该次 manifest 对应。

本文记录首次推送前的本地验收。云端 CI、tag/release 的实际状态以
[GitHub Actions](https://github.com/LCS-Official/torch-intel-npu/actions) 和
[GitHub Releases](https://github.com/LCS-Official/torch-intel-npu/releases) 为准。
没有 wheel、跨设备覆盖和全功能训练并不阻碍明确标注范围的实验版首发；
稳定 v1.0 仍需更广的设备/驱动/模型验证、独立 FP16 训练轨迹及长时间稳定性检查。
