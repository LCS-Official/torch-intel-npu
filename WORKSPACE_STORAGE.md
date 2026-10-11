# 工作区空间记录

2026-10-11 GitHub 首发验收：非 Quick 设备/训练/模型共53组、61,958次NPU调用，Git克隆内45项离线测试通过。
删除本轮6份JSON/日志（40,649字节）及两份临时Git验证副本（17,530,146字节）；
仅保留约6.1MiB的最终源码快照与其发布Git元数据，记录见 [RELEASE_READINESS.md](RELEASE_READINESS.md)。
未安装新依赖、下载模型或创建新图缓存，原运行环境与内核未替换。

2026-10-11 发布兼容性基线：设备/训练/独立兼容性/小模型检查完成，合计58组、14,154次NPU调用；
小模型FP16保留训练轨迹限制，结果与指纹见 [COMPATIBILITY.md](COMPATIBILITY.md)。
文字记录验收后清理本轮17个临时JSON/日志，共71,904字节，保留历史Markdown报告；
未新增依赖、模型或磁盘图缓存，继续保留原环境、工具链与当前内核/宿主二进制。

2026-10-11 源码安装验证：新建 Conda 环境，从导出源码构建宿主组件，完成真实 NPU 运算、训练与单层模型检查。
验收后删除 `.conda/release-validation`（逻辑约944.60MiB）和 `source-release-validation`（逻辑约14.95MiB）；
保留 requirements、构建/运行入口、[安装说明](INSTALL.md)、[验证报告](INSTALLATION_REPORT.md)与约6.1MiB的干净源码快照。
未替换原 `.conda/npunlock` 环境和当前 DLL/PYD，未新增模型下载；不清理与其他环境共享的 Conda 包缓存。

2026-10-10，按用户要求，保留内核与优化相关源码、运行/重建所需文件及文字报告，清理可重建实验输出。

2026-10-11 延续同一策略：首token优化产生的113个临时文件约596MiB，验收后清理；保留新增研究源码与文字报告。
新的免解压缓存以额外空间换取启动速度，仍默认关闭且上限512MiB。开发验证后删除；实际使用若要保留启动收益，需要跨进程保留缓存。
本轮没有在C盘新建图缓存或安装新依赖。最新结果见 [冷首token报告](npunlock-probe/research/results/startup-first-token/REPORT.md)。

2026-10-11 后续提前命中优化验收后，删除143个临时文件，共760,057,299字节（724.85MiB），无失败项。
包括旧/新及交替对照图缓存、边界检查图、hidden/logits数组和原始JSON；保留研究源码、before文字源码及[跳过IR重建报告](npunlock-probe/research/results/startup-ir-cache/REPORT.md)。
清理后原生文件校验、严格FP32算术、FP16矩乘和一层prefill/decode冒烟检查通过（14次NPU调用，资源回收正常）。

| 项目 | 清理前 GiB | 清理后 GiB |
|---|---:|---:|
| E盘项目逻辑大小 | 20.63 | 4.57 |
| E盘项目实际占用（硬链接去重） | 15.33 | 3.83 |
| E盘可用空间 | 0.48 | 11.99 |

本轮删除 **9938个文件**，没有失败项。E盘项目实际占用减少约 **11.50GiB**。
逻辑大小与实际占用不同，因为历史文件曾做NTFS透明压缩，Conda部分文件采用硬链接。

C盘删除本轮60份SmolLM图缓存（322.15MiB）及npunlock旧ETL跟踪（8MiB），合计约330.15MiB。
没有清理其他项目、用户Conda环境或通用系统缓存。

## 已清理

- 全部实验目录中的NPZ/NPY张量、logits、KV与校准输入输出。
- 生成的IR/XML、blob、编译图、设备profile、图片、运行日志和原始计时JSON。
- 历史实验复制的DLL/PYD/ELF与无扩展名源码指纹快照。
- 已安装工具对应的旧下载包（EXE/WHL/ZIP），不再保留安装包副本。
- 本轮移动到C盘的图缓存以及旧npunlock调试跟踪。

保留129份Markdown实验报告，保留研究脚本、C/C++/汇编内核源码、补丁和文字说明。
报告保存清理前已经完成验收的结论；历史原始数据链接可能失效，离线audit不能直接重算已删除的数据。
需要复测时重新执行对应实验，产生新的参考和测量值。

## 仍保留的必要内容

- 当前SHAVE ELF及其源码/manifest、npunlock原生库、session DLL、PyTorch设备PYD和构建清单。
- 当前运行代码、测试代码、研究脚本、参考开源编译器源码与构建工具链。
- 项目NPU/OpenVINO对照环境、GPU模型依赖和约0.25GiB的测试模型输入。
- Conda已解包包缓存中存在与环境共享的硬链接；不把其逻辑大小误当成可直接释放空间。

清理后实际运行 **严格FP32加法、FP16矩乘、单层SmolLM prefill/decode**，共14次NPU调用通过；
原生文件指纹验证正常，最终live/cache/quarantine/session全部归零，无context failure。

## 后续约定

每轮交付以Markdown报告为主，把主要数值及必要样本写进报告后清理大型实验产物。
图缓存默认关闭；若使用`NPUNLOCK_GRAPH_CACHE`，会以磁盘空间换取下次启动速度，删除后首次需要重编译。
用户主动启用的缓存目录需自行选择；项目提供的默认缓存目录已纳入以下清理入口。

所有本项目实验退出后可预览并清理：

```powershell
.\npunlock-probe\research\clean_experiment_artifacts.ps1
.\npunlock-probe\research\clean_experiment_artifacts.ps1 -Apply
```

脚本只处理明确的实验目录、项目下载件和两个固定NPU用户缓存目录，保留源码与Markdown报告；
拒绝目录联接/符号链接，不递归删除运行环境、当前内核或工具链。自定义缓存路径不自动清理。

[最新模型优化报告](npunlock-probe/research/results/model-cross-device-followup/REPORT.md)记录本轮CPU/GPU快路径、batch4/8和NPU缓存验收。

历史：2026-10-07曾清理下载件并压缩原始实验数据，当时约18.29→13.47GiB；本轮按新的保留策略直接清除了这些原始实验数据。
