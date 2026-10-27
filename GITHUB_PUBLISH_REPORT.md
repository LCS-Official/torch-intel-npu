# GitHub 首发交付记录

2026-10-11，已按授权公开发布 **v0.1.0 Experimental Preview**。这是实验源码预览版，不是稳定 v1.0。

- 公开仓库：[LCS-Official/torch-intel-npu](https://github.com/LCS-Official/torch-intel-npu)。
- 预发布版本：[v0.1.0 Experimental Preview](https://github.com/LCS-Official/torch-intel-npu/releases/tag/v0.1.0)。
- tag 对应提交：`03eee55b2c3133767ee5c16bf009c42832a09c4d`。
- GitHub 发布时间：2026-10-11 09:45:14 UTC，即北京时间 17:45:14。

## 发布内容与验证

公开仓库包含570个源码/报告/内核伴随文件及1个 `SOURCE_MANIFEST.json`，没有把整个开发工作区上传。
payload 为6,415,728字节（约6.12MiB）；GitHub tag 下载 ZIP 为2,012,032字节。
已从公开下载地址读取归档到内存，逐文件核对集合、长度与SHA256，全部与本地最终快照一致。
manifest 的SHA256为 `77ea5acb662186570dfa75646684ed7e7f5a19d782bb59a2256c834b9e415ae2`。

45项离线测试已在本地真实Git克隆内通过；`core.autocrlf=true` 下索引及克隆校验通过。
本轮非Quick真实NPU设备/训练/模型回归共53组、61,958次调用，细节见 [RELEASE_READINESS.md](RELEASE_READINESS.md)。
各进程资源回收检查通过。本轮没有修改计算内核、运行时或数值容差。

最终源码和本地Git元数据均未发现用户提供的访问令牌；认证信息只在请求/推送进程内使用，
远端URL没有嵌入凭据，提交身份使用GitHub noreply地址。
公开内容排除了Conda环境、模型、宿主DLL/PYD、图blob、缓存、实验数组/日志和旧本机依赖冻结清单。

## 云端CI限制

GitHub返回账号因账单问题锁定，Actions任务没有启动，没有执行任何测试步骤。
本地测试通过不能记作云端CI通过；首页、发布说明和验收报告已明确记录云端尚未执行。

- [最终main运行记录](https://github.com/LCS-Official/torch-intel-npu/actions/runs/38129391800)。
- [v0.1.0 tag运行记录](https://github.com/LCS-Official/torch-intel-npu/actions/runs/38129458361)。

需要账号所有者处理GitHub账户的账单/Actions限制；解除后可手动运行 `Offline source checks`。
手动运行包含严格发布manifest校验及同一组45项离线检查。没有修改账单设置或产生付费任务。

## 范围与清理

首发仍限定已验证的Windows/Lunar Lake NPU4000组合；普通FP16矩乘仍走FP32 SHAVE后转回FP16。
FP16 SmallResNet仅完成逐步同权重CPU参考对照，15步测试后段梯度相对L2误差最高约22.65%。
不认证通用LLM训练、GPU+NPU协同、任意模型兼容、跨驱动或长时间稳定性；暂无wheel。
完整契约见 [COMPATIBILITY.md](COMPATIBILITY.md)，安装见 [INSTALL.md](INSTALL.md)。

本轮6份硬件JSON/日志共40,649字节、两份Git验证副本共17,530,146字节及临时导出副本已清理。
GitHub下载归档只在内存中验证，没有留下ZIP或图片。保留原环境、模型、工具链、文字报告和实际发布的
`E:\Pytorch_xfire_NGPU\source-release` Git仓库；该仓库工作树干净，远端为上述公开仓库。
后续发布应从开发工作区重新导出并同步已审阅的源码变化，不能把完整开发工作区直接上传。
