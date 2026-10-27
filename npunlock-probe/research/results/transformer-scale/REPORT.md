# 标准 Transformer 扩大模型与索引列分块

日期：2026-10-19；本地开发版，未推送 GitHub，既有 source-release 发布快照未修改。
平台：Windows x64、Core Ultra 7 258V、NPU 8086:643e、驱动 32.0.100.5540、图编译器 8.3、Torch 2.14.1+cpu。

## 结论

标准 token 分类模型从上一轮的 443 参数扩大到 42,211 和 232,387 参数。
两种规模的 FP32，以及 232,387 参数模型的 FP16 模型精度，均通过两步训练、全部参数梯度/SGD 动量更新对照、在新模型/优化器中逐位断点重放和推理对照。
这里 FP16 模型的 CrossEntropy 显式使用 FP32 logits；不等于纯 FP16 损失、AMP 或长期训练验证。

实际复现并修复了 Embedding 容量阻塞：保留现有 SHAVE 索引内核的完整 packet 和工作量边界，按列拆分较宽的权重与梯度，设备 DMA 完成整理及拼接。
没有新增 C/ELF、重建宿主 PYD/DLL、安装依赖或下载模型；原有重复索引、padding、越界保护与 FP16 梯度舍入语义保留。

本轮扩大的是标准 nn.TransformerEncoder 容器的迁移范围。此前 793,604 参数持续训练使用显式 attention blocks，不能据此替代标准容器的验证。
这些仍是合成小模型检查，不代表 Hugging Face LLM 训练、通用 CUDA 支持或真实数据集质量评测。

## 正式完整模型结果

共同结构为 Embedding、可学习位置、两层标准 TransformerEncoder、最终 LayerNorm、padding 感知池化、Linear 分类头和 CrossEntropy。
batch=2、heads=4、GELU、Dropout=0.25；稠密 padding 推理使用 enable_nested_tensor=False，没有修改 PyTorch 模块源码或关闭全局 MHA fastpath。

| 配置 | FP32 small | FP32 medium | FP16 medium |
|---|---:|---:|---:|
| 参数 | 42,211 | 232,387 | 232,387 |
| vocabulary / width / sequence / FFN | 512 / 32 / 8 / 128 | 2048 / 64 / 16 / 256 | 2048 / 64 / 16 / 256 |
| 参数字节 | 168,844 | 929,548 | 464,774 |
| 训练 loss 第 1 步 | 1.260022402 | 1.196559787 | 1.196524262 |
| 训练 loss 第 2 步 | 1.307673097 | 1.079928875 | 1.080201626 |
| 完整验收 NPU 图提交 | 3,186 | 5,695 | 12,680 |
| 后端活跃缓冲区峰值，字节 | 1,820,685 | 7,151,029 | 5,410,951 |
| 全梯度 / 更新 / checkpoint / eval | 通过 | 通过 | 通过 |

三组共 21,561 次真实 NPU 图提交。活跃缓冲区峰值不包含驱动图、会话和空闲分配缓存，不能当作任务管理器总 NPU 内存或整个进程内存。

每步先在独立 CPU 模型上用相同 Philox mask 参考执行，再在 NPU 模型运行标准模块，核对 logits、FP32 loss、每一个参数梯度和 SGD 更新；两步均独立对照，保持 FP32 rtol=3e-5/atol=3e-6、FP16 rtol=0.003/atol=0.0005，未放宽阈值。
优化器使用 lr=0.02、momentum=0.5、foreach=False、fused=False。token 含重复索引、padding=0 和词表最后一行；embedding padding 行的梯度严格为零。
同 mask 的 CPU reference 仅用于验收，不改变实际 NPU 模块的实现；这不是 CPU/NPU 相同 seed 自动产生相同随机序列的承诺。

第 1 步后将权重、动量和 NPU RNG 状态转成显式 CPU checkpoint，在内存中 torch.save/load；以新模型和新优化器恢复并重放第 2 步。
loss、完整 state_dict、全部 optimizer state/param_groups 和 RNG 均与原来的第 2 步逐位一致。
随后使用相同权重做 CPU/NPU eval 对照，并确认 eval 不推进 RNG。
两步随机 mask 不同，尤其 small 的第 2 步 loss 上升，因此不根据本表判断收敛、准确率或模型质量。

## 阻塞复现与修复

修改前 medium 模型已能上传，首次前向停在 Embedding：

```text
NPU index packet requires 6+2*L+max(R,L)*D <= 65536 and R*D*L <= 1048576
```

词表 R=2048、宽度 D=64、索引数量 L=32。原 packet 需要 131,142 个 FP32 载体元素，工作量 R*D*L=4,194,304，均超过旧单 packet 的边界。
它的权重本身只有 131,072 元素，小于通用张量 1,048,576 元素上限，因此不是本机物理内存不足，也不需要先扩大通用 allocation。

新增纯几何规划 index_tile_width：

```text
d = min(D,
        floor((65530 - 2*L) / max(R,L)),
        floor(1048576 / (R*L)))
```

要求 d>=1，源与输出均不超过通用张量容量。medium 的 d=16，分成 4 块；每块 packet=32,838 元素，工作量=1,048,576。
通过现有转置 DMA 和连续 slice 提取列块，复用相同 index-gather/index-backward SHAVE ELF，再转置拼回原布局。
原本能放进单 packet 的输入仍沿用单 packet 路径。

每块都保留完整索引序列，设备内核先校验所有 INT64 索引再读取载荷，不在 CPU 读取索引或计算权重/梯度。
只拆列不拆索引，因此不会改变同一输出元素的重复索引累加顺序。
index_select 的 FP16 梯度仍在 FP32 累加后转回 half；Embedding 仍逐次 half 舍入。
out 的 shape/dtype/别名检查和保存索引的 version guard 保持原契约。

每块有一个 4 字节错误标志同步读取。medium 每次训练前后向共 10 次计算区间下载：8 个索引标志、1 个因果检测 BOOL 和1 个分类标签检查标志，总计 37 字节。
small 为 4 次、13 字节。另有 8 次各 32 字节的 Dropout seed/counter 元数据上传和首次命中的标量常量上传；没有 Tensor 载荷计算回退或 host_device_copies。
这些统计在 CPU 结果比较、日志和 checkpoint 下载之前截取，验收读取不混进计算区间。

## 索引专项及回归

新增 tiled 专项对 2048×35、32 个索引做 3 块列分解，覆盖最后宽度为 3 的非整除块。
FP16/FP32 各测 dim=0、dim=1 的 index_select 及 padding Embedding，前向原始值和全部梯度均与 CPU 逐位一致；并覆盖 out、重复/首尾索引、负数/越界/高 32 位非零 INT64 索引。
独立专项两组共 598 次图提交，扣除标量缓存后的 live buffer 增量均为 0。

随后在同一进程运行完整 Embedding 回归，10 组全部通过，共 2,291 次图提交：

| 回归组 | NPU 提交 |
|---|---:|
| index FP16 / FP32 | 306 / 216 |
| embedding FP16 / FP32 | 230 / 155 |
| edges FP16 / FP32 | 127 / 82 |
| tiled FP16 / FP32 | 337 / 261 |
| contracts | 33 |
| training | 544 |

保留原始特殊浮点位模式、INT64 越界、128 次重复索引、half/subnormal 梯度、转置布局、out、index version 和普通分类训练检查。
单列无法满足 packet/工作量的元数据拒绝检查仍然有效，不能用分块绕过内核的安全限制。

原 443 参数标准 token 分类器回归通过，4,207 次图提交：原三步 loss 仍为 0.986053348、0.834364653、0.854081511，全部梯度/动量更新、checkpoint 和 eval 检查通过。
正式扩大模型、独立分块专项与两项回归合计 28,657 次图提交；不含修改前阻塞探测和验收脚本作用域修正前的运行。

66 项离线检查通过：核心/低位/mask/RNG/SDPA/标准模块/Transformer CPU 控制 29 项、npunlock Python API 32 项、source exporter 5 项。
新增控制覆盖列块容量与边界、两种扩大配置的标准 CPU 结构/全部梯度及 padding 行。
完整离线命令显式使用项目 npunlock/python 的 PYTHONPATH；初次遗漏此路径时导入了环境中的旧包，纠正入口后通过，没有修改依赖来掩盖问题。

正式模型与专项、完整索引回归均正常 shutdown，context_failed=False，设备 live/cache/scalar/session/quarantine 全部归零。
正式模型在 shutdown 前已只剩标量缓存，无普通活跃 Tensor 泄漏。
初次 small 数值/重放/推理均通过，但验收脚本把仍在函数作用域内的模型引用算进 shutdown 检查；将工作负载放入独立函数作用域后重新执行全部 small 检查，释放检查通过。不是通过删除生命周期检查来判定通过。

## 时间观察与下一步

| 配置 | 第一次前后向，秒 | 同进程第 2 次前后向，秒 |
|---|---:|---:|
| FP32 small | 68.459 | 0.373 |
| FP32 medium | 92.957 | 2.218 |
| FP16 medium | 118.971 | 2.609 |

第一次包含首次图编译；表中是这次正确性验收的单次观察，不包含优化器/比较/checkpoint，不是稳定吞吐 benchmark，也没有做 CPU/GPU 对等性能测量。
普通 FP16 mm 仍使用 FP32 SHAVE 算术和 half 转换，不能据 FP16 存储更小就推断它会更快；本轮没有改为 DPU 执行。

当前仍有以下边界：

- 通用 FP16/FP32/BOOL 张量最多 1,048,576 元素、rank<=4、无空 Tensor；本轮没有提高该上限。
- SDPA/native attention score 最多 65,536 元素；本轮最长 sequence=16，尚未解决长序列 attention。
- 列分块要求一整列和完整索引能放进单 packet，不拆词表行或 token 数量。反向总工作量仍可能达到 R*D*L，分块扩容本身不是复杂度优化。
- 两步合成训练不验证长期收敛、真实模型质量或任意配置。未认证 encoder-decoder、NestedTensor、AMP、通用 CUDA 扩展或异步 stream。

下一轮优先根据更长序列/批次的标准模型继续暴露限制，再处理 attention 的完整 score 界限或索引反向的实际复杂度；性能优化需单独评测普通矩乘/DPU 接入，避免把不断新增零散 API 当作通用兼容完成度。

## 复现与源码

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_transformer_scale.py -ScriptArguments '--profile','small','--precision','float32'
.\run_torch_npu.ps1 -Script .\torch_npu_transformer_scale.py -ScriptArguments '--profile','medium','--precision','float32'
.\run_torch_npu.ps1 -Script .\torch_npu_transformer_scale.py -ScriptArguments '--profile','medium','--precision','float16'
.\run_torch_npu.ps1 -Script .\torch_npu_embedding_demo.py -ScriptArguments '--case','all','--label','scale-regression'
```

源码改动位于 torch_intel_npu.py、torch_npu_geometry.py；复用并参数化现有 token_classifier，新增 torch_npu_transformer_scale.py，扩展原有索引验收和 CPU 测试，更新双语 README/兼容性矩阵。
现有 CI 已包含 test_torch_npu_transformer，因此新控制不需新增依赖或另建工作流。
源码导出器自动选择新的根目录脚本和本报告；未改已有 source-release 或执行推送。

正式模型三组使用相同实现指纹：

```text
torch_intel_npu.py
0ae4cb693cfd746c6615afa63e32b700eb20052bf170264de322fe8afa667f3d
torch_npu_geometry.py
cdc8a5d7771ddd0d0223b30b5a670e44acf282c5e7a835b3ac2aab093f3b9a88
torch_npu_transformer_demo.py
f4406465fafac72277f5fe076e407e79d8e1811fc26ecea1f0d84d20486859ab
torch_npu_transformer_scale.py
62564e02af4e98bbf812b52f9b025bcb84eba4890dc1126ac2173df045ce80af
resident_index.c（未修改）
e11bc2d4ec1028b3d87280a0dfdc16958a394d273f9ef6c8cfe685c636818635
resident/manifest.json（未修改）
9f2e8638e164e7a753c3d191e87f822e8b0abc42d45c2220333b23ded8211889
```

## 清理与交付

验收后删除本轮 8 份原始 JSON，共 36,436 字节；清理前后 748 份保留源码/报告/内核/必要运行时文件的 SHA-256 均一致，随后补充本清理记录及存储台账。
源码导出选择共 744 文件，包含新增脚本、测试和本报告；没有复制新的发布目录。transformer-scale 目录最终仅保留 REPORT.md，未生成图表或磁盘 checkpoint。
