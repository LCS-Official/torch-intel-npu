# S46 分类损失原始Tensor版本保护

日期：2026-10-27。v0.2.0之后的开发修复；本轮只处理训练语义，不改量化算法或质量门槛。

## 问题与CPU对照

空间CE/NLL会先整理类轴并复制布局。此前，4D非连续标签经过contiguous后，NLL只保存副本，修改原标签或其共享storage别名时反向仍继续。另发现空间NLL输入自身也因布局复制丢失原始版本关联。错误不在损失数值内核，而在autograd保存的来源。

本机Torch2.14.1+cpu的行为需要按rank和复制情况区分：

| 来源 | CPU行为／本轮处理 |
|---|---|
| 4D硬标签，含非连续view | CE/NLL/native NLL2D在原标签或别名被修改后报版本错误；补齐复制源保护 |
| 3D/5D非连续硬标签 | CPU先复制，之后修改原始来源仍可反向；保留这个接受行为 |
| 4D NLL原始输入 | 连续或非连续输入都需检查原始版本 |
| 其他空间rank的NLL输入 | 连续输入检查原始版本，已复制的非连续来源按CPU行为接受修改 |
| CE原始logits | 不额外保存logits版本；正常修改行为保持CPU语义 |
| retained graph | 每次反向重新读取保存Tensor并检查版本 |

先在原实现真机运行110组检查，复现13处CPU应报错、NPU未报错的差异，共5933次调用；退出context健康、资源归零。它是故障复现，不计入正式通过数。

## 修复

在torch_npu_autograd加入共享LossSourceGuard，CE/NLL/native NLL2D在loss出口保存必要的原始Tensor引用。反向先解包saved_tensors，由PyTorch版本检查拒绝不合法修改，再进入原来的损失梯度计算。检查遵循上表，不把所有复制源都一律保存。

只改变Python autograd与来源管理：不复制原始payload、不增加SHAVE计算、不修改DLL/PYD/ELF，也不修改PyTorch安装。对照旧版的26组未修改输入用例，NPU调用数逐项一致。

## 数值与训练验收

环境沿用Windows x64、Core Ultra 7 258V／NPU4000 8086:643e、驱动32.0.100.5540、编译器8.3；Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3。所有NPU进程串行运行。

| 真机检查 | 组数 | NPU调用 | 最大绝对差 | 结果 |
|---|---:|---:|---:|---|
| versions / float32 | 122 | 10,240 | 2.384185791e-07 | 通过／退出归零 |
| versions / float16 | 122 | 14,539 | 0.00390625 | 通过／退出归零 |
| training / float32 | 1 | 468 | 2.980232239e-08 | 通过／退出归零 |
| training / float16 | 1 | 672 | 0.00048828125 | 通过／退出归零 |

正式4个进程合计25,919次NPU调用，结束live buffer、空闲buffer、标量buffer、图session和quarantine均归零，context_failed=False。

每种精度的versions为122组：119组版本行为对照（50组应拒绝、69组应接受），另3组no_grad。覆盖2D/3D/4D/5D、标签及输入的连续/非连续布局、weight修改、无关副本修改、retain_graph和原生NLL2D；4D的(2,3,33,365)含72,270输入元素、24,090标签，走跨packet路径。原13处差异归零；正确接受的路径完整比较CPU输出与梯度，梯度相对L2还须低于FP32的1e-4、FP16的1%。

训练每种精度独立执行三步：加权平滑硬标签CE＋软标签CE＋MSE，对照CPU全部参数、梯度和SGD更新，损失下降并复用缓存。以上时间/调用统计不是性能benchmark，不作训练加速承诺。

最终离线247项通过：根目录209、npunlock Python API32、导出器6。新增4个回归方法，包括真实生产类/包装入口的CPU oracle、不同rank复制语义、retain_graph、none/mean/sum及完整梯度。硬件验收后补充的最后一项只扩展离线测试，三份运行实现/验收脚本保持真机时指纹；没有因此重复硬件运行。

原27份resident C、66份ELF和66份汇编共159项哈希与manifest一致，原生session C源码及C++ backend与v0.2.0源码一致。无需新增依赖、环境、模型或工具链。

## 复现与范围

```powershell
$env:PYTHONPATH="$PWD;$PWD\npunlock\python"
python -B -m unittest test_torch_npu_losses
.\run_torch_npu.ps1 -Script torch_npu_loss_extended_demo.py -ScriptArguments @('--case','versions','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_loss_extended_demo.py -ScriptArguments @('--case','versions','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_loss_extended_demo.py -ScriptArguments @('--case','training','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_loss_extended_demo.py -ScriptArguments @('--case','training','--precision','float16')
```

已有v0.2.0运行环境同步torch_intel_npu.py和torch_npu_autograd.py后重启Python即可。独立开发分支保存本轮变化，v0.2.0 tag继续保留原快照；普通开发提交不改写旧SOURCE_MANIFEST，下一次正式发布再导出。

只认证上述一阶入口与本机Torch行为；不提供二阶梯度、任意view写入、绕过PyTorch版本计数的外部内存修改检测或长期FP16收敛保证。W8逐层误差定位仍为下一阶段，原16个低位质量失败和INT4/NF4整网边界未变化。

## 源码身份

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| torch_npu_autograd.py | `283f52d7b5939b5c8b79569fc51b7a4cc54495a4aecaf4be713953f7111afe79` |
| torch_npu_loss_extended_demo.py | `ea6953071fc55b60f0aae6b24a21b35b16b926cd2710df83d2f88dfc2eee0430` |
| test_torch_npu_losses.py | `4351ad3b01e11ea536bd4c2240024bd73b58b6b7947aabc224eceeafe49b218e` |

## 清理

只保留实现、测试、复现入口和本报告。原始JSON/日志在报告形成后清理，具体统计追加于此。

本轮清理13 份原始JSON/日志，共93905 字节；清理前后核对239份保留源码、文档、内核与运行组件SHA-256，全部一致。没有新增环境、模型、依赖、图表或持久图缓存，所有测试进程已退出。
