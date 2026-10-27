# S29：加权分类、可微软标签与广播MSE

日期：2026-10-23。本地开发版，尚未同步到GitHub。平台为Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`、驱动32.0.100.5540、图编译器8.3、Torch2.14.1+cpu、NumPy2.5.3。

这一轮补损失函数的模型迁移缺口，使用现有SHAVE log-softmax、NLL packet、数学/归约和DMA；没有新增C/ELF、pip依赖、OpenVINO runtime或宿主DLL/PYD构建。62枚resident ELF及其manifest保持原样。

## 接通的行为

| 接口 | 范围 |
|---|---|
| NLL/硬标签CrossEntropy | FP16/FP32，rank1–5类轴输入，INT64标签、可选同dtype `(C,)` 权重、ignore_index、none/mean/sum；CE支持label_smoothing |
| 加权均值 | 分母为未忽略标签的所选类别权重之和；支持零/负权重，不用“分母必须大于零”的替代分母 |
| NLL特殊值 | 未选中的NaN/Inf权重不参与归约；忽略行不会由这些权重污染，非法标签仍通过4字节设备标志报错 |
| 软标签CrossEntropy | 目标与输入同shape，FP16/FP32混合标签/权重，标签不要求归一化或非负；输入、软标签、类别权重均可求一阶梯度 |
| 软标签归约 | none沿类轴求和，mean按样本/空间位置数归一化；负ignore_index没有作用，非负ignore_index拒绝 |
| MSE | rank0–64合法广播、FP16/FP32混合dtype、双方梯度、三种reduction、标量/步长/空输入 |
| 原生MSE | `mse_loss.out`、`mse_loss_backward`及`.grad_input`；梯度输出为广播shape，dtype为self.dtype |
| 原生NLL | `nll_loss_forward.output`、`nll_loss_backward`及`.grad_input`；`nll_loss2d_forward/backward`及对应out、4D一阶autograd |
| 空分类 | 支持空batch/空间维度，none空输出、sum零、mean NaN；类别数仍必须大于零 |
| 资源/错误 | out精确shape/dtype、输入及输出间禁止别名；拒绝非法metadata、超范围INT64标签；保存的原权重/标签有版本检查 |

硬标签类别权重不支持求梯度，与CPU NLL保持一致；软标签CE中的类别权重可以求梯度。不对软标签概率作额外归一化，避免改变用户输入及PyTorch接口语义。

## 一个容易混淆的原生MSE细节

本机`F.mse_loss`会先广播输入和目标，再调用ATen。直接调用`aten.mse_loss_backward`则按原self.numel()决定mean因子，并返回广播shape。自定义autograd随后把梯度归约回原输入形状；目标侧使用交换self/target后的原生规则。

CPU示例：x=`ones(2,1)`、y=`zeros(1,3)`、mean损失均为1。`F.mse_loss`得到x.grad每行1、y.grad每列-2/3；直接ATen autograd得到x.grad每行3、y.grad每列-4/3。本轮分别对照这两个入口，未把原生ATen反向改成数学上预广播的版本。CPU原生执行后者会给出输出resize警告；项目out仍要求精确预分配，不实现自动resize。

## 数值验证与诊断

入口为根目录`torch_npu_loss_extended_demo.py`，case为`hard/soft/mse/native/contracts/training`，每种精度独立串行进程。CPU与NPU使用独立叶Tensor及训练参数；CPU参考不参与NPU执行路径。测试在下载比对结果之前检查显式复制计数：soft CE和MSE无payload下载；非空硬标签前向只允许一份校验标志下载，native手动反向重建标签packet也会读取标志。计数不涵盖编译器或驱动内部DMA。

逐元素门槛沿用既有验收：FP32 rtol=4e-5、atol=2e-6；FP16 rtol=.004、atol=.001，equal_nan=True。门槛按模型精度配置选择，Half配置中混合FP32输出/梯度也使用Half门槛。没有宣称逐位一致或FP64参考全部正确；Inf/NaN与CPU出现位置也必须相同。

新增CPU测试最初使用Torch默认Half门槛，在空间CE、smoothing=1、负权重时出现0.000244140625误差。本机CPU把类轴移到最后一轴后，log-softmax本身也出现同样幅度的舍入差异。正式CPU测试使用与既有项目一致的上述Half门槛，没有因此放宽硬件验收门槛。

极端权重追加检查发现了真实实现问题：原先保留FP32中间值，使Half NLL在权重60000或约1e-7时与CPU的溢出/下溢或mean梯度不同。现补入加权乘积、归约结果和mean梯度除法的Half边界，并增加unbatched/2D、三种reduction的CPU及真机回归。语义核对安装包对应commit的[PyTorch LossNLL.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL.cpp)及[LossNLL2d.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL2d.cpp)，不复制CPU级联累加实现。

旧S19拒绝fixture曾把已经在S23接通的正常ConvTranspose2d当成非法配置，导致本轮回归报“Invalid configuration accepted”；现改用非法groups=3，并在断言中记录action编号。MSE合法广播、类别权重和软标签旧拒绝fixture也改为真正非法的形状/dtype，没有删除拒绝检查。

## 实际训练与边界

组合训练为67参数的Linear(4,8)→Tanh→Linear(8,3)，固定3样本；损失为加权/平滑硬标签CE + 0.1×加权软标签CE + 0.2×回归MSE。FP16模型显式使用FP32损失。CPU与NPU各自执行三步SGD，比较全部参数梯度和更新参数，整体梯度相对L2附加门槛为FP32<=1e-4、FP16<=.01，缓存图数在第二/三步不再增长。

这是接口及短训练正确性验收，没有测量吞吐优势、长期收敛、真实数据集质量或大模型最大容量。普通训练仍走SHAVE，不自动变成DPU训练或CUDA运行时兼容。

仍保留以下范围：硬标签`目标数*(C+4)<=65,536`；rank1–5分类；softmax行宽<=65,536；单allocation256MiB和1,536图缓存；out精确预分配且不可别名；高阶梯度不支持。其他损失家族（BCE、KL散度、CTC等）的完整接口未认证。

SHAVE归约继续采用既有FP32分块累加，不复刻CPU逐项Half级联顺序；极端抵消、接近舍入分界或更多项的组合可能与CPU有更大差别。保留CPU Half特殊值行为的测试不意味着认证所有极值或所有训练轨迹；训练优先使用显式FP32损失。

## 复现

```powershell
.\run_torch_npu.ps1 -Script torch_npu_loss_extended_demo.py -ScriptArguments @('--case','hard','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_loss_extended_demo.py -ScriptArguments @('--case','training','--precision','float32')
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_losses
```

完整数值汇总、源码指纹与清理记录见下方。原始JSON/日志按用户存储偏好清理，不保留图表、checkpoint或磁盘图缓存；保留测试入口以便复现。

## 最终验收汇总

最终25个真机进程全部通过：新增12个进程、404个测试组合；旧回归13个进程。共35,407次NPU图提交，其中新增专项17,136次、旧回归18,271次。123项离线检查通过（根目录CI列表86项、Python API32项、源码导出5项）；其中7项为本轮新增损失函数测试。

| 新增case | FP32组合数 / NPU提交 | FP16组合数 / NPU提交 |
|---|---:|---:|
| hard | 79 / 3,231 | 79 / 5,693 |
| soft | 73 / 1,198 | 73 / 2,585 |
| mse | 34 / 490 | 34 / 866 |
| native | 13 / 738 | 13 / 1,041 |
| contracts | 2 / 62 | 2 / 98 |
| training | 1 / 465 | 1 / 669 |

`hard`含79组：rank1/2/3/5、空batch/空间、全部reduction及smoothing、忽略/零/负/未选NaN/Inf权重、步长输入和权重，以及权重60000/约1e-7的unbatched和2D边界。`soft`含73组：rank1/2/4/空batch、混合dtype、全部reduction/smoothing及完整base梯度。`mse`含34组：标量、行/列、rank64、空、步长、混合dtype、双方梯度和直接ATen行为。`native`含12组手动接口及1组4D NLL autograd，检查out身份与调用者total_weight被正确使用。`contracts`每种精度检查14个metadata拒绝、非法INT64及原权重/标签版本保护。

| 组合训练 | 三步NPU损失 | 三步整体梯度相对L2 | 三步图缓存数 |
|---|---|---|---|
| float32 | 1.400812268 → 1.376957178 → 1.355870605 | 8.33216e-08, 6.91664e-08, 8.93022e-08 | 83, 83, 83 |
| float16 | 1.400734067 → 1.376920104 → 1.355868578 | 0, 0.000213544, 0.000451149 | 102, 102, 102 |

| 旧回归记录 | NPU提交 | 结果 |
|---|---:|---|
| `torch-npu-device.json` | 4,171 | 通过 |
| `torch-npu-training.json` | 2,770 | 通过 |
| `classification/nll.json` | 1,638 | 通过 |
| `classification/contracts.json` | 37 | 通过 |
| `portability/s29-contracts-float16.json` | 22 | 通过 |
| `portability/s29-contracts-float32.json` | 18 | 通过 |
| `portability/s29-losses-float16.json` | 977 | 通过 |
| `portability/s29-losses-float32.json` | 490 | 通过 |
| `portability/s29-smoothing-float16.json` | 2,860 | 通过 |
| `portability/s29-smoothing-float32.json` | 1,306 | 通过 |
| `amp/s29-mlp.json` | 1,372 | 通过 |
| `sdpa-extended/training-float16.json` | 1,851 | 通过 |
| `sdpa-extended/training-float32.json` | 759 | 通过 |

D Quick为33组，T Quick为11组，旧分类NLL覆盖两种精度，旧分类拒绝路径也通过；S19多维损失/smoothing/contracts各两种精度、原AMP MLP及原S28 GQA模型两种精度全部通过。FP16边界修正后重跑全部新增专项、旧NLL、FP16迁移loss/smoothing、AMP及GQA；D/T等不受NLL舍入修正影响的早期回归未额外重复。原始失败fixture诊断保留在上文。

所有最终进程的关闭状态正常，context_failed=False，live buffers、cached blocks/scalars/sessions和quarantine均为零。新增专项在shutdown之前的live buffers也仅含缓存标量。没有遇到native设备失效或新生命周期错误。

## 源码指纹

以下为最终新增12份记录共同核对的SHA-256；旧回归记录中保留各自执行时的指纹，不把修正前的无关回归冒充为修正后执行。

| 文件 | SHA-256 |
|---|---|
| `torch_intel_npu.py` | `6ee469b37b855a938aee973c4b1bc42affde846e4728e3d2bb116e50a7f76800` |
| `torch_npu_autograd.py` | `c7cbe6e6654a44a3f3037bdbd09cb863792b96225cd86694178b01f037cc1a21` |
| `torch_npu_loss_extended_demo.py` | `7d8b78d7440f7cc6abda8d899f217b7de9d606946d1a30617c5a335e73afd4d6` |
| `test_torch_npu_losses.py` | `25306684d28d19b81c13d5349c4d4e2bc7bc6fb8ba16b80436620acb75e3e7ae` |
| `npunlock/tests/fixtures/npu4000/resident/manifest.json` | `329ec9430d4f9fbe2b8b721b5fc5d72aef38f2d6ceb44a33322aa6515cd64cfc` |

resident manifest的25份源文件、62枚ELF及其汇编均逐项校验通过。没有更新source-release快照或推送GitHub；源码导出选择规则会自动纳入新增根目录测试/验收脚本及本REPORT。

## 存储清理

2026-10-23 S29损失函数兼容性验收后清理：删除本轮60份原始JSON/日志、可重建字节码及临时校验清单，共795512字节（0.759MiB）；清理前后986份保留源码、报告、内核与必要运行时文件SHA-256一致，随后补充报告和本台账。保留加权NLL/硬标签CE、可微软标签/权重CE、广播混合dtype MSE、原生forward/backward/out及Half边界修复、验收/7项新增CPU测试、CI与双语说明。最终25个真机进程全部通过，新增404个测试组合、共35407次记录提交，123项离线检查通过；CPU Half级联求和、分类packet及高阶梯度等边界明确保留。复用原62枚resident ELF，没有依赖安装、模型下载、SHAVE或宿主重建、图表或磁盘checkpoint/图缓存；source-release与GitHub未修改。
