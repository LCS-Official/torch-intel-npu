"""Condense checked measurements into a report before deleting experiment artifacts."""
import json
from pathlib import Path
from model_cross_device import ROOT, sha
from model_cross_device_followup import OUT, BASE


def main():
    def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
    summary=read(OUT/'summary.json');cache=read(OUT/'cache-audit.json')
    paired=read(OUT/'xpu-b8-paired.json');guards=read(OUT/'regression/guards.json')
    assert summary['status']==cache['status']==paired['status']==guards['status']=='passed'
    rows=summary['rows'];npu={}
    for row in rows:
        if row['device']=='npu':npu[row['label']]=read(OUT/(row['label']+'.json'))
    lines=['# CPU / NPU / GPU 后续优化与缓存验收','',
        '2026-10-10，Core Ultra 7 258V / Arc 140V / Intel AI Boost 8086:643e。',
        '本轮完成 CPU 精度对照、GPU SDPA、batch4/8 与 NPU 编译图持久缓存。',
        '生产改动仅在 FrozenSmolLayer 的融合图编译入口接入可选缓存；模型图计算、SHAVE 内核、原生 DLL/PYD 和默认精度未改。','',
        '## 统一协议','',
        '- 本地 SmolLM2-135M-Instruct 完整30层，24-token 提示词，greedy生成32 token，不提前遇 EOS 停止。',
        '- batch1/4/8；各 batch 采用相同提示词的重复请求。吞吐为整个 batch 合计 token/s，单请求速度需除以 batch。',
        '- CPU/GPU 为完整 Transformers 模型；NPU 为30层 decoder及KV在NPU、embedding/final norm/lm_head/argmax在CPU。',
        '- 每组首次完整生成、额外预热2次、正式7次；表内为7次中位数。包含每步token返回主机，GPU每步同步。',
        '- 包含完整 prefill 输出头，模型加载、校验下载、证据保存和最终KV释放不计入热时间。与第一轮协议一致。',
        '- CPU/NPU：Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3；GPU：Python3.11.15、Torch2.12.0+xpu、NumPy2.4.6。',
        '- 共用 Transformers5.18.0、safetensors0.8.0、tokenizers0.23.2、huggingface-hub1.33.0。NPU驱动32.0.100.5540，图编译器8.3。',
        '- checkpoint SHA256：`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`。','',
        '## 热结果与准确性','',
        '固定token回放对照第一轮 CPU FP16 logits，门槛最大relative L2 < 3%；没有放宽门槛。',
        '所有配置自身重复生成的全部logits/token逐位一致。BF16两组未通过参考质量检查，其余配置全部生成相同的32 token。','',
        '| 配置 | batch | 线程 | 首token ms | 总生成 ms | 总token/s | 最大relative L2 | 质量门槛 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        label=r['label']
        lines.append(f"| {label} | {r['batch']} | {r['threads']} | {r['ttft_ms']:.3f} | {r['elapsed_ms']:.3f} | {r['tokens_per_second']:.3f} | {100*r['max_relative_l2']:.4f}% | {'通过' if r['quality_gate_passed'] else '未通过'} |")
    lines+=['','CPU FP32没有超过本轮CPU FP16；BF16最大误差6.21%、回放argmax为30/32且greedy轨迹不同，暂不推荐。',
        'GPU SDPA在batch1/4本轮分别达到32.32/122.55 token/s，比各自eager的25.45/105.45更快；',
        'profiler验证各组单步包含30次 `aten::scaled_dot_product_attention`，没有仅改标签。',
        'batch8独立进程结果eager为207.85、SDPA为196.09；不能由此认定SDPA在大batch必然更慢。','',
        'NPU batch4/8总吞吐167.39/288.05 token/s，单请求约41.85/36.01 token/s；',
        'batch并行提高合计吞吐，但没有把单请求延迟降低到batch1水平。NPU与CPU/GPU模型放置及Torch版本有差异。','',
        '## GPU batch8 配对复核','',
        paired['protocol']+'.',
        '同一模型使用公开 `set_attn_implementation` 切换，每模式预热3次；7对交替先后顺序，切换操作不计时。',
        '八个相同输入的完整logits/序列都检查相同，紧凑证据只保存一行；同模式重复输出逐位一致。','',
        '| 对次 | 先后顺序 | eager总 ms | SDPA总 ms | eager/SDPA耗时比 |',
        '|---:|---|---:|---:|---:|']
    for p in paired['pairs']:
        lines.append(f"| {p['index']+1} | {'→'.join(p['order'])} | {p['eager']['elapsed_ms']:.3f} | {p['sdpa']['elapsed_ms']:.3f} | {p['eager']['elapsed_ms']/p['sdpa']['elapsed_ms']:.3f} |")
    wins=sum(p['sdpa']['elapsed_ms']<p['eager']['elapsed_ms'] for p in paired['pairs'])
    lines += ['',f"本轮SDPA快{wins}/7对，逐对速度比中位数{paired['sdpa_speedup']['median']:.3f}，Q1–Q3为{paired['sdpa_speedup']['q1']:.3f}–{paired['sdpa_speedup']['q3']:.3f}。",
        '配对测试整体明显比此前独立进程慢，桌面负载/频率没有锁定；原因未单独测量。',
        '因此不自动统一切换所有batch，也不把这组波动作为稳定收益承诺。','',
        '## NPU 编译图缓存：已实现并通过硬件验收','',
        '| 新进程情况 | 首次完整生成 秒 | 缓存命中 | 融合图编译 | 缓存写入 |',
        '|---|---:|---:|---:|---:|']
    for label in ('npu-cache-fill','npu-cache-read','npu-cache-corrupt'):
        r=npu[label];c=r['graph_cache']
        lines.append(f"| {label} | {r['cold']['elapsed_ms']/1000:.3f} | {c['hits']} | {c['compiles']} | {c['writes']} |")
    lines+=['',
        '第一轮无磁盘缓存的首次完整生成68.124秒；本轮首次编译并写缓存76.409秒，其中缓存写入累计9.476秒。',
        '新进程读缓存6.229秒，60图全命中、零融合图编译；读取/解压累计2.416秒，仍需驱动bootstrap及创建session。',
        '上述是首次生成调用时间，模型加载另计，均为单次观察；不能将全部差值解释为纯编译时间。',
        '故意截断一个已备份缓存文件后，59命中、1拒绝、1重编译并写回，首次生成7.441秒。',
        '三种进程全部greedy/replay logits和token逐位一致；缓存不是精度优化或热内核提速，热样本差异含环境波动。','',
        '- `NPUNLOCK_GRAPH_CACHE` 显式指定目录时启用；默认关闭。键包含图/权重/内核指纹、驱动、编译器、设备和补丁实现指纹。',
        '- ZIP保存原生blob及JSON清单，验证成员、大小、SHA256、身份和补丁报告；不使用pickle。异常缓存回退到原编译路径。',
        '- 原子写入，单写者锁；总容量512MiB，保留256MiB磁盘余量，单blob不超过64MiB。达到限额跳过写入，不自动删除已有缓存。',
        '- 研究脚本自定义编译器hook仍执行原hook并绕过磁盘缓存；原生session执行故障不在缓存读取代码中重试。',
        '- 已知限制：写者崩溃可能留下`.writer.lock`，确认无写者后可手动删除；已有缓存读取仍可用。',
        f"- 本次60项压缩缓存占{cache['cached_bytes']/1024**2:.2f}MiB；按用户要求验收后清理，后续首次启用需重新建立。",'',
        '## 运行与回归检查','',
        f"17项独立缓存检查、{len(guards['checks'])}项冻结模型保护检查通过。",
        '离线重新计算17组性能统计/误差/证据指纹通过；缓存跨进程全部输出逐位一致，AST还原检查确认仅增加编译包装、图计算未改。',
        'NPU各热生成均为960次层图执行、930个chain graph、248个chain group，上传/下载各32次，零新session、零设备中间复制/packing。']
    for label in ('npu-cache-read','npu-b4-float16-eager-t8','npu-b8-float16-eager-t8'):
        r=npu[label];counts=[s['npu_delta']['driver_allocations'] for s in r['samples']]
        lines.append(f"- {label}：七轮新增驱动分配 {counts}；shutdown后live/cache/quarantine/session全部归零。")
    lines+=['','关键生产文件SHA256：','',f"- torch_npu_graph_cache.py：`{sha(ROOT/'torch_npu_graph_cache.py')}`。",
        f"- torch_npu_smollm.py：`{sha(ROOT/'torch_npu_smollm.py')}`。",'',
        '## 复现与保留策略','',
        '按用户要求仅保留代码、运行/构建所需文件和文字报告。历史原始数组、编译图、计时JSON及重复二进制快照已清理，实验源码保留；',
        '上文是清理前已完成验收的摘要，历史离线audit需原始文件，清理后无法原样复算。重新执行会生成新的实验数据。',
        '模型权重属于基准输入，约0.25GiB，与当前环境/工具链保留以便直接继续研究；它们不是实验输出。','',
        '先重新生成共同参考，再运行单个配置：','',
        '```powershell',
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\model_cross_device.py -ScriptArguments '--device','prepare','--threads','8','--label','fixture'",
        "$env:NPUNLOCK_GRAPH_CACHE=Join-Path $env:LOCALAPPDATA 'torch-intel-npu\\smollm-graph-cache'",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\model_cross_device_followup.py -ScriptArguments '--device','npu','--batch','1'",
        '# 新进程再次执行同一命令可验证缓存加载；改 batch 为4/8可测对应配置。',
        'Remove-Item Env:NPUNLOCK_GRAPH_CACHE',
        '```','',
        'CPU增加`--dtype float32`或`bfloat16`；GPU用已有blood环境和项目model-benchmark-deps，增加`--attention sdpa`。',
        'GPU环境设置方法见[第一轮报告](../model-cross-device/REPORT.md)，配对入口为`research/gpu_attention_paired.py`。',
        '本轮未测长上下文、多样提示词、功耗、LLM训练、GPU torch.compile或GPU+NPU同时运行。FrozenSmolStack仍为冻结推理路径。',
        '剩余已发现的问题：NPU batch8每轮仍有84次驱动分配，batch1/4为零。下一轮可先分析16MiB/256块缓存边界和释放顺序，再在有界容量下修复复用。',
        '随后扩展模型上下文/输入覆盖和稳定负载下的阶段剖析，再决定RMSNorm/SHAVE内核热点；不据此次结果优先改通用FP32 GEMM。','',
        '## 七轮总生成时间原始数值（ms，保留于报告）','',
        '| 配置 | 七轮样本 |','|---|---|']
    for r in rows:
        samples=read(OUT/(r['label']+'.json'))['summary']['elapsed_ms']['samples']
        lines.append(f"| {r['label']} | {', '.join(f'{v:.3f}' for v in samples)} |")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(OUT/'REPORT.md')


if __name__=='__main__':main()
