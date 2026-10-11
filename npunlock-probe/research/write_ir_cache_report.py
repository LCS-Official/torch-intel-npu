"""Audit temporary evidence before retaining the Markdown report only."""
import hashlib
import json
from pathlib import Path
import statistics
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/startup-ir-cache'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    labels=['baseline-fill',*[f'baseline-{i}' for i in range(1,4)],*[f'base-{i}' for i in range(1,4)],
        'candidate-fill',*[f'candidate-{i}' for i in range(1,4)],
        *[f'paired-{mode}-{i}' for i in range(1,4) for mode in ('full','fast')],'candidate-corrupt']
    data={name:read(OUT/(name+'.json')) for name in labels}
    expected=None;tokens=set()
    for name,row in data.items():
        assert row['status']=='passed' and row['quality']['relative_l2']<.03 and row['quality']['argmax_equal']
        assert row['checkpoint_sha256']=='5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c'
        path=OUT/(name+'-outputs.npz');assert sha(path)==row['outputs_sha256']
        with np.load(path) as archive:arrays=[archive[k].copy() for k in ('hidden','logits')]
        if expected is None:expected=arrays
        for a,b in zip(arrays,expected):np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8))
        tokens.add(row['token'])
        assert row['cold']['stats_delta']==dict(npu_calls=90,session_creations=31,driver_allocations=184,host_upload_copies=62,host_download_copies=1)
        final=row['final_stats'];assert final['closed'] and not final['context_failed']
        assert all(final[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
        folder=OUT/'before' if name.startswith(('base-','baseline-')) else ROOT
        for file in ('torch_npu_smollm.py','torch_npu_graph_cache.py'):assert row['sources'][file]==sha(folder/file)
    assert len(tokens)==1
    for mode in ('full','fast'):
        for i in range(1,4):
            row=data[f'paired-{mode}-{i}'];counts=row['cold_cache']
            assert counts['hits']==30 and counts['compiles']==counts['rejected']==counts['writes']==0
            assert counts['ir_skipped']==(30 if mode=='fast' else 0)
    damaged=data['candidate-corrupt']['cold_cache']
    assert {k:damaged[k] for k in ('hits','rejected','compiles','writes','ir_skipped')}==dict(hits=29,rejected=1,compiles=1,writes=1,ir_skipped=29)
    sizes={};sets=[]
    for directory in ('baseline-cache','candidate-cache','paired-full-ir-cache'):
        blobs=set();size=0
        for path in (OUT/directory).glob('*.npc'):
            size+=path.stat().st_size
            with zipfile.ZipFile(path) as archive:
                assert all(v.compress_type==zipfile.ZIP_STORED for v in archive.infolist())
                blob=archive.read('graph.blob');manifest=json.loads(archive.read('manifest.json'))
                digest=hashlib.sha256(blob).hexdigest();assert digest==manifest['blob_sha256']
                assert manifest['profile']['device']==0x643e and manifest['profile']['driver']==1005540
                if directory=='candidate-cache':assert manifest['io_contract']['outputs'][0]['shape']==[1,24,576]
                blobs.add(digest)
        assert len(blobs)==30 and size<512*1024*1024
        sizes[directory]=size;sets.append(blobs)
    assert sets[0]==sets[1]==sets[2]
    mapping=read(OUT/'paired-cache-map.json');assert len(mapping)==30 and {r['blob_sha256'] for r in mapping}==sets[0]
    unit=read(OUT/'prefill-checks.json');legacy=read(OUT.parent/'model-cross-device-followup/cache-checks.json')
    assert unit['status']==legacy['status']=='passed' and len(unit['checks'])==len(legacy['checks'])==19
    guards=read(OUT.parent/'model-cross-device-followup/regression/guards.json')
    assert guards['status']=='passed' and len(guards['checks'])==21
    for phase in ('fill','load'):
        row=read(OUT/('shape-'+phase+'.json'));assert row['status']=='passed'
        assert row['cache']['compiles']==(10 if phase=='fill' else 0)
        if phase=='load':
            assert len(row['checks'])==8 and row['cache']['hits']==10 and row['cache']['ir_skipped']==4
            assert any(c['operation']=='different_gamma_shared_graph' and c['new_sessions']==0 for c in row['checks'])
    def median(mode,field):return statistics.median(data[f'paired-{mode}-{i}'][field] for i in range(1,4))
    cold={mode:statistics.median(data[f'paired-{mode}-{i}']['cold']['ttft_ms'] for i in range(1,4)) for mode in ('full','fast')}
    lines=['# NPU 冷首 token：缓存命中跳过 IR 重建','',
        '2026-10-11，Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，驱动32.0.100.5540，编译器8.3，Torch2.14.1+cpu。','',
        f"已接入 FP16 整层 prefill 的提前缓存命中。相同30份编译blob、新进程交替对照，冷首token中位 **{cold['full']/1000:.3f}→{cold['fast']/1000:.3f}秒**，降低 **{(1-cold['fast']/cold['full'])*100:.1f}%**。",
        f"Python脚本入口到首token中位 **{median('full','entry_to_token_ms')/1000:.3f}→{median('fast','entry_to_token_ms')/1000:.3f}秒**，降低 **{(1-median('fast','entry_to_token_ms')/median('full','entry_to_token_ms'))*100:.1f}%**。",'',
        '## 测量定义','',
        '- SmolLM2-135M-Instruct，FP16，完整30层NPU decoder，batch1、24-token固定提示词。embedding、最终norm、lm_head与argmax在CPU。',
        '- 冷首token从模型准备完成后的embedding到首token可用，包含上传、首次图准备、NPU执行、下载和CPU完整prefix输出头。',
        '- 脚本入口计时另外包含Torch/Transformers导入、NPU初始化和模型/tokenizer准备；不含PowerShell及Python解释器启动。未把图准备挪到计时外。',
        '- 每次均为新进程；随后预热2次、记录7次热请求。未清空Windows文件系统页缓存，未锁定桌面后台负载或频率。',
        '- 主对照顺序为full1、fast1、full2、fast2、full3、fast3。所有主对照源码及hook一致；full模式透明包装graph入口以关闭提前命中，仍执行原来的完整建图路径。',
        '- 对照两目录使用相同blob/patch reports，逐项SHA验证。prepare_ir_comparison.py只重新关联缓存键，未修改blob或绕过运行时验证。','',
        '## 主对照','',
        '| 模式/进程 | 冷首token ms | 脚本入口到首token ms | 热首token中位 ms | 缓存命中/编译/跳过IR |',
        '|---|---:|---:|---:|---|']
    for i in range(1,4):
        for mode in ('full','fast'):
            row=data[f'paired-{mode}-{i}'];c=row['cold_cache']
            lines.append(f"| {mode}-{i} | {row['cold']['ttft_ms']:.3f} | {row['entry_to_token_ms']:.3f} | {row['warm_ttft']['median']:.3f} | {c['hits']}/{c['compiles']}/{c['ir_skipped']} |")
    lines+=['','热请求仍存在桌面负载波动，本次不宣称热吞吐提升。NPU计算blob与全部调用数量不变。','',
        '## 实现与安全边界','',
        '- 仅对未替换graph/serializer/compiler的FrozenSmolLayer、FP16、整层fused prefill提前查缓存。decode、W8A16、诊断分段和研究替换图沿用完整建图路径。',
        '- 键包含七组projection的实际FP16字节与形状、RoPE字节、RMSNorm内核、输入输出契约，以及模型图/serializer源码、缓存实现、NPU4000编译实现、驱动、设备与编译器profile。',
        '- 两组gamma是运行时输入，不嵌入blob；不同gamma可以复用同一图，仍分别上传和绑定。',
        '- 命中后用固定输入输出名称、dtype和shape构造轻量绑定元数据；它不是模型计算图，不会送入编译器。实际计算仍由原blob执行。',
        '- 未命中或文件损坏时延迟构造完整图，核对实际I/O契约后编译。原有ZIP成员/体积、profile、SHA、patch报告及ELF/DPU检查全部保留；native session失败不会盲目重试。',
        '- 磁盘缓存仍需NPUNLOCK_GRAPH_CACHE显式启用，默认关闭。每个缓存目录上限512MiB、预留256MiB；不自动删除旧条目。新键与旧键不同，旧缓存需要重建，旧条目占用仍计入上限。',
        f"- 30图缓存大小 {sizes['candidate-cache']/1024**2:.3f}MiB；I/O契约比旧存储格式仅增加 {(sizes['candidate-cache']-sizes['baseline-cache'])/1024:.3f}KiB。",'',
        '## 正确性与回归','',
        f'- 共{len(labels)}个完整30层新进程通过。所有hidden/logits跨进程逐字节相同，首token相同（ID {next(iter(tokens))}）；每进程热请求也逐位一致。',
        f"- 与CPU FP16参考首tokenlogits相对L2误差 {max(row['quality']['relative_l2'] for row in data.values())*100:.9f}%，argmax一致，原3%质量门槛保留。",
        '- 每次冷请求均90次NPU调用、31次session创建、184次driver allocation、62次显式上传和1次下载；不包含驱动内部DMA计数。',
        '- 19项既有缓存检查、19项提前缓存检查、21项既有输入/训练/存储保护检查通过。',
        '- 单层硬件检查覆盖(1,1,576)、(1,24,576)、(4,1,576)、(8,64,576)，prefill、允许长度的后继decode及原KV不可变；新进程10图全部命中，0编译，4图跳过IR。',
        '- 新进程额外验证两组gamma同时改变时，新旧路径hidden/KV逐位一致，结果随gamma改变，新增session为0。共8项硬件对照通过。',
        f"- 故意改坏1项I/O契约：29命中、1拒绝、1编译、1写回、29跳过IR，冷首token{data['candidate-corrupt']['cold']['ttft_ms']:.3f}ms；输出仍逐位相同。",
        '- 所有硬件进程退出时live/cached/quarantined buffer与session归零，context_failed=false。','',
        '## 补充样本与限制','',
        '| 样本 | 冷首token ms | 脚本入口到首token ms |',
        '|---|---:|---:|']
    for name in labels:
        if name.startswith('paired-'):continue
        row=data[name];lines.append(f"| {name} | {row['cold']['ttft_ms']:.3f} | {row['entry_to_token_ms']:.3f} |")
    lines+=['','base-2/3存在明显的导入/上传/其它阶段波动，未将其混入主交替对照。没有验证波动原因。',
        'candidate-fill仍需首次完整编译，约28.56秒；本轮改善的是已有磁盘缓存的新进程请求，不能把该收益用于无缓存启动。',
        '没有重测完整32-token生成、其它LLM、GPU协同或训练；未新增OpenVINO依赖。','',
        '## 剩余开销与下一步','',
        '初始完整图构造/序列化30层共785.827ms；主对照fast的轻量identity约77ms。还避免了为旧键拼接和散列大IR权重缓冲。',
        '各hook为包含式时间，存在嵌套；下面不能相加为总耗时。','',
        '| fast路径hook | 三进程中位 ms |','|---|---:|']
    for key in ('prefill_identity','cache_read','inspect_blob','session_create','norm_parameter','layer_run_many'):
        value=statistics.median(data[f'paired-fast-{i}']['cold']['hooks'][key]['ms'] for i in range(1,4))
        lines.append(f'| {key} | {value:.3f} |')
    lines+=['','下一步优先研究ELF/DPU证据检查中的重复解析，目标是在保留相同验证的条件下减少约0.29秒检查开销。',
        '若目标改成整个程序启动时间，则Torch/Transformers导入及CPU模型/适配器重复准备也需要单独测量；这部分目前仍约4.5秒。','',
        '## 复现与保留','',
        '运行入口为run_torch_npu.ps1 -Script npunlock-probe/research/startup_ir_cache.py -ScriptArguments --label,<名称>；启用的缓存目录通过NPUNLOCK_GRAPH_CACHE传入。',
        'prefill_cache_checks.py和graph_cache_checks.py不调用硬件；prefill_cache_hardware.py先--phase fill再用新进程--phase load。',
        '主A/B需要先填充candidate-cache，再运行prepare_ir_comparison.py；full模式传--full-ir并使用paired-full-ir-cache，fast模式使用candidate-cache。修改前baseline-cache存在时会额外验证旧blob一致性。重新复算全部历史样本需按before源码分别生成，不能直接读取已清理的数据。',
        '按用户约定，验收后删除本轮NPC/NPZ/原始JSON/日志，仅保留源码、before文字源码和本报告。清空缓存后，下次首次运行需要重新编译；实际应用要跨进程保留缓存才能获得命中收益。','',
        '## 验收源码指纹','',
        '| 文件 | SHA256 |','|---|---|']
    for path in [ROOT/'torch_npu_smollm.py',ROOT/'torch_npu_graph_cache.py',ROOT/'npunlock/python/npunlock/npu4000.py',
        *[ROOT/'npunlock-probe/research'/name for name in ('startup_ir_cache.py','prefill_cache_checks.py','prefill_cache_hardware.py','prepare_ir_comparison.py','corrupt_prefill_entry.py')]]:
        lines.append(f'| {path.relative_to(ROOT)} | {sha(path)} |')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Audit passed:',len(labels),'full-model processes, identical blobs/outputs; report written',flush=True)


if __name__=='__main__':main()
