"""Audit identical-blob startup pairs and retain their Markdown evidence."""
import json
from pathlib import Path
import statistics as st

import w8_prefill_cache as run


def read(tag):return json.loads((run.OUT/(tag+'.json')).read_text(encoding='utf-8'))
def median(rows,key):return st.median(row['cold'][key] for row in rows)
def fmt(value):return f'{value:.3f}'


def main():
    study=read('study');assert study['status']=='passed' and study['caches_removed']
    names=dict(fp16='FP16对照',w8='校准W8',mixed='混合50%',selected='逐层37.5%')
    all_records=[];variants={}
    for label in run.LABELS:
        fill=read(label+'-fill');anchor=fill['cold'];records=[read(tag) for tag in study['variants'][label]['rows']]
        assert len(records)==(3 if label=='fp16' else 6)
        for row in [fill,*records]:
            assert row['status']=='passed' and len(row['graphs'])==30 and len(row['samples'])==7
            assert row['cold']['array_sha256']==anchor['array_sha256'] and len(anchor['array_sha256'])==91
            assert row['cold']['logits_sha256']==anchor['logits_sha256']
            assert [g['blob_sha256'] for g in row['graphs']]==[g['blob_sha256'] for g in fill['graphs']]
            for sample in row['samples']:
                assert sample['array_sha256']==anchor['array_sha256'] and sample['logits_sha256']==anchor['logits_sha256']
                assert sample['delta']['npu_calls']==30 and sample['delta']['session_creations']==sample['delta']['driver_allocations']==0
            stats=row['final_stats'];assert stats['closed'] and not stats['context_failed']
            assert stats['npu_calls']==360
            assert all(stats[key]==0 for key in ('live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
            for path,digest in row['sources'].items():assert run.common.sha(run.ROOT/path)==digest
            for graph in row['graphs']:
                assert graph['integer_modes']==([] if label=='fp16' else [[0,2]])
            all_records.append(row)
        for row in records:
            counts=row['cold_cache']
            assert counts['hits']==30 and counts['compiles']==counts['rejected']==0
            assert counts['ir_skipped']==(30 if row['route']=='fast' else 0)
        if label!='fp16':
            mapping=study['variants'][label]['mapping'];assert len(mapping)==30
            assert [m['blob_sha256'] for m in mapping]==[g['blob_sha256'] for g in fill['graphs']]
            assert all(m['fast_key']!=m['full_key'] for m in mapping)
        variants[label]={route:[r for r in records if r['route']==route] for route in ('full','fast')}
    damaged=read('w8-corrupt');anchor=read('w8-fill')['cold'];counts=damaged['cold_cache']
    assert {key:counts[key] for key in ('hits','rejected','compiles','writes','ir_skipped')}==dict(hits=29,rejected=1,compiles=1,writes=1,ir_skipped=29)
    assert damaged['cold']['array_sha256']==anchor['array_sha256'] and damaged['cold']['logits_sha256']==anchor['logits_sha256']
    assert damaged['final_stats']['npu_calls']==360 and not damaged['final_stats']['context_failed']
    all_records.append(damaged)
    hardware=[]
    for label in ('fp16','w8','mixed'):
        for phase in ('fill','load'):
            row=json.loads((run.OUT/('hardware-'+label)/('shape-'+phase+'.json')).read_text(encoding='utf-8'))
            assert row['status']=='passed' and row['label']==label and len(row['checks'])==8
            assert row['cache']['compiles']==(10 if phase=='fill' else 0)
            assert row['cache']['ir_skipped']==(0 if phase=='fill' else 4)
            assert row['cache']['hits']==(0 if phase=='fill' else 10) and row['cache']['rejected']==0
            assert not row['final_stats']['context_failed'] and row['final_stats']['closed']
            assert all(row['final_stats'][key]==0 for key in ('live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
            assert row['runner_sha256']==run.common.sha(run.ROOT/'npunlock-probe/research/w8_prefill_cache_hardware.py')
            hardware.append((label,phase,row))
    old_checks=json.loads((run.ROOT/'npunlock-probe/research/results/startup-ir-cache/prefill-checks.json').read_text())
    checks=json.loads((run.ROOT/'npunlock-probe/research/results/model-cross-device-followup/cache-checks.json').read_text())
    assert old_checks['status']==checks['status']=='passed' and len(old_checks['checks'])==len(checks['checks'])==19
    total=sum(row['final_stats']['npu_calls'] for row in all_records)+sum(row['final_stats']['npu_calls'] for _,_,row in hardware)
    lines=['# W8与混合精度：prefill缓存提前命中', '',
        '2026-10-16，Core Ultra 7 258V / Lunar Lake NPU4000 `8086:643e`，驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu。', '',
        '已将现有FP16提前命中扩展到校准W8和显式混合精度；命中后跳过整层prefill IR重建。'
        '使用相同30份编译blob的新进程配对，首token耗时结果如下：', '',
        '| 配置 | 完整IR首token s | 提前命中首token s | 中位耗时降低 | 脚本入口：完整IR s | 脚本入口：提前命中 s |',
        '|---|---:|---:|---:|---:|---:|']
    for label in run.LABELS[1:]:
        full,fast=variants[label]['full'],variants[label]['fast']
        before,after=median(full,'ttft_ms'),median(fast,'ttft_ms')
        lines.append(f'| {names[label]} | {fmt(before/1000)} | {fmt(after/1000)} | {(1-after/before)*100:.2f}% | '
            f"{fmt(median(full,'entry_to_token_ms')/1000)} | {fmt(median(fast,'entry_to_token_ms')/1000)} |")
    control=variants['fp16']['fast']
    lines+=['',f"FP16原有快路径作为回归对照：首token中位{median(control,'ttft_ms')/1000:.3f}秒，"
        f"脚本入口中位{median(control,'entry_to_token_ms')/1000:.3f}秒；没有用FP16旧完整IR路径混入本轮低位收益。", '',
        '## 计时口径与固定条件', '',
        '- 本地SmolLM2-135M-Instruct、完整30层decoder及KV在NPU；CPU embedding/final norm/lm_head/argmax。'
        'batch1、固定24-token提示、CPU8线程、interop1线程；prefill的完整24位置CPU输出头保持原实现。',
        '- 每个配置复用前一轮固定逐行GPTQ校准权重，SHA256 '
        '`f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62`；'
        '混合50%保护全部Up/Down，逐层37.5%复用先前45项策略，没有重新选择或校准超参数。',
        '- 先建立30份提前命中缓存，再把同一blob/patch reports关联到完整IR键。逐层核对I/O契约与blob SHA256；'
        '每次正式请求均30次命中、0次编译。完整IR由透明graph包装器关闭提前命中，其他执行代码一致。',
        '- 每种低位策略3对全新进程，顺序full→fast、fast→full、full→fast；FP16另3个新进程快路径对照。'
        '每进程首次请求后预热2次、测7次热prefill，hash/数值验收在计时区间外。',
        '- 首token从模型准备完成后的embedding到argmax；脚本入口另外包括库导入、NPU初始化、模型和校准权重加载。'
        '未把IR准备挪到计时区间外，不含PowerShell/Python解释器创建前的时间。',
        '- 没有清空Windows文件系统页缓存、锁定频率或桌面负载；三对样本不能代表所有模型/机器。'
        '收益只针对已有磁盘缓存的新进程，不用于无缓存首次编译或热decode吞吐。', '',
        '## 全部正式样本', '',
        '| 配置/进程 | 首token ms | 脚本入口到首token ms | 热prefill中位 ms | 命中/编译/跳过IR |',
        '|---|---:|---:|---:|---|']
    for label in run.LABELS:
        for tag in study['variants'][label]['rows']:
            row=read(tag);cache=row['cold_cache']
            lines.append(f"| {tag} | {fmt(row['cold']['ttft_ms'])} | {fmt(row['cold']['entry_to_token_ms'])} | "
                f"{fmt(st.median(s['ttft_ms'] for s in row['samples']))} | {cache['hits']}/{cache['compiles']}/{cache['ir_skipped']} |")
    lines+=['', '## 实现与缓存身份', '',
        '- 生产改动主要在torch_npu_graph_cache.py。FrozenSmolLayer只增加对原weight_graph函数的身份记录，'
        '模型计算、量化器、C++、SHAVE、DLL/PYD、默认精度与公开API保持原样。',
        '- 沿用FP16的绑定元数据、延迟完整建图工厂与所有读取校验。W8身份额外包含mode、排序后的保护投影、'
        'torch_npu_lowbit源码，以及实际使用的每份PackedWeight的名称、shape、format、group_size、codes和scales。',
        '- 保留原FP16投影字节、RoPE、内核、shape/IO契约及源码依赖。保护投影使用原FP16，未被图使用的对应packed副本不影响身份；'
        '两组gamma仍为运行时输入，不进入常量指纹。',
        '- 替换graph、serializer、compiler或W8 weight_graph函数的研究路径绕过提前命中；子类也沿用完整路径。'
        'decode与诊断分段的路径没有扩展。',
        '- 键仍绑定设备、驱动、编译器、graph extension和缓存实现。因源码指纹变化，旧缓存需要重新建立一次；'
        '不会自动删除旧目录，旧条目仍计入容量。缓存保持显式opt-in、每目录512MiB上限、预留256MiB。', '',
        '## 数值、失效与资源验证', '',
        '- 10项模型/保护策略单元测试、19项既有cache检查、19项prefill检查、5项source-export检查通过，共53项。'
        '新增检查覆盖W8/混合绑定契约、codes/scales与保护策略失效、保护项原FP16字节、策略顺序无关、'
        '运行时gamma、研究hook和低位建图源码变化。',
        '- FP16身份算法与修改前实现对照一致；适配器删除新增的weight_graph身份记录这一行后，SHA256恢复为修改前值。'
        '源码整体指纹变化仍会使已有缓存失效，不能理解为旧缓存可直接复用。',
        '- 全部正式新进程与填充进程的最终hidden、30层K/V/position共91个Tensor、首token logits及argmax逐位一致；'
        '每进程7次热样本也逐位一致。',
        '- 所有30层图检查原生DPU证据：FP16没有整数权重矩阵任务；W8及混合的每层都包含amode=0/wmode=2的W8A16矩阵任务。',
        '- 故意损坏一个W8缓存IO契约，结果为29命中、1拒绝、1编译、1写回、29跳过IR；'
        f"输出仍逐位一致，首token{damaged['cold']['ttft_ms']:.3f}ms。原ZIP、SHA、profile及native检查保留。", '',
        '| 单层边界策略 | 阶段 | 检查项 | 缓存命中 | 编译 | 跳过IR | NPU调用 |', '|---|---|---:|---:|---:|---:|---:|']
    for label,phase,row in hardware:
        cache=row['cache'];lines.append(f"| {names[label]} | {phase} | {len(row['checks'])} | {cache['hits']} | {cache['compiles']} | {cache['ir_skipped']} | {row['final_stats']['npu_calls']} |")
    lines+=['', '边界形状为(1,1,576)、(1,24,576)、(4,1,576)、(8,64,576)，共48项检查。'
        '覆盖prefill、允许长度下的后继decode、旧KV不可变，以及两组gamma改变后共享图而输出仍与完整路径一致。'
        '边界测试使用第0层；完整30层只验证batch1/24-token启动，不能推广为全部30层在所有边界上的认证。',
        f'本轮共 **{total:,}次真实NPU调用**。所有进程关闭后live/cache/quarantine/session归零，context_failed=False。'
        '低位原质量门槛与默认配置不变；输出与旧路径一致不等于低位误差已消除。', '',
        '## 剩余启动开销', '',
        '下表为3个新进程cold请求的hook中位ms；hook含嵌套，不能相加。full_ir包括完整建图和序列化，'
        'identity为轻量绑定与指纹；读缓存、ELF证据检查、session和gamma准备仍保留。', '',
        '| 配置/路径 | full_ir | identity | cache_read | inspect_blob | session | gamma |', '|---|---:|---:|---:|---:|---:|---:|']
    for label in run.LABELS[1:]:
        for route in ('full','fast'):
            rows=variants[label][route]
            values=[st.median(r['cold']['hooks'].get(key,dict(ms=0.))['ms'] for r in rows)
                    for key in ('full_ir','identity','read','inspect_blob','session','gamma')]
            lines.append('| '+names[label]+'/'+route+' | '+' | '.join(fmt(v) for v in values)+' |')
    lines+=['', '提前命中新进程在首请求之前的准备（3次中位ms）：', '',
        '| 配置 | CPU模型/tokenizer/适配器/校准权重准备 | 其余入口准备、导入与后端初始化 |', '|---|---:|---:|']
    for label in run.LABELS:
        rows=variants[label]['fast']
        setup=st.median(r['setup_ms'] for r in rows)
        other=st.median(r['cold']['entry_to_token_ms']-r['cold']['ttft_ms']-r['setup_ms'] for r in rows)
        lines.append(f'| {names[label]} | {fmt(setup)} | {fmt(other)} |')
    lines+=['', '若继续优化程序入口到首token，下一步优先细分并减少CPU模型和权重的重复准备；'
        '现有测量中该部分约2.85–3.12秒，导入及后端初始化另外约1.82–1.87秒。'
        '模型准备后仍有约0.20–0.24秒ELF/DPU证据检查，可另行研究重复解析。'
        '当前数据没有证明这些开销都能安全消除。'
        '若优化稳态decode，则需另测图提交与设备任务，不能把本轮启动收益套用到token/s。', '',
        '## 复现与交付', '', '```powershell', "$env:PYTHONDONTWRITEBYTECODE='1'", "$env:PYTHONIOENCODING='utf-8'", "$env:NPUNLOCK_GRAPH_CACHE=$null",
        ".\\run_torch_npu.ps1 -Script .\\test_torch_npu_smollm_policy.py",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\graph_cache_checks.py",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\prefill_cache_checks.py",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\w8_prefill_cache.py -ScriptArguments '--case','prepare'",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\w8_prefill_cache.py -ScriptArguments '--case','study'",
        "foreach ($label in @('fp16','w8','mixed')) {",
        "  foreach ($phase in @('fill','load')) {",
        "    .\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\w8_prefill_cache_hardware.py -ScriptArguments '--label',$label,'--phase',$phase",
        '  }', '}', '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\w8_prefill_cache_report.py', '```', '',
        '按用户约定，验收后清理校准权重、NPC、计时/验收JSON等产物，仅保留生产与研究源码、修改前缓存源码及Markdown报告。'
        '研究A/B的系统临时缓存已逐策略删除；未新增依赖、模型、图片或持久图缓存，未推送GitHub。', '',
        '## 指纹', '', '| 文件 | SHA256 |', '|---|---|']
    sources=dict(all_records[0]['sources'])
    for path in ('test_torch_npu_smollm_policy.py','npunlock-probe/research/w8_prefill_cache_hardware.py',
                 'npunlock-probe/research/w8_prefill_cache_report.py','npunlock-probe/research/w8_efficiency_startup.py',
                 'npunlock-probe/research/w8_efficiency_report.py'):
        sources[path]=run.common.sha(run.ROOT/path)
    sources['graph-cache-before.py']=run.common.sha(run.OUT/'graph-cache-before.py')
    for path,digest in sources.items():lines.append(f'| `{path}` | `{digest}` |')
    lines+=['', '清理前临时权重与配对清单指纹：', '', '| 材料 | SHA256 |', '|---|---|']
    for name in ('weights.npz','prepare.json','selection.json','study.json'):
        lines.append(f'| `{name}` | `{run.common.sha(run.OUT/name)}` |')
    temporary=[p for p in run.OUT.rglob('*') if p.is_file() and p.name not in ('REPORT.md','graph-cache-before.py')]
    outside=[run.ROOT/'npunlock-probe/research/results/startup-ir-cache/prefill-checks.json',
             run.ROOT/'npunlock-probe/research/results/model-cross-device-followup/cache-checks.json']
    size=sum(p.stat().st_size for p in [*temporary,*outside])
    files=sum(row['cache_files'] for row in study['variants'].values())
    cached=sum(row['cache_bytes'] for row in study['variants'].values())
    lines+=['', '## 清理记录', '',
        f'验收后清理本轮{len(temporary)+len(outside)}份临时文件，共{size:,}字节（{size/1024**2:.2f}MiB），'
        '包括硬件检查NPC、校准权重、计时JSON和两份离线检查JSON；保留源码与本报告。',
        f'系统临时目录中的配对缓存共{files}份、{cached:,}字节（{cached/1024**2:.2f}MiB），已逐策略删除。'
        '删除后重做审计或复现需重新prepare并运行对应硬件步骤。']
    (run.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Audit passed, NPU calls',total)
    for label in run.LABELS[1:]:
        before=median(variants[label]['full'],'ttft_ms');after=median(variants[label]['fast'],'ttft_ms')
        print(label,round(before,3),'->',round(after,3),'ms',round((1-after/before)*100,2),'% lower')


if __name__=='__main__':main()
