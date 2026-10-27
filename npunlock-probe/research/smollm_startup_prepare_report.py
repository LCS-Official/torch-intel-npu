"""Audit lightweight CPU endpoints and paired cached NPU startup measurements."""
import statistics as st

import smollm_startup_prepare as run


def med(rows,key):return st.median(row[key] for row in rows)
def cold(rows,key):return st.median(row['cold'][key] for row in rows)
def fmt(value):return f'{value:.3f}'


def construction_check():
    import torch
    import torch_npu_smollm as smol
    from unittest.mock import patch
    torch.set_num_threads(8);ppl=run.helpers();packed=ppl.layers.mixed.weights()
    original=smol.FrozenSmolLayer.__init__;calls=[]
    def counted(self,*args,**kwargs):
        calls.append(1);original(self,*args,**kwargs)
    with patch.object(smol.FrozenSmolLayer,'__init__',counted):
        current=ppl.layers.make_stack('mixed',packed);new_count=len(calls)
        old=run.legacy_stack('mixed',packed);old_count=len(calls)-new_count
    assert new_count==30 and old_count==60
    for a,b in zip(current.layers,old.layers):
        assert a._weights==b._weights and a._packed==b._packed and a.fp16_projections==b.fp16_projections
        assert (a._cos,a._sin)==(b._cos,b._sin)
    run.write('construction-check',dict(status='passed',new_layer_constructions=new_count,
        old_layer_constructions=old_count,all_layer_bytes_equal=True))


def main():
    study=run.read('study');io_check=run.read('io-check')
    assert study['status']=='passed' and study['caches_removed']
    assert io_check['status']=='passed' and len(io_check['checks'])==12
    if not (run.OUT/'construction-check.json').exists():construction_check()
    construction=run.read('construction-check')
    assert construction==dict(status='passed',new_layer_constructions=30,old_layer_constructions=60,all_layer_bytes_equal=True)
    names=dict(fp16='FP16',w8='校准W8',mixed='混合50%',selected='逐层37.5%')
    variants={};all_rows=[]
    for label in run.previous.LABELS:
        fill=run.read(label+'-fill');rows=[run.read(tag) for tag in study['variants'][label]['rows']]
        assert len(rows)==6 and [row['route'] for row in rows]==['before','after','after','before','before','after']
        for row in [fill,*rows]:
            assert row['status']=='passed' and len(row['samples'])==7 and len(row['graphs'])==30
            assert row['cold']['arrays']==fill['cold']['arrays'] and len(row['cold']['arrays'])==91
            assert row['cold']['embedding_sha256']==fill['cold']['embedding_sha256']
            assert row['cold']['logits_sha256']==fill['cold']['logits_sha256'] and row['cold']['token']==fill['cold']['token']
            assert row['graphs']==fill['graphs']
            for sample in row['samples']:
                assert sample['arrays']==fill['cold']['arrays'] and sample['logits_sha256']==fill['cold']['logits_sha256']
                assert sample['delta']['npu_calls']==30 and sample['delta']['session_creations']==sample['delta']['driver_allocations']==0
            if row['generation']:
                assert row['generation']==fill['generation'] and len(row['generation']['tokens'][0])==56
                assert len(row['generation']['logits_sha256'])==32 and len(row['generation']['arrays'])==91
            assert row['final_stats']['npu_calls']==360+(row['generation']['npu_calls'] if row['generation'] else 0)
            assert row['final_stats']['closed'] and not row['final_stats']['context_failed']
            assert all(row['final_stats'][key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
            for path,digest in row['sources'].items():assert run.previous.common.sha(run.ROOT/path)==digest
            all_rows.append(row)
        for row in rows:
            counts=row['cold_cache']
            assert counts['hits']==counts['ir_skipped']==30 and counts['compiles']==counts['rejected']==0
        assert sum(bool(row['generation']) for row in rows)==2
        variants[label]={route:[row for row in rows if row['route']==route] for route in ('before','after')}
    total=sum(row['final_stats']['npu_calls'] for row in all_rows)
    lines=['# SmolLM启动：精简CPU模型与重复权重准备', '',
        '2026-10-16，Windows / Core Ultra 7 258V / Lunar Lake NPU4000 `8086:643e`，'
        '驱动32.0.100.5540、图编译器8.3、Torch2.14.1+cpu。', '',
        '新增可复用的 `FrozenSmolIO`：只加载CPU embedding、最终RMSNorm和共享输出权重，'
        '替代NPU推理时未使用的完整CPU decoder。另将混合50%的研究入口改为既有公开构造参数一次准备，'
        '省去30层的第二次构建。模型计算、BF16→FP16转换检查、量化权重及DPU/SHAVE/原生运行时未改变。', '',
        '## 真实NPU：已有缓存的新进程', '',
        '| 配置 | 原准备 ms | 新准备 ms | 原脚本入口到首token s | 新入口到首token s | 入口耗时降低 |',
        '|---|---:|---:|---:|---:|---:|']
    for label in run.previous.LABELS:
        before,after=variants[label]['before'],variants[label]['after']
        old,new=cold(before,'entry_to_token_ms'),cold(after,'entry_to_token_ms')
        lines.append(f'| {names[label]} | {fmt(med(before,"setup_ms"))} | {fmt(med(after,"setup_ms"))} | '
            f'{fmt(old/1000)} | {fmt(new/1000)} | {(1-new/old)*100:.2f}% |')
    lines+=['', '“准备”包括CPU模型/轻量入口、tokenizer、校准权重校验与解包、NPU层构造及末尾GC；'
        '脚本入口另包括导入和NPU初始化。没有把准备挪到入口计时之外。', '',
        '模型准备后到首token（每种3次中位数）：', '',
        '| 配置 | 原 ms | 新 ms |', '|---|---:|---:|']
    for label in run.previous.LABELS:
        lines.append(f'| {names[label]} | {fmt(cold(variants[label]["before"],"ttft_ms"))} | {fmt(cold(variants[label]["after"],"ttft_ms"))} |')
    lines+=['', '这轮收益来自CPU准备和模型构造；没有改善NPU芯片算力或认证热decode吞吐提升。'
        '准备后的首token存在桌面负载波动，应按原始配对数据阅读。', '',
        '## 方法', '',
        '- 完整30层SmolLM2-135M-Instruct、batch1、24-token固定提示，decoder与KV在NPU；CPU两端始终显式。'
        'Torch CPU8线程、interop1线程。沿用原来的完整24位置CPU输出头计算。',
        '- 四种精度保持固定：FP16、全部校准W8、全部Up/Down保留FP16、先前45项逐层保护。'
        '同一份逐行GPTQ校准权重、codes/scales与保护项未调整；权重NPZ SHA256 '
        '`f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62`。',
        '- 每种精度先填充60份prefill/decode缓存，再做3对全新进程，顺序before→after、after→before、before→after。'
        '相同缓存目录、相同编译blob；首请求30次命中、0次编译、30次跳过IR。',
        '- before保持完整CPU Transformers模型；混合50%还保留旧研究入口“先建30层再替换30层”的构造方式。'
        'after使用FrozenSmolIO及单次全局混合构造；逐层37.5%仍用原研究组合，没有新增公开逐层策略API。',
        '- 每进程首请求后预热2次、记录7次热prefill。各精度填充进程和第一对进程另运行完整greedy32-token生成，'
        '忽略EOS，最大消费55个位置；生成在启动计时与内存快照之后。',
        '- 输出/状态hash及数值检查在计时区间外。未锁定频率或桌面负载，未清空系统文件页缓存；'
        '三对样本只代表本机此次条件。无缓存编译、其它模型、长上下文、训练和GPU协同不在范围内。', '',
        '## 独立CPU分段', '',
        '独立CPU profile排除了profile前已完成的库导入；'
        '与上面的入口计时口径不同，不能将两表直接相减。hook为inclusive，不能与分段相加。', '',
        '| 配置 | 完整CPU模型 ms | tokenizer ms | 校准权重校验/解包 ms | stack ms | 层构造次数 |',
        '|---|---:|---:|---:|---:|---:|']
    for label in run.previous.LABELS:
        row=run.read('profile-'+label);s=row['stages_ms']
        lines.append(f'| {names[label]} | '+' | '.join(fmt(s[key]) for key in ('full_cpu_model','tokenizer','packed_load_verify','stack'))+
            f' | {row["hooks"]["layer_init"]["calls"]} |')
    lines+=['', 'CPU校准权重加载约0.69秒仍保留完整源文件/模型SHA检查、NPZ解压与PackedWeight验证。'
        '当前checkpoint是BF16；NPU层的原BF16→FP32→FP16准备及溢出/非有限检查保持原样。', '',
        '## 实际新进程的准备分段', '',
        '| 配置/路径 | CPU入口与tokenizer ms | 校准权重校验/解包 ms | stack ms | 总准备 ms |', '|---|---:|---:|---:|---:|']
    for label in run.previous.LABELS:
        for route in ('before','after'):
            rows=variants[label][route]
            values=[st.median(row['stages_ms'][key] for row in rows) for key in ('cpu_endpoints_and_tokenizer','packed_load_verify','stack')]
            lines.append('| '+names[label]+'/'+route+' | '+' | '.join(fmt(value) for value in [*values,med(rows,'setup_ms')])+' |')
    lines+=['', '分项中位数不必相加为总中位数；总准备额外包括GC。CPU入口分项还包含首次所需的Transformers类导入。', '',
        '## 进程内存', '',
        '| 配置 | 原准备后私有提交 MiB | 新准备后私有提交 MiB | 原启动阶段峰值提交 MiB | 新启动阶段峰值提交 MiB |',
        '|---|---:|---:|---:|---:|']
    for label in run.previous.LABELS:
        values=[]
        for route in ('before','after'):
            rows=variants[label][route];values.append(st.median(row['prepared_memory']['private_bytes'] for row in rows)/1024**2)
        for route in ('before','after'):
            rows=variants[label][route];values.append(st.median(row['startup_memory']['peak_commit_bytes'] for row in rows)/1024**2)
        lines.append('| '+names[label]+' | '+' | '.join(fmt(value) for value in values)+' |')
    lines+=['', 'Windows GetProcessMemoryInfo的PrivateUsage/PeakPagefileUsage为进程私有提交/截至快照的峰值提交，'
        '不是已写入pagefile的字节或NPU独占显存。启动峰值在后续32-token生成之前取样；'
        '包含导入、加载、缓存图与NPU会话、prefill及区间外hash验证。未把驱动/固件的所有内存归入此数。', '',
        '## 正确性与资源', '',
        '- FrozenSmolIO只依赖Torch与safetensors，不导入Transformers、npunlock或NPU运行时；加载两份CPU权重并生成自有FP16快照，'
        '输出矩阵复用embedding，不额外复制输出权重。tokenizer仍由调用者提供。',
        '- 5项新离线测试、10项既有模型/保护策略检查、5项source-export检查通过，共20项。'
        '新测试覆盖权重所有权、tied矩阵、checkpoint、dtype/shape/设备/训练模式、非法token ID、非有限值及FP16溢出拒绝。',
        '- 独立构造计数确认混合50%从60次层构造降至30次；所有层原FP16字节、PackedWeight、保护策略与RoPE字节一致。'
        'torch_npu_smollm.py、图缓存实现与原生二进制指纹未变，可复用上一轮图缓存。',
        '- 实际checkpoint下4种embedding形状与12种输出头样本（random/zero/最大有限FP16，含batch8/长度64）'
        '与官方FP16 Transformers输出逐位一致；不是只比较近似误差。',
        '- 每个精度的相同prefill blob、embedding、hidden、30层K/V/position共91个Tensor、首token logits与argmax跨新旧入口逐位一致；'
        '每进程的7次热样本也逐位一致。',
        '- 12次完整32-token生成的每步logits、全部生成token及最终91项状态逐位一致（每种精度：填充、before-1、after-1）。'
        '生成使用各自greedy路径，未强制喂参考token。此验证不改变低位模型质量门槛。',
        '- 各进程的prefill图验证原生DPU模式；W8/混合每层包含amode0/wmode2整数权重矩阵任务，FP16不含整数权重任务。'
        '热prefill没有新会话或驱动allocation。',
        f'- 本轮共 **{total:,}次真实NPU调用**；所有进程关闭后live/cache/quarantine/session归零，context_failed=False。', '',
        '## 逐对原始启动数据', '',
        '| 配置/进程 | 准备 ms | 模型准备后首token ms | 脚本入口到首token ms |', '|---|---:|---:|---:|']
    for label in run.previous.LABELS:
        for tag in study['variants'][label]['rows']:
            row=run.read(tag)
            lines.append(f'| {tag} | {fmt(row["setup_ms"])} | {fmt(row["cold"]["ttft_ms"])} | {fmt(row["cold"]["entry_to_token_ms"])} |')
    lines+=['', '## 使用与复现', '',
        '轻量入口是可复用的本地开发API；需要显式替换完整CPU模型的两端计算。'
        '既有完整CPU参考/校准脚本保留其参考模型，不会被自动替换。当前仅支持canonical tied SmolLM、D576、eps1e-5，'
        '不支持独立lm_head权重、训练或通用模型导入；默认NPU精度与图缓存开关不变。', '',
        '```python', 'from torch_npu_smollm_io import FrozenSmolIO',
        'io = FrozenSmolIO.from_checkpoint(model_directory)',
        'x = io.embed_tokens(cpu_token_ids).to("intel_npu")',
        'hidden, kv = stack.prefill(x)',
        'logits = io.logits(hidden.cpu())[:, -1].float()', '```', '',
        '```powershell', "$env:PYTHONDONTWRITEBYTECODE='1'", "$env:PYTHONIOENCODING='utf-8'", "$env:NPUNLOCK_GRAPH_CACHE=$null",
        '.\\run_torch_npu.ps1 -Script .\\test_torch_npu_smollm_io.py',
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_startup_prepare.py -ScriptArguments '--case','prepare'",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_startup_prepare.py -ScriptArguments '--case','profile'",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_startup_prepare.py -ScriptArguments '--case','io-check'",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_startup_prepare.py -ScriptArguments '--case','study'",
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_startup_prepare_report.py', '```', '',
        '本轮保留的profile在入口修改前采集，mixed为60次层构造；当前代码重新profile时会直接构建30层，数据随实现变化。'
        '正式study包含独立legacy_stack对照，不依赖删除校验、改变精度或重新编译不同计算图。', '',
        '## 剩余方向', '',
        '校准权重校验/解包和导入仍是启动成本；后续可评估不改变codes/scales的高效存储与解包方式，完整性检查保留。'
        '稳态NPU性能仍需单独研究图提交与设备任务，不把本轮CPU准备收益套用到tokens/s或TOPS。', '',
        '## 指纹', '', '| 对象 | SHA256 |', '|---|---|']
    sources=dict(all_rows[0]['sources'])
    for name in ('test_torch_npu_smollm_io.py','npunlock-probe/research/smollm_startup_prepare_report.py'):
        sources[name]=run.previous.common.sha(run.ROOT/name)
    for path,digest in sources.items():lines.append(f'| `{path}` | `{digest}` |')
    lines+=['', '校准文件与实验清单清理前指纹：', '', '| 材料 | SHA256 |', '|---|---|']
    for name in ('weights.npz','prepare.json','selection.json','study.json','io-check.json','construction-check.json'):
        lines.append(f'| `{name}` | `{run.previous.common.sha(run.OUT/name)}` |')
    temporary=[p for p in run.OUT.rglob('*') if p.is_file() and p.name!='REPORT.md']
    size=sum(p.stat().st_size for p in temporary)
    cache_files=sum(row['cache_files'] for row in study['variants'].values())
    cache_bytes=sum(row['cache_bytes'] for row in study['variants'].values())
    lines+=['', '## 交付与清理', '',
        f'验收后按用户约定清理{len(temporary)}份临时校准权重、原始计时/验收JSON及冗余适配器副本，'
        f'共{size:,}字节（{size/1024**2:.2f}MiB）；保留新增CPU入口、测试、研究源码和本报告。',
        f'系统临时目录的{cache_files}份图缓存、{cache_bytes:,}字节（{cache_bytes/1024**2:.2f}MiB）已逐精度删除，'
        '没有安装新依赖、下载模型、生成图片或保留持久图缓存，未推送GitHub。'
        '清理后重算审计需先重新prepare和执行测量。']
    (run.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Audit passed; NPU calls',total)
    for label in run.previous.LABELS:
        before,after=variants[label]['before'],variants[label]['after']
        print(label,round(cold(before,'entry_to_token_ms'),3),'->',round(cold(after,'entry_to_token_ms'),3),'ms')


if __name__=='__main__':main()
