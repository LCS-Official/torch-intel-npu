"""Validate temporary startup experiments, then retain their measurements in Markdown."""
import ast
import hashlib
import json
import statistics as st
import zipfile
import numpy as np
from model_cross_device import ROOT,sha
from startup_first_token import OUT


def main():
    baseline=('load-1','load-2','load-3');candidate=('stored-1','stored-2','stored-3')
    production=('promoted-fill','promoted-load-1','promoted-load-2','promoted-corrupt')
    labels=('fill',)+baseline+candidate+production
    reports={name:json.loads((OUT/(name+'.json')).read_text()) for name in labels}
    before=OUT/'before/torch_npu_graph_cache.py';current=ROOT/'torch_npu_graph_cache.py'
    with np.load(OUT/'fill-outputs.npz',allow_pickle=False) as data:anchor={k:data[k].copy() for k in data.files}
    for label,r in reports.items():
        assert r['status']=='passed' and r['quality']['relative_l2']<.03 and r['quality']['argmax_equal']
        assert r['token']==reports['fill']['token'] and len(r['warm'])==7
        assert r['sources']['torch_npu_smollm.py']==sha(ROOT/'torch_npu_smollm.py')
        assert r['sources']['torch_npu_graph_cache.py']==sha(current if label in production else before)
        assert r['checkpoint_sha256']==reports['fill']['checkpoint_sha256']
        assert r['final_stats']['closed'] and not r['final_stats']['context_failed']
        assert all(r['final_stats'][k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
        path=OUT/(label+'-outputs.npz');assert sha(path)==r['outputs_sha256']
        with np.load(path,allow_pickle=False) as data:
            assert set(data.files)==set(anchor)
            for key in anchor:np.testing.assert_array_equal(data[key].view(np.uint8),anchor[key].view(np.uint8))
    for label in baseline+candidate+production[1:3]:
        r=reports[label]['cache'];assert r['hits']==30 and r['compiles']==r['rejected']==r['writes']==0
    r=reports['promoted-fill']['cache'];assert r['writes']==r['compiles']==30 and r['hits']==0
    r=reports['promoted-corrupt']['cache'];assert r['hits']==29 and r['rejected']==r['compiles']==r['writes']==1
    a=ast.parse(before.read_text());b=ast.parse(current.read_text());changed=0
    for node in ast.walk(b):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='ZipFile':
            for keyword in node.keywords:
                if keyword.arg=='compression' and isinstance(keyword.value,ast.Attribute) and keyword.value.attr=='ZIP_STORED':
                    keyword.value.attr='ZIP_DEFLATED';node.keywords.append(ast.keyword(arg='compresslevel',value=ast.Constant(value=1)));changed+=1
    assert changed==1 and ast.dump(a)==ast.dump(b),'Production change exceeded ZIP storage format'
    checks=json.loads((ROOT/'npunlock-probe/research/results/model-cross-device-followup/cache-checks.json').read_text())
    assert checks['status']=='passed' and len(checks['checks'])==19 and checks['source_sha256']==sha(current)
    sizes={}
    for name,kind in (('cache',zipfile.ZIP_DEFLATED),('cache-stored',zipfile.ZIP_STORED),('cache-promoted',zipfile.ZIP_STORED)):
        files=list((OUT/name).glob('*.npc'));assert len(files)==30
        size=sum(p.stat().st_size for p in files);assert size<=512*1024*1024;sizes[name]=size
        assert not (OUT/name/'.writer.lock').exists() and not list((OUT/name).glob('*.tmp'))
        for path in files:
            with zipfile.ZipFile(path) as archive:
                assert all(v.compress_type==kind for v in archive.infolist())
                manifest=json.loads(archive.read('manifest.json'));blob=archive.read('graph.blob')
                assert hashlib.sha256(blob).hexdigest()==manifest['blob_sha256']
    def median(group,fn):return st.median(fn(reports[name]) for name in group)
    old=median(baseline,lambda r:r['cold']['ttft_ms']);new=median(candidate,lambda r:r['cold']['ttft_ms'])
    entry_old=median(baseline,lambda r:r['entry_to_token_ms']);entry_new=median(candidate,lambda r:r['entry_to_token_ms'])
    lines=['# NPU 冷首 token：方向诊断与缓存加载优化','',
        '2026-10-11，Core Ultra 7 258V / Intel AI Boost 8086:643e，驱动32.0.100.5540，图编译器8.3。',
        '结论：优先缩短已有编译图缓存命中后的准备开销。首个已接入优化为免解压图缓存，默认仍关闭磁盘缓存。','',
        '## 计时定义与范围','',
        '- 同一SmolLM2-135M-Instruct、FP16、完整30层、batch1、24-token提示词，只测prefill到首token。',
        '- 冷首token：模型加载完成后，从embedding开始到首token argmax可用，包括上传、图准备、NPU prefill、下载及CPU完整prefill输出头。',
        '- 脚本入口到首token：包含Torch/Transformers导入、NPU初始化、模型和tokenizer加载；从Python脚本开始计时，不含PowerShell/解释器启动。',
        '- 每个独立进程只有一次冷请求；额外预热2次再测7次热请求。NPU调用同步，未提前预热要测的图/session。',
        '- 新进程测试没有清空Windows文件系统页缓存；属于进程冷启动。后台负载/频率未锁定。',
        '- 上轮6.229秒是首次完整生成32 token，不能与本轮首token秒数直接比较。',
        '- 临时hook记录包含式时间，各项可能嵌套，不能简单相加；所有模式采用相同hook。没有提前转移工作来美化首token。','',
        '## 缓存加载格式对照','',
        '两种格式来自相同的30份blob和manifest，重打包时逐项验证字节一致；缓存键、模型图和运行路径相同。',
        '各取3个新进程，执行顺序为压缩1、免压缩1、压缩2、免压缩2、免压缩3、压缩3。','',
        '| 新进程 | 冷首token ms | 读缓存 ms | 脚本入口到首token ms | 热首token中位 ms |',
        '|---|---:|---:|---:|---:|']
    for label in baseline+candidate:
        r=reports[label]
        lines.append(f"| {label} | {r['cold']['ttft_ms']:.3f} | {r['cold']['hooks']['cache_read']['ms']:.3f} | {r['entry_to_token_ms']:.3f} | {r['warm_ttft']['median']:.3f} |")
    lines+=['',f'冷首token中位数 **{old/1000:.3f}→{new/1000:.3f}秒**，降低约{100*(1-new/old):.1f}%。',
        f'脚本入口到首token中位数 **{entry_old/1000:.3f}→{entry_new/1000:.3f}秒**，降低约{100*(1-entry_new/entry_old):.1f}%。',
        '这是小样本本机结果，不是跨负载速度保证；没有声称热内核或tokens/s提升。','',
        '| 冷prefill准备阶段 | 压缩缓存三次中位 ms | 免压缩缓存三次中位 ms |',
        '|---|---:|---:|']
    for key in ('graph_build_serialize','cache_read','inspect_blob','session_create','norm_parameter','layer_run_many'):
        lines.append(f"| {key} | {median(baseline,lambda r:r['cold']['hooks'][key]['ms']):.3f} | {median(candidate,lambda r:r['cold']['hooks'][key]['ms']):.3f} |")
    lines+=['','## 正式接入与恢复验收','',
        '生产代码仅将缓存写入从`ZIP_DEFLATED(level=1)`改成`ZIP_STORED`。AST还原检查确认无其他计算逻辑改动。',
        '图/权重/内核/驱动/编译器身份检查、SHA256、大小检查、原子写入、容量限制和失败回退全部保留。',
        '读取器仍能读取压缩格式；因缓存键保守包含实现源码哈希，本次代码更新会使旧键失效，首次需重建。','',
        '| 正式生产入口 | 冷首token ms | 脚本入口到首token ms | 命中 | 融合图编译 | 写入 |',
        '|---|---:|---:|---:|---:|---:|']
    for label in production:
        r=reports[label];c=r['cache']
        lines.append(f"| {label} | {r['cold']['ttft_ms']:.3f} | {r['entry_to_token_ms']:.3f} | {c['hits']} | {c['compiles']} | {c['writes']} |")
    lines+=['',f"首次写缓存累计耗时由旧格式试跑的{reports['fill']['cache']['write_ms']/1000:.3f}秒降到正式入口的{reports['promoted-fill']['cache']['write_ms']/1000:.3f}秒；两次首次编译未配对，不把整个调用差值都归因于格式。",
        '损坏一个临时条目后29命中、1拒绝、1重编译并恢复，无设备故障重试。',
        '19项缓存保护/格式兼容检查通过；11个进程全部首token logits和prefill hidden逐位一致，',
        f"与完整CPU FP16前向的logits相对L2为{100*reports['fill']['quality']['relative_l2']:.4f}%，argmax一致，仍使用3%门槛。",
        '每进程7次热请求均与本进程冷结果逐位一致，最终live/cache/quarantine/session归零，无context failure。','',
        '## 空间代价与使用','',
        f"30份prefill图：旧格式{sizes['cache']/1024**2:.2f}MiB，新格式{sizes['cache-promoted']/1024**2:.2f}MiB，增加约{(sizes['cache-promoted']-sizes['cache'])/1024**2:.2f}MiB。",
        '总缓存上限仍512MiB，保留256MiB磁盘余量；关闭缓存不新增这些文件。一个模型的不同batch/prefix形状不保证都能放入上限。',
        '本轮只生成30份prefill图，未测decode完整生成的正式新格式缓存容量。上一轮60图原生blob总量约438MiB，仅作容量参考。',
        '**按用户约定，本轮验收后清理所有临时数组、计时JSON和图缓存，仅保留源码和报告。**',
        '实际使用时要保留缓存才能获得下次新进程的命中速度；每次删除缓存会再次编译。本轮正式首次编译首token仍需约27秒。','',
        '```powershell',
        "$env:NPUNLOCK_GRAPH_CACHE=Join-Path $env:LOCALAPPDATA 'torch-intel-npu\\smollm-graph-cache'",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\startup_first_token.py -ScriptArguments '--label','first'",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\startup_first_token.py -ScriptArguments '--label','next-process'",
        'Remove-Item Env:NPUNLOCK_GRAPH_CACHE',
        '```','',
        '测试完成、报告生成后，使用`research/clean_experiment_artifacts.ps1 -Apply`清理可重建实验产物和默认缓存目录。','',
        '## 接下来的优化顺序','',
        '1. 缓存命中时仍有约0.8秒的图构建/权重序列化：研究复用经过严格身份校验的I/O合同，跳过重建完整IR；需防止权重/形状变更误命中。',
        '2. 减少启动依赖导入和不需要的CPU decoder权重加载。当前仅用CPU embedding/norm/head，但参考驱动仍加载完整HF模型。',
        '3. 再优化blob检查重复解析及常量gamma准备；保留矩阵精度证明和异常保护。',
        '4. 热首token/prefill计算与batch8重复分配另开受控对照；它们不解释本轮秒级冷启动等待。',
        '尚未实现1–4，未测试更多提示词/上下文、功耗、完整32-token新格式生成或LLM训练。',
        '首次试跑库导入21.56秒，后续约3–4秒；系统页缓存及后台原因未独立确认，保留该异常，不把它算作NPU内核改进。','',
        f'生产缓存源码SHA256：`{sha(current)}`。',
        f'模型实现SHA256未变：`{sha(ROOT/"torch_npu_smollm.py")}`。',
        '原始数据在清理前已完成上述离线验证；清理后历史audit不能原样复算，复测需重新生成数据。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',processes=len(labels),cold_ms_before=old,cold_ms_after=new,
        entry_ms_before=entry_old,entry_ms_after=entry_new,all_outputs_bit_equal=True,cache_checks=19),indent=2))


if __name__=='__main__':main()
