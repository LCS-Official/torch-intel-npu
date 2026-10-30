"""S58: paired exact/FP16-decoded fitting proxies on actual NPU down inputs."""
import argparse
import ast
import hashlib
from pathlib import Path
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import w8_tail_quality as previous
from torch_npu_lowbit import PackedWeight
from torch_npu_lowbit_correction import LowRankCorrection

base=previous.base
parent=previous.previous
basis=parent.previous
attention=previous.study
probe=attention.probe
runner=attention.previous
ROOT=previous.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-decode-proxy'
for module in (previous,parent,basis,attention,runner,runner.previous,probe,attention.device,
               attention.device.previous,base,base.mixed):module.OUT=OUT
NAME='mlp.down_proj.weight'
PROFILES=('exact-a50','decoded-a50')
FRESH=(
    ('teapot','Explain why a teapot usually has a handle.'),
    ('mirror','Why does a bathroom mirror become foggy after a hot shower?'),
    ('tiles','There are nine blue tiles and six white tiles. How many tiles are there in total?'),
    ('ticket','Write a polite message asking to change a train ticket to tomorrow.'),
    ('stapler','Describe how a stapler holds sheets of paper together.'),
    ('lighthouse','Use lighthouse, wind, and evening in one short descriptive sentence.'),
    ('museum','How is a museum different from a library?'),
    ('seeds','Suggest three simple ways to label seed packets in a garden.'))


def proxy_main(weight,x,mode):
    if (not isinstance(weight,PackedWeight) or weight.format!='int8' or weight.group_size!=weight.shape[1]
        or mode not in ('exact','decoded') or not isinstance(x,torch.Tensor) or x.device.type!='cpu'
        or x.dtype!=torch.float16 or x.ndim!=2 or not 1<=len(x)<=4096 or x.shape[1]!=weight.shape[1]
        or max(weight.shape)>4096 or not torch.isfinite(x).all()):
        raise ValueError('Requires bounded finite CPU Half inputs and per-row INT8 weights')
    codes=np.frombuffer(weight.data,dtype=np.int8).reshape(weight.shape).astype(np.float64)
    scales=np.frombuffer(weight.scales,dtype=np.float16).astype(np.float64)
    decoded=torch.from_numpy(codes*scales[:,None])
    if mode=='decoded':decoded=decoded.half().double()
    if not torch.isfinite(decoded).all():raise ValueError('FP16-decoded weight overflows')
    result=(x.double()@decoded.t()).half()
    if not torch.isfinite(result).all():raise ValueError('Projection overflows Half')
    return result


def replace_last_output(factors,up):
    if set(factors)!=set(range(30)) or NAME not in factors[29]:raise ValueError('Expected 30 prepared layers')
    old=factors[29][NAME]
    if (not isinstance(old,LowRankCorrection) or old.shape!=(576,1536) or old.rank!=32
        or not isinstance(up,torch.Tensor) or up.device.type!='cpu' or up.dtype!=torch.float16
        or tuple(up.shape)!=(576,32) or not torch.isfinite(up).all()):raise ValueError('Expected finite matching final down output factor')
    result=dict(factors);result[29]=dict(factors[29])
    result[29][NAME]=LowRankCorrection(old.shape,old.rank,old.down,up.contiguous().numpy().tobytes())
    return result


def data():
    if any((OUT/(name+'.json')).exists() for name in ('tail-data','fresh')):raise FileExistsError('Frozen data already exists')
    old=set(base.mixed.PROMPTS)|{(s,p) for _,s,p in (*base.mixed.HOLDOUT,*base.mixed.FRESH)}
    system='You are a helpful assistant.'
    for prompts in (previous.CALIBRATION,previous.FRESH,parent.FRESH):old|={(system,p) for _,p in prompts}
    for filename in ('w8_guarded_quality.py','w8_native_selection.py'):
        tree=ast.parse((Path(__file__).parent/filename).read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FRESH' for t in n.targets)]
        assert len(nodes)==1;old|={(system,p) for _,p in ast.literal_eval(nodes[0].value)}
    assert len(FRESH)==len({p for _,p in FRESH})==len({tag for tag,_ in FRESH})==8
    assert all((system,p) not in old for _,p in FRESH)
    cpu,tokenizer=base.mixed.common.model()
    for name,prompts,start in (('tail-data',previous.CALIBRATION,40),('fresh',FRESH,56)):
        cases=[]
        for i,(tag,prompt) in enumerate(prompts):
            ids=base.mixed.common.chat(tokenizer,system,prompt);assert 1<=ids.shape[1]<64
            cases.append(dict(tag=tag,prompt=prompt,index=start+i,prefix=ids.shape[1],offset=ids.shape[1]-1,
                token_ids=base.mixed.common.trajectory(cpu,ids,64).tolist()))
        base.write(name,dict(status='frozen',cases=cases,source_sha256=base.sha(Path(__file__)),
            policy='Original48 calibration trajectories only; eight new prompts frozen before observations or fitting.'))


def source_identity():
    paths={'source':Path(__file__),'fitter':Path(basis.__file__),'probe':Path(probe.__file__),
           'data_helper':Path(previous.__file__),'reference_helper':Path(parent.__file__),
           'weights':OUT/'weights.npz','factors':OUT/'corrections.npz','calibration':OUT/'tail-data.json','fresh':OUT/'fresh.json'}
    return {name+'_sha256':base.sha(path) for name,path in paths.items()}


def identity():
    row=base.read('decode-fit');current=source_identity()
    assert row['status']=='passed' and row['healthy'] and all(row[k]==v for k,v in current.items())
    assert base.sha(OUT/'decode-fit.npz')==row['fitted_factors_sha256']
    return dict(**current,fit_sha256=base.sha(OUT/'decode-fit.json'))


def parameters(profile):
    if profile not in ('baseline',*PROFILES):raise ValueError('Unknown decode-proxy profile')
    packed=base.mixed.weights();factors=base.corrections(32)
    if profile=='baseline':return packed,factors
    identity()
    with np.load(OUT/'decode-fit.npz',allow_pickle=False) as arrays:
        assert set(arrays.files)==set(PROFILES)
        factors=replace_last_output(factors,torch.from_numpy(arrays[profile].copy()))
    return packed,factors


def fit():
    if any((OUT/name).exists() for name in ('decode-fit.npz','decode-fit.json')):raise FileExistsError('Refusing to overwrite fit')
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolLayer,FrozenSmolStack
    report=dict(status='running',local_checks=[],controls=[],proxies={},native_calibration={},**source_identity())
    def work():
        cpu,_=base.mixed.common.model();packed,factors=parameters('baseline')
        stack=FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,packed_weights=packed,corrections=factors)
        rows=previous.fit_rows(basis.partition(base.read('prepare')['trajectories'],True),base.read('tail-data')['cases'])
        observed={k:[] for k in ('gated','reduced','main','corrected','target')};bindings=[]
        for index,row in enumerate(rows):
            x=cpu.model.embed_tokens(torch.tensor(row['token_ids'])).contiguous().to('intel_npu')
            for i,layer in enumerate(stack.layers[:29]):
                expected=layer.reference_cpu(x.cpu())[0] if index==0 else None;x=layer(x)
                if expected is not None:
                    error=attention.device.compare(x.cpu(),expected);report['local_checks'].append(dict(layer=i,**error))
                    assert error['relative_l2']<.003 and error['max_scaled_error']<.01
            layer=stack.layers[29];residual=layer._run_fused(x,segment='attention')[0]
            normalized=layer._norm_device(residual,'post_attention_layernorm.weight')
            graph,xml,_=layer.graph(tuple(x.shape),segment='ffn')
            values,_=probe.run_graph(npu,probe.ffn_probe_graph(graph,xml),[residual,normalized])
            if index in (0,23,47):
                whole=layer(x);check=attention.device.compare(values['output'].cpu(),whole.cpu())
                assert check['exact_fraction']==1;report['controls'].append(dict(trajectory=row['index'],**check))
                if index==0:
                    error=attention.device.compare(whole.cpu(),layer.reference_cpu(x.cpu())[0]);report['local_checks'].append(dict(layer=29,**error))
                    assert error['relative_l2']<.003 and error['max_scaled_error']<.01
            gated=values['gated'].cpu();target=F.linear(gated.float(),layer._cpu_weight(NAME).float()).half()
            for name,value in (('gated',gated),('reduced',values['down_reduced'].cpu()),('main',values['down_main'].cpu()),
                               ('corrected',values['down_corrected'].cpu()),('target',target)):observed[name].append(value)
            bindings.append((residual,values['gated']));print('native down calibration',index+1,'/48',flush=True)
        data={name:torch.cat(values) for name,values in observed.items()}
        assert len(rows)==48 and tuple(data['gated'].shape)==(3072,1536) and len(report['local_checks'])==30
        def sse(value):return float((value.double()-data['target'].double()).square().sum())
        report['native_calibration']['baseline']=dict(sse=sse(data['corrected']))
        tail=probe.down_tail_graph(graph,xml);control,_=probe.run_graph(npu,tail,list(bindings[0]))
        check=attention.device.compare(control['corrected'].cpu(),data['corrected'][:64])
        assert check['exact_fraction']==1;report['controls'].append(dict(trajectory=rows[0]['index'],retained_binding=True,**check))
        prior=factors[29][NAME];arrays={}
        for mode,profile in zip(('exact','decoded'),PROFILES,strict=True):
            main=proxy_main(packed[29][NAME],data['gated'],mode)
            factor,metrics=basis.fit_device_basis(data['reduced'],main,data['target'],prior,.5)
            up=factor.tensors()[1];arrays[profile]=up.numpy().copy()
            report['proxies'][profile]=dict(**metrics,main_vs_native=attention.device.compare(main,data['main']))
            changed=replace_last_output(factors,up)
            layer=FrozenSmolLayer.from_checkpoint(base.mixed.common.MODEL,layer_index=29,
                packed_weights=packed[29],corrections=changed[29])
            graph,xml,_=layer.graph((1,64,576),segment='ffn');tail=probe.down_tail_graph(graph,xml);outputs=[]
            for residual,gated in bindings:
                values,_=probe.run_graph(npu,tail,[residual,gated]);outputs.append(values['corrected'].cpu())
            report['native_calibration'][profile]=dict(sse=sse(torch.cat(outputs)))
            print('proxy fit',profile,report['proxies'][profile],report['native_calibration'][profile],flush=True)
        path=OUT/'decode-fit.npz';np.savez_compressed(path,**arrays)
        report.update(fitted_factors_sha256=base.sha(path),fit_indices=[r['index'] for r in rows],rows=3072,rank=32,
            changed_layer=29,changed_projection=NAME,alpha=.5,trainable_coefficients=576*32,
            observed_sha256={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in data.items()},stats=npu.stats())
        assert source_identity()=={k:report[k] for k in source_identity()}
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return attention.device.finish(report,'decode-fit',npu)


def select():
    if (OUT/'selection.json').exists():raise FileExistsError('Choice is already frozen')
    records={p:base.read('internal-'+p) for p in ('baseline',*PROFILES)};current=identity()
    assert all(all(r[k]==v for k,v in current.items()) for r in records.values())
    selected,candidate,rejected=runner.choose(records)
    base.write('selection',dict(status='frozen',selected=selected,candidate=candidate,rejected=rejected,**current,
        summaries={p:r.get('summary') for p,r in records.items()},records_sha256={p:base.sha(OUT/('internal-'+p+'.json')) for p in records},
        policy='Two fixed alpha=.5 proxies, last-layer down output factor only. Original internal gates/ranking/promotion; '
        'native calibration SSE is diagnostic. Freeze before eight new prompts; never refit on fresh results.'))
    print('DECODE PROXY SELECTED',selected,'DIAGNOSTIC',candidate,flush=True)


parent.PROFILES=PROFILES
parent.parameters=parameters
parent.identity=identity

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','fit','internal','select','fresh'))
    parser.add_argument('--profile',choices=('baseline',*PROFILES),default='baseline')
    parser.add_argument('--label',choices=('baseline','candidate'),default='candidate')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():
        if args.stage=='prepare':basis.prepare();result=data()
        elif args.stage=='fit':result=fit()
        elif args.stage=='internal':result=parent.internal(args.profile)
        elif args.stage=='select':result=select()
        else:result=parent.fresh(args.label)
    raise SystemExit(result or 0)
