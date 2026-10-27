"""Inspect device-only quantile ranks; diagnostic downloads are not a runtime path."""
import torch
import torch_intel_npu as n
import torch_npu_order_stats as stats
with torch.no_grad():
    q=n._scalar(1e-39,torch.float32)
    print('q',q.cpu(),q.cpu().view(torch.int32),flush=True)
    print('multiply',n._binary_op(q,1.,kind='mul').cpu(),flush=True)
    for value in (1e-39,1e-37,.25):
        result,a,b,w,*_=stats._stat(torch.tensor([0.,1e38],device='intel_npu'),None,False,'quantile',value,'linear')
        print(value,'result',result.cpu(),'a',a.cpu(),'b',b.cpu(),'w',w.cpu(),'bits',w.cpu().view(torch.int32),flush=True)
        del result,a,b,w
del q
n.shutdown()
