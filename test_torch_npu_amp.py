"""Offline AMP control-state contracts and independent CPU model references."""
import unittest
import pickle
import torch
from torch_npu_amp import GradScaler, register_grad_scaler
from torch_npu_amp_demo import amp_model
from torch_npu_amp_conv_demo import geometry_cases, sample, conv_oracles, reference_conv, model as conv_model


class AMPControls(unittest.TestCase):
    def test_standard_constructor_preserves_aliases_and_cpu_scaler(self):
        from torch.amp import GradScaler as alias
        from torch.amp.grad_scaler import GradScaler as module_alias
        previous=alias.__dict__.get('__new__')
        try:
            register_grad_scaler();register_grad_scaler()
            self.assertIs(torch.amp.GradScaler,alias)
            self.assertIs(module_alias,alias)
            for factory in (alias,module_alias):
                scaler=factory('intel_npu',128.,2.,.5,3)
                self.assertIsInstance(scaler,GradScaler)
                self.assertIsInstance(scaler,alias)
                self.assertEqual(scaler.state_dict()['scale'],128.)
                restored=factory(device='intel_npu');restored.load_state_dict(scaler.state_dict())
                self.assertEqual(restored.state_dict(),scaler.state_dict())
            cpu=alias('cpu',init_scale=8.)
            self.assertIs(type(cpu),alias)
            p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.1)
            cpu.scale(p.square()).backward();cpu.step(opt);cpu.update()
            torch.testing.assert_close(p,torch.tensor(.8))
            self.assertFalse(alias('intel_npu',enabled=False).is_enabled())
            self.assertIs(type(alias('xpu',enabled=False)),alias)
            restored=pickle.loads(pickle.dumps(scaler))
            self.assertEqual(restored.state_dict(),scaler.state_dict())
        finally:
            if previous is None:delattr(alias,'__new__')
            else:setattr(alias,'__new__',previous)

    def test_empty_and_nested_iterables_keep_lazy_control_initialization(self):
        scaler=GradScaler()
        self.assertEqual(scaler.scale([]),[])
        self.assertEqual(scaler.scale(([],())),([],()))
        self.assertEqual(list(scaler.scale(iter(()))),[])
        self.assertIsNone(scaler._scale)
        for invalid in ('loss',None,2.):
            with self.assertRaises(ValueError):scaler.scale(invalid)

    def test_fp32_opmath_matches_native_scale_promotion(self):
        for dtype in (torch.float16,torch.float32):
            for shape in ((),(4,)):
                value=torch.tensor(.125,dtype=dtype) if not shape else torch.tensor([0.,-.0,.125,-.25],dtype=dtype)
                scaler=torch.amp.GradScaler('cpu')
                actual=scaler.scale(value);expected=value.float()*scaler.get_scale()
                if value.ndim:expected=expected.to(value.dtype)
                self.assertEqual(actual.dtype,expected.dtype)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                self.assertTrue(torch.equal(torch.signbit(actual),torch.signbit(expected)))

    def test_scaler_state_validation_is_atomic(self):
        scaler=GradScaler(init_scale=128.,growth_interval=3)
        state=scaler.state_dict();state['_growth_tracker']=2
        scaler.load_state_dict(state);self.assertEqual(scaler.state_dict(),state)
        with self.assertRaisesRegex(RuntimeError,'empty'):scaler.load_state_dict({})
        self.assertEqual(scaler.state_dict(),state)
        for changed in ({'scale':float('nan')},{'scale':0.},{'growth_factor':1.},
                        {'backoff_factor':0.},{'growth_interval':0},{'_growth_tracker':3}):
            with self.assertRaises(ValueError):scaler.load_state_dict({**state,**changed})
            self.assertEqual(scaler.state_dict(),state)
        for kwargs in ({'init_scale':float('inf')},{'init_scale':1e-100},{'growth_factor':float('inf')},
                       {'backoff_factor':-1.},{'growth_interval':True}):
            with self.assertRaises(ValueError):GradScaler(**kwargs)

    def test_disabled_scaler_preserves_cpu_workflow(self):
        scaler=GradScaler(enabled=False);p=torch.nn.Parameter(torch.tensor(2.))
        opt=torch.optim.SGD([p],lr=.1)
        loss=p.square();self.assertIs(scaler.scale(loss),loss);scaler.scale(loss).backward()
        scaler.unscale_(opt);scaler.step(opt);scaler.update()
        torch.testing.assert_close(p,torch.tensor(1.6))
        self.assertEqual(scaler.state_dict(),{})

    def test_control_metadata_has_explicit_cpu_dtypes(self):
        previous=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            scaler=GradScaler(init_scale=128.)
            scaler._lazy_init_scale_growth_tracker(torch.device('cpu'))
            self.assertEqual((scaler._scale.device.type,scaler._scale.dtype),('cpu',torch.float32))
            self.assertEqual((scaler._growth_tracker.device.type,scaler._growth_tracker.dtype),('cpu',torch.int32))
            scaler.update(256);self.assertEqual(scaler.get_scale(),256.)
        finally:torch.set_default_dtype(previous)

    def test_cpu_scaler_overflow_control_sequence(self):
        p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.1)
        scaler=torch.amp.GradScaler('cpu',init_scale=8.,growth_interval=2)
        actual=[]
        for special in (None,None,float('inf'),None,float('nan'),None,None):
            scaler.scale(p.sum());p.grad=torch.tensor(scaler.get_scale() if special is None else special)
            scaler.step(opt);scaler.update();actual.append(scaler.get_scale())
        self.assertEqual(actual,[8.,16.,8.,8.,4.,4.,8.])

    def test_explicit_mixed_cpu_models_keep_fp32_master_gradients(self):
        torch.manual_seed(293)
        for kind in ('mlp','conv'):
            model=amp_model(kind,True)
            x=torch.randn(2,4)/4 if kind=='mlp' else torch.randn(2,1,4,4)/4
            with torch.backends.mkldnn.flags(enabled=False):
                logits=model(x);self.assertEqual(logits.dtype,torch.float16)
                loss=torch.nn.functional.cross_entropy(logits.float(),torch.tensor([0,2]))
                loss.backward()
            self.assertTrue(all(p.dtype==p.grad.dtype==torch.float32 for p in model.parameters()))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_cpu_conv_batch_rounding_explains_weight_difference(self):
        with torch.backends.mkldnn.flags(enabled=False):
            # Exact dyadics avoid tying this CI control to one CPU's GEMM/random path.
            x=torch.ones(3,1,1,1,dtype=torch.float16);w=torch.ones(1,1,1,1,dtype=torch.float16)
            b=torch.zeros(1,dtype=torch.float16);g=torch.tensor([2048.,1.,-2048.],dtype=torch.float16).view_as(x)
            result=conv_oracles(x,w,b,g,(1,1),(0,0),(1,1))
            self.assertEqual(result['native'].item(),0.)
            self.assertEqual(result['batch_rounding'].item(),0.)
            self.assertEqual(result['opmath'][2].item(),1.)
            self.assertEqual(result['fp64'][2].item(),1.)

    def test_cpu_opmath_conv_derivative_matches_dyadic_fp64(self):
        with torch.backends.mkldnn.flags(enabled=False):
            for case in geometry_cases():
                x,w,b,g=sample(case)
                values=[((torch.arange(t.numel())%13-6).float()/16).reshape(t.shape) for t in (x,w,b)]
                seed=((torch.arange(g.numel())%9-4).float()/32).reshape(g.shape).half()
                a=[t.clone().requires_grad_() for t in values];q=[t.clone().requires_grad_() for t in values]
                actual=reference_conv(*a,*case[3:])
                expected=torch.nn.functional.conv2d(*[t.half().double() for t in q],*case[3:]).half()
                actual.backward(seed);expected.backward(seed)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                for p,r in zip(a,q):torch.testing.assert_close(p.grad,r.grad,rtol=0,atol=0)

    def test_residual_opmath_control_has_all_fp32_master_gradients(self):
        torch.manual_seed(293);net=conv_model('residual','opmath')
        with torch.backends.mkldnn.flags(enabled=False):
            logits=net(torch.randn(4,1,6,6)/4)
            torch.nn.functional.cross_entropy(logits.float(),torch.tensor([0,1,2,0])).backward()
        self.assertEqual(sum(p.numel() for p in net.parameters()),105)
        self.assertTrue(all(p.dtype==p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all() for p in net.parameters()))


if __name__=='__main__':unittest.main()
