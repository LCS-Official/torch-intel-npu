"""CPU controls for the bounded token model and checkpoint payload."""
import io
from contextlib import redirect_stderr
import unittest
from unittest.mock import patch
import torch
from torch_npu_rng import RNGState
from torch_npu_standard_modules_demo import cpu_rng_reference
from torch_npu_transformer_demo import token_classifier,cpu_tree
import torch_npu_transformer_demo as demo
from torch_npu_transformer_scale import PROFILES


class TransformerReferenceTests(unittest.TestCase):
    def test_scaled_standard_model_shapes_and_all_gradients(self):
        for profile,count in (('small',42211),('medium',232387),('long',36835)):
            config=PROFILES[profile];model=token_classifier(**config)
            self.assertEqual(sum(p.numel() for p in model.parameters()),count)
            tokens=torch.ones(2,config['length'],dtype=torch.int64);tokens[:,-2:]=0
            with torch.backends.mkldnn.flags(enabled=False),cpu_rng_reference(RNGState(61).get_state()):
                logits=model(tokens);torch.nn.functional.cross_entropy(logits,torch.tensor([0,2])).backward()
            self.assertEqual(tuple(logits.shape),(2,3))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertEqual(torch.count_nonzero(model.embedding.weight.grad[0]).item(),0)

    def test_native_half_masked_softmax_rounding(self):
        for seed in range(5):
            scores=torch.randn(2,2,3,4,dtype=torch.float16,generator=torch.Generator().manual_seed(seed))
            mask=torch.zeros_like(scores,dtype=torch.bool);mask[...,2:]=True;mask[1,1,2]=True
            expected=torch.ops.aten._masked_softmax(scores,mask,-1,2)
            selected=scores.masked_fill(mask,float('-inf'))
            difference=selected-selected.max(-1,keepdim=True).values
            exponent=difference.float().exp().half().masked_fill(mask,0)
            inverse=(1/exponent.float().sum(-1,keepdim=True)).half()
            actual=exponent*inverse
            torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)

    def test_invalid_precision_is_rejected_before_hardware_import(self):
        for case,precision in (('classifier','float16'),('container','bool'),('boolean','float32')):
            with patch('sys.argv',['demo','--case',case,'--precision',precision]),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:demo.main()
                self.assertEqual(error.exception.code,2)

    def test_token_model_gradients_and_all_dropout_sites(self):
        model=token_classifier();tokens=torch.tensor([[2,2,0],[3,4,0]])
        before=torch.get_rng_state().clone()
        with cpu_rng_reference(RNGState(53).get_state()) as tracker:
            logits=model(tokens);torch.nn.functional.cross_entropy(logits,torch.tensor([0,2])).backward()
        self.assertEqual(tracker.offset,264)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(torch.count_nonzero(model.embedding.weight.grad[0]).item(),0)

    def test_cpu_checkpoint_clones_momentum_and_weights(self):
        model=token_classifier();opt=torch.optim.SGD(model.parameters(),lr=.02,momentum=.5,foreach=False)
        for parameter in model.parameters():parameter.grad=torch.ones_like(parameter)
        opt.step()
        saved=cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=RNGState(59).get_state()))
        snapshot=saved['model']['position'].clone()
        with torch.no_grad():model.position.add_(1)
        torch.testing.assert_close(saved['model']['position'],snapshot,rtol=0,atol=0)
        buffer=io.BytesIO();torch.save(saved,buffer);buffer.seek(0);restored=torch.load(buffer,weights_only=True)
        for key,state in restored['optimizer']['state'].items():
            self.assertEqual(state['momentum_buffer'].device.type,'cpu')
            torch.testing.assert_close(state['momentum_buffer'],saved['optimizer']['state'][key]['momentum_buffer'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
