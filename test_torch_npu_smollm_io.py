"""Offline CPU endpoint tests; no checkpoint, Transformers or NPU required."""
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
if importlib.util.find_spec('safetensors') is None:
    raise unittest.SkipTest('optional safetensors dependency is not installed')
from safetensors.torch import save_file

from torch_npu_smollm_io import FrozenSmolIO


class SmolIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator=torch.Generator().manual_seed(105)
        cls.config=dict(model_type='llama',hidden_size=576,tie_word_embeddings=True,rms_norm_eps=1e-5,vocab_size=64)
        cls.embedding=(torch.randn(64,576,generator=generator)*.03).bfloat16()
        cls.norm=(1+torch.randn(576,generator=generator)*.1).bfloat16()

    def make(self):return FrozenSmolIO(self.config,self.embedding,self.norm)

    def test_embedding_and_tied_output_matrix(self):
        io=self.make();ids=torch.tensor([[0,1,63],[2,7,19]])
        self.assertTrue(torch.equal(io.embed_tokens(ids),F.embedding(ids,self.embedding.half())))
        self.assertEqual(set(dict(io.named_buffers())),{'embedding','norm'})
        self.assertEqual(list(io.parameters()),[])
        x=io.embed_tokens(ids)
        reference=F.rms_norm(x.float(),(576,),eps=1e-5).half()*self.norm.half()
        torch.testing.assert_close(io.logits(x),F.linear(reference,self.embedding.half()),atol=.002,rtol=.002)
        self.assertEqual(io.logits(x).dtype,torch.float16)

    def test_owned_fp16_snapshots(self):
        embedding=self.embedding.half();norm=self.norm.half()
        io=FrozenSmolIO(self.config,embedding,norm);ids=torch.tensor([[4,8]])
        anchor=io.embed_tokens(ids).clone();weight=io.norm.clone()
        embedding.zero_();norm.zero_()
        self.assertTrue(torch.equal(io.embed_tokens(ids),anchor))
        self.assertTrue(torch.equal(io.norm,weight))

    def test_checkpoint_load_and_canonical_tying(self):
        with tempfile.TemporaryDirectory(prefix='smol-io-fixture-') as directory:
            path=Path(directory);(path/'config.json').write_text(json.dumps(self.config),encoding='utf-8')
            tensors={'model.embed_tokens.weight':self.embedding,'model.norm.weight':self.norm}
            save_file(tensors,str(path/'model.safetensors'))
            io=FrozenSmolIO.from_checkpoint(path)
            self.assertTrue(torch.equal(io.embedding,self.embedding.half()))
            self.assertTrue(torch.equal(io.norm,self.norm.half()))
            save_file(dict(tensors,**{'lm_head.weight':self.embedding.clone()}),str(path/'model.safetensors'))
            with self.assertRaises(ValueError):FrozenSmolIO.from_checkpoint(path)

    def test_parameter_guards_and_fp16_overflow(self):
        for updates in (dict(hidden_size=575),dict(tie_word_embeddings=False),dict(rms_norm_eps=0),
                        dict(vocab_size=True),dict(model_type='bert')):
            with self.assertRaises(ValueError):FrozenSmolIO(dict(self.config,**updates),self.embedding,self.norm)
        for bad in (self.embedding.int(),self.embedding[:,:575],self.embedding.float().fill_(float('nan'))):
            with self.assertRaises(ValueError):FrozenSmolIO(self.config,bad,self.norm)
        with self.assertRaisesRegex(ValueError,'overflows'):
            FrozenSmolIO(self.config,torch.full((64,576),1e10),self.norm)
        with self.assertRaises(ValueError):FrozenSmolIO(self.config,self.embedding,self.norm[:575])

    def test_input_mode_and_device_guards(self):
        io=self.make()
        for ids in (torch.ones(1,2,dtype=torch.int32),torch.ones(2,dtype=torch.int64),
                    torch.ones(1,0,dtype=torch.int64),torch.ones(1,2,dtype=torch.int64,device='meta')):
            with self.assertRaises(ValueError):io.embed_tokens(ids)
        for hidden in (torch.ones(1,2,576),torch.ones(1,2,575,dtype=torch.float16),
                       torch.ones(1,0,576,dtype=torch.float16),torch.ones(1,2,576,dtype=torch.float16,device='meta')):
            with self.assertRaises(ValueError):io.logits(hidden)
        for value in (-1,64):
            with self.assertRaises(IndexError):io.embed_tokens(torch.tensor([[value]]))
        io.train()
        with self.assertRaises(ValueError):io.embed_tokens(torch.tensor([[1]]))
        io.eval().float()
        with self.assertRaises(ValueError):io.logits(torch.zeros(1,1,576,dtype=torch.float16))


if __name__=='__main__':
    torch.set_num_threads(8);unittest.main()
