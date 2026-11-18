"""Scientific invariants, including analytical gradients and native baseline smoke."""
import unittest, math
import torch
from torch import nn
from revision_core import *

class Toy(nn.Module):
    def forward(self,x):return torch.stack((x[:,:,0].mean(1)*5,x[:,:,1].mean(1)*3,x[:,:,2].mean(1)*2),1)

class Tests(unittest.TestCase):
    def test_metric_worst_off_target(self):
        family='enhancer';low=torch.tensor([v[0] for v in RANGES[family]]);scale=torch.tensor([v[1]-v[0] for v in RANGES[family]])
        y=low+scale*torch.tensor([[.8,.7,.1],[.4,.9,.0]])
        torch.testing.assert_close(score(y,family,'k562','mean'),torch.tensor([.4,-.05]),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(score(y,family,'k562','min'),torch.tensor([.1,-.5]),atol=1e-6,rtol=1e-6)
    def test_kl_distribution_and_extreme_logits(self):
        p=torch.tensor([[[1.,2.,-4.,3.],[100.,-100.,0.,1.]]],requires_grad=True)
        torch.testing.assert_close(stable_kl(p,p.detach(),.7),torch.zeros(1),atol=1e-7,rtol=0)
        k=stable_kl(p,torch.zeros_like(p),.7);self.assertGreaterEqual(float(k),0);k.sum().backward();self.assertTrue(torch.isfinite(p.grad).all())
    def test_edit_penalty_hand_calculation(self):
        x=one_hot(['A']);z=torch.tensor([[[math.log(.7),math.log(.1),math.log(.1),math.log(.1)]]])
        torch.testing.assert_close(expected_edits(z,x,1),torch.tensor([.3]))
    def test_st_matches_analytical_softmax_jacobian(self):
        z=torch.tensor([[[.2,.1,-.3,.7]]],requires_grad=True);v=torch.tensor([1.,4.,-2.,3.])
        q=hard_st(z,sample=True,generator=torch.Generator().manual_seed(6));(q*v).sum().backward()
        prob=z.detach().softmax(-1);expected=prob*(v-(prob*v).sum(-1,keepdim=True))
        torch.testing.assert_close(z.grad,expected);self.assertEqual(float(q.detach().sum()),1.)
    def test_reference_and_restart(self):
        q=one_hot(['ACGTACGT']);ref=q*2;digest=tensor_hash(ref)
        a,_=restart_logits(ref,q,.5,10);b,_=restart_logits(ref,q,.5,11)
        self.assertEqual(tensor_hash(ref),digest);self.assertNotEqual(tensor_hash(a),tensor_hash(b));self.assertEqual(decode(a),['ACGTACGT'])
    def test_all_methods_finite_counted_and_fixed_steps(self):
        q=one_hot(['ACGT'*5])
        for name in ['todna','fsp','ledidi']:
            rec=RecordedPredictor(Toy(),'enhancer','k562');rec.start='ACGT'*5
            if name=='todna':r=run_todna(rec,q,q*2,3,.1,1.,0.,.01,10);expected=5
            elif name=='fsp':r=run_fastseqprop(rec,q,3,.1,2,10);expected=9
            else:r=run_ledidi(rec,q,3,1.,2,10,1.);expected=9
            self.assertEqual(rec.sequence_calls,expected);self.assertTrue(r['endpoints']);self.assertGreater(r['backward_sequence_calls'],0)
    def test_batched_restarts_match_independent_optimizers(self):
        q=one_hot(['ACGT'*5]);seeds=[21,22,23];single=[]
        for seed in seeds:
            rec=RecordedPredictor(Toy(),'enhancer','k562')
            single.append(run_todna(rec,q,q*2,4,.1,.3,.4,.1,seed))
        rec=RecordedPredictor(Toy(),'enhancer','k562');rec.trajectory=[0,1,2]
        batch=run_todna_batch(rec,q,q*2,4,.1,.3,.4,.1,seeds)
        for a,b in zip(single,batch):
            self.assertEqual(a['endpoints'],b['endpoints'])
            for ha,hb in zip(a['history'],b['history']):
                for k in ['reward','kl','expected_edit_fraction','gradient_norm']:self.assertAlmostEqual(ha[k],hb[k],places=6)
        self.assertEqual(rec.sequence_calls,18)
if __name__=='__main__':unittest.main()
