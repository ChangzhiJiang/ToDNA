import unittest
import torch
from revision_core import *
from revision_budget import SharedInitialPredictor
from test_revision_core import Toy

class BudgetTests(unittest.TestCase):
    def test_deduplication_preserves_optimizer_and_equalizes_calls(self):
        q=one_hot(['ACGT'*5]);seq='ACGT'*5
        original=RecordedPredictor(Toy(),'enhancer','k562');original.start=seq;original.trajectory=[0,1]
        reference=run_todna_batch(original,q,q*2,3,.1,1.,0.,.01,[1,2])
        for method in ['todna','fastseqprop','ledidi']:
            rec=SharedInitialPredictor(Toy(),'enhancer','k562');rec.start=seq
            if method=='todna':
                rec.trajectory=[0,1];result=run_todna_batch(rec,q,q*2,3,.1,1.,0.,.01,[1,2])
                self.assertEqual(reference,result)
            elif method=='fastseqprop':run_fastseqprop(rec,q,3,.1,2,1)
            else:run_ledidi(rec,q,3,1.,2,1,1.)
            self.assertEqual(rec.sequence_calls,9)
            self.assertEqual(sum(r['stage']=='initial' for r in rec.queries),1)
    def test_cannot_share_different_initial_sequences(self):
        rec=SharedInitialPredictor(Toy(),'enhancer','k562');rec.start='AC'
        with torch.no_grad(),self.assertRaises(ValueError):rec(one_hot(['AC','GT']))
if __name__=='__main__':unittest.main()
