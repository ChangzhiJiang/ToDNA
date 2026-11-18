"""Shared initial-query accounting for the prospective revision comparison."""
import torch
from revision_core import RecordedPredictor,decode

class SharedInitialPredictor(RecordedPredictor):
    def forward(self,x):
        if self.stage=='initial' and len(x)>1:
            seqs=decode(x)
            if len(set(seqs))!=1 or seqs[0]!=self.start:
                raise ValueError('Initial-query sharing requires identical original starts')
            if torch.is_grad_enabled():raise RuntimeError('Initial sharing must be outside differentiation')
            return super().forward(x[:1]).expand(len(x),-1)
        return super().forward(x)
