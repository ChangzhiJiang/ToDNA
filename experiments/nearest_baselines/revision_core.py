"""Explicit search math for the revision; no evaluation model is imported here."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

BASES='ACGT'
TASKS={'enhancer':['k562','hepg2','sknsh'],'promoter':['jurkat','k562','thp1']}
RANGES={'enhancer':[(-5.857445,10.781755),(-6.051336,10.992575),(-7.283977,12.888308)],
        'promoter':[(-5.574782,8.555965),(-4.088671,10.781755),(-7.271035,6.797082)]}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def tensor_hash(x):return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)

def one_hot(seqs,device='cpu'):
    return F.one_hot(torch.tensor([[BASES.index(b) for b in s] for s in seqs],device=device),4).float()

def decode(q):return [''.join(BASES[int(i)] for i in row) for row in q.detach().argmax(-1).cpu()]

def hard_st(logits,temperature=1.,sample=False,generator=None):
    soft=(logits/temperature).softmax(-1)
    indices=torch.multinomial(soft.detach().reshape(-1,4),1,generator=generator).reshape(soft.shape[:-1]) if sample else soft.argmax(-1)
    hard=F.one_hot(indices,4).to(soft.dtype)
    return hard+soft-soft.detach()

def stable_kl(logits,reference_logits,temperature):
    logq=F.log_softmax(logits/temperature,-1)
    logp=F.log_softmax(reference_logits/temperature,-1)
    return (logq.exp()*(logq-logp)).sum(-1).mean(-1)

def expected_edits(logits,initial,temperature):
    return (1-(logits/temperature).softmax(-1).gather(-1,initial.argmax(-1,keepdim=True)).squeeze(-1)).mean(-1)

def score(pred,family,target,objective='mean',smooth_tau=.1):
    low=pred.new_tensor([x[0] for x in RANGES[family]]);scale=pred.new_tensor([x[1]-x[0] for x in RANGES[family]])
    normalized=(pred-low)/scale; idx=TASKS[family].index(target);off=[j for j in range(3) if j!=idx]
    if objective=='target':return normalized[:,idx]
    if objective=='mean':return normalized[:,idx]-normalized[:,off].mean(-1)
    if objective=='min':return normalized[:,idx]-normalized[:,off].max(-1).values
    if objective=='smooth_min':return normalized[:,idx]-smooth_tau*torch.logsumexp(normalized[:,off]/smooth_tau,-1)
    raise ValueError(objective)

def restart_logits(reference,initial,std,seed,margin=.25):
    # Restart perturbation never changes the KL reference tensor.
    gen=torch.Generator(device='cpu').manual_seed(seed)
    noise=torch.randn(reference.shape,generator=gen,dtype=torch.float32).to(reference.device)
    logits=reference.clone()+std*noise
    mask=initial.bool();chosen=logits.gather(-1,initial.argmax(-1,keepdim=True))
    maximum=logits.masked_fill(mask,-torch.inf).max(-1,keepdim=True).values
    logits=logits+initial*(maximum-chosen+margin).clamp_min(0)
    return logits,tensor_hash(noise)

class FastSeqPropParameters(nn.Module):
    """PyTorch port of seqprop InstanceNormalization(axis=-2) on [1,L,4,1].

    One policy per start, independent base-specific scale/bias. Sampling uses
    categorical hard forward and ordinary softmax backward, not Gumbel gradients.
    Reference commit: 61677a2bc469257b95231fd5f280f7eff2fb0b35.
    """
    def __init__(self,initial,p_init=.5,epsilon=1e-3):
        super().__init__();self.epsilon=epsilon
        self.weights=nn.Parameter((initial*p_init+(1-initial)*(1-p_init)/3).log())
        self.gamma=nn.Parameter(torch.ones(1,1,4,device=initial.device))
        self.beta=nn.Parameter(torch.zeros(1,1,4,device=initial.device))
    def forward(self):
        w=self.weights;return (w-w.mean(1,keepdim=True))/(w.std(1,unbiased=False,keepdim=True)+self.epsilon)*self.gamma+self.beta

class RecordedPredictor(nn.Module):
    """Count every sequence-level Design call and retain all hard queries."""
    def __init__(self,predictor,family,target,objective='mean',smooth_tau=.1):
        super().__init__();self.predictor=predictor;self.family=family;self.target=target
        self.objective=objective;self.smooth_tau=smooth_tau;self.queries=[];self.calls=0;self.sequence_calls=0
        self.start='';self.trajectory=0;self.stage='initial';self.step=0
    def forward(self,x):
        pred=self.predictor(x)
        if pred.shape!=(x.shape[0],3) or not torch.isfinite(pred).all():raise RuntimeError('Invalid predictor output')
        rewards=score(pred,self.family,self.target,self.objective,self.smooth_tau)
        # Fail rather than record a soft distribution as a discrete candidate.
        if not torch.allclose(x.detach(),one_hot(decode(x),x.device),atol=1e-6,rtol=0):raise ValueError('Non-discrete oracle input')
        seqs=decode(x);values=pred.detach().cpu().tolist();rr=rewards.detach().cpu().tolist()
        for sample_index,(s,y,v) in enumerate(zip(seqs,values,rr)):
            self.queries.append({'sequence':s,'activity':y,'reward':v,'initial_sequence':self.start,
                'trajectory':self.trajectory[sample_index] if isinstance(self.trajectory,list) else self.trajectory,
                'step':self.step,'stage':self.stage,'query_index':self.sequence_calls})
            self.sequence_calls+=1
        self.calls+=1;return pred
    def reward(self,x):return score(self(x),self.family,self.target,self.objective,self.smooth_tau)

class ChannelFirstModel(nn.Module):
    def __init__(self,recorded):super().__init__();self.recorded=recorded;self.first_call=recorded.calls
    def forward(self,x):
        step=self.recorded.calls-self.first_call
        self.recorded.stage='initial' if step==0 else 'search';self.recorded.step=step
        return self.recorded(x.transpose(1,2))

def run_todna(rec,initial,reference,steps,lr,kl_weight,edit_weight,noise_std,seed,temperature=.7):
    logits,noise_hash=restart_logits(reference,initial,noise_std,seed)
    logits=nn.Parameter(logits);ref=reference.detach().clone();ref_hash=tensor_hash(ref)
    opt=torch.optim.Adam([logits],lr=lr);history=[];rec.stage='initial';rec.step=0
    with torch.no_grad():rec.reward(hard_st(logits,temperature))
    for step in range(steps):
        rec.stage='search';rec.step=step+1;opt.zero_grad(set_to_none=True)
        reward=rec.reward(hard_st(logits,temperature));kl=stable_kl(logits,ref,temperature);edits=expected_edits(logits,initial,temperature)
        loss=(-reward+kl_weight*kl+edit_weight*edits).mean();loss.backward()
        if not torch.isfinite(logits.grad).all():raise RuntimeError('Nonfinite gradient')
        grad_norm=float(nn.utils.clip_grad_norm_([logits],5.));opt.step()
        history.append({'step':step+1,'reward':float(reward.detach().mean()),'kl':float(kl.detach().mean()),'expected_edit_fraction':float(edits.detach().mean()),'gradient_norm':grad_norm})
    rec.stage='endpoint';rec.step=steps
    with torch.no_grad():q=hard_st(logits,temperature);rec.reward(q)
    if tensor_hash(ref)!=ref_hash:raise RuntimeError('Reference changed')
    return {'endpoints':decode(q),'history':history,'noise_sha256':noise_hash,'reference_sha256':ref_hash,
            'final_logits_sha256':tensor_hash(logits),'backward_sequence_calls':steps}

def run_fastseqprop(rec,initial,steps,lr,n_samples,seed):
    model=FastSeqPropParameters(initial);opt=torch.optim.Adam(model.parameters(),lr=lr)
    gen=torch.Generator(device=initial.device).manual_seed(seed);history=[]
    rec.stage='initial';rec.step=0
    with torch.no_grad():rec.reward(initial)
    for step in range(steps):
        opt.zero_grad(set_to_none=True);rec.stage='search';rec.step=step+1
        q=hard_st(model().expand(n_samples,-1,-1),sample=True,generator=gen)
        reward=rec.reward(q);loss=-reward.mean();loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Bad FSP gradient')
        opt.step();history.append({'step':step+1,'mean_reward':float(reward.detach().mean())})
    rec.stage='endpoint';rec.step=steps
    with torch.no_grad():q=hard_st(model().expand(n_samples,-1,-1),sample=True,generator=gen);rec.reward(q)
    return {'endpoints':decode(q),'history':history,'backward_sequence_calls':steps*n_samples}

def run_todna_batch(rec,initial,reference,steps,lr,kl_weight,edit_weight,noise_std,seeds,temperature=.7):
    """Independent restarts batched only for oracle throughput.

    Sum per-trajectory losses and clip each row separately. Adam moments remain
    independent per logit element, as in separate optimizers with equal steps.
    """
    prepared=[restart_logits(reference,initial,noise_std,s) for s in seeds]
    logits=nn.Parameter(torch.cat([x[0] for x in prepared]));ref=reference.detach().expand(len(seeds),-1,-1).clone()
    initial=initial.expand(len(seeds),-1,-1);opt=torch.optim.Adam([logits],lr=lr)
    hashes=[tensor_hash(row.unsqueeze(0)) for row in ref];history=[[] for _ in seeds]
    rec.stage='initial';rec.step=0
    with torch.no_grad():rec.reward(hard_st(logits,temperature))
    for step in range(steps):
        opt.zero_grad(set_to_none=True);rec.stage='search';rec.step=step+1
        reward=rec.reward(hard_st(logits,temperature));kl=stable_kl(logits,ref,temperature);edits=expected_edits(logits,initial,temperature)
        (-reward+kl_weight*kl+edit_weight*edits).sum().backward()
        if not torch.isfinite(logits.grad).all():raise RuntimeError('Nonfinite gradient')
        norms=logits.grad.flatten(1).norm(2,1)
        logits.grad.mul_((5./(norms+1e-6)).clamp(max=1).view(-1,1,1));opt.step()
        values=torch.stack([reward.detach(),kl.detach(),edits.detach(),norms.detach()],1).cpu().tolist()
        for h,v in zip(history,values):h.append(dict(step=step+1,reward=v[0],kl=v[1],expected_edit_fraction=v[2],gradient_norm=v[3]))
    rec.stage='endpoint';rec.step=steps
    with torch.no_grad():q=hard_st(logits,temperature);rec.reward(q)
    sequences=decode(q)
    return [{'endpoints':[s],'history':h,'noise_sha256':prepared[j][1],'reference_sha256':hashes[j],
        'final_logits_sha256':tensor_hash(logits[j:j+1]),'backward_sequence_calls':steps} for j,(s,h) in enumerate(zip(sequences,history))]

def run_ledidi(rec,initial,steps,lr,n_samples,seed,edit_weight):
    from ledidi import Ledidi
    calls_before=rec.sequence_calls
    def output_loss(pred,_):return -score(pred,rec.family,rec.target,rec.objective,rec.smooth_tau).mean()
    rec.stage='search';rec.step=-1
    designer=Ledidi(ChannelFirstModel(rec),shape=(4,initial.shape[1]),output_loss=output_loss,
        # Native penalty counts edited bases. Divide coefficient by L to match
        # the units of a mean edit-fraction coefficient; this is documented.
        l=edit_weight/initial.shape[1],tau=1.,batch_size=n_samples,max_iter=steps,
        early_stopping_iter=steps+1,lr=lr,return_history=True,verbose=False,random_state=seed).to(initial.device)
    output,history=designer.fit_transform(initial.transpose(1,2),initial.new_zeros(1,3))
    rec.stage='endpoint';rec.step=steps
    with torch.no_grad():rec(output.transpose(1,2))
    if len(history['total_loss'])!=steps:raise RuntimeError('Unexpected Ledidi early stop')
    return {'endpoints':decode(output.transpose(1,2)),'history':{k:v for k,v in history.items() if k!='edits'},
            'backward_sequence_calls':steps*n_samples,'native_query_count':rec.sequence_calls-calls_before}
