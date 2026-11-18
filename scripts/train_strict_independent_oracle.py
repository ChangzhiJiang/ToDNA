"""Train a fresh predictor using only its role-specific immutable train/val files."""
import argparse, hashlib, json, os, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
if not hasattr(np,'product'):np.product=np.prod
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',choices=['enhancer','promoter'],required=True);p.add_argument('--role',choices=['design','evaluation'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
 root=a.root;data=root/'outputs/strict_independent_v1/data'/a.family;out=root/'outputs/strict_independent_v1/assets'/a.family/a.role
 if out.exists():raise FileExistsError('Refuse to reuse/overwrite a previous training directory')
 audit=json.loads((data/'complete.json').read_text());cols=['K562_log2FC','HepG2_log2FC','SKNSH_log2FC'] if a.family=='enhancer' else ['JURKAT','K562','THP1']
 frames=[];hashes={}
 for split in ['train','validation']:
  name=f'{a.role}_{split}';path=data/f'{name}.csv';h=sha(path);assert h==audit['partitions'][name]['sha256'];hashes[name]=h
  df=pd.read_csv(path).loc[:,['sequence',*cols]];assert len(df)==audit['partitions'][name]['n'];frames.append(df)
 assert not(set(frames[0].sequence)&set(frames[1].sequence))
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 from grelu.data.dataset import DFSeqDataset
 from grelu.lightning import LightningModel
 import grelu.resources
 input_length=200 if a.family=='enhancer' else 2114
 state=root/'assets/external_validation/grelu_enformer_human_state_dict.h5'
 class Local:
  def download(self,root):
   dest=Path(root)/'human.h5'
   if not dest.exists():os.symlink(state.resolve(),dest)
   return str(root)
 original=grelu.resources.get_artifact;grelu.resources.get_artifact=lambda *args,**kw:Local()
 out.mkdir(parents=True)
 params={'task':'regression','loss':'mse','logger':'csv','name':a.role,'lr':1e-4,'optimizer':'adam','batch_size':16 if a.family=='enhancer' else 8,'max_epochs':10,'devices':[0],'num_workers':4,'save_dir':str(out),'checkpoint':{'dirpath':str(out/'checkpoints'),'filename':'epoch{epoch:02d}-val{val_loss:.4f}','monitor':'val_loss','mode':'min','save_top_k':1,'save_last':True,'save_weights_only':True}}
 try:
  model=LightningModel({'model_type':'EnformerModel' if a.family=='enhancer' else 'EnformerPretrainedModel','n_tasks':3},params)
  if a.family=='enhancer':
   pretrained=torch.load(root/'assets/enformer_pretrained/human_state_dict.h5',map_location='cpu',weights_only=False)
   if 'state_dict' in pretrained:pretrained=pretrained['state_dict']
   emb={k.removeprefix('embedding.'):v for k,v in pretrained.items() if k.startswith('embedding.')}
   if not emb:emb={k.removeprefix('model.embedding.'):v for k,v in pretrained.items() if k.startswith('model.embedding.')}
   assert emb;model.model.embedding.load_state_dict(emb,strict=True);del pretrained,emb
  train=DFSeqDataset(frames[0],seq_len=input_length,rc=False,max_seq_shift=2,seed=a.seed,augment_mode='random')
  val=DFSeqDataset(frames[1],seq_len=input_length,rc=False)
  trainer=model.train_on_dataset(train,val)
 finally:grelu.resources.get_artifact=original
 best=Path(trainer.checkpoint_callback.best_model_path);assert best.is_file()
 destination=out/'oracle.ckpt';os.link(best,destination)
 metadata={'protocol':'strict_independent_v1','family':a.family,'role':a.role,'seed':a.seed,'input_length':input_length,'data_hashes':hashes,'split_manifest_sha256':sha(data/'complete.json'),'checkpoint_sha256':sha(destination),'training':params,'initialization':'Generic pretraining only; no old task-specific predictor weights','sealed_test_opened':False,'validation_used_only_for_own_role_checkpoint_selection':True}
 (out/'complete.json').write_text(json.dumps(metadata,indent=2)+'\n');print('TRAINING_COMPLETE',a.family,a.role,flush=True)
if __name__=='__main__':main()
