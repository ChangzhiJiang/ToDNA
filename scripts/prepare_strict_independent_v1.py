"""Prepare immutable within-source partitions; never reuse old learned assets."""
import argparse, hashlib, json, re, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(s): return min(s,s.translate(str.maketrans('ACGT','TGCA'))[::-1])
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',choices=['enhancer','promoter'],required=True);a=p.parse_args()
 root=a.root; fam=a.family; out=root/'outputs/strict_independent_v1/data'/fam;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists(): raise SystemExit('Already prepared; refusing to change frozen partitions')
 sources=[root/'assets/gosai/Table_S2__MPRA_dataset.txt'] if fam=='enhancer' else [root/f'data/ctrl_dna_promoter_matched_v1/design_{s}.csv' for s in ['train','validation','test']]
 df=pd.concat([pd.read_csv(s,sep='\t' if fam=='enhancer' else ',') for s in sources],ignore_index=True)
 cols=['K562_log2FC','HepG2_log2FC','SKNSH_log2FC'] if fam=='enhancer' else ['JURKAT','K562','THP1'];L=200 if fam=='enhancer' else 250
 df['sequence']=df.sequence.str.upper();valid=df.sequence.str.fullmatch('[ACGT]{'+str(L)+'}') & np.isfinite(df[cols].to_numpy()).all(1)
 excluded=int((~valid).sum());df=df.loc[valid].reset_index(drop=True)
 # Keep repeated labels/records in one component, not silently average their targets.
 seq=out/'sequences.txt'; seq.write_text('\n'.join(df.sequence)+'\n')
 components=out/'components.txt'
 subprocess.run([str(root/'outputs/strict_independent_v1/bin/strict_hamming_components'),str(seq),str(components)],check=True)
 ids=np.loadtxt(components,dtype=int);assert len(ids)==len(df)
 # Union explicit enhancer source loci as well, including allele/construct variants.
 parent={int(i):int(i) for i in set(ids)}
 def find(x):
  while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
  return x
 if fam=='enhancer':
  seen={}
  for i,label in enumerate(df.IDs.astype(str)):
   match=re.match(r'^(?:chr)?([0-9XYM]+):(\d+):',label)
   if match:
    key=match.groups();r=find(int(ids[i]))
    if key in seen: parent[r]=find(seen[key])
    else: seen[key]=r
 ids=np.array([find(int(i)) for i in ids]);df['component']=ids
 # Labels never determine split assignment. Greedy size balancing of whole groups.
 sizes=df.groupby('component').size();rng=np.random.default_rng(20260909)
 order=list(sizes.index);rng.shuffle(order);order.sort(key=lambda g:-int(sizes[g]))
 partitions=['design_train','design_validation','evaluation_train','evaluation_validation','sealed_test'];weights=np.array([.36,.04,.36,.04,.20]);targets=weights*len(df);counts=np.zeros(5,int);assignment={}
 for g in order:
  j=int(np.argmax((targets-counts)/targets));assignment[g]=partitions[j];counts[j]+=int(sizes[g])
 df['partition']=df.component.map(assignment)
 if np.any(np.abs(counts/len(df)-weights)>.03): raise RuntimeError('Components prevent planned proportions; review required, do not split clusters')
 canonical=df.sequence.map(canon)
 assert pd.DataFrame({'s':canonical,'p':df.partition}).groupby('s').p.nunique().max()==1
 assert df.groupby('component').partition.nunique().max()==1
 manifest={}
 for part in partitions:
  x=df[df.partition==part];path=out/f'{part}.csv';x[['sequence',*cols]].to_csv(path,index=False);manifest[part]={'n':len(x),'sha256':sha(path),'components':x.component.nunique()}
  if part.startswith('design_'):
   path=out/'lora'/('train.jsonl' if part=='design_train' else 'validation.jsonl');path.parent.mkdir(exist_ok=True)
   with path.open('w') as f:
    for s in x.sequence: f.write(json.dumps({'sequence':s})+'\n')
   manifest['lora_'+part]={'n':len(x),'sha256':sha(path)}
 df[['component','partition']].assign(sequence_sha256=df.sequence.map(lambda s:hashlib.sha256(s.encode()).hexdigest())).to_csv(out/'membership.csv',index=False)
 audit={'protocol':'strict_independent_v1','family':fam,'seed':20260909,'excluded':excluded,'eligible':len(df),'components':len(sizes),'largest_component':int(sizes.max()),'source_hashes':{str(s):sha(s) for s in sources},'partitions':manifest,'isolation':'Exact forward/RC plus all aligned equal-length pairs with >=90% identity are in one component; enhancer source-locus variants also grouped. Indel/shifted homology not certified.','pretraining_exposure':'Unknown; no claim of independence from Evo2/Enformer pretraining.','historical_exposure':'Previously inspected corpus; new split is not prospectively unseen external data.'}
 (out/'complete.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
