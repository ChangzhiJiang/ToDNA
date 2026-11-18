#!/usr/bin/env python3
"""Create fixed 32-start LoRA contexts for the supplementary comparison."""
import argparse,gc,json,time
from pathlib import Path
import torch
from revision_core import TASKS,sha,write_json,decode

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root
    out=r/'results/comparison_contexts_v1';out.mkdir(parents=True,exist_ok=False);started=time.time();groups={}
    for family,tasks in TASKS.items():
        for task in tasks:
            path=r/'inputs/formal72_v2/search/ctrl_dna'/family/'seed20260920/results'/f'{task}_starting_top128.json'
            rows=json.loads(path.read_text())['sequences'];indices=list(range(0,128,4));assert len(rows)==128
            groups[family,task]={'family':family,'task':task,'source':str(path),'source_sha256':sha(path),
                'indices':indices,'sequences':[rows[i]['sequence'] for i in indices],'logits':{}}
    import evo2
    original=evo2.Evo2
    def offline(name,*a,**k):
        k['local_path']=str(r/'assets/evo2_7b/evo2_7b.pt');return original(name,*a,**k)
    evo2.Evo2=offline
    from evo2_nabla_dna.dto import Evo2Dto
    for family in TASKS:
        dto=Evo2Dto(model_name='evo2_7b',lora_adapter=r/'inputs/assets'/family/'lora/best_adapter.pt')
        dto.disable_flash_attention();dto.model.eval()
        for task in TASKS[family]:
            g=groups[family,task];values=[]
            for seq in g['sequences']:
                logits=dto.logits_for_sequence(seq,initial_sequence_margin=.25).cpu()
                if decode(logits)!=[seq]:raise RuntimeError('Start changed')
                values.append(logits)
            g['logits']['lora']=torch.cat(values);torch.save(g,out/f'{family}_{task}.pt')
            print(json.dumps({'family':family,'task':task,'n':len(values)}),flush=True)
        del dto;gc.collect();torch.cuda.empty_cache()
    write_json(out/'manifest.json',{'protocol':'conditional_search_comparison_contexts_v1','pool_seed':20260920,
        'n_starts':32,'start_selection':'ranks 0,4,...,124 from archived Design Top128','source_sha256':sha(__file__),
        'files':{p.name:sha(p) for p in sorted(out.glob('*.pt'))},'evaluation_loaded':False})
    write_json(out/'complete.json',{'state':'COMPLETE','elapsed_seconds':time.time()-started})
if __name__=='__main__':main()
