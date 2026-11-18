#!/usr/bin/env python3
"""Six-task implementation pilot, with complete query archives and post-lock evaluation."""
import argparse,gc,json,time,platform
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from revision_core import *
from evo2_nabla_dna.predictor import load_activity_checkpoint

def arms():
    values=[]
    for steps in [10,40]:
        values.extend([
            dict(name=f'todna_s{steps}',method='todna',steps=steps,lr=.1,kl=1.,edit=0.),
            dict(name=f'fastseqprop_s{steps}',method='fastseqprop',steps=steps,lr=.1,kl=0.,edit=0.),
            dict(name=f'ledidi_s{steps}',method='ledidi',steps=steps,lr=1.,kl=0.,edit=1.),
        ])
    values.extend([
        dict(name='todna_none',method='todna',steps=10,lr=.1,kl=0.,edit=0.),
        dict(name='todna_edit',method='todna',steps=10,lr=.1,kl=0.,edit=1.),
        dict(name='todna_kl_edit',method='todna',steps=10,lr=.1,kl=1.,edit=1.),
        dict(name='todna_base_logits',method='todna',steps=10,lr=.1,kl=1.,edit=0.,context='base'),
    ])
    for objective in ['target','min','smooth_min']:
        values.append(dict(name='reward_'+objective,method='todna',steps=10,lr=.1,kl=0.,edit=0.,objective=objective))
    return values

def save_run(out,rec,details,meta,top_k):
    # Shared postprocessing uses all evaluated hard candidates, first occurrence
    # per sequence. It is a new protocol; never relabel it as the historical endpoint-only table.
    distinct={}
    for row in rec.queries:
        distinct.setdefault(row['sequence'],row)
    selected=sorted(distinct.values(),key=lambda row:(-row['reward'],row['sequence']))[:top_k]
    with (out/'queries.jsonl').open('w') as f:
        for row in rec.queries:f.write(json.dumps(row)+'\n')
    write_json(out/'trajectories.json',details)
    lock={'protocol':'revision_pilot_all_queried_candidates_v1','meta':meta,'selection':'Design objective only; unique sequence, first query, then top-k',
        'requested_top_k':top_k,'actual_top_k':len(selected),'top_k_complete':len(selected)==top_k,'candidates':selected}
    write_json(out/'lock.json',lock)
    summary={'state':'COMPLETE','lock_sha256':sha(out/'lock.json'),'queries_sha256':sha(out/'queries.jsonl'),
        'design_sequence_calls':rec.sequence_calls,'design_forward_batches':rec.calls,
        'backward_sequence_calls':sum(d['backward_sequence_calls'] for d in details),
        'query_unique_fraction':len(set(row['sequence'] for row in rec.queries))/len(rec.queries),
        'candidate_unique_count':len(distinct),'selected_count':len(selected),
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),**meta}
    write_json(out/'complete.json',summary);return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=['search','evaluate'],required=True)
    ap.add_argument('--families',nargs='+',default=['enhancer','promoter']);ap.add_argument('--only',nargs='*');ap.add_argument('--n-starts',type=int,default=8)
    ap.add_argument('--seeds',nargs='+',type=int,default=[2026110601,2026110602,2026110603]);ap.add_argument('--samples',type=int,default=4)
    ap.add_argument('--output-name',default='pilot_v1');args=ap.parse_args();root=args.root;output=root/'results'/args.output_name
    device='cuda';torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    selected_arms=[a for a in arms() if not args.only or a['name'] in args.only]
    if args.phase=='search':
        output.mkdir(parents=True,exist_ok=True)
        manifest={'phase':'implementation pilot, not confirmatory result','source_sha256':sha(__file__),'core_sha256':sha(Path(__file__).with_name('revision_core.py')),
            'seeds':args.seeds,'n_starts':args.n_starts,'samples_per_start':args.samples,'arms':arms(),'fixed_pool_seed':20260920,
            'noise_std_todna':.01,'reference_perturbed':False,'temperature_todna':.7,'smooth_tau':.1,
            'evaluation_loaded_during_search':False,'top_k':16,'torch':torch.__version__,'python':platform.python_version(),
            'interpretation':'Matched Design data and all-query selection; method-specific native optimization. 10/40 steps diagnose convergence only; no final method selection from Evaluation.'}
        manifest_path=output/'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('Pilot manifest changed; use a new output name')
        write_json(manifest_path,manifest)
        for family in args.families:
            checkpoint=root/'inputs/assets'/family/'design/oracle.ckpt';checkpoint_hash=sha(checkpoint);predictor=load_activity_checkpoint(str(checkpoint),device).eval()
            for p in predictor.parameters():p.requires_grad_(False)
            for task in TASKS[family]:
                context_path=root/'results/pilot_contexts_v1'/f'{family}_{task}.pt'
                if context_path.exists():
                    g=torch.load(context_path,map_location='cpu',weights_only=False)
                elif all(arm['method']!='todna' for arm in selected_arms):
                    source=root/'inputs/formal72_v2/search/ctrl_dna'/family/'seed20260920/results'/f'{task}_starting_top128.json'
                    rows=json.loads(source.read_text())['sequences']
                    g={'sequences':[rows[i*len(rows)//args.n_starts]['sequence'] for i in range(args.n_starts)]}
                else:raise FileNotFoundError(context_path)
                seqs=g['sequences'][:args.n_starts]
                for arm in selected_arms:
                    for seed in args.seeds:
                        out=output/family/task/arm['name']/f'seed{seed}'
                        if (out/'complete.json').exists():continue
                        if out.exists():raise RuntimeError(f'Incomplete prior run requires inspection: {out}')
                        out.mkdir(parents=True);start=time.time();torch.cuda.reset_peak_memory_stats()
                        rec=RecordedPredictor(predictor,family,task,arm.get('objective','mean'));details=[]
                        for i,seq in enumerate(seqs):
                            rec.start=seq;initial=one_hot([seq],device);base_seed=seed+i*1000
                            if arm['method']=='todna':
                                ref=g['logits'][arm.get('context','lora')][i:i+1].to(device)
                                rec.trajectory=[i*args.samples+j for j in range(args.samples)]
                                results=run_todna_batch(rec,initial,ref,arm['steps'],arm['lr'],arm['kl'],arm['edit'],.01,[base_seed+j for j in range(args.samples)])
                                for j,result in enumerate(results):details.append({'start_index':i,'restart':j,'search_seed':base_seed+j,**result})
                            elif arm['method']=='fastseqprop':
                                rec.trajectory=i;details.append({'start_index':i,**run_fastseqprop(rec,initial,arm['steps'],arm['lr'],args.samples,base_seed)})
                            else:
                                rec.trajectory=i;details.append({'start_index':i,**run_ledidi(rec,initial,arm['steps'],arm['lr'],args.samples,base_seed,arm['edit'])})
                        torch.cuda.synchronize();elapsed=time.time()-start
                        meta={'family':family,'task':task,'arm':arm,'seed':seed,'elapsed_seconds':elapsed,'design_sha256':checkpoint_hash}
                        report=save_run(out,rec,details,meta,16)
                        print(json.dumps({k:report[k] for k in ['family','task','arm','seed','elapsed_seconds','design_sequence_calls','candidate_unique_count','selected_count']}),flush=True)
                        write_json(output/'progress.json',{'last_completed':str(out),'completed_runs':len(list(output.glob('*/*/*/seed*/complete.json')))})
            del predictor;gc.collect();torch.cuda.empty_cache()
        n_done=len(list(output.glob('*/*/*/seed*/complete.json')));n_expected=sum(len(x) for x in TASKS.values())*len(arms())*len(args.seeds)
        if n_done==n_expected:
            write_json(output/'search_complete.json',{'state':'COMPLETE','runs':n_done,
                'locks':{str(p.relative_to(output)):sha(p) for p in sorted(output.glob('*/*/*/seed*/lock.json'))}})
        else:write_json(output/'search_progress.json',{'state':'PARTIAL','completed_runs':n_done,'expected_runs':n_expected})
    else:
        completion=json.loads((output/'search_complete.json').read_text())
        rows=[]
        for family in args.families:
            checkpoint=root/'inputs/assets'/family/'evaluation/oracle.ckpt';checkpoint_hash=sha(checkpoint);predictor=load_activity_checkpoint(str(checkpoint),device).eval()
            for rel,expected in completion['locks'].items():
                if not rel.startswith(family+'/'):continue
                path=output/rel
                if sha(path)!=expected:raise ValueError('Lock changed before evaluation')
                lock=json.loads(path.read_text());seqs=[x['sequence'] for x in lock['candidates']];meta=lock['meta'];pred=[]
                with torch.no_grad():
                    for i in range(0,len(seqs),4):pred.append(predictor(one_hot(seqs[i:i+4],device)).cpu())
                y=torch.cat(pred);mean=score(y,family,meta['task'],'mean');mingap=score(y,family,meta['task'],'min')
                record={'family':family,'task':meta['task'],'arm':meta['arm']['name'],'seed':meta['seed'],'n':len(seqs),
                    'mean_margin':float(mean.mean()),'min_gap':float(mingap.mean()),'positive_min_gap_fraction':float((mingap>0).float().mean()),
                    'lock_sha256':expected,'evaluation_checkpoint_sha256':checkpoint_hash,'raw_activity_mean':y.mean(0).tolist()}
                write_json(path.parent/'evaluation.json',{'summary':record,'sequences':seqs,'predictions':y.tolist()});rows.append(record)
            del predictor;gc.collect();torch.cuda.empty_cache()
        write_json(output/'evaluation_summary.json',rows);write_json(output/'evaluation_complete.json',{'state':'COMPLETE','n_runs':len(rows),'scope':'pilot only; no configuration selected using these scores'})
if __name__=='__main__':main()
