#!/usr/bin/env python3
"""Equal-query development and six-seed conditional nearest-method comparison."""
import argparse,gc,json,statistics,time
from pathlib import Path
import torch
from revision_core import *
from revision_budget import SharedInitialPredictor
from evo2_nabla_dna.predictor import load_activity_checkpoint

def save_run(out,rec,details,meta,top_k):
    distinct={}
    for row in rec.queries:distinct.setdefault(row['sequence'],row)
    selected=sorted(distinct.values(),key=lambda row:(-row['reward'],row['sequence']))[:top_k]
    with (out/'queries.jsonl').open('w') as f:
        for row in rec.queries:f.write(json.dumps(row)+'\n')
    write_json(out/'trajectories.json',details)
    write_json(out/'lock.json',{'protocol':'conditional_search_nearest_baselines_v1','meta':meta,
        'selection':'Design normalized mean-margin only; first occurrence per unique sequence then top-k',
        'requested_top_k':top_k,'actual_top_k':len(selected),'top_k_complete':len(selected)==top_k,'candidates':selected})
    summary={'state':'COMPLETE','lock_sha256':sha(out/'lock.json'),'queries_sha256':sha(out/'queries.jsonl'),
        'design_sequence_calls':rec.sequence_calls,'design_forward_batches':rec.calls,
        'backward_sequence_calls':sum(d['backward_sequence_calls'] for d in details),
        'query_unique_fraction':len(distinct)/len(rec.queries),'candidate_unique_count':len(distinct),
        'selected_count':len(selected),'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),**meta}
    write_json(out/'complete.json',summary);return summary

def run_search(root,cfg,phase):
    development=phase=='development';name='nearest_development_v1' if development else 'nearest_comparison_v1'
    output=root/'results'/name;output.mkdir(parents=True,exist_ok=True)
    frozen=None if development else json.loads((root/'results/nearest_development_v1/frozen_selection.json').read_text())
    seeds=cfg['development_seeds'] if development else cfg['search_seeds']
    n=cfg['development_starts'] if development else cfg['n_starts'];topk=cfg['development_top_k'] if development else cfg['top_k']
    manifest={'protocol':cfg,'phase':phase,'config_sha256':sha(root/'configs/comparison_v1.json'),
        'source_hashes':{name:sha(Path(__file__).with_name(name)) for name in ['run_nearest_comparison.py','revision_core.py','revision_budget.py','run_revision_pilot.py']},
        'context_manifest_sha256':sha(root/'results/comparison_contexts_v1/manifest.json'),
        'selection_sha256':None if development else sha(root/'results/nearest_development_v1/frozen_selection.json'),
        'evaluation_loaded_during_search':False}
    mp=output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('Protocol changed')
    write_json(mp,manifest)
    for family,tasks in TASKS.items():
        checkpoint=root/'inputs/assets'/family/'design/oracle.ckpt';ch=sha(checkpoint)
        predictor=load_activity_checkpoint(str(checkpoint),'cuda').eval()
        for p in predictor.parameters():p.requires_grad_(False)
        for task in tasks:
            cp=root/'results/comparison_contexts_v1'/f'{family}_{task}.pt';g=torch.load(cp,weights_only=False,map_location='cpu')
            selected=[i*len(g['sequences'])//n for i in range(n)]
            for method in cfg['methods']:
                rates=cfg['learning_rates'][method] if development else [frozen['selected'][f'{family}/{task}/{method}']['lr']]
                for lr in rates:
                    armname=f'{method}_lr{lr:g}' if development else method
                    arm={'name':armname,'method':method,'lr':lr,'steps':cfg['steps'],
                        'kl':cfg['kl_weight_todna'] if method=='todna' else 0.,'edit':cfg['edit_fraction_weight_ledidi'] if method=='ledidi' else 0.}
                    for seed in seeds:
                        out=output/family/task/armname/f'seed{seed}'
                        if (out/'complete.json').exists():
                            d=json.loads((out/'complete.json').read_text())
                            if sha(out/'lock.json')!=d['lock_sha256'] or sha(out/'queries.jsonl')!=d['queries_sha256']:raise ValueError('Existing run modified')
                            continue
                        if out.exists():raise RuntimeError(f'Incomplete run needs inspection: {out}')
                        out.mkdir(parents=True);rec=SharedInitialPredictor(predictor,family,task);details=[];started=time.time();torch.cuda.reset_peak_memory_stats()
                        ns=cfg['samples_per_start']
                        for position,index in enumerate(selected):
                            seq=g['sequences'][index];rec.start=seq;initial=one_hot([seq],'cuda');rs=seed+position*1000
                            if method=='todna':
                                rec.trajectory=[position*ns+j for j in range(ns)]
                                values=run_todna_batch(rec,initial,g['logits']['lora'][index:index+1].cuda(),cfg['steps'],lr,arm['kl'],0.,cfg['noise_std_todna'],[rs+j for j in range(ns)],cfg['temperature_todna'])
                                for j,value in enumerate(values):details.append({'start_index':index,'restart':j,'search_seed':rs+j,**value})
                            elif method=='fastseqprop':
                                rec.trajectory=position;details.append({'start_index':index,**run_fastseqprop(rec,initial,cfg['steps'],lr,ns,rs)})
                            else:
                                rec.trajectory=position;details.append({'start_index':index,**run_ledidi(rec,initial,cfg['steps'],lr,ns,rs,arm['edit'])})
                        torch.cuda.synchronize();expected=n*(1+ns*(cfg['steps']+1))
                        if rec.sequence_calls!=expected:raise RuntimeError(f'Budget mismatch: {rec.sequence_calls} vs {expected}')
                        meta={'family':family,'task':task,'arm':arm,'seed':seed,'elapsed_seconds':time.time()-started,
                              'design_sha256':ch,'context_sha256':sha(cp),'n_starts':n,'samples_per_start':ns,'phase':phase}
                        summary=save_run(out,rec,details,meta,topk)
                        progress={'completed_runs':len(list(output.glob('*/*/*/seed*/complete.json'))),'family':family,'task':task,'method':method,'lr':lr,'seed':seed,'seconds':meta['elapsed_seconds'],'calls':expected,'selected_count':summary['selected_count']}
                        write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        del predictor;gc.collect();torch.cuda.empty_cache()
    paths=sorted(output.glob('*/*/*/seed*/lock.json'));expected=cfg['development_runs'] if development else cfg['formal_runs']
    if len(paths)!=expected:raise RuntimeError(f'Incomplete stage {len(paths)}/{expected}')
    write_json(output/'search_complete.json',{'state':'COMPLETE','runs':len(paths),'locks':{str(p.relative_to(output)):sha(p) for p in paths}})
    if development:
        selected={};scores={}
        for family,tasks in TASKS.items():
            for task in tasks:
                for method in cfg['methods']:
                    candidates=[]
                    for lr in cfg['learning_rates'][method]:
                        arm=f'{method}_lr{lr:g}';values=[]
                        for seed in seeds:
                            lock=json.loads((output/family/task/arm/f'seed{seed}'/'lock.json').read_text())
                            values.append(statistics.mean(x['reward'] for x in lock['candidates']))
                        candidates.append({'lr':lr,'mean_design_reward':statistics.mean(values),'per_seed_design_reward':values})
                    key=f'{family}/{task}/{method}';selected[key]=sorted(candidates,key=lambda v:(-v['mean_design_reward'],v['lr']))[0];scores[key]=candidates
        write_json(output/'frozen_selection.json',{'state':'FROZEN','selected':selected,'all_development_scores':scores,
            'search_complete_sha256':sha(output/'search_complete.json'),'selection_rule':cfg['development_selection'],
            'evaluation_read':False,'freeze_epoch':time.time()})

def evaluate(root,cfg):
    output=root/'results/nearest_comparison_v1';completion=json.loads((output/'search_complete.json').read_text());rows=[]
    if completion['runs']!=cfg['formal_runs']:raise RuntimeError('Missing search runs')
    for family in TASKS:
        checkpoint=root/'inputs/assets'/family/'evaluation/oracle.ckpt';ch=sha(checkpoint)
        predictor=load_activity_checkpoint(str(checkpoint),'cuda').eval()
        for rel,digest in completion['locks'].items():
            if not rel.startswith(family+'/'):continue
            path=output/rel
            if sha(path)!=digest:raise ValueError('Candidate lock changed')
            lock=json.loads(path.read_text());seqs=[x['sequence'] for x in lock['candidates']];meta=lock['meta'];values=[]
            with torch.inference_mode():
                for i in range(0,len(seqs),32):values.append(predictor(one_hot(seqs[i:i+32],'cuda')).float().cpu())
            y=torch.cat(values)
            if not torch.isfinite(y).all():raise RuntimeError('Nonfinite Evaluation')
            mm=score(y,family,meta['task']);mg=score(y,family,meta['task'],'min')
            row={'family':family,'task':meta['task'],'arm':meta['arm']['method'],'seed':meta['seed'],'n':len(seqs),
                'mean_margin':float(mm.mean()),'min_gap':float(mg.mean()),'positive_min_gap_fraction':float((mg>0).float().mean()),
                'lock_sha256':digest,'evaluation_checkpoint_sha256':ch,'raw_activity_mean':y.mean(0).tolist()}
            write_json(path.parent/'evaluation.json',{'summary':row,'sequences':seqs,'predictions':y.tolist()});rows.append(row)
        del predictor;gc.collect();torch.cuda.empty_cache()
    write_json(output/'evaluation_summary.json',rows);write_json(output/'evaluation_complete.json',{'state':'COMPLETE','n_runs':len(rows),'scope':cfg['scope']})

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['development','formal','evaluate'],required=True)
    a=p.parse_args();cfg=json.loads((a.root/'configs/comparison_v1.json').read_text());torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    if a.phase=='evaluate':evaluate(a.root,cfg)
    else:run_search(a.root,cfg,a.phase)
if __name__=='__main__':main()
