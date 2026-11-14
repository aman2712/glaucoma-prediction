"""Frozen grouped outer CV, inner epoch/threshold selection, E0 versus E2 only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch
import train


def digest(path):
    return train.core.sha256_file(path)


def assignments(development):
    data=development.sort_values('image_id').reset_index(drop=True)
    if data.analysis_group.eq('').any() or data.image_id.duplicated().any():
        raise ValueError('Missing group or duplicated ID')
    result=[]
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=73115)
    for fold,(remaining,held) in enumerate(outer.split(data,data.binary_label,data.analysis_group)):
        pool=data.iloc[remaining]
        inner=StratifiedGroupKFold(5,shuffle=True,random_state=83115+fold)
        fit,select=next(inner.split(pool,pool.binary_label,pool.analysis_group))
        for role,rows in [('fit',pool.iloc[fit]),('select',pool.iloc[select]),('held',data.iloc[held])]:
            for row in rows.itertuples():
                result.append(dict(image_id=row.image_id,analysis_group=row.analysis_group,
                                   binary_label=int(row.binary_label),fold=fold,role=role))
    table=pd.DataFrame(result)
    for fold,rows in table.groupby('fold'):
        if rows.image_id.duplicated().any() or (rows.groupby('analysis_group').role.nunique()>1).any():
            raise ValueError(f'Group/identity leakage in fold {fold}')
        for _,part in rows.groupby('role'):
            if set(part.binary_label)!={0,1}:raise ValueError('A fold role lost a class')
    if set(table[table.role=='held'].image_id)!=set(data.image_id):
        raise ValueError('Incomplete outer coverage')
    return table


def prepare(study,manifest):
    parts=train.load_development(manifest)
    dev=pd.concat(list(parts.values()),ignore_index=True)
    table=assignments(dev)
    study.mkdir(parents=True,exist_ok=False)
    table.to_csv(study/'assignments.csv',index=False)
    dev.to_csv(study/'development.csv',index=False)
    sources={str(p.relative_to(train.HERE)):digest(p) for p in
             [Path(__file__),train.HERE/'train.py',train.HERE/'baseline_utils.py',train.HERE/'plan.json']}
    protocol=dict(study='E0_vs_E2_grouped_nested_split_confirmation',folds=5,seeds=[2440,2441,2442],
                  experiments=['E0','E2'],outer_seed=73115,inner_seed_base=83115,
                  source_manifest_sha256=digest(manifest),assignments_sha256=digest(study/'assignments.csv'),
                  development_sha256=digest(study/'development.csv'),sources=sources,
                  primary='Mean paired outer-fold AUROC difference, E2 minus E0, using mean-logit seed ensembles within fold',
                  secondary='Per-seed mean outer-fold AUROC, AP, fixed-inner-threshold metrics and calibration',
                  uncertainty='2000 paired analysis-group bootstrap resamples within each outer fold; conditional on fitted models',
                  evidence_limit='Post-selection development split sensitivity analysis; not independent confirmation; unknown patient linkage remains',
                  selection='Inner select role chooses epoch and threshold; held role never enters training/selection',
                  success_rule='Primary difference positive with grouped-bootstrap 95% interval above zero; report all seed/fold results regardless')
    train.core.write_json(study/'protocol.json',protocol)
    table.groupby(['fold','role']).binary_label.agg(['count','sum']).to_csv(study/'fold_counts.csv')
    print((study/'fold_counts.csv').read_text(),flush=True)
    print('Protocol frozen:',study,flush=True)


def verify(study):
    protocol=json.loads((study/'protocol.json').read_text())
    for file,key in [('assignments.csv','assignments_sha256'),('development.csv','development_sha256')]:
        if digest(study/file)!=protocol[key]:raise ValueError('Frozen assignments changed')
    for relative,value in protocol['sources'].items():
        if digest(train.HERE/relative)!=value:raise ValueError(f'Code changed after freeze: {relative}')
    return protocol


def execute(study,artifacts,task):
    protocol=verify(study)
    if not 0<=task<30:raise ValueError('Task must be 0..29')
    fold=task//6;experiment=['E0','E2'][(task%6)//3];seed=protocol['seeds'][task%3]
    dest=study/'runs'/f'{experiment}_fold{fold}_seed{seed}'
    if dest.exists():raise FileExistsError('Run exists; inspect it before retrying')
    if not torch.cuda.is_available():raise RuntimeError('Submit using a GPU allocation')
    data=train.core.read_table(study/'development.csv')
    data['binary_label']=data.binary_label.astype(int)
    mapping=pd.read_csv(study/'assignments.csv')
    role=mapping[mapping.fold==fold].set_index('image_id').role
    data['cv_role']=data.image_id.map(role)
    parts={}
    for split,cvrole in [('train','fit'),('validation','select')]:
        part=data[data.cv_role==cvrole].copy();part['split']=split
        parts[split]=part.reset_index(drop=True)
    config=train.configuration(experiment,seed)
    config.update(outer_fold=fold,study_protocol_sha256=digest(study/'protocol.json'))
    device=torch.device('cuda')
    result=train.run(config,parts,artifacts,dest,device,manifest_hash=protocol['source_manifest_sha256'])
    # Epoch and threshold are now fixed. Only here do we open the outer-held images.
    held=data[data.cv_role=='held'].copy();held['split']='validation'
    train.core.check_images(artifacts,held)
    model=train.AnatomyModel(config,pretrained=False).to(device)
    checkpoint=torch.load(dest/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state'],strict=True)
    _,pred=train.epoch(model,train.loader(held,artifacts,config,False,{}),device)
    pred=train.core.attach_metadata(pred,held)
    pred.to_csv(dest/'outer_predictions.csv',index=False)
    metrics=train.core.metric_values(pred.label,pred.probability,result['threshold'])
    train.core.write_json(dest/'outer_metrics.json',metrics)
    train.core.write_json(dest/'outer_completion.json',dict(complete=True,fold=fold,experiment=experiment,
                          seed=seed,protocol_sha256=digest(study/'protocol.json'),
                          checkpoint_sha256=digest(dest/'best.pt'),threshold=result['threshold']))
    print('Outer evaluation complete:',fold,experiment,seed,metrics,flush=True)


def aggregate(study):
    verify(study)
    records=[];ensemble={};seed_records=[]
    for fold in range(5):
        for exp in ['E0','E2']:
            frames=[]
            for seed in [2440,2441,2442]:
                run=study/'runs'/f'{exp}_fold{fold}_seed{seed}'
                if not (run/'outer_completion.json').is_file():
                    raise ValueError(f'Missing outer completion: {run}')
                item=json.loads((run/'outer_metrics.json').read_text())
                records.append(dict(fold=fold,experiment=exp,seed=seed,**item))
                p=pd.read_csv(run/'outer_predictions.csv').sort_values('image_id').reset_index(drop=True)
                frames.append(p)
            reference=frames[0]
            for p in frames[1:]:
                if not p[['image_id','label','analysis_group']].equals(reference[['image_id','label','analysis_group']]):
                    raise ValueError('Unpaired predictions')
            logits=[np.log(np.clip(p.probability.to_numpy(),1e-7,1-1e-7)/(1-np.clip(p.probability.to_numpy(),1e-7,1-1e-7))) for p in frames]
            reference=reference.copy();reference['probability']=1/(1+np.exp(-np.mean(logits,axis=0)))
            ensemble[(fold,exp)]=reference
    out=study/'comparison'
    out.mkdir(exist_ok=False)
    pd.DataFrame(records).to_csv(out/'per_seed_fold_metrics.csv',index=False)
    rows=[]
    for fold in range(5):
        a,b=ensemble[(fold,'E0')],ensemble[(fold,'E2')]
        if not a[['image_id','label','analysis_group']].equals(b[['image_id','label','analysis_group']]):
            raise ValueError('Unpaired arm predictions')
        for exp,p in [('E0',a),('E2',b)]:
            rows.append(dict(fold=fold,experiment=exp,auroc=train.core.roc_auc_score(p.label,p.probability),
                             average_precision=train.core.average_precision_score(p.label,p.probability)))
    table=pd.DataFrame(rows);table.to_csv(out/'ensemble_fold_metrics.csv',index=False)
    diff=table.pivot(index='fold',columns='experiment',values='auroc');point=float((diff.E2-diff.E0).mean())
    rng=np.random.default_rng(93115);samples=[]
    for _ in range(2000):
        differences=[]
        for fold in range(5):
            a,b=ensemble[(fold,'E0')],ensemble[(fold,'E2')]
            groups=[np.asarray(i) for i in a.groupby('analysis_group').indices.values()]
            idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
            if len(np.unique(a.label.to_numpy()[idx]))<2:break
            differences.append(train.core.roc_auc_score(b.label.to_numpy()[idx],b.probability.to_numpy()[idx])-
                               train.core.roc_auc_score(a.label.to_numpy()[idx],a.probability.to_numpy()[idx]))
        if len(differences)==5:samples.append(float(np.mean(differences)))
    lo,hi=np.quantile(samples,[.025,.975])
    result=dict(mean_paired_fold_auroc_difference=point,lower_95=float(lo),upper_95=float(hi),
                valid_bootstrap_resamples=len(samples),success_rule_met=bool(point>0 and lo>0),
                interpretation='Development split sensitivity analysis after model selection; not independent clinical validation')
    train.core.write_json(out/'primary.json',result)
    print(table.to_string(index=False));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','summarize'])
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path('splits/all.csv'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    p.add_argument('--task',type=int)
    args=p.parse_args()
    if args.action=='prepare':prepare(args.study,args.manifest)
    elif args.action=='run':execute(args.study,args.artifacts,args.task)
    else:aggregate(args.study)
