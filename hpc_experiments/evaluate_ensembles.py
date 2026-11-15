"""Freeze validation-only ensemble decisions, then score held-out images once."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import train


def ensemble(frames):
    frames=[x.sort_values('image_id').reset_index(drop=True) for x in frames]
    first=frames[0].copy()
    for frame in frames[1:]:
        if not frame[['image_id','label','analysis_group']].equals(first[['image_id','label','analysis_group']]):
            raise ValueError('Unpaired ensemble predictions')
    values=[np.clip(x.probability.to_numpy(),1e-7,1-1e-7) for x in frames]
    first['probability']=1/(1+np.exp(-np.mean([np.log(x/(1-x)) for x in values],axis=0)))
    return first


def prepare(out,runs,selection,manifest,historical):
    parts=train.load_development(manifest)
    chosen=json.loads(selection.read_text())
    arms={};hashes={}
    for arm in ['E0','E2']:
        entries=[];frames=[]
        for name in chosen:
            path=runs/name
            summary=json.loads((path/'summary.json').read_text())
            if summary['config']['experiment']!=arm:continue
            if summary['synthetic'] or json.loads((path/'status.json').read_text())['status']!='complete':
                raise ValueError('Use completed real-data runs')
            if summary['manifest_sha256']!=train.EXPECTED_MANIFEST:raise ValueError('Wrong cohort')
            if train.core.sha256_file(path/'best.pt')!=summary['checkpoint_sha256']:raise ValueError('Changed checkpoint')
            predictions=pd.read_csv(path/'validation_predictions.csv')
            if set(predictions.image_id)!=set(parts['validation'].image_id):raise ValueError('Wrong validation cohort')
            truth=parts['validation'].set_index('image_id').binary_label
            if not np.array_equal(predictions.label,truth.loc[predictions.image_id].to_numpy()):raise ValueError('Validation labels changed')
            frames.append(predictions)
            entries.append(dict(run=str(path.resolve()),seed=summary['config']['seed'],best_epoch=summary['best_epoch'],
                                checkpoint_sha256=summary['checkpoint_sha256'],
                                validation_sha256=train.core.sha256_file(path/'validation_predictions.csv')))
            hashes[name]=summary['development_images_sha256']
        if sorted(x['seed'] for x in entries)!=[2440,2441,2442]:raise ValueError('Need exactly three prespecified seeds per arm')
        prediction=ensemble(frames)
        threshold=train.core.select_threshold(prediction.label,prediction.probability,.9)
        arms[arm]=dict(runs=entries,threshold=threshold,validation_metrics=train.core.metric_values(prediction.label,prediction.probability,threshold))
    if len(set(hashes.values()))!=1:raise ValueError('Development images differ across runs')
    prior={}
    if historical:
        for partition in ['in_domain_test','external_test']:
            p=historical/'final_evaluation'/f'{partition}_predictions.csv'
            prior[partition]=dict(path=str(p.resolve()),sha256=train.core.sha256_file(p))
    out.mkdir(parents=True,exist_ok=False)
    train.core.write_json(out/'protocol.json',dict(arms=arms,manifest=str(manifest.resolve()),
                          manifest_sha256=train.core.sha256_file(manifest),historical=prior,
                          method='mean of clipped probability logits across three seeds',
                          threshold_source='CHAKSU original validation only; >=90% sensitivity',
                          historical_threshold=0.00786661822348833,
                          source_sha256={p.name:train.core.sha256_file(p) for p in
                                         [Path(__file__),train.HERE/'train.py',train.HERE/'baseline_utils.py']},
                          bootstrap_resamples=2000,bootstrap_seed=151126,
                          primary_comparison='E2 ensemble minus E0 ensemble AUROC; separate CHAKSU and PAPILA estimates',
                          test_reuse='Prior baselines and P01 already evaluated these cohorts. This is exploratory reused-test evidence, not unexposed confirmation.'))
    print('Frozen ensemble protocol:',out/'protocol.json',flush=True)
    for arm,entry in arms.items():print(arm,'validation threshold',entry['threshold'],flush=True)


class TestPhotos(Dataset):
    def __init__(self,rows,root):
        self.rows=rows.reset_index(drop=True);self.root=root
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(train.core.MEAN,train.core.STD)])
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows.iloc[index]
        with Image.open(train.core.image_path(self.root,r.classification_image)) as im:x=self.transform(im.convert('RGB'))
        return x,torch.tensor(float(r.binary_label)),torch.zeros(3),torch.tensor(False),r.image_id


def bootstrap(predictions,thresholds,seed,iterations):
    names=list(predictions);reference=predictions[names[0]].sort_values('image_id').reset_index(drop=True)
    aligned={}
    for name in names:
        p=predictions[name].sort_values('image_id').reset_index(drop=True)
        if not p[['image_id','label','analysis_group']].equals(reference[['image_id','label','analysis_group']]):raise ValueError('Unpaired comparator')
        aligned[name]=p.probability.to_numpy()
    groups=[np.asarray(i) for i in reference.groupby('analysis_group').indices.values()]
    y=reference.label.to_numpy();rng=np.random.default_rng(seed)
    keys=['auroc','average_precision','sensitivity','specificity','balanced_accuracy','brier']
    samples={name:{key:[] for key in keys} for name in names}
    deltas={name:[] for name in names if name!='E2'}
    for _ in range(iterations):
        idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        if len(np.unique(y[idx]))<2:continue
        values={name:train.core.metric_values(y[idx],scores[idx],thresholds[name]) for name,scores in aligned.items()}
        for name in names:
            for key in keys:samples[name][key].append(values[name][key])
        for name in deltas:deltas[name].append(values['E2']['auroc']-values[name]['auroc'])
    rows=[];paired=[]
    for name in names:
        point=train.core.metric_values(y,aligned[name],thresholds[name])
        for key in keys:
            low,high=np.quantile(samples[name][key],[.025,.975])
            rows.append(dict(model=name,metric=key,estimate=point[key],lower_95=low,upper_95=high,
                             valid_resamples=len(samples[name][key]),analysis_groups=len(groups)))
    for name,values in deltas.items():
        low,high=np.quantile(values,[.025,.975])
        paired.append(dict(comparison='E2 minus '+name,metric='auroc',
                           estimate=train.core.roc_auc_score(y,aligned['E2'])-train.core.roc_auc_score(y,aligned[name]),
                           lower_95=low,upper_95=high,valid_resamples=len(values),analysis_groups=len(groups)))
    return pd.DataFrame(rows),pd.DataFrame(paired)


def evaluate(out,artifacts):
    protocol=json.loads((out/'protocol.json').read_text())
    if train.core.sha256_file(protocol['manifest'])!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    for name,value in protocol['source_sha256'].items():
        if train.core.sha256_file(train.HERE/name)!=value:raise ValueError('Source changed after freezing')
    for arm in protocol['arms'].values():
        for entry in arm['runs']:
            path=Path(entry['run'])
            if train.core.sha256_file(path/'best.pt')!=entry['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            if train.core.sha256_file(path/'validation_predictions.csv')!=entry['validation_sha256']:raise ValueError('Validation changed')
    for entry in protocol['historical'].values():
        if train.core.sha256_file(entry['path'])!=entry['sha256']:raise ValueError('Historical scores changed')
    if not torch.cuda.is_available():raise RuntimeError('Use a GPU allocation')
    dest=out/'evaluation';dest.mkdir(exist_ok=False)
    device=torch.device('cuda');all_rows=train.core.read_table(protocol['manifest'])
    metrics=[];intervals=[];paired=[]
    for partition,expected in [('in_domain_test',335),('external_test',420)]:
        rows=all_rows[(all_rows.split==partition)&(all_rows.classification_eligible=='1')].copy()
        rows['binary_label']=rows.binary_label.astype(int)
        if len(rows)!=expected:raise ValueError('Wrong test cohort')
        train.core.check_images(artifacts,rows)
        batches=DataLoader(TestPhotos(rows,artifacts),batch_size=16,shuffle=False,num_workers=0)
        predictions={};thresholds={}
        for name,arm in protocol['arms'].items():
            frames=[]
            for entry in arm['runs']:
                checkpoint=torch.load(Path(entry['run'])/'best.pt',map_location='cpu',weights_only=True)
                model=train.AnatomyModel(checkpoint['config'],pretrained=False).to(device)
                model.load_state_dict(checkpoint['model_state'],strict=True)
                _,p=train.epoch(model,batches,device)
                p=train.core.attach_metadata(p,rows)
                p.to_csv(dest/f'{partition}_{name}_seed{entry["seed"]}_predictions.csv',index=False)
                frames.append(p);del model
            predictions[name]=ensemble(frames);thresholds[name]=arm['threshold']
        if protocol['historical']:
            historical=pd.read_csv(protocol['historical'][partition]['path'])
            if set(historical.image_id)!=set(rows.image_id):raise ValueError('Historical cohort mismatch')
            # Attach current frozen group metadata; verify labels against the manifest.
            predictions['historical_B']=train.core.attach_metadata(historical[['image_id','label','probability']],rows)
            thresholds['historical_B']=protocol['historical_threshold']
        for name,p in predictions.items():
            p.to_csv(dest/f'{partition}_{name}_predictions.csv',index=False)
            metrics.append(dict(partition=partition,model=name,**train.core.metric_values(p.label,p.probability,thresholds[name])))
            _,cameras=train.core.evaluation_tables(p,thresholds[name])
            cameras.to_csv(dest/f'{partition}_{name}_camera_metrics.csv',index=False)
            train.core.diagnostic_plot(p,thresholds[name],partition+' '+name,dest/f'{partition}_{name}_curves.png')
            import matplotlib.pyplot as plt
            plt.close('all')
        ci,delta=bootstrap(predictions,thresholds,protocol['bootstrap_seed'],protocol['bootstrap_resamples'])
        ci.insert(0,'partition',partition);delta.insert(0,'partition',partition)
        intervals.append(ci);paired.append(delta)
        print('Completed partition',partition,flush=True)
    pd.DataFrame(metrics).to_csv(dest/'metrics.csv',index=False)
    pd.concat(intervals).to_csv(dest/'confidence_intervals.csv',index=False)
    pd.concat(paired).to_csv(dest/'paired_auroc_differences.csv',index=False)
    train.core.write_json(dest/'completion.json',dict(complete=True,protocol_sha256=train.core.sha256_file(out/'protocol.json')))
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print(pd.concat(paired).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','evaluate']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--runs',type=Path,default=Path('hpc_experiments/runs'))
    p.add_argument('--selection',type=Path,default=Path('hpc_experiments/results/all_six_review_20261114/selected_runs.json'))
    p.add_argument('--manifest',type=Path,default=Path('splits/all.csv'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    p.add_argument('--historical',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.out,a.runs,a.selection,a.manifest,a.historical)
    else:evaluate(a.out,a.artifacts)
