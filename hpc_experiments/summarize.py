"""Build meeting tables from completed development runs; exclude synthetic checks."""
import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(root, selected_runs=None):
    if selected_runs is not None:
        if not selected_runs or len(selected_runs)!=len(set(selected_runs)):
            raise ValueError('Selection must contain unique run directory names')
        for name in selected_runs:
            if Path(name).name!=name or not (Path(root)/name/'summary.json').is_file():
                raise ValueError(f'Invalid or missing selected run: {name}')
    records = []
    for path in sorted(Path(root).glob('*/summary.json')):
        if selected_runs is not None and path.parent.name not in selected_runs:
            continue
        entry = json.loads(path.read_text())
        if entry.get('synthetic'):
            if selected_runs is not None:
                raise ValueError('Selected run is synthetic')
            continue
        if json.loads((path.parent/'status.json').read_text())['status'] != 'complete':
            if selected_runs is not None:
                raise ValueError('Selected run is incomplete')
            continue
        config = entry['config']
        metrics = entry['metrics']
        records.append(dict(experiment=config['experiment'], seed=config['seed'],
                            name=config['name'], hypothesis=config['hypothesis'],
                            best_epoch=entry['best_epoch'], threshold=entry['threshold'],
                            manifest_sha256=entry['manifest_sha256'],
                            development_images_sha256=entry['development_images_sha256'],
                            source_fingerprint=entry['source_fingerprint'],
                            settings=json.dumps({k:v for k,v in config.items() if k!='seed'},sort_keys=True),
                            mask_audit=json.dumps(entry.get('mask_audit',{}),sort_keys=True),
                            **{key:metrics[key] for key in ['auroc','average_precision','sensitivity','specificity','brier','log_loss']},
                            run_directory=str(path.parent), outcome=entry['outcome']))
    if not records:
        raise ValueError('No completed real-data runs; synthetic software checks are excluded')
    runs = pd.DataFrame(records)
    if runs.duplicated(['experiment','seed']).any():
        raise ValueError('Multiple runs for the same experiment/seed. Select a prespecified run set; do not cherry-pick.')
    if runs.manifest_sha256.nunique()!=1 or runs.development_images_sha256.nunique()!=1:
        raise ValueError('Do not pool runs from different cohorts or prepared image exports')
    if runs.source_fingerprint.nunique()!=1 or (runs.groupby('experiment').settings.nunique()>1).any():
        raise ValueError('Code/configuration changed within the study; report a separately versioned experiment')
    masks=runs.loc[runs.mask_audit!='{}','mask_audit']
    if masks.nunique()>1:
        raise ValueError('Supervision masks changed between anatomy experiments')
    rows = []
    for exp, group in runs.groupby('experiment'):
        row = dict(experiment=exp, name=group.name.iloc[0], n_seeds=len(group),
                   seeds=','.join(map(str,sorted(group.seed))),
                   complete_three_seed_comparison=set(group.seed)=={2440,2441,2442})
        for metric in ['auroc','average_precision','sensitivity','specificity','brier','log_loss']:
            row[metric+'_mean'] = group[metric].mean()
            row[metric+'_sd'] = group[metric].std(ddof=1)
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(['auroc_mean','average_precision_mean'],ascending=False)
    paired = []
    for treatment,control in [('E1','E0'),('E2','E0'),('E3','E0'),('E4','E3'),('E5','E4')]:
        left=runs[runs.experiment==treatment].set_index('seed')
        right=runs[runs.experiment==control].set_index('seed')
        for seed in sorted(set(left.index)&set(right.index)):
            paired.append(dict(treatment=treatment,control=control,seed=seed,
                               delta_validation_auroc=left.loc[seed,'auroc']-right.loc[seed,'auroc'],
                               delta_validation_ap=left.loc[seed,'average_precision']-right.loc[seed,'average_precision']))
    return runs,comparison,pd.DataFrame(paired)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,
                        help='New output directory, e.g. hpc_experiments/results/meeting_01')
    parser.add_argument('--selection-file',type=Path,
                        help='JSON list of run directory basenames, chosen independently of scores')
    args=parser.parse_args()
    selected=json.loads(args.selection_file.read_text()) if args.selection_file else None
    runs,comparison,paired=summarize(args.runs,selected)
    args.output.mkdir(parents=True,exist_ok=False)
    if selected is not None:
        (args.output/'selected_runs.json').write_text(json.dumps(selected,indent=2)+'\n')
    runs.to_csv(args.output/'experiment_log.csv',index=False)
    comparison.to_csv(args.output/'validation_comparison.csv',index=False)
    paired.to_csv(args.output/'paired_seed_differences.csv',index=False)
    print(comparison.to_string(index=False))
    print('Selection uses validation only. Three-seed results are required for the planned comparison.')
