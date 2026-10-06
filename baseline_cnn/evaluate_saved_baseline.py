"""Evaluate the existing baseline checkpoint; never train or select on test data."""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import pandas as pd
import torch
from baseline_core import (
    final_evaluation, freeze_run, select_threshold, sha256_file,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--artifacts-dir', required=True, type=Path)
    parser.add_argument('--check-only', action='store_true',
                        help='Check saved protocol on CPU without evaluating tests or writing files.')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    root = args.artifacts_dir.resolve()
    if not (root / 'prepared').is_dir():
        raise ValueError('Missing artifacts/prepared directory')
    if (run / 'final_evaluation').exists():
        raise ValueError('final_evaluation already exists. Inspect it; do not overwrite or delete it.')
    audit = json.loads((run / 'input_audit.json').read_text())
    manifest_hash = sha256_file(run / 'effective_splits/all.csv')
    if manifest_hash != audit['effective_manifest_sha256']:
        raise ValueError('Effective manifest differs from the recorded training run')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    config = json.loads((run / 'config.json').read_text())
    if config != checkpoint['config']:
        raise ValueError('Saved configuration differs from checkpoint')
    if checkpoint['best_epoch'] != 8 or config['model_name'] != 'resnet18':
        raise ValueError('Expected the recorded epoch-8 ResNet-18 baseline')
    metrics = pd.read_csv(run / 'validation_metrics.csv')
    selected = metrics.loc[metrics.operating_point == 'validation_selected']
    if len(selected) != 1:
        raise ValueError('Expected exactly one validation-selected operating point')
    threshold = float(selected.iloc[0]['threshold'])
    if not math.isclose(threshold, 0.00786661822348833, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Threshold differs from the recorded baseline')
    pred = pd.read_csv(run / 'validation_predictions.csv')
    derived = select_threshold(pred.label, pred.probability, config['target_sensitivity'])
    if not math.isclose(threshold, derived, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Threshold does not match saved validation predictions')
    print('Existing checkpoint:', run / 'best.pt', flush=True)
    print('Selected epoch:', checkpoint['best_epoch'], 'Threshold:', threshold, flush=True)
    print('Effective manifest checksum verified. No training will run.', flush=True)
    if args.check_only:
        print('Preflight passed. No test inference or output writes performed.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Submit this command through a Slurm GPU job.')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    freeze_run(run, threshold, manifest_hash, checkpoint)
    comparison = final_evaluation(run, root, torch.device('cuda'), iterations=1000)
    print(comparison.to_string(index=False), flush=True)
    print('Complete. Results:', run / 'final_evaluation', flush=True)


if __name__ == '__main__':
    main()
