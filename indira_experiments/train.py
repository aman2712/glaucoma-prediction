"""Development-only anatomy-informed experiments; no held-out evaluation entry point."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
import baseline_utils as core

EXPECTED_MANIFEST = '4590df4ee6e6ce8ed722786cdb49220b632468184222fb9760c396e5cbf37094'
RATIO_NAMES = ['vertical_cdr', 'horizontal_cdr', 'area_cdr']


def configuration(experiment, seed, plan=None):
    plan = plan or json.loads((HERE / 'plan.json').read_text())
    entry = plan['experiments'][experiment]
    config = dict(plan['base'], **entry['changes'], seed=seed,
                  experiment=experiment, name=entry['name'], hypothesis=entry['hypothesis'])
    if config['fusion'] and not config['auxiliary']:
        raise ValueError('Fusion requires the supervised anatomy head')
    if config['partial'] and config['model_name'] != 'resnet18':
        raise ValueError('Partial fine-tuning is implemented only for ResNet-18')
    if not 0 < config['warmup_epochs'] < config['epochs']:
        raise ValueError('Require at least one head-only and one fine-tuning epoch')
    return config


def load_development(manifest):
    manifest = Path(manifest).resolve()
    if core.sha256_file(manifest) != EXPECTED_MANIFEST:
        raise ValueError('Use the current team splits/all.csv (4590df4e...).')
    table = core.read_table(manifest)
    required = {'image_id', 'analysis_group', 'patient_id', 'pixel_sha256', 'split',
                'classification_eligible', 'binary_label', 'classification_image',
                'segmentation_eligible', 'disc_mask', 'cup_mask', 'camera', 'dataset'}
    if required - set(table):
        raise ValueError(f'Manifest missing: {sorted(required-set(table))}')
    if table.image_id.duplicated().any():
        raise ValueError('Duplicate image IDs')
    active = table[table.split.isin(core.PARTITIONS)]
    for key in ['analysis_group', 'patient_id', 'pixel_sha256']:
        known = active[active[key] != '']
        if (known.groupby(key).split.nunique() > 1).any():
            raise ValueError(f'Known {key} crosses partitions')
    parts = {}
    for split, expected in [('train', 808), ('validation', 200)]:
        rows = table[(table.split == split) & (table.classification_eligible == '1')].copy()
        if len(rows) != expected or set(rows.binary_label) != {'0', '1'}:
            raise ValueError(f'Unexpected {split} cohort')
        if set(rows.dataset) != {'chaksu'}:
            raise ValueError('Development must contain CHAKSU only')
        rows['binary_label'] = rows.binary_label.astype(int)
        parts[split] = rows.reset_index(drop=True)
    return parts


def safe_mask_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not relative or Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError('Mask must have a relative path within artifacts')
    return path


def ratios_from_masks(disc, cup):
    disc, cup = np.asarray(disc) > 0, np.asarray(cup) > 0
    if disc.ndim != 2 or disc.shape != cup.shape:
        raise ValueError('Disc/cup masks must have the same 2D shape')
    if not disc.any() or not cup.any() or (cup & ~disc).any():
        raise ValueError('Empty or non-nested anatomy target; check segmentation eligibility')
    dy, dx = np.where(disc)
    cy, cx = np.where(cup)
    return np.asarray([(cy.max()-cy.min()+1)/(dy.max()-dy.min()+1),
                       (cx.max()-cx.min()+1)/(dx.max()-dx.min()+1),
                       cup.sum()/disc.sum()], dtype=np.float32)


def anatomy_targets(rows, root):
    """Read only development masks. Invalid eligibility is a missing target, not a zero CDR."""
    if not set(rows.split).issubset({'train', 'validation'}):
        raise ValueError('Anatomical supervision is development-only')
    result = {}
    digest = hashlib.sha256()
    for row in rows.sort_values('image_id').itertuples():
        if str(row.segmentation_eligible) != '1':
            continue
        paths = [safe_mask_path(root, row.disc_mask), safe_mask_path(root, row.cup_mask)]
        masks = []
        for path in paths:
            digest.update(str(path.relative_to(Path(root).resolve())).encode())
            digest.update(core.sha256_file(path).encode())
            with Image.open(path) as mask:
                if mask.mode not in ('1', 'L', 'I', 'I;16'):
                    raise ValueError(f'Expected a single-channel binary mask: {path}')
                arr = np.asarray(mask)
                if arr.ndim != 2 or not set(np.unique(arr)).issubset({0, 1, 255, 65535}):
                    raise ValueError(f'Mask is not binary: {path}')
                masks.append(arr)
        result[row.image_id] = ratios_from_masks(*masks)
    if not result:
        raise ValueError('No usable development masks; obtain the intact prepared masks')
    return result, digest.hexdigest()


class DevelopmentDataset(Dataset):
    def __init__(self, rows, root, config, training, targets=None):
        if not set(rows.split).issubset({'train', 'validation'}):
            raise ValueError('The experiment runner cannot load test images')
        self.rows = rows.reset_index(drop=True)
        self.root, self.targets = Path(root), targets or {}
        jitter = (.30, .30, .15) if config['strong_colour'] else (.10, .10, .05)
        ops = [transforms.ColorJitter(brightness=jitter[0], contrast=jitter[1], saturation=jitter[2])] if training else []
        self.transform = transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(core.MEAN, core.STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(core.image_path(self.root, row.classification_image)) as image:
            x = self.transform(image.convert('RGB'))
        target = self.targets.get(row.image_id)
        return (x, torch.tensor(float(row.binary_label)),
                torch.tensor(target if target is not None else np.zeros(3), dtype=torch.float32),
                torch.tensor(target is not None), row.image_id)


class AnatomyModel(nn.Module):
    """At inference only an RGB tensor is accepted; expert CDRs are loss targets only."""
    def __init__(self, config, pretrained=True):
        super().__init__()
        self.config = dict(config)
        if config['model_name'] == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights if pretrained else None)
            width = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif config['model_name'] == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights if pretrained else None)
            width = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Identity()
        else:
            raise ValueError('Unknown architecture')
        self.weights_id = str(weights) if pretrained else 'NONE_SYNTHETIC_ONLY'
        # Independent initializers keep the shared classifier identical in E0/E3
        # and the anatomy head identical in E3/E4 for each paired seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'] + 101)
            self.classifier = nn.Linear(width + (3 if config['fusion'] else 0), 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'] + 102)
            self.anatomy = nn.Linear(width, 3) if config['auxiliary'] else None

    def forward(self, image):
        features = self.backbone(image)
        ratios = torch.sigmoid(self.anatomy(features)) if self.anatomy is not None else None
        inputs = torch.cat([features, ratios], dim=1) if self.config['fusion'] else features
        return self.classifier(inputs).flatten(), ratios

    def set_phase(self, head_only):
        for name, param in self.backbone.named_parameters():
            param.requires_grad_(not head_only and (not self.config['partial'] or name.startswith('layer4.')))
        self.train()
        if head_only:
            self.backbone.eval()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def losses(logits, ratios, labels, targets, available, weight):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    # Only supervised rows contribute; no anatomy-imputation pseudo-labels.
    auxiliary = (nn.functional.smooth_l1_loss(ratios[available], targets[available])
                 if ratios is not None and available.any() else bce.new_zeros(()))
    return bce + weight * auxiliary, bce, auxiliary


def loader(rows, root, config, training, targets):
    return DataLoader(DevelopmentDataset(rows, root, config, training, targets),
                      batch_size=config['batch_size'], shuffle=training, num_workers=0,
                      generator=torch.Generator().manual_seed(config['seed']))


def epoch(model, batches, device, optimizer=None, head_only=False):
    if optimizer is None:
        model.eval()
    else:
        model.set_phase(head_only)
    totals = dict(loss=0., bce=0., anatomy_sum=0., anatomy_n=0, n=0)
    records = []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, target, valid, ids in batches:
            x, y, target, valid = [v.to(device) for v in (x, y, target, valid)]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits, ratios = model(x)
            loss, bce, auxiliary = losses(logits, ratios, y, target, valid, model.config['anatomy_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            size = len(y)
            totals['loss'] += loss.item()*size
            totals['bce'] += bce.item()*size
            totals['n'] += size
            if ratios is not None:
                totals['anatomy_sum'] += auxiliary.item()*valid.sum().item()
                totals['anatomy_n'] += valid.sum().item()
            for i, image_id in enumerate(ids):
                row = dict(image_id=image_id, label=int(y[i].item()), probability=float(logits[i].sigmoid().item()))
                if ratios is not None:
                    row.update({name: float(ratios[i,j].item()) for j,name in enumerate(RATIO_NAMES)})
                records.append(row)
    stats = {key: totals[key]/totals['n'] for key in ['loss', 'bce']}
    stats['anatomy_smooth_l1'] = totals['anatomy_sum']/totals['anatomy_n'] if totals['anatomy_n'] else None
    return stats, pd.DataFrame(records)


def git_value(*args):
    proc = subprocess.run(['git', '-C', str(HERE.parent), *args], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else 'UNAVAILABLE'


def run(config, parts, root, out, device, pretrained=True, manifest_hash=EXPECTED_MANIFEST):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    core.write_json(out/'status.json', dict(status='running', synthetic=not pretrained))
    try:
        core.seed_everything(config['seed'])
        core.write_json(out/'config.json', config)
        versions = {name: importlib.metadata.version(name) for name in
                    ['torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow']}
        sources = {str(p.relative_to(HERE.parent)): core.sha256_file(p) for p in
                   [Path(__file__), HERE/'plan.json', HERE/'baseline_utils.py']}
        core.write_json(out/'provenance.json', dict(manifest_sha256=manifest_hash,
                        git_commit=git_value('rev-parse', 'HEAD'), git_status=git_value('status', '--short'),
                        source_sha256=sources, python=sys.version, packages=versions,
                        device=str(device), cuda=torch.version.cuda,
                        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                        synthetic=not pretrained))
        for filename, path in [('train_source.py', Path(__file__)), ('plan.json', HERE/'plan.json'),
                               ('baseline_utils.py', HERE/'baseline_utils.py')]:
            (out/filename).write_bytes(path.read_bytes())
        environment = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
        (out/'pip_freeze.txt').write_text(environment.stdout if environment.returncode == 0 else environment.stderr)
        image_digest = hashlib.sha256()
        for split in ['train', 'validation']:
            core.check_images(root, parts[split])
            parts[split].to_csv(out/f'{split}_manifest.csv', index=False)
            for row in parts[split].sort_values('image_id').itertuples():
                image_digest.update(row.image_id.encode())
                image_digest.update(core.sha256_file(core.image_path(root,row.classification_image)).encode())
        image_hash = image_digest.hexdigest()
        core.write_json(out/'prepared_images.json', dict(development_images_sha256=image_hash,
                        train=len(parts['train']),validation=len(parts['validation'])))
        target_maps = {'train': {}, 'validation': {}}
        mask_audit = {}
        if config['auxiliary']:
            for split in ['train', 'validation']:
                target_maps[split], digest = anatomy_targets(parts[split], root)
                mask_audit[split] = dict(supervised_images=len(target_maps[split]), masks_sha256=digest)
                pd.DataFrame([dict(image_id=k, **dict(zip(RATIO_NAMES, v)))
                              for k,v in target_maps[split].items()]).to_csv(out/f'{split}_anatomy_targets.csv', index=False)
        core.write_json(out/'mask_audit.json', mask_audit)
        model = AnatomyModel(config, pretrained).to(device)
        core.write_json(out/'model.json', dict(weights=model.weights_id,
                        parameters=sum(p.numel() for p in model.parameters()),
                        inference_inputs='RGB photograph only', anatomy_targets='expert masks, development only'))
        train_batches = loader(parts['train'], root, config, True, target_maps['train'])
        val_batches = loader(parts['validation'], root, config, False, target_maps['validation'])
        best_auc, best_bce, wait, history = -float('inf'), float('inf'), 0, []
        best_path = out/'best.pt'
        optimizer = None
        for index in range(config['epochs']):
            head_only = index < config['warmup_epochs']
            if index in [0, config['warmup_epochs']]:
                model.set_phase(head_only)
                heads = list(model.classifier.parameters()) + (list(model.anatomy.parameters()) if model.anatomy is not None else [])
                groups = [dict(params=heads, lr=config['head_lr'])]
                backbone = [p for p in model.backbone.parameters() if p.requires_grad]
                if backbone:
                    groups.append(dict(params=backbone, lr=config['backbone_lr']))
                optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
            started = time.monotonic()
            train_stats, _ = epoch(model, train_batches, device, optimizer, head_only)
            val_stats, pred = epoch(model, val_batches, device)
            metrics = core.metric_values(pred.label, pred.probability, .5)
            entry = dict(epoch=index+1, phase='head' if head_only else 'fine_tune',
                         train_bce=train_stats['bce'], validation_bce=val_stats['bce'],
                         train_loss=train_stats['loss'], validation_loss=val_stats['loss'],
                         train_anatomy_loss=train_stats['anatomy_smooth_l1'],
                         validation_anatomy_loss=val_stats['anatomy_smooth_l1'],
                         validation_auroc=metrics['auroc'], validation_ap=metrics['average_precision'],
                         seconds=time.monotonic()-started)
            history.append(entry)
            pd.DataFrame(history).to_csv(out/'history.csv', index=False)
            auc, bce = metrics['auroc'], val_stats['bce']
            improved = not head_only and (auc > best_auc+1e-8 or (abs(auc-best_auc)<=1e-8 and bce<best_bce))
            if improved:
                best_auc, best_bce, wait = auc, bce, 0
                torch.save(dict(model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                config=config, weights=model.weights_id, best_epoch=index+1), best_path)
            elif not head_only:
                wait += 1
            print(json.dumps(entry), flush=True)
            if not head_only and wait >= config['patience']:
                break
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        _, predictions = epoch(model, val_batches, device)
        predictions = core.attach_metadata(predictions, parts['validation'])
        threshold = core.select_threshold(predictions.label, predictions.probability, config['target_sensitivity'])
        predictions.to_csv(out/'validation_predictions.csv', index=False)
        overall, cameras = core.evaluation_tables(predictions, threshold)
        overall.to_csv(out/'validation_metrics.csv', index=False)
        cameras.to_csv(out/'validation_camera_metrics.csv', index=False)
        if config['auxiliary']:
            joined = predictions[predictions.image_id.isin(target_maps['validation'])]
            mae = {name: float(np.mean([abs(getattr(row,name)-target_maps['validation'][row.image_id][j])
                                       for row in joined.itertuples()])) for j,name in enumerate(RATIO_NAMES)}
            core.write_json(out/'validation_anatomy_mae.json', dict(n=len(joined), **mae))
        import matplotlib.pyplot as plt
        core.diagnostic_plot(predictions, threshold, 'Development validation', out/'validation_curves.png')
        plt.close('all')
        core.error_gallery(predictions, root, threshold, out/'validation_errors.png')
        plt.close('all')
        h = pd.DataFrame(history)
        fig, axes = plt.subplots(1,2,figsize=(10,3))
        axes[0].plot(h.epoch,h.train_bce,label='Train BCE'); axes[0].plot(h.epoch,h.validation_bce,label='Validation BCE')
        axes[1].plot(h.epoch,h.validation_auroc,label='Validation AUROC'); axes[1].plot(h.epoch,h.validation_ap,label='Validation AP')
        for ax in axes:
            ax.set_xlabel('Epoch'); ax.legend()
        fig.tight_layout(); fig.savefig(out/'learning_curves.png',dpi=150); plt.close(fig)
        summary = dict(config=config, synthetic=not pretrained, best_epoch=checkpoint['best_epoch'],
                       checkpoint_sha256=core.sha256_file(best_path), manifest_sha256=manifest_hash,
                       development_images_sha256=image_hash,
                       source_fingerprint=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
                       mask_audit=mask_audit,
                       threshold=threshold, metrics=core.metric_values(predictions.label,predictions.probability,threshold),
                       outcome='Complete; inspect validation only. No test inference performed.')
        core.write_json(out/'summary.json', summary)
        core.write_json(out/'status.json', dict(status='complete', synthetic=not pretrained))
        return summary
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        core.write_json(out/'status.json', dict(status='failed', synthetic=not pretrained))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=list(json.loads((HERE/'plan.json').read_text())['experiments']))
    parser.add_argument('--seed', type=int, default=2440)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=HERE/'runs')
    parser.add_argument('--device', choices=['cuda','cpu','mps'], default='cuda')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = configuration(args.experiment,args.seed)
    parts = load_development(args.manifest)
    for rows in parts.values():
        core.check_images(args.artifacts,rows)
    if config['auxiliary']:
        for rows in parts.values():
            targets, _ = anatomy_targets(rows,args.artifacts)
            print('Anatomical targets:',len(targets),'of',len(rows),flush=True)
    if args.check_only:
        print('Preflight passed: exact effective manifest, development images and required masks.'); return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: submit on a Slurm GPU allocation')
    run_id = f'{args.experiment}_seed{args.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}'
    summary = run(config,parts,args.artifacts.resolve(),args.output/run_id,torch.device(args.device))
    print('Run complete:',args.output/run_id,flush=True)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
