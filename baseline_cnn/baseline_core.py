"""Image-only baseline helpers. Embedded verbatim in the delivered notebook.

Only RGB photographs are model inputs. Labels/identities serve training/auditing.
No expert masks, measurements, votes, camera IDs, or clinical covariates enter CNN.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import random
import re
import time
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, confusion_matrix,
    brier_score_loss, log_loss, precision_recall_curve,
)

PARTITIONS = ('train', 'validation', 'in_domain_test', 'external_test')
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    # Invalid/undefined metrics are represented by null, never nonstandard NaN.
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [clean(v) for v in obj]
        if isinstance(obj, np.generic):
            return clean(obj.item())
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        if isinstance(obj, Path):
            return str(obj)
        return obj
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def read_table(path):
    # Empty patient IDs must stay empty, rather than becoming the string 'nan'.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


class Components:
    def __init__(self, ids):
        self.parent = {x: x for x in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def bosch_filename_key(row):
    # Precautionary identity constraint, NOT a claim of verified patient identity.
    if row['dataset'] == 'chaksu' and row['camera'] == 'Bosch':
        match = re.fullmatch(r'chaksu_Bosch_(?:train|test)_P(\d+)_Image[12]', row['image_id'], re.I)
        if match:
            return 'possible_bosch_P' + str(int(match.group(1)))
    return ''


def load_and_audit(artifacts):
    """Validate the real frozen manifest before selecting any training rows.

    Reads test metadata only to audit identity leakage; does not open test images.
    Counts in each CSV must agree with the hashed all.csv, not merely a summary.
    """
    root = Path(artifacts).expanduser().resolve()
    required = [root / 'prepared_manifest.csv', root / 'splits/all.csv',
                root / 'splits/split_summary.json']
    required += [root / 'splits' / f'{p}_classification.csv' for p in PARTITIONS]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            'The handoff summaries are not training data. Obtain the completed artifacts/ '
            'folder from your teammate (prepared images, prepared_manifest.csv, splits/). '
            'Do not invent/recreate the frozen split from the reported counts. Missing:\n' + '\n'.join(missing))
    summary = json.loads((root / 'splits/split_summary.json').read_text())
    for key, path in [('all_splits_sha256', root / 'splits/all.csv'),
                      ('manifest_sha256', root / 'prepared_manifest.csv')]:
        if not summary.get(key) or sha256_file(path) != summary[key]:
            raise ValueError(f'Frozen data integrity check failed: {key}')
    all_rows = read_table(root / 'splits/all.csv')
    fields = {'image_id', 'dataset', 'chaksu_release', 'camera', 'patient_id',
              'patient_id_verified', 'pixel_sha256', 'group_id', 'split',
              'classification_eligible', 'binary_label', 'original_label', 'classification_image'}
    if fields - set(all_rows):
        raise ValueError(f'Missing manifest columns: {sorted(fields - set(all_rows))}')
    if all_rows.empty or all_rows.image_id.duplicated().any() or (all_rows.image_id == '').any():
        raise ValueError('Empty or repeated image identity in all.csv')
    if not all_rows.classification_eligible.isin(['0', '1']).all():
        raise ValueError('Invalid classification eligibility flag')
    if (all_rows.pixel_sha256 == '').any() or (all_rows.group_id == '').any():
        raise ValueError('Missing duplicate hash or frozen group identity')
    for partition in PARTITIONS:
        table = read_table(root / 'splits' / f'{partition}_classification.csv')
        expected = all_rows[(all_rows.split == partition) & (all_rows.classification_eligible == '1')]
        if set(table.columns) != set(all_rows.columns):
            raise ValueError(f'{partition}: CSV columns differ from all.csv')
        left = table.sort_values('image_id').reset_index(drop=True)[all_rows.columns]
        right = expected.sort_values('image_id').reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f'{partition}: rows differ from frozen all.csv')
        if len(table) != summary.get('classification_counts', {}).get(partition):
            raise ValueError(f'{partition}: count differs from summary')
    active = all_rows[all_rows.split.isin(PARTITIONS)]
    for _, r in active.iterrows():
        expected = ('papila', '') if r['split'] == 'external_test' else (
            'chaksu', 'test' if r['split'] == 'in_domain_test' else 'train')
        if (r['dataset'], r['chaksu_release']) != expected:
            raise ValueError(f'Dataset/release crossed evaluation roles: {r.image_id}')
        if r.classification_eligible == '1':
            if r.binary_label not in ('0', '1'):
                raise ValueError(f'Nonbinary classification target: {r.image_id}')
            original = ' '.join(r.original_label.upper().split()).replace('SUSUPECT', 'SUSPECT')
            mapping = {'NORMAL': '0', 'GLAUCOMA SUSPECT': '1'} if r.dataset == 'chaksu' else {'0': '0', '1': '1'}
            if mapping.get(original) != r.binary_label:
                raise ValueError(f'Unexpected label mapping: {r.image_id}')
        if r.dataset == 'papila' and (not r.patient_id or r.patient_id_verified != '1'):
            raise ValueError('PAPILA requires verified patient identities')
    if summary.get('near_duplicate_reviews_pending', 0):
        raise ValueError('Resolve pending duplicate review before training')
    # Existing known identities must already be disjoint across active tiers.
    for field in ('patient_id', 'pixel_sha256', 'group_id'):
        subset = active[active[field] != '']
        if (subset.groupby(field)['split'].nunique() > 1).any():
            raise ValueError(f'Known {field} crosses splits; repair upstream before training')
    return root, all_rows, summary


def effective_partitions(all_rows):
    """Create a conservative derivative split without altering the original.

    Merge possible P-number identities with existing transitive duplicate groups.
    Quarantine entire components spanning source releases. If a possible identity
    spans train/validation, put its full development component into TRAIN.
    The deterministic policy uses no pixel interpretation or model performance.
    """
    df = all_rows.copy()
    df['original_split'] = df['split']
    df['possible_filename_group'] = [bosch_filename_key(r) for _, r in df.iterrows()]
    uf = Components(df.image_id.tolist())
    for field in ('group_id', 'patient_id', 'pixel_sha256', 'possible_filename_group'):
        first = {}
        for _, r in df.iterrows():
            key = r[field]
            if key:
                if key in first:
                    uf.join(first[key], r.image_id)
                else:
                    first[key] = r.image_id
    df['analysis_group'] = [uf.find(x) for x in df.image_id]
    df['split_change_reason'] = ''
    for _, indices in df.groupby('analysis_group').groups.items():
        group = df.loc[indices]
        datasets = set(group.dataset)
        releases = set(group.loc[group.dataset == 'chaksu', 'chaksu_release'])
        old = set(group.split)
        reason = None
        if any(s.startswith('quarantine') for s in old):
            destination, reason = 'quarantine_inherited', 'component includes previously quarantined image'
        elif len(datasets) > 1:
            destination, reason = 'quarantine_cross_dataset', 'possible shared identity across datasets'
        elif len(releases) > 1:
            destination, reason = 'quarantine_possible_patient_overlap', 'possible Bosch filename identity spans releases'
        elif old == {'train', 'validation'}:
            destination, reason = 'train', 'possible filename identity crosses development split; keep together in train'
        if reason:
            df.loc[indices, 'split'] = destination
            df.loc[indices, 'split_change_reason'] = reason
    active = df[df.split.isin(PARTITIONS)]
    if (active.groupby('analysis_group')['split'].nunique() > 1).any():
        raise ValueError('Effective groups still cross evaluation tiers')
    partitions = {}
    for p in PARTITIONS:
        rows = df[(df.split == p) & (df.classification_eligible == '1')].copy().reset_index(drop=True)
        if set(rows.binary_label) != {'0', '1'}:
            raise ValueError(f'{p}: precautionary grouping leaves fewer than two classes; revise protocol')
        rows['binary_label'] = rows.binary_label.astype(int)
        partitions[p] = rows
    return df, partitions


def image_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError('classification_image must be relative to artifacts/')
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Image path escapes artifacts/')
    if path.name != 'image_224.png' and path.name != 'image_classification.png':
        raise ValueError('Expected the pipeline RGB classification image, not a mask or overlay')
    return path


def check_images(root, rows):
    # Call on development only until final evaluation is explicitly enabled.
    for r in rows.itertuples():
        path = image_path(root, r.classification_image)
        if not path.is_file():
            raise FileNotFoundError(f'Missing prepared image: {path}')
        with Image.open(path) as im:
            if im.mode != 'RGB' or im.size != (224, 224):
                raise ValueError(f'Expected original pipeline RGB 224x224 export: {path}')
            im.load()


def make_transform(training=False, augment=True):
    operations = []
    if training and augment:
        # Full image retained; no random crops or anatomical orientation changes.
        operations.append(transforms.ColorJitter(brightness=.10, contrast=.10, saturation=.05))
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class FundusDataset(Dataset):
    def __init__(self, rows, root, training=False, augment=True):
        self.rows = rows.reset_index(drop=True).copy()
        self.root = Path(root)
        self.transform = make_transform(training, augment)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(image_path(self.root, row.classification_image)) as image:
            x = self.transform(image.convert('RGB'))
        return x, torch.tensor(float(row.binary_label), dtype=torch.float32), row.image_id


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # Warn if the chosen accelerator lacks deterministic support; record device.
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(request='auto'):
    if request != 'auto':
        return torch.device(request)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_loader(rows, root, config, training=False):
    # workers=0 works when Dataset is defined in a Windows/macOS Jupyter cell.
    generator = torch.Generator().manual_seed(config['seed'])
    return DataLoader(FundusDataset(rows, root, training, config['augment']),
                      batch_size=config['batch_size'], shuffle=training,
                      num_workers=0, drop_last=False, generator=generator)


def make_model(name, pretrained=True):
    builders = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    }
    if name not in builders:
        raise ValueError(f'Unknown architecture: {name}')
    builder, weights = builders[name]
    # Weight download failures must propagate: never silently train from scratch.
    model = builder(weights=weights if pretrained else None)
    if name.startswith('resnet'):
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model, str(weights) if pretrained else 'NONE_TEST_ONLY'


def head_parameters(model, name):
    return list(model.fc.parameters()) if name.startswith('resnet') else list(model.classifier.parameters())


def training_mode(model, head_only=False):
    if head_only:
        # Frozen backbone: freeze BatchNorm buffers and EfficientNet dropout too.
        model.eval()
    else:
        model.train()
    # Keep pretrained normalization statistics stable for small/imbalanced data.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def run_epoch(model, loader, device, optimizer=None, head_only=False):
    training = optimizer is not None
    if training:
        training_mode(model, head_only)
    else:
        model.eval()
    losses, targets, probs, ids = 0., [], [], []
    # Unweighted BCE estimates the original training-distribution posterior.
    # Class imbalance is addressed by ranking metrics and a validation threshold.
    criterion = nn.BCEWithLogitsLoss()
    with torch.set_grad_enabled(training):
        for x, y, image_ids in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x).reshape(-1)
            if logits.shape != y.shape:
                raise ValueError('Logit/target shape mismatch')
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
                optimizer.step()
            losses += loss.item() * len(y)
            targets.extend(y.detach().cpu().numpy().astype(int).tolist())
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            ids.extend(image_ids)
    if not targets:
        raise ValueError('Empty data loader')
    pred = pd.DataFrame({'image_id': ids, 'label': targets, 'probability': probs})
    return losses / len(targets), pred


def fit_model(train_rows, val_rows, root, out, config, device, pretrained=True):
    out = Path(out)
    if (out / 'best.pt').exists():
        raise FileExistsError('A checkpoint exists. Use a new run directory for a new experiment.')
    if config['epochs'] <= config['warmup_epochs']:
        raise ValueError('epochs must exceed warmup_epochs so the backbone is fine-tuned')
    seed_everything(config['seed'])
    model, weight_id = make_model(config['model_name'], pretrained)
    model.to(device)
    head = head_parameters(model, config['model_name'])
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    train_loader = make_loader(train_rows, root, config, training=True)
    val_loader = make_loader(val_rows, root, config)
    best_auc, best_loss, bad_epochs, history = -np.inf, np.inf, 0, []
    optimizer = None
    for epoch in range(config['epochs']):
        head_only = epoch < config['warmup_epochs']
        if epoch == 0 or epoch == config['warmup_epochs']:
            for p in backbone:
                p.requires_grad_(not head_only)
            groups = [{'params': head, 'lr': config['head_lr']}]
            if not head_only:
                groups.append({'params': backbone, 'lr': config['backbone_lr']})
            optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
            bad_epochs = 0
        started = time.monotonic()
        train_loss, _ = run_epoch(model, train_loader, device, optimizer, head_only)
        val_loss, predictions = run_epoch(model, val_loader, device)
        auc = float(roc_auc_score(predictions.label, predictions.probability))
        ap = float(average_precision_score(predictions.label, predictions.probability))
        entry = dict(epoch=epoch + 1, phase='head' if head_only else 'fine_tune',
                     train_bce=train_loss, validation_bce=val_loss,
                     validation_auroc=auc, validation_average_precision=ap,
                     seconds=time.monotonic() - started)
        history.append(entry)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        # Compare only fine-tuning checkpoints: selected model is end-to-end tuned.
        improved = not head_only and (auc > best_auc + 1e-8 or (abs(auc - best_auc) <= 1e-8 and val_loss < best_loss))
        if improved:
            best_auc, best_loss, bad_epochs = auc, val_loss, 0
            checkpoint = {'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          'config': dict(config), 'weights': weight_id, 'best_epoch': epoch + 1,
                          'validation_auroc': auc, 'validation_bce': val_loss}
            torch.save(checkpoint, out / 'best.pt')
        elif not head_only:
            bad_epochs += 1
        print(f"Epoch {epoch + 1:02d} [{entry['phase']}]: train BCE={train_loss:.4f}; "
              f"val BCE={val_loss:.4f}, AUROC={auc:.4f}, AP={ap:.4f}", flush=True)
        if not head_only and bad_epochs >= config['patience']:
            print('Early stopping on validation AUROC (BCE breaks ties).')
            break
    model, checkpoint = restore_model(out / 'best.pt', device)
    return model, checkpoint, pd.DataFrame(history)


def restore_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['config']['model_name'], pretrained=False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(device).eval()
    return model, checkpoint


def checked_arrays(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError('Expected nonempty 1D labels and probabilities of equal length')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected binary labels and finite probabilities in [0,1]')
    return y.astype(int), p


def select_threshold(y, p, target_sensitivity=.90):
    y, p = checked_arrays(y, p)
    if len(np.unique(y)) != 2 or not 0 < target_sensitivity <= 1:
        raise ValueError('Threshold selection needs both classes and target sensitivity in (0,1]')
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    good = np.flatnonzero(np.isfinite(thresholds) & (tpr >= target_sensitivity))
    # Maximize specificity subject to required validation sensitivity. Ties use
    # the largest threshold. This does not promise the same sensitivity on test.
    chosen = max(good, key=lambda i: (1 - fpr[i], thresholds[i]))
    return float(thresholds[chosen])


def metric_values(y, p, threshold=.5):
    y, p = checked_arrays(y, p)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be finite and in [0,1]')
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float('nan')
    sensitivity, specificity = div(tp, tp + fn), div(tn, tn + fp)
    both = len(np.unique(y)) == 2
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == k) * abs(y[bins == k].mean() - p[bins == k].mean())
              for k in range(10) if (bins == k).any())
    return dict(n=len(y), positives=int(y.sum()), negatives=int((1-y).sum()),
                prevalence=float(y.mean()), threshold=threshold,
                auroc=float(roc_auc_score(y, p)) if both else float('nan'),
                average_precision=float(average_precision_score(y, p)) if both else float('nan'),
                sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=(sensitivity + specificity) / 2,
                accuracy=div(tp + tn, len(y)), precision=div(tp, tp + fp),
                npv=div(tn, tn + fn), f1=div(2 * tp, 2 * tp + fp + fn),
                brier=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])),
                ece_10_bins=float(ece), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def attach_metadata(predictions, rows):
    if predictions.image_id.duplicated().any() or set(predictions.image_id) != set(rows.image_id):
        raise ValueError('Prediction identity mismatch or duplication')
    keep = ['image_id', 'binary_label', 'dataset', 'camera', 'patient_id', 'analysis_group', 'classification_image']
    result = predictions.merge(rows[keep], on='image_id', how='left', validate='one_to_one')
    if not np.array_equal(result.label.to_numpy(), result.binary_label.astype(int).to_numpy()):
        raise ValueError('Prediction label alignment failed')
    return result


def grouped_bootstrap(predictions, threshold, seed=2440, iterations=1000):
    """Percentile CIs conditional on a fixed trained model and threshold.

    Resample entire analysis components (patient + duplicate + precautionary ID).
    CHAKSU unknown identities remain unresolved, so its CIs are provisional.
    """
    rng = np.random.default_rng(seed)
    groups = [np.asarray(v) for v in predictions.reset_index(drop=True).groupby('analysis_group').indices.values()]
    if len(groups) < 2:
        raise ValueError('At least two independent groups are needed for a bootstrap')
    y, p = checked_arrays(predictions.label, predictions.probability)
    keys = ['auroc', 'average_precision', 'sensitivity', 'specificity', 'balanced_accuracy', 'brier']
    samples = {key: [] for key in keys}
    for _ in range(iterations):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        values = metric_values(y[indices], p[indices], threshold)
        for key in keys:
            if np.isfinite(values[key]):
                samples[key].append(values[key])
    point = metric_values(y, p, threshold)
    records = []
    for key in keys:
        values = samples[key]
        lo, hi = np.quantile(values, [.025, .975]) if len(values) >= 2 else (np.nan, np.nan)
        records.append(dict(metric=key, estimate=point[key], lower_95=lo, upper_95=hi,
                            valid_resamples=len(values), requested_resamples=iterations,
                            bootstrap_groups=len(groups)))
    return pd.DataFrame(records)


def evaluation_tables(predictions, threshold):
    overall = pd.DataFrame([dict(operating_point='default_0.5', **metric_values(predictions.label, predictions.probability, .5)),
                            dict(operating_point='validation_selected', **metric_values(predictions.label, predictions.probability, threshold))])
    cameras = []
    for camera, group in predictions.groupby('camera'):
        values = metric_values(group.label, group.probability, threshold)
        cameras.append(dict(camera=camera, small_class_count=min(values['positives'], values['negatives']) < 10, **values))
    return overall, pd.DataFrame(cameras)


def diagnostic_plot(predictions, threshold, title, destination):
    import matplotlib.pyplot as plt
    y, p = checked_arrays(predictions.label, predictions.probability)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, label=f'AUROC {roc_auc_score(y, p):.3f}')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlabel='False positive rate', ylabel='Sensitivity', title='ROC')
    axes[0].legend()
    precision, recall, _ = precision_recall_curve(y, p)
    axes[1].plot(recall, precision, label=f'AP {average_precision_score(y, p):.3f}')
    axes[1].axhline(y.mean(), linestyle='--', color='gray', label='Prevalence')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall')
    axes[1].legend()
    bins = np.minimum((p * 5).astype(int), 4)
    xs, ys, counts = [], [], []
    for b in range(5):
        mask = bins == b
        if mask.any():
            xs.append(p[mask].mean()); ys.append(y[mask].mean()); counts.append(mask.sum())
    axes[2].plot([0, 1], [0, 1], '--', color='gray')
    axes[2].scatter(xs, ys, s=np.asarray(counts) * 2 + 15)
    axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted score', ylabel='Observed positive fraction',
                title='Reliability (marker size = bin count)')
    fig.suptitle(f'{title}; fixed threshold = {threshold:.4f}')
    fig.tight_layout(); fig.savefig(destination, dpi=150, bbox_inches='tight')
    return fig


def error_gallery(predictions, root, threshold, destination, max_each=3):
    import matplotlib.pyplot as plt
    fn = predictions[(predictions.label == 1) & (predictions.probability < threshold)].sort_values('probability').head(max_each)
    fp = predictions[(predictions.label == 0) & (predictions.probability >= threshold)].sort_values('probability', ascending=False).head(max_each)
    examples = pd.concat([fn.assign(error='FN'), fp.assign(error='FP')])
    if examples.empty:
        print('No errors at this operating point.'); return None
    fig, axes = plt.subplots(1, len(examples), figsize=(3.3 * len(examples), 3.5), squeeze=False)
    for ax, row in zip(axes[0], examples.itertuples()):
        with Image.open(image_path(root, row.classification_image)) as im:
            ax.imshow(im.copy())
        ax.set_title(f'{row.error}: y={row.label}, p={row.probability:.3f}\n{row.camera}', fontsize=10)
        ax.set_xlabel(row.image_id, fontsize=6); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(destination, dpi=150, bbox_inches='tight')
    return fig


def letterbox_photo(path):
    # Exactly match prepare_data.padded_resize for new, unprepared images.
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        side = max(im.size)
        canvas = Image.new('RGB', (side, side), 0)
        canvas.paste(im, ((side-im.width)//2, (side-im.height)//2))
        return canvas.resize((224, 224), Image.Resampling.BILINEAR)


def freeze_run(out, threshold, data_hash, checkpoint):
    out = Path(out)
    path = out / 'frozen_protocol.json'
    record = dict(checkpoint_sha256=sha256_file(out / 'best.pt'),
                  effective_manifest_sha256=data_hash,
                  threshold=float(threshold), threshold_source='CHAKSU validation only',
                  target_validation_sensitivity=checkpoint['config']['target_sensitivity'],
                  model_name=checkpoint['config']['model_name'], weights=checkpoint['weights'],
                  best_epoch=checkpoint['best_epoch'],
                  label_definition='CHAKSU normal vs glaucoma suspect; PAPILA healthy vs clinical glaucoma',
                  probability_note='Uncalibrated sigmoid score; not a validated clinical disease probability')
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError('Frozen run changed. Use a new development run; never tune on test.')
    else:
        write_json(path, record)
    return record


def verify_frozen(out):
    out = Path(out)
    record = json.loads((out / 'frozen_protocol.json').read_text())
    if sha256_file(out / 'best.pt') != record['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after freezing')
    if sha256_file(out / 'effective_splits/all.csv') != record['effective_manifest_sha256']:
        raise ValueError('Effective manifest changed after freezing')
    return record


def final_evaluation(out, root, device, iterations=1000):
    """Load only the frozen checkpoint, threshold, and saved derivative splits."""
    out = Path(out)
    protocol = verify_frozen(out)
    # Integrity bind per-partition files to the hashed effective manifest.
    all_rows = read_table(out / 'effective_splits/all.csv')
    model, checkpoint = restore_model(out / 'best.pt', device)
    config = checkpoint['config']
    if checkpoint['weights'] == 'NONE_TEST_ONLY':
        warnings.warn('Synthetic software test only: no pretrained weights.')
    final_dir = out / 'final_evaluation'
    final_dir.mkdir(exist_ok=False)
    # If interrupted, inspect this folder; do not silently overwrite test results.
    results = []
    for partition in ('in_domain_test', 'external_test'):
        rows = all_rows[(all_rows.split == partition) & (all_rows.classification_eligible == '1')].copy()
        rows['binary_label'] = rows.binary_label.astype(int)
        check_images(root, rows)
        loader = make_loader(rows, root, config)
        _, pred = run_epoch(model, loader, device)
        pred = attach_metadata(pred, rows)
        pred.to_csv(final_dir / f'{partition}_predictions.csv', index=False)
        overall, cameras = evaluation_tables(pred, protocol['threshold'])
        overall.insert(0, 'partition', partition)
        overall.to_csv(final_dir / f'{partition}_metrics.csv', index=False)
        cameras.to_csv(final_dir / f'{partition}_camera_metrics.csv', index=False)
        grouped_bootstrap(pred, protocol['threshold'], config['seed'], iterations).to_csv(
            final_dir / f'{partition}_confidence_intervals.csv', index=False)
        fig = diagnostic_plot(pred, protocol['threshold'], partition, final_dir / f'{partition}_curves.png')
        import matplotlib.pyplot as plt
        plt.close(fig)
        fig = error_gallery(pred, root, protocol['threshold'], final_dir / f'{partition}_errors.png')
        if fig is not None:
            plt.close(fig)
        results.append(overall)
    result = pd.concat(results, ignore_index=True)
    result.to_csv(final_dir / 'comparison.csv', index=False)
    write_json(final_dir / 'completion.json', dict(protocol=protocol, complete=True))
    return result
