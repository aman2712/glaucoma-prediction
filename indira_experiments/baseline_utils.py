"""Frozen baseline utility subset from commit 4d02f19; no training or test evaluator.

Retained within this experimental branch so the existing common project modules
remain unchanged. Only development datasets are passed to these utilities.
"""
from pathlib import Path
import hashlib
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, brier_score_loss, log_loss, precision_recall_curve
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

