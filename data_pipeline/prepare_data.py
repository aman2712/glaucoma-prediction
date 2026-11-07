"""
Pipeline: inventory -> prepare -> split. See README.md before interpreting outputs.

Three evaluation tiers are produced, in increasing order of how "unseen" they are:
  train / validation   - from the CHAKSU *train* release only.
  in_domain_test        - the CHAKSU *test* release, held out untouched (same dataset,
                           same population, official publisher split).
  external_test         - PAPILA, a fully separate dataset/site/camera, touched only
                           after the model and thresholds are frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openpyxl
from PIL import Image, ImageDraw, ImageFilter

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def config_load(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for key in ('chaksu_train_root', 'chaksu_test_root', 'papila_root'):
        config[key] = str((path.parent / config[key]).resolve())
    if config['chaksu_mask_reference'] not in ('STAPLE', 'Majority', 'Median'):
        raise ValueError('Unsupported CHAKSU mask reference')
    if config['papila_reference_expert'] not in (1, 2):
        raise ValueError('PAPILA reference expert must be 1 or 2')
    return config


def dataset_root(config, row):
    if row['dataset'] == 'papila':
        return Path(config['papila_root'])
    return Path(config[f"chaksu_{row['chaksu_release']}_root"])


def images(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if not p.name.startswith('.') and p.suffix.lower() in IMAGE_EXTENSIONS)


def unique_stems(paths):
    result = {}
    for p in paths:
        key = p.stem.lower()
        if key in result:
            raise ValueError(f'Duplicate filename stem: {p.parent}/{p.stem}')
        result[key] = p
    return result


def chaksu_label(value):
    value = ' '.join(value.upper().split()).replace('SUSUPECT', 'SUSPECT')
    if value not in ('NORMAL', 'GLAUCOMA SUSPECT'):
        raise ValueError(f'Unrecognized CHAKSU label: {value!r}')
    return int(value == 'GLAUCOMA SUSPECT')


def label_stem(value):
    #released table uses Image101.jpg-Image101-1.jpg, not a plain filename
    return re.split(r'\.(?:jpg|jpeg|png|tif|tiff)(?:-|$)', value.strip(),
                    maxsplit=1, flags=re.I)[0].lower()


def base_row(dataset, stem, camera, image_path, release=''):
    image_id = f'{dataset}_{camera}_{stem}' if not release else f'{dataset}_{camera}_{release}_{stem}'
    source_release = f'chaksu_{release}' if dataset == 'chaksu' else 'papila'
    return dict(image_id=image_id, dataset=dataset,
                patient_id='', patient_id_verified=0, eye='', camera=camera,
                chaksu_release=release, source_release=source_release,
                image_path=str(image_path), original_label='', binary_label='',
                classification_eligible=1, exclusion_reason='', label_source='',
                suspect_votes='', disc_reference='', cup_reference='',
                alternate_disc_reference='', alternate_cup_reference='',
                annotation_reference='')


def inventory_chaksu(root, release, mask_reference):
    """Inventory one CHAKSU release (train or test) folder tree."""
    rows, audit = [], {}
    for camera in ('Bosch', 'Forus', 'Remidio'):
        folder = root / '1.0_Original_Fundus_Images' / camera
        found = unique_stems(images(folder))
        labels_file = root / '6.0_Glaucoma_Decision' / f'Glaucoma_Decision_Comparison_{camera}_majority.csv'
        labels = {}
        for entry in read_csv(labels_file):
            key = label_stem(entry['Images'])
            if key in labels:
                raise ValueError(f'Duplicate label key in {labels_file.name}')
            votes = [chaksu_label(entry[f'Expert.{i}']) for i in range(1, 6)]
            majority = int(sum(votes) >= 3)
            # The released label tables are inconsistent across camera/release: most use
            # majority decision, but some (observed on CHAKSU test-release Forus/Remidio)
            # use glaucoma decision instead. Accept either, but cross-check it
            # against the five expert votes.
            for column in ('Majority Decision', 'Glaucoma Decision'):
                if column in entry:
                    supplied_column = column
                    break
            else:
                raise ValueError(f'No majority/glaucoma decision column in {labels_file.name}')
            if majority != chaksu_label(entry[supplied_column]):
                raise ValueError(f'Supplied {supplied_column!r} disagrees with five expert votes '
                                 f'in {labels_file.name}')
            labels[key] = (majority, sum(votes))
        reference = root / '5.0_OD_OC_Mean_Median_Majority_STAPLE' / camera
        disc = unique_stems(images(reference / 'Disc' / mask_reference))
        cup = unique_stems(images(reference / 'Cup' / mask_reference))
        for key, path in found.items():
            if key not in labels or key not in disc or key not in cup:
                raise ValueError(f'Missing label/mask for CHAKSU {release} {camera} {path.stem}')
            label, votes = labels[key]
            row = base_row('chaksu', path.stem, camera, path.relative_to(root), release=release)
            row.update(original_label='GLAUCOMA SUSPECT' if label else 'NORMAL',
                       binary_label=label, suspect_votes=votes,
                       label_source=str(labels_file.relative_to(root)),
                       disc_reference=str(disc[key].relative_to(root)),
                       cup_reference=str(cup[key].relative_to(root)),
                       annotation_reference=mask_reference)
            rows.append(row)
        audit[camera] = dict(images=len(found), label_rows=len(labels),
                             label_rows_without_release_image=len(set(labels) - set(found)))
    return rows, audit


def inventory(config, out):
    if (out / 'inventory.csv').exists():
        raise ValueError('Inventory already exists, use a new output directory for a revised run.')
    rows = []
    chaksu_audit = {}
    for release in ('train', 'test'):
        release_rows, release_audit = inventory_chaksu(
            Path(config[f'chaksu_{release}_root']), release, config['chaksu_mask_reference'])
        rows.extend(release_rows)
        chaksu_audit[release] = release_audit

    root = Path(config['papila_root'])
    known = {}
    for eye in ('OD', 'OS'):
        source = root / 'ClinicalData' / f'patient_data_{eye.lower()}.xlsx'
        book = openpyxl.load_workbook(source, read_only=True, data_only=True)
        for values in book.active.values:
            identifier = str(values[0])
            if not re.fullmatch(r'#\d+', identifier):
                continue  #three header rows
            patient = int(identifier[1:])
            raw = values[3]
            if raw not in (0, 1, 2):
                raise ValueError('Unexpected PAPILA diagnosis code')
            key = f'RET{patient:03d}{eye}'
            if key in known:
                raise ValueError('Duplicate PAPILA diagnosis')
            known[key] = (patient, eye, int(raw), str(source.relative_to(root)))
        book.close()
    expert = config['papila_reference_expert']
    for path in images(root / 'FundusImages'):
        patient, eye, raw, source = known.pop(path.stem)
        row = base_row('papila', path.stem, 'Topcon', path.relative_to(root))
        row.update(patient_id=f'papila_{patient:03d}', patient_id_verified=1, eye=eye,
                   original_label=raw, binary_label=raw if raw in (0, 1) else '',
                   classification_eligible=int(raw in (0, 1)),
                   exclusion_reason='suspicious_label' if raw == 2 else '',
                   label_source=source, annotation_reference=f'expert_{expert}')
        for kind in ('disc', 'cup'):
            for alternate, who in ((False, expert), (True, 3 - expert)):
                p = Path('ExpertsSegmentations/Contours') / f'{path.stem}_{kind}_exp{who}.txt'
                if not (root / p).exists():
                    raise ValueError(f'Missing PAPILA contour: {p.name}')
                row[('alternate_' if alternate else '') + kind + '_reference'] = str(p)
        rows.append(row)
    if known:
        raise ValueError('PAPILA clinical records without image, inspect source version')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Non-unique image IDs')

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'inventory.csv', rows)
    write_json(out / 'config.resolved.json', config)
    write_json(out / 'inventory_summary.json', {
        'counts': dict(Counter(f"{r['source_release']}/{r['camera']}/label={r['original_label']}" for r in rows)),
        'images': len(rows),
        'chaksu_source_join': chaksu_audit,
        'papila_patients': len({r['patient_id'] for r in rows if r['dataset'] == 'papila'}),
        'chaksu_patient_linkage': ('Unknown. No patient identity inferred from filenames. Provide '
                                    '--patient-groups before splitting. Without it, both the CHAKSU '
                                    'train/validation split and the CHAKSU in-domain test are only '
                                    'provisional (image-level, not patient-level).'),
        'status': 'Inventory only; image/mask geometry and duplication not yet audited.'})
    n_train = sum(r['dataset'] == 'chaksu' and r['chaksu_release'] == 'train' for r in rows)
    n_test = sum(r['dataset'] == 'chaksu' and r['chaksu_release'] == 'test' for r in rows)
    n_papila = sum(r['dataset'] == 'papila' for r in rows)
    print(f'Inventoried {len(rows)} images ({n_train} CHAKSU-train, {n_test} CHAKSU-test, '
          f'{n_papila} PAPILA). PAPILA reserved external, CHAKSU-test reserved as in-domain held-out.',
          flush=True)


def binary_mask(path, size, threshold=128):
    with Image.open(path) as im:
        if im.size != size:
            raise ValueError('Reference mask and photograph have different dimensions')
        # Alpha is opacity, not foreground. CHAKSU STAPLE PNGs may be RGBA.
        gray = np.array(im.convert('L'))
    return gray >= threshold


def contour_mask(path, size):
    points = np.loadtxt(path)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('Invalid contour coordinate array')
    width, height = size
    if (points[:, 0] < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] < 0).any() or (points[:, 1] >= height).any():
        raise ValueError('Contour outside original image coordinates')
    #Publisher helper uses x=column, y=row, with no 1-based offset.
    im = Image.new('L', size, 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in points], fill=255)
    return np.array(im) > 0


def measurements(disc, cup):
    if disc.shape != cup.shape or not disc.any() or not cup.any():
        raise ValueError('Empty or shape-mismatched masks')
    if np.any(cup & ~disc):
        raise ValueError('Cup extends outside disc, review rather than silently clip')
    dy, dx = np.where(disc)
    cy, cx = np.where(cup)
    da, ca = int(disc.sum()), int(cup.sum())
    return dict(vcdr=float(np.ptp(cy) + 1) / float(np.ptp(dy) + 1),
                hcdr=float(np.ptp(cx) + 1) / float(np.ptp(dx) + 1), acdr=ca / da,
                disc_area_px=da, cup_area_px=ca, rim_area_px=da - ca,
                rim_to_disc_area=(da - ca) / da)


def padded_resize(im, size, is_mask=False):
    width, height = im.size
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    canvas = Image.new(im.mode, (side, side), 0)
    canvas.paste(im, (left, top))
    result = canvas.resize((size, size), Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR)
    return result, dict(original_width=width, original_height=height,
                        pad_left=left, pad_top=top, padded_side=side, output_size=size,
                        scale=size / side)


def prepare(config, out, limit=0):
    dest = out / ('smoke' if limit else 'prepared')
    if dest.exists():
        raise ValueError(f'{dest.name} exists, use a fresh output directory for a new run.')
    source = read_csv(out / 'inventory.csv')
    if limit:
        buckets = defaultdict(list)
        for row in source:
            buckets[(row['dataset'], row['camera'])].append(row)
        source = [r for i in range(limit) for bucket in buckets.values() for r in bucket[i:i+1]][:limit]
    dest.mkdir(parents=True)
    result, features, qc, previews = [], [], [], Counter()
    for number, row in enumerate(source, 1):
        root = dataset_root(config, row)
        path = root / row['image_path']
        with Image.open(path) as opened:
            if opened.getexif().get(274, 1) != 1:
                raise ValueError('Nontrivial EXIF orientation: annotation alignment needs explicit handling')
            im = opened.convert('RGB')
        raw = np.asarray(im)
        row.update(width=im.width, height=im.height, sha256=sha_file(path),
                   pixel_sha256=hashlib.sha256(str(im.size).encode() + raw.tobytes()).hexdigest())
        # dHash is a candidate generator only, visual review must establish duplication
        small = np.asarray(im.convert('L').resize((9, 8)))
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        row['dhash'] = f'{sum(int(b) << i for i, b in enumerate(bits)):016x}'
        loader = contour_mask if row['dataset'] == 'papila' else binary_mask
        masks = {}
        for kind in ('disc', 'cup'):
            kwargs = {} if row['dataset'] == 'papila' else {'threshold': config['mask_threshold']}
            masks[kind] = loader(root / row[kind + '_reference'], im.size, **kwargs)
        disc, cup = masks['disc'], masks['cup']
        outside = int((cup & ~disc).sum())
        valid = bool(disc.any() and cup.any() and outside == 0)
        row.update(segmentation_eligible=int(valid), cup_outside_disc_pixels=outside)
        feature = dict(image_id=row['image_id'], dataset=row['dataset'],
                       feature_source='expert_reference', reference=row['annotation_reference'],
                       valid=int(valid), vcdr='', hcdr='', acdr='', disc_area_px='',
                       cup_area_px='', rim_area_px='', rim_to_disc_area='')
        if valid:
            feature.update(measurements(disc, cup))
        features.append(feature)
        item = dest / row['dataset'] / row['image_id']
        item.mkdir(parents=True)
        clf, geometry = padded_resize(im, config['classification_size'])
        cls_name = 'image_224.png' if config['classification_size'] == 224 else 'image_classification.png'
        clf.save(item / cls_name)
        row['classification_image'] = str((item / cls_name).relative_to(out))
        seg, seg_geometry = padded_resize(im, config['segmentation_size'])
        seg.save(item / 'image_segmentation.png')
        row['segmentation_image'] = str((item / 'image_segmentation.png').relative_to(out))
        for kind, mask in masks.items():
            native = Image.fromarray(mask.astype('uint8') * 255)
            native.save(item / f'{kind}_native.png')
            resized, _ = padded_resize(native, config['segmentation_size'], True)
            resized.save(item / f'{kind}_segmentation.png')
            row[kind + '_mask'] = str((item / f'{kind}_segmentation.png').relative_to(out))
            row[kind + '_native_mask'] = str((item / f'{kind}_native.png').relative_to(out))
        write_json(item / 'geometry.json', {'classification': geometry, 'segmentation': seg_geometry})
        domain = (row['dataset'], row['camera'])
        preview_path = ''
        # Local-only previews, no uploading or model inspection of fundus photographs
        if previews[domain] < 3 or not valid:
            overlay = seg.copy()
            for kind, color in [('disc', (0, 255, 0)), ('cup', (255, 70, 70))]:
                mask = Image.open(item / f'{kind}_segmentation.png').convert('L')
                edge = mask.filter(ImageFilter.MaxFilter(3))
                edge = Image.fromarray((np.array(edge) != np.array(mask)).astype('uint8') * 255)
                overlay.paste(color, mask=edge)
            overlay.save(item / 'overlay_review.png')
            preview_path = str((item / 'overlay_review.png').relative_to(out))
            previews[domain] += 1
        qc.append(dict(image_id=row['image_id'], dataset=row['dataset'], camera=row['camera'],
                       cup_outside_disc_pixels=outside, empty_disc=int(not disc.any()),
                       empty_cup=int(not cup.any()), overlay=preview_path,
                       reviewer='', status='pending', comment=''))
        result.append(row)
        if number % 50 == 0:
            print(f'Prepared {number}/{len(source)}', flush=True)
    prefix = 'smoke_' if limit else ''
    write_csv(out / f'{prefix}prepared_manifest.csv', result)
    write_csv(out / f'{prefix}expert_measurements.csv', features)
    write_csv(out / f'{prefix}quality_review.csv', qc)
    groups = defaultdict(list)
    for r in result:
        groups[r['pixel_sha256']].append(r['image_id'])
    duplicate_sets = [v for v in groups.values() if len(v) > 1]
    candidates = []
    for i, a in enumerate(result):
        for b in result[i + 1:]:
            distance = (int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count()
            if distance <= 4 and a['pixel_sha256'] != b['pixel_sha256']:
                candidates.append(dict(image_id_a=a['image_id'], image_id_b=b['image_id'],
                                       dhash_distance=distance, decision='pending', reviewer=''))
    write_csv(out / f'{prefix}near_duplicate_review.csv', candidates,
              ['image_id_a', 'image_id_b', 'dhash_distance', 'decision', 'reviewer'])
    write_json(out / f'{prefix}preparation_summary.json', {
        'images': len(result), 'invalid_reference_geometry': sum(int(r['segmentation_eligible']) == 0 for r in result),
        'exact_pixel_duplicate_sets': duplicate_sets, 'near_duplicate_candidates': len(candidates),
        'smoke_only': bool(limit), 'raw_files_modified': False,
        'expert_features_for_oracle_only': True, 'visual_review': 'Pending human review',
        'python': platform.python_version(), 'numpy': np.__version__,
        'config_sha256': sha_file(out / 'config.resolved.json'),
        'script_sha256': sha_file(__file__)})
    print(f'Prepared {len(result)} images, invalid masks flagged, not repaired. Human QC pending.', flush=True)


class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, key):
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def join(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def grouped_validation(groups, fraction, seed):
    if len(groups) < 2:
        raise ValueError('Insufficient independent development groups')
    total = Counter((r['camera'], str(r['binary_label'])) for g in groups for r in g)
    best, best_score = None, float('inf')
    rng = random.Random(seed)
    # Choose by balance
    for _ in range(500):
        order = list(range(len(groups)))
        rng.shuffle(order)
        selected, n = [], 0
        for i in order:
            if n >= sum(total.values()) * fraction:
                break
            selected.append(i)
            n += len(groups[i])
        counts = Counter((r['camera'], str(r['binary_label'])) for i in selected for r in groups[i])
        if any(sum(v for (c, y), v in counts.items() if y == label) == 0 or
               sum(v for (c, y), v in total.items() if y == label) == sum(v for (c, y), v in counts.items() if y == label)
               for label in ('0', '1')):
            continue
        score = sum(abs(counts[k] / v - fraction) for k, v in total.items())
        if score < best_score:
            best, best_score = selected, score
    if best is None:
        raise ValueError('Cannot make train/validation with both labels, inspect grouping')
    return set(best)


def split(config, out, patient_file=None, allow_provisional=False, limitation_note=''):
    dest = out / 'splits'
    if dest.exists():
        raise ValueError('Frozen splits exist. Use a new output run to change the protocol.')
    rows = read_csv(out / 'prepared_manifest.csv')
    known_ids = {r['image_id'] for r in rows}
    if patient_file:
        seen = set()
        by_id = {r['image_id']: r for r in rows}
        for entry in read_csv(patient_file):
            key, patient = entry['image_id'], entry['patient_id'].strip()
            if key not in known_ids or key in seen or not patient:
                raise ValueError('Unknown, repeated, or empty patient mapping')
            if by_id[key]['dataset'] != 'chaksu':
                raise ValueError('PAPILA IDs already verified, overrides only support CHAKSU')
            seen.add(key)
            by_id[key].update(patient_id='chaksu_' + patient, patient_id_verified='1')
    missing = sum(r['dataset'] == 'chaksu' and r['patient_id_verified'] != '1' for r in rows)
    reviews = read_csv(out / 'near_duplicate_review.csv')
    if any(r['decision'] not in ('pending', 'duplicate', 'not_duplicate') for r in reviews):
        raise ValueError('Near duplicate decision must be pending, duplicate, or not_duplicate')
    pending = sum(r['decision'] == 'pending' for r in reviews)
    if (missing or pending) and not allow_provisional:
        raise ValueError(f'{missing} CHAKSU images lack verified patient groups; {pending} duplicate reviews pending. '
                         'Complete linkage/review, or use --allow-provisional together with --limitation-note '
                         'and report this limitation explicitly.')
    if (missing or pending) and not limitation_note.strip():
        raise ValueError('Provisional split requires --limitation-note describing precisely what is unverified '
                          'and why proceeding is acceptable for this milestone.')

    uf = UnionFind(known_ids)
    for field in ('patient_id', 'pixel_sha256'):
        first = {}
        for row in rows:
            value = row[field]
            if not value:
                continue
            if value in first:
                uf.join(first[value], row['image_id'])
            else:
                first[value] = row['image_id']
    for row in reviews:
        if row['decision'] == 'duplicate':
            uf.join(row['image_id_a'], row['image_id_b'])
    components = defaultdict(list)
    for row in rows:
        components[uf.find(row['image_id'])].append(row)

    development = []
    for group in components.values():
        datasets = {r['dataset'] for r in group}
        releases = {r['chaksu_release'] for r in group if r['dataset'] == 'chaksu'}
        group_id = min(r['image_id'] for r in group)
        for row in group:
            row['group_id'] = group_id
            row['split'] = ''
        if 'papila' in datasets:
            # A PAPILA duplicate must never let a CHAKSU twin enter development or the in-domain test.
            for row in group:
                row['split'] = 'external_test' if row['dataset'] == 'papila' else 'quarantine_external_overlap'
        elif len(releases) > 1:
            # Same identity/pixels appear in both CHAKSU releases: exclude from both tiers rather
            # than let it artificially link "development" and "held-out" performance.
            for row in group:
                row['split'] = 'quarantine_release_overlap'
        elif releases == {'train'}:
            development.append(group)
        elif releases == {'test'}:
            for row in group:
                row['split'] = 'in_domain_test'
        else:
            raise ValueError('Unexpected group composition, inspect grouping')

    development.sort(key=lambda g: g[0]['group_id'])
    validation = grouped_validation(development, config['validation_fraction'], config['seed'])
    for i, group in enumerate(development):
        for row in group:
            row['split'] = 'validation' if i in validation else 'train'

    if any(r['split'] == '' for r in rows):
        raise ValueError('Unassigned split found, inspect grouping logic')
    assert all(r['split'] == 'external_test' for r in rows if r['dataset'] == 'papila')
    assert all(r['split'] in ('train', 'validation') for r in rows
               if r['dataset'] == 'chaksu' and r['chaksu_release'] == 'train'
               and r['split'] not in ('quarantine_external_overlap', 'quarantine_release_overlap'))
    assert all(r['split'] == 'in_domain_test' for r in rows
               if r['dataset'] == 'chaksu' and r['chaksu_release'] == 'test'
               and r['split'] not in ('quarantine_external_overlap', 'quarantine_release_overlap'))

    indomain_labels = {str(r['binary_label']) for r in rows
                        if r['split'] == 'in_domain_test' and r['classification_eligible'] == '1'}
    if indomain_labels and len(indomain_labels) < 2:
        raise ValueError('In-domain (CHAKSU test-release) set does not contain both classes; inspect labels')

    dest.mkdir()
    write_csv(dest / 'all.csv', rows)
    for name in ('train', 'validation', 'in_domain_test', 'external_test'):
        selected = [r for r in rows if r['split'] == name]
        write_csv(dest / f'{name}_all_tasks.csv', selected, list(rows[0]))
        write_csv(dest / f'{name}_classification.csv', [r for r in selected if r['classification_eligible'] == '1'], list(rows[0]))
        write_csv(dest / f'{name}_segmentation.csv', [r for r in selected if r['segmentation_eligible'] == '1'], list(rows[0]))
    write_json(dest / 'split_summary.json', {
        'status': 'PROVISIONAL' if missing or pending else 'GROUPED_IDENTITIES_PROVIDED',
        'limitation_note': limitation_note.strip() if (missing or pending) else '',
        'unknown_chaksu_patient_images': missing, 'near_duplicate_reviews_pending': pending,
        'seed': config['seed'], 'validation_fraction_target': config['validation_fraction'],
        'counts': dict(Counter(f"{r['split']}/{r['dataset']}/{r['camera']}/label={r['binary_label']}" for r in rows)),
        'classification_counts': dict(Counter(r['split'] for r in rows if r['classification_eligible'] == '1')),
        'segmentation_counts': dict(Counter(r['split'] for r in rows if r['segmentation_eligible'] == '1')),
        'quarantined_external_overlap': sum(r['split'] == 'quarantine_external_overlap' for r in rows),
        'quarantined_release_overlap': sum(r['split'] == 'quarantine_release_overlap' for r in rows),
        'manifest_sha256': sha_file(out / 'prepared_manifest.csv'),
        'all_splits_sha256': sha_file(dest / 'all.csv'),
        'patient_mapping_sha256': sha_file(patient_file) if patient_file else None,
        'duplicate_review_sha256': sha_file(out / 'near_duplicate_review.csv'),
        'interpretation': ('Known groups and reviewed duplicates held together. CHAKSU train-release is '
                            'split into train/validation; CHAKSU test-release is held out untouched as an '
                            'in-domain test; PAPILA is fully external. Unknown patient linkage prevents a '
                            'strict patient-disjoint claim until resolved. Human quality review still required.')})
    print('Saved fixed splits (train/validation/in_domain_test/external_test); see split_summary.json.',
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'prepare', 'split'])
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--limit', type=int, default=0, help='Smoke preparation only, cannot be split')
    parser.add_argument('--patient-groups', type=Path)
    parser.add_argument('--allow-provisional', action='store_true')
    parser.add_argument('--limitation-note', type=str, default='',
                         help='Required together with --allow-provisional: state precisely what is '
                              'unverified and why proceeding is acceptable at this milestone.')
    args = parser.parse_args()
    config = config_load(args.config)
    out = args.out.resolve()
    if args.command != 'inventory':
        if json.loads((out / 'config.resolved.json').read_text()) != config:
            parser.error('Configuration changed since inventory, start a new output run.')
    if args.command == 'inventory':
        inventory(config, out)
    elif args.command == 'prepare':
        prepare(config, out, args.limit)
    else:
        split(config, out, args.patient_groups, args.allow_provisional, args.limitation_note)


if __name__ == '__main__':
    main()
