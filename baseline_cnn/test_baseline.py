"""Software verification with synthetic images; never a clinical experiment."""
from pathlib import Path
import contextlib
import copy
import csv
import gc
import hashlib
import json
import tempfile
import unittest

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import torch

import baseline_core as b

HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)


def fixture(root):
    root = Path(root)
    (root / 'splits').mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(2440)
    sizes = {'train': 16, 'validation': 12, 'in_domain_test': 8, 'external_test': 8}
    for partition, count in sizes.items():
        dataset = 'papila' if partition == 'external_test' else 'chaksu'
        release = '' if dataset == 'papila' else ('test' if partition == 'in_domain_test' else 'train')
        for i in range(count):
            label = i % 2
            image_id = f'{dataset}_{partition}_{i:03d}'
            relative = f'prepared/{dataset}/{image_id}/image_224.png'
            path = root / relative
            path.parent.mkdir(parents=True)
            pixels = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            rows.append(dict(image_id=image_id, dataset=dataset, chaksu_release=release,
                             camera='Topcon' if dataset == 'papila' else 'Remidio',
                             patient_id=f'papila_{i // 2:03d}' if dataset == 'papila' else '',
                             patient_id_verified='1' if dataset == 'papila' else '0',
                             pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                             group_id=f'papila_{i // 2:03d}' if dataset == 'papila' else image_id,
                             split=partition, classification_eligible='1', segmentation_eligible='0',
                             binary_label=str(label), original_label=str(label) if dataset == 'papila' else (
                                 'GLAUCOMA SUSPECT' if label else 'NORMAL'), classification_image=relative))
    df = pd.DataFrame(rows)
    write_fixture(root, df)
    return df


def write_fixture(root, df):
    df.to_csv(root / 'prepared_manifest.csv', index=False)
    df.to_csv(root / 'splits/all.csv', index=False)
    for p in b.PARTITIONS:
        df[(df.split == p) & (df.classification_eligible == '1')].to_csv(
            root / 'splits' / f'{p}_classification.csv', index=False)
    b.write_json(root / 'splits/split_summary.json', dict(
        status='PROVISIONAL', near_duplicate_reviews_pending=0,
        all_splits_sha256=b.sha256_file(root / 'splits/all.csv'),
        manifest_sha256=b.sha256_file(root / 'prepared_manifest.csv'),
        classification_counts={p: int(((df.split == p) & (df.classification_eligible == '1')).sum()) for p in b.PARTITIONS}))


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='glaucoma_software_test_')
        cls.root = Path(cls.temp.name)
        cls.original = fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_missing_export_fails_explicitly(self):
        with self.assertRaisesRegex(FileNotFoundError, 'handoff summaries'):
            b.load_and_audit(self.root / 'missing')

    def test_real_schema_round_trip_and_classification_keeps_bad_masks(self):
        _, df, _ = b.load_and_audit(self.root)
        effective, parts = b.effective_partitions(df)
        self.assertEqual(len(parts['train']), 16)
        self.assertEqual(set(parts['train'].segmentation_eligible), {'0'})
        self.assertTrue((effective.split == effective.original_split).all())
        self.assertEqual(set(df[df.dataset == 'chaksu'].patient_id), {''})

    def test_manifest_hash_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            with (root / 'splits/all.csv').open('a') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                b.load_and_audit(root)

    def test_individual_split_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); df = fixture(root)
            path = root / 'splits/train_classification.csv'
            damaged = b.read_table(path); damaged.loc[0, 'binary_label'] = '1'
            damaged.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'differ'):
                b.load_and_audit(root)

    def test_known_patient_leak_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); df = fixture(root)
            df.loc[[0, 16], 'patient_id'] = 'shared_known_identity'
            write_fixture(root, df)
            with self.assertRaisesRegex(ValueError, 'crosses splits'):
                b.load_and_audit(root)

    def test_papila_suspect_cannot_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); df = fixture(root)
            i = df[df.split == 'external_test'].index[0]
            df.loc[i, 'original_label'] = '2'
            write_fixture(root, df)
            with self.assertRaisesRegex(ValueError, 'label mapping'):
                b.load_and_audit(root)

    def test_filename_policy_groups_and_quarantines_transitively(self):
        df = self.original.copy()
        # Unmapped train/test P10 pair; another training duplicate shares group.
        df.loc[0, ['image_id', 'camera']] = ['chaksu_Bosch_train_P10_Image1', 'Bosch']
        test_i = df[df.split == 'in_domain_test'].index[0]
        df.loc[test_i, ['image_id', 'camera']] = ['chaksu_Bosch_test_P10_Image2', 'Bosch']
        df.loc[1, 'group_id'] = df.loc[0, 'group_id']
        # P1 across train/validation must stay together in development training.
        df.loc[2, ['image_id', 'camera']] = ['chaksu_Bosch_train_P1_Image1', 'Bosch']
        val_i = df[df.split == 'validation'].index[0]
        df.loc[val_i, ['image_id', 'camera']] = ['chaksu_Bosch_train_P1_Image2', 'Bosch']
        effective, parts = b.effective_partitions(df)
        self.assertTrue(effective.loc[[0, 1, test_i], 'split'].str.startswith('quarantine').all())
        self.assertEqual(effective.loc[val_i, 'split'], 'train')
        self.assertEqual(effective.loc[2, 'analysis_group'], effective.loc[val_i, 'analysis_group'])
        self.assertEqual(df.loc[val_i, 'split'], 'validation')  # Input preserved.

    def test_split_policy_order_invariance(self):
        a, _ = b.effective_partitions(self.original)
        c, _ = b.effective_partitions(self.original.sample(frac=1, random_state=7))
        cols = ['image_id', 'split', 'analysis_group']
        pd.testing.assert_frame_equal(a[cols].sort_values('image_id').reset_index(drop=True),
                                      c[cols].sort_values('image_id').reset_index(drop=True))

    def test_training_input_exactly_matches_imagenet_normalization(self):
        _, parts = b.effective_partitions(self.original)
        rows = parts['train']; b.check_images(self.root, rows)
        x, y, image_id = b.FundusDataset(rows, self.root)[0]
        with Image.open(self.root / rows.iloc[0].classification_image) as im:
            rgb = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255
        expected = (rgb - np.asarray(b.MEAN)[:, None, None]) / np.asarray(b.STD)[:, None, None]
        np.testing.assert_allclose(x.numpy(), expected, atol=1e-6)
        self.assertEqual(x.shape, (3, 224, 224))
        self.assertEqual(float(y), 0)

    def test_path_guard_and_overlay_guard(self):
        for path in ('../outside/image_224.png', 'prepared/x/overlay_review.png', '/tmp/image_224.png'):
            with self.assertRaises(ValueError):
                b.image_path(self.root, path)

    def test_metrics_and_threshold_ties(self):
        y, p = [0, 0, 1, 1], [.1, .8, .7, .9]
        metrics = b.metric_values(y, p, .5)
        self.assertAlmostEqual(metrics['auroc'], .75)
        self.assertAlmostEqual(metrics['brier'], (.01 + .64 + .09 + .01) / 4)
        self.assertEqual((metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']), (1, 1, 0, 2))
        threshold = b.select_threshold(y, p, 1.0)
        self.assertAlmostEqual(threshold, .7)
        self.assertEqual(b.metric_values(y, p, threshold)['sensitivity'], 1.)
        self.assertEqual(b.select_threshold([0, 1], [.4, .4], .9), .4)

    def test_single_class_and_no_predicted_positive(self):
        values = b.metric_values([0, 0], [.1, .2])
        self.assertTrue(np.isnan(values['auroc']))
        self.assertTrue(np.isnan(values['sensitivity']))
        self.assertTrue(np.isnan(values['precision']))
        with self.assertRaises(ValueError):
            b.select_threshold([1, 1], [.2, .3])

    def test_probability_validation(self):
        for p in ([np.nan, .2], [-.1, .8], [.2, 1.1]):
            with self.assertRaises(ValueError):
                b.metric_values([0, 1], p)

    def test_prediction_alignment_is_by_image_id(self):
        _, parts = b.effective_partitions(self.original)
        rows = parts['validation']
        pred = pd.DataFrame(dict(image_id=rows.image_id, label=rows.binary_label, probability=.5)).iloc[::-1]
        joined = b.attach_metadata(pred, rows)
        self.assertEqual(len(joined), len(rows))
        bad = pred.copy(); bad['label'] = 1 - bad['label']
        with self.assertRaisesRegex(ValueError, 'alignment'):
            b.attach_metadata(bad, rows)

    def test_cluster_bootstrap_reproducible(self):
        pred = pd.DataFrame(dict(label=[0, 1, 0, 1, 0, 1], probability=[.1, .9, .2, .8, .3, .7],
                                 analysis_group=['a', 'a', 'b', 'b', 'c', 'c']))
        x = b.grouped_bootstrap(pred, .5, iterations=30)
        y = b.grouped_bootstrap(pred, .5, iterations=30)
        pd.testing.assert_frame_equal(x, y)
        self.assertTrue((x.bootstrap_groups == 3).all())
        self.assertEqual(x[x.metric == 'auroc'].lower_95.iloc[0], 1.)

    def test_each_architecture_returns_one_finite_logit(self):
        for name in ('resnet18', 'resnet50', 'efficientnet_b0'):
            model, _ = b.make_model(name, pretrained=False)
            model.eval()
            with torch.inference_mode():
                logits = model(torch.zeros(1, 3, 224, 224))
            self.assertEqual(tuple(logits.shape), (1, 1))
            self.assertTrue(torch.isfinite(logits).all())
            del model; gc.collect()

    def test_batchnorm_running_stats_held_fixed(self):
        model, _ = b.make_model('resnet18', pretrained=False)
        before = model.bn1.running_mean.clone()
        b.training_mode(model)
        model(torch.randn(2, 3, 224, 224)).sum().backward()
        torch.testing.assert_close(before, model.bn1.running_mean)
        self.assertTrue(model.conv1.weight.grad is not None)

    def test_new_photo_letterboxing_matches_upstream_geometry(self):
        path = self.root / 'new_photo.png'
        Image.new('RGB', (80, 40), (200, 100, 50)).save(path)
        im = b.letterbox_photo(path)
        self.assertEqual(im.size, (224, 224))
        self.assertEqual(im.getpixel((112, 0)), (0, 0, 0))
        self.assertEqual(im.getpixel((112, 112)), (200, 100, 50))

    def test_strict_json_with_undefined_metric(self):
        path = self.root / 'undefined.json'
        b.write_json(path, {'value': float('nan')})
        self.assertIsNone(json.loads(path.read_text())['value'])

    def test_notebook_schema_and_code_compile(self):
        nb = nbformat.read(HERE / '03_image_only_cnn_baseline.ipynb', as_version=4)
        nbformat.validate(nb)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'notebook_cell_{i}', 'exec')
                self.assertFalse(cell.outputs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
