import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_data import (binary_mask, chaksu_label, contour_mask, grouped_validation,
                          label_stem, measurements, padded_resize, UnionFind,
                          split, write_csv, write_json, read_csv)


class PreparationTests(unittest.TestCase):
    def test_alpha_is_not_segmentation(self):
        with tempfile.TemporaryDirectory() as t:
            a = np.zeros((10, 10, 4), dtype=np.uint8)
            a[:, :, 3] = 255
            a[3:7, 3:7, :3] = 255
            p = Path(t) / 'mask.png'
            Image.fromarray(a).save(p)
            self.assertEqual(binary_mask(p, (10, 10)).sum(), 16)

    def test_measurements_on_known_rectangles(self):
        disc = np.zeros((20, 20), bool)
        cup = disc.copy()
        disc[2:12, 4:12] = True
        cup[4:9, 6:10] = True
        values = measurements(disc, cup)
        self.assertEqual(values['vcdr'], .5)
        self.assertEqual(values['hcdr'], .5)
        self.assertEqual(values['acdr'], .25)
        self.assertEqual(values['rim_area_px'], 60)

    def test_reject_invalid_containment_and_empty_mask(self):
        disc = np.zeros((10, 10), bool)
        cup = disc.copy()
        disc[1:4, 1:4] = True
        with self.assertRaises(ValueError):
            measurements(disc, cup)
        cup[8, 8] = True
        with self.assertRaises(ValueError):
            measurements(disc, cup)

    def test_polygon_xy_and_bounds(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'contour.txt'
            np.savetxt(p, [[5, 1], [7, 1], [7, 3], [5, 3]])
            a = contour_mask(p, (12, 6))
            self.assertTrue(a[2, 6])
            self.assertFalse(a[4, 2])
            np.savetxt(p, [[-1, 0], [1, 0], [1, 2]])
            with self.assertRaises(ValueError):
                contour_mask(p, (12, 6))

    def test_letterboxing_preserves_mask_values(self):
        a = Image.new('L', (8, 4), 255)
        result, transform = padded_resize(a, 16, True)
        arr = np.array(result)
        self.assertEqual(set(np.unique(arr)), {0, 255})
        self.assertEqual(transform['pad_top'], 2)
        self.assertEqual(arr[4:12].sum(), 8 * 16 * 255)

    def test_actual_source_label_conventions(self):
        self.assertEqual(label_stem('Image101.jpg-Image101-1.jpg'), 'image101')
        self.assertEqual(label_stem('P7_Image2.JPG'), 'p7_image2')
        self.assertEqual(chaksu_label('GLAUCOMA  SUSUPECT'), 1)
        with self.assertRaises(ValueError):
            chaksu_label('ungraded')

    def test_grouped_split_deterministic(self):
        groups = [[dict(camera='A', binary_label=i % 2)] * 2 for i in range(20)]
        a = grouped_validation(groups, .2, 2440)
        self.assertEqual(a, grouped_validation(groups, .2, 2440))
        self.assertEqual(len(a), 4)
        self.assertEqual({groups[i][0]['binary_label'] for i in a}, {0, 1})

    def test_transitive_duplicate_groups(self):
        uf = UnionFind(['a', 'b', 'c'])
        uf.join('a', 'b')
        uf.join('b', 'c')
        self.assertEqual(uf.find('a'), uf.find('c'))

    def _chaksu_rows(self, n, release):
        # Balanced across two labels, unique pixel hashes, unlinked patient identity
        # (the default, realistic state before --patient-groups is supplied).
        return [dict(image_id=f'{release}{i}', dataset='chaksu', patient_id='',
                     patient_id_verified='0', pixel_sha256=f'{release}{i}', camera='A',
                     chaksu_release=release, binary_label=str(i % 2),
                     classification_eligible='1', segmentation_eligible='1')
                for i in range(n)]

    def test_split_protects_external_and_rejects_unverified(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            rows = self._chaksu_rows(40, 'train')
            for i in range(2):
                rows.append(dict(image_id=f'p{i}', dataset='papila', patient_id='p1',
                                 patient_id_verified='1', pixel_sha256='0' if i == 0 else 'p2', camera='B',
                                 chaksu_release='', binary_label='1' if i == 0 else '',
                                 classification_eligible='1' if i == 0 else '0',
                                 segmentation_eligible='1'))
            # p0 duplicates c0's pixel hash: p0 is PAPILA, so c0 (the CHAKSU twin) must be
            # excluded, not p0 -- PAPILA identity/labels are already externally verified.
            rows[0]['pixel_sha256'] = '0'
            write_csv(out / 'prepared_manifest.csv', rows)
            write_csv(out / 'near_duplicate_review.csv', [],
                      ['image_id_a', 'image_id_b', 'dhash_distance', 'decision', 'reviewer'])
            cfg = dict(seed=2440, validation_fraction=.2)
            with self.assertRaises(ValueError):
                split(cfg, out)
            with self.assertRaises(ValueError):
                # allow_provisional alone is not enough; must state the limitation.
                split(cfg, out, allow_provisional=True)
            split(cfg, out, allow_provisional=True,
                  limitation_note='CHAKSU patient identity unverified in this test fixture.')
            result = {r['image_id']: r for r in read_csv(out / 'splits/all.csv')}
            self.assertEqual(result['train0']['split'], 'quarantine_external_overlap')
            self.assertEqual(result['p0']['split'], 'external_test')
            self.assertEqual(result['p1']['split'], 'external_test')
            self.assertEqual(len(read_csv(out / 'splits/external_test_classification.csv')), 1)
            with self.assertRaises(ValueError):
                # Splits are frozen once written; rerunning must not silently overwrite.
                split(cfg, out, allow_provisional=True, limitation_note='again')

    def test_release_overlap_is_quarantined_not_leaked(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            rows = self._chaksu_rows(10, 'train') + self._chaksu_rows(10, 'test')
            # Same identity appears in both CHAKSU releases -- must be excluded from both
            # development and the in-domain held-out test, not counted as generalization.
            rows[0]['pixel_sha256'] = 'shared'
            rows[10]['pixel_sha256'] = 'shared'
            write_csv(out / 'prepared_manifest.csv', rows)
            write_csv(out / 'near_duplicate_review.csv', [],
                      ['image_id_a', 'image_id_b', 'dhash_distance', 'decision', 'reviewer'])
            cfg = dict(seed=2440, validation_fraction=.2)
            split(cfg, out, allow_provisional=True, limitation_note='unverified in test fixture')
            result = {r['image_id']: r for r in read_csv(out / 'splits/all.csv')}
            self.assertEqual(result['train0']['split'], 'quarantine_release_overlap')
            self.assertEqual(result['test0']['split'], 'quarantine_release_overlap')
            self.assertEqual(result['test1']['split'], 'in_domain_test')
            self.assertNotIn(result['train0']['split'], ('train', 'validation'))

    def test_in_domain_test_must_keep_both_classes(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            rows = self._chaksu_rows(10, 'train')
            # Every CHAKSU-test-release row is the same label -- the in-domain
            # held-out set would be useless for a binary metric like AUROC.
            single_class = self._chaksu_rows(4, 'test')
            for r in single_class:
                r['binary_label'] = '0'
            rows += single_class
            write_csv(out / 'prepared_manifest.csv', rows)
            write_csv(out / 'near_duplicate_review.csv', [],
                      ['image_id_a', 'image_id_b', 'dhash_distance', 'decision', 'reviewer'])
            cfg = dict(seed=2440, validation_fraction=.2)
            with self.assertRaises(ValueError):
                split(cfg, out, allow_provisional=True, limitation_note='unverified in test fixture')


if __name__ == '__main__':
    unittest.main()
