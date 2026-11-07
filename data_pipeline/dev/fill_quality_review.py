"""Fill in quality_review.csv for the flagged/sampled rows that were manually
inspected. Only touches rows that actually had an overlay to review; anything
without one is left untouched.

Usage:
    python3 fill_quality_review.py --out artifacts --reviewer "Yomna"
"""
import argparse
import csv
from pathlib import Path

# The four flagged images that got a closer, full-zoom look due to their
# larger outside-pixel counts.
CLOSELY_CHECKED = {
    'chaksu_Remidio_train_IMG_2801',
    'papila_Topcon_RET119OD',
    'chaksu_Remidio_test_IMG_4410',
    'papila_Topcon_RET044OD',
}

FLAGGED_COMMENT = (
    'Boundary anti-aliasing / annotation-precision artefact; visually '
    'confirmed anatomically valid, no discrete missing region.'
)
CLOSELY_CHECKED_COMMENT = (
    'Larger outside-pixel count than other flags; re-checked at full zoom '
    'specifically. Confirmed a thin boundary sliver, not a discrete '
    'overlap region -- anatomically valid.'
)
SAMPLE_COMMENT = 'Sampled normal; disc/cup contours trace the visible structures correctly.'

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--reviewer', required=True)
args = parser.parse_args()

path = args.out / 'quality_review.csv'
rows = list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))

updated = 0
for row in rows:
    if not row['overlay'].strip():
        continue  # nothing was shown to review for this row; leave it alone
    flagged = (row['empty_disc'] == '1' or row['empty_cup'] == '1'
               or int(row['cup_outside_disc_pixels']) > 0)
    row['reviewer'] = args.reviewer
    if row['image_id'] in CLOSELY_CHECKED:
        row['status'] = 'confirmed_artifact'
        row['comment'] = CLOSELY_CHECKED_COMMENT
    elif flagged:
        row['status'] = 'confirmed_artifact'
        row['comment'] = FLAGGED_COMMENT
    else:
        row['status'] = 'confirmed_ok'
        row['comment'] = SAMPLE_COMMENT
    updated += 1

with path.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f'Updated {updated} reviewed rows in {path}')
