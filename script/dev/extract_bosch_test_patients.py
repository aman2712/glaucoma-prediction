"""Build patient_groups.csv entries for CHAKSU images where patient identity is
recoverable from the filename itself: CHAKSU Bosch test-release files are named
P<patient_number>_Image<1 or 2>.JPG. Every other CHAKSU camera/release has no
such pattern (plain sequential numbers) and is left unmapped, this script
only ever writes rows it can justify directly from a filename.

Usage:
    python3 extract_bosch_test_patients.py --out artifacts --write patient_groups.csv
"""
import argparse
import csv
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--write', type=Path, required=True,
                        help='Output patient_groups.csv (image_id,patient_id)')
    args = parser.parse_args()

    rows = []
    with (args.out / 'prepared_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        manifest = list(csv.DictReader(f))

    matched, unmatched_bosch_test = 0, 0
    for row in manifest:
        if row['dataset'] != 'chaksu' or row['camera'] != 'Bosch' or row['chaksu_release'] != 'test':
            continue
        stem = Path(row['image_path']).stem  # e.g. "P16_Image1"
        m = re.fullmatch(r'P(\d+)_Image[12]', stem, re.IGNORECASE)
        if m:
            rows.append(dict(image_id=row['image_id'], patient_id=f'chaksu_bosch_test_P{int(m.group(1))}'))
            matched += 1
        else:
            unmatched_bosch_test += 1
            print(f'  no P## pattern match: {stem} ({row["image_id"]})')

    with args.write.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'patient_id'])
        w.writeheader()
        w.writerows(rows)

    n_patients = len({r['patient_id'] for r in rows})
    print(f'\nMatched {matched} CHAKSU Bosch-test images to {n_patients} patients.')
    if unmatched_bosch_test:
        print(f'{unmatched_bosch_test} Bosch-test images did NOT match the P##_ImageN pattern -- inspect these before trusting this mapping.')
    print(f'Wrote {args.write}')
    print('Every other CHAKSU image (Bosch train, all Forus, all Remidio) has no recoverable '
          'patient signal and is intentionally left out of this file.')


if __name__ == '__main__':
    main()
