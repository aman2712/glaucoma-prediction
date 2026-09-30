"""Copy quality_review.csv rows that actually have an overlay image into one
folder, named for quick sorting/scanning in Finder, instead of opening each
row's path one at a time from the terminal.

Usage:
    python3 make_quality_review_folder.py --out artifacts --dest ~/Desktop/quality_review
"""
import argparse
import csv
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()

    with (args.out / 'quality_review.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))

    to_review = [r for r in rows if r['overlay'].strip()]
    args.dest.mkdir(parents=True, exist_ok=True)
    for row in to_review:
        flagged = 'FLAGGED' if (row['empty_disc'] == '1' or row['empty_cup'] == '1'
                                or int(row['cup_outside_disc_pixels']) > 0) else 'sample'
        name = f"{flagged}_{row['dataset']}_{row['camera']}_{row['image_id']}.png"
        shutil.copy(args.out / row['overlay'], args.dest / name)

    print(f'Copied {len(to_review)} of {len(rows)} rows to {args.dest} '
          f'({sum(1 for n in args.dest.iterdir() if n.name.startswith("FLAGGED"))} flagged, '
          f'rest are sampled normals).')
    print('Open in Finder (Gallery view), go through each, then fill in reviewer/comment/status '
          'in quality_review.csv directly.')


if __name__ == '__main__':
    main()
