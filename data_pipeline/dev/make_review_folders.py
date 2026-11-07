"""Copy each near-duplicate candidate pair into its own folder for quick visual
review in Finder, instead of opening images one at a time from the terminal.

Usage:
    python3 make_review_folders.py --out artifacts --dest ~/Desktop/duplicate_review
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

    manifest = {}
    with (args.out / 'prepared_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            manifest[row['image_id']] = row['classification_image']

    with (args.out / 'near_duplicate_review.csv').open(encoding='utf-8-sig', newline='') as f:
        pairs = list(csv.DictReader(f))

    args.dest.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(pairs, 1):
        a, b = row['image_id_a'], row['image_id_b']
        folder = args.dest / f'pair_{i:02d}_distance_{row["dhash_distance"]}'
        folder.mkdir(exist_ok=True)
        shutil.copy(args.out / manifest[a], folder / f'A_{a}.png')
        shutil.copy(args.out / manifest[b], folder / f'B_{b}.png')
    print(f'Wrote {len(pairs)} pair folders to {args.dest}')
    print('Open it and flip through each pair_NN folder.')


if __name__ == '__main__':
    main()
