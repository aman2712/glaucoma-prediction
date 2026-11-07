"""Recompute near-duplicate candidates using a hash that discriminates
fundus photos by cropping the shared black vignette/background out first, use a higher-resolution hash, and only compare
images from the same camera.

Usage:
    python3 recompute_duplicates_v2.py --out artifacts --histogram
    python3 recompute_duplicates_v2.py --out artifacts --threshold 8
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def content_hash(path, crop_fraction=0.6, hash_size=16):
    with Image.open(path) as im:
        gray = im.convert('L')
        w, h = gray.size
        cw, ch = int(w * crop_fraction), int(h * crop_fraction)
        left, top = (w - cw) // 2, (h - ch) // 2
        gray = gray.crop((left, top, left + cw, top + ch))
        small = np.asarray(gray.resize((hash_size + 1, hash_size)))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value, len(bits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threshold', type=int, default=8)
    parser.add_argument('--histogram', action='store_true')
    parser.add_argument('--crop-fraction', type=float, default=0.6,
                        help='Central fraction of the image kept before hashing; '
                             'excludes the shared black vignette/background.')
    parser.add_argument('--hash-size', type=int, default=16,
                        help='Hash grid size (hash_size^2 bits); 16 -> 256 bits.')
    args = parser.parse_args()

    rows = read_csv(args.out / 'prepared_manifest.csv')
    print(f'Hashing {len(rows)} images (crop_fraction={args.crop_fraction}, '
          f'hash_size={args.hash_size})...')
    hashes, nbits = {}, None
    for i, row in enumerate(rows, 1):
        path = args.out / row['classification_image']
        value, nbits = content_hash(path, args.crop_fraction, args.hash_size)
        hashes[row['image_id']] = value
        if i % 200 == 0:
            print(f'  hashed {i}/{len(rows)}', flush=True)

    by_camera = defaultdict(list)
    for row in rows:
        by_camera[(row['dataset'], row['camera'])].append(row['image_id'])
    total_pairs = sum(len(v) * (len(v) - 1) // 2 for v in by_camera.values())
    print(f'{total_pairs} same-camera pairs to check (of {len(rows) * (len(rows) - 1) // 2} total)')

    if args.histogram:
        counts = Counter()
        for ids in by_camera.values():
            for i, a in enumerate(ids):
                ha = hashes[a]
                for b in ids[i + 1:]:
                    counts[(ha ^ hashes[b]).bit_count()] += 1
        print(f'\nHamming distance distribution (bits differing out of {nbits}):')
        cumulative = 0
        step = max(1, nbits // 40)
        for d in range(0, nbits // 2, step):
            c = sum(counts.get(x, 0) for x in range(d, d + step))
            cumulative += c
            if c:
                print(f'  distance {d:3d}-{d + step - 1:3d}: {c:8d} pairs   (cumulative: {cumulative})')
        print('\nLook for a gap now, with background excluded, genuine duplicates '
              'should sit far closer to 0 than everything else.')
        return

    id_to_row = {r['image_id']: r for r in rows}
    candidates = []
    for ids in by_camera.values():
        for i, a in enumerate(ids):
            ha = hashes[a]
            for b in ids[i + 1:]:
                if id_to_row[a]['pixel_sha256'] == id_to_row[b]['pixel_sha256']:
                    continue
                distance = (ha ^ hashes[b]).bit_count()
                if distance <= args.threshold:
                    candidates.append(dict(image_id_a=a, image_id_b=b,
                                           dhash_distance=distance, decision='pending', reviewer=''))
    write_csv(args.out / 'near_duplicate_review.csv', candidates,
              ['image_id_a', 'image_id_b', 'dhash_distance', 'decision', 'reviewer'])
    print(f'Wrote {len(candidates)} candidate pairs (threshold <= {args.threshold} bits)')


if __name__ == '__main__':
    main()
