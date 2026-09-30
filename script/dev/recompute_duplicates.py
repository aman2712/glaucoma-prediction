"""Recompute near duplicate candidates from an existing prepared_manifest.csv
without re-running the image-processing stage of prepare_data.py.

Usage:
    python3 recompute_duplicates.py --out artifacts --threshold 4 --histogram
    python3 recompute_duplicates.py --out artifacts --threshold 2
"""
import argparse
import csv
from collections import Counter
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threshold', type=int, default=4)
    parser.add_argument('--histogram', action='store_true')
    args = parser.parse_args()

    rows = read_csv(args.out / 'prepared_manifest.csv')
    n = len(rows)
    print(f'{n} images, {n * (n - 1) // 2} pairs to check...')

    if args.histogram:
        # Sample-based histogram: full O(n^2) distance computation, but only
        # counting distances rather than storing every pair.
        counts = Counter()
        for i, a in enumerate(rows):
            ah = int(a['dhash'], 16)
            for b in rows[i + 1:]:
                bh = int(b['dhash'], 16)
                counts[(ah ^ bh).bit_count()] += 1
        print('\nHamming distance distribution (bits differing out of 64):')
        total = sum(counts.values())
        cumulative = 0
        for d in range(0, 17):
            c = counts.get(d, 0)
            cumulative += c
            if c:
                print(f'  distance {d:2d}: {c:8d} pairs   (cumulative: {cumulative:8d} / {total})')
        print('\nLook for a natural gap, genuine duplicates cluster near 0; '
              'everything past the gap is coincidental similarity, not duplication. '
              'Pick --threshold just below that gap.')
        return

    candidates = []
    for i, a in enumerate(rows):
        ah = int(a['dhash'], 16)
        for b in rows[i + 1:]:
            if a['pixel_sha256'] == b['pixel_sha256']:
                continue  # already an exact duplicate, not a near one
            distance = (ah ^ int(b['dhash'], 16)).bit_count()
            if distance <= args.threshold:
                candidates.append(dict(image_id_a=a['image_id'], image_id_b=b['image_id'],
                                       dhash_distance=distance, decision='pending', reviewer=''))
    write_csv(args.out / 'near_duplicate_review.csv', candidates,
              ['image_id_a', 'image_id_b', 'dhash_distance', 'decision', 'reviewer'])
    print(f'Wrote {len(candidates)} candidate pairs to near_duplicate_review.csv '
          f'(threshold <= {args.threshold} bits)')


if __name__ == '__main__':
    main()
