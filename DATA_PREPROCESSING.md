# Data Preprocessing and Splitting Documentation

Record of what was done, what was found, why each decision was made, and how to reproduce it. The source files (`inventory_summary.json`,`preparation_summary.json`, `splits/split_summary.json`) are committed under `docs/pipeline_evidence/` and this document should be read alongside them.

## 1. Data sources

### CHAKSU (Kumar et al., 2023, *Scientific Data* 10:70, doi:10.1038/s41597-023-01943-4)

Collected at Kasturba Medical College, Manipal, India (IISc–MAHE collaboration),
subjects aged 18–76, Indian ethnicity, over a two-year collection period. 1,345
fundus images across three cameras, released as a publisher-defined Train
(1,009 images) and Test (336 images) split, each with five expert
optic-disc/optic-cup annotations and a binary glaucoma decision.

The paper (Table 2, p.5) reports per-camera counts that our `inventory` stage
reproduced:

| Camera | Train | Test |
|---|---|---|
| Remidio (2448×3264) | 810 | 264 |
| Bosch (1920×1440) | 104 | 41 |
| Forus (2048×1536) | 95 | 31 |
| **Total** | **1009** | **336** |

We checked both the bundled README and the full peer-reviewed paper specifically
for any documentation of patient identity, filename conventions, or how many
images each subject contributed. **Neither document contains this information
anywhere.** The paper describes recruitment and the five-expert annotation
protocol in detail but never states the number of unique subjects or whether a
subject could contribute more than one image. This absence is directly relevant
to §4 below.

The paper's Table 3 also reports the agreement between each expert and the
STAPLE-fused ground truth (Se = 0.989, Sp = 0.999, Dice = 0.976 on average) —
the highest agreement of the four fusion methods offered (mean, median,
majority, STAPLE). This is the paper's own justification for treating STAPLE
as the reference mask, which is why `config.json` sets
`"chaksu_mask_reference": "STAPLE"` rather than an arbitrary choice.

### PAPILA (Kovalyk et al., 2022, *Scientific Data* 9:291, doi:10.1038/s41597-022-01388-1)

Collected at Hospital General Universitario Reina Sofía, Murcia, Spain
(2018–2020). 244 patients, both eyes each (488 images), Topcon TRC-NW400
camera, diagnosis coded per eye as 0 (healthy), 1 (glaucoma), or 2
("suspect"). Unlike CHAKSU, **patient identity is a first-class part of this
dataset** — it is the dataset's contribution (fundus images *of both
eyes of the same patient*), encoded directly in the filename
(`RET###OD`/`RET###OS`).

The paper's Table 1 reports 333 healthy / 155 glaucoma-or-suspect (merging
classes 1 and 2 for comparison with CHAKSU), and Table 4 breaks this
down eye-wise as 333 healthy, 87 glaucoma, 68 suspect. Our `inventory` stage
reproduced this: `papila/Topcon/label=0: 333`,
`label=1: 87`, `label=2: 68` (87 + 68 = 155, matching Table 1's merged
figure). This cross-check against the publisher's own reported numbers is
strong evidence the PAPILA side of `inventory_chaksu`/`inventory` logic is
correct but lacks consistency.

PAPILA's class-2 ("suspect") images are excluded from classification
(`classification_eligible = 0`, `exclusion_reason = suspicious_label`) because
the diagnosis itself is ambiguous, so it is not a data quality problem. The
paper's own multiclass experiments show the suspect class does not sit as an intermediate class between healthy and glaucoma in embedding space, and removing it measurably improved their baseline AUC (their Test #2). Excluding it here follows the same reasoning the dataset's own
authors used.

## 2. Pipeline stages

### `inventory`

Walks both CHAKSU releases and PAPILA, joins each image to its label and
mask/contour references, and writes `inventory.csv`. Run:

```
Inventoried 1833 images (1009 CHAKSU-train, 336 CHAKSU-test, 488 PAPILA).
```

One real data bug was found and fixed here: the released CHAKSU label tables
are not consistently named across cameras and releases. Train-release
Bosch/Forus/Remidio and test-release Bosch all use a `Majority Decision`
column, but test release Forus and Remidio use `Glaucoma Decision` for the
same field. The original code hard-required `Majority Decision` and raised
`KeyError: 'Majority Decision'` partway through the real run. The fix accepts
either column name but still cross-checks the value against the five expert
votes, so a data problem would still be caught, only the column
*name* is now tolerant, not the value.

A second interesting data notation: `inventory_summary.json`'s
`chaksu_source_join` shows Bosch's label table has 145 rows in both the
train and test folders (`label_rows: 145` for each), while Forus and Remidio's
label tables match their own release's image count exactly. This means the
publisher's Bosch label CSV is not release partitioned because it lists all 145
Bosch images (104 train + 41 test) regardless of which folder is being
inventoried. The pipeline already handles this safely, since it only keeps
label rows matching images actually present in that release's image folder
(`label_rows_without_release_image: 41` for train, `104` for test, exactly
the complementary release's count in each case).

### `prepare`

Loads every image once: checks EXIF orientation, builds the binary mask
(CHAKSU) or fills the polygon from expert coordinates (PAPILA), rejects masks
where the cup extends outside the disc, computes oracle CDR/area measurements
into `expert_measurements.csv` (kept separate from `load_images.py`'s inputs),
and produces the two resized/letterboxed image versions. Run:

```
Prepared 1833 images, invalid_reference_geometry: 33 (1.8%)
```

33 of 1,833 images had an invalid reference mask (empty disc/cup, or cup
extending outside disc) and are flagged `segmentation_eligible = 0` rather
than silently repaired or dropped — they still exist in `prepared_manifest.csv`
for the classification task, they're just excluded from segmentation-specific
splits.

### `split`

Produces four frozen partitions, described and justified in §3–4. The
mechanism (union-find over shared patient identity, exact pixel hash, and
manually reviewed duplicates) is unchanged from earlier design work, what follows is the data findings.

## 3. Why three tiers instead of two, and what the real numbers show

CHAKSU's own Test release is used as a third tier (`in_domain_test`) between
`validation` and PAPILA's `external_test`, rather than being discarded or
merged into training. This is because it's the same population and acquisition
effort as `train`/`validation`, so it is a weaker generalisation check
than PAPILA, but it's a publisher-defined held-out set, cheap, and lets the eventual error analysis separate "doesn't
generalise even within CHAKSU" from "generalises within CHAKSU but not to a
new site." The final frozen split:

| Partition | Images (classification) | Images (segmentation) |
|---|---|---|
| `train` | 807 | 787 |
| `validation` | 202 | 200 |
| `in_domain_test` | 336 | 327 |
| `external_test` | 420 | 486 |

(`train` + `validation` = 1,009 = CHAKSU's full train release, confirming no
CHAKSU images were lost to quarantine - `in_domain_test` = 336 = CHAKSU's full
test release, same reasoning - `external_test` classification count of 420 =
488 − 68 excluded PAPILA suspect images.)

Per-camera class balance in the frozen split (`split_summary.json`):

| | Bosch (0/1) | Forus (0/1) | Remidio (0/1) |
|---|---|---|---|
| `train` | 68 / 15 | 65 / 13 | 565 / 81 |
| `validation` | 17 / 4 | 14 / 3 | 143 / 21 |
| `in_domain_test` | 40 / 1 | 29 / 2 | 216 / 48 |

**A Data Limitation**: Bosch's
`in_domain_test` has only 1 glaucoma-suspect image, and Bosch/Forus
`validation` have only 4 and 3 positives respectively. Any per-camera
sensitivity/specificity computed on these small positive counts will be
extremely noisy where a single image flipping right/wrong swings the metric by a
large margin. 

## 4. Duplicate detection and handling

### Exact duplicates

`prepare`'s pixel-hash check found 7 exact-pixel-identical pairs, all
within the Bosch camera: 6 within the train release (`Image102`/`Image120`,
`Image107`/`Image128`, `Image108`/`Image144`, `Image109`/`Image145`,
`Image156`/`Image157`, `Image168`/`Image169`) and 1 within the test release
(`P16_Image1`/`P29_Image2`). None cross the train/test release boundary which is important since it means CHAKSU's exact duplication does not threaten the train/in-domain-test separation. The union-find grouping in `split` keeps each pair together automatically, so none of these seven pairs could ever be split across train and validation.

### Near duplicates

The first near duplicate detector we ran (a standard 64-bit dHash on the full
letterboxed image, threshold ≤ 4 bits, all cameras compared against each
other) returned 50,333 candidate pairs out of 1,833 images, effectively
unusable, and a red flag that the method itself was broken. Inspecting the Hamming-distance histogram confirmed this: even at
distance 0 (a "perfect" dHash match) there were 637 pairs, when only 7 were
genuinely pixel-identical. The dHash was picking up the shared circular
vignette common to all fundus photos, not the retinal
content, a known failure mode of coarse perceptual hashing on images that
share a dominant global structure.

**Fix**: crop the central 60% of each image (removing the shared vignette),
raise the hash to 256 bits, and restrict comparisons to the same camera only
(since a real duplicate cannot cross devices). Re-running the histogram on real data
showed a clear, genuine gap this time: only 33 pairs below distance 12, then a
23× jump to 759 pairs in the 12–17 band. A threshold of 11 was chosen from
this gap (not guessed), producing **26 candidate pairs**, all within the
Remidio camera.

### Manual review of the 26 candidates

Every one of the 26 candidates was visually inspected (image pairs copied into
labeled folders via `make_review_folders.py` for side-by-side comparison). **Result: all 26 were marked `not_duplicate`.** Two of the 26
crossed the CHAKSU train/test release boundary
(`Remidio_train_IMG_2452`↔`Remidio_test_IMG_3922`,
`Remidio_train_IMG_2674`↔`Remidio_test_IMG_3816`); both were confirmed as different photos, so no images were quarantined for release overlap
via this path either (`quarantined_release_overlap: 0` in the final split
summary, corroborating the exact-hash finding in §4.1).

This negative result is still a meaningful finding: it demonstrates the
detector, once fixed, correctly separates true near duplicates from
coincidentally similar (but distinct) photos of different discs.

## 5. Patient linkage

CHAKSU publishes no patient identifiers in its label tables. Before falling
back to a provisional split, three sources were checked directly:

1. **The CHAKSu supplied README** describes
   file structure and per-camera counts, says nothing about filenames or
   subject identity.
2. **The peer-reviewed *Scientific Data* paper** (Kumar et al., 2023) had
   **no mention anywhere of filename conventions, a subject list, or how many
   images each subject contributed.**
3. **The actual released filenames** — inspected directly. CHAKSU Bosch
   train, Forus (both releases), and Remidio (both releases) use plain
   sequential numbering (`Image101.JPG`, `1.png`, `IMG_2450.JPG`) with no
   identifiable structure. **CHAKSU Bosch test-release** filenames, however,
   follow a `P<n>_Image[1|2].JPG` pattern. This was verified visually, not
   assumed: `P11_Image1`/`P11_Image2` show the same eye photographed twice;
   `P16_Image1`/`P16_Image2` show two different eyes (left/right) of
   apparently the same person. Either way, `P<n>` functions as a genuine
   two-image patient identifier for this one release.

`extract_bosch_test_patients.py` extracts this mapping directly from the
41 Bosch-test filenames (`patient_groups.csv`), producing real, filename-
derived patient identity for those 41 images. This is not an ID we created but rather it is read directly from information the dataset's own undocumented file-naming already
contains.

However, the other 1,304 CHAKSU images (Bosch
train + all Forus + all Remidio) have no recoverable patient signal anywhere
in the released data, documentation, or literature. Inventing sequential IDs
for these would not add information, it would assert "every image is a
different patient" without evidence, which could actively hide a leak if
false.

The frozen split was produced with:
```
--patient-groups patient_groups.csv --allow-provisional --limitation-note "Patient identity recovered from filename for CHAKSU Bosch test-release images only (P##_ImageN naming, verified visually; confirmed absent from both the CHAKSU README and the paper). No patient identifier exists in the released data for CHAKSU Bosch train-release or for any Forus/Remidio images; these remain image-level, not patient-level, for this milestone."
```
`split_summary.json` records `"status": "PROVISIONAL"`,
`"unknown_chaksu_patient_images": 1304` (= 1,345 − 41, confirming the Bosch
linkage was applied and nothing else was), and the limitation note verbatim.

### Decision Justification

- The tier that matters most for a generalisation claim is unaffected.
  PAPILA (`external_test`) has genuine, publisher-verified patient identity
  from its clinical spreadsheets, the provisional status only touches the
  *internal* CHAKSU train/validation/in-domain-test boundary.
- The risk is bounded and directional. If an unlinked CHAKSU patient's
  images happened to land in two different tiers, the worst-case effect is an
  optimistic bias on that specific internal comparison since it cannot invalidate
  the PAPILA external result, and it cannot make the model itself wrong.
- **It reflects a data availability limit, checked at three independent
  sources. The 41 image resolution proves the
  mechanism works wherever real signal exists and the remaining 1,304 have none
  to find.

## 6. Reproducibility

- Deterministic given `config.json`'s `seed` (2440), used for the
  validation-fraction search.
- Every output is hashed into `split_summary.json`:
  `manifest_sha256`, `all_splits_sha256`, `duplicate_review_sha256`,
  `patient_mapping_sha256` — a given split can be tied back to the exact
  inputs that produced it.
- `preparation_summary.json` records the exact Python/NumPy versions and a
  SHA-256 of `prepare_data.py` itself.
- Full command sequence to reproduce from scratch:
  ```bash
  python3 prepare_data.py inventory --config config.json --out artifacts
  python3 prepare_data.py prepare   --config config.json --out artifacts
  # review artifacts/near_duplicate_review.csv (see §4) and
  # artifacts/quality_review.csv before splitting
  python3 extract_bosch_test_patients.py --out artifacts --write patient_groups.csv
  python3 prepare_data.py split --config config.json --out artifacts \
    --patient-groups patient_groups.csv --allow-provisional \
    --limitation-note "<see §5 for exact text>"
  ```
- Unit tests (`script/tests/test_preparation.py`, 11 tests) cover mask
  validation, measurement correctness, contour rasterization, letterboxing,
  label-string parsing, deterministic grouped splitting, transitive duplicate
  grouping, and the split-quarantine logic (external overlap, release
  overlap, and the in-domain-test both-classes guard) — run with
  `python3 -m unittest test_preparation -v` from `script/tests/`.
