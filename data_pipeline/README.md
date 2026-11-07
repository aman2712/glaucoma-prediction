# Data Preparation

Glaucoma prescreening from fundus images. Development on **CHAKSU**
(Kumar et al., 2023), external test on **PAPILA** (Kovalyk et al., 2022).

**For the full account of every decision, the bugs found running this against
real data, the literature checked, and why the split is provisional, see
[`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md).**

## Evaluation tiers 

| Split | Source | Classification images | When it may be touched |
|---|---|---|---|
| `train` | CHAKSU train release | 807 | Freely, during development |
| `validation` | CHAKSU train release | 202 | Freely, during development |
| `in_domain_test` | CHAKSU test release | 336 | Only after model + hyperparameters are frozen |
| `external_test` | PAPILA | 420 | Only after model + thresholds are frozen |

Full dataset: 1,833 images total (1,009 CHAKSU-train + 336 CHAKSU-test + 488
PAPILA). See `../docs/pipeline_evidence/split_summary.json`'s `limitation_note` and
`DATA_PREPROCESSING.md` §5 for data issues and how they are provisionally handled.

## Pipeline

```bash
python prepare_data.py inventory --config config.json --out artifacts
python prepare_data.py prepare   --config config.json --out artifacts
# review artifacts/near_duplicate_review.csv and artifacts/quality_review.csv,
# then optionally: python extract_bosch_test_patients.py --out artifacts --write patient_groups.csv
python prepare_data.py split --config config.json --out artifacts \
  --patient-groups patient_groups.csv --allow-provisional \
  --limitation-note "..."
```

`config.json` needs `chaksu_train_root`, `chaksu_test_root`, and `papila_root`
pointing at your local downloads (siblings of this folder by default, see
`../Train`, `../Test`, `../PapilaDB-...`). Datasets are **not** committed to
this repo, download them separately (CHAKSU: figshare/IISc-MAHE release;
PAPILA: figshare, doi:10.1038/s41597-022-01388-1) and place them per the
layout above.

## What needs to be reviewed before `split`

- `near_duplicate_review.csv` - candidate near duplicate pairs (dHash on the
  cropped, camera-restricted image). Set each `decision` to `duplicate`/`not_duplicate`.
- `quality_review.csv` — per-image mask sanity flags.
- CHAKSU patient linkage — see `DATA_PREPROCESSING.md` §5. Real, filename-
  derived linkage exists for the Bosch test release only
  (`extract_bosch_test_patients.py`); everything else requires
  `--allow-provisional --limitation-note`.

## Leakage guards built into `split`

- Images sharing a `patient_id`, an exact pixel hash, or a manually confirmed
  duplicate decision are grouped and never split across tiers.
- A CHAKSU image identity spanning both CHAKSU releases is excluded from both
  (`quarantine_release_overlap`); one spanning CHAKSU and PAPILA is excluded
  from CHAKSU's side (`quarantine_external_overlap`). Both are `0` in the
  current real run.
- `split` refuses to finish if `in_domain_test` loses a whole class to
  quarantine.
- Splits are frozen once written - rerunning `split` on an existing `--out`
  raises rather than silently overwriting.

## Environment

The repository-level `requirements.txt` covers this pipeline and the models.
`load_images.py` additionally needs `torch`/`torchvision` (its own docstring
says so).

## Tests

```bash
cd tests && python3 -m unittest test_preparation -v
```
11 tests, no dataset required, covers mask/measurement correctness and the
split-quarantine logic against synthetic fixtures.
