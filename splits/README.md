# Common split of record

Which image belongs to which split, for every model. Built once from the data pipeline's frozen split plus a precautionary grouping of possible repeat patients (`src/glaucoma_project/data/effective_splits.py`), and not recomputed by any model.

- `all.csv`: one row per image (1,833), with its `split` and its `analysis_group`. A model takes the rows of its own splits that its task can use (`classification_eligible` or `segmentation_eligible`).
- `changes.csv`: the four images that moved relative to the pipeline's split, with the reason.
- `common_split.json`: SHA-256 of the pipeline file and of `all.csv`, and the counts per split.

Rebuild and compare with the local data export: `python -m glaucoma_project.data.common_split --check`. Never edit these files; a change of split is a new, documented decision.
