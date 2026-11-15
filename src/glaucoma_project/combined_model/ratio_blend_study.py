"""Experiment X7: the P01b blend with the ratios of the robust segmentation network (X5).

The cross-fitted image scores are those of P01b, which are reused and not retrained: the development
fold scores (``image_runs_dir``) fit the blend, and the saved test image logits (``image_scores``) are blended.

Usage, in order (the design is in docs/experiment_log.md, "X-series design" and the X7 clarification):
  python -m glaucoma_project.measurements.classification_inputs --manifest docs/frozen_x5.json --adapt --out FILE
  python -m glaucoma_project.combined_model.ratio_blend_study CONFIG fit        development data only
  python -m glaucoma_project.combined_model.ratio_blend_study CONFIG freeze
  python -m glaucoma_project.combined_model.ratio_blend_study CONFIG score --out DIR   both test sets, once
  python -m glaucoma_project.combined_model.ratio_blend_study CONFIG analyse --scores DIR --out DIR
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from glaucoma_project import COMMON_SPLIT_DIR, REPO_ROOT
from glaucoma_project.combined_model.blend import (RATIOS, blend_logits, choose_regularisation,
                                                   development_diagnostic, development_table, fit_blend)
from glaucoma_project.combined_model.blend_study import (TEST_PARTITIONS, compare_scores, read_held_out_logits,
                                                         seed_directory)
from glaucoma_project.files import write_json
from glaucoma_project.freeze import freeze, verify
from glaucoma_project.combined_model.variant_study import load_study, manifest_path, runs_directory

CODE_PATHS = [f"src/glaucoma_project/{name}" for name in
              ("combined_model/ratio_blend_study.py", "combined_model/blend_study.py", "combined_model/blend.py",
               "combined_model/analysis.py", "classification", "evaluation", "files.py", "freeze.py")]


def _path(study: dict, key: str) -> Path:
    return REPO_ROOT / study[key]


def _image_study(study: dict) -> dict:
    """The P01b study whose saved fold scores supply the image score."""
    return {**study, "runs_dir": study["image_runs_dir"]}


def fit_study(study: dict) -> dict:
    """Choose C by grouped cross-validation and fit the blend on development images (new ratios, old image scores)."""
    blend_path = runs_directory(study) / "blend.joblib"
    if blend_path.exists():
        raise FileExistsError(f"{blend_path} already exists: the blend is already fitted")
    table = development_table(read_held_out_logits(_image_study(study)), pd.read_csv(_path(study, "measurements")))
    c = choose_regularisation(table, study["c_grid"], study["n_folds"], study["fold_seed"])
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit_blend(table, c), blend_path)
    summary = {"c": c, "n_development_images": len(table),
               **development_diagnostic(table, c, study["n_folds"], study["fold_seed"])}
    write_json(runs_directory(study) / "development.json", summary)
    return summary


def frozen_inputs(study: dict) -> dict[str, Path]:
    """Every file the test scoring reads: the blend, the ratios, the image scores, the references, the split."""
    image_study = _image_study(study)
    files = {"blend": runs_directory(study) / "blend.joblib", "measurements": _path(study, "measurements"),
             "common_split": COMMON_SPLIT_DIR / "all.csv"}
    files |= {f"oof_seed{seed}": seed_directory(image_study, seed) / "oof_logits.csv" for seed in study["seeds"]}
    files |= {f"image_scores_{name}": _path(study, "image_scores") / f"{name}.csv" for name in TEST_PARTITIONS}
    files |= {f"reference_{name}": REPO_ROOT / path for name, path in study["references"].items()}
    return files


def freeze_study(study: dict) -> dict:
    files = frozen_inputs(study)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Not finished: {missing}")
    return freeze(files, {"study": study}, manifest_path(study), CODE_PATHS + [f"configs/{study['name']}.yaml"])


def score_test_sets(study: dict, out_dir: Path, manifest: Path | None = None) -> None:
    """Blend the saved P01b test image logits with the new ratios; one table per test set. Refuses to run twice."""
    verify(manifest or manifest_path(study))
    out_dir.mkdir(parents=True)  # an existing folder means the test sets were already used
    blend = joblib.load(runs_directory(study) / "blend.joblib")
    measurements = pd.read_csv(_path(study, "measurements"))[["image_id", *RATIOS]]
    for name in TEST_PARTITIONS:
        saved = pd.read_csv(_path(study, "image_scores") / f"{name}.csv", float_precision="round_trip")
        table = saved.drop(columns=["blend_logit"]).merge(
            measurements, on="image_id", how="left", validate="one_to_one").assign(p01b_blend_logit=saved.blend_logit)
        table["blend_logit"] = blend_logits(blend, table)
        table.drop(columns=RATIOS).to_csv(out_dir / f"{name}.csv", index=False)


def _reference(study: dict, name: str, partition: str) -> pd.DataFrame:
    table = pd.read_csv(REPO_ROOT / study["references"][name])
    table = table[table.split == partition]
    return table[table.source == "predicted"] if "source" in table else table


def analyse_study(study: dict, scores_dir: Path, out: Path) -> dict[str, pd.DataFrame]:
    """The blend minus the first reference (primary, the best single-input measurement result) and minus the image
    score, the P01b blend and the other references (secondary), per test set, on the images both have scored."""
    out.mkdir(parents=True, exist_ok=True)
    names = list(study["references"])
    primary, secondary = [], []
    for partition in TEST_PARTITIONS:
        scores = pd.read_csv(scores_dir / f"{partition}.csv")
        for name in names:
            shared = scores.merge(_reference(study, name, partition)[["image_id", "score"]], on="image_id")
            row = {"n": len(shared), **compare_scores(f"{study['name']} blend minus {name}", partition, shared,
                                                      "blend_logit", "score", study)}
            (primary if name == names[0] else secondary).append(row)
        for label, column in (("image score", "image_logit"), ("p01b blend", "p01b_blend_logit")):
            secondary.append({"n": len(scores), **compare_scores(f"{study['name']} blend minus {label}", partition,
                                                                 scores, "blend_logit", column, study)})
    tables = {"primary": pd.DataFrame(primary), "secondary": pd.DataFrame(secondary)}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", type=Path)
    parser.add_argument("action", choices=["fit", "freeze", "score", "analyse"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    study = load_study(args.study)
    if args.action == "fit":
        print(json.dumps(fit_study(study), indent=2))
    elif args.action == "freeze":
        print(freeze_study(study)["git_commit"])
    elif args.action == "score":
        score_test_sets(study, args.out or runs_directory(study) / "final")
    else:
        print(analyse_study(study, args.scores, args.out)["primary"].to_string(index=False))


if __name__ == "__main__":
    main()
