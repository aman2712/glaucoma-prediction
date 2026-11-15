# Frozen EfficientNet test evaluation — 15 November 2026

E2 was selected using the six-method development study and its grouped E0/E2 split-sensitivity comparison. The held-out evaluation uses the original 808/200-trained E2 checkpoints, not the cross-validation models. Three seeds (2440–2442) are combined by the mean of probability logits, clipping probabilities to [1e-7, 1-1e-7] consistently for validation and test inference.

Primary comparator: a newly evaluated, matched three-seed E0 ResNet-18 ensemble with the same original development split and seeds. Secondary comparator: the historical single-seed ResNet baseline's already saved scores. This separates architecture effects from simply adding ensemble members.

Thresholds were selected on the original 200 CHAKSU validation images, maximizing specificity subject to sensitivity >=90%: E0 ensemble 0.016863073282639456; E2 ensemble 0.023144414657426053. The historical baseline retains its original threshold 0.00786661822348833. No threshold or checkpoint is selected from either test set.

The protocol binds checkpoints, saved validation predictions, source code and the common split by SHA-256 before inference. It records prior test exposure explicitly: both cohorts have already been evaluated by earlier baselines/P01. Consequently new test results are exploratory reused-test evidence and do not establish unexposed external confirmation or clinical readiness.

Evaluation job: 300953. Study directory on HPC: `hpc_experiments/runs/efficientnet_heldout_20261115/`. Its `protocol.json` is written before test pixels are opened; `evaluation/completion.json` is written only after both partitions finish. The evaluator refuses to overwrite the evaluation directory. No training code is called during inference.

Report CHAKSU (335 images, 51 positives) and PAPILA (420 images, 87 positives) separately. Primary difference: E2 ensemble minus E0 ensemble AUROC, with 2,000 paired analysis-group bootstrap resamples, conditional on the trained checkpoints. Per-arm intervals additionally cover AP, sensitivity, specificity, balanced accuracy and Brier score. Use identical bootstrap indices for paired comparisons. Known group dependence is preserved; unknown CHAKSU identities remain unresolved.

The measurement baseline's common-split published results use segmentation-eligible classification cohorts (326 CHAKSU images/50 positives; 418 PAPILA images/85 positives), Youden threshold selection and a different training procedure. If its original per-image scores cannot be obtained, label its summary numbers as descriptive historical results and avoid a paired significance claim. Re-evaluating image scores on the same eligibility subset can improve cohort comparability but cannot reconstruct the missing baseline scores.

Saved predictions and all checkpoint/source artifacts remain in ignored HPC storage. Publish aggregate metric tables, confidence intervals, paired differences and diagnostic curves after verifying completion. Do not publish photographs or image-level identifiers by default.
