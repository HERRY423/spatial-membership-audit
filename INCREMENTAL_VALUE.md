# Incremental value checks: development use

`spatial_membership_audit.incremental_validation` supplies fixed shrinkage, train-only feature transforms, fixed-penalty fits, paired group resampling and a strict internal gain gate. It is a development evaluator, not a validated physical-depth corrector or biological warning system.

The completed local exercise is documented in [the full report](../../review/incremental_value_2026-10-26/report.md). A fixed half shrinkage gave a positive aggregate descriptive result in one historical specimen, with unfavorable category contrasts retained. Adding CV to a combined baseline did not establish stable internal prediction gain. Neither branch passed its complete prespecified internal gate; independent validation remains unestablished.

## Contracts

- `shrink_depth_retention(mean_a, mean_b, strength=0.5)` accepts aligned nonnegative gene means. It shrinks positive-gene log contrasts towards the panel-total contrast. Half shrinkage does not equalize every gene's expected count. Zero-boundary genes keep explicit original deletion semantics, including at strength=0; to obtain the original scalar on boundary genes use the separate scalar definition.
- `train_only_design(...)` estimates centering/scaling on training rows only. Classes must be declared; missing/nonfinite features are rejected rather than imputed. The caller must freeze specimen/region grouping and never split a dependent unit across train/test.
- `fit_predict_fixed(..., task="binary"|"continuous", penalty=1)` uses only training responses. It does not accept validation responses or select hyperparameters. Single-class binary training fails explicitly. Nonnegative continuous predictions are a declared behavior.
- `group_bootstrap_difference(...)` preserves all rows in each sampled group. Its interval conditions on the fitted model and observed groups; it is not automatically a patient interval or a simultaneous guarantee.
- `incremental_gate(...)` requires a favorable effect and interval, all specified strata improving and no coverage loss. It never upgrades an internal pass into external validation.

Retain no-processing/scalar and simple source-side baselines on identical records. Preserve finite zero, unestimable ratios, missing features, failed fits and all planned denominators. Report all comparisons and prior exposure. Numerical solver agreement does not establish scientific benefit.

No automatic CV threshold filtering or automatic depth correction is added to the CLI. A failed or unvalidated value gate supports reporting the diagnostic with its baselines, not silently switching to a favorable subgroup or fitted model.

## L75 follow-up: bounded model check

A subsequent [L75 diagnostic](../../review/l75_risk_resolution_2026-10-26/report.md) retained the original data and task, and selected penalties only within outer-training spatial folds for linear models and quadratic expansions of the numeric predictors. Pair identity was additive; pair-by-CV interactions were not tested. Adding CV changed pooled spatial Brier by +0.000194 and +0.004696, respectively; both improved only two of five spatial blocks. The quadratic model's local held-region improvement (−0.003967) is retained as an exploratory result, but neither family passed the original stability gate. These results support the current diagnostic-only use of CV; they are not an equivalence test or evidence that every CV model is ineffective. Original frozen results remain available. The complete predictions, tuning records, numerical repair and independent software checks are linked in the report.

## Reproduce the saved local example

The frozen [execution contract](../../review/incremental_value_2026-10-26/contract.md) defines the exact data, splits, seven model candidates, two numerical sensitivities, losses and budgets. The [full predictions](../../review/incremental_value_2026-10-26/science/risk_02/all_heldout_predictions.csv) and [depth records](../../review/incremental_value_2026-10-26/science/depth_01/all_paired_endpoints.csv) can be inspected without new fitting. The first failed assertion and its scope-only correction remain preserved. Do not rerun the frozen scripts into existing output folders or reuse the exhausted/remaining budget as authorization for a different study.
