# Phase 1 model evaluation

The selected deployable model is **RandomForest**, trained using **pre_sep17** data. On its untouched final test set, Precision@300 was **0.383** (95% bootstrap CI 0.330–0.440); ROC-AUC was **0.589** and PR-AUC **0.302**. The High threshold is 0.6603, targeting 2.03% of calls.

The CI versus Level1_speed_limit_rate includes zero (-0.010 to 0.103). The CI versus Level1_5_smoothed_interaction_rate includes zero (-0.010 to 0.097).

The Sep-17 recommendation is **pre_sep17**, based on the held-out comparison in `sep17_sensitivity.csv`, not an assumed data-quality rule.

## Limitations

- STATS19 covers police-attended collisions with a casualty, not every incoming road-safety call.
- "Unknown" categories correlate with low severity partly because reporting and attendance differ, so the model may partly learn reporting behaviour.
- The five call-time features create many tied scores; reported Precision@K averages 20 deterministic tie-breaks.
- This observational dataset cannot show that enhanced response changes outcomes.
- These estimates are specific to 2023 UK data and need monitoring before operational use.
