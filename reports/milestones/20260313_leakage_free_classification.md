# Leakage-Free Classification Milestone

## Metadata
- Report timestamp: 2026-03-13T23:38:03
- Dataset used: data/raw/NISQ-FaultLogs-100K.csv
- Split strategy: Single deterministic 80/15/5 train/validation/test split shared across compared models.
- Experiment scope: Global leakage-free fault classification, qubit-count-stratified classification, fidelity regression reference, SHAP interpretation, and targeted subgroup tuning.
- Subgroup: qubit_count
- Run group: 2026-03-13 leakage-free classification and regression milestone
- Report config: `reports\milestone_configs\20260313_leakage_free_classification.yaml`

## What Was Compared
- Models compared: Dummy Most Frequent, Logistic Regression, Random Forest, XGBoost, Random Forest Regressor, XGBoost Regressor
- Feature sets compared: baseline_raw, enhanced_topology
- Tuning was run: yes
- SHAP was run: yes
- Prediction contexts: pre_execution
- Excluded feature columns: fidelity, fidelity_loss, bit_errors, observed_error_rate, bit_error_density, timestamp
- Classification uses the leakage-free pre_execution feature policy.
- Regression is included as a reference result rather than the thesis headline for fault-type classification.

## Best Raw Results
- Best global validation result: XGBoost (validation macro-F1 0.4992, test macro-F1 0.5072, source `experiments\model_benchmark\20260313_221036\model_comparison.csv`)
- Best global held-out test result: XGBoost (validation macro-F1 0.4992, test macro-F1 0.5072, source `experiments\model_benchmark\20260313_221036\model_comparison.csv`)
- Best stratified validation result: Random Forest, subgroup qubit_count = 8 (validation macro-F1 0.5261, test macro-F1 0.5435, source `experiments\qubit_stratified\20260313_223849\best_model_by_qubit_count.csv`)
- Best stratified held-out test result: XGBoost, subgroup qubit_count = 6 (validation macro-F1 0.5216, test macro-F1 0.5493, source `experiments\qubit_stratified\20260313_223849\best_model_by_qubit_count.csv`)
- Regression reference: Random Forest Regressor (validation R2 0.9410, test R2 0.9412, test MAE 0.0429, source `experiments\fidelity_regression\20260313_203701\model_comparison.csv`)

Subgroup winners
| Subgroup | Model | Validation macro-F1 | Test macro-F1 |
| --- | --- | ---: | ---: |
| qubit_count = 5 | Random Forest | 0.5135 | 0.4767 |
| qubit_count = 6 | XGBoost | 0.5216 | 0.5493 |
| qubit_count = 7 | Random Forest | 0.5071 | 0.4839 |
| qubit_count = 8 | Random Forest | 0.5261 | 0.5435 |
| qubit_count = 9 | Logistic Regression | 0.5119 | 0.5202 |
| qubit_count = 10 | Logistic Regression | 0.5068 | 0.5037 |
| qubit_count = 11 | XGBoost | 0.5244 | 0.4995 |
| qubit_count = 12 | XGBoost | 0.4814 | 0.4673 |
| qubit_count = 13 | XGBoost | 0.5036 | 0.5066 |
| qubit_count = 14 | XGBoost | 0.5076 | 0.5197 |
| qubit_count = 15 | Logistic Regression | 0.4971 | 0.4448 |

Tuning comparisons
- qubit_count = 6: untuned test macro-F1 0.5493 vs tuned 0.5401; validation delta 0.0044, test delta -0.0092. Validation improved but held-out test performance worsened, which is consistent with overfitting during tuning.
- qubit_count = 8: untuned test macro-F1 0.5435 vs tuned 0.5152; validation delta 0.0153, test delta -0.0283. Validation improved but held-out test performance worsened, which is consistent with overfitting during tuning.

SHAP highlights
- SHAP explanation: XGBoost on the test split. Top features: t2_time (0.049), t1_time (0.044), error_rate_gate (0.042), readout_error (0.037), gate_depth (0.030). Source `experiments\model_benchmark\20260313_221036\shap_analysis\shap_feature_importance.csv`
- SHAP explanation for qubit_count = 8: Random Forest on the test split. Top features: readout_error (0.011), t2_time (0.010), error_rate_gate (0.010), t1_time (0.009), gate_depth (0.008). Source `experiments\qubit_stratified\20260313_223849\qubit_count_8\shap_analysis\shap_feature_importance.csv`
- SHAP explanation for qubit_count = 6: XGBoost on the test split. Top features: t1_time (0.129), gate_depth (0.127), readout_error (0.122), t2_time (0.113), error_rate_gate (0.111). Source `experiments\qubit_stratified\20260313_223849\qubit_count_6\shap_analysis\shap_feature_importance.csv`

## Main Scientific Takeaway
- Plain-language conclusion: Leakage-free global fault classification remains weak, but qubit-count-stratified models recover stronger signal, while fidelity regression stays strong.
- Scientific meaning: The public snapshot appears to carry useful hardware and noise information, but not enough topology diversity for a single strong global fault classifier. Regime-specific models by qubit_count are therefore the more defensible scientific result.

Negative results to preserve
- Global leakage-free fault classification stayed near 0.51 macro-F1, so it should not be framed as a strong general classifier.
- Targeted tuning improved validation scores for q=6 and q=8 but reduced held-out test macro-F1 in both subgroups.

## Important Caveats
- All classification claims in this report are leakage-free pre-execution claims only; post-observation columns such as fidelity and bit-error-derived fields were excluded.
- The public Kaggle snapshot appears to have constant gate_types, which limits topology variation and weakens claims about gate-sequence signal.
- Stratified subgroup results come from smaller per-qubit subsets than the global benchmark and should be described as regime-specific evidence.
- The best validation subgroup and the strongest held-out test subgroup are different, so model selection and retrospective comparison must be kept separate in the thesis.

## Methodological Warnings
- Do not present a validation winner as automatically identical to the best thesis headline result.
- Do not treat tuned subgroup runs as the main result here because they improved validation but hurt held-out generalization.

## Thesis Framing Recommendation
- Recommended headline result: Use qubit_count = 8 as the headline subgroup because it won under the planned validation criterion and remained strong on held-out test.
- More trustworthy result: Treat the untuned q=8 Random Forest subgroup winner as more trustworthy than the tuned alternatives because tuning hurt held-out generalization.
- Held-out test comparator: Use qubit_count = 6 as the strongest held-out comparator because its untuned XGBoost model achieved the highest subgroup test macro-F1.
- Present the global leakage-free result as an intentionally strict baseline that remained weak rather than as a failure to find any signal.
- Contrast the weak global classification result with the strong fidelity regression baseline to show that the dataset still contains meaningful structure.
- State explicitly that SHAP highlighted hardware, noise, and depth features more strongly than topology-style variation in this public snapshot.

## Recommended Next Steps
- Keep the untuned q=8 subgroup as the current thesis headline unless a new dataset with richer topology variation changes the ranking.
- If more experiments are needed, prioritize new data with non-constant gate_types over deeper hyperparameter search.
- Regenerate the milestone report after any new dataset snapshot, feature policy change, or materially new subgrouping strategy.

## Thesis Reuse Sentences
- Under a leakage-free pre-execution feature policy, global fault-type classification remained weak, but qubit-count-stratified models recovered stronger and more scientifically interpretable signal.

## Artifact References
- Global classification comparison table: `experiments\model_benchmark\20260313_221036\model_comparison.csv`
- Global classification feature policy: `experiments\model_benchmark\20260313_221036\feature_policy.json`
- Global classification split summary: `experiments\model_benchmark\20260313_221036\split_summary.json`
- Stratified best model by qubit count: `experiments\qubit_stratified\20260313_223849\best_model_by_qubit_count.csv`
- Stratified full comparison table: `experiments\qubit_stratified\20260313_223849\qubit_model_comparison.csv`
- Stratified feature policy by qubit count: `experiments\qubit_stratified\20260313_223849\feature_policy_by_qubit.json`
- Stratified subset metadata by qubit count: `experiments\qubit_stratified\20260313_223849\subset_metadata_by_qubit.json`
- Fidelity regression comparison table: `experiments\fidelity_regression\20260313_203701\model_comparison.csv`
- Tuned comparison table for qubit_count = 6: `experiments\tuned_classification\20260313_225202\tuned_model_comparison.csv`
- Tuned subset metadata for qubit_count = 6: `experiments\tuned_classification\20260313_225202\subset_metadata.json`
- Tuned comparison table for qubit_count = 8: `experiments\tuned_classification\20260313_225237\tuned_model_comparison.csv`
- Tuned subset metadata for qubit_count = 8: `experiments\tuned_classification\20260313_225237\subset_metadata.json`
- SHAP explanation feature importance: `experiments\model_benchmark\20260313_221036\shap_analysis\shap_feature_importance.csv`
- SHAP explanation metadata: `experiments\model_benchmark\20260313_221036\shap_analysis\shap_metadata.json`
- SHAP explanation for qubit_count = 8 feature importance: `experiments\qubit_stratified\20260313_223849\qubit_count_8\shap_analysis\shap_feature_importance.csv`
- SHAP explanation for qubit_count = 8 metadata: `experiments\qubit_stratified\20260313_223849\qubit_count_8\shap_analysis\shap_metadata.json`
- SHAP explanation for qubit_count = 6 feature importance: `experiments\qubit_stratified\20260313_223849\qubit_count_6\shap_analysis\shap_feature_importance.csv`
- SHAP explanation for qubit_count = 6 metadata: `experiments\qubit_stratified\20260313_223849\qubit_count_6\shap_analysis\shap_metadata.json`
- Additional artifact: checklist.md: `checklist.md`
