# Leakage-Free Reliability Regression

## Target
- Target name: reliability
- Definition: `reliability = 1 - (bit_errors / qubit_count)`
- Target bounds: `[0, 1]`
- Observed bitstring column used only for target construction: `bitstring_aligned`
- Ideal bitstring column used only for target construction: `ideal_bitstring_aligned`
- Row count: 99967
- Reliability mean: 0.5359
- Reliability median: 0.5385
- Reliability min/max: 0.0000 / 1.0000

## Leakage-Free Feature Policy
- Prediction context: pre-execution
- Allowed raw features: qubit_count, gate_depth, error_rate_gate, t1_time, t2_time, readout_error, device_type
- Allowed engineered pre-run features: num_cx, two_qubit_ratio, unique_gates, cx_density, t2_t1_ratio
- Used feature columns: qubit_count, gate_depth, error_rate_gate, t1_time, t2_time, readout_error, device_type, cx_density, t2_t1_ratio
- Dropped zero-variance columns: num_cx, two_qubit_ratio, unique_gates
- Forbidden columns present in source feature table: fidelity, bit_errors, observed_error_rate
- Other non-allowed columns present in source feature table: shots, fidelity, bit_errors, observed_error_rate

## Results
- Best validation model: Dummy Mean
- Best validation R2: -0.0002
- Best held-out test R2: -0.0002
- Best held-out test MAE: 0.1421

| Model | Validation R2 | Test R2 | Validation MAE | Test MAE |
| --- | ---: | ---: | ---: | ---: |
| Dummy Mean | -0.0002 | -0.0002 | 0.1410 | 0.1421 |
| Random Forest Regressor | -0.0004 | 0.0010 | 0.1410 | 0.1421 |
| XGBoost Regressor | -0.0056 | -0.0035 | 0.1410 | 0.1421 |

## Notes
- Post-run columns such as `bitstring`, `ideal_bitstring`, `fidelity`, `bit_errors`, `observed_error_rate`, and target-derived fields are excluded from model inputs.
- Root-level `metrics.json`, `model.joblib`, and plots correspond to the best validation model for quick thesis reuse.
- Per-model subdirectories preserve the full comparison suite.