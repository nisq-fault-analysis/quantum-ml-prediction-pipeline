# Grouped Release Ablation Summary

## Leakage Audit
- Grouped split passed: True
- Group column: `base_circuit_id`
- Overlapping groups found: 0

## Best Model By Ablation
- both: XGBoost Regressor (validation R2=0.8519, test R2=0.8579, test MAE=0.0393)
- both_with_local_features: XGBoost Regressor (validation R2=0.8599, test R2=0.8653, test MAE=0.0387)
- both_without_local_features: XGBoost Regressor (validation R2=0.8615, test R2=0.8668, test MAE=0.0384)
- raw_only: XGBoost Regressor (validation R2=0.9279, test R2=0.9250, test MAE=0.0258)
- transpiled_only: XGBoost Regressor (validation R2=0.8284, test R2=0.8356, test MAE=0.0502)

## Diagnostics
- Local-feature gain diagnostic available: True
- Routing-sensitive test delta with local features: delta R2=0.0000, delta MAE=-0.0000

## Worst Slices
- family=qaoa_like (n=1380): MAE=0.0398, R2=0.9273
- family=entanglement_heavy (n=1256): MAE=0.0384, R2=0.9378
- family=trotter_ising (n=1378): MAE=0.0356, R2=0.8761
- family=mixed_random (n=781): MAE=0.0328, R2=0.6697
- qubit_count=6 (n=2282): MAE=0.0324, R2=0.9127