# Scientific Assumptions and Research Hygiene

## 1. Primary Research Assumptions

1. **Forecasting Under Regime Change (Non-Causal Frame)**:
   The comparison between dToU (Dynamic Time-of-Use) and Standard tariff cohorts evaluates forecasting robustness and uncertainty calibration under behavioral variance. This study does **NOT** claim causal identification of price elasticities or causal difference-in-differences treatment effects, as tariff assignment in field trials may involve customer self-selection or unobserved co-interventions.

2. **Aggregation and Feeder Representation**:
   Cohort aggregates represent arithmetic sums of customer load profiles available in the trial. They must **NOT** be claimed as physical medium-voltage/low-voltage distribution feeder circuits unless explicit network topology, line impedances, and transformer mapping data are provided.

3. **Exogenous Information and Weather Hygiene**:
   Realized future weather data (temperature, solar irradiance) are **NOT** utilized in primary day-ahead forecast models to avoid oracle leakage. Any secondary experiments incorporating observed weather are explicitly labeled as `perfect_weather_sensitivity` oracle benchmarks.

4. **Sequential Operational Realism**:
   At forecast origin $t_0$, predictions for horizons $h \in \{1, \dots, 48\}$ are generated simultaneously using strictly lagged information $\{y_\tau\}_{\tau \le t_0}$. Adaptive conformal parameters are updated strictly after ground truth $y_{t_0+h}$ becomes observable in chronological order.

---

## 2. Leakage Prevention Rules

1. **Chronological Splitting**: Train $\to$ Calibration $\to$ Validation $\to$ Test partitions are strictly chronological. No random cross-validation is permitted.
2. **Preprocessing & Feature Scaling**: StandardScalers, imputation values, and categorical encodings are fitted strictly on the training partition and applied to downstream splits.
3. **Hyperparameter Selection**: Detector thresholds ($\tau_{\text{shift}}$) and adaptation learning rates ($\gamma, \gamma_{\text{fast}}, \gamma_{\text{slow}}$) are tuned exclusively on validation data, never on the final test set.
4. **Automated Leakage Testing**: Automated tests (`test_features_no_leakage.py`, `test_splits.py`, `test_sequential_updates.py`) verify the absence of future information in all feature pipelines.
