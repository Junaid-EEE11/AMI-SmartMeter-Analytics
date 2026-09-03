# Central Research Question and Objectives

## 1. Central Research Question

> **When residential smart-meter demand undergoes behavioral or temporal distribution shift, how badly do conventional point and probabilistic load forecasts lose calibration, and can adaptive conformal methods restore reliable uncertainty coverage without producing operationally useless prediction intervals?**

---

## 2. Pre-Declared Research Questions

### RQ1: Degradation Under Shift
*How much does forecast accuracy and prediction-interval calibration deteriorate under temporal or behavioral distribution shift?*
- **Primary Endpoint**: Change in Absolute Coverage Error (ACE) and Winkler Interval Score between stationary baseline periods and periods of structural/temporal shift.
- **Diagnostic Metrics**: Empirical coverage at nominal $90\%$ confidence ($1-\alpha=0.90$), Mean Prediction Interval Width (MPIW), and root mean square error (RMSE).

### RQ2: Cohort Disparity (dToU vs. Standard Tariff)
*Are calibration failures larger for the dynamic Time-of-Use (dToU) cohort than for the standard-tariff cohort during relevant shifted periods?*
- **Investigation**: Compare coverage degradation between the $\approx 1,100$ dToU cohort customers and the standard tariff comparison cohort across the 2012–2013 transition.
- **Scientific Guardrail**: Differences are analyzed as forecasting under regime change and behavioral variance; they are NOT automatically asserted as causal treatment effects without formal identification assumptions.

### RQ3: Rolling vs. Static Conformal Calibration
*Can rolling or adaptive conformal methods maintain nominal coverage better than static conformal prediction after a shift?*
- **Comparison**: Static split conformal prediction (fixed calibration window) vs. sliding rolling-window conformal recalibration vs. Adaptive Conformal Inference (ACI).

### RQ4: Shift-Aware Adaptive Conformal Prediction (SA-ACP) Trade-Off
*Does the proposed Shift-Aware Adaptive Conformal Prediction (SA-ACP) method improve the trade-off between empirical coverage and interval width compared with standard Adaptive Conformal Inference (ACI)?*
- **Investigation**: Compare SA-ACP (which dynamically adjusts adaptation step size $\gamma$ based on a sequential non-parametric shift statistic) against standard ACI with fixed $\gamma$.

### RQ5: Systematic Failure Characterization
*Where do all conformal and probabilistic forecasting methods fail?*
- **Failure Analysis**: Characterize breakdown modes conditional on:
  1. Forecast horizon ($h \in \{1, \dots, 48\}$ half-hours ahead),
  2. Peak demand hours vs. off-peak baseload,
  3. Extreme load ramps ($\Delta y_t$),
  4. Seasonal transitions (winter cold snaps vs. mild spring/autumn),
  5. Cohort behavioral volatility,
  6. Shift detector false alarms vs. missed shifts.

---

## 3. Operational Relevance to Power Distribution Systems

Smart meter (AMI) aggregated forecasting is foundational to low-voltage distribution management:
- **Transformer Overload Prevention**: Feeder/cohort forecasts inform dynamic thermal ratings and loading headroom.
- **Demand Response Readiness**: Dynamic tariff response induces load displacement and rebound peaks; uncalibrated forecasting risks severe capacity shortfalls.
- **Reserve and Storage Dispatch**: Operating reserves sized on nominal $90\%$ prediction intervals fail if true coverage drops to $70\%$.
