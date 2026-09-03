# Pre-Registered Hypotheses

*Status: Formally registered prior to test-set evaluation and empirical model fitting.*
*Guiding Rule: The research pipeline must never rewrite or alter these hypotheses based on observed experiment outcomes.*

---

## Hypothesis 1 (H1: Static Miscalibration Under Shift)
> **Static prediction intervals calibrated on pre-shift data will exhibit significantly larger Absolute Coverage Error (ACE) during high-shift or regime-change periods than during relatively stable stationary periods.**

- **Rationale**: Standard split conformal prediction assumes exchangeability of calibration and test residuals. Distribution drift (tariff response, seasonal transition, temperature extremes) violates exchangeability, causing systematic under-coverage.
- **Verification Metric**: $\text{ACE}_{\text{shift}} > \text{ACE}_{\text{stable}}$ for static conformal prediction ($p < 0.05$ via paired block bootstrap).

---

## Hypothesis 2 (H2: Adaptive Coverage Restoration)
> **Adaptive conformal prediction approaches (Rolling Conformal, ACI, and SA-ACP) will significantly reduce Absolute Coverage Error under distribution shift relative to fixed static conformal calibration.**

- **Rationale**: Sequential feedback mechanisms adjust critical nonconformity quantiles or miscoverage parameters $\alpha_t$ in response to observed prediction errors, recovering nominal coverage over time.
- **Verification Metric**: $\text{ACE}_{\text{adaptive}} < \text{ACE}_{\text{static}}$ evaluated over the full out-of-distribution test horizon.

---

## Hypothesis 3 (H3: Coverage-Width Trade-Off)
> **Restoring valid empirical coverage under distribution shift will involve a fundamental operational trade-off: improved coverage (lower ACE) will require an increase in Mean Prediction Interval Width (MPIW) and higher Winkler Interval Score during high-uncertainty regimes.**

- **Rationale**: When underlying load variance expands or predictability drops, narrower intervals cannot achieve $90\%$ coverage. Any valid calibration mechanism must expand interval widths to capture broadened error tails.
- **Verification Metric**: Strong negative correlation between coverage deficit and MPIW across methods.

---

## Hypothesis 4 (H4: SA-ACP Responsiveness Under Severe Shift)
> **The proposed Shift-Aware Adaptive Conformal Prediction (SA-ACP) method will provide its largest performance benefit (measured by lower ACE and faster recovery time without excessive average width) during periods identified as substantial residual/distribution shifts.**

- **Rationale**: Standard ACI uses a constant step-size $\gamma$. During abrupt shifts, small $\gamma$ adapts too slowly, while large $\gamma$ causes volatility during stationary regimes. SA-ACP dynamically modulates $\gamma_t$ based on a sequential non-parametric shift detector, enabling rapid expansion when a shift occurs and stabilization thereafter.
- **Verification Metric**: During high-shift sub-periods, $\text{ACE}_{\text{SA-ACP}} < \text{ACE}_{\text{ACI}}$ while maintaining competitive overall Winkler score.

---

## Hypothesis 5 (H5: Heterogeneity Across Load Regimes)
> **Differences between conformal methods will be highly heterogeneous across operational load regimes (e.g., peak demand hours vs. overnight baseload, high ramp events vs. flat profiles); therefore, aggregate summary metrics alone will conceal critical operational failure modes.**

- **Rationale**: Aggregate empirical coverage averages over 48 daily half-hours. High overnight coverage can mask severe under-coverage during evening peak hours when grid capacity is tightest.
- **Verification Metric**: Subgroup conditional evaluation will reveal significant conditional coverage deficits during peak and high-ramp hours even for methods showing $\approx 90\%$ marginal coverage.
