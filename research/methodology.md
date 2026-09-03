# Mathematical Methodology and Algorithmic Formulations

## 1. Problem Formulation & Temporal Structure

Let $y_t \in \mathbb{R}_{+}$ denote the aggregated electrical demand (in kWh or kW) for a designated consumer cohort at discrete half-hour index $t \in \{1, \dots, T\}$.

At forecast origin $t_0$, the objective is to generate day-ahead probabilistic forecasts across horizons $h \in \{1, 2, \dots, 48\}$, representing target timestamps $t = t_0 + h$.

The feature vector available at forecast origin $t_0$ for target time $t_0 + h$ is denoted $\mathbf{x}_{t_0, h} \in \mathbb{R}^d$. 
Strict temporal non-leakage dictates:
$$\mathbf{x}_{t_0, h} = f(\{y_\tau\}_{\tau \le t_0}, \mathbf{z}_{t_0 + h}^{\text{cal}})$$
where $\{y_\tau\}_{\tau \le t_0}$ is historical load observed strictly on or before $t_0$, and $\mathbf{z}_{t_0 + h}^{\text{cal}}$ represents deterministic calendar variables known in advance for target time $t_0 + h$.

---

## 2. Base Forecasting Models

### 2.1 Point Forecasting
- **B0: Seasonal Naive**:
  $$\hat{y}_{t_0+h}^{\text{day}} = y_{t_0 + h - 48}, \quad \hat{y}_{t_0+h}^{\text{week}} = y_{t_0 + h - 336}$$
- **B1: Ridge Regression**:
  $$\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \sum_{i} \left( y_i - \mathbf{w}^\top \mathbf{x}_i \right)^2 + \lambda \|\mathbf{w}\|_2^2$$
- **B2: Histogram Gradient-Boosted Trees (HistGBR)**:
  Minimizes mean squared error $\mathcal{L}(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$ via second-order tree boosting on binned continuous features.

### 2.2 Quantile Regression (B3 & P0)
For nominal coverage $1 - \alpha$, lower quantile $\tau_{\text{lo}} = \alpha/2$ and upper quantile $\tau_{\text{hi}} = 1 - \alpha/2$ (e.g., $\tau \in \{0.05, 0.50, 0.95\}$ for $90\%$ nominal coverage):
$$\mathcal{L}_{\tau}(y, \hat{q}) = \max\left(\tau (y - \hat{q}), (\tau - 1)(y - \hat{q})\right) = (y - \hat{q})\left(\tau - \mathbb{I}(y < \hat{q})\right)$$
Quantile models $\hat{q}_{\tau_{\text{lo}}}(\mathbf{x})$ and $\hat{q}_{\tau_{\text{hi}}}(\mathbf{x})$ are trained directly using the pinball loss.

---

## 3. Conformal Prediction Formulations

### 3.1 P1: Static Split Conformal Prediction
Given a fixed calibration set $\mathcal{D}_{\text{cal}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n_{\text{cal}}}$, compute nonconformity scores:
$$s_i = |y_i - \hat{\mu}(\mathbf{x}_i)|$$
The conformal quantile at nominal level $1 - \alpha$ is:
$$\hat{Q}_{1-\alpha}(s) = \text{Quantile}\left(\{s_i\}_{i=1}^{n_{\text{cal}}}, \frac{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil}{n_{\text{cal}}}\right)$$
Prediction interval for a test sample $\mathbf{x}_{t}$:
$$\mathcal{C}(\mathbf{x}_t) = \left[ \hat{\mu}(\mathbf{x}_t) - \hat{Q}_{1-\alpha}(s), \; \hat{\mu}(\mathbf{x}_t) + \hat{Q}_{1-\alpha}(s) \right]$$

### 3.2 P2: Conformalized Quantile Regression (CQR)
Given base quantile estimators $\hat{q}_{\alpha/2}(\mathbf{x})$ and $\hat{q}_{1-\alpha/2}(\mathbf{x})$, define the nonconformity score on $\mathcal{D}_{\text{cal}}$:
$$s_i = \max\left(\hat{q}_{\alpha/2}(\mathbf{x}_i) - y_i, \; y_i - \hat{q}_{1-\alpha/2}(\mathbf{x}_i)\right)$$
If $y_i$ falls strictly within $[\hat{q}_{\alpha/2}(\mathbf{x}_i), \hat{q}_{1-\alpha/2}(\mathbf{x}_i)]$, $s_i \le 0$; otherwise $s_i > 0$ represents the absolute boundary violation.
The calibrated prediction interval is:
$$\mathcal{C}(\mathbf{x}_t) = \left[ \hat{q}_{\alpha/2}(\mathbf{x}_t) - \hat{Q}_{1-\alpha}(s), \; \hat{q}_{1-\alpha/2}(\mathbf{x}_t) + \hat{Q}_{1-\alpha}(s) \right]$$

### 3.3 P3: Rolling-Window Conformal Recalibration
Rather than fixing $\mathcal{D}_{\text{cal}}$, maintain a trailing FIFO buffer $\mathcal{S}_t = \{s_{t-W}, \dots, s_{t-1}\}$ of length $W$ comprising the most recently observed nonconformity scores. At step $t$:
$$\hat{Q}_{t, 1-\alpha} = \text{Quantile}\left(\mathcal{S}_t, \frac{\lceil (W + 1)(1 - \alpha) \rceil}{W}\right)$$

### 3.4 P4: Adaptive Conformal Inference (ACI)
Introduced by Gibbs & Candès (2021), ACI dynamically updates the target miscoverage level $\alpha_t \in (0, 1)$ at each time step based on observed binary coverage errors.
Let $\text{err}_t = \mathbb{I}\left(y_t \notin \mathcal{C}_t(\alpha_t)\right)$ denote the miscoverage indicator. The sequential update rule is:
$$\alpha_{t+1} = \text{clip}\left(\alpha_t + \gamma (\alpha - \text{err}_t), \; \alpha_{\min}, \; \alpha_{\max}\right)$$
- If coverage fails ($\text{err}_t = 1$), $\alpha_{t+1} = \alpha_t + \gamma(\alpha - 1) < \alpha_t$, reducing the target miscoverage level and thereby widening subsequent prediction intervals.
- If coverage succeeds ($\text{err}_t = 0$), $\alpha_{t+1} = \alpha_t + \gamma \alpha > \alpha_t$, incrementally increasing target miscoverage and tightening intervals.
- Parameters: learning rate $\gamma > 0$, bounds $[\alpha_{\min}, \alpha_{\max}] \subset (0, 1)$ (e.g., $[0.01, 0.50]$).

---

## 4. Proposed Method: Shift-Aware Adaptive Conformal Prediction (SA-ACP)

### 4.1 Motivation
Standard ACI applies a static adaptation rate $\gamma$. Under sudden structural or behavioral distribution shifts (e.g., extreme weather, dynamic tariff activations), small $\gamma$ results in prolonged under-coverage due to slow adaptation, while a globally large $\gamma$ induces high interval volatility during stationary periods.

SA-ACP introduces a continuous non-parametric distribution shift detector that monitors recent nonconformity scores relative to a stationary reference window, dynamically modulating $\gamma_t$ and responsive quantile offsets.

### 4.2 Shift Detector Formulation
Let $\mathcal{R} = \{s_1^{\text{ref}}, \dots, s_{N_{\text{ref}}}^{\text{ref}}\}$ denote the reference nonconformity scores from the validation/calibration window.
Let $\mathcal{W}_t = \{s_{t-W+1}, \dots, s_t\}$ denote the trailing window of $W$ recent test nonconformity scores.

We compute the 1-Wasserstein distance (Earth Mover's Distance) between the empirical distributions:
$$\mathcal{W}_1(\hat{P}_{\mathcal{W}_t}, \hat{P}_{\mathcal{R}}) = \int_{-\infty}^{\infty} |\hat{F}_{\mathcal{W}_t}(u) - \hat{F}_{\mathcal{R}}(u)| \, du$$
Normalized shift score:
$$\delta_t = \frac{\mathcal{W}_1(\hat{P}_{\mathcal{W}_t}, \hat{P}_{\mathcal{R}})}{\sigma_{\mathcal{R}} + \epsilon}$$
Binary shift flag with validation-calibrated threshold $\tau_{\text{shift}}$:
$$\text{ShiftFlag}_t = \mathbb{I}(\delta_t \ge \tau_{\text{shift}})$$

### 4.3 Adaptive Modulation Rule
$$\gamma_t = \begin{cases} 
\gamma_{\text{fast}} & \text{if } \text{ShiftFlag}_t = 1 \\ 
\gamma_{\text{slow}} & \text{if } \text{ShiftFlag}_t = 0 
\end{cases}$$
The miscoverage parameter $\alpha_t$ updates as:
$$\alpha_{t+1} = \text{clip}\left(\alpha_t + \gamma_t (\alpha - \text{err}_t) - \eta \cdot \delta_t \cdot \text{err}_t, \; \alpha_{\min}, \; \alpha_{\max}\right)$$
where $\eta \ge 0$ is an optional immediate shift penalty that sharpens interval expansion on miscoverage during active shifts.

```mermaid
flowchart LR
    A[Point / Quantile Forecast] --> B[Conformal Calibration Layer]
    B --> C[Prediction Interval [L_t, U_t]]
    C --> D[Realized Load y_t Observed]
    D --> E[Compute Nonconformity Score s_t]
    E --> F[Update Trailing Window W_t]
    F --> G[1-Wasserstein Shift Detector vs. Reference]
    G --> H{Shift Detected?}
    H -- Yes --> I[Set gamma_t = gamma_fast + Shift Penalty]
    H -- No --> J[Set gamma_t = gamma_slow]
    I --> K[Update alpha_t+1]
    J --> K[Update alpha_t+1]
    K --> B
```

---

## 5. Evaluation Metrics

### 5.1 Point Forecast Metrics
- **MAE**: $\frac{1}{N}\sum |y_t - \hat{y}_t|$
- **RMSE**: $\sqrt{\frac{1}{N}\sum (y_t - \hat{y}_t)^2}$
- **NMAE**: $\frac{\text{MAE}}{\frac{1}{N}\sum y_t}$
- **sMAPE**: $\frac{100\%}{N}\sum \frac{2|y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t| + 10^{-5}}$

### 5.2 Probabilistic / Conformal Metrics
- **Empirical Coverage**:
  $$\text{Cov} = \frac{1}{N}\sum_{t=1}^N \mathbb{I}(L_t \le y_t \le U_t)$$
- **Absolute Coverage Error (ACE)**:
  $$\text{ACE} = |\text{Cov} - (1 - \alpha)|$$
- **Mean Prediction Interval Width (MPIW)**:
  $$\text{MPIW} = \frac{1}{N}\sum_{t=1}^N (U_t - L_t)$$
- **Winkler Interval Score ($IS_\alpha$)**:
  $$IS_\alpha(L_t, U_t, y_t) = (U_t - L_t) + \frac{2}{\alpha}(L_t - y_t)\mathbb{I}(y_t < L_t) + \frac{2}{\alpha}(y_t - U_t)\mathbb{I}(y_t > U_t)$$

---

## 6. Statistical Inference: Paired Day-Level Block Bootstrap

To account for strong serial autocorrelation across half-hourly intervals within days, statistical comparison between methods (e.g., SA-ACP vs. ACI) is conducted using a **paired block bootstrap at the daily level**:
1. Group test-set predictions into $K$ daily blocks $\{D_1, D_2, \dots, D_K\}$, where each block contains 48 half-hourly observations.
2. For $b = 1, \dots, B$ (default $B = 2,000$):
   - Sample $K$ daily blocks with replacement: $\{D_{k_1}^*, \dots, D_{k_K}^*\}$.
   - Evaluate metric difference $\Delta M^{*(b)} = M(\text{Method}_1) - M(\text{Method}_2)$ over the concatenated resampled days.
3. Compute empirical mean difference $\bar{\Delta M}$, $95\%$ percentile confidence interval $[\Delta M_{0.025}^*, \Delta M_{0.975}^*]$, and bootstrap $p$-value.
