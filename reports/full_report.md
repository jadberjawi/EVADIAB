# EVADIAB Multimodal Modeling Summary  
**Clinical-only vs Radiomics-only vs Clinical + Radiomics**  
*(Interpretation-focused report for internal thesis/workflow use)*

---

## 1. Goal

This work aimed to understand how well **baseline clinical variables** and **imaging-derived radiomic features** predict a binary outcome (**MACE or death**) in the EVADIAB cohort, and whether combining both modalities improves prediction.

The focus was on:
- establishing solid baselines,
- quantifying performance variability in a small cohort,
- interpreting what the results imply about each modality.

---

## 2. Cohorts and Inputs

### 2.1 Clinical-only cohort
- Clinical dataset size was approximately **n ≈ 167** originally.
- Clinical variables used included:
  - Demographics: age, sex
  - Labs/metabolic: HbA1c, creatinine, lipid profile, BMI
  - History/behavioral: smoking status, CAD history
- Label distribution was moderate imbalance (event rate ~0.44).

### 2.2 Radiomics cohort (matched subset)
- After matching patients with available radiomics in both conditions, the merged dataset used for radiomics and fusion analyses had:
  - **n = 161 patients**  
  - **event rate ≈ 0.45**
- Radiomics were provided for two imaging states per patient:
  - **REST**
  - **STRESS**
- 5 radiomic features per state → **10 radiomics features total per patient**:
  - `ngtdm_Contrast_{REST,STRESS}`
  - `entropy_Lrb_{REST,STRESS}`
  - `shape_Flatness_{REST,STRESS}`
  - `glcm_Idmn_{REST,STRESS}`
  - `firstorder_Kurtosis_{REST,STRESS}`

**Important design decision:** REST and STRESS were not treated as separate samples. They were used as two feature sets for the same patient.

---

## 3. Modeling and Evaluation Strategy

### 3.1 Models evaluated
Three baselines were evaluated:
1. **Clinical-only** logistic regression
2. **Radiomics-only** logistic regression (REST+STRESS features)
3. **Clinical + radiomics** logistic regression (early fusion)

Logistic regression was used because:
- the cohort is small,
- the number of features is low-to-moderate,
- it is stable and interpretable compared with more complex nonlinear models.

### 3.2 Metrics
Reported metrics:
- **ROC-AUC** (discrimination / ranking)
- **PR-AUC** (useful given event rate ~0.45)
- **Brier score** (probability calibration)
- Accuracy was logged but not used as a primary decision criterion.

### 3.3 Stability analysis
To quantify robustness:
- **50 repeated stratified train/test splits** were run for each setting.
- Performance distributions (histograms) were plotted.

This was essential because single-split results can be optimistic in small datasets.

---

## 4. Results (high-level)

### 4.1 Clinical-only
Clinical-only modeling showed **limited and unstable** predictive performance across splits.
- Some splits looked promising, but average generalization was modest.
- This implies that baseline clinical variables alone may be insufficient for reliable individualized risk prediction in this cohort, especially at this sample size.

**Interpretation:**
- Clinical variables contain signal, but it is weak relative to cohort variability.
- Performance sensitivity to data partitioning suggests either:
  - heterogeneity in patient subtypes, and/or
  - limited sample size for stable estimation.

### 4.2 Radiomics-only
Radiomics-only modeling (REST + STRESS radiomics) showed **stronger and more consistent** performance than clinical-only.
- Mean ROC-AUC was higher than clinical-only.
- PR-AUC was consistently above baseline (event rate).
- This suggests that radiomic descriptors encode prognostic information that is not fully captured by baseline clinical variables.

**Interpretation:**
- Radiomics likely capture structural/functional signatures visible in imaging (e.g., heterogeneity, texture patterns, morphological descriptors) that correlate with future adverse outcomes.

### 4.3 Clinical + Radiomics (early fusion)
The combined model achieved very strong performance on some individual splits (including an excellent single-split result).
However, across 50 splits:
- the average performance was **similar to radiomics-only**, not consistently better.
- calibration (Brier) did not consistently improve.

**Interpretation:**
- Radiomics appear to dominate predictive signal in this cohort.
- Clinical variables may be partially redundant with imaging-derived biomarkers, or too noisy/weak to add stable incremental value at this sample size.
- The occasional “very high” split performance suggests potential synergy for certain partitions, but the effect is not robust enough to claim consistent additive benefit.

---

## 5. Key Conclusions

1. **Clinical-only prediction is limited and unstable** in this cohort at n≈160–170.
2. **Radiomics-only provides a clear improvement** over clinical-only, with more stable discrimination.
3. **Clinical + radiomics did not yield consistent improvement over radiomics-only**, suggesting:
   - radiomics capture most of the usable signal, and/or
   - incremental clinical value is small relative to variability at this sample size.

---

## 6. Practical Implications for Next Steps

- For this dataset and current feature set, **radiomics are the most valuable modality**.
- If the goal is building a robust predictor, prioritize:
  - better radiomics quality control (repeatability),
  - possibly additional imaging-derived features,
  - or increasing sample size for stable gains.

Optional future checks (not required, but informative):
- **Delta radiomics**: (STRESS − REST) features may better represent physiological response to stress.
- **Calibration refinement**: if probabilities are used clinically, apply calibration methods and report calibration curves.
- **Interpretability**: compute odds ratios / feature importance for radiomics features to connect signals back to physiology.

---

## 7. Summary (one sentence)
In EVADIAB, **radiomics carry stronger and more stable predictive information than baseline clinical variables**, and combining clinical data with radiomics does not consistently improve performance beyond radiomics alone under repeated-split evaluation.

---
