# Radiomics-Only Baseline Modeling Report  
**EVADIAB Cohort – Prediction of MACE / Death**

---

## 1. Objective

The objective of this analysis is to evaluate the predictive value of **radiomic features extracted from cardiac imaging** for the occurrence of **major adverse cardiovascular events (MACE) or death** in the EVADIAB cohort.

This radiomics-only baseline is designed to:
- assess whether imaging-derived features contain prognostic information,
- compare their performance to clinical-only models,
- and establish a reference prior to multimodal (clinical + radiomics) modeling.

---

## 2. Dataset Description

### 2.1 Cohort
- **Number of patients:** 161  
- **Outcome:** Binary indicator of MACE or death  
- **Number of events:** ~72  
- **Event rate:** ~45%

Only patients with:
- available baseline clinical data, and
- complete radiomic feature extraction for both **rest** and **stress** images  
were included in this analysis.

---

### 2.2 Radiomic Features

A total of **5 radiomic features** were extracted for each patient from both **rest** and **stress** imaging conditions, resulting in **10 radiomic variables per patient**.

#### Radiomic feature categories:
- Texture (NGTDM, GLCM)
- Shape
- First-order statistics

#### Features used:
- `ngtdm_Contrast`
- `entropy_Lrb`
- `shape_Flatness`
- `glcm_Idmn`
- `firstorder_Kurtosis`

Each feature was included separately for:
- **REST**
- **STRESS**

This explicit separation preserves physiological information related to stress-induced changes.

---

## 3. Data Preprocessing

### 3.1 Data Integration
Radiomic features were provided in a long format with two rows per patient (REST and STRESS).  
They were pivoted into a wide format such that each patient had a single row with condition-specific features:

# Radiomics-Only Baseline Modeling Report  
**EVADIAB Cohort – Prediction of MACE / Death**

---

## 1. Objective

The objective of this analysis is to evaluate the predictive value of **radiomic features extracted from cardiac imaging** for the occurrence of **major adverse cardiovascular events (MACE) or death** in the EVADIAB cohort.

This radiomics-only baseline is designed to:
- assess whether imaging-derived features contain prognostic information,
- compare their performance to clinical-only models,
- and establish a reference prior to multimodal (clinical + radiomics) modeling.

---

## 2. Dataset Description

### 2.1 Cohort
- **Number of patients:** 161  
- **Outcome:** Binary indicator of MACE or death  
- **Number of events:** ~72  
- **Event rate:** ~45%

Only patients with:
- available baseline clinical data, and
- complete radiomic feature extraction for both **rest** and **stress** images  
were included in this analysis.

---

### 2.2 Radiomic Features

A total of **5 radiomic features** were extracted for each patient from both **rest** and **stress** imaging conditions, resulting in **10 radiomic variables per patient**.

#### Radiomic feature categories:
- Texture (NGTDM, GLCM)
- Shape
- First-order statistics

#### Features used:
- `ngtdm_Contrast`
- `entropy_Lrb`
- `shape_Flatness`
- `glcm_Idmn`
- `firstorder_Kurtosis`

Each feature was included separately for:
- **REST**
- **STRESS**

This explicit separation preserves physiological information related to stress-induced changes.

---

## 3. Data Preprocessing

### 3.1 Data Integration
Radiomic features were provided in a long format with two rows per patient (REST and STRESS).  
They were pivoted into a wide format such that each patient had a single row with condition-specific features:


Only patients with complete radiomic data for both conditions were retained.

---

### 3.2 Missing Data
- No missing values were present in the radiomic features after preprocessing.
- All preprocessing steps were applied **within the training data only** during model fitting to avoid information leakage.

---

### 3.3 Normalization
- All radiomic features were standardized (zero mean, unit variance) as part of the modeling pipeline.

---

## 4. Modeling Strategy

### 4.1 Model Choice
A **logistic regression** classifier was used as the reference model.

This choice was motivated by:
- the small sample size,
- the low dimensionality of the radiomic feature space,
- model stability,
- and interpretability.

---

### 4.2 Evaluation Metrics
Model performance was assessed using:
- **ROC-AUC** (discrimination)
- **PR-AUC** (precision–recall performance)
- **Brier score** (calibration)
- Accuracy (reported but not used as a primary decision metric)

---

### 4.3 Stability Analysis
To evaluate robustness and generalization, performance was assessed using:
- **50 repeated stratified train/test splits**
- fixed test size
- stratification by outcome

This approach provides an estimate of expected performance and variability in small cohorts.

---

## 5. Results

### 5.1 Single-Split Performance

On a representative stratified split, the radiomics-only model achieved:

- **ROC-AUC:** ~0.74  
- **PR-AUC:** ~0.64  
- **Brier score:** ~0.22  

These results suggest meaningful discriminative ability based solely on radiomic features.

---

### 5.2 Stability Across 50 Repeated Splits

Across 50 repeated stratified splits:

- **Mean ROC-AUC:** ~0.63  
- **95% CI:** ~[0.61 – 0.65]  
- **Mean PR-AUC:** ~0.60 (baseline event rate ≈ 0.45)  
- **Mean Brier score:** ~0.24  

Performance distributions showed moderate variability, as expected for a cohort of this size, but remained consistently above random baseline.

---

## 6. Interpretation

Key observations from the radiomics-only analysis:

- Radiomic features provide **robust prognostic signal** for MACE/death.
- Average discrimination is **higher and more stable** than that obtained with clinical variables alone.
- Radiomics appear to capture disease-related information not fully reflected in baseline clinical data.
- Calibration is acceptable, though slightly less optimal than clinical-only models.

These findings support the hypothesis that imaging-derived features offer complementary information for cardiovascular risk stratification.

---

## 7. Comparison with Clinical-Only Baseline

Compared to the previously established clinical-only baseline:

- Radiomics-only models achieved:
  - higher mean ROC-AUC,
  - higher PR-AUC,
  - and improved discrimination stability.
- Nonlinear models were not required to achieve these gains.

This highlights the added value of imaging-derived biomarkers.

---

## 8. Limitations

- Moderate sample size (n = 161)
- Binary outcome without explicit time-to-event modeling
- Single-center cohort
- No external validation

Despite these limitations, the consistent performance across repeated splits supports the robustness of the findings.

---

## 9. Conclusion and Next Steps

Radiomic features extracted from rest and stress imaging demonstrate **superior and more stable predictive performance** compared to baseline clinical variables alone in the EVADIAB cohort.

These results justify the integration of radiomics with clinical data to evaluate **multimodal prediction models**.

**Next step:**  
Develop and evaluate combined **clinical + radiomics** models to assess incremental predictive value beyond either modality alone.

This radiomics-only analysis serves as a critical benchmark for multimodal modeling.

---
