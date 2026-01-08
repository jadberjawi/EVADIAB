# Clinical-Only Baseline Modeling Report  
**EVADIAB Cohort – Prediction of MACE / Death**

---

## 1. Objective

The objective of this analysis is to evaluate the predictive value of **baseline clinical variables alone** for the occurrence of an outcome of **major adverse cardiovascular events (MACE) or death** in the EVADIAB cohort.  
This clinical-only baseline serves as a reference point before integrating more complex data sources such as radiomics.

---

## 2. Dataset Description

### 2.1 Cohort
- **Number of patients:** 167  
- **Outcome:** Binary indicator of MACE or death  
- **Number of events:** 72  
- **Event rate:** ~43%

Each row corresponds to one patient, with baseline clinical data collected prior to outcome occurrence.

---

### 2.2 Clinical Features Used

Only baseline clinical variables were considered:

- **Demographics**
  - Age
  - Sex (male/female)

- **Metabolic Profile**
  - Hemoglobin A1c (HbA1c)
  - Body Mass Index (BMI)

- **Lipid Profile**
  - LDL cholesterol
  - HDL cholesterol
  - Triglycerides

- **Renal Function**
  - Creatinine

- **Clinical History**
  - Smoking status (non-smoker / smoker / heavy smoker)
  - History of coronary artery disease (CAD)

No imaging, radiomic, or longitudinal variables were included at this stage.

---

## 3. Data Quality and Preprocessing

### 3.1 Missing Data
- Missing values were primarily present in laboratory variables:
  - Creatinine, HbA1c, LDL, HDL, triglycerides: **~4–9% missing**
- Demographic and clinical history variables had no missing values.

**Handling strategy:**
- Median imputation was applied.
- Imputation statistics were computed on the **training set only** to avoid data leakage.

---

### 3.2 Encoding and Normalization
- Binary encoding was used for:
  - Sex
  - History of CAD
- Smoking status was treated as an **ordinal variable** (0 / 1 / 2).
- Continuous variables were standardized for models requiring normalization (logistic regression).

---

## 4. Exploratory Data Analysis (EDA)

Exploratory analysis showed clinically plausible patterns:

- Patients with events tended to be:
  - Older
  - Have higher creatinine levels
  - Exhibit poorer glycemic control (higher HbA1c)
  - Have a higher prevalence of smoking and prior CAD
- Substantial overlap between event and non-event groups was observed for most variables.

These findings suggest the presence of signal in clinical variables, but also indicate limited separability based on clinical data alone.

---

## 5. Modeling Strategy

### 5.1 Models Evaluated
Three models of increasing complexity were evaluated:

1. **Logistic Regression** (reference baseline, interpretable)
2. **Random Forest**
3. **XGBoost**

This stepwise approach ensures that gains from increased model complexity are critically assessed.

---

### 5.2 Evaluation Metrics
Models were evaluated using clinically relevant metrics:

- **ROC-AUC** (discrimination)
- **PR-AUC** (precision–recall performance)
- **Brier score** (calibration)
- Accuracy was reported but not used as a primary decision metric.

---

### 5.3 Stability Analysis
To assess robustness, model performance was evaluated using:
- **50 repeated stratified train/test splits**
- Fixed test size and stratification by outcome

This approach provides insight into performance variability and expected generalization.

---

## 6. Results

### 6.1 Single-Split Performance (Logistic Regression)
- ROC-AUC ≈ **0.72**
- PR-AUC ≈ **0.69**
- Brier score ≈ **0.21**

These results suggested moderate discrimination on a single split.

---

### 6.2 Stability Across 50 Repeated Splits

#### Logistic Regression (Clinical-only)

- **Mean ROC-AUC:** ~0.54  
- **95% CI:** ~[0.52 – 0.57]  
- **Mean PR-AUC:** ~0.56 (baseline event rate ≈ 0.43)  
- **Mean Brier score:** ~0.26  

Performance exhibited **substantial variability across splits**, with ROC-AUC values ranging broadly depending on patient allocation.

#### Comparison with Nonlinear Models
- Random Forest and XGBoost did **not improve** performance.
- Both models showed:
  - Lower ROC-AUC
  - Worse calibration
  - Increased instability

---

## 7. Interpretation

Key findings from the clinical-only modeling:

- Baseline clinical variables provide **limited-to-moderate predictive performance**.
- Performance estimates from single splits can be **optimistic** in small cohorts.
- Repeated-split analysis reveals **high sensitivity to data partitioning**, highlighting the challenges of prediction in small clinical datasets.
- More complex nonlinear models did not outperform logistic regression, suggesting that most of the predictive signal is approximately linear.

---

## 8. Limitations

- Relatively small sample size (n = 167)
- Binary outcome without explicit time-to-event modeling
- Single-center cohort
- No external validation at this stage

These limitations are inherent to the dataset and motivate cautious interpretation.

---

## 9. Conclusion and Next Steps

This analysis establishes a **robust and transparent clinical-only baseline** for the EVADIAB cohort.  
The limited and unstable predictive performance indicates that **clinical variables alone are insufficient for robust individual risk prediction**.

**Next step:**  
Integrate **radiomic features** derived from imaging data to assess whether multimodal models can improve risk stratification beyond clinical baselines.

This clinical-only baseline will serve as the reference for all subsequent multimodal analyses.

---
