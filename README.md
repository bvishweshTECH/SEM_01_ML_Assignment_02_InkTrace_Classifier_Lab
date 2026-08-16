# InkTrace Classifier Lab

## a. Problem statement
Build and compare five required classification models and expose them through an interactive Streamlit evaluation dashboard.

## b. Dataset description
**Dataset family:** Optical Recognition of Handwritten Digits  
**Repository:** UCI Machine Learning Repository, Dataset ID 80  
**Task:** Multi-class classification  
**Features:** 64 integer pixel features  
**Classes:** 10 (digits 0–9)

The UCI repository lists 5,620 original instances and 64 features. This executable package uses the 1,797-instance standardized benchmark distributed by scikit-learn from the same dataset family, retaining all 64 features and 10 classes. It therefore exceeds the institute minimum of 12 features and 500 instances.

**Strict-UCI option:** `refresh_from_uci.py` is included. It retrieves the exact UCI repository data through `ucimlrepo` and writes `uci_full_dataset.csv`. If the institute insists on the full UCI file, regenerate the final models/metrics from that file before submission.

## c. Github Repository Link
`https://github.com/bvishweshTECH/SEM_01_ML_Assignment_02_InkTrace_Classifier_Lab`

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9596 | 0.9982 | 0.9600 | 0.9596 | 0.9593 | 0.9552 |
| Decision Tree | 0.8611 | 0.9363 | 0.8619 | 0.8611 | 0.8588 | 0.8462 |
| kNN | 0.9773 | 0.9968 | 0.9788 | 0.9773 | 0.9776 | 0.9749 |
| Naive Bayes | 0.8460 | 0.9741 | 0.8775 | 0.8460 | 0.8467 | 0.8323 |
| Random Forest (Ensemble) | 0.9823 | 0.9997 | 0.9827 | 0.9823 | 0.9821 | 0.9805 |

**AUC:** macro one-vs-rest ROC AUC. **Precision/Recall/F1:** weighted averages.

### Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Strong standardized linear baseline with high AUC and balanced metrics. |
| Decision Tree | Weaker single-tree generalization than the ensemble and distance-based approaches. |
| kNN | Very strong because local pixel-intensity neighborhoods are informative for digit recognition. |
| Naive Bayes | Fast baseline, but correlated pixel features limit its conditional-independence assumption. |
| Random Forest (Ensemble) | Best overall model on this held-out split and the highest across the six reported metrics. |

### Overall Winner for your dataset?
**Random Forest (Ensemble)** — Accuracy 0.9823, AUC 0.9997, F1 0.9821, MCC 0.9805.

## Streamlit features
- CSV test-data upload
- Model selection dropdown
- Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix
- Classification report
- Prediction preview with confidence
- Customized InkTrace UI

## Project structure
```text
InkTrace_Classifier_Lab/
├── app.py
├── train_models.py
├── refresh_from_uci.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model/*.joblib
├── artifacts/*.json
└── data/training_preview.csv
```

## Local execution
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Create a GitHub repository, upload the project, then deploy `app.py` using Streamlit Community Cloud. Replace the two link placeholders in the final PDF after deployment.

## Academic-integrity note
This package is customized with a distinct project name, non-default split, model settings, variable naming, application layout, and observations. The student should review and understand the code and create the GitHub history personally rather than submitting generated material blindly.

## Dataset citation
Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository. Dataset ID 80. DOI: 10.24432/C50P49.
