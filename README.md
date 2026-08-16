# BITS M.Tech AIML
# Machine Learning : Assignment-02

## InkTrace Classifier Lab

### 1. Problem Statement

Build and compare five classification models for handwritten-digit recognition and expose the trained models through an interactive Streamlit evaluation dashboard. The project covers model training, evaluation using multiple classification metrics, visualization of results, and deployment through Streamlit Community Cloud.

### 2. Dataset Description

**Dataset family:** Optical Recognition of Handwritten Digits  
**Source family:** UCI Machine Learning Repository, Dataset ID 80  
**Task:** Multi-class classification  
**Instances used:** 1,797  
**Input features:** 64 integer pixel features  
**Classes:** 10 (digits 0–9)  
**Target column:** `target`

The original Optical Recognition of Handwritten Digits dataset is associated with the UCI Machine Learning Repository. The executable version of this project uses the 1,797-instance standardized benchmark distributed by scikit-learn from the same dataset family, retaining all 64 pixel features and 10 digit classes. The project therefore exceeds the institute minimum requirement of 12 features and 500 instances.

The 10 classes represent the handwritten digits **0 through 9**.

### 3. Data Splitting Strategy

To keep the experiment reproducible and evaluate the models on unseen data, the packaged benchmark is divided into training and held-out test sets.

- **Training:** approximately 78% (1,401 instances)
- **Final Test:** 22% (396 instances)
- **Split type:** Stratified train/test split
- **Random seed:** `73`

The split is stratified so that the ten digit classes remain represented proportionally in the training and held-out test sets. The test set is not used for model fitting and is used to calculate the final reported metrics.

The exact configuration used by the executable project is recorded in `artifacts/experiment_config.json`.

### 4. Models Used

The following five classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

All five models are trained using the same training/test split so that their performance can be compared consistently.

### 5. Evaluation Metrics

For every model, the following evaluation metrics are calculated:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Because this is a ten-class classification problem:

- **AUC** is calculated using macro one-vs-rest multiclass ROC AUC.
- **Precision, Recall and F1** use weighted averaging.
- **MCC** is calculated using the multiclass predictions and true labels.

### 6. Model Performance Comparison

The following table presents the performance of all five classification models on the held-out test dataset containing 396 observations.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9596 | 0.9982 | 0.9600 | 0.9596 | 0.9593 | 0.9552 |
| Decision Tree | 0.8611 | 0.9363 | 0.8619 | 0.8611 | 0.8588 | 0.8462 |
| kNN | 0.9773 | 0.9968 | 0.9788 | 0.9773 | 0.9776 | 0.9749 |
| Naive Bayes | 0.8460 | 0.9741 | 0.8775 | 0.8460 | 0.8467 | 0.8323 |
| **Random Forest (Ensemble)** | **0.9823** | **0.9997** | **0.9827** | **0.9823** | **0.9821** | **0.9805** |

### 7. Model Performance Observations

#### Logistic Regression

Logistic Regression provides a strong linear baseline for handwritten-digit recognition. It achieved **95.96% accuracy**, **0.9982 AUC**, and **0.9593 F1 score**, with a high MCC of **0.9552**. Its strong AUC indicates very good class-separation capability, although its overall performance is below kNN and Random Forest on the held-out test set.

#### Decision Tree

The Decision Tree achieved **86.11% accuracy**, **0.9363 AUC**, and **0.8588 F1 score**, with an MCC of **0.8462**. Although it provides a useful nonlinear classifier, its performance is substantially lower than the ensemble and kNN models for this dataset. This suggests that a single tree does not capture the digit-pattern relationships as effectively as the stronger models.

#### kNN

kNN achieved **97.73% accuracy**, **0.9968 AUC**, **0.9776 F1 score**, and **0.9749 MCC**. Its strong performance indicates that handwritten digits with similar pixel-intensity patterns tend to be close to one another in the feature space. kNN is therefore highly effective for this dataset, although it remains slightly below Random Forest across the reported metrics.

#### Naive Bayes

Gaussian Naive Bayes achieved **84.60% accuracy**, **0.9741 AUC**, and **0.8467 F1 score**, with an MCC of **0.8323**. It provides a fast probabilistic baseline, but its conditional-independence assumption is less suitable for pixel features that can exhibit relationships with one another. Consequently, it produced the weakest overall performance among the five evaluated models.

#### Random Forest (Ensemble)

Random Forest achieved the **best overall performance**, with **98.23% accuracy**, **0.9997 AUC**, **0.9821 F1 score**, and **0.9805 MCC**. It also achieved the highest precision (**0.9827**) and recall (**0.9823**). The results indicate that the ensemble approach captures the nonlinear patterns in the handwritten-digit features effectively and provides the strongest overall classification performance on the held-out test set.

#### Overall Winner

**Random Forest (Ensemble) is the overall winner for this dataset.** It achieved the highest Accuracy, AUC, Precision, Recall, F1 and MCC among the five evaluated models on the held-out test data. Therefore, based on the reported evaluation metrics, Random Forest provides the strongest overall classification performance for this handwritten-digit recognition task.

### 8. Repository Structure

```text
InkTrace_Classifier_Lab/
│
├── app.py
├── train_models.py
├── refresh_from_uci.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── random_forest_ensemble.joblib
│   └── README.md
│
└── artifacts/
    ├── metrics.json
    ├── confusion_matrices.json
    └── experiment_config.json
```

### 9. How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The required packages are pinned to compatible major/minor ranges in `requirements.txt`, including Streamlit, scikit-learn, pandas, NumPy, Matplotlib and joblib.

Train and evaluate all models:

```bash
python train_models.py
```

This command loads the packaged handwritten-digit benchmark, creates the stratified training/held-out-test split, trains all five classifiers, calculates the six evaluation metrics, saves the trained model files, writes `test_data.csv`, and stores the experiment metrics in `artifacts/metrics.json`.

Run the Streamlit application:

```bash
streamlit run app.py
```

The application opens an interactive dashboard where the supplied `test_data.csv` can be uploaded and one of the five trained models can be selected.

### 10. Streamlit Features

The application includes:

- CSV test-data upload
- Model-selection dropdown
- Logistic Regression
- Decision Tree Classifier
- kNN
- Gaussian Naive Bayes
- Random Forest (Ensemble)
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Confusion matrix
- Classification report
- Prediction preview with confidence
- Customized InkTrace user interface

The application allows the user to upload the supplied held-out test data, select a trained classification model, generate predictions and view the corresponding evaluation results. Metrics are recalculated from the uploaded test data.

### 11. Final Submission Links

**GitHub Repository:** [ML Assignment 02- InkTrace Classifier Lab](https://github.com/bvishweshTECH/SEM_01_ML_Assignment_02_InkTrace_Classifier_Lab)

**Live Streamlit App:** (https://sem01mlassignment02inktraceclassifierlab-tsa9at6fpxkmdqmd4c3zp.streamlit.app/)

### 12. Pre-Submission Checklist

Before submitting the final PDF, verify the following:

- All five model files are present in the `model/` folder.
- `test_data.csv` contains the supplied held-out test data used for evaluation.
- `artifacts/metrics.json` contains the final reported model metrics.
- `requirements.txt` is present and tested.
- GitHub repository contains all required project files.
- GitHub Repository link opens correctly.
- Streamlit Community Cloud application opens successfully.
- The test CSV upload works in the deployed application.
- The model-selection dropdown works.
- Accuracy, AUC, Precision, Recall, F1 and MCC are displayed.
- Confusion matrix is displayed.
- Classification report is displayed.
- Prediction preview is displayed.
- BITS Virtual Lab execution has been completed.
- The BITS Virtual Lab screenshot is included in the final PDF.
- README content is included in the final PDF as required.
- GitHub Repository and Streamlit App links are updated before final submission.
- The reported observations match the actual final metrics.
- Random Forest (Ensemble) is identified as the overall winner based on the reported results.
- The student has reviewed and understood the code, model settings and results.
