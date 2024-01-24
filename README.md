# Credit Risk Analysis
** Supervised Machine Learning , Resampling **

## Overview
This project focuses on predicting credit risk using various machine learning models, given an imbalanced dataset with a higher number of good loans than risky ones. The LendingClub's credit card credit dataset was split into training and testing subsets, followed by resampling. Six machine learning models were trained to predict credit risk, and their performance was assessed based on balanced accuracy score, precision, and sensitivity.

### Models:
- Logistic regression model with random oversampled data
- Logistic regression model with synthetic minority oversampled (SMOTE) data
- Logistic regression model with cluster centroid undersampled data
- Logistic regression model with oversampled and undersampled combination (SMOTEENN) data
- Balanced random forest classification model
- Easy ensemble adaptive boosting classification model

### Evaluations:
- Accuracy score of the model
- Confusion matrix
- Imbalanced classification report

## Results
### Models with Unscaled Dataset
1. **Naive Random Oversampling Model**
   :arrow_lower_right:
   - Balanced accuracy score: **65%**
   - Precision of high risk: **1%**
   - Recall (Sensitivity) of high risk: **71%**
   
   ![Naive Random Oversampling Model](https://user-images.githubusercontent.com/105877888/191419580-ea8d49c7-9c61-42d8-9ad4-c53d70b5adfa.png)

2. **SMOTE Model**
   :arrow_lower_right:
   - Balanced accuracy score: **66%**
   - Precision of high risk: **1%**
   - Recall (Sensitivity) of high risk: **63%**
   
   ![SMOTE Model](https://user-images.githubusercontent.com/105877888/191419600-a5dae719-e96f-4167-859d-011b6456c484.png)

3. **Cluster Centroids Model**
   :arrow_lower_right:
   - Balanced accuracy score: **54%**
   - Precision of high risk: **1%**
   - Recall (Sensitivity) of high risk: **69%**
   
   ![Cluster Centroids Model](https://user-images.githubusercontent.com/105877888/191419613-29057895-4a4a-415c-85b4-2f648be5cd08.png)

4. **SMOTEENN Model**
   :arrow_lower_right:
   - Balanced accuracy score: **62%**
   - Precision of high risk: **1%**
   - Recall (Sensitivity) of high risk: **68%**
   
   ![SMOTEENN Model](https://user-images.githubusercontent.com/105877888/191419624-f5cc523f-09ef-488a-9f27-7878a5f85d33.png)

5. **Balanced Random Forest Model**
   :arrow_lower_right:
   - Balanced accuracy score: **79%**
   - Precision of high risk: **3%**
   - Recall (Sensitivity) of high risk: **70%**
   
   ![Balanced Random Forest Model](https://user-images.githubusercontent.com/105877888/191419634-b2ac04ef-f3e5-4ccd-97f7-ef512a8bc989.png)

6. **Easy Ensemble Model**
   :arrow_lower_right:
   - Balanced accuracy score: **93%**
   - Precision of high risk: **9%**
   - Recall (Sensitivity) of high risk: **92%**
   
   ![Easy Ensemble Model](https://user-images.githubusercontent.com/105877888/191419652-5e1ebed0-4f8d-4c52-b48a-02a6f7bdcdc1.png)

### Models with Scaled Data
- The table below shows model performance before and after scaling with StandardScaler (`mean = 0, SD = 1`).

| Model                               | Balanced accuracy score | Precision of high risk | Recall of high risk | F1 score of high risk | 
|:------                              |:------                  |:------                |:------              | :------               | 
| **Naive Random Oversampling Model** | 0.84 > `0.66`           | 0.03 > `0.01`          | 0.83 > `0.71`       | 0.06 > `0.02`         |
| **SMOTE Oversampling Model**        | 0.84 > `0.66`           | 0.03 > `0.01`          | 0.81 > `0.63`       | 0.07 > `0.02`         |
| **Cluster Centroids Undersampling Model** | 0.81 > `0.54`     | 0.02 > `0.01`          | 0.86 > `0.69`       | 0.04 > `0.01`         |
| **SMOTEENN Combined Resampling Model** | 0.85 > `0.62`         | 0.03 > `0.01`          | 0.84 > `0.68`       | 0.06 > `0.02`         |
| **Balanced Random Forest Model**    | 0.79 = `0.79`           | 0.03 = `0.03`          | 0.70 = `0.70`       | 0.06 = `0.06`         |
| **Easy Ensemble Model**             | 0.93 = `0.93`           | 0.09 = `0.09`          | 0.92 = `0.92`       | 0.16 = `0.16`         |
 
## Summary
- Logistic Regression Models with Naive Random Oversampling, SMOTE Oversampling, Cluster Centroids Undersampling, and SMOTEENN Oversampling and Undersampling Combination achieved accuracy scores of 65%, 66%, 54%, and 62%, respectively. Balanced Random Forest Model and Easy Ensemble Model outperformed with accuracy scores of 79% and 93%.

- All six models exhibited precision of high risk under 10%, suggesting a considerable number of false positives. Easy Ensemble Model slightly outperformed others in this aspect.

- Feature scaling significantly improved the performance of Logistic regression models but had minimal impact on Balanced Random Forest Model and Easy Ensemble Model, which are tree-based models and do not require feature scaling.

- Easy Ensemble Model demonstrated a higher recall score (92%), indicating its effectiveness in detecting potentially high-risk loans. However, the low precision of high-risk cases (9%) and F1 score (16%) highlight the imbalance between sensitivity and precision.

- Recommendations: Easy Ensemble Model performed best in predicting high-risk credit, with a satisfactory accuracy score (93%) and recall of high risk (92%). However, it exhibited a low precision of high-risk cases (9%), indicating a risk of falsely labeling low-risk applicants as high risk. LendingClub should consider the trade-off between sensitivity and precision. Additionally, for further study, deep learning neural network binary classification can be explored, and cross-validation can be applied to enhance model predictions.