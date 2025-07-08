# Data Mining Final Project

## Overview
This project implements a comprehensive data mining pipeline on a health-related dataset. The objective is to preprocess the dataset, perform feature engineering and selection, apply various sampling techniques, and evaluate multiple classification models using different subsets of features. The final goal is to classify the target variable `Class` (binary: Y/N) with optimal performance.

---

## Project Structure

- **Preprocessing:**
  - Removed columns with no variance, redundant ID columns, and non-informative features.
  - Converted low-cardinality columns to factors.
  - Handled missing values using mode (for categorical) and median (for numeric).
  - Capped outliers and applied log transformations where appropriate.
  - Standardized numeric columns.
  - Retained important categorical features after rare category analysis.

- **Sampling Techniques:**
  - Random Under-Sampling (RUS)
  - Cluster-based Undersampling (k-means)

- **Feature Selection Methods:**
  - Boruta
  - Random Forest (Information Gain)
  - Linear Discriminant Analysis (LDA)

- **Models Evaluated:**
  - K-Nearest Neighbors (KNN)
  - Decision Trees (Rpart)
  - AdaBoost
  - Random Forest
  - Support Vector Machines (SVM-RBF)
  - XGBoost

- **Performance Metrics:**
  - True Positive Rate (Recall), False Positive Rate, Precision, F1 Score
  - Matthews Correlation Coefficient (MCC)
  - Kappa Statistic
  - Area Under the ROC Curve (AUC-ROC)
  - Weighted metrics across both classes

---

## How to Run

1. **Dependencies**
   - R version >= 4.0.0
   - Required Libraries:  
     `caret`, `e1071`, `Metrics`, `randomForest`, `xgboost`, `rpart`, `pROC`, `dplyr`, `tidyr`, `ggplot2`, `corrplot`, `caTools`, `Boruta`, `kknn`, `MASS`, `ada`

2. **Input Files:**
   - `project_data.csv` (Original dataset)

3. **Output Files:**
   - `preprocessed_data.csv`
   - `initial_train.csv`, `initial_test.csv`
   - Feature-selected datasets: `rus_boruta_train.csv`, `rus_info_gain_train.csv`, `rus_lda_train.csv`, `cluster_boruta_train.csv`, etc.

4. **Execution Steps:**
   - Run the entire R script from top to bottom.
   - Visualizations and performance metrics will be displayed in the console and plots window.

---

## Key Highlights

- **Data Cleaning:**
  - Outlier detection and capping
  - Imputation for missing data
  - Correlation-based feature removal

- **Sampling Approaches:**
  - Balanced datasets using RUS and k-means

- **Feature Engineering:**
  - Log transformation, factor conversion, dummy variable support
  - Detection and handling of rare categories

- **Model Evaluation:**
  - Comparative performance of models across three different feature selection techniques and two sampling strategies.
  - Evaluation includes ROC curves and weighted scoring across both classes.

---

## Performance
- **Best Model: Random Forest with Information Gain + Random Under-Sampling (RUS)**
  - Achieved high True Positive Rates (TPR) for both Class 0 (0.800) and Class 1 (0.916).
  - Weighted Precision - 0.914
  - Weighted F-score - 0.842
  - ROC - 0.932
  - MCC - 0.514
  - Kappa - 0.447

## Author
Yash Rao

## License
This project is for academic and educational purposes only.
