# Credit-Card-Fraud-Detection

A machine learning-based project to detect fraudulent credit card transactions using various supervised classification algorithms. This project evaluates multiple models and identifies the most accurate one using performance metrics and visualization techniques.

## Overview

Credit card fraud has been on the rise, causing significant financial loss and distress to customers. This project applies machine learning techniques to effectively identify and classify fraudulent transactions from legitimate ones.

The dataset used is sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), consisting of anonymized transaction features over two days.


## Objectives

- Detect fraudulent credit card transactions.
- Compare the performance of four classification algorithms:  
  - K-Nearest Neighbors (KNN)  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Naïve Bayes
- Select the best model based on accuracy and misclassification rate.


## Machine Learning Techniques Used

| Algorithm                    | Accuracy |
|------------------------------|----------|
| Support Vector Machine (SVM) | 99.94%   |
| Logistic Regression          | 99.92%   |
| K-Nearest Neighbors (KNN)    | 99.89% (K=3), 99.88% (K=7) |
| Naïve Bayes                  | 97.76%   |

Evaluation was performed using confusion matrices, classification metrics, and visualizations.


## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,808 transactions  
- **Attributes:** 31  
  - 28 anonymized PCA features  
  - Time, Amount, Class (0 = Non-Fraud, 1 = Fraud)


## Tools & Technologies

### Languages:
- Python

### Libraries:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

### Tools:
- RStudio
- Weka
- Sublime Text (for ARFF file formatting)


## Methodology

This project follows the **CRISP-DM** (Cross Industry Standard Process for Data Mining) approach:
1. **Business Understanding**
2. **Data Understanding**
3. **Data Preparation**
4. **Modeling**
5. **Evaluation & Deployment**


## Data Preprocessing

- Converted `Class` column to categorical (`Fraud`, `Not Fraud`)
- Normalized/standardized relevant features
- No missing values or duplicates were found
- Dataset split: 70% training, 30% testing


## Evaluation

All models were tested using metrics such as:
- Accuracy
- True Positive Rate (Recall)
- False Positive Rate
- Confusion Matrix

The **SVM** model outperformed others with an accuracy of **99.94%**, making it the best-suited model for this dataset.


## Future Enhancements

- Use of ensemble models (Random Forest, XGBoost)
- Apply SMOTE or ADASYN to handle class imbalance
- Real-time detection with streaming data
- Integration with mobile alerts system
- Use geo-location or IP data for anomaly detection


## Co-Authors  
- [@Jay-1232](https://github.com/jay-1232)
- Supervised by: **Prof. Swati Pandey**


## License

This project is for academic and educational purposes only.
