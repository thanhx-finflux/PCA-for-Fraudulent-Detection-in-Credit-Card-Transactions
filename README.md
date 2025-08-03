# PCA-for-Fraudulent-Detection-in-Credit-Card-Transactions
Principal Component Analysis (PCA) reduces the dimensionlity of Kaggle Credit Card Fraud Detection dataset.

This project applies Principal Component Analysis (PCA) to reduce the dimensionality of the Kaggle Credit Crad Fraud Detection data (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and trains a Random Forest Classifier to detect traudulent transactions.
The pipeline address financial risk management - high fraud recall, cost control - PCA for efficiency, and performance reporting (ROC-AUC)

**Problem**: Detect traudulent credit card transactions (0 = legitimate, 1 = fraudulent) uisng a 30 feature dataset. PCA reduces dimensionlity for effeciency and visualization, minimizing financial risk.

**Apply**: PCA, SMOTE, Random Forest, HalvingRandomSearchCV for hyperparameter tunning, Confusion matrix, ROC curve, Cross-validation for monitoring performance.

# Dataset
- Source: Credit Card Fraud Detection dataset[](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- License: CC0 1.0 Public Domain Dedication
- Citation: Machine Learning Group - ULB

**Dataset**: Kaggle's Credict Fruad Detection dataset (284,0807 transactions, 30 numerical features: V1-V28, Amount, Time, binary target Class, Fraud cases are rate 0.17% high imbalance.

- 284,807 transactions, 30 features, no missing values, and about 0.4% duplicates.
- 492 fraudulent transactions about 0.17% of the dataset, it is highly imbalanced.
- Amount variable has a long tail distribution, with most transactions being small amounts. 
- Time variable has cyclical patterns
- Low correlation due to prior PCA
- PCA transformed features (V1 to V28) are not highly correlated with each other


# Conclusion
- PCA was applied to reduce dimensionality while retaining 90% of the variance.
- In case of large datasets, we can use tuning with HalvingRandomSearchCV instead of GridSearchCV and RandomizedSearchCV because it is more efficient and faster.
- ROC AUC score for the best halving model has been improved compred to the initial model.
- ROC AUC score indicates that the model is performing well in distingushing between fraudulent and non-fraudulent transactions.
- We use recall and precision to evaluate the model performance, especially in fraud detection where false negatives are more critical.
- We use precision to evaluate the model performance, especially in fraud detection where false positives are more critical.
- The f1-score is a good metric to evaluate the model performance in fraud detection, as it balances precision and recall.
- The model is performing well in detecting fraudulent transactions with a high recall and precision.

# Author
Thanh Xuyen Nguyen - https://www.linkedin.com/in/xuyen-thanh-nguyen-0518/
