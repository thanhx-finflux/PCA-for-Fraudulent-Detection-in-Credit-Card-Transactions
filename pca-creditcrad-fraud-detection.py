# %% [markdown]
# 1 Problem definition
# 
# Dectect fruadulent credit card transactions (0 = legitimate, 1 = fraudulent) using a 30 feature dataset.
# PCA is used to reduce the dimensionality of the dataset for efficiency and visualization, minimizing the financial risk and computational cost.

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


# %%
# Load the dataset
data = pd.read_csv('creditcard.csv')

# %%
print('Data shape:', data.shape)

# %%
print('Columns:', data.columns.tolist())

# %%
# Basic data information
data.info()

# %% [markdown]
# **Difination of variables**
# - 'Time': Number of seconds elapsed between this transaction and the first transaction in the dataset.
# - 'V1' to 'V28': PCA transformed features.
# - 'Amount': Transaction amount.
# - 'Class': Target variable, where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction.

# %%
# Frist 5 rows of the dataset
data.head()

# %%
print('Missing values:\n', data.isna().sum())

# %%
print('Duplicates:\n', data.duplicated().sum())
print('Percentage of duplicates:', data.duplicated().mean()* 100, '%')

# %% [markdown]
# 2 Data Understanding and Exploration

# %%
# Summary statistics of the dataset
print('Summary statistics:\n', data.describe())

# %%
# Class distribution
print('Class distribution:\n', data['Class'].value_counts())
print('Percentage of fraudulent transactions:', data['Class'].value_counts(normalize = True) * 100)
# Plot class distribution
plt.figure(figsize = (8, 6))
sns.countplot(x= 'Class', data = data, palette = 'Set1')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Fraudulent', 'Fraudulent'])
plt.show()

# %%
# Amount and Time distribution
plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data['Amount'], bins = 50, kde =True, color = 'blue')
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
sns.histplot(data['Time'], bins = 50, kde = True, color = 'green')
plt.title('Time Distribution')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()


# %%
# Correlation heatmap
plt.figure(figsize = (12, 10))
sns.heatmap(data.corr(), cmap = 'coolwarm', square = True, cbar_kws = {'shrink': 0.8})
plt.title('Correlation Heatmap')
plt.show()

# %%
# Amount by Class
plt.figure(figsize = (8, 6))
sns.boxplot(x = 'Class', y = 'Amount', data = data, palette = 'Set1')
plt.title('Transaction Amount by Class')
plt.xlabel('Class')
plt.ylabel('Amount')
plt.xticks([0, 1], ['Non-Fraudulent', 'Fraudulent'])
plt.show()

# %% [markdown]
# - 284,807 transactions, 30 features, no missing values, and about 0.4% duplicates.
# - 492 fraudulent transactions about 0.17% of the dataset, it is highly imbalanced.
# - Amount variable has a long tail distribution, with most transactions being small amounts. 
# - Time variable has cyclical patterns
# - Low correlation due to prior PCA
# - PCA transformed features (V1 to V28) are not highly correlated with each other

# %% [markdown]
# 3. Data Preprocessing and Feature Engineering

# %%
# Remoce dupicates
data1 = data.drop_duplicates()
print('Data shape after removing duplicates:', data1.shape)

# %%
# Separate features and target variable
X = data1.drop(['Class'], axis = 1)
y = data1['Class']

# %%
# Standardize Amount and Time
scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

# %%
# Visualize explained variance
pca0 = PCA()
pca0.fit(X)
cumulative_variance = np.cumsum(pca0.explained_variance_ratio_)
plt.figure(figsize = (8, 6))
plt.plot(np.arange(1, len(cumulative_variance) + 1), 
         cumulative_variance, 
         marker = 'o',
         label = 'Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
ticks = np.arange(0, X.shape[1] + 1, 5)
plt.xticks(ticks, ticks)
plt.grid()
plt.legend()
plt.show()


# %%
# Apply PCA and retain 90% of variance
pca = PCA(n_components = 0.90)
X_pca = pca.fit_transform(X)
X_pca

# %%
# PCA results
print('Original number of features:', X.shape[1])
print('\nNumber of features after PCA:', X_pca.shape[1])
print('\nNumber of components explaining 90% variance:', pca.n_components_)
print('\nExplained variance ratio:', pca.explained_variance_ratio_)
print('\nTotal variance captured by PCA:', np.sum(pca.explained_variance_ratio_))


# %%
# Contribution of each component and importance of features
pca_components = pd.DataFrame(pca.components_.T, 
                              columns = [f'PC{i + 1}' for i in range(pca.n_components_)],
                              index = X.columns) # Original feature names
# Display PCA components
pca_components

# %%
# Plot PCA components contributio
pca_components.plot(kind = 'bar', figsize = (12, 8))
plt.title('PCA Components Contribution')
plt.xlabel('Features')
plt.ylabel('Contribution')
plt.legend(title = 'PCA Components', 
           loc = 'upper left', 
           fontsize = 'small', 
           bbox_to_anchor=(1.0, 1.0))
plt.show()

# %%
# Calculate total contribution of each feature
feature_contribution = pca_components.abs().sum(axis = 1).sort_values(ascending = False)
# Display feature contributions
plt.figure(figsize = (12, 6))
plt.barh(y = feature_contribution.index,
         width = feature_contribution.values, 
         color = 'skyblue'
         )
plt.title('Feature Contribution to PCA Components')
plt.xlabel('Total Contribution')
plt.ylabel('Features')
plt.grid(axis = 'x')
plt.show()


# %%
# Visualize PCA
plt.figure(figsize = (8, 6))
plt.scatter(x = X_pca[:, 0], y = X_pca[:, 1], c = y, cmap = 'coolwarm', alpha = 0.6)
plt.title('PCA Visualization of First Two Components')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label = 'Class')
plt.show()

# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 100, 
                                                    stratify = y)


# %%
# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state = 100)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print('Shape of training features after SMOTE:', X_train_res.shape)
print('Shape of training target after SMOTE:', y_train_res.shape)

# %% [markdown]
# 4. Model selection and traning

# %%
# Train Random Forest Classifier
model_rf = RandomForestClassifier(random_state = 100, n_jobs = -1)
# Train the model
model_rf.fit(X_train_res, y_train_res)

# %%
# Predict on the test set
y_pred = model_rf.predict(X_test)
y_pred_proba = model_rf.predict_proba(X_test)[:, 1]

# %% [markdown]
# 5. Model Evaluation and Optimization

# %%
# Evualation metrics
print('Classification Report:\n', classification_report(y_test, y_pred))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred_proba))

# %%
# Confusion matrix
plt.figure(figsize = (8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, 
            annot = True, 
            fmt = 'd', 
            cmap = 'RdBu', 
            xticklabels = ['Non-Fraudulent', 'Fraudulent'], 
            yticklabels = ['Non-Fraudulent', 'Fraudulent'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize = (8, 6))
plt.plot(fpr, tpr, color = 'blue', label = f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], color = 'red', linestyle = '--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# %%
# HalvingRandomSearchCV for hyperparameter tuning
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
# Define the Random Forest model
model_rf = RandomForestClassifier(random_state = 100, n_jobs = -1)
# Define the parameter distribution for hyperparameter tuning
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Use a representative subset for faster evaluation
X_train_df = pd.DataFrame(X_train_res, columns = [f'PC{i + 1}'for i in range(X_train_res.shape[1])])
y_train_df = pd.Series(y_train_res, name = 'Class')
# Stratified sampling using train_test_split
from sklearn.model_selection import train_test_split
X_sample, _, y_sample, _ = train_test_split(
    X_train_df, y_train_df, 
    train_size=200000, 
    random_state=100, 
    stratify=y_train_df
)

# Because tuning hyperparameters can be computationally expensive, we use HalvingRandomSearchCV
# Check core cpu to ensure it is not too high core cpu to avoid overloading the system and to ensure the prcess runs smoothly
# Incase my system has 20 cores, I will use 10 cores for the tuning process
# import os check core cpu
import os
n_cores = os.cpu_count() // 2
halving_random_search = HalvingRandomSearchCV(estimator = model_rf,
                                                  param_distributions = param_dist,
                                                  scoring = 'roc_auc',
                                                  cv = 5,
                                                  n_jobs = n_cores,
                                                  verbose = 2,
                                                  random_state = 100,
                                                  factor = 2,
                                                  min_resources = 10000, # This is a more efficient option than 'exhaust', 
                                                  # 'smallest' is also a good option
                                                  # 'exhaust' is not recommended for large datasets
                                                  max_resources = 'auto',
                                                  aggressive_elimination = True)
# Fit the model
halving_random_search.fit(X_sample, y_sample)
# Best parameters and score
print('Best parameters:', halving_random_search.best_params_)
print('Best ROC AUC score:', halving_random_search.best_score_)
# Best estimator
best_halving_model = halving_random_search.best_estimator_

# %% [markdown]
# In case of large datasets, we can use tuning with HalvingRandomSearchCV instead of GridSearchCV and RandomizedSearchCV becuase it is more efficient and faster. 

# %%
# Predict on the test set using the best halving model
y_pred_halving = best_halving_model.predict(X_test)
y_pred_proba_halving = best_halving_model.predict_proba(X_test)[:, 1]

# %%
# Evaluation metrics for the best halving model
print('Classification Report for Halving Model:\n', classification_report(y_test, y_pred_halving))
print('ROC AUC Score for Halving Model:', roc_auc_score(y_test, y_pred_proba_halving))

# %%
# Confusion matrix for the best halving model
plt.figure(figsize = (8, 6))
conf_matrix_halving = confusion_matrix(y_test, y_pred_halving)
sns.heatmap(conf_matrix_halving,
            annot = True,
            fmt = 'd',
            cmap = 'RdBu')
plt.xticklabels = ['Non-Fraudulent', 'Fraudulent'],
plt.yticklabels = ['Non-Fraudulent', 'Fraudulent'],
plt.xlabel = 'Predicted',
plt.ylabel = 'Actual'
plt.title('Confusion Matrix for Halving Model')
plt.show()

# %%
# ROC curve for the best halving model
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_halving)
plt.figure(figsize = (8, 6))
plt.plot(fpr, tpr, color = 'blue', label = f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba_halving):.2f})')
plt.plot([0, 1], [0, 1], color = 'red', linestyle = '--')
plt.title('ROC Curve for Halving Model')
plt.legend()
plt.show()

# %%
# Cross-validation for the best halving model
auc_scores = []
# Using cross_val_score to evaluate stability of the best halving model
for seed in [42, 52, 142]:
    x_sample, _, y_sample, _ = train_test_split(
        X_train_df, y_train_df, 
        train_size=200000, 
        random_state=seed, 
        stratify=y_train_df
    )
    cv_scores_halving = cross_val_score(best_halving_model, x_sample, y_sample, cv = 5, scoring = 'roc_auc', n_jobs = n_cores, verbose = 2)
    auc_scores.append(np.mean(cv_scores_halving))
print('Cross-validation ROC AUC scores for Halving Model:', auc_scores)
print('Mean ROC AUC score for Halving Model:', np.mean(auc_scores))
print('Standard deviation of ROC AUC scores for Halving Model:', np.std(auc_scores))

# %%
# Conclusion
# PCA was applied to reduce dimensionality while retaining 90% of the variance.
# In case of large datasets, we can use tuning with HalvingRandomSearchCV instead of GridSearchCV and RandomizedSearchCV because it is more efficient and faster.
# ROC AUC score for the best halving model has been improved compred to the initial model.
# ROC AUC score indicates that the model is performing well in distingushing between fraudulent and non-fraudulent transactions.
# We use recall and precision to evaluate the model performance, especially in fraud detection where false negatives are more critical.
# We use precision to evaluate the model performance, especially in fraud detection where false positives are more critical.
# The f1-score is a good metric to evaluate the model performance in fraud detection, as it balances precision and recall.
# The model is performing well in detecting fraudulent transactions with a high recall and precision.


