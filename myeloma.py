import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor


# Define the cMSE evaluation function
def cMSE(y_hat, y, c):
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]

# Load the training dataset
df = pd.read_csv('train_data.csv', index_col=0)

imputer_iterative = IterativeImputer()
imputer_knn = KNNImputer()

# Use IterativeImputer
df_iterative = df.copy()
df_iterative.iloc[:, :] = imputer_iterative.fit_transform(df)

# Use KNNImputer
df_knn = df.copy()
df_knn.iloc[:, :] = imputer_knn.fit_transform(df)
# Handle missing values for features only
features = df_iterative.iloc[:, :].drop(['SurvivalTime', 'Censored'], axis=1)

# Split the data into features and target arrays
X = features
y = df_iterative.iloc[:, :]['SurvivalTime']

# Filter out rows with NaN SurvivalTime for training
train_indices = y.notna()
X_train = X[train_indices]
y_train = y[train_indices].values
c_train = df_iterative.iloc[:, :].loc[train_indices, 'Censored'].values

# Prepare the data for XGBoost
X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(X_train, y_train, c_train, test_size=0.7, random_state=42)

# Convert the dataset to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Define the XGBoost model parameters
params = {
    'max_depth': 10,
    'min_child_weight': 30,
    'eta': 0.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective':'reg:squarederror',
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=999, evals=[(dval, "Val")], early_stopping_rounds=5)

# Predict on validation set
y_val_pred = bst.predict(dval)
print(y_val_pred.shape)

# Evaluate the model
validation_cMSE = cMSE(y_val_pred, y_val, c_val)
print(f'Validation cMSE: {validation_cMSE}')

# Load the test dataset
test_data = pd.read_csv('test_data.csv', index_col=0)

# Handle missing values for features only in the test set
# It's important to use `transform` here to apply the same transformation as the training set


# Predict on the test set
dtest = xgb.DMatrix(test_data)
test_predictions = bst.predict(dtest)

# Prepare the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_data.index,
    'TARGET': test_predictions
})

# Output predictions to a CSV file for all IDs in the test set
submission_df.to_csv('test_predictions.csv', index=False)
