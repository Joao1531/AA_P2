import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Define the cMSE evaluation function
def cMSE(y_hat, y, c):
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]

# Load the training dataset
df = pd.read_csv('train_data.csv', index_col=0)

mean_GeneticRisk = df['GeneticRisk'].mean()
print(mean_GeneticRisk)
df['GeneticRisk'].fillna(mean_GeneticRisk,inplace=True)

# Value is an integer so the mean doesn't work.
mode_ComorbilityIndex = df['ComorbidityIndex'].mode()[0]
print(mode_ComorbilityIndex)
df['ComorbidityIndex'].fillna(mode_ComorbilityIndex,inplace=True)

# We try with the mode and with a missing string, since it depends on each persons genetics
mode_TreatmentResponse = df['TreatmentResponse'].mode()[0]
print(mode_TreatmentResponse)
df['TreatmentResponse'].fillna(mode_TreatmentResponse,inplace=True)

#Check the mean of survival time when the value of the censored column is 0
mean_SurvivalTime =  df[df['Censored']==0]['SurvivalTime'].mean()
print(mean_SurvivalTime)
df['SurvivalTime'].fillna(mean_SurvivalTime,inplace=True)

# Handle missing values for features only
features = df.drop(['SurvivalTime', 'Censored'], axis=1)

# Split the data into features and target arrays
X = features
y = df['SurvivalTime']

# Filter out rows with NaN SurvivalTime for training
train_indices = y.notna()
X_train = X[train_indices]
y_train = y[train_indices].values
c_train = df.loc[train_indices, 'Censored'].values

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
