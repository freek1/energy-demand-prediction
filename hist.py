# Main imports
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Constants
MODEL_NAME = 'histgradientboosting'

train = pd.read_csv('preprocessed_data/train.csv')
val = pd.read_csv('preprocessed_data/val.csv')
test = pd.read_csv('preprocessed_data/test.csv')

print('HistGradientBoost \n ----------')
# Split the data into training and test sets
X_train = train.drop('demand_kW', axis=1)
y_train = np.array(train['demand_kW']).ravel()

X_val = val.drop('demand_kW', axis=1)
y_val = np.array(val['demand_kW']).ravel()

X_test = test.drop('demand_kW', axis=1)

pipeline = HistGradientBoostingRegressor(max_iter=1000, loss='squared_error', l2_regularization=0.3)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)

print('Mean crossval score (cv=10)', np.mean(cross_val_score(pipeline, X_train, y_train, cv=10)))

y_pred_val = np.array(list(pipeline.predict(X_val)))
y_pred_train = np.array(list(pipeline.predict(X_train)))

# Add predicted validation values
model_predictions = pd.read_csv('output/model_predictions_val.csv')
model_predictions[MODEL_NAME] = y_pred_val

# Save data
model_predictions.to_csv('output/model_predictions_val.csv', index=False)