# Main imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Constants
MODEL_NAME = 'random_forest'

train = pd.read_csv('preprocessed_data/train.csv')
val = pd.read_csv('preprocessed_data/val.csv')
test = pd.read_csv('preprocessed_data/test.csv')

print('Random forest \n ----------')
# Split the data into training and test sets
X_train = train.drop('demand_kW', axis=1)
y_train = np.array(train['demand_kW']).ravel()

X_val = val.drop('demand_kW', axis=1)
y_val = np.array(val['demand_kW']).ravel()

X_test = test.drop('demand_kW', axis=1)

# Decision Tree
pipeline = RandomForestRegressor(n_estimators=10, random_state=0)
# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

print('Mean crossval score (cv=10)', np.mean(cross_val_score(pipeline, X_train, y_train, cv=10)))

y_pred_val = np.array(list(pipeline.predict(X_val)))
y_pred_train = np.array(list(pipeline.predict(X_train)))

# Add predicted validation values
model_predictions = pd.read_csv('output/model_predictions_val.csv')
model_predictions[MODEL_NAME] = y_pred_val

# Save data
model_predictions.to_csv('output/model_predictions_val.csv', index=False)
