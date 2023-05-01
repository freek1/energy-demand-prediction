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
pipeline = RandomForestRegressor(n_estimators=10, random_state=0, bootstrap = True)
# Fit the pipeline to the training data
pipeline.fit(X_train.append(X_val), np.append(y_train, y_val))

y_test = np.array(list(pipeline.predict(X_test)))


original_demand = pd.read_csv('data/demand_kWtrain_val.csv')
original_demand = original_demand.iloc[273988:]
original_demand = original_demand.drop('demand_kW', axis=1)
y_test_pd = pd.DataFrame({'datetime_local': original_demand.datetime_local,'demand_kW': y_test})
y_test_pd.to_csv("output/y_test.csv", index = False)

original_demand = pd.read_csv('data/demand_kWtrain_val.csv')
assert(len(y_test_pd) == original_demand['demand_kW'].isnull().sum())