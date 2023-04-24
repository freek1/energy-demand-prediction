import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
model_predictions_val = pd.read_csv('output/model_predictions_val.csv')

y_val = model_predictions_val['demand_kW']
y_val_decision_tree = model_predictions_val['decision_tree_val']
y_val_random_forest = model_predictions_val['random_forest_val']

# Compute evaluations
correlation_decision_tree = np.corrcoef(y_val, y_val_decision_tree)[0,1]
print('R^2 decision tree:', correlation_decision_tree**2)

correlation_random_forest = np.corrcoef(y_val, y_val_random_forest)[0,1]
print('R^2 random forest:', correlation_random_forest**2)

plt.plot(y_val, y_val_decision_tree, 'o', label='Decision tree')
plt.xlabel('True demand [kW]')
plt.ylabel('Predicted demand [kW]')
plt.title('Model predictions of demand in kW')
plt.legend()
plt.show()

plt.plot(y_val, y_val_random_forest, 'o', label='Random forest')
plt.xlabel('True demand [kW]')
plt.ylabel('Predicted demand [kW]')
plt.title('Model predictions of demand in kW')
plt.legend()
plt.show()
