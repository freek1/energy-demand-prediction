import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

# Load data
model_predictions_val = pd.read_csv('output/model_predictions_val.csv')

y_val = model_predictions_val['demand_kW']
model_predictions = model_predictions_val.drop('demand_kW', axis = 1)

# Evaluation and plotting
plot_size = len(model_predictions.columns)
fig, ax = plt.subplots(1, plot_size, sharey= True)
fig.set_figwidth(7*plot_size)
for i, model in enumerate(model_predictions):
    y_pred = model_predictions[model]
    # Compute evaluations
    correlation = np.corrcoef(y_val, y_pred)[0,1]
    R2 = correlation**2

    ax[i].plot(y_val, y_pred, 'o', label=f'R2 = {R2}')
    ax[i].set_xlabel('True demand [kW]')
    ax[i].set_ylabel('Predicted demand [kW]')
    ax[i].set_title(f'Model predictions of {model}')
    ax[i].legend()
plt.tight_layout()
plt.show()

# Plotting errors
plot_size = len(model_predictions.columns)
fig, ax = plt.subplots(plot_size, 1)
values = np.zeros((len(model_predictions.columns), 2))
for i, model in enumerate(model_predictions):
    y_pred = model_predictions[model]
    values[:,i] = [mean_squared_error(y_val, y_pred), mean_absolute_error(y_val, y_pred)]

ax[0].set_title("Mean Squared Error of Validation Set")
ax[0].barh(model_predictions.columns, values[0,:])
ax[0].set_ylabel("Regressor")
ax[0].set_xlabel("y_test - y_pred")

ax[1].set_title("Mean Absolute Error of Validation Set")
ax[1].barh(model_predictions.columns, values[1,:])
ax[1].set_ylabel("Regressor")
ax[1].set_xlabel("y_test - y_pred")
plt.tight_layout()
plt.show()
