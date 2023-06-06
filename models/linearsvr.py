# Main imports
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score

def predict():
    ''' Preforms the prediction of linear svr and generates data in the csv '''
    # Constants
    MODEL_NAME = 'Linear_SVR'

    train = pd.read_csv('preprocessed_data/train.csv')
    val = pd.read_csv('preprocessed_data/val.csv')
    test = pd.read_csv('preprocessed_data/test.csv')

    print(f'{MODEL_NAME} \n ----------')
    # Split the data into training and test sets
    X_train = train.drop('demand_kW', axis=1)
    y_train = np.array(train['demand_kW']).ravel()

    X_val = val.drop('demand_kW', axis=1)
    y_val = np.array(val['demand_kW']).ravel()

    X_test = test.drop('demand_kW', axis=1)

    # SVR
    pipeline = LinearSVR(C=1.0, epsilon=0.2, max_iter=5000)
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

if __name__ == '__main__':
    predict()
