from utils import preprocess, explore_dataset, evaluation, test_prediction
from models import decision_tree, random_forest, hist, linearsvr
import pandas as pd

def main():
    ''' Runs the pipeline '''
    # Get data (csv files not pushed to repo, since it pertains company data)
    weather = pd.read_csv('data/weather.csv')
    pallet_history = pd.read_csv('data/Pallet_history_Gold_Spike.csv')
    inbound = pd.read_csv('data/inbound_loads.csv')
    outbound = pd.read_csv('data/outbound_laods.csv')
    demand = pd.read_csv('data/demand_kWtrain_val.csv')

    # Can explore the dataset if you like:
    explore_dataset.explore()

    # Then preprocess the data into usable csvs (are pushed to repo, otherwise checking this code is useless as it does nothing)
    preprocess.preprocess(weather, pallet_history, inbound, outbound, demand)

    # Then run the models you like
    # They will all append their predictions to the output/model_predictions.csv file
    decision_tree.predict()
    random_forest.predict()
    hist.predict()
    linearsvr.predict()

    # Evaluate each model
    evaluation.evaluate()

    # And finally you can test the predictions of the random forest model, our best fitted model
    test_prediction.test()

if __name__ == '__main__':
    main()