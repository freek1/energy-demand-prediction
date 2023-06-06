# Main imports
import pandas as pd

def explore():
    ''' quick exploration of given dataset '''
    weather = pd.read_csv('data/weather.csv')
    pallet_history = pd.read_csv('data/Pallet_history_Gold_Spike.csv')
    inbound = pd.read_csv('data/inbound_loads.csv')
    outbound = pd.read_csv('data/outbound_laods.csv')
    demand = pd.read_csv('data/demand_kWtrain_val.csv')



    weather.info(verbose=True)
    print("\n")
    pallet_history.info(verbose=True)
    print("\n")
    inbound.info(verbose=True)
    print("\n")
    outbound.info(verbose=True)
    print("\n")
    demand.info(verbose=True)


    # after preprocessing
    preprocessed = True

    if preprocessed:
        train = pd.read_csv('preprocessed_data/train.csv')
        test = pd.read_csv('preprocessed_data/test.csv')
        val = pd.read_csv('preprocessed_data/val.csv')

        train.info(verbose=True)
        print("\n")
        test.info(verbose=True)
        print("\n")
        val.info(verbose=True)


if __name__ == '__main__':
    explore()