# Main imports
import pandas as pd
import numpy as np

# Additional imports
import time
import datetime
import random

# Extras
pd.options.mode.chained_assignment = None


weather = pd.read_csv('data/weather.csv')
pallet_history = pd.read_csv('data/Pallet_history_Gold_Spike.csv')
inbound = pd.read_csv('data/inbound_loads.csv')
outbound = pd.read_csv('data/outbound_laods.csv')
demand = pd.read_csv('data/demand_kWtrain_val.csv')

#################
# Preprocess inbound csv:
#################
print('\n #####\nInbound: \n ##### \n')
inbound_post = inbound[inbound.carrier_code != 'CANCEL']
inbound_post = inbound[inbound.carrier_code != '']

inbound_post['truck_signin_datetime'] = pd.to_datetime(inbound_post['truck_signin_datetime'])

# Compute delta times
inbound_load_time = pd.to_datetime(inbound_post['load_finish_datetime']) - pd.to_datetime(inbound_post['load_start_datetime'])
inbound_truck_time = pd.to_datetime(inbound_post['truck_signin_datetime']) - pd.to_datetime(inbound_post['signout_datetime'])

# Drop unnecessary columns
inbound_post = inbound_post.drop(['Unnamed: 0', 'warehouse_order_number', 'customer_code', 'load_reference_number', 'carrier_code', 'weight_uom', 'load_finish_datetime', 'load_start_datetime', 'dock_door_number', 'trailer_number', 'signout_datetime'], axis=1)

# Add time deltas
inbound_post['load_time'] = inbound_load_time
inbound_post['truck_time'] = inbound_truck_time

print(inbound_post.columns)

inbound_post['load_time'] = inbound_post['load_time'].dt.seconds
inbound_post['truck_time'] = inbound_post['truck_time'].dt.seconds

print('With NaN:', inbound_post.shape)

# Drop rows with >0 NaN values
inbound_post_nan = inbound_post.dropna().reset_index(drop=True)

print('Without NaN:', inbound_post_nan.shape)

#################
# Preprocess outbound csv:
#################
print('\n #####\nOutbound: \n ##### \n')
outbound_post = outbound[outbound.carrier_code != 'CANCEL']
outbound_post = outbound[outbound.carrier_code != 'VOID']
outbound_post = outbound[outbound.carrier_code != '']

outbound_post['truck_signin_datetime'] = pd.to_datetime(outbound_post['truck_signin_datetime'])

# Compute delta times
outbound_load_time = pd.to_datetime(outbound_post['load_finish_datetime']) - pd.to_datetime(outbound_post['load_start_datetime'])
outbound_truck_time = pd.to_datetime(outbound_post['truck_signin_datetime']) - pd.to_datetime(outbound_post['signout_datetime'])

# Drop unnecessary columns
outbound_post = outbound_post.drop(['Unnamed: 0', 'warehouse_order_number', 'customer_code', 'load_reference_number', 'carrier_code', 'weight_uom', 'load_finish_datetime', 'load_start_datetime', 'dock_door_number', 'trailer_number', 'signout_datetime'], axis=1)

# Add time deltas
outbound_post['load_time'] = outbound_load_time
outbound_post['truck_time'] = outbound_truck_time

print(outbound_post.columns)

outbound_post['load_time'] = outbound_post['load_time'].dt.seconds
outbound_post['truck_time'] = outbound_post['truck_time'].dt.seconds

print('With NaN:', outbound_post.shape)

# Drop rows with >0 NaN values
outbound_post_nan = outbound_post.dropna().reset_index(drop=True)

print('Without NaN:', outbound_post_nan.shape)

#################
# Preprocess demand csv:
#################
print('\n #####\nDemand:  \n ##### \n')

demand['datetime_local'] = pd.to_datetime(demand['datetime_local'])

# How to split time series:
def split_train_val_test(data, end_known_idx, split=0.7):
    '''Splits dataset into train val and test
    
    Test set is inferred from end_known_idx (int)
    Train size is dictated by split (float)
    Train and val are randomly split in correct proportions
    
    Input: 
        data: pd dataframe
        end_known_idx: index of start of test set (int)
        split: float indicating size of train set (known -> train, val)
    Output:
        (train_set, val_set, test_set) tuple
    '''
    test_set = data.iloc[end_known_idx:]
    # Randomly split train and val out of known with prob. split
    train_idx = random.sample(set(np.arange(end_known_idx)), int(split * end_known_idx))
    train_set = data.iloc[train_idx]
    val_idx = set(np.arange(end_known_idx)) - set(train_idx)
    val_set = data.iloc[list(val_idx)]
    return train_set, val_set, test_set

end_known_idx = demand[demand.demand_kW > 1].index[-1]+1
train_val_split = 0.7 # 70% train, 30% val
# end_train_idx = int((train_val_split) * end_known_idx)
# demand_train = demand.loc[0:end_train_idx-1]
# demand_val = demand.loc[end_train_idx:end_known_idx]
# demand_test = demand.iloc[end_known_idx+1:-2]
demand_train, demand_val, demand_test = split_train_val_test(demand, end_known_idx, split=train_val_split)

print('Full dataset:', demand.shape)
print('Answers known until index: ', end_known_idx)
print(f'Training set, {int(train_val_split*100)}%:', demand_train.shape)
print(f'Validation set, {int(100-train_val_split*100)}%:', demand_val.shape)
print('Test set', demand_test.shape)

#################
# Preprocess weather csv:
#################
print('\n #####\nWeather:  \n ##### \n')

weather_post = weather.copy()
UTC6 = pd.to_datetime(weather_post['datetime_UTC']) - pd.Timedelta(hours=6)
weather_post['datetime_america'] = UTC6
weather_post = weather_post.drop('datetime_UTC', axis=1)
weather_post = weather_post.drop(['datetime', 'hour', 'Unnamed: 0'], axis=1)

print(weather_post.head())
print(weather_post.columns)
print(weather_post.shape)




# Combine weather and demand data to train, val, and test sets

weather_post.sort_values("datetime_america", inplace=True)
demand_train.sort_values("datetime_local", inplace=True)
demand_val.sort_values("datetime_local", inplace=True)
demand_test.sort_values("datetime_local", inplace=True)
print("weather", weather_post.shape, '\n')
print("demand_train", demand_train.shape)
print("demand_val", demand_val.shape)
print("demand_test", demand_test.shape, '\n')
demand_weather_train = pd.merge_asof(demand_train, weather_post, left_on='datetime_local', right_on='datetime_america', direction='nearest')
demand_weather_val = pd.merge_asof(demand_val, weather_post, left_on='datetime_local', right_on='datetime_america', direction='nearest')
demand_weather_test = pd.merge_asof(demand_test, weather_post, left_on='datetime_local', right_on='datetime_america', direction='nearest')

demand_weather_train = demand_weather_train.drop(['Unnamed: 0','datetime_america'], axis=1)
demand_weather_val = demand_weather_val.drop(['Unnamed: 0','datetime_america'], axis=1)
demand_weather_test = demand_weather_test.drop(['Unnamed: 0','datetime_america'], axis=1)

print("train merged", demand_weather_train.shape)
print("val merged", demand_weather_val.shape)
print("test merged", demand_weather_test.shape)





# Merging inbound with the above demand/weather dataset. Maximum 15 minutes difference in order to merge.
# Results in NaN and NaT where > 15 mins difference.
inbound_post_nan.sort_values("truck_signin_datetime", inplace=True)
print("train pre", demand_weather_train.shape)
print("val pre", demand_weather_val.shape)
print("test pre", demand_weather_test.shape, '\n')

demand_inbound_merge_train = pd.merge_asof(demand_weather_train, inbound_post_nan, 
                                     left_on='datetime_local', 
                                     right_on='truck_signin_datetime', 
                                     direction='nearest', 
                                     tolerance=datetime.timedelta(minutes = 15))
demand_inbound_merge_val = pd.merge_asof(demand_weather_val, inbound_post_nan, 
                                     left_on='datetime_local', 
                                     right_on='truck_signin_datetime', 
                                     direction='nearest', 
                                     tolerance=datetime.timedelta(minutes = 15))
demand_inbound_merge_test = pd.merge_asof(demand_weather_test, inbound_post_nan, 
                                     left_on='datetime_local', 
                                     right_on='truck_signin_datetime', 
                                     direction='nearest', 
                                     tolerance=datetime.timedelta(minutes = 15))

# Add only load time and truck time
demand_inbound_merge_train1 = demand_inbound_merge_train.drop(
    ['truck_signin_datetime', 'front_temperature', 'middle_temperature', 'back_temperature', 'case_quantity', 'pallet_count'],
      axis=1)
demand_inbound_merge_val1 = demand_inbound_merge_val.drop(
    ['truck_signin_datetime', 'front_temperature', 'middle_temperature', 'back_temperature', 'case_quantity', 'pallet_count'],
      axis=1)
demand_inbound_merge_test1 = demand_inbound_merge_test.drop(
    ['truck_signin_datetime', 'front_temperature', 'middle_temperature', 'back_temperature', 'case_quantity', 'pallet_count'],
      axis=1)
# If they are NaN, replace them with 0.0, since 0 seconds has passed.
demand_inbound_merge_train2 = demand_inbound_merge_train1.fillna(value=0, axis=1)
demand_inbound_merge_val2 = demand_inbound_merge_val1.fillna(value=0, axis=1)
demand_inbound_merge_test2 = demand_inbound_merge_test1.fillna(value=0, axis=1)

print("train post", demand_inbound_merge_train2.shape)
print("val post", demand_inbound_merge_val2.shape)
print("test post", demand_inbound_merge_test2.shape)





# Merging inbound with the above demand/weather dataset. Maximum 15 minutes difference in order to merge.
# Results in NaN and NaT where > 15 mins difference.
outbound_post_nan.sort_values("truck_signin_datetime", inplace=True)
print("train pre", demand_inbound_merge_train2.shape)
print("val pre", demand_inbound_merge_val2.shape)
print("test pre", demand_inbound_merge_test2.shape, '\n')

demand_inbound_merge_train3 = pd.merge_asof(demand_inbound_merge_train2, outbound_post_nan, 
                                     left_on='datetime_local', 
                                     right_on='truck_signin_datetime', 
                                     direction='nearest', 
                                     suffixes=('_in', '_out'),
                                     tolerance=datetime.timedelta(minutes = 15))
demand_inbound_merge_val3 = pd.merge_asof(demand_inbound_merge_val2, outbound_post_nan, 
                                     left_on='datetime_local', 
                                     right_on='truck_signin_datetime', 
                                     direction='nearest', 
                                     suffixes=('_in', '_out'),
                                     tolerance=datetime.timedelta(minutes = 15))
demand_inbound_merge_test3 = pd.merge_asof(demand_inbound_merge_test2, outbound_post_nan, 
                                     left_on='datetime_local', 
                                     right_on='truck_signin_datetime', 
                                     direction='nearest', 
                                     suffixes=('_in', '_out'),
                                     tolerance=datetime.timedelta(minutes = 15))

# Add only load time and truck time
demand_inbound_merge_train4 = demand_inbound_merge_train3.drop(
    ['truck_signin_datetime', 'case_quantity', 'pallet_count'],
      axis=1)
demand_inbound_merge_val4 = demand_inbound_merge_val3.drop(
    ['truck_signin_datetime', 'case_quantity', 'pallet_count'],
      axis=1)
demand_inbound_merge_test4 = demand_inbound_merge_test3.drop(
    ['truck_signin_datetime', 'case_quantity', 'pallet_count'],
      axis=1)

# If they are NaN, replace them with 0.0, since 0 seconds has passed.
demand_inbound_merge_train5 = demand_inbound_merge_train4.fillna(value=0, axis=1)
demand_inbound_merge_val5 = demand_inbound_merge_val4.fillna(value=0, axis=1)
demand_inbound_merge_test5 = demand_inbound_merge_test4.fillna(value=0, axis=1)

# Sum load time and truck time
demand_inbound_merge_train5['load_time'] = demand_inbound_merge_train5['load_time_in'] + demand_inbound_merge_train5['load_time_out']
demand_inbound_merge_val5['load_time'] = demand_inbound_merge_val5['load_time_in'] + demand_inbound_merge_val5['load_time_out']
demand_inbound_merge_test5['load_time'] = demand_inbound_merge_test5['load_time_in'] + demand_inbound_merge_test5['load_time_out']

demand_inbound_merge_train5['truck_time'] = demand_inbound_merge_train5['truck_time_in'] + demand_inbound_merge_train5['truck_time_out']
demand_inbound_merge_val5['truck_time'] = demand_inbound_merge_val5['truck_time_in'] + demand_inbound_merge_val5['truck_time_out']
demand_inbound_merge_test5['truck_time'] = demand_inbound_merge_test5['truck_time_in'] + demand_inbound_merge_test5['truck_time_out']

demand_inbound_merge_train5['net_weight'] = demand_inbound_merge_train5['net_weight_in'] + demand_inbound_merge_train5['net_weight_out']
demand_inbound_merge_val5['net_weight'] = demand_inbound_merge_val5['net_weight_in'] + demand_inbound_merge_val5['net_weight_out']
demand_inbound_merge_test5['net_weight'] = demand_inbound_merge_test5['net_weight_in'] + demand_inbound_merge_test5['net_weight_out']

demand_inbound_merge_train6 = demand_inbound_merge_train5.drop(
    ['load_time_in', 'load_time_out', 'truck_time_in', 'truck_time_out', "net_weight_in", "net_weight_out"],
      axis=1)
demand_inbound_merge_val6 = demand_inbound_merge_val5.drop(
    ['load_time_in', 'load_time_out', 'truck_time_in', 'truck_time_out', "net_weight_in", "net_weight_out"],
      axis=1)
demand_inbound_merge_test6 = demand_inbound_merge_test5.drop(
    ['load_time_in', 'load_time_out', 'truck_time_in', 'truck_time_out', "net_weight_in", "net_weight_out"],
      axis=1)


print("train post", demand_inbound_merge_train6.shape)
print("val post", demand_inbound_merge_val6.shape)
print("test post", demand_inbound_merge_test6.shape)



demand_inbound_merge_train6['datetime_local'] = demand_inbound_merge_train6['datetime_local'].apply(lambda x: time.mktime(x.timetuple()))
demand_inbound_merge_val6['datetime_local'] = demand_inbound_merge_val6['datetime_local'].apply(lambda x: time.mktime(x.timetuple()))
demand_inbound_merge_test6['datetime_local'] = demand_inbound_merge_test6['datetime_local'].apply(lambda x: time.mktime(x.timetuple()))

train = demand_inbound_merge_train6
val = demand_inbound_merge_val6
test = demand_inbound_merge_test6

print('\n\n ------')
print('train:',train.shape)
print('val:',val.shape)
print('test:',test.shape)

# Save train, val, test
train.to_csv('preprocessed_data/train.csv', index=False)
val.to_csv('preprocessed_data/val.csv', index=False)
test.to_csv('preprocessed_data/test.csv', index=False)

# Save y_val
y_val = val['demand_kW']
y_val.to_csv('output/model_predictions_val.csv', index=False)
