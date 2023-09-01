
import numpy as np
import os
import pandas as pd

from configurations import basic_parameters, features_logger as logger
#from multiprocessing import Pool, cpu_count
from mylibs import features
#from numba import njit, prange



# =============================================================================
# Basic parameters
# =============================================================================

directory_of_the_script = os.path.dirname(os.path.abspath(__file__))

ticker                     = basic_parameters['ticker']
data_features              = basic_parameters['data_features']
nlevels                    = basic_parameters['nlevels']
lookback_window            = basic_parameters['lookback_window']
lookahead_window           = basic_parameters['lookahead_window']
csv_name                   = basic_parameters['csv_name']
csv_path                   = basic_parameters['csv_path']

theNames = ['exchange', 'symbol', 'timestamp', 'local_timestamp']
start_cols_num = len(theNames)

asks_bids_names = []
for i in range(nlevels):
    asks_bids_names.extend([f'asks[{i}].price', f'asks[{i}].amount', f'bids[{i}].price', f'bids[{i}].amount'])

 

# =============================================================================
# Lists of order book files
# =============================================================================


#----- get files
colums_list = [2, *range(4, start_cols_num+nlevels*4)]
logger.info(f'read colums: {colums_list}')
logger.info('start read LOB csvs')
lob_file = pd.read_csv(directory_of_the_script+csv_path, usecols=colums_list)
logger.info(lob_file[:10])

#columns_titles = ['timestamp', 'ask_price', 'ask_amount', 'bid_price', 'bid_amount'] # to make it like 'book_snapshot' csv
#lob_file = lob_file.reindex(columns = columns_titles)
#logger.info(lob_file[:10])

logger.info('finish read LOB csvs')

data_arr_timestamp = lob_file['timestamp'].values.reshape(len(lob_file), 1)
data_arr = lob_file.iloc[:, 1:].values

logger.info(f'data_arr.shape: {data_arr.shape}') # (4032277, 20)
logger.info(data_arr[:10])

del lob_file

# =============================================================================
# Features
# =============================================================================

#----- Imbalance Level
imbLevel = features.imbalance_level(data_arr, nlevels)
imbLevel = np.array(imbLevel).T
logger.info(f'imbLevel.shape: {imbLevel.shape}')


#----- Derivative per level
imbDer = features.derivative_per_level(data_arr, nlevels, lookback_window)
imbDer = np.array(imbDer).T
logger.info(f'imbDer.shape: {imbDer.shape}')


#----- Mid price
midDer = features.mid_derivative(data_arr, nlevels, lookback_window)
midDer = np.array(midDer).T
logger.info(f'midDer.shape: {midDer.shape}')


#----- Spread
spreadDer = features.spread_derivative(data_arr, nlevels, lookback_window)
spreadDer = np.array(spreadDer).T
logger.info(f'spreadDer.shape: {spreadDer.shape}')


#----- Volume Spread derivatives
volumeDer = features.volume_spread_derivative(data_arr, nlevels, lookback_window)
volumeDer = np.array(volumeDer).T
logger.info(f'volumeDer.shape: {volumeDer.shape}')


#----- Price differences # if 1 level only : ValueError: need at least one array to concatenate
AskPriceDiff = features.ask_price_diff(data_arr, nlevels, lookback_window)
AskPriceDiff = np.concatenate(AskPriceDiff, axis=1)
logger.info(f'AskPriceDiff.shape: {AskPriceDiff.shape}')


#----- Mean Price and Volume
mean_price_and_volume = features.mean_price_and_volume(data_arr, nlevels, lookback_window)
logger.info(f'mean_price_and_volume.shape: {mean_price_and_volume.shape}')


#----- Accumulated differences Price and Volume
acc_price_size_diff = features.accDiff(data_arr, nlevels, lookback_window)
logger.info(f'acc_price_size_diff.shape: {acc_price_size_diff.shape}')


#----- Price and Volume Derivatives # swap
asks_0_price = features.the_big_array(data_arr, nlevels, lookback_window)
asks_0_price = np.array(asks_0_price).T
logger.info(f'asks_0_price.shape: {asks_0_price.shape}')


# =============================================================================
# Собираем все массивы в один большой массив
# =============================================================================

#----- создаем пустые массивы
logger.info('создаем пустые массивы')
big_arr_width = (
    1 + imbLevel.shape[1]
    + imbDer.shape[1]
    + midDer.shape[1]
    + spreadDer.shape[1]
    + volumeDer.shape[1]
    + AskPriceDiff.shape[1]
    + mean_price_and_volume.shape[1]
    + acc_price_size_diff.shape[1]
    + asks_0_price.shape[1]
)
big_arr_height = len(data_arr)
big_arr = np.zeros((big_arr_height, big_arr_width), dtype=np.int64)


logger.info('заполняем пустые массивы')
#----- заполняем пустые массивы
start = 0
end = 1
big_arr[:, start:end] = data_arr_timestamp

start = end
end = end + imbLevel.shape[1]
big_arr[:, start:end] = imbLevel

start = end
end = end + imbDer.shape[1]
big_arr[:, start:end] = imbDer

start = end
end = end + midDer.shape[1]
big_arr[:, start:end] = midDer

start = end
end = end + spreadDer.shape[1]
big_arr[:, start:end] = spreadDer

start = end
end = end + volumeDer.shape[1]
big_arr[:, start:end] = volumeDer

start = end
end = end + AskPriceDiff.shape[1]
big_arr[:, start:end] = AskPriceDiff

start = end
end = end + mean_price_and_volume.shape[1]
big_arr[:, start:end] = mean_price_and_volume

start = end
end = end + acc_price_size_diff.shape[1]
big_arr[:, start:end] = acc_price_size_diff

start = end
end = end + asks_0_price.shape[1]
big_arr[:, start:end] = asks_0_price


# =============================================================================
# сохраняем один большой массив в CSV. При этом исключаем lookback_window и lookahead_window
# =============================================================================
logger.info('сохраняем в CSV')

os.makedirs(data_features, exist_ok=True)

theName = csv_name[:-4] + f'_X_lb{lookback_window}_lev{nlevels}.csv'
np.savetxt(directory_of_the_script+'/'+data_features+theName, big_arr[lookback_window:-lookahead_window], delimiter=',', fmt='%d')
logger.info('finish')
