"""
Definition of the labels used in the Random Forest model
 
labels are defines as following:
    -1: Dowmward movement
     0: Stationary
    +1: Upward movement 

Fees:
    Asks: 0.04
    Bids: 0.04


The labels are defined over 0-1000 ticks time horizons:

"""

import numba
import numpy as np
import os
import pandas as pd
import scipy.stats as st
import time

from configurations import basic_parameters, labels_logger as logger

# =============================================================================
# BASIC PARAMETERS
# =============================================================================

directory_of_the_script = os.path.dirname(os.path.abspath(__file__))

lookahead_window    = basic_parameters['lookahead_window']
csv_path            = basic_parameters['csv_path']
drawdown_limit      = basic_parameters['drawdown_limit']
output_filename = f'_Y{lookahead_window}Ticks.csv'


# =============================================================================
# Read CSV file
# =============================================================================


logger.info('start read csvs')
lob_file = directory_of_the_script + csv_path
#df = pd.read_csv(lob_file, usecols=['timestamp', 'asks[0].price', 'bids[0].price'])
df = pd.read_csv(lob_file, usecols=['timestamp', 'ask_price', 'bid_price']) # columns names for book_ticker CSV
logger.info('finish read csvs')


# =============================================================================
# Y LABELS FUNCTION 
# =============================================================================
# outputFileName = '_Y1000Ticks.csv'
# fwdTimeLength = 10
# f = 'binance-futures_book_snapshot_25_2021-01-04_BTCUSDT.csv'

@numba.njit(parallel=True)
def ttp(f_array, lookahead_window):

    # 0 - long profit, 1 - short profit
    profit_arr = np.zeros((f_array.shape[0], 2), dtype=np.float64)

    for open_tick in numba.prange(f_array.shape[0]):
        #if open_tick % 100000 == 0:
            #print('its alive!!!', open_tick, f_array[open_tick])

        open_ask_price = f_array[open_tick, 0]
        open_bid_price = f_array[open_tick, 1]
        price_move_prcnt_long = 0
        price_move_prcnt_short = 0
        max_profit_long = 0
        max_profit_short = 0

        for close_tick in range(open_tick+1, f_array.shape[0]):
            ticks_to_profit = close_tick - open_tick
            if ticks_to_profit > lookahead_window: # lookahead window limitation
                break

            close_ask_price = f_array[close_tick, 0]
            close_bid_price = f_array[close_tick, 1]
            # движение цены по которой продавцы готовы продать актив:
            #price_move_prcnt_long = (close_ask_price-open_ask_price) / (open_ask_price/100)

            # возможный профит при покупке по цене open_ask_price и продаже по цене close_bid_price
            # в процентах от цены покупки
            price_move_prcnt_long = (close_bid_price-open_ask_price) / (open_ask_price/100) 

            # хитро учитываем спред
            #price_move_prcnt_short = (close_bid_price-open_bid_price) / (open_bid_price/100)
            # нормально детектим движение вниз
            #price_move_prcnt_short = (open_bid_price-close_bid_price) / (open_bid_price/100)


            if price_move_prcnt_long > max_profit_long:
                max_profit_long = price_move_prcnt_long
                profit_arr[open_tick, 0] = np.around(max_profit_long, 10)
            elif price_move_prcnt_long < drawdown_limit: # go to next open_tick if drawdown is to big
                break

            

            #if price_move_prcnt_short > max_profit_short:
            #    max_profit_short = price_move_prcnt_short
            #    profit_arr[open_tick, 1] = np.around(max_profit_short, 10)

    return profit_arr

def labels(bp, df):
    lookahead_window        = bp['lookahead_window']
    csv_name                = bp['csv_name']
    lookback_window         = bp['lookback_window']
    data_features           = bp['data_features']

    logger.info(f'start to extract labels from file: {csv_name}')

    
    #logger.info(df)

    df['ask_plus_fee'] = df['ask_price'] + df['ask_price'] * (bp['fee']/100)
    df['bid_plus_fee'] = df['bid_price'] - df['bid_price'] * (bp['fee']/100)

    f_array = df[['ask_plus_fee', 'bid_plus_fee']].to_numpy()

    start_time = time.time() # For profiling only

    profit_arr = ttp(f_array, lookahead_window)

    logger.info('--- %s seconds ---' % round(time.time() - start_time, 2)) # For profiling only
    logger.info(profit_arr[:10])
    logger.info(profit_arr.shape)
    logger.info(f'profit_arr: {profit_arr[:, 0][np.nonzero(profit_arr[:, 0])][:10]} nozero')
    logger.info(f'profit_arr: {profit_arr[:, 0][np.nonzero(profit_arr[:, 0])].shape} shape')


    '''
    start_time = time.time() # For profiling only
    percents = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for p in percents:
        long_profit_prcnt = st.scoreatpercentile(profit_arr[:, 0], p) # Вычисляем персентиль профита лонг
        short_profit_prcnt = st.scoreatpercentile(profit_arr[:, 1], p) # Вычисляем персентиль профита шорт
        format_float_long = '{:.10f}'.format(long_profit_prcnt)
        format_float_short = '{:.10f}'.format(short_profit_prcnt)
        logger.info(f'percent {p} || Вычисляем персентиль профита лонг: {format_float_long}')
        logger.info(f'percent {p} || Вычисляем персентиль профита шорт: {format_float_short}')

    logger.info('--- %s seconds ---' % round(time.time() - start_time, 2)) # For profiling only
    '''

    nozero_arr = profit_arr[:, 0][np.nonzero(profit_arr[:, 0])]
    long_profit_prcnt = st.scoreatpercentile(nozero_arr, 50) # Вычисляем 50й персентиль профита лонг
    logger.info(f'50 percentile of all LONG profits: {long_profit_prcnt}%')
    #short_profit_prcnt = st.scoreatpercentile(profit_arr[:, 1], 50) # Вычисляем 50й персентиль профита шорт
    logger.info(f'max LONG profit: {profit_arr[:, 0].max()}%')
    profit_arr[:, 0][np.where(profit_arr[:, 0]>0.01)]
    logger.info(f'LONG profits > 0.01%: {profit_arr[:, 0][np.where(profit_arr[:, 0]>0.01)].shape} shape')
    logger.info(f'min LONG profit: {profit_arr[:, 0].min()}%')


    #logger.info(f'50 percentile of all SHORT profits: {short_profit_prcnt}%')

    #best_direction_arr = np.where(profit_arr[:, 0]>long_profit_prcnt, 1, np.where(profit_arr[:, 1]>short_profit_prcnt, -1, 0)).reshape(-1, 1)
    best_direction_arr = np.where(profit_arr[:, 0]>long_profit_prcnt, 1, 0).reshape(-1, 1)

    logger.info(f'long profit by percentile > 50: {np.where(profit_arr[:, 0]>long_profit_prcnt, 1, 0).sum()}')

    logger.info(f'best_direction_arr.shape: {best_direction_arr.shape}')
    logger.info(f'profit_arr.shape: {profit_arr.shape}')

    #fin_arr = np.concatenate((best_direction_arr, profit_arr), axis=1)

    logger.info(f'long profit by 1: {np.where(best_direction_arr[:, 0]==1, 1, 0).sum()}')
    logger.info(f'no profit by 0: {np.where(best_direction_arr[:, 0]==0, 1, 0).sum()}')
    #logger.info(f'short profit by -1: {np.where(best_direction_arr[:, 0]==-1, 1, 0).sum()}')


    #----- сохраняем массив в CSV. При этом исключаем lookback_window и lookahead_window
    logger.info(f'make {output_filename}')

    os.makedirs(directory_of_the_script + '/' + data_features, exist_ok=True) # if not exist makedir for CSV files
    labels_file = directory_of_the_script + '/' + data_features + bp['csv_name'][:-4] + f'_Y_la{lookahead_window}Ticks.csv'

    np.savetxt(labels_file, best_direction_arr[lookback_window:-lookahead_window], delimiter=',', fmt='%d')

    del best_direction_arr, df

# =============================================================================
# 3 - RUN LABELS FUNCTION 
# =============================================================================



labels(bp=basic_parameters, df=df)

logger.info('the END')
