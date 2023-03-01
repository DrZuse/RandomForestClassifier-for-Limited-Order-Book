from configurations import features_logger as logger
from multiprocessing import Pool, cpu_count
from scipy import stats

#import numba
import numpy as np

#####################################

def imbalance_level_pool(args):
    i, queue = args
    #--- Define Imbalance
    asks_column_number = (i*4)+1 # columns 1, 5, 9 ... # asks[...].amount
    bids_column_number = (i*4)+3 # columns 3, 7, 11 ... # bids[...].amount
    asks_amount = queue[:, asks_column_number]
    bids_amount = queue[:, bids_column_number]
    lev = (asks_amount-bids_amount) / (asks_amount+bids_amount) # в предыдущих версиях < 13 была ошибка (bids_amount-asks_amount)
    theLevel = np.round(10 * (lev - lev.min()) / (lev.max() - lev.min()), 0).astype(np.int32) # Deciles
    return theLevel

def imbalance_level(queue, nlevels):
    args = [(i, queue) for i in range(nlevels)]
    with Pool(cpu_count()) as p:
        theLevel = p.map(imbalance_level_pool, args)
    return theLevel

#####################################

def derivative_per_level_pool(args):
    i, queue, lw = args
    asks_column_number = (i*4)+1 # columns 1, 5, 9 ... # asks[...].amount
    bids_column_number = (i*4)+3 # columns 3, 7, 11 ... # bids[...].amount
    asks_amount = queue[:, asks_column_number]
    bids_amount = queue[:, bids_column_number]
    lev = (asks_amount-bids_amount) / (bids_amount+asks_amount) # в предыдущих версиях < 13 была ошибка (bids_amount-asks_amount)
    lev = np.array(lev).T
    #--- Define Imbalance Derivative
    # start = (np.abs(k-lw)+k-lw)/2 
    # if k-lookback_window > 0: start = k-lookback_window else start = 0
    theDerivative = [np.round(stats.percentileofscore(lev[(np.abs(k-lw)+k-lw)//2:k], lev[k])/10, 0).astype(np.int8) for k in range(len(queue))] # Deciles
    return theDerivative

def derivative_per_level(queue, nlevels, lookback_window):
    args = [(i, queue, lookback_window) for i in range(nlevels)]
    with Pool(cpu_count()) as p:
        theDerivative = p.map(derivative_per_level_pool, args)
    return theDerivative

#####################################

def mid_derivative_pool(args):
    i, queue, lw = args
    asks_column_number = (i*4)+0 # columns 0, 4, 8 ... # asks[...].price
    bids_column_number = (i*4)+2 # columns 2, 6, 10 ... # bids[...].price
    asks_price = queue[:, asks_column_number]
    bids_price = queue[:, bids_column_number]
    #--- Define Levels
    mid = (bids_price+asks_price)/2
    #--- Define Derivatives
    theMidDerivative = [np.round(stats.percentileofscore(mid[(np.abs(k-lw)+k-lw)//2:k], mid[k])/10, 0).astype(np.int8) for k in range(len(queue))] # Deciles
    return theMidDerivative
    
def mid_derivative(queue, nlevels, lookback_window):
    args = [(i, queue, lookback_window) for i in range(nlevels)]
    with Pool(cpu_count()) as p:
        theMidDerivative = p.map(mid_derivative_pool, args)
    return theMidDerivative

#####################################

def spread_derivative_pool(args):
    i, queue, lw = args
    asks_column_number = (i*4)+0 # columns 0, 4, 8 ... # asks[...].price
    bids_column_number = (i*4)+2 # columns 2, 6, 10 ... # bids[...].price
    asks_price = queue[:, asks_column_number]
    bids_price = queue[:, bids_column_number]
    #--- Define Levels
    spread = bids_price - asks_price
    #--- Define Derivatives
    theSpreadDerivative = [np.round(stats.percentileofscore(spread[(np.abs(k-lw)+k-lw)//2:k], spread[k])/10, 0).astype(np.int8) for k in range(len(queue))] # Deciles
    return theSpreadDerivative

def spread_derivative(queue, nlevels, lookback_window):
    args = [(i, queue, lookback_window) for i in range(nlevels)]
    with Pool(cpu_count()) as p:
        theSpreadDerivative = p.map(spread_derivative_pool, args)
    return theSpreadDerivative

#####################################

def volume_spread_derivative_pool(args):
    i, queue, lw = args
    asks_column_number = (i*4)+1 # columns 1, 5, 9 ... # asks[...].amount
    bids_column_number = (i*4)+3 # columns 3, 7, 11 ... # bids[...].amount
    asks_amount = queue[:, asks_column_number]
    bids_amount = queue[:, bids_column_number]
    #--- Define Levels
    volumeSpread = bids_amount - asks_amount
    #--- Define Derivatives
    theVolumeSpreadDerivative = [np.round(stats.percentileofscore(volumeSpread[(np.abs(k-lw)+k-lw)//2:k], volumeSpread[k])/10, 0).astype(np.int8) for k in range(len(queue))] # Deciles
    return theVolumeSpreadDerivative

def volume_spread_derivative(queue, nlevels, lookback_window):
    args = [(i, queue, lookback_window) for i in range(nlevels)]
    with Pool(cpu_count()) as p:
        theVolumeSpreadDerivative = p.map(volume_spread_derivative_pool, args)
    return theVolumeSpreadDerivative
    
#####################################

def ask_price_diff_pool(args):
    i, queue, lw = args
    #--- Define asks
    asks_column_number = (i*4)+0 # columns 0, 4, 8 ... # asks[{i}].price
    asks_column_number_1 = (i*4)+4 # columns 4, 8, 12 ... # asks[{i+1}].price
    asks_price = queue[:, asks_column_number]
    asks_price_1 = queue[:, asks_column_number_1]
    aa = np.array(np.abs(asks_price_1+asks_price))
    theAskDiffDerivative = [np.round(stats.percentileofscore(aa[(np.abs(k-lw)+k-lw)//2:k], aa[k])/10, 0).astype(np.int8) for k in range(len(queue))] # Deciles
    theAskDiffDerivative = np.array(theAskDiffDerivative).reshape(len(queue), 1)
    #--- Define bids
    bids_column_number = (i*4)+2 # columns 2, 6, 10 ... # bids[{i}].price
    bids_column_number_1 = (i*4)+4 # columns 6, 10, 14 ... # bids[{i+1}].price
    bids_price = queue[:, bids_column_number]
    bids_price_1 = queue[:, bids_column_number_1]
    bb = np.array(np.abs(bids_price_1+bids_price))
    theBidDiffDerivative = [np.round(stats.percentileofscore(bb[(np.abs(k-lw)+k-lw)//2:k], bb[k])/10, 0).astype(np.int8) for k in range(len(queue))] # Deciles
    theBidDiffDerivative = np.array(theBidDiffDerivative).reshape(len(queue), 1)
    #--- result
    AskPriceDiff = np.concatenate((theAskDiffDerivative, theBidDiffDerivative), axis=1)
    return AskPriceDiff

def ask_price_diff(queue, nlevels, lookback_window):
    args = [(i, queue, lookback_window) for i in range(nlevels-1)]
    with Pool(cpu_count()) as p:
        AskPriceDiff = p.map(ask_price_diff_pool, args)
    return AskPriceDiff

#####################################

def mean_pool(args):
    k, mean, lw = args
    return np.round(stats.percentileofscore(mean[(np.abs(k-lw)+k-lw)//2:k], mean[k])/10, 0).astype(np.int8)

def mean_price_and_volume(queue, nlevels, lookback_window):
    for i in range(nlevels):
        #i = 0
        asks_price_column_number = (i*4)+0 # columns 0, 4, 8 ... # asks[...].price
        asks_price = queue[:, asks_price_column_number]
        bids_price_column_number = (i*4)+2 # columns 2, 6, 10 ... # bids[...].price
        bids_price = queue[:, bids_price_column_number]

        asks_amount_column_number = (i*4)+1 # columns 1, 5, 9 ... # asks[...].amount
        asks_amount = queue[:, asks_amount_column_number]
        bids_amount_column_number = (i*4)+3 # columns 3, 7, 11 ... # bids[...].amount
        bids_amount = queue[:, bids_amount_column_number]

        if i == 0:
            dtAccAsk = asks_price
            dtAccBid = bids_price
            dtAccAskVol = asks_amount
            dtAccBidVol = bids_amount
        
        if i != 0:
            dtAccAsk = dtAccAsk + asks_price
            dtAccBid = dtAccBid + bids_price
            dtAccAskVol = dtAccAskVol + asks_amount
            dtAccBidVol = dtAccBidVol + bids_amount

    meanAsk = np.array(dtAccAsk/nlevels) 
    meanBid = np.array(dtAccBid/nlevels)
    meanAskSize = np.array(dtAccAskVol/nlevels)
    meanBidSize = np.array(dtAccBidVol/nlevels)

    #aa = [np.round(stats.percentileofscore(meanAsk[:k], meanAsk[k])/10, 0).astype(np.int32) for k in range(len(queue))] # Deciles
    meanAsk_args = [(k, meanAsk, lookback_window) for k in range(len(queue))]
    with Pool(cpu_count()) as p:
        meanAsk = p.map(mean_pool, meanAsk_args)
    meanAsk = np.array(meanAsk).reshape(len(queue), 1)

    meanBid_args = [(k, meanBid, lookback_window) for k in range(len(queue))]
    with Pool(cpu_count()) as p:
        meanBid = p.map(mean_pool, meanBid_args)
    meanBid = np.array(meanBid).reshape(len(queue), 1)

    meanAskSize_args = [(k, meanAskSize, lookback_window) for k in range(len(queue))]
    with Pool(cpu_count()) as p:
        meanAskSize = p.map(mean_pool, meanAskSize_args)
    meanAskSize = np.array(meanAskSize).reshape(len(queue), 1)

    meanBidSize_args = [(k, meanBidSize, lookback_window) for k in range(len(queue))]
    with Pool(cpu_count()) as p:
        meanBidSize = p.map(mean_pool, meanBidSize_args)
    meanBidSize = np.array(meanBidSize).reshape(len(queue), 1)

    mean_price_and_volume = np.concatenate((meanAsk, meanBid, meanAskSize, meanBidSize), axis=1)

    return mean_price_and_volume

#####################################

def accDiff_pool(args):
    k, acc_diff, lw = args
    return np.round(stats.percentileofscore(acc_diff[(np.abs(k-lw)+k-lw)//2:k], acc_diff[k])/10, 0).astype(np.int8)

def accDiff(queue, nlevels, lookback_window):
    acc_price_diff = np.zeros(len(queue), dtype=np.int8)
    acc_size_diff = np.zeros(len(queue), dtype=np.int8)

    for i in range(nlevels):
        #i = 0
        asks_price_column_number = (i*4)+0 # columns 0, 4, 8 ... # asks[...].price
        asks_price = (queue[:, asks_price_column_number]) * 100 # *100 for decimal fix. TODO: ajust zeros
        bids_price_column_number = (i*4)+2 # columns 2, 6, 10 ... # bids[...].price
        bids_price = (queue[:, bids_price_column_number]) * 100

        asks_amount_column_number = (i*4)+1 # columns 1, 5, 9 ... # asks[...].amount
        asks_amount = (queue[:, asks_amount_column_number]) * 100_000 # *100 for decimal fix. TODO: ajust zeros
        bids_amount_column_number = (i*4)+3 # columns 3, 7, 11 ... # bids[...].amount
        bids_amount = (queue[:, bids_amount_column_number]) * 100_000

        acc_price_diff = acc_price_diff + (bids_price - asks_price)
        acc_size_diff = acc_size_diff + (bids_amount - asks_amount)

    acc_price_diff = acc_price_diff // 100 # / 100 for decimal fix. TODO: ajust zeros
    acc_size_diff = acc_size_diff // 100_000

    accPriceDiff_args = [(k, acc_price_diff, lookback_window) for k in range(len(queue))]
    with Pool(cpu_count()) as p:
        accPriceDiff = p.map(accDiff_pool, accPriceDiff_args)
    accPriceDiff = np.array(accPriceDiff).reshape(len(queue), 1)

    accSizeDiff_args = [(k, acc_size_diff, lookback_window) for k in range(len(queue))]
    with Pool(cpu_count()) as p:
        accSizeDiff = p.map(accDiff_pool, accSizeDiff_args)
    accSizeDiff = np.array(accSizeDiff).reshape(len(queue), 1)

    acc_price_size_diff = np.concatenate((accPriceDiff, accSizeDiff), axis=1)
    return acc_price_size_diff
    
#####################################

def the_big_array_pool(args):
    i, queue, lw = args
    #   i = 0
    #   j = 0
    theBigArray = [np.round(stats.percentileofscore(queue[(np.abs(k-lw)+k-lw)//2:k, i], queue[k, i])/10, 0).astype(np.int8) for k in range(len(queue))]
    return theBigArray

def the_big_array(queue, nlevels, lookback_window):
    args = [(i, queue, lookback_window) for i in range(nlevels*4)]
    with Pool(cpu_count()) as p:
        theBigArray = p.map(the_big_array_pool, args)
    return theBigArray

#####################################