import numpy as np
import os
import pandas as pd
import pickle

from configurations import predict_logger as logger, basic_parameters
from sklearn.ensemble import RandomForestClassifier # needed to use model aploaded from pickle byte stream


# =============================================================================
# Basic parameters
# =============================================================================

data_calibration        = basic_parameters['data_calibration']
ticker                  = basic_parameters['ticker']
data_features           = basic_parameters['data_features']
nlevels                 = basic_parameters['nlevels']
drawdown_limit          = basic_parameters['drawdown_limit']
fee                     = basic_parameters['fee']
lookahead_window        = basic_parameters['lookahead_window']
lookback_window         = basic_parameters['lookback_window']
csv_name                = basic_parameters['csv_name']
results_path            = basic_parameters['results_path']
trees                   = basic_parameters['trees']

directory_of_the_script = os.path.dirname(os.path.abspath(__file__))

features_csv = directory_of_the_script + '/' + data_features + csv_name[:-4] + f'_X_lb{lookback_window}_lev{nlevels}.csv'
labels_csv = directory_of_the_script + '/' + data_features + csv_name[:-4] + f'_Y_la{lookahead_window}Ticks_lb{lookback_window}.csv'

# =============================================================================
# Read CSV files
# =============================================================================

features_arr = pd.read_csv(features_csv, header=None, dtype=np.int64).values # int64 - для первой колонки с timestamp, хотя но она и нафиг не нужна
labels_arr = pd.read_csv(labels_csv, header=None, dtype=np.int32).values
logger.info(f'features_arr.shape: {features_arr.shape}')
logger.info(f'labels_arr.shape: {labels_arr.shape}')


# =============================================================================
# Calibration & test
# =============================================================================

#--- load fitted model object from a byte stream
#model_dump_file = directory_of_the_script + data_calibration + f'{ticker}_model_dump_tr{trees}_lb{lookback_window}_la{lookahead_window}Ticks.pickle'
model_dump_file = directory_of_the_script + '/' + data_calibration + f'model_dump_{csv_name[:-4]}_lev{nlevels}_tr{trees}_lb{lookback_window}_la{lookahead_window}Ticks.pickle'
with open(model_dump_file, 'rb') as f:
    theFit = pickle.load(f)

# 80% to fit, 20% to test
fit_size = (features_arr.shape[0]//100) * 80

logger.info('start predict_proba(features_arr[fit_size:, 1:])')
pp_fit = theFit.predict_proba(features_arr[fit_size:, 1:])
logger.info(pp_fit.shape) # Probabilities: pp_fit[0] - noprofit | pp_fit[1] profit 
logger.info(pp_fit[:10])
logger.info('end predict_proba(features_arr[fit_size:, 1:])')

labels_arr = labels_arr[fit_size:] # отсекаем fit часть


# =============================================================================
# Gather results
# =============================================================================

newDtTest = np.concatenate([pp_fit, labels_arr], axis=1)
logger.info(newDtTest.shape)


pp_res_file = directory_of_the_script + '/' + results_path + f'results_PP_{csv_name[:-4]}_lb{lookback_window}_la{lookahead_window}_lv{nlevels}_fee{fee}_dd{drawdown_limit}.csv'
os.makedirs(results_path, exist_ok=True)
np.savetxt(pp_res_file, newDtTest, delimiter=',')

print(newDtTest[:10])


# UP - price go up, ST - price stay still, DW - price go down
# pp - predict_proba
# UP_real_cnt - real UP labels '1'
# UP_ratio - ratio of correct predictions

treshold = [*range(5, 100, 5), *range(96, 100)]
collect_results = pd.DataFrame(
                            index = range(len(treshold)),
                            columns = [
                                'treshold',
                                'UP_ratio', 'UP_pp_cnt', 'UP_pp_OK', 'UP_real_cnt',
                                'ST_ratio', 'ST_pp_cnt', 'ST_pp_OK', 'ST_real_cnt',
                            ]
                        )

for i, thp in enumerate(treshold):
    th = thp/100

    #---- up
    UP_pp_cnt = np.where(pp_fit[:, 1] > th, 1, 0).sum()
    UP_pp_OK = np.where((pp_fit[:, 1] > th) & (labels_arr[:, 0]==1), 1, 0).sum()

    if UP_pp_cnt > 0:
        UP_ratio = UP_pp_OK / UP_pp_cnt
    else:
        UP_ratio = None

    UP_real_cnt = np.where(labels_arr[:, 0]==1, 1, 0).sum()

    #---- still
    ST_pp_cnt = np.where(pp_fit[:, 0] > th, 1, 0).sum()
    ST_pp_OK = np.where((pp_fit[:, 0] > th) & (labels_arr[:, 0]==0), 1, 0).sum()

    if ST_pp_cnt > 0:
        ST_ratio = ST_pp_OK / ST_pp_cnt
    else:
        ST_ratio = None

    ST_real_cnt = np.where(labels_arr[:, 0]==0, 1, 0).sum()

    #---- down
    '''
    DW_pp_cnt = np.where(pp_fit[:, 0] > th, 1, 0).sum()
    DW_pp_OK = np.where((pp_fit[:, 0] > th) & (labels_arr[:, 0]==-1), 1, 0).sum()

    if DW_pp_cnt > 0:
        DW_ratio = DW_pp_OK / DW_pp_cnt
    else:
        DW_ratio = None

    DW_real_cnt = np.where(labels_arr[:, 0]==-1, 1, 0).sum()
    '''

    collect_results.iloc[i, 0] = f'{thp}%'

    collect_results.iloc[i, 1] = UP_ratio
    collect_results.iloc[i, 2] = UP_pp_cnt
    collect_results.iloc[i, 3] = UP_pp_OK
    collect_results.iloc[i, 4] = UP_real_cnt

    collect_results.iloc[i, 5] = ST_ratio
    collect_results.iloc[i, 6] = ST_pp_cnt
    collect_results.iloc[i, 7] = ST_pp_OK
    collect_results.iloc[i, 8] = ST_real_cnt

    '''
    collect_results.iloc[i, 9] = DW_ratio
    collect_results.iloc[i, 10] = DW_pp_cnt
    collect_results.iloc[i, 11] = DW_pp_OK
    collect_results.iloc[i, 12] = DW_real_cnt
    '''


print(collect_results)
collect_results_file = directory_of_the_script + '/' + results_path + f'results_TH_{csv_name[:-4]}_lb{lookback_window}_la{lookahead_window}_lev{nlevels}_fee{fee}_dd{drawdown_limit}.csv'
collect_results.to_csv(collect_results_file, header=True, index=False)