import numpy as np
import os
import pandas as pd
import pickle

from configurations import fit_logger as logger, basic_parameters
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# Basic parameters
# =============================================================================

data_calibration        = basic_parameters['data_calibration']
ticker                  = basic_parameters['ticker']
data_features           = basic_parameters['data_features']
nlevels                 = basic_parameters['nlevels']
lookahead_window        = basic_parameters['lookahead_window']
lookback_window         = basic_parameters['lookback_window']
csv_name                = basic_parameters['csv_name']
trees                   = basic_parameters['trees']
leafs                   = basic_parameters['leafs']
splits                  = basic_parameters['splits']

directory_of_the_script = os.path.dirname(os.path.abspath(__file__))
features_csv = directory_of_the_script + '/' + data_features + csv_name[:-4] + f'_X_lb{lookback_window}_lev{nlevels}.csv'
labels_csv = directory_of_the_script + '/' + data_features + csv_name[:-4] + f'_Y_la{lookahead_window}Ticks.csv'


# =============================================================================
# Read CSV files
# =============================================================================

features_arr = pd.read_csv(features_csv, header=None, dtype=np.int64).values # int64 - для первой колонки с timestamp, хотя но она и нафиг не нужна
labels_arr = pd.read_csv(labels_csv, header=None, dtype=np.int32).values.reshape(-1)
logger.info(f'features_arr.shape: {features_arr.shape}')
logger.info(f'labels_arr.shape: {labels_arr.shape}')

# =============================================================================
# Fit & save
# =============================================================================
#--- Create RFC


rfc = RandomForestClassifier(
                            n_jobs              = -1,       # How many processors is it allowed to use. -1 means there is no restriction, 1 means it can only use one processor
                            n_estimators        = trees,    # The number of trees in the forest
                            min_samples_leaf    = leafs,    # Minimum number of observations (i.e. samples) in terminal leaf
                            min_samples_split   = splits,   # represents the minimum number of samples (i.e. observations) required to split an internal node. 
                            oob_score           = True,     # This is a random forest cross validation method. It is very similar to leave one out validation technique 
                            max_depth           = None,     # The maximum depth of the tree
                            verbose             = 0,        # To check progress of the estimation
                            max_features        = 'sqrt'    # The number of features to consider when looking for the best split. None = no limit
                        )



# 80% to fit, 20% to test
fit_size = (features_arr.shape[0]//100) * 80

#--- Random forest estimation
logger.info('start fit')
theFit = rfc.fit(features_arr[:fit_size, 1:], labels_arr[:fit_size])
logger.info('end fit')

#--- save fitted model object into a byte stream
os.makedirs(data_calibration, exist_ok=True)
model_dump_file = directory_of_the_script + '/' + data_calibration + f'model_dump_{csv_name[:-4]}_lev{nlevels}_tr{trees}_lb{lookback_window}_la{lookahead_window}Ticks.pickle'
with open(model_dump_file, 'wb') as f:
    pickle.dump(theFit, f)


logger.info('the end')
