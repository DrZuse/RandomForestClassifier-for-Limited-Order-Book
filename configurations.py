import logging
import os


# =============================================================================
# Basic parameters
# =============================================================================

basic_parameters = dict()
basic_parameters['path_to_full_csvs']       = '/../big_dataframes/tardis/'
basic_parameters['exchange']                = 'binance'
basic_parameters['market']                  = 'spot' # spot, futures
#basic_parameters['csv_name']                = 'binance_book_snapshot_25_2023-02-01_BTCUSDT.csv'
basic_parameters['csv_name']                = 'binance_book_ticker_2023-02-01_BTCUSDT.csv'
basic_parameters['ticker']                  = 'BTCUSDT'
basic_parameters['csv_path']                = basic_parameters['path_to_full_csvs'] + basic_parameters['exchange'] + '/' + basic_parameters['market'] + '/' + basic_parameters['csv_name']
basic_parameters['data_features']           = 'data/features/'
basic_parameters['data_calibration']        = 'data/calibration/'
basic_parameters['results_path']            = 'results/'
basic_parameters['nlevels']                 = 1 # 20 is the maximum for binance
basic_parameters['lookback_window']         = 1000
basic_parameters['lookahead_window']        = 1000
#basic_parameters['drawdown_limit']          = 0.005
#basic_parameters['fee']                     = 0.04
basic_parameters['drawdown_limit']          = 0
basic_parameters['fee']                     = 0
basic_parameters['trees']                   = 200
basic_parameters['leafs']                   = 200
basic_parameters['splits']                  = 300



# =============================================================================
# Logs
# =============================================================================

logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

formatter = logging.Formatter('\n\r[%(asctime)s.%(msecs)03d] %(levelname)s [%(filename)s.%(process)d.%(thread)d.%(funcName)s:%(lineno)d]\n\r%(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)

# To setup as many loggers as you want
# example: main_logger = setup_logger('main_logger')
def setup_logger(name, level=logging.DEBUG, logs_dir=logs_dir):
    
    log_file = logs_dir + '/' + name + '.log'

    handler = logging.FileHandler(log_file, mode='w')        
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger

main_logger = setup_logger('main_logger', logging.INFO, logs_dir)
features_logger = setup_logger('features_logger', logging.INFO, logs_dir)
labels_logger = setup_logger('labels_logger', logging.INFO, logs_dir)
fit_logger = setup_logger('fit_logger', logging.INFO, logs_dir)
predict_logger = setup_logger('predict_logger', logging.INFO, logs_dir)


