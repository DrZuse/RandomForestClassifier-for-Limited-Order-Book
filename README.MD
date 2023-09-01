RandomForestClassifier for predicting the direction of the BTCUSDT crypto pair price. Limited Order Book (LOB) data of varying depth is used to generate labels and features.

## configurations.py
Contains all the necessary settings and the path to the LOB CSV file.

## 1_features.py
Reads the LOB CSV and generates features. It uses the mylibs/features.py library, which, in turn, utilizes multiprocessing to distribute the workload among CPU cores.

## Features include absolute levels and deciles ranking compared to the last [lookback_window] ticks:
- Imbalance level
- Derivative per level
- Spread derivative
- Volume spread derivative
- Ask price difference
- Mean price and volume
- Accumulated difference

At the end, all features are saved in the file data/features/{LOB_file_name}_X_lb{lookback_window}_lev{nlevels}.csv.

## 2_labels.py
Labels are chosen based on the desired profit magnitude. If the price movement contributes to profit, the label is set to 1; otherwise, it's set to 0. To improve computation speed, the numba library is used with parallel=True (Numba is 10X faster than pure Python).

## 3_fit.py
The RandomForestClassifier is trained on 80% of the data from LOB. The resulting model is saved using pickle in the data/calibration directory.

## 4_predict.py
Prediction is made on the remaining 20% of the data from LOB.
![alt text](https://github.com/DrZuse/rf-binance_realtime_bookticker_v0.01/blob/master/predict.png?raw=true)
