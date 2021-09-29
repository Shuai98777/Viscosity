
import heapq
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import joblib

pkl_model = 'VIS_FCST_auto.pickle'


def load_data(x):
    test = pd.DataFrame(x)
    test = test[['Temperature', 'Pressure8601', 'Current', 'FlowRate']]
    return test


def model(test):
    model = joblib.load(pkl_model)
    pred = model.predict(test)
    return pred


if __name__ == '__main__':
    x = pd.read_csv('test.csv') ###need to change the actual data flow
    test = load_data(x)
    pred = model(test)
    print(pred)
