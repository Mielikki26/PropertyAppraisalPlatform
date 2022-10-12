import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
cur_dir = os.getcwd()
csvs = os.listdir(str(cur_dir))
for filename in csvs:
    if filename[-4:] != '.csv':
        continue
    df = pd.read_csv(filename, index_col=None, header=0)
    df['MAPE'] = abs((df['Price'] - df['Predictions'])/df['Price'])
    df['MAE'] = abs(df['Price'] - df['Predictions'])
    df.to_csv(filename, index=False)