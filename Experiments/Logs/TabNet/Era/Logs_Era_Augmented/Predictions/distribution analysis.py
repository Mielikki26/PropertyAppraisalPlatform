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
    bins = np.arange(0, df['MAPE'].max(), 0.15)

    df['MAPE'].plot.hist(stacked=True, bins=bins)
    plt.title('Histogram of MAPE distribution in dataset: ' + filename)
    plt.xlabel('MAPE')
    plt.ylabel('')
    plt.show()