import os
import pandas as pd
import numpy as np
cur_dir = os.getcwd()

arr = os.listdir(cur_dir + '\Research\Datasets\CreatedDatasets\\')

for i in arr:
    new_name = i[:-7] + 'cleaned.csv'
    df = pd.read_csv(cur_dir + '\Research\Datasets\CreatedDatasets\\' + i)
    df['Price'].replace('  ', np.nan, inplace=True)
    df= df.dropna(subset=['Price'])
    df.to_csv(cur_dir + '\Research\Datasets\CreatedDatasets\\' + new_name, index=False)