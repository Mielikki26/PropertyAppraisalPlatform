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
    df2 = df[df['MAPE'] >= 1]
    df3 = df[df['MAPE'] < 1]
    print('-----------------------\n'
          'DATASET:     '+ filename)
    print('size:        ' + str(df.shape[0]))
    print('avg MAPE:    ' + str(df['MAPE'].mean()))
    print('avg Price:   ' + str(df['Price'].mean()))
    print('avg MAE:     ' + str(df['MAE'].mean()))
    print('max MAPE:    ' + str(df['MAPE'].max()))
    print()
    print('WITHOUT +1.0 MAPES')
    print('size:        ' + str(df3.shape[0]))
    print('avg MAPE:    ' + str(df3['MAPE'].mean()))
    print('avg Price:   ' + str(df3['Price'].mean()))
    print('avg MAE:     ' + str(df3['MAE'].mean()))
    print()
    print('+1.0 mape:   ' + str(df2.shape[0]) + '  ({:.1f}'.format((df2.shape[0]/df.shape[0])*100) + '%)')
    print('avg Price:   ' + str(df2['Price'].mean()))
    print('avg MAE:     ' + str(df2['MAE'].mean()))
    print('-----------------------')