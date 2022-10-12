import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

cur_dir = os.getcwd()

df = pd.read_csv("DataBase_Era.csv")
df2 = df[df['Zone'] != 'N/D']
df3 = df[df['Parish'] != 'N/D']
print(df.shape)
print(df2.shape)
print(df3.shape)