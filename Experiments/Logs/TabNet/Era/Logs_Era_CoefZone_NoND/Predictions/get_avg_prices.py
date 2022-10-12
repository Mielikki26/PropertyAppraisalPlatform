import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

df = pd.read_csv("Era_Coef_Zone.csv")
df2 = df.drop_duplicates('Zone', keep='first')
df2 = df2[["Zone", "AvgZone"]]

df_preds = pd.read_csv("Era_Coef_Zone_preds.csv")
df_preds["AvgZone"] = 0.0
df_preds["Price"] = 0.0
df_preds["PricePredicted"] = 0.0
array = df2.to_numpy()
for idx, j in enumerate(array):
    df_preds.loc[df_preds["Zone"] == j[0], 'AvgZone'] = j[1]

df_preds["Price"] = (df_preds["AvgZone"]*df_preds["CoefZone"])*df_preds["Floor area sq meter"]
df_preds["PricePredicted"] = (df_preds["AvgZone"]*df_preds["Predictions"])*df_preds["Floor area sq meter"]

df_preds["MAPE"] = abs((df_preds["Price"] - df_preds["PricePredicted"])/df_preds["Price"])*100

df_preds.to_csv(r'Era_Coef_Zone_preds.csv', index=False)