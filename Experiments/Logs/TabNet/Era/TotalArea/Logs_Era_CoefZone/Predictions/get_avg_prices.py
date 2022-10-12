import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

df = pd.read_csv("Era_Coef_TotalArea.csv")
df2 = df.drop_duplicates('Zone', keep='first')
#df3 = df.drop_duplicates('Parish', keep='first')
df2 = df2[["Zone", "AvgZone"]]
#df3 = df3[["Parish", "AvgParish"]]

df_preds = pd.read_csv("Era_Coef_TotalArea_preds.csv")
df_preds["AvgZone"] = 0.0
#df_preds["AvgParish"] = 0.0
df_preds["Price"] = 0.0
df_preds["PricePredicted"] = 0.0
array = df2.to_numpy()
for idx, j in enumerate(array):
    df_preds.loc[df_preds["Zone"] == j[0], 'AvgZone'] = j[1]

#array = df3.to_numpy()
#for idx, j in enumerate(array):
#    df_preds.loc[df_preds["Parish"] == j[0], 'AvgParish'] = j[1]

df_preds["Price"] = (df_preds["AvgZone"]*df_preds["CoefZone"])*df_preds["TotalArea"]
df_preds["PricePredicted"] = (df_preds["AvgZone"]*df_preds["Predictions"])*df_preds["TotalArea"]
#df_preds["PricePredicted"] = (df_preds["AvgParish"]*df_preds["Predictions"]))*df_preds["Floor area sq meter"]

df_preds["MAPE"] = abs((df_preds["Price"] - df_preds["PricePredicted"])/df_preds["Price"])*100

df_preds.to_csv(r'Era_Coef_TotalArea_preds.csv', index=False)