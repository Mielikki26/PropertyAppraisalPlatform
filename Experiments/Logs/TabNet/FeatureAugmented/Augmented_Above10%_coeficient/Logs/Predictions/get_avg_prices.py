import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

#directories
cur_dir = os.getcwd()
datasets_dir = str(cur_dir)
datasets_list = os.listdir(str(cur_dir))
original_datasets_list = os.listdir(str(cur_dir) + r'\Original\\')

df_all = pd.read_csv("all_data_preds.csv")
df_all["AvgZone"] = 0.0
df_all["Price"] = 0.0
df_all["PricePredicted"] = 0.0

for filename in original_datasets_list:
    if filename not in original_datasets_list:
        continue
    df = pd.read_csv('Original\\' + filename)
    df2 = df.drop_duplicates('suburb', keep='first')
    df2 = df2[["suburb", "Average"]]

    df_preds = pd.read_csv(filename[:-4] + '_preds.csv')
    df_preds["Average"] = 0.0
    df_preds["Price"] = 0.0
    df_preds["PricePredicted"] = 0.0

    for i in df2["suburb"].unique():
        avg=df2.loc[df2['suburb'] == i, 'Average'].iloc[0]
        df_preds.loc[df_preds["suburb"] == i, 'Average'] = avg
        df_all.loc[df_all["suburb"] == i, 'Average'] = avg

    df_preds["Price"] = (df_preds["Average"]*df_preds["Coeficient"])*df_preds["Area"]
    df_preds["PricePredicted"] = (df_preds["Average"]*df_preds["Prediction"])*df_preds["Area"]

    df_preds["MAPE"] = abs((df_preds["Price"] - df_preds["PricePredicted"])/df_preds["Price"])*100

    df_preds.to_csv(filename[:-4] + '_preds.csv', index=False)

df_all["Price"] = (df_all["Average"] * df_all["Coeficient"]) * df_all["Area"]
df_all["PricePredicted"] = (df_all["Average"] * df_all["Prediction"]) * df_all["Area"]
df_all["MAPE"] = abs((df_all["Price"] - df_all["PricePredicted"]) / df_all["Price"]) * 100
df_all.to_csv("all_data_preds.csv", index=False)