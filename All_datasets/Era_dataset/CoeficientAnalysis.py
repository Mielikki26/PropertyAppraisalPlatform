import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

cur_dir = os.getcwd()
fd = r"\Coeficientes\FloorArea(No ND)\\" #files_directory
if not os.path.exists(cur_dir + fd):
    os.makedirs(cur_dir + fd)

df = pd.read_csv("DataBase_Era.csv")
df = df[df['Floor area sq meter'] > 0]
df = df.reset_index(drop=True)
df["Price/meter"] = df["Price"]/df["Floor area sq meter"]

df2 = df[df['Zone'] != 'N/D']
df2["CoefZone"] = 0.0
df2["AvgZone"] = 0.0
for i in df2["Zone"].unique():
    df3 = df2[df2['Zone'] == i]
    avg = df3["Price/meter"].mean()
    df2.loc[df2["Zone"] == i, 'CoefZone'] = (df2['Price/meter'] / avg)
    df2.loc[df2["Zone"] == i, 'AvgZone'] = avg
df2.to_csv(cur_dir + fd + r'\Era_Coef_Zone.csv', index=False)

bins = np.arange(0, 5, 0.05)
df2['CoefZone'].plot.hist(stacked=True, bins=bins)
plt.title('Histogram of CoefZone distribution:')
plt.xlabel('CoefZone')
plt.ylabel('')
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.savefig(cur_dir + fd + r'CoefZone.png')
plt.close()


df2 = df[df['Parish'] != 'N/D']
df2["CoefParish"] = 0.0
df2["AvgParish"] = 0.0
for i in df2["Parish"].unique():
    df3 = df2[df2['Parish'] == i]
    avg = df3["Price/meter"].mean()
    df2.loc[df2["Parish"] == i, 'CoefParish'] = (df2['Price/meter'] / avg)
    df2.loc[df2["Parish"] == i, 'AvgParish'] = avg
df2.to_csv(cur_dir + fd + r'\Era_Coef_Parish.csv', index=False)

bins = np.arange(0, 5, 0.05)
df2['CoefParish'].plot.hist(stacked=True, bins=bins)
plt.title('Histogram of CoefParish distribution:')
plt.xlabel('CoefParish')
plt.ylabel('')
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.savefig(cur_dir + fd + r'CoefParish.png')
plt.close()

