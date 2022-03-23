import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_datasets(method, datasets_list):
    list_all = []
    for i in datasets_list:  # for each dataset
        if '.csv' not in i:
            continue

        df = pd.read_csv(cur_dir + '\\' + method + '\\' + i)

        if df.shape[0] == 0:
            continue

        df["Dataset"] = i[:-4]
        df["Method"] = method
        list_all.append(df)

    df_all = pd.concat(list_all)
    return df_all

cur_dir = os.getcwd()
grubbs_datasets_list = os.listdir(cur_dir + r'\Grubbs\\')
iqr_datasets_list = os.listdir(cur_dir + r'\IQR\\')
zscore_datasets_list = os.listdir(cur_dir + r'\Zscore\\')

grubbs_datasets = get_datasets("Grubbs", grubbs_datasets_list)
iqr_datasets = get_datasets("IQR", iqr_datasets_list)
zscore_datasets = get_datasets("Zscore", zscore_datasets_list)

final_df = pd.concat([grubbs_datasets, iqr_datasets, zscore_datasets])

fig, ax = plt.subplots(figsize=(18, 9))
sns.boxplot(x="Dataset", y="Area", hue="Method", data=final_df)
ax.set_title("Outlier removal method comparison")
plt.xticks(
        rotation=10,
        horizontalalignment='right',
        fontweight='light',
        fontsize=10
    )
plt.ticklabel_format(style='plain', axis='y')
plt.show()