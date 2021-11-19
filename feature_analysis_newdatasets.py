import os
import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cur_dir = os.getcwd()

arr = os.listdir(cur_dir + '\Research\Datasets\CreatedDatasets\\')

for i in arr:
    if i[-12:] != "newclean.csv":
        continue
    df = pd.read_csv(cur_dir + '\Research\Datasets\CreatedDatasets\\' + i)

    print("\n------------------------------------------\n")
    print("Dataset " + i + ":")
    print('Dataset shape:' + str(df.shape))
    print('Range of value for Price: ' + str(df["Price"].min()) + " - " + str(df["Price"].max()))
    print('Range of value for Area:' + str(df["Area"].min()) + " - " + str(df["Area"].max()))
    print('Range of value for Beds:' + str(df["Beds"].min()) + " - " + str(df["Beds"].max()))
    print('Range of value for Baths:' + str(df["Baths"].min()) + " - " + str(df["Baths"].max()))
    print('Range of value for Latitude:' + str(df["Latitude"].min()) + " - " + str(df["Latitude"].max()))
    print('Range of value for Longitude:' + str(df["Longitude"].min()) + " - " + str(df["Longitude"].max()))
    print('Range of value for Month:' + str(df["Month"].min()) + " - " + str(df["Month"].max()))
    print('Range of value for Year:' + str(df["Year"].min()) + " - " + str(df["Year"].max()))
    print(str(len(df)-len(df.drop_duplicates())) + ' rows are duplicated!')

    if df.shape[0] == 0:
        continue

    correlations = df.corr()
    mask = np.zeros_like(correlations, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(20, 20))
    sns.set_style('white')
    ax = sns.heatmap(correlations*100, annot=True, fmt='.0f', mask=mask, cbar=False)
    ax.set_title("Dataset " + str(i))
    #plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Correlations_before.png')
    plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Correlations_after.png')

    df.hist(figsize=(20, 20))
    pl.suptitle("Dataset " + str(i))
    #plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\FeatureDistribution_before.png')
    plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\FeatureDistribution_after.png')