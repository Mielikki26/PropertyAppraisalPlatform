import os
import pandas as pd
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
    print('Unique date values: ' + str(df['Date'].unique()))
    print(str(len(df)-len(df.drop_duplicates())) + ' rows are duplicated!')



    ax = sns.violinplot(x='Price', data=df)
    ax.set_title("Dataset " + str(i))
    #plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Price_before.png')
    plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Price_after.png')

    ax = sns.violinplot(x='Area', data=df)
    ax.set_title("Dataset " + str(i))
    #plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Area_before.png')
    plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Area_after.png')

    ax = sns.violinplot(x='Beds', data=df)
    ax.set_title("Dataset " + str(i))
    #plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Beds_before.png')
    plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Beds_after.png')

    ax = sns.violinplot(x='Baths', data=df)
    ax.set_title("Dataset " + str(i))
    #plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Baths_before.png')
    plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Baths_after.png')

    ax = sns.violinplot(x='Latitude', data=df)
    ax.set_title("Dataset " + str(i))
    #plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Latitude_before.png')
    plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Latitude_after.png')

    ax = sns.violinplot(x='Longitude', data=df)
    ax.set_title("Dataset " + str(i))
    #plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Longitude_before.png')
    plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Longitude_after.png')