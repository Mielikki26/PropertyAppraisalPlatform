import os
import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cur_dir = os.getcwd()

def boxplot(list,label): #create and save boxplots of all datasets combined into one
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.boxplot(data=list, x=label, y="Label", ax=ax)
    plt.ticklabel_format(style='plain', axis='x')
    #plt.savefig(cur_dir + r'\Research\Datasets\Plots\boxplots\\' + label + r'Distribution_Outliers.png')
    plt.savefig(cur_dir + r'\Research\Datasets\Plots\boxplots\\' + label + r'Distribution_NoOutliers.png')

def distplot(list, label): #create and save distplots of all datasets combined into one
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 8))
    for idx, i in enumerate(list):
        ax = sns.distplot(i, kde=False, label=label_all[idx])
    plt.ticklabel_format(style='plain')
    plt.legend()
    ax.set(xlabel=label, ylabel='Count')
    #plt.savefig(cur_dir + r'\Research\Datasets\Plots\distplots\\' + label + r'Distribution_Outliers.png')
    plt.savefig(cur_dir + r'\Research\Datasets\Plots\distplots\\' + label + r'Distribution_NoOutliers.png')

#get datasets list
#datasets_list = os.listdir(cur_dir + r'\Research\Datasets\CreatedDatasets\NewDatasets\\')
datasets_list = os.listdir(cur_dir + r'\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\')

#variables to save the joined dataset
df_all = []
list_all = []
label_all = []
price_all = []
area_all = []
baths_all = []
beds_all = []

for i in datasets_list: #for each dataset
    if '.csv' not in i:
        continue

    #df = pd.read_csv(cur_dir + r'\Research\Datasets\CreatedDatasets\NewDatasets\\' + i)
    df = pd.read_csv(cur_dir + r'\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\' + i)

    if df.shape[0] == 0:
        continue

    #correlation matrix:
    #correlations = df.corr()
    #mask = np.zeros_like(correlations, dtype=bool)
    #mask[np.triu_indices_from(mask)] = True
    #plt.figure(figsize=(20, 20))
    #sns.set_style('white')
    #ax = sns.heatmap(correlations, annot=True, fmt='.0f', mask=mask, cbar=False)
    #ax.set_title("Dataset " + str(i))
    #plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Correlations_before.png')
    #plt.savefig(cur_dir + '\Research\Datasets\CreatedDatasets\Outlier_Removal_Images\\' + i[:-4] + '\Correlations_after.png')

    #save plot of histogram for current dataset
    df.hist(figsize=(20, 20))
    pl.suptitle("Dataset " + str(i))
    plt.savefig(cur_dir + r'\Research\Datasets\Plots\distplots\\' + i[:-7] + r'FeatureDistribution.png')

    label_all.append([i])
    price_all.append([df["Price"]])
    area_all.append([df["Area"]])
    baths_all.append([df["Baths"]])
    beds_all.append([df["Beds"]])
    df["Label"] = i
    list_all.append(df)


df_all = pd.concat(list_all)

distplot(price_all, "Price")
boxplot(df_all, "Price")
distplot(area_all, "Area")
boxplot(df_all, "Area")
distplot(baths_all, "Baths")
boxplot(df_all, "Baths")
distplot(beds_all, "Beds")
boxplot(df_all, "Beds")