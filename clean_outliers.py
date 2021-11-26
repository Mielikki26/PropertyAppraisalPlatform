import os
import pandas as pd
import numpy as np
from scipy.stats import t
cur_dir = os.getcwd()

#Removes the outliers from the dataset passed using IQR score
def remove_outliers_IQR(df, range):
    q1 = df["Price"].quantile(0.25)
    q3 = df["Price"].quantile(0.75)
    iqr = q3-q1
    lb = q1-(iqr * range)
    ub = q3+(iqr * range)
    df = df[(df["Price"] < ub) & (df["Price"] > lb)]

    q1 = df["Area"].quantile(0.25)
    q3 = df["Area"].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - (iqr * range)
    ub = q3 + (iqr * range)
    df = df[(df["Area"] < ub) & (df["Area"] > lb)]

    q1 = df["Beds"].quantile(0.25)
    q3 = df["Beds"].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - (iqr * range)
    ub = q3 + (iqr * range)
    df = df[(df["Beds"] < ub) & (df["Beds"] > lb)]

    q1 = df["Baths"].quantile(0.25)
    q3 = df["Baths"].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - (iqr * range)
    ub = q3 + (iqr * range)
    df = df[(df["Baths"] < ub) & (df["Baths"] > lb)]

    return df

#Removes the outliers from the dataset passed using grubbs test
def remove_outliers_Grubbs(values, alpha, df):
    flag = 0
    indexes = []
    while flag == 0:
        std = values.std()
        mean = values.mean()
        index = np.argmax(abs(values - mean))
        G = max(abs(values - mean)) / std
        N = len(values)
        uppercv_squared = np.square(t.ppf(1 - alpha / (2 * N), N - 2))
        twosided = ((N - 1) / np.sqrt(N)) * np.sqrt(uppercv_squared / (N - 2 + uppercv_squared))
        if G > twosided:
            indexes.append(index)
            values = np.delete(values, index)
        else:
            flag = 1

    for i in indexes:
        df = df.drop(df.index[i])

    return df

#Removes the outliers from the dataset passed using zscore test
def remove_outliers_zscore(label, threshold, df):
    flag = 1
    while(flag):
        values = df[label].to_numpy()
        std = values.std()
        mean = values.mean()
        indexes = []
        for idx, i in enumerate(values):
            z = (i - mean) / std
            if z > threshold:
                indexes.append(idx)

        if indexes == []:
            flag = 0
        else:
            df = df.drop(df.index[indexes])

    return df


dataset_list = os.listdir(cur_dir + r'\Research\Datasets\CreatedDatasets\NewDatasets\\') #get datasets list

for i in dataset_list: #for each dataset
    if '.csv' not in i:
        continue
    new_name = i[:-4] + '_noOutliers.csv'
    df = pd.read_csv(cur_dir + r'\Research\Datasets\CreatedDatasets\NewDatasets\\' + i)

    print("\n------------------------------------------\n")
    print("Dataset " + i + ":")
    print('Inicial dataset shape:' + str(df.shape))

    #df = remove_outliers_IQR(df, 1.5)

    #df = remove_outliers_Grubbs(df["Price"].to_numpy(), 0.05, df)
    #df = remove_outliers_Grubbs(df["Area"].to_numpy(), 0.05, df)
    #df = remove_outliers_Grubbs(df["Beds"].to_numpy(), 0.05, df)
    #df = remove_outliers_Grubbs(df["Baths"].to_numpy(), 0.05, df)

    df = remove_outliers_zscore("Price", 3, df)
    df = remove_outliers_zscore("Area", 3, df)
    df = remove_outliers_zscore("Beds", 3, df)
    df = remove_outliers_zscore("Baths", 3, df)

    print("Outliers were removed! Dataset shape is now: " + str(df.shape))

    if df.shape[0] > 1000:
        df.to_csv(cur_dir + r'\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\' + new_name, index=False)