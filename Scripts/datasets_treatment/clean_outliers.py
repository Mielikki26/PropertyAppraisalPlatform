import os
import pandas as pd
import numpy as np
from scipy.stats import t
cur_dir = os.getcwd()

#Removes the outliers from the dataset passed using IQR score
def remove_outliers_IQR(df, range):
    cols = ['Price', 'Area', 'Beds', 'Baths']

    for i in cols:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1

        df = df[~((df[i] < (Q1 - range * IQR)) | (df[i] > (Q3 + range * IQR)))]

    return df

#Removes the outliers from the dataset passed using grubbs test
def remove_outliers_Grubbs(values, alpha, df):
    flag = 1
    indexes = []
    while(flag):
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
            flag = 0

    for i in indexes:
        df = df.drop(df.index[i])

    return df

#Removes the outliers from the dataset passed using zscore test
def remove_outliers_zscore(label, threshold, df):
    flag = 1
    while(flag):
        indexes = []
        values = df[label].to_numpy()
        std = values.std()
        mean = values.mean()
        for idx, i in enumerate(values):
            z = (i - mean) / std
            if z > threshold:
                indexes.append(idx)

        if indexes == []:
            flag = 0
        else:
            df = df.drop(df.index[indexes])

    return df

datasets_dir = cur_dir + r'\Research\Datasets\CreatedDatasets\NewDatasets1000\\'
save_folder = datasets_dir + r'WithoutOutliers\\'
dataset_list = os.listdir(datasets_dir) #get datasets list

if not os.path.exists(save_folder):
    os.makedirs(save_folder + 'IQR')
    os.makedirs(save_folder + 'Grubbs')
    os.makedirs(save_folder + 'Zscore')

for i in dataset_list: #for each dataset
    if '.csv' not in i:
        continue
    df = pd.read_csv(datasets_dir + i)

    print("\n------------------------------------------\n")
    print("Dataset " + i + ":")
    print('Inicial dataset shape:' + str(df.shape))

    df_new = df.copy()
    df_new = remove_outliers_IQR(df_new, 1.5)

    if df_new.shape[0] > 1000:
        df_new.to_csv(save_folder +  r'IQR\\' + i, index=False)
        print('Final dataset shape for IQR:' + str(df_new.shape))

    df_new = df.copy()
    df_new = remove_outliers_Grubbs(df_new["Price"].to_numpy(), 0.05, df_new)
    df_new = remove_outliers_Grubbs(df_new["Area"].to_numpy(), 0.05, df_new)
    df_new = remove_outliers_Grubbs(df_new["Beds"].to_numpy(), 0.05, df_new)
    df_new = remove_outliers_Grubbs(df_new["Baths"].to_numpy(), 0.05, df_new)

    if df_new.shape[0] > 1000:
        df_new.to_csv(save_folder +  r'Grubbs\\' + i, index=False)
        print('Final dataset shape for Grubbs:' + str(df_new.shape))

    df_new = df.copy()
    df_new = remove_outliers_zscore("Price", 3, df_new)
    df_new = remove_outliers_zscore("Area", 3, df_new)
    df_new = remove_outliers_zscore("Beds", 3, df_new)
    df_new = remove_outliers_zscore("Baths", 3, df_new)

    if df_new.shape[0] > 1000:
        df_new.to_csv(save_folder +  r'Zscore\\' + i, index=False)
        print('Final dataset shape for Zscore:' + str(df_new.shape))