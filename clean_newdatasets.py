import os
import pandas as pd
import numpy as np
cur_dir = os.getcwd()

def cleannegatives(value, flag):
    if flag == 1 and value < 0:
        return np.nan
    elif flag == 2 and value <= 0:
        return np.nan
    else:
        return value

def remove_outliers(df):
    q1 = df["Price"].quantile(0.25)
    q3 = df["Price"].quantile(0.75)
    iqr = q3-q1
    lb = q1-(iqr*1.5)
    ub = q3+(iqr*1.5)
    df = df[(df["Price"] < ub) & (df["Price"] > lb)]

    q1 = df["Area"].quantile(0.25)
    q3 = df["Area"].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - (iqr * 1.5)
    ub = q3 + (iqr * 1.5)
    df = df[(df["Area"] < ub) & (df["Area"] > lb)]

    q1 = df["Beds"].quantile(0.25)
    q3 = df["Beds"].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - (iqr * 1.5)
    ub = q3 + (iqr * 1.5)
    df = df[(df["Beds"] < ub) & (df["Beds"] > lb)]

    q1 = df["Baths"].quantile(0.25)
    q3 = df["Baths"].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - (iqr * 1.5)
    ub = q3 + (iqr * 1.5)
    df = df[(df["Baths"] < ub) & (df["Baths"] > lb)]

    return df


arr = os.listdir(cur_dir + '\Research\Datasets\CreatedDatasets\\')

for i in arr:
    if i[-7:] != "new.csv":
        continue
    new_name = i[:-4] + 'clean.csv'
    df = pd.read_csv(cur_dir + '\Research\Datasets\CreatedDatasets\\' + i)

    print("\n------------------------------------------\n")
    print("Dataset " + i + ":")
    print('Inicial dataset shape:' + str(df.shape))

    df['Price'].replace('  ', np.nan, inplace=True)
    df['Price'] = df.apply(lambda x: cleannegatives(x['Price'],2), axis=1)
    df = df.dropna(subset=['Price'])
    df['Area'].replace('  ', np.nan, inplace=True)
    df['Area'] = df.apply(lambda x: cleannegatives(x['Area'], 2), axis=1)
    df = df.dropna(subset=['Area'])
    df['Baths'].replace('  ', 0, inplace=True)
    df['Baths'] = df.apply(lambda x: cleannegatives(x['Baths'], 1), axis=1)
    df['Beds'].replace('  ', 0, inplace=True)
    df['Beds'] = df.apply(lambda x: cleannegatives(x['Beds'], 1), axis=1)
    df['Latitude'].replace('  ', np.nan, inplace=True)
    df = df.dropna(subset=['Latitude'])
    df['Longitude'].replace('  ', np.nan, inplace=True)
    df = df.dropna(subset=['Longitude'])
    df['Month'].replace('  ', np.nan, inplace=True)
    df = df.dropna(subset=['Month'])
    df['Year'].replace('  ', np.nan, inplace=True)
    df = df.dropna(subset=['Year'])
    print('Dropped rows with empty features:' + str(df.shape))

    df = df.drop_duplicates()
    print('Dropped duplicates:' + str(df.shape))

    df = remove_outliers(df)
    print('After outlier removal: ' + str(df.shape))

    if df.shape[0] > 1000:
        df.to_csv(cur_dir + '\Research\Datasets\CreatedDatasets\\' + new_name, index=False)