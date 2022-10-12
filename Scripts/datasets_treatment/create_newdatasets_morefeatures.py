import pandas as pd
import numpy as np
import os
pd.set_option('mode.chained_assignment', None)
dir = r'Research\Datasets'
save_folder = dir + r'\CreatedDatasets\MoreFeatures_noSA_datasets'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#clean garbage data(empty cells, bad data such as negative prices, etc...)
def clean_garbage_data(df):
    features = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    for i in features:
        df[i] = df[i][pd.to_numeric(df[i], errors='coerce').notnull()]

    df = df[df['Price'] >= 10000]
    df = df[df['Area'] >= 30]
    df = df[df['Baths'] >= 0]
    df = df[df['Beds'] >= 0]
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def Melbourne_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\Melbourne_housing_FULL.csv")
    df['Price'] = df['Price'].multiply(0.636)
    df['Month'] = df['Date'].str[-7:-5]
    df['Year'] = df['Date'].str[-4:]
    df2 = df[['Price','BuildingArea','Bathroom','Bedroom2','Lattitude','Longtitude', 'Month', 'Year',
              'Distance','Postcode','Car','Landsize','Propertycount']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year',
                   'Distance','Postcode','Car','Landsize','Propertycount']

    df2['Car'].fillna(0, inplace=True)
    df2['Landsize'].fillna(df2['Area'], inplace=True)
    df2['Landsize'] = np.where(df2['Landsize'] < df2['Area'], df2['Area'], df2['Landsize'])

    features = ['Distance','Postcode','Car','Landsize','Propertycount']
    for i in features:
        df2[i] = df2[i][pd.to_numeric(df2[i], errors='coerce').notnull()]
        df2[i].fillna(0, inplace=True)

    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\Melbourne housing.csv', index=False)

def kc_house_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\kc_house_data.csv")
    df['Price'] = df['price'].multiply(0.874)
    df['sqft_living'] = df['sqft_living'].multiply(0.09290304)
    df['sqft_lot'] = df['sqft_lot'].multiply(0.09290304)
    df['sqft_above'] = df['sqft_above'].multiply(0.09290304)
    df['sqft_basement'] = df['sqft_basement'].multiply(0.09290304)
    df['Month'] = df['date'].str[4:6]
    df['Year'] = df['date'].str[:4]
    df2 = df[['Price','sqft_living','bathrooms','bedrooms','lat','long', 'Month', 'Year',
              'sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_renovated','zipcode']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year',
              'sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_renovated','zipcode']

    df2['sqft_lot'].fillna(df2['Area'], inplace=True)
    df2['sqft_lot'] = np.where(df2['sqft_lot'] < df2['Area'], df2['Area'], df2['sqft_lot'])
    df2['sqft_above'].fillna(0, inplace=True)
    df2['sqft_above'] = np.where(df2['sqft_above'] < 0, 0, df2['sqft_above'])
    df2['sqft_basement'].fillna(0, inplace=True)
    df2['sqft_basement'] = np.where(df2['sqft_basement'] < 0, 0, df2['sqft_basement'])

    features = ['sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_renovated','zipcode']
    for i in features:
        df2[i] = df2[i][pd.to_numeric(df2[i], errors='coerce').notnull()]
        df2[i].fillna(0, inplace=True)

    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\kc house data.csv', index=False)

def DC_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\DC_Properties.csv")
    df['PRICE'] = df['PRICE'].multiply(0.874)
    df['LANDAREA'] = df['LANDAREA'].multiply(0.09290304)
    df['Month'] = df['SALEDATE'].str[5:7]
    df['Year'] = df['SALEDATE'].str[:4]
    df2 = df[['PRICE','LANDAREA','BATHRM','BEDRM','LATITUDE','LONGITUDE', 'Month', 'Year',
              'HF_BATHRM','AC','NUM_UNITS','ROOMS','AYB','YR_RMDL','EYB','STORIES','QUALIFIED',
              'SALE_NUM','KITCHENS','FIREPLACES','SOURCE','ZIPCODE']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year',
                   'HF_BATHRM','AC','NUM_UNITS','ROOMS','AYB','YR_RMDL','EYB','STORIES','QUALIFIED',
                   'SALE_NUM','KITCHENS','FIREPLACES','SOURCE','ZIPCODE']

    df2['AC'].fillna('N', inplace=True)
    df2['AC'] = np.where(df2['AC'] != 'Y', 0, df2['AC'])
    df2['AC'] = np.where(df2['AC'] == 'Y', 1, df2['AC'])

    df2['QUALIFIED'] = np.where(df2['QUALIFIED'] != 'Q', 0, df2['QUALIFIED'])
    df2['QUALIFIED'] = np.where(df2['QUALIFIED'] == 'Q', 1, df2['QUALIFIED'])

    df2['SOURCE'] = np.where(df2['SOURCE'] != 'Residential', 0, df2['SOURCE'])
    df2['SOURCE'] = np.where(df2['SOURCE'] == 'Residential', 1, df2['SOURCE'])

    features = ['HF_BATHRM','NUM_UNITS','ROOMS','AYB','YR_RMDL','EYB','STORIES',
                'SALE_NUM','KITCHENS','FIREPLACES','SOURCE','ZIPCODE']
    for i in features:
        df2[i] = df2[i][pd.to_numeric(df2[i], errors='coerce').notnull()]
        df2[i].fillna(0, inplace=True)

    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\DC Properties.csv', index=False)

def perth():
    df = pd.read_csv(dir + "\OriginalDatasets\\all_perth_310121.csv")
    df['PRICE'] = df['PRICE'].multiply(0.63)
    df[['Month', 'Year']] = df['DATE_SOLD'].str.split('-', 1, expand=True)
    df2 = df[['PRICE','FLOOR_AREA','BATHROOMS','BEDROOMS','LATITUDE','LONGITUDE', 'Month', 'Year',
              'SUBURB', 'GARAGE', 'LAND_AREA', 'BUILD_YEAR', 'CBD_DIST', 'NEAREST_STN_DIST', 'POSTCODE', 'NEAREST_SCH_DIST']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year',
                   'SUBURB', 'GARAGE', 'LAND_AREA', 'BUILD_YEAR', 'CBD_DIST', 'NEAREST_STN_DIST', 'POSTCODE', 'NEAREST_SCH_DIST']

    df2['LAND_AREA'].fillna(df2['Area'], inplace=True)
    df2['LAND_AREA'] = np.where(df2['LAND_AREA'] < df2['Area'], df2['Area'], df2['LAND_AREA'])

    features = ['GARAGE', 'LAND_AREA', 'BUILD_YEAR', 'CBD_DIST', 'NEAREST_STN_DIST', 'POSTCODE', 'NEAREST_SCH_DIST']
    for i in features:
        df2[i] = df2[i][pd.to_numeric(df2[i], errors='coerce').notnull()]
        df2[i].fillna(0, inplace=True)

    df2['SUBURB'].fillna("-", inplace=True)

    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\all perth.csv', index=False)

Melbourne_dataset()
kc_house_dataset()
DC_dataset()
perth()