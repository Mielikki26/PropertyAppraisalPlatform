import pandas as pd
import numpy as np
import os
pd.set_option('mode.chained_assignment', None)
dir = r'Research\Datasets'
save_folder = dir + r'\CreatedDatasets\SouthAmericanMoreFeatures_2'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#clean garbage data(empty cells, bad data such as negative prices, etc...)
def clean_garbage_data(df):
    df['Price'] = df['Price'][pd.to_numeric(df['Price'], errors='coerce').notnull()]
    df = df[df['Price'] >= 10000]

    df['Area'] = df['Area'][pd.to_numeric(df['Area'], errors='coerce').notnull()]
    df = df[df['Area'] >= 30]

    df['TotalArea'] = df['TotalArea'][pd.to_numeric(df['TotalArea'], errors='coerce').notnull()]
    df['TotalArea'].fillna(df['Area'], inplace=True)
    df['TotalArea'] = np.where(df['TotalArea'] < df['Area'], df['Area'], df['TotalArea'])

    df['Baths'] = df['Baths'][pd.to_numeric(df['Baths'], errors='coerce').notnull()]
    df = df[df['Baths'] >= 0]

    df['Beds'] = df['Beds'][pd.to_numeric(df['Beds'], errors='coerce').notnull()]
    df = df[df['Beds'] >= 0]

    df['Rooms'] = df['Rooms'][pd.to_numeric(df['Rooms'], errors='coerce').notnull()]
    df['Rooms'].fillna(0, inplace=True)
    df['Rooms'] = np.where(df['Rooms'] < 0, 0, df['Rooms'])

    df['Latitude'] = df['Latitude'][pd.to_numeric(df['Latitude'], errors='coerce').notnull()]
    df['Longitude'] = df['Longitude'][pd.to_numeric(df['Longitude'], errors='coerce').notnull()]
    df['Month'] = df['Month'][pd.to_numeric(df['Month'], errors='coerce').notnull()]
    df['Year'] = df['Year'][pd.to_numeric(df['Year'], errors='coerce').notnull()]

    df['Country'].fillna("-", inplace=True)
    df['Province'].fillna("-", inplace=True)
    df['City'].fillna("-", inplace=True)
    df['PropertyType'].fillna("-", inplace=True)
    df['OperationType'].fillna("-", inplace=True)

    df['Country'] = df['Country'].str.replace(' ', '-')
    df['Province'] = df['Province'].str.replace(' ', '-')
    df['City'] = df['City'].str.replace(' ', '-')
    df['PropertyType'] = df['PropertyType'].str.replace(' ', '-')
    df['OperationType'] = df['OperationType'].str.replace(' ', '-')

    df = df.dropna()
    df = df.drop_duplicates()

    return df

#convert the prices if there is a currency feature attached
def conversionprice(price, currency): #conversion depending on currency feature for the datasets that use it
    if currency == "USD":
        price = price * 0.874
    elif currency == "ARS":
        price = price * 0.00872
    elif currency == "UYU":
        price = price * 0.02
    elif currency == "PEN":
        price = price * 0.217
    elif currency == "COP":
        price = price * 0.000225
    return price

def southamerican_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\ar_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'Month', 'Year',
              'l1', 'l2', 'l3', 'rooms', 'surface_total', 'property_type', 'operation_type']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year',
                   'Country', 'Province', 'City', 'Rooms', 'TotalArea', 'PropertyType', 'OperationType']
    df2 = clean_garbage_data(df2)
    df2 = df2[df2['Latitude'] <= 0]
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\ar properties.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\co_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew', 'surface_covered', 'bathrooms', 'bedrooms', 'lat', 'lon', 'Month', 'Year',
              'l1', 'l2', 'l3', 'rooms', 'surface_total', 'property_type', 'operation_type']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year',
                   'Country', 'Province', 'City', 'Rooms', 'TotalArea', 'PropertyType', 'OperationType']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\co properties.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\ec_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew', 'surface_covered', 'bathrooms', 'bedrooms', 'lat', 'lon', 'Month', 'Year',
              'l1', 'l2', 'l3', 'rooms', 'surface_total', 'property_type', 'operation_type']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year',
                   'Country', 'Province', 'City', 'Rooms', 'TotalArea', 'PropertyType', 'OperationType']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\ec properties.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\pe_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew', 'surface_covered', 'bathrooms', 'bedrooms', 'lat', 'lon', 'Month', 'Year',
              'l1', 'l2', 'l3', 'rooms', 'surface_total', 'property_type', 'operation_type']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year',
                   'Country', 'Province', 'City', 'Rooms', 'TotalArea', 'PropertyType', 'OperationType']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\pe properties.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\uy_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew', 'surface_covered', 'bathrooms', 'bedrooms', 'lat', 'lon', 'Month', 'Year',
              'l1', 'l2', 'l3', 'rooms', 'surface_total', 'property_type', 'operation_type']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year',
                   'Country', 'Province', 'City', 'Rooms', 'TotalArea', 'PropertyType', 'OperationType']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\uy properties.csv', index=False)

southamerican_dataset()