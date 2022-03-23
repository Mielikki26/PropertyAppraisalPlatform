import pandas as pd
import numpy as np
import os
pd.set_option('mode.chained_assignment', None)
dir = r'Research\Datasets'
save_folder = dir + r'\CreatedDatasets\NewDatasets'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#clean garbage data(empty cells, bad data such as negative prices, etc...)
def clean_garbage_data(df):
    df['Price'] = df['Price'][pd.to_numeric(df['Price'], errors='coerce').notnull()]
    df = df[df['Price'] >= 10000]
    df['Area'] = df['Area'][pd.to_numeric(df['Area'], errors='coerce').notnull()]
    df = df[df['Area'] >= 30]
    df['Baths'] = df['Baths'][pd.to_numeric(df['Baths'], errors='coerce').notnull()]
    df = df[df['Baths'] >= 0]
    df['Beds'] = df['Beds'][pd.to_numeric(df['Beds'], errors='coerce').notnull()]
    df = df[df['Beds'] >= 0]
    df['Latitude'] = df['Latitude'][pd.to_numeric(df['Latitude'], errors='coerce').notnull()]
    df['Longitude'] = df['Longitude'][pd.to_numeric(df['Longitude'], errors='coerce').notnull()]
    df['Month'] = df['Month'][pd.to_numeric(df['Month'], errors='coerce').notnull()]
    df['Year'] = df['Year'][pd.to_numeric(df['Year'], errors='coerce').notnull()]
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
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = clean_garbage_data(df2)
    df2 = df2[df2['Latitude'] <= 0]
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\ar properties.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\co_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\co properties.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\ec_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\ec properties.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\pe_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\pe properties.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\uy_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\uy properties.csv', index=False)

def Melbourne_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\Melbourne_housing_FULL.csv")
    df['Price'] = df['Price'].multiply(0.636)
    df['Month'] = df['Date'].str[-7:-5]
    df['Year'] = df['Date'].str[-4:]
    df2 = df[['Price','BuildingArea','Bathroom','Bedroom2','Lattitude','Longtitude', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\Melbourne housing.csv', index=False)

def kc_house_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\kc_house_data.csv")
    df['Price'] = df['price'].multiply(0.874)
    df['sqft_living'] = df['sqft_living'].multiply(0.09290304)
    df['Month'] = df['date'].str[4:6]
    df['Year'] = df['date'].str[:4]
    df2 = df[['Price','sqft_living','bathrooms','bedrooms','lat','long', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\kc house data.csv', index=False)

def DC_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\DC_Properties.csv")
    df['PRICE'] = df['PRICE'].multiply(0.874)
    df['LANDAREA'] = df['LANDAREA'].multiply(0.09290304)
    df['Month'] = df['SALEDATE'].str[5:7]
    df['Year'] = df['SALEDATE'].str[:4]
    df2 = df[['PRICE','LANDAREA','BATHRM','BEDRM','LATITUDE','LONGITUDE', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\DC Properties.csv', index=False)

def zameen():
    df = pd.read_csv(dir + "\OriginalDatasets\\Zameen Property Data Pakistan.csv")
    df['price'] = df['price'].multiply(0.0051)
    df['area_sqft'] = df['area_sqft'].multiply(0.09290304)
    df['Month'] = df['date_added'].str[:2]
    df['Year'] = df['date_added'].str[-4:]
    df2 = df[['price','area_sqft','baths','bedrooms','latitude','longitude', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
       df2.to_csv(save_folder + r'\Zameen Data.csv', index=False)

def perth():
    df = pd.read_csv(dir + "\OriginalDatasets\\all_perth_310121.csv")
    df['PRICE'] = df['PRICE'].multiply(0.63)
    df[['Month', 'Year']] = df['DATE_SOLD'].str.split('-', 1, expand=True)
    df2 = df[['PRICE','FLOOR_AREA','BATHROOMS','BEDROOMS','LATITUDE','LONGITUDE', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = clean_garbage_data(df2)
    if df2.shape[0] > 1000:
        df2.to_csv(save_folder + r'\all perth.csv', index=False)

southamerican_dataset()
Melbourne_dataset()
kc_house_dataset()
DC_dataset()
zameen()
perth()