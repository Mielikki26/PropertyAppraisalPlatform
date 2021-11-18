import numpy as np
import pandas as pd
dir = r'Research\Datasets'

def conversiondate(date, flag):
    newdate = ""
    if flag == 1:
        newdate = date[5] + date[6] + "/" + date[0] + date[1] + date[2] + date[3]
    elif flag == 2:
        newdate = date[0] + date[1] + "/" + date[6] + date[7] + date[8] + date[9]
    elif flag == 3:
        newdate = date[0] + date[1] + "/" + date[3] + date[4] + date[5] + date[6]
    elif flag == 4:
        if len(date) == 9:
            newdate = date[2:]
        else:
            newdate = date[3:]
    elif flag == 5:
        newdate = date[4] + date[5] + "/" + date[0] + date[1] + date[2] + date[3]
    elif flag == 6:
        if pd.isnull(date) is False:
            newdate = date[5] + date[6] + "/" + date[0] + date[1] + date[2] + date[3]
    return newdate

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
    df['start_date'] = df.apply(lambda x: conversiondate(x['start_date'], 1), axis=1)
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'start_date']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Date']
    df2.to_csv(dir + '\CreatedDatasets\\ar_properties_new.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\co_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['start_date'] = df.apply(lambda x: conversiondate(x['start_date'], 1), axis=1)
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'start_date']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Date']
    df2.to_csv(dir + '\CreatedDatasets\\co_properties_new.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\ec_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['start_date'] = df.apply(lambda x: conversiondate(x['start_date'], 1), axis=1)
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'start_date']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Date']
    df2.to_csv(dir + '\CreatedDatasets\\ec_properties_new.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\pe_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['start_date'] = df.apply(lambda x: conversiondate(x['start_date'], 1), axis=1)
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'start_date']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Date']
    df2.to_csv(dir + '\CreatedDatasets\\pe_properties_new.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\uy_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['start_date'] = df.apply(lambda x: conversiondate(x['start_date'], 1), axis=1)
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'start_date']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Date']
    df2.to_csv(dir + '\CreatedDatasets\\uy_properties_new.csv', index=False)

def Melbourne_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\Melbourne_housing_FULL.csv")
    df['Price'] = df['Price'].multiply(0.636)
    df['Date'] = df.apply(lambda x: conversiondate(x['Date'], 4), axis=1)
    df2 = df[['Price','BuildingArea','Bathroom','Bedroom2','Lattitude','Longtitude', 'Date']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Date']
    df2.to_csv(dir + '\CreatedDatasets\Melbourne_housing_FULL_new.csv', index=False)

def kc_house_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\kc_house_data.csv")
    df['Price'] = df['price'].multiply(0.874)
    df['sqft_living'] = df['sqft_living'].multiply(0.09290304)
    df['date'] = df.apply(lambda x: conversiondate(x['date'], 5), axis=1)
    df2 = df[['Price','sqft_living','bathrooms','bedrooms','lat','long', 'date']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Date']
    df2.to_csv(dir + '\CreatedDatasets\kc_house_data_new.csv', index=False)

def DC_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\DC_Properties.csv")
    df['PRICE'] = df['PRICE'].multiply(0.874)
    df['LANDAREA'] = df['LANDAREA'].multiply(0.09290304)
    df['SALEDATE'] = df.apply(lambda x: conversiondate(x['SALEDATE'], 6), axis=1)
    df2 = df[['PRICE','LANDAREA','BATHRM','BEDRM','LATITUDE','LONGITUDE', 'SALEDATE']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Date']
    df2.to_csv(dir + '\CreatedDatasets\DC_Properties_new.csv', index=False)

def zameen():
    df = pd.read_csv(dir + "\OriginalDatasets\\Zameen Property Data Pakistan.csv")
    df['price'] = df['price'].multiply(0.0051)
    df['area_sqft'] = df['area_sqft'].multiply(0.09290304)
    df['date_added'] = df.apply(lambda x: conversiondate(x['date_added'], 2), axis=1)
    df2 = df[['price','area_sqft','baths','bedrooms','latitude','longitude', 'date_added']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Date']
    df2.to_csv(dir + '\CreatedDatasets\Zameen Property Data Pakistan_new.csv', index=False)

def perth():
    df = pd.read_csv(dir + "\OriginalDatasets\\all_perth_310121.csv")
    df['PRICE'] = df['PRICE'].multiply(0.874)
    df['LAND_AREA'] = df['LAND_AREA'].multiply(0.09290304)
    df['DATE_SOLD'] = df.apply(lambda x: conversiondate(x['DATE_SOLD'], 3), axis=1)
    df2 = df[['PRICE','LAND_AREA','BATHROOMS','BEDROOMS','LATITUDE','LONGITUDE', 'DATE_SOLD']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Date']
    df2.to_csv(dir + '\CreatedDatasets\\all_perth_310121_new.csv', index=False)

southamerican_dataset()
Melbourne_dataset()
kc_house_dataset()
DC_dataset()
zameen()
perth()