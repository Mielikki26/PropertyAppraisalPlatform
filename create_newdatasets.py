import pandas as pd
dir = r'Research\Datasets'

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
    df2 = df2.apply(pd.to_numeric)
    df2.to_csv(dir + '\CreatedDatasets\\ar_properties_new.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\co_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = df2.apply(pd.to_numeric)
    df2.to_csv(dir + '\CreatedDatasets\\co_properties_new.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\ec_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = df2.apply(pd.to_numeric)
    df2.to_csv(dir + '\CreatedDatasets\\ec_properties_new.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\pe_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = df2.apply(pd.to_numeric)
    df2.to_csv(dir + '\CreatedDatasets\\pe_properties_new.csv', index=False)

    df = pd.read_csv(dir + "\OriginalDatasets\\uy_properties.csv")
    df['pricenew'] = df.apply(lambda x: conversionprice(x['price'], x['currency']), axis=1)
    df['Month'] = df['start_date'].str[5:7]
    df['Year'] = df['start_date'].str[:4]
    df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = df2.apply(pd.to_numeric)
    df2.to_csv(dir + '\CreatedDatasets\\uy_properties_new.csv', index=False)

def Melbourne_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\Melbourne_housing_FULL.csv")
    df['Price'] = df['Price'].multiply(0.636)
    df['Month'] = df['Date'].str[-7:-5]
    df['Year'] = df['Date'].str[-4:]
    df2 = df[['Price','BuildingArea','Bathroom','Bedroom2','Lattitude','Longtitude', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = df2.apply(pd.to_numeric)
    df2.to_csv(dir + '\CreatedDatasets\Melbourne_housing_FULL_new.csv', index=False)

def kc_house_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\kc_house_data.csv")
    df['Price'] = df['price'].multiply(0.874)
    df['sqft_living'] = df['sqft_living'].multiply(0.09290304)
    df['Month'] = df['date'].str[4:6]
    df['Year'] = df['date'].str[:4]
    df2 = df[['Price','sqft_living','bathrooms','bedrooms','lat','long', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = df2.apply(pd.to_numeric)
    df2.to_csv(dir + '\CreatedDatasets\kc_house_data_new.csv', index=False)

def DC_dataset():
    df = pd.read_csv(dir + "\OriginalDatasets\\DC_Properties.csv")
    df['PRICE'] = df['PRICE'].multiply(0.874)
    df['LANDAREA'] = df['LANDAREA'].multiply(0.09290304)
    df['Month'] = df['SALEDATE'].str[5:7]
    df['Year'] = df['SALEDATE'].str[:4]
    df2 = df[['PRICE','LANDAREA','BATHRM','BEDRM','LATITUDE','LONGITUDE', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = df2.apply(pd.to_numeric)
    df2.to_csv(dir + '\CreatedDatasets\DC_Properties_new.csv', index=False)

def zameen():
    df = pd.read_csv(dir + "\OriginalDatasets\\Zameen Property Data Pakistan.csv")
    df['price'] = df['price'].multiply(0.0051)
    df['area_sqft'] = df['area_sqft'].multiply(0.09290304)
    df['Month'] = df['date_added'].str[:2]
    df['Year'] = df['date_added'].str[-4:]
    df2 = df[['price','area_sqft','baths','bedrooms','latitude','longitude', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = df2.apply(pd.to_numeric)
    df2.to_csv(dir + '\CreatedDatasets\Zameen Property Data Pakistan_new.csv', index=False)

def perth():
    df = pd.read_csv(dir + "\OriginalDatasets\\all_perth_310121.csv")
    df['PRICE'] = df['PRICE'].multiply(0.874)
    df['LAND_AREA'] = df['LAND_AREA'].multiply(0.09290304)
    df['Month'] = df['DATE_SOLD'].str[:2]
    df['Year'] = df['DATE_SOLD'].str[-4:]
    df2 = df[['PRICE','LAND_AREA','BATHROOMS','BEDROOMS','LATITUDE','LONGITUDE', 'Month', 'Year']]
    df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude', 'Month', 'Year']
    df2 = df2.apply(pd.to_numeric)
    df2.to_csv(dir + '\CreatedDatasets\\all_perth_310121_new.csv', index=False)

#southamerican_dataset()
#Melbourne_dataset()
#kc_house_dataset()
#DC_dataset()
#zameen()
perth()