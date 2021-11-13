import pandas as pd
dir = r'Research\Datasets'

def conversion(price, currency): #conversion depending on currency feature for the datasets that use it
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

df = pd.read_csv(dir + "\OriginalDatasets\\ar_properties.csv")
df['pricenew'] = df.apply(lambda x: conversion(x['price'], x['currency']), axis=1)
df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon']]
df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude']
df2.to_csv(dir + '\CreatedDatasets\\ar_properties_new.csv', index=False)

df = pd.read_csv(dir + "\OriginalDatasets\\co_properties.csv")
df['pricenew'] = df.apply(lambda x: conversion(x['price'], x['currency']), axis=1)
df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon']]
df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude']
df2.to_csv(dir + '\CreatedDatasets\\co_properties_new.csv', index=False)

df = pd.read_csv(dir + "\OriginalDatasets\\ec_properties.csv")
df['pricenew'] = df.apply(lambda x: conversion(x['price'], x['currency']), axis=1)
df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon']]
df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude']
df2.to_csv(dir + '\CreatedDatasets\\ec_properties_new.csv', index=False)

df = pd.read_csv(dir + "\OriginalDatasets\\pe_properties.csv")
df['pricenew'] = df.apply(lambda x: conversion(x['price'], x['currency']), axis=1)
df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon']]
df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude']
df2.to_csv(dir + '\CreatedDatasets\\pe_properties_new.csv', index=False)

df = pd.read_csv(dir + "\OriginalDatasets\\uy_properties.csv")
df['pricenew'] = df.apply(lambda x: conversion(x['price'], x['currency']), axis=1)
df2 = df[['pricenew','surface_covered','bathrooms','bedrooms','lat','lon']]
df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude']
df2.to_csv(dir + '\CreatedDatasets\\uy_properties_new.csv', index=False)

df = pd.read_csv(dir + "\OriginalDatasets\\Melbourne_housing_FULL.csv")
df['Price'] = df['Price'].multiply(0.636)
df2 = df[['Price','BuildingArea','Bathroom','Bedroom2','Lattitude','Longtitude']]
df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude']
df2.to_csv(dir + '\CreatedDatasets\Melbourne_housing_FULL_new.csv', index=False)

df = pd.read_csv(dir + "\OriginalDatasets\\kc_house_data.csv")
df['Price'] = df['price'].multiply(0.874)
df['sqft_living'] = df['sqft_living'].multiply(0.09290304)
df2 = df[['Price','sqft_living','bathrooms','bedrooms','lat','long']]
df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude']
df2.to_csv(dir + '\CreatedDatasets\kc_house_data_new.csv', index=False)

df = pd.read_csv(dir + "\OriginalDatasets\\Analyze_Boston.csv")
df['bldg_price'] = df['bldg_price'].multiply(0.874)
df['square_foot'] = df['square_foot'].multiply(0.09290304)
df2 = df[['bldg_price','square_foot','full_bth','bedrooms','latitude','longitude']]
df2.columns = ['Price', 'Area', 'Baths', 'Beds', 'Latitude', 'Longitude']
df2.to_csv(dir + '\CreatedDatasets\Analyze_Boston_new.csv', index=False)

