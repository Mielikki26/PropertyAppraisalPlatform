# Created Dataset:

## Features:

- Price - The price of the property in €;
- Area - The area of the property in m^2;
- Baths - The number of bathrooms in the property;
- Beds - The number of bedrooms in the property;
- Latitude - Latitude of the property;
- Longitude - Longitude of the property.

# Melbourne Housing Market dataset : 

Melbourne_housing_FULL.csv

## Features used and conversions made:

- Price = Price*0,636;

- Area = BuildingArea;

- Baths = Bathroom;

- Beds = Bedroom2;

- Latitude = Lattitude;

- Longitude = Longtitude;

# South American Countries datasets:

ar_properties.csv

co_properties.csv

ec_properties.csv

pe_properties.csv

uy_properties.csv

## Features used and conversions made:

- Price:

  if currency is "USD":	Price = price * 0.874

  if currency is "ARS":	Price = price * 0.00872

  if currency is "UYU":	Price = price * 0.02

  if currency is "PEN":	Price = price * 0.217

  if currency is "COP":	Price = price * 0.000225

- Area = surface_covered

- Baths = bathrooms

- Beds = bedrooms

- Latitude = lat

- Longitude = lon

# King County dataset:

kc_house_data.csv

## Features used and conversions made:

- Price = price * 0,874

- Area = sqft_living * 0.09290304
- Baths = bathrooms
- Beds = bedrooms
- Latitude = lat
- Longitude = long

# Analyze Boston dataset:

Analyze_Boston.csv

## Features used and conversions made:

- Price = bldg_price * 0,874
- Area = square_foot * 0.09290304
- Baths = full_bth
- Beds = bedrooms
- Latitude = latitude
- Longitude = longitude