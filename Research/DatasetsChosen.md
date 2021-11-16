# Created Dataset:

## Features:

- Price - The price of the property in €;
- Area - The area of the property in m^2;
- Baths - The number of bathrooms in the property;
- Beds - The number of bedrooms in the property;
- Latitude - Latitude of the property;
- Longitude - Longitude of the property;
- Date - Date of the sale. (dd/mm/yy);

# Melbourne Housing Market dataset : 

Melbourne_housing_FULL.csv

## Features used and conversions made:

- Price = Price*0,636;
- Area = BuildingArea;
- Baths = Bathroom;
- Beds = Bedroom2;
- Latitude = Lattitude;
- Longitude = Longtitude;
- Date = Date;

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

- Area = surface_covered;

- Baths = bathrooms;

- Beds = bedrooms;

- Latitude = lat;

- Longitude = lon;

- date = start_date;

# King County dataset:

kc_house_data.csv

## Features used and conversions made:

- Price = price * 0,874;
- Area = sqft_living * 0.09290304;
- Baths = bathrooms;
- Beds = bedrooms;
- Latitude = lat;
- Longitude = long;
- Date = date (format change applied);

# DC dataset:

DC_Properties.csv

## Features used and conversions made:

- Price = PRICE * 0,874;
- Area = LANDAREA * 0.09290304;
- Baths = BATHRM;
- Beds = BEDRM;
- Latitude = LATITUDE;
- Longitude = LONGITUDE;
- Date = SALEDATE (format change applied);

# Zameen Property Data Pakistan dataset:

Zameen Property Data Pakistan.csv

## Features used and conversions made:

- Price = price * 0.0051;
- Area = area_sqft* 0.09290304;
- Baths = baths;
- Beds = bedrooms;
- Latitude = latitude;
- Longitude = longitude;
- Date = date_added (format change applied);

# Perth dataset:

all_perth_310121.csv

## Features used and conversions made:

- Price = PRICE* 0,874;
- Area = LAND_AREA* 0.09290304;
- Baths = BATHROOMS;
- Beds = BEDROOMS;
- Latitude = LATITUDE;
- Longitude = LONGITUDE;
- Date = DATE_SOLD(format change applied);

