# Created Dataset:

## Features:

- Price - The price of the property in €;
- Area - The area of the property in m^2;
- Baths - The number of bathrooms in the property;
- Beds - The number of bedrooms in the property;
- Latitude - Latitude of the property;
- Longitude - Longitude of the property;
- Month - Month of the sale;
- Year - Year of the sale.

# Melbourne Housing Market dataset : 

Melbourne_housing_FULL.csv

## Features used and conversions made:

- Price = Price*0,636;
- Area = BuildingArea;
- Baths = Bathroom;
- Beds = Bedroom2;
- Latitude = Lattitude;
- Longitude = Longtitude;
- Month and Year = Date(separated into respective values);

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

- Month and Year = start_date(separated into respective values);

# King County dataset:

kc_house_data.csv

## Features used and conversions made:

- Price = price * 0,874;
- Area = sqft_living * 0.09290304;
- Baths = bathrooms;
- Beds = bedrooms;
- Latitude = lat;
- Longitude = long;
- Month and Year = date(separated into respective values);

# DC dataset:

DC_Properties.csv

## Features used and conversions made:

- Price = PRICE * 0,874;
- Area = LANDAREA * 0.09290304;
- Baths = BATHRM;
- Beds = BEDRM;
- Latitude = LATITUDE;
- Longitude = LONGITUDE;
- Month and Year = SALEDATE(separated into respective values);

# Zameen Property Data Pakistan dataset:

Zameen Property Data Pakistan.csv

## Features used and conversions made:

- Price = price * 0.0051;
- Area = area_sqft* 0.09290304;
- Baths = baths;
- Beds = bedrooms;
- Latitude = latitude;
- Longitude = longitude;
- Month and Year = date_added(separated into respective values);

# Perth dataset:

all_perth_310121.csv

## Features used and conversions made:

- Price = PRICE* 0.63;
- Area = FLOOR_AREA;
- Baths = BATHROOMS;
- Beds = BEDROOMS;
- Latitude = LATITUDE;
- Longitude = LONGITUDE;
- Month and Year = DATE_SOLD(separated into respective values);

# Data Exploration and Cleaning realized:

- The original datasets were used to create new datasets with only the features chosen as described above;
- The new datasets were explored and the following changes were made:
  - Duplicated rows were deleted;
  - Rows with empty cells were deleted;
  - Rows with nonsensical data(such as negative Prices or Areas) were deleted;
- Outliers were Removed using the following algorithms(all were implemented and after further analysis only one will be chosen):
  - IQR score.
  - Grubbs test.
  - Zscore.
  
- With these changes the dataset ec_properties.csv was completely removed because it had less than 1000 data points.
- Plots were created for each dataset representing the differences before and after outlier removal with each algorithm.

