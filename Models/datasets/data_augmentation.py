import os
import pandas as pd
from geopy.geocoders import Nominatim

cur_dir = os.getcwd()
datasets_list = os.listdir(cur_dir)
geolocator = Nominatim(user_agent="MauroGaudêncio")

for dataset in datasets_list:  # for each dataset
    if '.csv' not in dataset:
        continue

    df = pd.read_csv(cur_dir + '\\' + dataset)

    if df.shape[0] == 0:
        continue

    for i in range(df.shape[0]):
        location = geolocator.reverse(str(df["Latitude"][i]) + "," + str(df["Longitude"][i]))
        address = location.raw['address']
        print(location.raw['address'])
        #suburb = address.get('suburb', '')
        #country = address.get('country', '')
        #print(country + " - " + suburb)