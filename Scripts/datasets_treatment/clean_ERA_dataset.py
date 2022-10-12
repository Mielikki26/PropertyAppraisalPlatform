import pandas as pd
import numpy as np
import os
pd.set_option('mode.chained_assignment', None)
cur_dir = os.getcwd()
save_folder = str(cur_dir) + r'\CreatedDatasets\ERA_dataset'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def ERA_dataset():
    df = pd.read_excel("DataBase_Era.xlsx", engine='openpyxl')
    df[['Latitude', 'Longitude']] = df['Latitude and Longitude'].str.split(',', expand=True)
    df2 = df[['Price', 'Number of Bedrooms', 'WC', 'Latitude', 'Longitude','Agency', 'Property Type',
              'Transaction Type', 'Status', 'Zone', 'Parish', 'County', 'Region',
              'Number of Rooms', 'Typology', 'Floor area sq meter', 'Land area sq meter', 'Energy certification',
              'vista', 'renovado', 'renovação', 'parqueamento', 'garagem', 'obras', 'remodelado',
              'remodelação', 'com elevador', 'box', 'arrenda', 'arrendamento', 'inquilin',
              'financiamento', 'logradouro', 'R/C']]

    yes_no_features = ['vista', 'renovado', 'renovação', 'parqueamento', 'garagem', 'obras', 'remodelado',
              'remodelação', 'com elevador', 'box', 'arrenda', 'arrendamento', 'inquilin',
              'financiamento', 'logradouro', 'R/C']
    for i in yes_no_features:
        df2[i] = np.where(df2[i] != 'Yes', 0, df2[i])
        df2[i] = np.where(df2[i] == 'Yes', 1, df2[i])

    values = ['F', 'E', 'D', 'C', 'B-', 'B', 'A', 'A+']
    for i in values:
        df2['Energy certification'] = np.where(df2['Energy certification'] == i, values.index(i)+1,
                                               df2['Energy certification'])
    df2['Energy certification'].fillna(0, inplace=True)

    df2['Typology'].fillna('0', inplace=True)
    df2['Typology'] = df2['Typology'].map(lambda x: x.lstrip('T'))
    df2['Typology'] = df2['Typology'].apply(pd.to_numeric)

    continuos_features = ['Floor area sq meter','Land area sq meter', 'WC', 'Number of Bedrooms', 'Number of Rooms']
    for i in continuos_features:
        df2[i] = df2[i][pd.to_numeric(df2[i], errors='coerce').notnull()]
        df2[i].fillna(0, inplace=True)

    df2['Latitude'] = df2['Latitude'].apply(pd.to_numeric)
    df2['Longitude'] = df2['Longitude'].apply(pd.to_numeric)
    df2 = df2[df2['Latitude'] > 0]
    df2 = df2[df2['Longitude'] < 0]

    categorical_features = ['Agency', 'Property Type', 'Transaction Type', 'Status', 'Zone',
                            'Parish', 'County', 'Region']
    for i in categorical_features:
        df2[i].fillna('N/D', inplace=True)

    df2['Price'] = df2['Price'][pd.to_numeric(df2['Price'], errors='coerce').notnull()]

    df2 = df2.dropna()
    #df2 = df2.drop_duplicates()
    df2.to_csv(save_folder + r'\DataBase_Era.csv', index=False)


ERA_dataset()