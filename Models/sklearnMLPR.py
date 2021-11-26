import pandas as pd
from pathlib import Path
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import sys

def time():
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

def normalize(df):
    print("Normalization used is StandardScaler")
    # https://stackoverflow.com/questions/60998512/how-to-scale-all-columns-except-last-column
    scalar = StandardScaler()
    standardized_features = pd.DataFrame(scalar.fit_transform(df[features].copy()), columns=features)
    old_shape = df.shape
    df.drop(features, axis=1, inplace=True)
    df = pd.concat([df, standardized_features], axis=1)
    assert old_shape == df.shape, "something went wrong!"

    return df

stdoutOrigin=sys.stdout
sys.stdout = open("logMLPRtest.txt", "w")

print("start: ")
time()

cur_dir = os.getcwd()
parent_dir = Path(cur_dir).parent.absolute()
datasets_dir = str(parent_dir) + r'\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\'
datasets_list = os.listdir(str(parent_dir) + r'\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\')

features = ["Area", "Baths", "Beds", "Latitude", "Longitude", "Month", "Year"]
labels = ["Price"]

#paramsNN = {
#    "activation": ["identity", "logistic", "relu"],
#    "solver": ["adam", "sgd"],
#    "learning_rate": ["adaptive", "invscaling"],
#    "learning_rate_init": [0.001, 0.01, 0.1, 1],
#    "max_iter": [500, 1500, 3000, 5000, 10000, 15000]
#}
paramsNN = {
    "activation": ["identity", "logistic", "relu"],
    "solver": ["adam", "sgd"],
    "learning_rate": ["adaptive", "invscaling"],
    "learning_rate_init": [0.001, 0.01, 0.1, 1],
    "max_iter": [5, 10, 20, 50]
}


print("MLPR SKLEARN LOGS WITH FOLLOWING PARAMS:\n" + str(paramsNN))

gsNN = GridSearchCV(estimator=MLPRegressor(), param_grid=paramsNN, cv=5, scoring="neg_mean_squared_error")

i = "Zameen Property Data Pakistan_new_noOutliers.csv"

df = pd.read_csv(datasets_dir + i)

print("------------------")
print(df)
df = normalize(df)
print("------------------")
print(df)
print("------------------")
X = df[features]
Y = df[labels]

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=1)

print("Train/test division is 70/30")

print("Starting search for best params: ")
time()
gsNN.fit(x_train, y_train.values.ravel())
print("Best params for the dataset :" + i + " were the following:")
time()
print(gsNN.best_params_)
print("Applying prediction with the params for all datasets...")

for i in datasets_list:
    print("Dataset \"" + i)
    df = pd.read_csv(datasets_dir + i)

    df = normalize(df)
    X = df[features]
    Y = df[labels]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=1)

    model = MLPRegressor(**gsNN.best_params_)
    model.fit(x_train, y_train.values.ravel())

    predictions = model.predict(x_test)
    print("For the dataset \"" + i + "\" the mean squared error is of " + str(mean_squared_error(y_test,predictions)))
    time()

sys.stdout.close()
sys.stdout=stdoutOrigin