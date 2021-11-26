import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import sys

def time():
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

def normalize_and_split(df):
    print("Normalization used is StandardScaler, train/test division is 70/30")
    # https://stackoverflow.com/questions/60998512/how-to-scale-all-columns-except-last-column
    scalar = StandardScaler()
    standardized_features = pd.DataFrame(scalar.fit_transform(df[features].copy()), columns=features)
    old_shape = df.shape
    df.drop(features, axis=1, inplace=True)
    df = pd.concat([df, standardized_features], axis=1)
    assert old_shape == df.shape, "something went wrong!"

    X = df[features]
    Y = df[labels]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test

print("start: ")
time()

stdoutOrigin=sys.stdout
sys.stdout = open("logRF.txt", "w")

cur_dir = os.getcwd()
parent_dir = Path(cur_dir).parent.absolute()
datasets_dir = str(parent_dir) + '\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\'
datasets_list = os.listdir(str(parent_dir) + '\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\')

features = ["Area", "Baths", "Beds", "Latitude", "Longitude", "Month", "Year"]
labels = ["Price"]

params = {
    "n_estimators": [10,20,50,100,150,300,500],
    "max_features": ["auto", "sqrt", "log2", None],
    "max_depth": list(range(1, 20))
}
gs = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params, cv=5, scoring="neg_mean_squared_error")

print("RANDOM FOREST LOGS WITH FOLLOWING PARAMS:\n" + str(params))

i = "Zameen Property Data Pakistan_new_noOutliers.csv"

df = pd.read_csv(datasets_dir + i)

x_train, x_test, y_train, y_test = normalize_and_split(df)

print("Starting search for best params: ")
time()
gs.fit(x_train, y_train.values.ravel())
print("Best params for the dataset :" + i + " were the following:")
time()
print(gs.best_params_)
print("Applying prediction with the params for all datasets...")

for i in datasets_list:
    print("Dataset \"" + i)
    df = pd.read_csv(datasets_dir + i)
    x_train, x_test, y_train, y_test = normalize_and_split(df)

    model = RandomForestRegressor(**gs.best_params_)
    model.fit(x_train, y_train.values.ravel())

    predictions = model.predict(x_test)
    print("For the dataset \"" + i + "\" the mean squared error is of " + mean_squared_error(y_test, predictions))
    print("List of feature importance:\n" + str(list(model.feature_importances_)))
    time()

sys.stdout.close()
sys.stdout=stdoutOrigin