import pandas as pd
from pathlib import Path
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
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

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

cur_dir = os.getcwd()
parent_dir = Path(cur_dir).parent.absolute()
datasets_dir = str(parent_dir) + r'\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\'
datasets_list = os.listdir(str(parent_dir) + r'\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\')


stdoutOrigin=sys.stdout
sys.stdout = open(cur_dir + r'\RF\logRF.txt', "w")

print("start: ")
time()

features = ["Area", "Baths", "Beds", "Latitude", "Longitude", "Month", "Year"]
labels = ["Price"]

for filename in datasets_list:
    print("------------------------------------------------------------------")
    print("Dataset: " + filename)
    df = pd.read_csv(datasets_dir + filename, index_col=None, header=0)

    df = normalize(df)
    df = df.reset_index()

    print("Dataset total size after normalization: " + str(df.shape))

    X = df[features]
    Y = df[labels]

    X = X.to_numpy()
    Y = Y.to_numpy().reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=1)
    print("Train/test division is 70/30")

    params = {
        "n_estimators": [1,5],
        "max_features": ["auto", "sqrt", "log2", None],
        "max_depth": list(range(5, 15))
    }
    gs = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params, cv=5, scoring="neg_mean_squared_error")

    print("RANDOM FOREST LOGS WITH FOLLOWING PARAMS:\n" + str(params))

    print("Starting search for best params: ")
    time()
    gs.fit(x_train, y_train.ravel())
    print("Best params were the following:")
    time()
    print(gs.best_params_)
    print("Applying prediction with the params...")


    clf = RandomForestRegressor(**gs.best_params_)
    clf.fit(x_train, y_train.ravel())

    predictions = clf.predict(x_test)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!True vs Preds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for idx, i in enumerate(y_test):
        print(str(y_test[idx]) + " --- " + str(predictions[idx]))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, predictions, c='crimson')
    plt.yscale('log')
    plt.xscale('log')
    p1 = max(max(predictions), max(y_test))
    p2 = min(min(predictions), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.savefig(cur_dir + r'\RF\\' + r'Predictions vs Actual.png')


    print("Feature importance:")
    print(list(zip(features, clf.feature_importances_)))
    print("Mean squared error is of " + str(mean_squared_error(y_test, predictions)))
    print("Mean absolute error:" + str(mean_absolute_error(y_pred=predictions, y_true=y_test)))
    print("MAPE:" + str(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test)))
    time()
    print("------------------------------------------------------------------")

sys.stdout.close()
sys.stdout=stdoutOrigin