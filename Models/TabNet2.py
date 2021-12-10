import pandas as pd
from pathlib import Path
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import sys

def time():
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

def normalize(df):
    print("Normalization used is StandardScaler, using kfold=5")
    # https://stackoverflow.com/questions/60998512/how-to-scale-all-columns-except-last-column
    scalar = StandardScaler()
    standardized_features = pd.DataFrame(scalar.fit_transform(df[features].copy()), columns=features)
    old_shape = df.shape
    df.drop(features, axis=1, inplace=True)
    df = pd.concat([df, standardized_features], axis=1)
    assert old_shape == df.shape, "something went wrong!"

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean())

    return df


cur_dir = os.getcwd()
parent_dir = Path(cur_dir).parent.absolute()
datasets_dir = str(parent_dir) + r'\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\'
datasets_list = os.listdir(str(parent_dir) + r'\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\')

stdoutOrigin=sys.stdout
sys.stdout = open(cur_dir + r'\TabNet\log12TabNet.txt', "w")

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


    X, X_test, y, y_test = train_test_split(X, Y, test_size=0.2)

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        clf = TabNetRegressor()
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric=['rmse', 'mse', 'mae'],
            max_epochs=5000,
            patience=500,
            num_workers=0,
            drop_last=False
        )

    print("TABNET LOGS:\n")
    print("Starting training: ")
    time()

    print("Training ended: ")
    time()
    print("Predictions starting:")
    time()

    predictions = clf.predict(X_test)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!True vs Preds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for idx,i in enumerate(y_test):
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
    plt.savefig(cur_dir + r'\TabNet\\' + r'Predictions vs Actual2.png')

    print("Predictions ended: ")
    time()

    print("Feature importance:")
    print(list(zip(features, clf.feature_importances_)))
    print(f"BEST VALID SCORE : {clf.best_cost}")
    print("Mean squared error:" + str(mean_squared_error(y_pred=predictions, y_true=y_test)))
    print("Mean absolute error:" + str(mean_absolute_error(y_pred=predictions, y_true=y_test)))
    print("MAPE:" + str(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test)))
    print("------------------------------------------------------------------")

sys.stdout.close()
sys.stdout=stdoutOrigin

# save tabnet model
saving_path_name = "./tabnet_model_test_2"
saved_filepath = clf.save_model(saving_path_name)