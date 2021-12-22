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
from sklearn.metrics import r2_score
from statistics import mean
import json

fd = "\TabNet\Logs1\\" #files_directory
m_mapes = []
m_r2s = []
m_maes = []
features = ["Area", "Baths", "Beds", "Latitude", "Longitude", "Month", "Year"]
labels = ["Price"]
all = labels+features
cur_dir = os.getcwd()
parent_dir = Path(cur_dir).parent.absolute()
datasets_dir = str(parent_dir) + r'\Research\Datasets\CreatedDatasets\Datasets_in_use\\'
datasets_list = os.listdir(str(parent_dir) + r'\Research\Datasets\CreatedDatasets\Datasets_in_use\\')

if not os.path.exists(cur_dir + fd):
    os.mkdir(cur_dir + fd)

stdoutOrigin=sys.stdout
sys.stdout = open(cur_dir + fd + r'logTabNet.txt', "w")
print("TABNET LOGS:\n")

def time():
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    return current_time


def normalize(df):
    print("Normalization used is StandardScaler, using kfold=2")
    # https://stackoverflow.com/questions/60998512/how-to-scale-all-columns-except-last-column
    scalar = StandardScaler()
    # uncomment for price normalization too and comment the one after it
    #standardized_features = pd.DataFrame(scalar.fit_transform(df[all].copy()), columns=all)
    standardized_features = pd.DataFrame(scalar.fit_transform(df[features].copy()), columns=features)
    old_shape = df.shape
    df.drop(features, axis=1, inplace=True)
    df = pd.concat([df, standardized_features], axis=1)
    assert old_shape == df.shape, "something went wrong!"

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean())

    return df

print("start: "+ str(time()))

j = 0
for filename in datasets_list:
    mapes = []
    r2s = []
    maes = []
    print("------------------------------------------------------------------")
    print("Dataset: " + filename)
    df = pd.read_csv(datasets_dir + filename, index_col=None, header=0)

    df = normalize(df)
    df = df.reset_index()

    print("Dataset total size after normalization: " + str(df.shape))
    print("Splitting 80/20")

    X = df[features]
    Y = df[labels]

    X = X.to_numpy()
    Y = Y.to_numpy().reshape(-1, 1)


    X, X_test, y, y_test = train_test_split(X, Y, test_size=0.2)

    for i in range(10):
        j += 1
        print("j = " + str(j))

        kf = KFold(n_splits=2, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            clf = TabNetRegressor()
            clf.fit(
                X_train=X_train, y_train=y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric=['rmse', 'mse', 'mae'],
                max_epochs=15000,
                patience = 150,
                num_workers=0,
                drop_last=False
            )

        print("Starting training: " + str(time()))
        print("Training ended: " + str(time()))
        print("Predictions starting:" + str(time()))


        # plot losses
        #try:
        #    plt.plot(clf.history['loss'])
        #    plt.savefig(cur_dir + fd + r'Loss_' + filename + str(j) + '.png')
        #except:
        #    pass

        predictions = clf.predict(X_test)

        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!True vs Preds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #for idx,i in enumerate(y_test):
        #    print(str(y_test[idx]) + " --- " + str(predictions[idx]))
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        #plt.figure(figsize=(10, 10))
        #plt.scatter(y_test, predictions, c='crimson')
        #plt.yscale('log')
        #plt.xscale('log')
        #p1 = max(max(predictions), max(y_test))
        #p2 = min(min(predictions), min(y_test))
        #plt.plot([p1, p2], [p1, p2], 'b-')
        #plt.xlabel('True Values', fontsize=15)
        #plt.ylabel('Predictions', fontsize=15)
        #plt.axis('equal')
        #plt.savefig(cur_dir + fd + r'Predictions vs Actual_' + filename + str(j) + '.png')

        print("Predictions ended: " + str(time()))

        print("Feature importance:")
        print(list(zip(features, clf.feature_importances_)))
        print(f"BEST VALID SCORE : {clf.best_cost}")
        print("Mean squared error:" + str(mean_squared_error(y_pred=predictions, y_true=y_test)))
        print("Mean absolute error:" + str(mean_absolute_error(y_pred=predictions, y_true=y_test)))
        print("MAPE:" + str(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test)))
        print("R2 score:" + str(r2_score(y_pred=predictions, y_true=y_test)))
        print("------------------------------------------------------------------")

        mapes.append(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test))
        r2s.append(r2_score(y_pred=predictions, y_true=y_test))
        maes.append(mean_absolute_error(y_pred=predictions, y_true=y_test))

        # save tabnet model
        #saving_path_name = cur_dir + fd + "tabnet_model_test_" + filename + str(j)
        #saved_filepath = clf.save_model(saving_path_name)
        #plt.close('all')

    m_mapes.append((mean(mapes), np.std(mapes)))
    m_r2s.append((mean(r2s), np.std(r2s)))
    m_maes.append((mean(maes), np.std(maes)))

tabnet_data = {'model': "TabNet", 'datasets': datasets_list, 'm_mapes': m_mapes, 'm_r2s': m_r2s, 'm_maes': m_maes}

if not os.path.exists("Results.json"):
    open("Results.json", 'w').close()

with open('Results.json', 'r+') as f:
    try:
        json_data = json.load(f)
    except:
        json_data = []
    json_data.append(tabnet_data)
    f.seek(0)
    f.truncate()
    json.dump(json_data, f, indent=4, separators=(',',': '))

sys.stdout.close()
sys.stdout = stdoutOrigin