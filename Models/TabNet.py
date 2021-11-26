import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
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

    X = X.to_numpy()
    Y = Y.to_numpy().reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test

stdoutOrigin=sys.stdout
sys.stdout = open("logTabNet.txt", "w")

print("start: ")
time()

cur_dir = os.getcwd()
parent_dir = Path(cur_dir).parent.absolute()
datasets_dir = str(parent_dir) + r'\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\'
datasets_list = os.listdir(str(parent_dir) + r'\Research\Datasets\CreatedDatasets\NewDatasets_noOutliers\\')

features = ["Area", "Baths", "Beds", "Latitude", "Longitude", "Month", "Year"]
labels = ["Price"]

i = "Zameen Property Data Pakistan_new_noOutliers.csv"
#for i in datasets_list:
df = pd.read_csv(datasets_dir + i)

X_train, X_test, y_train, y_test = normalize_and_split(df)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

print("TABNET LOGS:\n")
print("Starting training: ")
time()

clf = TabNetRegressor()

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric=['rmse', 'rmsle', 'mae', 'mse'],
    max_epochs=1000,
    patience=50,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

print("Training ended: ")
time()
print("Predictions starting:")
time()

preds = clf.predict(X_test)

print("Predictions ended: ")
time()

y_true = y_test

test_score = mean_squared_error(y_pred=preds, y_true=y_true)

print(f"BEST VALID SCORE FOR {i} : {clf.best_cost}")
print(f"FINAL TEST SCORE FOR {i} : {test_score}")
print(list(clf.feature_importances_))
print("Mean squared error:" + str(mean_squared_error(y_pred=preds, y_true=y_true)))

sys.stdout.close()
sys.stdout=stdoutOrigin

# save tabnet model
saving_path_name = "./tabnet_model_test_1"
saved_filepath = clf.save_model(saving_path_name)
# define new model with basic parameters and load state dict weights
loaded_clf = TabNetRegressor()
loaded_clf.load_model(saved_filepath)
loaded_preds = loaded_clf.predict(X_test)
loaded_test_mse = mean_squared_error(loaded_preds, y_test)

print(f"FINAL TEST SCORE FOR {i} : {loaded_test_mse}")
assert(test_score == loaded_test_mse)

