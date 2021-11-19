import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

cur_dir = os.getcwd()
parent_dir = Path(cur_dir).parent.absolute()
datasets_dir = str(parent_dir) + '\Research\Datasets\CreatedDatasets\\'
datasets_list = os.listdir(str(parent_dir) + '\Research\Datasets\CreatedDatasets\\')

features = ["Area", "Baths", "Beds", "Latitude", "Longitude", "Month", "Year"]
labels = ["Price"]

params = {
    "n_estimators": [5, 10, 20, 50, 100],
    "max_features": ["auto", "sqrt", "log2", None],
    "max_depth": list(range(1, 10))
}
gs = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params, cv=5, scoring="neg_mean_squared_error")

i = "Zameen Property Data Pakistan_newclean.csv"

df = pd.read_csv(datasets_dir + i)
print("Dataset being used: " + i)

#https://stackoverflow.com/questions/60998512/how-to-scale-all-columns-except-last-column
scalar = StandardScaler()
standardized_features = pd.DataFrame(scalar.fit_transform(df[features].copy()), columns=features)
old_shape = df.shape
df.drop(features, axis=1, inplace=True)
df = pd.concat([df, standardized_features], axis=1)
assert old_shape == df.shape, "something went wrong!"

X = df[features]
Y = df[labels]

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=1)

gs.fit(x_train, y_train.values.ravel())
print(gs.best_params_)

model = RandomForestRegressor(**gs.best_params_)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(mean_squared_error(y_test,predictions))
print(list(model.feature_importances_))