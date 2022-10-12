import os
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from statistics import mean
import numpy as np
import json

def get_datasets_names(folder):
    new_dir = dir[:-1] + folder + r"\\"
    dataset_list = os.listdir(str(new_dir))
    datasets = []
    for idx, val in enumerate(dataset_list):
        if "all_data" in val:
            continue
        else:
            datasets.append(val.split("_")[0])
    datasets.append("all_data")
    return datasets

def calc_true_values(datasets):
    avg_mape = []
    avg_r2 = []
    avg_mae = []

    cols = ["Parish", "AvgParish"]

    #df_all = pd.DataFrame(columns = cols)
    for i in datasets: #for each dataset

        # we will do all_data later
        #if i == "all_data":
        #    continue

        mapes = []
        r2s = []
        maes = []

        for og_data in og_datasets_list:  # get original dataset
            if i in og_data:  # of the same name
                # we only care about suburb and average columns
                og_df = pd.read_csv(og_datasets_dir + r"\\" + og_data)
                og_df2 = og_df.drop_duplicates('Parish', keep='first')
                og_df2 = og_df2[cols]
                #df_all = df_all.append(og_df2)

        for j in sub_folders: #iterate all 5 of each
            datasets_list = os.listdir(dir[:-1] + j + r"\\")

            for k in datasets_list:
                if i[:-4] in k:
                    df = pd.read_csv(dir[:-1] + j + r"\\" + k, index_col=None, header=0)

                    #calculate suburb average for the prediction datasets
                    df["AvgParish"] = 0.0
                    for sub in og_df2["Parish"].unique():
                        avg = og_df2.loc[og_df2['Parish'] == sub, 'AvgParish'].iloc[0]
                        df.loc[df["Parish"] == sub, 'AvgParish'] = avg

                    #calculate true values
                    df["True_Price"] = (df["AvgParish"] * df["CoefParish"]) * df["Floor area sq meter"]
                    df["True_PricePredicted"] = (df["AvgParish"] * df["Predictions"]) * df["Floor area sq meter"]

                    #append values for this iteration in temp list
                    mapes.append(mean_absolute_percentage_error(y_pred=df["True_PricePredicted"].to_numpy(),
                                                                y_true=df["True_Price"].to_numpy()))
                    r2s.append(r2_score(y_pred=df["True_PricePredicted"].to_numpy(),
                                        y_true=df["True_Price"].to_numpy()))
                    maes.append(mean_absolute_error(y_pred=df["True_PricePredicted"].to_numpy(),
                                                    y_true=df["True_Price"].to_numpy()))

        #append final values in final list
        avg_mape.append((mean(mapes), np.std(mapes)))
        avg_r2.append((mean(r2s), np.std(r2s)))
        avg_mae.append((mean(maes), np.std(maes)))
    """
    mapes = []
    r2s = []
    maes = []
    for j in sub_folders:  # iterate all 5 of each
        datasets_list = os.listdir(dir[:-1] + j + r"\\")
        for k in datasets_list:
            if "all_data" in k:
                df = pd.read_csv(dir[:-1] + j + r"\\" + k, index_col=None, header=0)

                # calculate suburb average for the prediction datasets
                df["Average"] = 0.0
                for i in df_all["suburb"].unique():
                    avg = df_all.loc[df_all['suburb'] == i, 'Average'].iloc[0]
                    df.loc[df["suburb"] == i, 'Average'] = avg

                # calculate true values
                df["True_Price"] = (df["Average"] * df["Price"]) * df["Area"]
                df["True_PricePredicted"] = (df["Average"] * df["Prediction"]) * df["Area"]

                # append values for this iteration in temp list
                mapes.append(mean_absolute_percentage_error(y_pred=df["True_PricePredicted"].to_numpy(),
                                                            y_true=df["True_Price"].to_numpy()))
                r2s.append(r2_score(y_pred=df["True_PricePredicted"].to_numpy(),
                                    y_true=df["True_Price"].to_numpy()))
                maes.append(mean_absolute_error(y_pred=df["True_PricePredicted"].to_numpy(),
                                                y_true=df["True_Price"].to_numpy()))

    # append final values in final list
    avg_mape.append((mean(mapes), np.std(mapes)))
    avg_r2.append((mean(r2s), np.std(r2s)))
    avg_mae.append((mean(maes), np.std(maes)))"""

    return avg_mape, avg_r2, avg_mae


dir = os.getcwd() + r'\RF\5x\Era\Log&SS_Coeficient\Logs\Predictions\\'
json_file = "resultsRF_ERA.json"

sub_folders = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

og_datasets_dir = r"C:\Users\Mauro\PycharmProjects\Estagio\Models\Era_datasets"
og_datasets_list = os.listdir(og_datasets_dir)

datasets = ["Era_Coef_Parish.csv"]

avg_mape, avg_r2, avg_mae = calc_true_values(datasets)

#save json
mlpr_data = {'model': "MLPR", 'datasets': datasets, 'avg_mape': avg_mape, 'avg_r2': avg_r2, 'avg_mae': avg_mae}

if not os.path.exists(dir[:-1] + json_file):
    open(dir[:-1] + json_file, 'w').close()
with open(dir[:-1] + json_file, 'r+') as f:
    f.seek(0)
    f.truncate()
    json.dump(mlpr_data, f, indent=4, separators=(',',': '))
