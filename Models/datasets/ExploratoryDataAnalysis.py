import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

features = ["Price", "Area", "Baths", "Beds", "Latitude", "Longitude", "Month", "Year"]
cur_dir = os.getcwd()

def create_boxplots(datasets_list):
    list_all = []
    for i in datasets_list:  # for each dataset
        if '.csv' not in i:
            continue

        df = pd.read_csv(cur_dir + '\\' + i)

        if df.shape[0] == 0:
            continue

        df["Dataset"] = i[:-4]
        list_all.append(df)

    df_all = pd.concat(list_all)

    if not os.path.exists(cur_dir + '\Boxplots\\'):
        os.mkdir(cur_dir + '\Boxplots\\')

    for i in features:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x="Dataset", y=i, data=df_all)
        ax.set_title("Boxplot comparing the feature distribuition of " + i)
        plt.xticks(
            rotation=10,
            horizontalalignment='right',
            fontweight='light',
            fontsize=10
        )
        plt.ticklabel_format(style='plain', axis='y')
        plt.savefig(cur_dir + '\Boxplots\\' + i + 'Distribution.png')
        plt.close()

def create_histplots(datasets_list):
    for i in datasets_list:  # for each dataset
        if '.csv' not in i:
            continue

        df = pd.read_csv(cur_dir + '\\' + i)

        if df.shape[0] == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(i + '(size = ' + str(df.shape[0]) + ' rows)')
        plt.xlabel("Baths")
        plt.hist(df["Year"])


        plt.show()

def create_scatterplots(datasets_list, feature):
    for i in datasets_list:  # for each dataset
        if '.csv' not in i:
            continue

        df = pd.read_csv(cur_dir + '\\' + i)

        if df.shape[0] == 0:
            continue

        if not os.path.exists(cur_dir + '\Scatterplots\\' + feature + '\\'):
            os.makedirs(cur_dir + '\Scatterplots\\' + feature + '\\')

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(i + '(size = ' + str(df.shape[0]) + ' rows)')
        plt.xlabel("Price")
        plt.ylabel(feature)
        plt.scatter(data=df, x="Price", y=feature, c="orange")
        plt.savefig(cur_dir + '\Scatterplots\\' + feature + '\\' + i[:-4] + '.png')
        plt.close()


datasets_list = os.listdir(cur_dir)

create_boxplots(datasets_list)
#create_histplots(datasets_list)
create_scatterplots(datasets_list, "Area")
create_scatterplots(datasets_list, "Baths")
create_scatterplots(datasets_list, "Beds")
create_scatterplots(datasets_list, "Year")
create_scatterplots(datasets_list, "Month")
