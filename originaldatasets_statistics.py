import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

publicos = []  #store features of public datasets

nrpublicos = 0 #nr of public datasets
nrprivados = 0 #nr of private datasets

df = pd.read_excel (r'Research\DatasetsAnalyzed.xlsx')
flag = 0  #used to know if dataset is public of not | public -> 1 | private -> 0
for i in range(1,df.shape[0],2):
    for j in df.iloc[i]:
        if type(j) != str:  #if cell is empty -> continue
            continue
        elif j == "PUBLIC": #flag for public
            nrpublicos += 1
            flag = 1
            continue
        elif j == "PRIVATE": #flag for private
            flag = 0
            nrprivados += 1
            continue
        elif flag == 1: #add feature to list
            publicos.append(j)

fig, ax = plt.subplots(figsize=(15,10)) #define plot size

countpublicos = Counter(publicos) #count ocorrences of each feature
pub = Counter({k: i for k, i in countpublicos.items() if i >= 5}) #store those who occured more than x times
pub = dict(sorted(pub.items(), key=lambda item: -item[1])) #sorting

#show plot
ax.bar(pub.keys(), pub.values())
ax.set_title(str(nrpublicos) + " datasets publicos")
plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize=10
    )
plt.show()
#print datasets that contain all wanted features
wanted_features = [i[0] for i in pub.items() if i[1] >= 20]
wanted_features.remove('Type')
wanted_features.remove('Rooms') #7
wanted_features.remove('YearBuilt') #2
print(wanted_features)

wanted_datasets = []
flag = 0
for i in range(nrprivados*2,df.shape[0]):
    temp_list = []
    if flag == 0:
        name = df.iloc[i][0]
        flag = 1
        continue
    for j in df.iloc[i]:
        if type(j) != str:
            continue
        else:
            temp_list.append(j)
    if set(wanted_features).issubset(set(temp_list)):
        wanted_datasets.append(name)
    flag = 0
print(wanted_datasets)
print(len(wanted_datasets))