import json

f = open('Results.json')

data = json.load(f)

def round_list_values(mapes, r2s):#, maes):
    for idx,i in enumerate(mapes):
        mapes[idx] = [j * 100 for j in mapes[idx]]
        mapes[idx] = ['%.2f' % elem for elem in mapes[idx]]
    for idx,i in enumerate(r2s):
        r2s[idx] = ['%.2f' % elem for elem in r2s[idx]]
#    for idx,i in enumerate(maes):
#        maes[idx] = ['%.0f' % elem for elem in maes[idx]]

tabnet_data = []
rf_data = []
mlpr_data = []

for i in data:
    if i['model'] == "TabNet" and tabnet_data == []:
        tabnet_data = i
    elif i['model'] == "RF" and rf_data == []:
        rf_data = i
    elif i['model'] == "MLPR" and mlpr_data == []:
        mlpr_data = i
    else:
        print("Error in json")

if tabnet_data['datasets'] != rf_data['datasets'] and tabnet_data['datasets'] != mlpr_data['datasets']:
    print("Error in json")

round_list_values(tabnet_data['m_mapes'], tabnet_data['m_r2s'])
round_list_values(rf_data['m_mapes'], rf_data['m_r2s'])
round_list_values(mlpr_data['m_mapes'], mlpr_data['m_r2s'])

def create_table(tabnet_data, rf_data, mlpr_data):
    template_start = r"\begin{table}[H]"  + "\n" +\
                     r"\begin{tabular}{ccc|cc|cc}"  + "\n" + \
                     r"\cline{2-7}" + "\n" + \
                     r"\multirow{2}{*}{} & \multicolumn{2}{c|}{TabNet} & \multicolumn{2}{c|}{RF}     & \multicolumn{2}{c}{MLPR}    \\ \cline{2-7} "  + "\n" +\
                     r"                  & MAPE & R\textasciicircum{}2 & MAPE & R\textasciicircum{}2 & MAPE & R\textasciicircum{}2 \\ \hline" + "\n"
    template_end = r"\end{tabular}"  + "\n" +\
                   r"\end{table}"
    body = ""
    for i in range(len(tabnet_data['datasets'])):
        body += tabnet_data['datasets'][i][:-4] + " & " + str(tabnet_data['m_mapes'][i][0]) + " $\pm$ " + str(tabnet_data['m_mapes'][i][1]) + " & " + str(tabnet_data['m_r2s'][i][0])\
               + " $\pm$ " + "{\\footnotesize" + str(tabnet_data['m_r2s'][i][1]) + "} & " + str(rf_data['m_mapes'][i][0]) + " $\pm$ " + str(rf_data['m_mapes'][i][1]) + " & " + str(rf_data['m_r2s'][i][0])\
               + " $\pm$ " + str(rf_data['m_r2s'][i][1]) + " & " + str(mlpr_data['m_mapes'][i][0]) + " $\pm$ " + str(mlpr_data['m_mapes'][i][1]) + " & " + str(mlpr_data['m_r2s'][i][0])\
               + " $\pm$ " + str(mlpr_data['m_r2s'][i][1]) + " \\\\ \hline \n"

    final_table = template_start + body + template_end
    return final_table

print(create_table(tabnet_data, rf_data, mlpr_data))


#with maes
#def create_table(tabnet_data, rf_data, mlpr_data):
#    template_start = r"\begin{table}[]"  + "\n" +\
#                     r"\begin{tabular}{cccccccccc}"  + "\n" + \
#                     r"\cline{2-10}" + "\n" + \
#                     r"\multirow{2}{*}{} & \multicolumn{3}{c}{TabNet}         & \multicolumn{3}{c}{RF}            & \multicolumn{3}{c}{MLPR}          \\ \cline{2-10}"  + "\n" +\
#                     r"                  & MAPE  & R\textasciicircum{}2 & MAE & MAPE & R\textasciicircum{}2 & MAE & MAPE & R\textasciicircum{}2 & MAE \\ \hline" + "\n"
#    template_end = r"\end{tabular}"  + "\n" +\
#                   r"\end{table}"
#    body = ""
#    for i in range(len(tabnet_data['datasets'])):
#        body += tabnet_data['datasets'][i][:-4] + " & " + str(tabnet_data['m_mapes'][i][0]) + " $\pm$ " + str(tabnet_data['m_mapes'][i][1]) + " & " + str(tabnet_data['m_r2s'][i][0])\
#               + " $\pm$ " + str(tabnet_data['m_r2s'][i][1]) + " & " + str(tabnet_data['m_maes'][i][0]) + " $\pm$ " + str(tabnet_data['m_maes'][i][1]) + " & " + str(rf_data['m_mapes'][i][0])\
#               + " $\pm$ " + str(rf_data['m_mapes'][i][1]) + " & " + str(rf_data['m_r2s'][i][0]) + " $\pm$ " + str(rf_data['m_r2s'][i][1]) + " & " + str(rf_data['m_maes'][i][0])\
#               + " $\pm$ " + str(rf_data['m_maes'][i][1]) + " & " + str(mlpr_data['m_mapes'][i][0]) + " $\pm$ " + str(mlpr_data['m_mapes'][i][1]) + " & " + str(mlpr_data['m_r2s'][i][0])\
#               + " $\pm$ " + str(mlpr_data['m_r2s'][i][1]) + " & " + str(mlpr_data['m_maes'][i][0]) + " $\pm$ " + str(mlpr_data['m_maes'][i][1]) + " \\\\ \hline \n"
#
#    final_table = template_start + body + template_end
#    return final_table