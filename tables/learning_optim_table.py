import os, sys
import seaborn as sns
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from methods.methods import optim_methods, learning_methods

# from collections import OrderedDict

methods = learning_methods
methods = optim_methods



# experiments = ["0","1","2","3","4"]
# experiments = ["mvs4"]

tables=["iou","chamfer","normal","components","boundary_edges","non-manifold_edges"]
tables={k:{} for k in tables}
cites=[]

# make tables sorted by evaluation metric
for i,m in enumerate(methods):
    cites.append(m["cite"])

    if m in learning_methods:
        e = "reconbench"
        path = "/mnt/raphael/ShapeNet_out/benchmark"
        file = os.path.join(path, m["path"].format(e), "results.csv")
        df = pd.read_csv(file)
        print(m,e)
        df = df.iloc[-1]
    else:
        e = "mvs4"
        path = "/mnt/raphael/reconbench_out"
        file = os.path.join(path, m["path"], "results.csv")
        df = pd.read_csv(file)
        print(m,e)
        df = df.iloc[0]
    for k in tables:
        # if not tables[k]:
        #     tables[k][e] = {}
        # if not tables[k]:
        #     tables[k][e]={}
        if not e in tables[k]:
            tables[k][e]={}
        tables[k][e][m["name"]]=df[k]

for k in tables:

    df = pd.DataFrame(tables[k])
    df.columns = ['reconbench']
    tables[k] = df

fr=pd.concat([tables["iou"],tables["normal"]],axis=1)*100
sr=pd.concat([tables["chamfer"],tables["components"],tables["boundary_edges"],tables["non-manifold_edges"]],axis=1)

# fr.insert(0,"citations",cites)
# sr.insert(0,"citations",cites)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = fr.max(0)
def f_tex(x): # highlight min/max row, from here: https://github.com/pandas-dev/pandas/issues/38328#issuecomment-824539429
    if isinstance(x,str):
        return x
    if x in data.values:
        return '\\textbf{' +f'{x:0.3g}'+ '}'
    else:
        return f'{x:0.3g}'
print(fr.to_latex(bold_rows=True,  escape=False, formatters = [f_tex]*len(fr.columns)))

data = sr.min(0)
def f_tex(x):
    if isinstance(x,str):
        return x
    if x in data.values:
        return '\\textbf{' +f'{x:0.3g}'+ '}'
    else:
        return f'{x:0.3g}'
print(sr.to_latex(bold_rows=True,  escape=False, formatters = [f_tex]*len(sr.columns)))

# data = tr.min(0)
# def f_tex(x):
#     if isinstance(x,str):
#         return x
#     if x in data.values:
#         return '\\textbf{' +f'{x:0.3g}'+ '}'
#     else:
#         return f'{x:0.3g}'
# print(tr.to_latex(bold_rows=True,  escape=False, formatters = [f_tex]*len(tr.columns)))

a=4



