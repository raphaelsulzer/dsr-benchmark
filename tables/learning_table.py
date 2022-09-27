import os, sys
import seaborn as sns
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from methods.methods import learning_methods as methods

colors = sns.color_palette("Set2", len(methods))

spath = "/mnt/raphael/ShapeNet_out/benchmark"
mpath = "/mnt/raphael/ModelNet10_out/benchmark"

experiments = ["shapenet3000","shapenet10000","modelnet","reconbench","modelnet_shapenet"]

tables=["iou","chamfer","normal","components","boundary_edges","non-manifold_edges"]
tables={k:{} for k in tables}
cites=[]

# make tables sorted by evaluation metric
for i,m in enumerate(methods):
    cites.append(m["cite"])
    for j,e in enumerate(experiments):



        if(e == "modelnet_shapenet"):
            file = os.path.join(mpath, m["path"].format("shapenet"), "results.csv")
            if (not os.path.exists(file)):
                file = os.path.join(spath, m["path"].format("shapenet3000"), "results.csv")
        else:
            file = os.path.join(spath, m["path"].format(e), "results.csv")

        df = pd.read_csv(file)
        df = df.iloc[-1]
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
    df['Mean'] = df.mean(numeric_only=True, axis=1)
    df.columns = ['E1', 'E2', 'E3', 'E4', 'E5', "Mean"]
    tables[k] = df


fr=pd.concat([tables["iou"],tables["normal"]],axis=1)*100
sr=pd.concat([tables["chamfer"],tables["components"]],axis=1)
tr=pd.concat([tables["boundary_edges"],tables["non-manifold_edges"]],axis=1)

fr.insert(0,"citations",cites)
sr.insert(0,"citations",cites)
tr.insert(0,"citations",cites)

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

data = tr.min(0)
def f_tex(x):
    if isinstance(x,str):
        return x
    if x in data.values:
        return '\\textbf{' +f'{x:0.3g}'+ '}'
    else:
        return f'{x:0.3g}'
print(tr.to_latex(bold_rows=True,  escape=False, formatters = [f_tex]*len(tr.columns)))

a=4



