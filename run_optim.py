from tqdm import tqdm

from datasets.modelnet10 import ModelNet10
from datasets.shapenet import ShapeNet
from datasets.berger import Berger
from evaluator.eval import MeshEvaluator

import os
import pandas as pd


methods = []


d=dict()
d["name"] = "IGR"
d["cite"] = "\cite{Gropp2020}"
d["path"] = "igr"
methods.append(d)

d=dict()
d["name"] = "LIG"
d["cite"] = "\cite{lig}"
d["path"] = "lig"
methods.append(d)

d=dict()
d["name"] = "P2M"
d["cite"] = "\cite{point2mesh}"
d["path"] = "p2m/poisson"
methods.append(d)

d=dict()
d["name"] = "SAP"
d["cite"] = "\cite{Peng2021SAP}"
d["path"] = "sap"
methods.append(d)

d=dict()
d["name"] = "DSE"
d["cite"] = "\cite{rakotosaona2021dse}"
d["path"] = "dse"
methods.append(d)

d=dict()
d["name"] = "SPSR"
d["cite"] = "\cite{screened_poisson}"
d["path"] = "poisson"
methods.append(d)

d=dict()
d["name"] = "Labatut~\etal"
d["cite"] = "\cite{Labatut2009a}"
d["path"] = "labatut"
methods.append(d)


experiment = "reconbench"
dataset = Berger()
models = dataset.getModels(type="berger",scan=["0","1","2","3","4"])
outpath = "/mnt/raphael/reconbench_out"

results_all = []


for m in methods:
    print(m["name"])
    evaluator = MeshEvaluator(n_points=100000)
    outpath = os.path.join("/mnt/raphael/reconbench_out",m["path"])
    res_dict = evaluator.eval(models,outpath)
    results_all.append(res_dict.loc["mean"].rename(m["name"]))


eval_df = pd.DataFrame(results_all)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 3)
print(eval_df)

eval_df.drop(labels='watertight',axis=1, inplace=True)
eval_df.to_csv(os.path.join("/mnt/raphael/ModelNet10_out/benchmark", experiment+"_all.csv"),sep='&',float_format='%.3g',line_terminator='\\\\\n')


