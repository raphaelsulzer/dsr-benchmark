from tqdm import tqdm

from datasets.modelnet10 import ModelNet10
from datasets.shapenet import ShapeNet
from datasets.berger import Berger
from evaluator.eval import MeshEvaluator

import os
import pandas as pd


methods = []

# d=dict()
# d["name"] = "ConvONet-2D~\cite{Peng2020}"
# d["path"] = "conv_onet/2dAP/{}/meshes"
# methods.append(d)
#
# d=dict()
# d["name"] = "ConvONet-3D~\cite{Peng2020}"
# d["path"] = "conv_onet/3dAP/{}/meshes"
# methods.append(d)
#
# d=dict()
# d["name"] = "Shape~As~Points~\cite{Peng2021SAP}"
# d["path"] = "sap/AP/{}/meshes"
# methods.append(d)

# d=dict()
# d["name"] = "LIG~\cite{lig}"
# d["path"] = "lig/{}"
# d["modelnet_shapenet"] = 0.664
# methods.append(d)
#
# d=dict()
# d["name"] = "DGNN~\cite{dgnn}"
# d["path"] = "dgnn/tr/{}"
# methods.append(d)

# d=dict()
# d["name"] = "POCO~\cite{boulch2022poco}"
# d["path"] = "poco/aux/gen_ShapeNetViewpoints_test100_3000/meshes/raphael/ShapeNet"
# methods.append(d)
#
# d=dict()
# d["name"] = "P2S~\cite{points2surf}"
# d["path"] = "p2s/shapenet/aux_global/rec/mesh"
# methods.append(d)

d=dict()
d["name"] = "P2S~\cite{points2surf}"
d["path"] = "p2s/shapenet/conventional/rec/mesh"
methods.append(d)

# d=dict()
# d["name"] = "SPSR~\cite{screened_poisson}"
# d["path"] = "poisson/{}"
# methods.append(d)
#
#
# d=dict()
# d["name"] = "Vu~\etal~\cite{Vu2012}"
# d["path"] = "labatut/{}"
# methods.append(d)


# experiment = "modelnet"
# dataset = ModelNet10()
# split = "test"
# models = dataset.getModels(splits=[split],classes=["bathtub",  "bed",  "desk",  "dresser",  "nightstand",  "toilet"])[split]

# experiment = "modelnet"
# dataset = ModelNet10()
# split = "test"
# models = dataset.getModels(splits=[split])[split]


# experiment = "shapenet3000"
# dataset = ShapeNet()
# split = "test100"
# models = dataset.getModels(splits=[split],scan='4')[split]

# experiment = "shapenet10000"
# dataset = ShapeNet()
# split = "test100"
# models = dataset.getModels(splits=[split],scan='6')[split]

experiment = "shapenet"
dataset = ShapeNet()
split = "test10"
models = dataset.getModels(splits=[split],scan_conf='4')[split]



# experiment = "reconbench"
# dataset = Berger()
# models = dataset.getModels(scan=["4"])

results_all = []

data_path = "/mnt/raphael/ModelNet10_out"
# data_path = os.path.join(data_path,"benchmark")
# data_path = os.path.join(data_path,"dgnn","ablation")
data_path = os.path.join(data_path,"dsrv")

for m in methods:
    transform=False
    print(m["name"])
    evaluator = MeshEvaluator(n_points=100000)
    outpath = os.path.join(data_path,m["path"].format(experiment))
    # outpath = os.path.join(data_path,"baseline","modelnet")
    if(m["name"] not in ["DGNN~\cite{dgnn}","P2S~\cite{points2surf}"]):
        transform = True
    res_dict = evaluator.eval(models,outpath,transform,method=m["name"])
    results_all.append(res_dict.loc["mean"].rename(m["name"]))


eval_df = pd.DataFrame(results_all)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 3)
print(eval_df)

eval_df.drop(labels='watertight',axis=1, inplace=True)
eval_df.to_csv(os.path.join(data_path, experiment+"_all.csv"),sep='&',float_format='%.3g',line_terminator='\\\\\n')


