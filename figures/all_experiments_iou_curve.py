import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
import pandas as pd
import numpy as np

from methods.methods import learning_methods as methods
# methods = ["conv_onet","dgnn","labatut","lig","poco","poisson","sap"]

# methods = []
#
# d=dict()
# d["name"] = "ConvONet2D"
# d["path"] = "conv_onet/2d/{}/meshes"
# d["modelnet_shapenet"] = 0.683
# methods.append(d)
#
# d=dict()
# d["name"] = "ConvONet3D"
# d["path"] = "conv_onet/3d/{}/meshes"
# d["modelnet_shapenet"] = 0.510
# methods.append(d)
#
# d=dict()
# d["name"] = "SAP"
# d["path"] = "sap/tr/{}/meshes"
# d["modelnet_shapenet"] = 0.549
# methods.append(d)
#
# d=dict()
# d["name"] = "LIG"
# d["path"] = "lig/{}"
# d["modelnet_shapenet"] = 0.616
# methods.append(d)
#
# d=dict()
# d["name"] = "DGNN"
# d["path"] = "dgnn/tr/{}"
# d["modelnet_shapenet"] = 0.845
# methods.append(d)
#
# d=dict()
# d["name"] = "POCO"
# d["path"] = "poco/tr/{}/meshes"
# d["modelnet_shapenet"] = 0.409
# methods.append(d)
#
# d=dict()
# d["name"] = "SPSR"
# d["path"] = "poisson/{}"
# d["modelnet_shapenet"] = 0.741
# methods.append(d)
#
#
# d=dict()
# d["name"] = "Labatut $\it{et~al.}$"
# d["path"] = "labatut/{}"
# d["modelnet_shapenet"] = 0.803
# methods.append(d)


colors = sns.color_palette("Set2", len(methods))

spath = "/mnt/raphael/ShapeNet_out/benchmark"
mpath = "/mnt/raphael/ModelNet10_out/benchmark"

# experiments = ["shapenet3000","shapenet10000","modelnet","reconbench","modelnet_shapenet"]
experiments = ["shapenet3000","shapenet10000","modelnet","reconbench","shapenet"]

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.figure(figsize=(12,7))
names = []

for i,m in enumerate(methods):



    names.append(m["name"])
    ious = []
    for j,e in enumerate(experiments):

        path = spath
        if e == "shapenet":
            path = mpath

        if e in m.keys():
            ious.append(m[e])
            continue
        rfile = os.path.join(path,m["path"].format(e),"results.csv")

        df = pd.read_csv(rfile)
        ious.append(df.iloc[-1]["iou"])

    plt.plot(np.arange(len(experiments))+1,ious,color=colors[i],marker='s',markersize=11)


plt.ylabel("Volumetric IoU")
plt.xlabel("Experiment")
plt.legend(names,loc=1, bbox_to_anchor=(0.9,0.5))
plt.xticks([1,2,3,4,5])
plt.savefig("/mnt/raphael/ShapeNet_out/benchmark/iou_all_figure.png")

plt.show()
a=4



