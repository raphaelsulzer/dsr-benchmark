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


# colors = sns.color_palette("Set2", len(methods))

colors = ["#ccece6","#66c2a4","#238b45","#005824","#f03b20","#d0d1e6","#034e7b"]

spath = "/mnt/raphael/ShapeNet_out/benchmark"
mpath = "/mnt/raphael/ModelNet10_out/benchmark"

# experiments = ["shapenet3000","shapenet10000","modelnet","reconbench","modelnet_shapenet"]
experiments = ["shapenet3000","shapenet10000","modelnet","reconbench","shapenet"]

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

fig=plt.figure(figsize=(15,8))
ax = fig.add_subplot(1,1,1)


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

    # plt.plot(np.arange(len(experiments))+1,ious,color=colors[i],marker='s',markersize=11)
    space_between_experiments=3
    ax.bar(np.arange(len(experiments))*
           (len(experiments)+space_between_experiments)+i,ious,width=0.9,color=colors[i])
    # plt.plot(np.arange(len(experiments))+1,ious,color=colors[i],marker='s',markersize=11)




plt.ylabel("Volumetric IoU")
plt.xlabel("Experiment")
plt.xticks([3,11,19,27,35],[1,2,3,4,5])

# plt.legend(names,loc=1, bbox_to_anchor=(1.5,0.5))

box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

ax.set_position([box.x0, box.y0,
                 box.width, box.height * 0.8])

# Put a legend below current axis
ax.legend(names,loc='upper center', bbox_to_anchor=(0.5, 1.3),
          fancybox=True, shadow=True, ncol=5)


plt.savefig("/mnt/raphael/ShapeNet_out/benchmark/all_experiments_barplot_presentation.png")

plt.show()
a=4



