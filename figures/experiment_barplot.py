import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
import pandas as pd
import numpy as np

# from methods.methods import learning_methods as methods
# methods = ["conv_onet","dgnn","labatut","lig","poco","poisson","sap"]

methods = []

d=dict()
d["name"] = "CONet2D"
d["path"] = "conv_onet/2d/{}/meshes"
methods.append(d)

d=dict()
d["name"] = "CONet3D"
d["path"] = "conv_onet/3d/{}/meshes"
methods.append(d)

d=dict()
d["name"] = "SAP"
d["path"] = "sap/tr/{}/meshes"
methods.append(d)

# d=dict()
# d["name"] = "LIG"
# d["path"] = "lig/{}"
# methods.append(d)



d=dict()
d["name"] = "POCO"
d["path"] = "poco/tr/{}/meshes"
methods.append(d)

d=dict()
d["name"] = "DGNN"
d["path"] = "dgnn/tr/{}"
methods.append(d)

d=dict()
d["name"] = "SPSR"
d["path"] = "poisson/{}"
methods.append(d)


d=dict()
d["name"] = "Labatut $\it{et~al.}$"
d["path"] = "labatut/{}"
methods.append(d)


# colors = sns.color_palette("Set2", len(methods))

colors = ["#ccece6","#66c2a4","#238b45","#005824","#f03b20","#d0d1e6","#034e7b"]
# colors = ["#ccece6","#66c2a4","#238b45","#005824","#d0d1e6","#034e7b"]

spath = "/mnt/raphael/ShapeNet_out/benchmark"
mpath = "/mnt/raphael/ModelNet10_out/benchmark"

# experiments = ["shapenet3000","shapenet10000","modelnet","reconbench","modelnet_shapenet"]
experiments = ["shapenet3000","shapenet10000","modelnet","reconbench","shapenet"]
experiments = ["shapenet10000"]

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)



# metric="iou"
metrics=["iou","components"]
metric_names = ["Volumetric IoU","Number of Components"]

for k,metric in enumerate(metrics):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
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
            ious.append(df.iloc[-1][metric])

        # plt.plot(np.arange(len(experiments))+1,ious,color=colors[i],marker='s',markersize=11)
        space_between_experiments=3
        ax.bar(np.arange(1)*
               (1 + space_between_experiments)+i,ious,width=0.9,color=colors[i])
        print(m["name"], ious[0])

        if metric == 'iou':
            ax.text(np.arange(1) *
                    (1 + space_between_experiments) + i-0.4,
                    ious[0]*1.03, "{:.2f}".format(ious[0]), color='black', fontweight='bold')
        else:
            ax.text(np.arange(1) *
                    (1 + space_between_experiments) + i-0.2,
                    ious[0]+11.0, "{:.0f}".format(ious[0]), color='black', fontweight='bold')


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.ylabel(metric_names[k])

    if metric == 'iou':
        plt.gca().set_ylim(bottom=0.0)
        # ax.set_ylim([0.4, 0.9])
    else:
        plt.gca().invert_yaxis()
        plt.gca().set_ylim(top=1)
        locs,labels=plt.yticks()
        locs[0]=1
        plt.yticks(locs)

    # plt.ylabel("Chamfer distance")
    plt.xlabel("Method")
    ax.xaxis.set_label_coords(0.5, -0.28)
    plt.xticks(np.arange(len(methods)),names,rotation=45)


    # plt.legend(names,loc=1, bbox_to_anchor=(1.5,0.5))

    box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])

    ax.set_position([box.x0+0.05, box.y0+0.1,
                     box.width, box.height * 0.8])


    plt.savefig("/mnt/raphael/ShapeNet_out/benchmark/plots/experiment_{}_barplot_{}_with_dgnn.png".format(experiments[0],metric),
                transparent=True)

    plt.show(block=False)
    a=4



