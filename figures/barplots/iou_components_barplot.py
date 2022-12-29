import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
import pandas as pd
import numpy as np

from methods.methods import learning_methods as methods

# colors = ["#ccece6","#66c2a4","#238b45","#005824","#f03b20","#d0d1e6","#034e7b"]
colors = ["#dbddf2","#afb3e1","#555ec0","#373f94","#191d44","#fdc4b7","#fb6b4b"]

spath = "/mnt/raphael/ShapeNet_out/benchmark"
mpath = "/mnt/raphael/ModelNet10_out/benchmark"

# experiments = ["shapenet3000","shapenet10000","modelnet","reconbench","modelnet_shapenet"]
experiments = ["shapenet3000","shapenet10000","modelnet","reconbench","shapenet"]

font = {'family' : 'DejaVu Sans',
        # 'weight' : 'bold',
        'size'   : 26}

matplotlib.rc('font', **font)

fig, axes = plt.subplots(figsize=(24,16), nrows=2)
# plt.tight_layout()
# fig.tight_layout()

metric_names = ["Volumetric IoU","Number of Components"]

a=0
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
    axes[a].bar(np.arange(len(experiments))*
           (len(experiments)+space_between_experiments)+i,ious,width=0.9,color=colors[i])
    # plt.plot(np.arange(len(experiments))+1,ious,color=colors[i],marker='s',markersize=11)

axes[a].set_ylabel(metric_names[a],labelpad=20)

axes[a].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

axes[a].spines['top'].set_visible(False)
axes[a].spines['right'].set_visible(False)
axes[a].spines['bottom'].set_visible(False)
axes[a].spines['left'].set_visible(False)

### Legend
# box = axes[a].get_position()
# axes[a].set_position([box.x0, box.y0,
#                  box.width, box.height * 0.8])
# Put a legend below current axis
axes[a].legend(names,loc='upper center', bbox_to_anchor=(0.5, 1.3),
          fancybox=True, shadow=True, ncol=7)

a=1
names = []
# axes=[]
# axes.append(ax)
# axes.append(ax)
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
        ious.append(df.iloc[-1]["components"])

    # plt.plot(np.arange(len(experiments))+1,ious,color=colors[i],marker='s',markersize=11)
    space_between_experiments=3
    axes[a].bar(np.arange(len(experiments))*
           (len(experiments)+space_between_experiments)+i,ious,width=0.9,color=colors[i])


axes[a].spines['top'].set_visible(False)
axes[a].spines['right'].set_visible(False)
axes[a].spines['bottom'].set_visible(False)
axes[a].spines['left'].set_visible(False)

axes[a].invert_yaxis()
axes[a].set_ylim(top=1)
axes[a].set_yscale('log')
# locs, labels = axes[a].yticks()
# locs[0] = 1
# axes[a].yticks(locs)
axes[a].set_ylabel(metric_names[a],labelpad=20)

# axes[a].xticks([3,11,19,27,35],[1,2,3,4,5])
axes[a].set(xticks=[3,11,19,27,35], xticklabels=["E1","E2","E3","E4","E5"])

plt.xlabel("Experiment",labelpad=20)
plt.subplots_adjust(hspace=0.08)

plt.savefig("/mnt/raphael/ShapeNet_out/benchmark/iou_components_barplot.png")

plt.show()
a=4



