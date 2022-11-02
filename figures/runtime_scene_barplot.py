import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['hatch.linewidth'] = 4.0

# from methods.methods import learning_methods as methods
# methods = ["conv_onet","dgnn","labatut","lig","poco","poisson","sap"]

methods = []

d=dict()
d["name"] = "CONet3D\nsliding-window"
d["path"] = "conv_onet/3d/{}/meshes"
d["runtime"] = [164]
methods.append(d)

d=dict()
d["name"] = "DGNN"
d["path"] = "dgnn/tr/{}"
d["runtime"] = [58]
methods.append(d)

d=dict()
d["name"] = "Labatut $\it{et~al.}$"
d["path"] = "labatut/{}"
d["runtime"] = [31]
methods.append(d)



colors = ["#66c2a4","#f03b20","#034e7b"]
font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)



# metric="iou"
metrics=["iou"]
metric_names = ["Volumetric IoU"]

experiment = "runtime"


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
names = []

for i,m in enumerate(methods):

    names.append(m["name"])

    ious = m[experiment]

    # plt.plot(np.arange(len(experiments))+1,ious,color=colors[i],marker='s',markersize=11)
    space_between_experiments=1
    top = ious[0]
    if len(ious) > 1:
        top = ious[1]

    ax.bar(space_between_experiments * i, top, width=0.3, color=colors[i])

    ax.text((space_between_experiments * i) - 0.075,
            top + 3.12, "{:.0f}".format(top), color='black', fontweight='bold')

    print(m["name"], ious[0])


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.ylabel("Runtime (in seconds)")


# plt.ylabel("Chamfer distance")
plt.xlabel("Method")
ax.xaxis.set_label_coords(0.5, -0.28)
plt.xticks(np.arange(len(methods)),names,rotation=0)

# plt.legend(names,loc=1, bbox_to_anchor=(1.5,0.5))
box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

ax.set_position([box.x0+0.05, box.y0+0.1,
                 box.width, box.height * 0.8])

plt.savefig("/mnt/raphael/ModelNet10_out/dsrv/plots/experiment_{}_barplot_scene.png".format(experiment),
            transparent=True)

plt.show(block=False)
a=4



