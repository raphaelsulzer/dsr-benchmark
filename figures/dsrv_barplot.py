import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['hatch.linewidth'] = 4.0

# from methods.methods import learning_methods as methods
# methods = ["conv_onet","dgnn","labatut","lig","poco","poisson","sap"]

methods = []

d=dict()
d["name"] = "CONet2D"
d["path"] = "conv_onet/2d/{}/meshes"
d["modelnet"] = [0.853,0.889]
d["shapenet"] = [0.683,0.780]
d["runtime"] = [0.51,0.53]
d["components"] = [3.2,3.0]
methods.append(d)

d=dict()
d["name"] = "CONet3D"
d["path"] = "conv_onet/3d/{}/meshes"
d["modelnet"] = [0.885,0.923]
d["shapenet"] = [0.510,0.823]
d["runtime"] = [0.40,0.41]
d["components"] = [1.5,2.7]
methods.append(d)

d=dict()
d["name"] = "SAP"
d["path"] = "sap/tr/{}/meshes"
d["modelnet"] = [0.903,0.914]
d["shapenet"] = [0.549,0.809]
d["runtime"] = [0.088,0.114]
d["components"] = [10.5,3.4]
methods.append(d)

d=dict()
d["name"] = "POCO"
d["path"] = "poco/tr/{}/meshes"
d["modelnet"] = [0.907,0.917]
d["shapenet"] = [0.409,0.815]
d["runtime"] = [15.74,15.77]
d["components"] = [16.3,6.2]
methods.append(d)

d=dict()
d["name"] = "P2S"
d["path"] = "lig/{}"
d["modelnet"] = [0.842,0.859]
d["shapenet"] = [0.807,0.836]
d["runtime"] = [80.57,83.27]
d["components"] = [7.8,2.9]
methods.append(d)

d=dict()
d["name"] = "DGNN"
d["path"] = "dgnn/tr/{}"
d["modelnet"] = [0.866]
d["shapenet"] = [0.844]
d["runtime"] = [0.39]
d["components"] = [1.3]
methods.append(d)

d=dict()
d["name"] = "SPSR"
d["path"] = "poisson/{}"
d["modelnet"] = [0.807]
d["shapenet"] = [0.771]
d["runtime"] = [1.25]
d["components"] = [3.2]
methods.append(d)

d=dict()
d["name"] = "Labatut $\it{et~al.}$"
d["path"] = "labatut/{}"
d["modelnet"] = [0.839]
d["shapenet"] = [0.803]
d["runtime"] = [0.18]
d["components"] = [1.2]
methods.append(d)


# colors = sns.color_palette("Set2", len(methods))

colors = ["#ccece6","#66c2a4","#238b45","#005824","#0E381C","#d0d1e6","#034e7b"]
colors = ["#ccece6","#66c2a4","#238b45","#005824","#0E381C","#f03b20","#d0d1e6","#034e7b"]
font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)



# metric="iou"
metrics=["iou"]
metric_names = ["Volumetric IoU","Number of Components","Runtime (in seconds)"]
experiments = ["shapenet","components","runtime"]
ek = 0


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
names = []

for i,m in enumerate(methods):

    names.append(m["name"])

    ious = m[experiments[ek]]

    # plt.plot(np.arange(len(experiments))+1,ious,color=colors[i],marker='s',markersize=11)
    space_between_experiments=1
    # ax.bar(space_between_experiments*i,ious[0],width=0.9,color='white',hatch='/',edgecolor=colors[i])
    if m["name"] == "Labatut $\it{et~al.}$":
        ax.bar(space_between_experiments * i, ious[0], width=0.9, color=colors[i],hatch="/",edgecolor="orange")
    else:
        ax.bar(space_between_experiments*i,ious[0],width=0.9,color=colors[i])
    top = ious[0]
    if len(ious) > 1:
        top = ious[1]
        ax.bar(space_between_experiments*i,ious[1]-ious[0],width=0.9,color='orange',bottom=ious[0])
        # ax.text((space_between_experiments * i) - 0.47,
        #         top* 1.02, "+{:.2f}".format(ious[1] - ious[0]), color='black', fontweight='bold')
        ax.text((space_between_experiments * i) - 0.47,
                top* 1.02, "{:.2f}".format(ious[1]), color='black', fontweight='bold')
    else:
        ax.text((space_between_experiments * i) - 0.47,
                top* 1.02, "{:.2f}".format(top), color='black', fontweight='bold')
        # top = ious[0]

    print(m["name"], ious[0])


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)


if experiments[ek] == "runtime":
    ax.set_yscale('log')
else:
    plt.gca().set_ylim(bottom=0.4)

plt.ylabel(metric_names[ek])



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

plt.savefig("/mnt/raphael/ModelNet10_out/dsrv/plots/experiment_{}_barplot_with_dgnn.png".format(experiments[ek]),
            transparent=True)

plt.show(block=False)
a=4



