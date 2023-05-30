import os, sys, subprocess
sys.path.append("/home/raphael/remote_python/benchmark")
from tqdm import tqdm
from datasets.modelnet10 import ModelNet10
from datasets.ksr42 import KSR42
import time

dataset = ModelNet10()
split = "test"
models = dataset.getModels(splits=[split],classes=["bathtub", "bed", "desk", "dresser", "nightstand", "toilet"])[split]
outpath = "/mnt/raphael/ModelNet10_out/benchmark/poisson/modelnet"


dataset = KSR42()
models = dataset.getModels(classes="Advanced")


# dataset = ShapeNet()
# split = "test100"
# models = dataset.getModels(splits=[split], scan="6")[split]
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/poisson/shapenet10000"

# dataset = Berger()
# models = dataset.getModels(scan_conf=["mvs4"])
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/poisson/reconbench"



depth = [8]
boundary = [2]

# os.makedirs(os.path.join(outpath,"ply"),exist_ok=True)

replace = False

# for m in tqdm(models,ncols=50):
#
#     # # if not os.path.isfile(os.path.join(outpath, "..", "ply", m["class"], m["model"] + ".ply")) or replace:
#     # scan = np.load(m["scan"])
#     # pcd = o3d.geometry.PointCloud()
#     #
#     # # orient normals towards sensor
#     # points = scan["points"]
#     # normals = scan["normals"]
#     # sensor_vec = scan["sensor_position"] - points
#     #
#     # ip = np.einsum('ij,ij->i',normals, sensor_vec)
#     # normals[ip<0] = -normals[ip<0]
#     #
#     # normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
#     # pcd.points = o3d.utility.Vector3dVector(points)
#     # pcd.normals = o3d.utility.Vector3dVector(normals)
#     #
#     #
#     # o3d.io.write_point_cloud(m["scan"][:-3]+"ply", pcd)
#
#     for b in boundary:
#         for d in depth:
#             try:
#                 os.makedirs(os.path.join(outpath,m["class"]), exist_ok=True)
#                 command = ["/home/raphael/cpp/PoissonReconOri/Bin/Linux/PoissonRecon",
#                            "--in", m["scan_ply"],
#                            "--out", os.path.join(outpath,m["class"],m["model"]),
#                            "--depth", str(d),
#                            "--bType", str(b)]
#                 print("run command: ", *command)
#                 p = subprocess.Popen(command)
#                 p.wait()
#
#             except Exception as e:
#                 raise
#                 print(e)
#                 print("Skipping {}/{}".format(m["class"], m["model"]))
#
# a=5

t0 = time.time()

for m in tqdm(models,ncols=50):

    b = boundary[0]
    d = depth[0]
    # try:
    os.makedirs(os.path.join(outpath,m["class"]), exist_ok=True)
    command = ["/home/raphael/cpp/PoissonReconOri/Bin/Linux/PoissonRecon",
               "--in", m["scan_ply"],
               "--out", os.path.join(outpath,m["class"],m["model"]),
               "--depth", str(d),
               "--bType", str(b)]
    # print("run command: ", *command)
    # p = subprocess.Popen(command)
    p = subprocess.Popen(command, shell=False,
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p.wait()

    # except Exception as e:
    #     raise
    #     print(e)
    #     print("Skipping {}/{}".format(m["class"], m["model"]))

tt = time.time() - t0

print("models: ",len(models))
print("full time: ",tt)
print("average time: ",tt/len(models))



