import numpy as np
import os, sys, subprocess
sys.path.append("/home/raphael/remote_python/benchmark")
from tqdm import tqdm
from datasets.modelnet10 import ModelNet10
from datasets.shapenet import ShapeNet
from datasets.berger import Berger
import open3d as o3d

# dataset = ModelNet10()
# split = "test"
# models = dataset.getModels(splits=[split],classes=["bathtub", "bed", "desk", "dresser", "nightstand", "toilet"])[split]
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/poisson/modelnet"

# dataset = ShapeNet()
# split = "test100"
# models = dataset.getModels(splits=[split], scan="6")[split]
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/poisson/shapenet10000"

# dataset = Berger()
# models = dataset.getModels(scan=["4"])
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/poisson/reconbench"


depth = [10]
boundary = [2]

# 16,0.01000,1.00000


os.makedirs(os.path.join(outpath,"ply"),exist_ok=True)

replace = False

for m in tqdm(models,ncols=50):

    # if not os.path.isfile(os.path.join(outpath, "..", "ply", m["class"], m["model"] + ".ply")) or replace:
    scan = np.load(m["scan"])
    pcd = o3d.geometry.PointCloud()

    # orient normals towards sensor
    points = scan["points"]
    normals = scan["normals"]
    sensor_vec = scan["sensor_position"] - points

    ip = np.einsum('ij,ij->i',normals, sensor_vec)
    normals[ip<0] = -normals[ip<0]

    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)


    o3d.io.write_point_cloud(m["scan"][:-3]+"ply", pcd)

    for b in boundary:
        for d in depth:
            try:
                os.makedirs(os.path.join(outpath,m["class"]), exist_ok=True)
                command = ["/home/raphael/cpp/PoissonReconOri/Bin/Linux/PoissonRecon",
                           "--in", m["scan"][:-3]+"ply",
                           "--out", os.path.join(outpath,m["class"],m["model"]),
                           "--depth", str(d),
                           "--bType", str(b)]
                print("run command: ", *command)
                p = subprocess.Popen(command)
                p.wait()

            except Exception as e:
                raise
                print(e)
                print("Skipping {}/{}".format(m["class"], m["model"]))

a=5
# make a grid search over 10% of the test set





