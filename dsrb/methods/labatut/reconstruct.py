import time

import numpy as np
import os, sys, subprocess
sys.path.append("/home/raphael/remote_python/benchmark")
from tqdm import tqdm
from datasets.modelnet10 import ModelNet10
from datasets.shapenet import ShapeNet
from datasets.berger import Berger
from datasets.real import Real

# dataset = ModelNet10()
# split = "test"
# models = dataset.getModels(splits=[split],classes=["bathtub", "bed", "desk", "dresser", "nightstand", "toilet"])[split]
# outpath = "/mnt/raphael/ModelNet10_out/benchmark/labatut/modelnet"

ds = Real(classes="50000")
models = ds.getModels()
outpath = os.path.join("/mnt/raphael/real_out/labatut/shapenet3000")

# dataset = ShapeNet()
# split = "test10"
# models = dataset.getModels(splits=[split],scan_conf="4")[split]
# outpath = os.path.join("/mnt/raphael/ShapeNet_out/benchmark/labatut/shapenet3000")

# dataset = Berger()
# models = dataset.getModels(scan=["4"])
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/labatut/reconbench"

# dataset = Berger()
# models = dataset.getModels(type="berger")
# outpath = "/mnt/raphael/reconbench_out/labatut"

# sigma = [0.01]
# lam = [1.0]
# alpha = [16]

#### the shapenet experiments are run with this setting:
# sigma = [0.01]
# lam = [2.5]
# alpha = [32]

sigma = [0.001]
lam = [5]
alpha = [32]


t0 = time.time()

for m in tqdm(models,ncols=50):
    s = sigma[0]
    l = lam[0]
    a = alpha[0]
    os.makedirs(os.path.join(outpath, m["class"]), exist_ok=True)
    command = ["/home/raphael/cpp/mesh-tools/build/release/labatut",
               "-i", m["scan_ply"],
               "-o", os.path.join(outpath,m["class"],m["model"]),
               "-s", "npz",
               "--sigma", str(s),
               "--alpha", str(a),
               "--gco", "angle-"+str(l),
               "--closed","1"]
    print("run command: ", *command)
    # p = subprocess.Popen(command, shell=False,
    # stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p = subprocess.Popen(command)
    p.wait()

tt = time.time() - t0

print("models: ",len(models))
print("full time: ",tt)
print("average time: ",tt/len(models))

a=5
# make a grid search over 10% of the test set





