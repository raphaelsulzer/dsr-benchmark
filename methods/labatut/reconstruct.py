import numpy as np
import os, sys, subprocess
sys.path.append("/home/raphael/remote_python/benchmark")
from tqdm import tqdm
from datasets.modelnet10 import ModelNet10
from datasets.shapenet import ShapeNet
from datasets.berger import Berger

# dataset = ModelNet10()
# split = "test"
# models = dataset.getModels(splits=[split],classes=["bathtub", "bed", "desk", "dresser", "nightstand", "toilet"])[split]
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/labatut/modelnet"

# dataset = ShapeNet()
# split = "test100"
# models = dataset.getModels(splits=[split],scan="4")[split]
# outpath = os.path.join("/mnt/raphael/ShapeNet_out/benchmark/labatut/shapenet3000")

dataset = Berger()
models = dataset.getModels(scan=["4"])
outpath = "/mnt/raphael/ShapeNet_out/benchmark/labatut/reconbench"

# sigma = [0.01]
# lam = [1.0]
# alpha = [16]


# sigma = [0.001]
# lam = [5]
# alpha = [32]

sigma = [0.01]
lam = [2.5]
alpha = [32]



# 16,0.01000,1.00000



for m in tqdm(models,ncols=50):
    for s in sigma:
        for l in lam:
            for a in alpha:
                os.makedirs(os.path.join(outpath, m["class"]), exist_ok=True)
                command = ["/home/raphael/cpp/mesh-tools/build/release/labatut",
                           "-i", m["scan"],
                           "-o", os.path.join(outpath,m["class"],m["model"]),
                           "-s", "npz",
                           "--sigma", str(s),
                           "--alpha", str(a),
                           "--gco", "angle-"+str(l),
                           "--closed","1"]
                print("run command: ", *command)
                p = subprocess.Popen(command)
                p.wait()
                b=5

a=5
# make a grid search over 10% of the test set





