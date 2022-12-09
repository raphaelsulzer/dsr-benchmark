import os, sys, subprocess
sys.path.append("/home/raphael/remote_python/benchmark")
from tqdm import tqdm
from datasets.modelnet10 import ModelNet10
from datasets.ksr42 import KSR42
import time

POISSON_EXE = "/home/rsulzer/cpp/PoissonRecon/Bin/Linux/PoissonRecon"

dataset = KSR42()
models = dataset.getModels()


# dataset = ShapeNet()
# split = "test100"
# models = dataset.getModels(splits=[split], scan="6")[split]
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/poisson/shapenet10000"

# dataset = Berger()
# models = dataset.getModels(scan_conf=["mvs4"])
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/poisson/reconbench"

depth = [8]
boundary = [2]

t0 = time.time()



for m in tqdm(models,ncols=50):

    b = boundary[0]
    d = depth[0]
    # try:
    os.makedirs(os.path.join(dataset.path,m["class"]), exist_ok=True)
    command = [POISSON_EXE,
               "--in", m["scan_ply"],
               "--out", os.path.join(dataset.path,m["class"],m["mesh"]),
               "--depth", str(d),
               "--bType", str(b)]
    print("run command: ", *command)
    p = subprocess.Popen(command)
    # p = subprocess.Popen(command, shell=False,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p.wait()

    # except Exception as e:
    #     raise
    #     print(e)
    #     print("Skipping {}/{}".format(m["class"], m["model"]))

tt = time.time() - t0

print("models: ",len(models))
print("full time: ",tt)
print("average time: ",tt/len(models))



