import os, sys, subprocess, trimesh
import pandas as pd
sys.path.append("/home/raphael/remote_python/benchmark")

from tqdm import tqdm

from datasets.modelnet10 import ModelNet10
from datasets.shapenet import ShapeNet
from datasets.berger import Berger


dataset = ShapeNet(classes=["02691156"])
split = "test100"
models = dataset.getModels(splits=[split], scan="4")[split]
models = models[:5]
for m in models:
    outpath = os.path.join("/home/adminlocal/PhD/data/benchmark/sampling_figure",m["model"])
    command = ["scp","enpc:"+m["scan"],outpath]
    print(*command)
    # p = subprocess.Popen(command)
    # p.wait()

    command = ["scp","enpc:"+m["mesh"],outpath]
    print(*command)
    # p = subprocess.Popen(command)
    # p.wait()

models = dataset.getModels(splits=[split], scan="6")[split]
models = models[:5]
for m in models:
    outpath = os.path.join("/home/adminlocal/PhD/data/benchmark/sampling_figure",m["model"])
    command = ["scp", "enpc:" + m["scan"], outpath]
    print(*command)
    # p = subprocess.Popen(command)
    # p.wait()

    command = ["scp", "enpc:" + m["mesh"], outpath]
    print(*command)
    # p = subprocess.Popen(command)
    # p.wait()