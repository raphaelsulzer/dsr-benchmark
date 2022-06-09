import numpy as np
import os, sys, subprocess, trimesh
import pandas as pd
sys.path.append("/home/raphael/remote_python/benchmark")
import shutil

from tqdm import tqdm

from datasets.modelnet10 import ModelNet10
from datasets.shapenet import ShapeNet
from datasets.berger import Berger


dataset = ShapeNet()
split = "train"
models = dataset.getModels(splits=[split], scan="4")[split]

for m in models:

    path = os.path.abspath(os.path.join(os.path.dirname(m["scan"]),"..","dse"))
    if(os.path.exists(path)):
        shutil.rmtree(path)