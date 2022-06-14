from tqdm import tqdm

from datasets.modelnet10 import ModelNet10
from datasets.shapenet import ShapeNet
from datasets.berger import Berger
from evaluator.eval import MeshEvaluator

import os
import pandas as pd

import shutil


dataset = Berger()
models = dataset.getModels(type="berger")
inpath = "/mnt/raphael/reconbench/igr"
outpath = "/mnt/raphael/reconbench_out/igr"

for m in models:

    src = os.path.join(inpath,m["model"],m["class"],"plots","igr_30000_"+m["model"]+".ply")
    dest = os.path.join(outpath,m["class"])

    os.makedirs(dest, exist_ok=True)
    dest = os.path.join(dest,m["model"]+".ply")

    shutil.copyfile(src,dest)

