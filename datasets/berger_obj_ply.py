import os, sys, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
import numpy as np
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from berger import Berger

dataset = Berger()
models = dataset.getModels(type="berger",scan=["0","1"])


for m in models:

    print(m["class"],m["model"])

    mesh = trimesh.load(os.path.join("/mnt/raphael/reconbench_out/p2m/poisson",m["class"],m["model"],"recon_iter_3000.obj"))
    mesh.export(os.path.join("/mnt/raphael/reconbench_out/p2m/poisson",m["class"],m["model"]+".ply"))

