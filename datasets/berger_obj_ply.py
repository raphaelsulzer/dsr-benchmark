import os, sys, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
import numpy as np
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from berger import Berger

dataset = Berger()
models = dataset.getModels(type="berger")


for m in models:
    try:
        src = os.path.join("/mnt/raphael/reconbench_out/p2m/poisson",m["class"],m["model"]+".ply")
        dest = os.path.join("/mnt/raphael/reconbench_out/p2m/poisson",m["class"],m["model"]+".obj")
        os.rename(src,dest)
    except:
        pass