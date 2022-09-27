import os, sys, subprocess, shutil
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
import numpy as np
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from berger import Berger

dataset = Berger()
models = dataset.getModels(scan_conf="mvs4")


for m in models:

    print(m["class"],m["model"])

    infile = os.path.join("/mnt/raphael/reconbench_out/mvs/igr",m["model"],'4',"plots","igr_100000_"+m["model"]+".ply")
    outpath = os.path.join("/mnt/raphael/reconbench_out/igr",m["class"])
    os.makedirs(outpath,exist_ok=True)
    outfile = os.path.join(outpath,m["model"]+".ply")

    shutil.copyfile(infile,outfile)

