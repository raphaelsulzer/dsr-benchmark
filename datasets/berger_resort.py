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

    # infile = os.path.join("/mnt/raphael/reconbench_out/mvs/igr",m["model"],m["class"],"plots","igr_100000_"+m["model"]+".ply")
    # outpath = os.path.join("/mnt/raphael/reconbench_out/igr",m["class"])

    ## Labatut
    # infile = os.path.join("/mnt/raphael/ShapeNet_out/benchmark/labatut/reconbench",m["model"],m["model"]+"_rt_2.5.ply")
    # outpath = os.path.join("/mnt/raphael/reconbench_out/labatut",m["class"])

    ## SAP
    infile = os.path.join("/mnt/raphael/reconbench_out/mvs/sap/mvs4",m["model"],"vis/mesh","1600.ply")
    outpath = os.path.join("/mnt/raphael/reconbench_out/sap",m["class"])

    os.makedirs(outpath,exist_ok=True)
    outfile = os.path.join(outpath,m["model"]+".ply")

    shutil.copyfile(infile,outfile)

