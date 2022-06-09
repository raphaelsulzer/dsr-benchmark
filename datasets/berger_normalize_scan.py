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

    scan = np.load(m["scan"])

    points = scan["points"]/75.0
    normals = scan["normals"]
    sensor_position = scan["sensor_position"]/75.0


    outpath = os.path.join("/mnt/raphael/reconbench/scan_berger_1",m["class"])
    os.makedirs(outpath,exist_ok=True)
    filename = os.path.join(outpath,m["scan_conf"])

    np.savez(filename, points=points, normals=normals, sensor_position=sensor_position)



    a=5


