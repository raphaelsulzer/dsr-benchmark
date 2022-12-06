import numpy as np
import os, sys, subprocess, trimesh
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "datasets"))
from modelnet10 import ModelNet10
from shapenet import ShapeNet
from berger import Berger
import vedo

light_dict = dict()
cam_dict = dict()
light_dict["modelnet"] = [-1, 1, 1]
cam_dict["modelnet"] = dict(pos=(-1.950, 0.9284, 0.4101),
           focalPoint=(0.08787, -0.02095, -0.04916),
           viewup=(0.1802, -0.08722, 0.9798),
           distance=2.295,
           clippingRange=(0.3031, 4.935))
light_dict["shapenet"] = [1, 1, 1]
cam_dict["shapenet"] = dict(pos=(1.493, 0.7150, 0.9952),
           focalPoint=(0.06518, 0.02076, 0.09272),
           viewup=(-0.3424, 0.9242, -0.1691),
           distance=1.826,
           clippingRange=(1.166, 2.902))

# path = "/home/adminlocal/PhD/data/benchmark/meshes/modelnet"
# files = os.listdir(path)
# img_path = "/home/adminlocal/PhD/data/benchmark/images/modelnet"
# os.makedirs(img_path,exist_ok=True)
# dataset = "modelnet"

path = "/home/adminlocal/PhD/data/benchmark/meshes/shapenet"
folders = os.listdir(path)
img_path = "/home/adminlocal/PhD/data/benchmark/images/shapenet"
os.makedirs(img_path,exist_ok=True)
dataset = "shapenet"

for fol in folders:
    files = os.listdir(os.path.join(path,fol))
    for f in files:

        data = vedo.load(os.path.join(path,fol,f,"mesh","mesh.off"))
        data = vedo.Mesh(data, c=[180, 180, 180])
        data = data.computeNormals().phong()
        image_file = os.path.join(img_path, fol+"_"+f + ".png")

        light = light_dict[dataset]

        p2 = vedo.Point(light, c='y')
        l2 = vedo.Light(p2, c='w', intensity=1)

        cam = cam_dict[dataset]

        interactive = False

        p = vedo.show(data, l2, size=(700, 700), camera=cam, interactive=interactive)
        vedo.io.screenshot(image_file)
        p.close()











