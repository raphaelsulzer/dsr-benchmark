import os, sys, subprocess, trimesh
import pandas as pd
sys.path.append("/home/raphael/remote_python/benchmark")

from tqdm import tqdm

from datasets.modelnet10 import ModelNet10
from datasets.shapenet import ShapeNet
from datasets.berger import Berger

import vedo
import open3d as o3d

path = "/home/adminlocal/PhD/data/benchmark/scan_example"
mesh = os.path.join(path,"mesh.off")
mesh = trimesh.load(mesh)

n=3000
uniform_points = mesh.sample(n)


trimesh.export(uniform_points,os.path.join(path,"uniform.ply"))
