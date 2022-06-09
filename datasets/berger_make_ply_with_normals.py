import os, sys, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
import numpy as np
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from berger import Berger

dataset = Berger()
models = dataset.getModels(type="berger")

import open3d as o3d

for m in models:

    scan = np.load(m["scan"])
    pcd = o3d.geometry.PointCloud()

    # orient normals towards sensor
    points = scan["points"]

    pcd.points = o3d.utility.Vector3dVector(points)


    pcd.estimate_normals()

    normals = np.asarray(pcd.normals)
    sensor_vec = scan["sensor_position"] - points

    ip = np.einsum('ij,ij->i',normals, sensor_vec)
    normals[ip<0] = -normals[ip<0]

    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    pcd.normals = o3d.utility.Vector3dVector(normals)


    o3d.io.write_point_cloud(m["scan"][:-3]+"ply", pcd)


    a=5


