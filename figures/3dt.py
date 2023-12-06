import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import Delaunay
from itertools import combinations
from pycompod import PolyhedralComplexExporter

pcd = o3d.io.read_point_cloud("./pointcloud.ply")
points = np.asarray(pcd.points)

# mesh = trimesh.load("./pointcloud.obj")
# points = mesh.vertices

tri = Delaunay(points)

triangles = []
for sim in tri.simplices:

    for triangle in combinations(sim,3):
        triangles.append(triangle)


triangles = np.array(triangles)
triangles = np.unique(triangles,axis=0)

exp = PolyhedralComplexExporter()
exp.write_surface("./3dt.obj",points=tri.points,facets=triangles)

a=5