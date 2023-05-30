import open3d as o3d
import numpy as np


def npz2ply(infile,outfile,normal_type=None):

    data = np.load(infile)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data["points"])
    if normal_type == "normals":
        pcd.normals = o3d.utility.Vector3dVector(data["normals"])
    elif normal_type == "sensor_vec":
        pcd.normals = o3d.utility.Vector3dVector(data["sensor_pos"] - data["points"])
    elif normal_type == "sensor_pos":
        pcd.normals = o3d.utility.Vector3dVector(data["sensor_pos"])
    else:
        print("specify a normal type = ['normals', 'sensor_vec']")

    if "colors" in data.keys():
        pcd.colors = o3d.utility.Vector3dVector(data["colors"])

    o3d.io.write_point_cloud(outfile, pcd)