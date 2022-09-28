import os, sys, subprocess, trimesh
import open3d as o3d
import numpy as np


sure_dir = "/home/adminlocal/PhD/cpp/mesh-tools/build/release"

# mesh = os.path.join(path,"mesh.off")
# mesh = trimesh.load(mesh)
stddev = 0.0025

## uniform
# n=3000
# uniform_points = mesh.sample(n)
# noise = stddev * np.random.randn(*uniform_points.shape)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(uniform_points+noise)
# o3d.io.write_point_cloud(os.path.join(path,"uniform_"+str(n)+".ply"),pcd)
#
# command = [sure_dir+"/normal",
#            "-w", path,
#            "-i", "uniform_"+str(n)+".ply",
#            "--neighborhood", "30"]
# print(*command)
# p = subprocess.Popen(command)
# p.wait()


## scan
# cams = 5
# n=3000
# command = [sure_dir+"/scan",
#            "-w", path,
#            "-i", "mesh.off",
#            "-o", "scan_"+str(n),
#            "--cameras", str(cams),
#            "--points", str(n),
#            "--noise", str(stddev),
#            "--normal_method", "jet",
#            "--normal_neighborhood", "30",
#            "--export", "all"]
# print(*command)
# p = subprocess.Popen(command)
# p.wait()



# lidar.ply is Ignatius02.ply lidar scan with the following sensor coords:
# -1.851682 -1.862650 -5.837185

# clouds = ["lidar.ply",
#           "scan_100000.ply",
#           "uniform_100000.ply",
#           "mvs.ply"]
#
# cam = dict(pos=(0.8373, -0.6862, 0.4311),
#            focalPoint=(-0.07846, 0.02467, 0.2840),
#            viewup=(-0.09310, 0.08535, 0.9920),
#            distance=1.169,
#            clippingRange=(0.1948, 2.214))
#
# # mesh = vedo.load(mesh)
#
# for c in clouds:
#
#     file = os.path.join(path,c)
#
#     if file.endswith(".ply"):
#         data = vedo.load(file)
#         points = vedo.Points(data, r=15.0)
#         points = points.computeNormalsWithPCA()
#
#     else:
#         data = np.load(file)
#
#         points = data["points"]
#         sensor_vec = data["sensor_position"] - points
#         sensor_vec_norm = sensor_vec / np.linalg.norm(sensor_vec, axis=1)[:, np.newaxis]
#
#         points = vedo.Points(points, r=15.0)
#         points = points.computeNormalsWithPCA()
#
#     light=[0.3, -0.3, 0.5]
#     p2 = vedo.Point(light, c='y')
#     l2 = vedo.Light(p2, c='w', intensity=1)
#
#
#
#     image_file = os.path.join(path,"img",file.split(".")[0]+".png")
#
#
#
#     # ### without aux
#     interactive = True
#     p = vedo.show(points, p2,l2, size=(700, 700), camera=cam, interactive=interactive)
#     vedo.io.screenshot(image_file)
#     p.close()
#
#     a=5

class Ignatius:

    def __init__(self):
        # self.path = "/home/adminlocal/PhD/PhD-Thesis-Harvard/python/scanning/data/bunny"
        # self.path = "/home/adminlocal/PhD/data/benchmark/Ignatius/"

        self.path = "/Users/mba222/PhD/data/Ignatius"

    def uniform_sampling(mesh,stddev):
        n=250000
        uniform_points = mesh.sample(n)
        noise = stddev * np.random.randn(*uniform_points.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(uniform_points+noise)
        o3d.io.write_point_cloud(os.path.join(path,"uniform_"+str(n)+".ply"),pcd)

        command = [sure_dir+"/normal",
                   "-w", path,
                   "-i", "uniform_"+str(n)+".ply",
                   "--neighborhood", "30"]
        print(*command)
        p = subprocess.Popen(command)
        p.wait()

    def orientMVS(self):

        import open3d as o3d

        pc=o3d.io.read_point_cloud(os.path.join(self.path,"mvs_sensor.ply"))

        trans=np.loadtxt(os.path.join(self.path,"Ignatius","Ignatius_trans.txt"))

        pc=pc.transform(trans)

        trans = np.eye(4,4)
        trans[0,3]=-0.2
        trans[1,3]=-0.03
        trans[2,3]=0.03

        pc=pc.transform(trans)


        o3d.io.write_point_cloud(os.path.join(self.path,"mvs_sensor_trans.ply"),pc)


        a=5

    def scanMVS(self):
        cams=5
        n=250000
        command = [sure_dir+"/scan",
                   "-w", path,
                   "-i", "mesh.off",
                   "-o", "scan_"+str(n),
                   "--cameras", str(cams),
                   "--points", str(n),
                   "--noise", str(stddev),
                   # "--normal_method", "jet",
                   # "--normal_neighborhood", "30",
                   "--export", "all",
                   "-e","v"]
        # TODO: export this scan with sensors as normal field
        print(*command)
        p = subprocess.Popen(command)
        p.wait()


if __name__ == "__main__":

    ign = Ignatius()

    ign.orientMVS()







