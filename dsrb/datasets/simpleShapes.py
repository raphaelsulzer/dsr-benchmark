import os, sys, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libmesh import check_mesh_contains
from tqdm import tqdm
import numpy as np
from glob import glob
import trimesh
import open3d as o3d

DEBUG = 0


class simpleShapes:

    def __init__(self,path="/home/rsulzer/data/simpleShapes", mesh_tools_dir="/home/rsulzer/cpp/mesh-tools/build/release"):

        self.path = path
        self.model_dicts = []
        self.mesh_tools_dir = mesh_tools_dir
        self.POISSON_EXE = "/home/rsulzer/cpp/PoissonRecon/Bin/Linux/PoissonRecon"


    def getModels(self,hint=None):

        models = np.genfromtxt(os.path.join(self.path,"models.lst"),dtype=str)
        for m in models:

            if hint is not None:
                if hint not in m:
                    continue

            d = {}
            d["class"] = ""
            d["model"] = m
            d["scan_ply"] = glob(os.path.join(self.path,d["class"],m,'*.ply'))[0]

            d["occ"] = os.path.join(self.path,d["class"],m,"eval","points.npz")
            d["pointcloud"] = os.path.join(self.path,d["class"],m,"eval","pointcloud.npz")

            d["pointcloud_ply"] = os.path.join(self.path,d["class"],m,"pointcloud.ply")
            d["mesh"] = os.path.join(self.path,d["class"],m,"mesh_unit.off")
            d["planes"] = os.path.join(self.path,d["class"],m,"planes","planes.vg")

            d["ksr"] = {}
            d["ksr"]["surface"] = os.path.join(self.path,d["class"],m,"ksr",'{}',"surface.off")
            d["ksr"]["partition"] = os.path.join(self.path,d["class"],m,"ksr",'{}',"partition.kgraph")

            d["abspy"] = {}
            d["abspy"]["surface"] = os.path.join(self.path,d["class"],m,"abspy",'{}',"surface.off")
            d["abspy"]["partition"] = os.path.join(self.path,d["class"],m,"abspy",'{}',"partition.ply")

            self.model_dicts.append(d)

        return self.model_dicts



    def scan(self,scan_setting="4",scanner_dir="/home/raphael/cpp/mesh-tools/build/release/scan",
             normal_method='jet', normal_neighborhood=30, normal_orient=1):

        if(len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)

        scan = scan_settings[scan_setting]

        for m in self.model_dicts:

            os.makedirs(os.path.join(self.path,"scan",m["class"]),exist_ok=True)

            command = [scanner_dir,
                       "-w", str(self.path),
                       "-i", "mesh/1/"+m["class"]+".off",
                       "-o", str(os.path.join("scan",m["class"],scan_setting)),
                       "--points", scan["points"],
                       "--noise", scan["noise"],
                       "--outliers", scan["outliers"],
                       "--cameras", scan["cameras"],
                       "--normal_method", "jet",
                       "--export", "all",
                       "--normal_neighborhood", normal_neighborhood,
                       "--normal_method", normal_method,
                       "--normal_orient", normal_orient]
            print(*command)
            p = subprocess.Popen(command)
            p.wait()


    def estimNormals(self, method='jet', neighborhood=30, orient=1):
        if (len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)

        for m in tqdm(self.model_dicts, ncols=50):
            try:
                command = [str(os.path.join(self.mesh_tools_dir, "normal")),
                           "-w", str(self.path),
                           "-s", "npz",
                           "-i", str(os.path.join("scan", m["model"], self.scan_conf + ".npz")),
                           "-o", str(os.path.join("scan", m["model"], self.scan_conf)),
                           "--method", method,
                           "--neighborhood", str(neighborhood),
                           "--orient", str(orient),
                           "--overwrite", "1"]
                print(*command)
                p = subprocess.Popen(command)
                p.wait()
            except Exception as e:
                print(e)
                print("Skipping {}/{}".format(m["class"], m["model"]))


    def makePoisson(self, depth=8, boundary=2):

        for m in tqdm(self.model_dicts, ncols=50):
            # try:
            command = [self.POISSON_EXE,
                       "--in", m["scan_ply"],
                       "--out", os.path.join(self.path, m["class"], m["mesh"]),
                       "--depth", str(depth),
                       "--bType", str(boundary)]
            print("run command: ", *command)
            p = subprocess.Popen(command)
            # p = subprocess.Popen(command, shell=False,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p.wait()



    def standardize(self,padding=0.1):

        for m in tqdm(self.model_dicts):

            mesh = o3d.io.read_triangle_mesh(m["mesh"])
            center = mesh.get_axis_aligned_bounding_box().get_center()

            mesh = mesh.translate(-center)


            max_bound = np.vstack([np.abs(mesh.get_min_bound()), mesh.get_max_bound()])
            col_index = np.argmax(max_bound,axis=1)[1]
            scale = np.abs(mesh.get_min_bound()[col_index]) + np.abs(mesh.get_max_bound()[col_index])


            mesh = mesh.scale((1-padding) / scale, [0, 0, 0])

            o3d.io.write_triangle_mesh(os.path.splitext(m["mesh"])[0]+"_unit.off",mesh)



    def makePointcloudPLY(self,n_points=100000,std_noise=0.0):

        print("Writing pointclouds for reconstruction input...\n")

        for m in tqdm(self.model_dicts):

            data = np.load(m["pointcloud"])

            ind = np.random.randint(0, data["points"].shape[0], size = (n_points,))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data["points"][ind])
            pcd.normals = o3d.utility.Vector3dVector(data["normals"][ind])
            o3d.io.write_point_cloud(m["pointcloud_ply"], pcd)




    def sample(self,n_points=100000):

        print("Sample points on surface and in bounding box for evaluation...\n")


        if(len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)


        loc = np.zeros(3)
        scale = 1.0


        for m in tqdm(self.model_dicts):

            mesh = trimesh.load(m["mesh"])

            # surface points
            points_surface, fid = mesh.sample(n_points,return_index=True)
            normals = mesh.face_normals[fid]

            fpath = os.path.join(self.path,m["class"],m["model"],"eval")
            os.makedirs(fpath,exist_ok=True)
            filename = os.path.join(fpath,"pointcloud.npz")
            np.savez(filename, points=points_surface, normals=normals, loc=loc, scale=scale)

            if DEBUG:
                print('Writing points: %s' % filename)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_surface)
                pcd.normals = o3d.utility.Vector3dVector(normals)
                o3d.io.write_point_cloud(os.path.join(fpath,"pointcloud.ply"), pcd)



            # IoU points

            n_points_uniform = int(n_points * 0.5)
            n_points_surface = n_points - n_points_uniform

            boxsize = 1
            points_uniform = np.random.rand(n_points_uniform, 3)
            points_uniform = boxsize * (points_uniform - 0.5)
            points_surface = mesh.sample(n_points_surface)
            points_surface += 0.05 * np.random.randn(n_points_surface, 3)
            points = np.concatenate([points_uniform, points_surface], axis=0)

            occupancies = check_mesh_contains(mesh, points)

            colors = np.zeros(shape=(n_points, 3)) + [0, 0, 1]
            colors[occupancies] = [1,0,0]



            dtype = np.float16
            points = points.astype(dtype)
            occupancies = np.packbits(occupancies)

            filename = os.path.join(self.path,m["class"],m["model"],"eval","points.npz")
            np.savez(filename, points=points, occupancies=occupancies,loc=loc, scale=scale)

            if DEBUG:
                print('Writing points: %s' % filename)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(os.path.join(fpath,"points.ply"), pcd)




if __name__ == '__main__':

    ds = simpleShapes()

    ds.getModels()

    # ds.standardize()
    # ds.sample(n_points=1000000)

    ds.makePointcloudPLY(n_points=100000)







