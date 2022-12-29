import numpy as np
import os
import open3d as o3d
import subprocess
from tqdm import tqdm

class Real:



    def __init__(self,path="/mnt/raphael/real/input", outpath="/mnt/raphael/real_out",
                 classes=[],
                MESH_TOOLS_DIR="/home/raphael/cpp/mesh-tools/build/release",
                POISSON_EXE = "/home/raphael/cpp/PoissonReconOri/Bin/Linux/PoissonRecon"
    ):



        self.path = path
        self.outpath = outpath
        self.MESH_TOOLS_DIR = MESH_TOOLS_DIR
        self.POISSON_EXE = POISSON_EXE
        self.model_dicts = []

        if not isinstance(classes,list):
            classes = [classes]

        if len(classes) == 0:
            with open(os.path.join(self.path, "classes.lst"), 'r') as f:
                categories = f.read().split('\n')
            if '' in categories:
                categories.remove('')
            self.classes = categories
        else:
            self.classes = classes

    def getModels(self,hint=None):

        self.model_dicts = []

        for  c in self.classes:

            models = np.genfromtxt(os.path.join(self.path,c,"test.lst"),dtype=str)
            for m in models:

                if hint is not None:
                    if hint not in m:
                        continue

                d = {}
                d["class"] = c
                d["model"] = m
                d["scan_ply"] = os.path.join(self.path,d["class"],d["model"],'scan','pointcloud.ply')
                d["scan"] = os.path.join(self.path,d["class"],d["model"],'scan','pointcloud.npz')

                d["poisson"] = os.path.join(self.path,d["class"],d["model"],"poisson.ply")

                # d["scan_ply"] = glob(os.path.join(self.path,d["class"],m,'*.ply'))[0]

                d["occ"] = os.path.join(self.path,d["class"],d["model"],"eval","points.npz")
                d["pointcloud"] = os.path.join(self.path,d["class"],d["model"],"eval","pointcloud.npz")

                d["pointcloud_ply"] = os.path.join(self.path,d["class"],d["model"],"pointcloud.ply")
                d["mesh"] = os.path.join(self.path,d["class"],d["model"],"mesh_unit.off")
                d["planes"] = os.path.join(self.path,d["class"],d["model"],"planes","planes.vg")


                # d["ksr"] = {}
                # d["ksr"]["surface"] = os.path.join(self.path,d["class"],m,"ksr",'{}',"surface.off").format(ksr_k)
                # d["ksr"]["partition"] = os.path.join(self.path,d["class"],m,"ksr",'{}',"partition.kgraph").format(ksr_k)
                #
                # d["abspy"] = {}
                # d["abspy"]["surface"] = os.path.join(self.path,d["class"],m,"abspy",'{}',"surface.off").format(abspy_k)
                # d["abspy"]["partition"] = os.path.join(self.path,d["class"],m,"abspy",'{}',"partition.obj").format(abspy_k)

                self.model_dicts.append(d)

        return self.model_dicts



    def makeDummyEval(self):

        for m in self.model_dicts:

            os.makedirs(os.path.join(self.path,m["class"],m["model"],"eval"),exist_ok=True)

            points = np.zeros(shape=(1,3))
            np.savez(os.path.join(self.path,m["class"],m["model"],"eval","pointcloud.npz"),points=points)
            np.savez(os.path.join(self.path,m["class"],m["model"],"eval","points.npz"),points=points)


    def makeSubset(self,n_points):

        """"make a subset NPZ and PLY file with sensor oriented normals"""


        model_names = []
        for m in self.model_dicts:

            print(m["model"])

            os.makedirs(os.path.join(self.path, str(n_points), m["model"],"scan"), exist_ok=True)

            data = np.load(os.path.join(self.path,m["class"],m["model"],"scan","pointcloud.npz"))

            points = data["points"]
            if 'sensor_pos' in data.keys():
                sensors = data["sensor_pos"]
            else:
                sensors = data["sensor_position"]


            rnd = np.random.default_rng()
            indices = rnd.choice(points.shape[0], size=n_points, replace=False)
            # indices = np.random.randint(points.shape[0], size=n_points)
            points = points[indices,:]
            sensors = sensors[indices,:]
            sensor_vec = sensors - points


            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals)
            ip = np.einsum('ij,ij->i',normals, sensor_vec)

            normals[ip<0] = -normals[ip<0]
            normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
            pcd.normals = o3d.utility.Vector3dVector(normals)




            # pcd.normals = o3d.utility.Vector3dVector(sensors - points)

            if 'colors' in data.keys():
                colors = data["colors"][indices,:]
                pcd.colors = o3d.utility.Vector3dVector(colors)

            np.savez(os.path.join(self.path, str(n_points), m["model"], "scan", "pointcloud.npz"),points=points,colors=colors,normals=normals,sensor_position=sensors)

            o3d.io.write_point_cloud(os.path.join(self.path, str(n_points), m["model"], "scan", "pointcloud.ply"), pcd)

            model_names.append(m["model"])

        with open(os.path.join(self.path,str(n_points),"test.lst"), "w") as f:
            for m in model_names:
                f.write(m+"\n")


    def makeFulll(self):

        """"make a PLY file with sensor oriented normals"""
        model_names = []
        for m in self.model_dicts:

            print(m["model"])

            data = np.load(os.path.join(self.path, m["class"], m["model"], "scan", "pointcloud.npz"))

            points = data["points"]
            if 'sensor_pos' in data.keys():
                sensors = data["sensor_pos"]
            else:
                sensors = data["sensor_position"]

            sensor_vec = sensors - points

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals)
            ip = np.einsum('ij,ij->i', normals, sensor_vec)

            normals[ip < 0] = -normals[ip < 0]
            normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
            pcd.normals = o3d.utility.Vector3dVector(normals)

            # pcd.normals = o3d.utility.Vector3dVector(sensors - points)

            if 'colors' in data.keys():
                colors = data["colors"]
                pcd.colors = o3d.utility.Vector3dVector(colors)

            o3d.io.write_point_cloud(os.path.join(self.path, "full", m["model"], "scan", "pointcloud.ply"), pcd)
            model_names.append(m["model"])


    def makePoisson(self,depth="6"):

        boundary = 2

        for m in tqdm(self.model_dicts, ncols=50):
            # try:
            # os.makedirs(os.path.join(outpath, m["class"]), exist_ok=True)
            command = [self.POISSON_EXE,
                       "--in", m["scan_ply"],
                       "--out", m["poisson"],
                       "--depth", str(depth),
                       "--bType", str(boundary)]
            print("run command: ", *command)
            p = subprocess.Popen(command)
            # p = subprocess.Popen(command, shell=False,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p.wait()


    def dgnnFeats(self):

        for m in self.model_dicts:
            # extract features from mpu
            os.makedirs(os.path.join(self.path,m["class"],m["model"],"dgnn"),exist_ok=True)
            command = [self.mesh_tools_dir + "/feat",
                       "-w", str(os.path.join(self.path,m["class"],m["model"])),
                       "-i", str(os.path.join("scan","pointcloud")),
                       "-o", str(os.path.join("dgnn","0")),
                       "-s", "npz",
                       "-e", ""]
            print("run command: ", command)
            p = subprocess.Popen(command)
            p.wait()

    def dgnnSurfaceFromPrediction(self):

        for m in self.model_dicts:
            command = [self.mesh_tools_dir + "/occ2mesh",
                       "-i", str(os.path.join(self.path,m["class"],m["model"],"scan","pointcloud")),
                       "-o", str(os.path.join(self.outpath,"dgnn","shapenet3000",m["class"],m["model"])),
                       "-p", str(os.path.join(self.outpath,"dgnn","prediction",m["class"],m["model"])),
                       "--gco", "cc-1.0",
                       "-s", "npz",
                       "-e", "i"]
            print("run command: ", command)
            p = subprocess.Popen(command)
            p.wait()





if __name__ == '__main__':

    ds = Real(classes="full")

    ds.getModels()

    # ds.makePoisson()

    ds.makeFulll()

    # ds.makeSubset(50000)

    # ds.makeDummyEval()
    #
    # ds.dgnnFeats()

    # ds.dgnnSurfaceFromPrediction()


