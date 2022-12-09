import os, sys, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import numpy as np
from glob import glob
import trimesh

class KSR42:

    def __init__(self,path="/home/rsulzer/data/KSR42_dataset",
                 classes=[], mesh_tools_dir="/home/rsulzer/cpp/mesh-tools/build/release"):

        self.path = path
        self.classes = classes if isinstance(classes,list) else [classes]

        self.model_dicts = []
        self.mesh_tools_dir = mesh_tools_dir

        if not classes:
            with open(os.path.join(self.path, "classes.lst"), 'r') as f:
                categories = f.read().split('\n')
            if '' in categories:
                categories.remove('')
            self.classes = categories

    def getModels(self,ksr_k=1,abspy_k=1):

        for c in self.classes:

            models = np.genfromtxt(os.path.join(self.path,c,"models.lst"),dtype=str)
            for m in models:

                d = {}
                d["class"] = c
                d["model"] = m
                d["scan_ply"] = glob(os.path.join(self.path,c,m,'*.ply'))[0]

                d["occ"] = os.path.join(self.path,"eval",c,"points.npz")
                d["pointcloud"] = os.path.join(self.path,"eval",c,"pointcloud.npz")
                d["mesh"] = os.path.join(self.path,c,m,"mesh.ply")
                d["planes"] = os.path.join(self.path,c,m,"planes.vg")

                d["ksr"] = {}
                d["ksr"]["surface"] = os.path.join(self.path,c,m,"ksr",'{}',"surface.off").format(ksr_k)
                d["ksr"]["partition"] = os.path.join(self.path,c,m,"ksr",'{}',"partition.kgraph").format(ksr_k)

                d["abspy"] = {}
                d["abspy"]["surface"] = os.path.join(self.path,c,m,"abspy",'{}',"surface.off").format(abspy_k)
                d["abspy"]["partition"] = os.path.join(self.path,c,m,"abspy",'{}',"partition.obj").format(abspy_k)

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

    def normalize(self):

        for m in self.model_dicts:



    def sample(self,n_points=100000):


        if(len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)


        loc = np.zeros(3)
        scale = 1.0


        for m in self.model_dicts:

            mesh = trimesh.load(m["mesh"])

            # surface points
            points_surface, fid = mesh.sample(n_points,return_index=True)
            normals = mesh.face_normals[fid]

            fpath = os.path.join(self.path,m["class"],m["model"],"eval","pointcloud.npz")
            filename = os.path.join(fpath,"pointcloud.npz")
            print('Writing points: %s' % filename)
            np.savez(filename, points=points_surface, normals=normals, loc=loc, scale=scale)



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

            dtype = np.float16
            points = points.astype(dtype)
            occupancies = np.packbits(occupancies)

            filename = os.path.join(self.path,"eval",m["class"],"points.npz")
            print('Writing points: %s' % filename)
            np.savez(filename, points=points, occupancies=occupancies,loc=loc, scale=scale)






