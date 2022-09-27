import os, sys, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
import numpy as np
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from libmesh import check_mesh_contains
from tqdm import tqdm

class Berger:

    def __init__(self,path="/mnt/raphael/reconbench",
                 classes=[],
                 mesh_tools_dir="/home/raphael/cpp/mesh-tools/build/release"
                 ):

        self.path = path
        self.classes = classes
        self.model_dicts = []
        self.mesh_tools_dir = mesh_tools_dir

        if not classes:
            with open(os.path.join(self.path, "classes.lst"), 'r') as f:
                categories = f.read().split('\n')
            if '' in categories:
                categories.remove('')
            self.classes = categories

    def getModels(self,scan_conf=["0","1","2","3","4"],reduce=None):

        self.scan_conf = [scan_conf] if isinstance(scan_conf, str) else scan_conf


        for s in self.scan_conf:
            for c in self.classes:

                    d = {}
                    d["class"] = s
                    d["model"] = c
                    d["scan_conf"] = s
                    if "mvs" in s:
                        d["scan"] = os.path.join(self.path,"scan",c,s[3:]+".npz")
                        d["scan_ply"] = os.path.join(self.path,"scan",c,s[3:]+".ply")
                    else:
                        d["scan"] = os.path.join(self.path,"scan_berger_1",c,s+".npz")
                        d["scan_ply"] = os.path.join(self.path,"scan_berger_1",c,s+".ply")

                    d["convex_hull"] = os.path.join(self.path,"p2m","convex_hull",s,c+".obj")
                    d["poisson_6"] = os.path.join(self.path,"p2m","poisson",s,c+".ply")

                    d["occ"] = os.path.join(self.path,"eval",c,"points.npz")
                    d["pointcloud"] = os.path.join(self.path,"eval",c,"pointcloud.npz")

                    d["mesh"] = os.path.join(self.path,"mesh","1",c+".off")
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

    # def sample(self,n_points=100000):
    #
    #
    #     if(len(self.model_dicts) < 1):
    #         print("\nERROR: run getModels() first!")
    #         sys.exit(1)
    #
    #
    #     loc = np.zeros(3)
    #     scale = 1.0
    #
    #
    #     for m in self.model_dicts:
    #
    #         mesh = trimesh.load(m["mesh"])
    #
    #         # surface points
    #         points_surface, fid = mesh.sample(n_points,return_index=True)
    #         normals = mesh.face_normals[fid]
    #
    #         filename = os.path.join(self.path,"eval",m["class"],"pointcloud.npz")
    #         print('Writing points: %s' % filename)
    #         np.savez(filename, points=points_surface, normals=normals, loc=loc, scale=scale)
    #
    #
    #
    #         # IoU points
    #
    #         n_points_uniform = int(n_points * 0.5)
    #         n_points_surface = n_points - n_points_uniform
    #
    #         boxsize = 1
    #         points_uniform = np.random.rand(n_points_uniform, 3)
    #         points_uniform = boxsize * (points_uniform - 0.5)
    #         points_surface = mesh.sample(n_points_surface)
    #         points_surface += 0.05 * np.random.randn(n_points_surface, 3)
    #         points = np.concatenate([points_uniform, points_surface], axis=0)
    #
    #         occupancies = check_mesh_contains(mesh, points)
    #
    #         dtype = np.float16
    #         points = points.astype(dtype)
    #         occupancies = np.packbits(occupancies)
    #
    #         filename = os.path.join(self.path,"eval",m["class"],"points.npz")
    #         print('Writing points: %s' % filename)
    #         np.savez(filename, points=points, occupancies=occupancies,loc=loc, scale=scale)






