import os, sys, subprocess, pathlib
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import trimesh
import open3d as o3d
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libmesh import check_mesh_contains
from pathlib import Path



with open(os.path.join(pathlib.Path(__file__).parent,"DATASETDIR.txt"),'r') as f:
    DATASETDIR=f.readline()

DEBUG = 1
class Berger:

    def __init__(self,path=os.path.join(DATASETDIR,"reconbench"),
                 classes=[],
                 mesh_tools_dir="/home/raphael/cpp/mesh-tools/build/release"
                 ):
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

    def getModels(self,scan_conf=["0","1","2","3","4"],hint=None):


        self.scan_conf = scan_conf if isinstance(scan_conf, list) else [scan_conf]

        for s in self.scan_conf:
            for c in self.classes:

                if hint is not None:
                    if hint not in c:
                        continue

                d = {}
                d["class"] = s
                d["model"] = c
                d["scan_conf"] = s
                d["path"] = self.path
                if "mvs" in s:
                    d["scan"] = os.path.join(self.path,"scan",c,s[3:]+".npz")
                    d["scan_ply"] = os.path.join(self.path,"scan",c,s[3:]+".ply")
                else:
                    d["scan"] = os.path.join(self.path,"scan","{}_{}.npz".format(c,s))
                    d["scan_ply"] = os.path.join(self.path,"scan",c,s+".ply")

                d["convex_hull"] = os.path.join(self.path,"p2m","convex_hull",s,c+".obj")
                d["poisson_6"] = os.path.join(self.path,"p2m","poisson",s,c+".ply")

                d["occ"] = os.path.join(self.path,"eval",c,"points.npz")
                d["pointcloud"] = os.path.join(self.path,"eval",c,"pointcloud.npz")

                d["pointcloud_ply"] = os.path.join(self.path,"scan_ply","with_normals",c+"_"+s+".ply")

                d["mesh"] = os.path.join(self.path,"mesh",c+"_light.off")

                d["planes"] = os.path.join(self.path,"planes",c,s,"planes.npz")
                d["ransac"] = os.path.join(self.path,"ransac",c,s,"planes.npz")

                # d["partition"] = os.path.join(self.path,'{}','{}',c,s,"partition.ply")
                # d["compact_surface"] = os.path.join(self.path,'{}','{}',c,s,"surface.off")
                d["ksr"] = {}
                d["ksr"]["surface"] = os.path.join(self.path,"ksr",'{}',c,s,"surface.off")
                d["ksr"]["partition"] = os.path.join(self.path,"ksr",'{}',c,s,"partition.ply")

                d["abspy"] = {}
                d["abspy"]["surface"] = os.path.join(self.path,"abspy",'{}',c,s,"surface.off")
                d["abspy"]["partition"] = os.path.join(self.path,"abspy",'{}',c,s,"partition.ply")

                d["coacd"] = {}
                d["coacd"]["partition"] = os.path.join(self.path,"coacd",c,s,"in_cells.ply")
                d["coacd"]["surface"] = os.path.join(self.path,"coacd",c,s,"in_cells.ply")

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



    def scanBerger(self,args):
        import configparser

        # make the scans
        config = configparser.ConfigParser()
        conf_path = os.path.join(args.working_dir, "confs/bumps_" + str(args.conf) + ".cnf")
        print("Read scanning configuration from: " + conf_path)
        config.read(conf_path)
        infile = os.path.join(args.working_dir, "mesh.mpu")
        if (args.normal_type == 1):
            outfile = os.path.join(args.working_dir, "berger_normals.ply")
        elif (args.normal_type == 4):
            outfile = os.path.join(args.working_dir, "berger_sensors.ply")

        sensor_file = os.path.join(args.working_dir, "sensors")
        # pathdir = os.path.join(args.user_dir, args.reconbench_dir)
        pathdir = args.program_dir

        command = []

        command.append(pathdir + "/" + config.get("uniform", "exec_name"))
        command.append(pathdir)
        command.append(infile)
        command.append(outfile)
        command.append(sensor_file)

        # required
        command.append(config.get("uniform", "camera_res_x"))
        command.append(config.get("uniform", "camera_res_y"))
        command.append(config.get("uniform", "scan_res"))

        # optional

        if config.has_option("uniform", "min_range"):
            command.append("min_range")
            command.append(config.get("uniform", "min_range"))

        if config.has_option("uniform", "max_range"):
            command.append("max_range")
            command.append(config.get("uniform", "max_range"))

        if config.has_option("uniform", "num_stripes"):
            command.append("num_stripes")
            command.append(config.get("uniform", "num_stripes"))

        if config.has_option("uniform", "laser_fov"):
            command.append("laser_fov")
            command.append(config.get("uniform", "laser_fov"))

        if config.has_option("uniform", "peak_threshold"):
            command.append("peak_threshold")
            command.append(config.get("uniform", "peak_threshold"))

        if config.has_option("uniform", "std_threshold"):
            command.append("std_threshold")
            command.append(config.get("uniform", "std_threshold"))

        if config.has_option("uniform", "additive_noise"):
            command.append("additive_noise")
            command.append(config.get("uniform", "additive_noise"))

        if config.has_option("uniform", "outlier_percentage"):
            command.append("outlier_percentage")
            command.append(config.get("uniform", "outlier_percentage"))

        if config.has_option("uniform", "laser_smoother"):
            command.append("laser_smoother")
            command.append(config.get("uniform", "laser_smoother"))

        if config.has_option("uniform", "registration_error"):
            command.append("registration_error")
            command.append(config.get("uniform", "registration_error"))

        # if config.has_option("uniform", "normal_type"):
        #         command.append("normal_type")
        #         command.append(config.get("uniform", "normal_type"))
        command.append("normal_type")
        command.append(str(args.normal_type))

        if config.has_option("uniform", "pca_knn"):
            command.append("pca_knn")
            command.append(config.get("uniform", "pca_knn"))

        if config.has_option("uniform", "random_sample_rotation"):
            command.append("random_sample_rotation")
            command.append(config.get("uniform", "random_sample_rotation"))

        print(*command)
        subprocess.check_call(command)

        # ### usage:
        # args = Munch()
        # args.working_dir = "/home/adminlocal/PhD/data/benchmark/scan_example"
        # args.program_dir = "/home/adminlocal/PhD/cpp/reconbench-CMake/build/release"
        # # args.working_dir = "/mnt/raphael/scan_example"
        # # args.program_dir = "/home/raphael/cpp/reconbench-CMake/build/release"
        # args.conf = "5"
        # args.normal_type = 4
        # scanShape(args)




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

    def orientNormals(self):

        for m in tqdm(self.model_dicts, ncols=50):
            scan = np.load(m["scan"])
            pcd = o3d.geometry.PointCloud()

            # orient normals towards sensor
            points = scan["points"]

            pcd.points = o3d.utility.Vector3dVector(points)

            pcd.estimate_normals()

            normals = np.asarray(pcd.normals)
            sensor_vec = scan["sensor_position"] - points

            ip = np.einsum('ij,ij->i', normals, sensor_vec)
            normals[ip < 0] = -normals[ip < 0]

            normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
            pcd.normals = o3d.utility.Vector3dVector(normals)

            o3d.io.write_point_cloud(m["scan"][:-3] + "ply", pcd)

            a = 5

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


    def makeEval(self,n_points=100000,padding=0.1):

        print("Sample points on surface and in bounding box for evaluation...\n")


        if(len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)


        # loc = np.zeros(3)
        # scale = 75.0
        loc = np.ones(3)+5
        scale = 1.0
        padding = padding*scale


        for m in tqdm(self.model_dicts):

            dtype = np.float32


            mesh = trimesh.load(m["mesh"])

            # surface points
            points_surface, fid = mesh.sample(n_points,return_index=True)
            normals = mesh.face_normals[fid]

            points_surface = points_surface.astype(dtype)
            normals = normals.astype(dtype)

            os.makedirs(os.path.dirname(m["pointcloud"]),exist_ok=True)
            np.savez(m["pointcloud"], points=points_surface, normals=normals, loc=loc, scale=scale)

            if DEBUG:
                print('Writing points: %s' % m["pointcloud"])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_surface)
                pcd.normals = o3d.utility.Vector3dVector(normals)
                o3d.io.write_point_cloud(str(Path(m["pointcloud"]).with_suffix(".ply")), pcd)



            # IoU points

            n_points_uniform = int(n_points * 0.5)
            n_points_surface = n_points - n_points_uniform

            points_uniform = np.random.rand(n_points_uniform, 3)
            points_uniform = (scale+padding) * (points_uniform - 0.5)

            points_surface = mesh.sample(n_points_surface)
            points_surface += np.random.randn(n_points_surface, 3)
            points = np.concatenate([points_uniform, points_surface], axis=0)

            occupancies = check_mesh_contains(mesh, points)

            colors = np.zeros(shape=(n_points, 3)) + [0, 0, 1]
            colors[occupancies] = [1,0,0]



            points = points.astype(dtype)
            occupancies = np.packbits(occupancies)

            np.savez(m["occ"], points=points, occupancies=occupancies,loc=loc, scale=scale)

            if DEBUG:
                print('Writing points: %s' % m["occ"])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(str(Path(m["occ"]).with_suffix(".ply")), pcd)


if __name__ == '__main__':

    # model = "split_cube"
    # model = "slanted_cube"
    # model = "double_slanted_cube"
    models = ["cube","split_cube","double_split_cube","slanted_cube","double_slanted_cube","complex_cube"]
    ds = Berger(classes=models)
    ds.getModels(scan_conf="1")
    # ds.standardize()
    ds.makeEval(n_points=100000,padding=0.1)
