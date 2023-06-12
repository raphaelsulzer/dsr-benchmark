import os, sys, subprocess, pathlib
import shutil

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
from glob import glob
from default_dataset import DATASET


DEBUG = 1
class Scalability(DATASET):

    def __init__(self, models=[], list="models.lst"):
        super().__init__()
        self.path = os.path.join(self.path, "scalability_test")

        self.model_dicts = []

        if not models:
            with open(os.path.join(self.path, list), 'r') as f:
                models = f.read().split('\n')
            if '' in models:
                models.remove('')
            self.models = models

    def getModels(self,scale=["40","100","250","500","2k","10k"],hint=None):


        self.scale = scale if isinstance(scale, list) else [scale]

        for s in self.scale:
            for m in self.models:

                if hint is not None:
                    if hint not in m:
                        continue

                d = {}
                d["model"] = m
                if s in ["40","100","250","500","1000"]:
                    d["n_sample_points"] = 150000
                elif s in ["2k"]:
                    d["n_sample_points"] = 250000
                elif s in ["10k","50k"]:
                    d["n_sample_points"] = 1000000
                else:
                    raise NotImplementedError

                d["scale"] = s
                d["class"] = s
                d["path"] = self.path


                d["eval"] = dict()
                d["eval"]["occ"] = os.path.join(self.path,"eval",m,"points.npz")
                d["eval"]["pointcloud"] = os.path.join(self.path,"eval",m,"pointcloud.npz")
                d["eval"]["polygons"] = os.path.join(self.path,"eval",m,"polygon_samples.npz")

                # d["pointcloud_ply"] = os.path.join(self.path,"scan_ply","with_normals",c+"_"+s+".ply")
                d["pointcloud_ply"] = os.path.join(self.path,"input_pointcloud",m+".ply")
                d["pointcloud"] = os.path.join(self.path,"input_pointcloud",m+".npz")

                d["mesh"] = os.path.join(self.path,"mesh",m+".off")

                d["planes_vg"] = glob(os.path.join(self.path,"planes",m,"{}_{}".format(m,s),'*.vg'))[0]
                d["planes"] = str(Path(d["planes_vg"]).with_suffix('.npz'))

                # d["plane_params"] = os.path.join(self.path,"planes",s,m,"params.txt")
                # d["ransac"] = os.path.join(self.path,"ransac",s,m,"planes.npz")

                # d["partition"] = os.path.join(self.path,'{}','{}',c,s,"partition.ply")
                # d["compact_surface"] = os.path.join(self.path,'{}','{}',c,s,"surface.off")
                d["ksr"] = {}
                d["ksr"]["surface"] = os.path.join(self.path,"ksr",'{}','{}',s,m,"surface.off")
                d["ksr"]["partition"] = os.path.join(self.path,"ksr",'{}','{}',s,m,"partition.ply")

                d["abspy"] = {}
                d["abspy"]["surface"] = os.path.join(self.path,"abspy",'{}','{}',s,m,"surface.off")
                d["abspy"]["partition"] = os.path.join(self.path,"abspy",'{}','{}',s,m,"partition.ply")

                d["coacd"] = {}
                d["coacd"]["partition"] = os.path.join(self.path,"coacd",s,m,"in_cells.ply")
                d["coacd"]["surface"] = os.path.join(self.path,"coacd",s,m,"in_cells.ply")

                d["qem"] = {}
                d["qem"]["partition"] = os.path.join(self.path,"qem",'{}',s,m,"in_cells.ply")
                d["qem"]["surface"] = os.path.join(self.path,"qem",'{}',s,m,"surface.ply")

                self.model_dicts.append(d)

        return self.model_dicts

    def move(self):

        for m in self.model_dicts:
            try:
                dir = os.listdir(os.path.join(m["planes"]))[0]
                shutil.rmtree(os.path.join(m["planes"], dir))
                for file in glob(os.path.join(m["planes"],dir,'*')):
                    fname = os.path.split(file)[-1]
                    # os.rename(file,os.path.join(m["planes"],fname))
            except Exception as e:
                print(e)

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
                       "--in", m["pointcloud_ply"],
                       "--out", m["mesh"],
                       "--depth", str(depth),
                       "--bType", str(boundary)]
            print("run command: ", *command)
            p = subprocess.Popen(command)
            # p = subprocess.Popen(command, shell=False,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p.wait()



    def convert_pc(self):

        for m in tqdm(self.model_dicts):
            pcd = o3d.io.read_point_cloud(m["pointcloud_ply"])
            np.savez(m["pointcloud"], points=np.asarray(pcd.points), normals=np.asarray(pcd.normals))

    def convert_mesh(self):

        for m in tqdm(self.model_dicts):
            plymesh = str(m["mesh"]+".ply")
            mesh = o3d.io.read_triangle_mesh(plymesh)
            o3d.io.write_triangle_mesh(m["mesh"],mesh)


    def make_eval(self,n_points=100000,unit=False,surface=True,occ=True):

        print("Sample points on surface and in bounding box for evaluation...\n")


        if(len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)





        for m in tqdm(self.model_dicts):

            try:

                # if not unit:
                #     m["mesh"] = m["ori_mesh"]

                mesh = trimesh.load(m["mesh"])
                fpath = os.path.dirname(m["eval"]["pointcloud"])
                os.makedirs(fpath, exist_ok=True)

                if surface:

                    # surface points
                    points_surface, fid = mesh.sample(n_points,return_index=True)
                    normals = mesh.face_normals[fid]

                    np.savez(m["eval"]["pointcloud"], points=points_surface, normals=normals)

                    if DEBUG:
                        print('Writing points: %s' % m["eval"]["pointcloud"])
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points_surface)
                        pcd.normals = o3d.utility.Vector3dVector(normals)
                        o3d.io.write_point_cloud(str(Path(m["eval"]["pointcloud"]).with_suffix(".ply")), pcd)


                # IoU points
                if occ:

                    n_points_uniform = int(n_points * 0.5)
                    n_points_surface = n_points - n_points_uniform

                    if not unit:
                        o3dmesh = o3d.io.read_triangle_mesh(m["mesh"])
                        o3dmesh = o3dmesh.scale(1.1,o3dmesh.get_center())
                        min=o3dmesh.get_min_bound()
                        max=o3dmesh.get_max_bound()
                        points_uniform = np.random.uniform(low=min,high=max,size=(n_points_uniform,3))
                        bb_diag = np.linalg.norm(min-max)
                    else:
                        bb_diag = sqrt(3)
                        points_uniform = np.random.rand(n_points_uniform, 3)
                        points_uniform = points_uniform - 0.5



                    points_surface = mesh.sample(n_points_surface)
                    points_surface += 0.05 * bb_diag * np.random.randn(n_points_surface, 3)
                    points = np.concatenate([points_uniform, points_surface], axis=0)

                    occupancies = check_mesh_contains(mesh, points)

                    colors = np.zeros(shape=(n_points, 3)) + [0, 0, 1]
                    colors[occupancies] = [1,0,0]



                    dtype = np.float16
                    points = points.astype(dtype)
                    occupancies = np.packbits(occupancies)

                    np.savez(m["eval"]["occ"], points=points, occupancies=occupancies)

                    if DEBUG:
                        print('Writing points: %s' % m["eval"]["occ"])
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                        o3d.io.write_point_cloud(str(Path(m["eval"]["occ"]).with_suffix(".ply")), pcd)


            except Exception as e:
                print(e)
                print("Problem with {}".format(m["model"]))


if __name__ == '__main__':

    # model = "split_cube"
    # model = "slanted_cube"
    # model = "double_slanted_cube"
    ds = Scalability()
    ds.getModels(scale=['40'])
    # ds.convert_pc()
    #
    # ds.makePoisson(depth=10)

    # ds.standardize()
    ds.make_eval(n_points=100000)

    # ds.move()
