import os, sys, subprocess, shutil, trimesh

import pandas as pd
import numpy as np
from tqdm import tqdm
import open3d as o3d
from pathlib import Path
from glob import glob
from copy import deepcopy

from libmesh import check_mesh_contains
from dsrb import DefaultDataset, scan_settings

DEBUG = 1
class ScalabilityDataset(DefaultDataset):

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

    def get_models(self,scales=["40","70","100","250","500","2k","5k","10k"],names=None):

        if not isinstance(names,list) and names is not None:
            names = [names]

        if not isinstance(scales,list):
            scales = [scales]

        for s in scales:
            if not isinstance(s,str): s = str(s)
            for m in self.models:

                if names is not None:
                    if m not in names:
                        continue

                d = {}
                d["model"] = m

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

                # d["planes_vg"] = glob(os.path.join(self.path,"planes",m,"{}_{}".format(m,s),'*.vg'))[0]

                d["planes"] = os.path.join(self.path,"planes",m, s, m+".npz")
                d["planes_ply"] = os.path.join(self.path,"planes",m, s, m+".ply")
                d["plane_params"] = os.path.join(self.path,"planes",m, s, m+".json")

                d["output"] = {}
                d["output"]["surface"] = os.path.join(self.path, "{}", m, s,  "surface.ply")
                d["output"]["surface_simplified"] = os.path.join(self.path, "{}", m, s,  "surface_simplified.obj")
                d["output"]["partition"] = os.path.join(self.path, "{}", m, s,  "partition.ply")
                d["output"]["partition_pickle"] = os.path.join(self.path, "{}", m, s, "partition")
                d["output"]["in_cells"] = os.path.join(self.path, "{}", m, s, "in_cells.obj")
                d["output"]["settings"] = os.path.join(self.path, "{}", m, s,  "settings.yaml")

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



    def scan_berger(self,args):
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




    def estimate_normals(self, method='jet', neighborhood=30, orient=1):
        if (len(self.model_dicts) < 1):
            print("\nERROR: run get_models() first!")
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

    def orient_normals(self):

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

    def make_poisson(self, depth=10, boundary=2):

        for m in tqdm(self.model_dicts, ncols=50):
            # try:
            command = [self.poisson_exe,
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

    def detect_planes(self,params):

        from pypsdr import psdr
        import json

        for model in self.model_dicts:

            print("Process {}/{}".format(model["class"],model["model"]))

            par = deepcopy(params)
            ## get bounding box diagonal
            pcd = o3d.io.read_point_cloud(model["pointcloud_ply"])
            diag = np.linalg.norm(pcd.get_min_bound() - pcd.get_max_bound())
            par["epsilon"] *= diag
            ## extract planes
            psd = psdr(1)
            psd.load_points(np.asarray(pcd.points), np.asarray(pcd.normals))
            psd.detect(**par)
            # psd.set_discretization()
            psd.refine()
            psd.save(model["planes"])
            psd.save(model["planes_ply"])


            with open(model["plane_params"], "w") as outfile:
                # json_data refers to the above JSON
                json.dump(params, outfile)

    def plot_plane_params(self):

        df = pd.DataFrame()
        for model in self.model_dicts:
            if os.path.isfile(model["planes"]):
                data = np.load(model["planes"])
                df.loc[model["model"],model["class"]] = len(data["group_parameters"])
            else:
                df.loc[model["model"],model["class"]] = -99


        print(df)

    def move_compod_reconstructions(self):

        path = "/home/rsulzer/data/scalability_test/coacd"

        scales = os.listdir(path)

        for s in scales:

            models = os.listdir(os.path.join(path,s))

            for m in models:

                try:
                    infile = os.path.join(path,s,m,"in_cells.ply")
                    outfile = os.path.join("/home/rsulzer/data/scalability_test/coacd_c",m,s,"in_cells.ply")
                    os.makedirs(os.path.join("/home/rsulzer/data/scalability_test/coacd_c",m,s),exist_ok=True)
                    os.rename(infile,outfile)
                except:
                    pass











def make_coacd_comparison_dataset():

    ds = ScalabilityDataset()
    # ds.get_models(scales=["6k"],names=["temple"])
    ds.get_models(scales=["250"],names=["castle"])
    # ds.get_models()

    # 40
    # params = {"min_inliers": 4000, "epsilon": 0.02, "normal_th": 0.5}
    # Horse
    # params = {"min_inliers": 3800, "epsilon": 0.02, "normal_th": 0.6}
    # droid
    # params = {"min_inliers": 6500, "epsilon": 0.08, "normal_th": 0.48}
    # sazabi
    # params = {"min_inliers": 6500, "epsilon": 0.07, "normal_th": 0.49}
    # temple
    # params = {"min_inliers": 8000, "epsilon": 0.085, "normal_th": 0.48}
    # city
    # params = {"min_inliers": 10000, "epsilon": 0.9, "normal_th": 0.5}
    # castle
    params = {"min_inliers": 2000, "epsilon": 0.05, "normal_th": 0.6}

    # 70
    # params = {"min_inliers": 2200, "epsilon": 0.016, "normal_th": 0.62}
    # # "tarbosaurus",
    # params = {"min_inliers": 2800, "epsilon": 0.018, "normal_th": 0.6}
    # # "forbiddentower"
    # params = {"min_inliers": 3200, "epsilon": 0.025, "normal_th": 0.59}
    # # "droid"
    # params = {"min_inliers": 5000, "epsilon": 0.06, "normal_th": 0.57}
    # # "temple"
    # params = {"min_inliers": 5000, "epsilon": 0.06, "normal_th": 0.55}
    # # "sazabi"
    # # params = {"min_inliers": 4500, "epsilon": 0.065, "normal_th": 0.6}
    # # runninghorse
    # params = {"min_inliers": 2000, "epsilon": 0.014, "normal_th": 0.65}
    # castle
    # params = {"min_inliers": 900, "epsilon": 0.03, "normal_th": 0.55}



    # 100
    # params = {"min_inliers": 1000, "epsilon": 0.015, "normal_th": 0.6}
    # Horse
    # params = {"min_inliers": 900, "epsilon": 0.015, "normal_th": 0.65}
    # forbidden tower
    # params = {"min_inliers": 1500, "epsilon": 0.018, "normal_th": 0.6}
    # droid
    # params = {"min_inliers": 2500, "epsilon": 0.04, "normal_th": 0.55}
    # sazabi
    # params = {"min_inliers": 3000, "epsilon": 0.045, "normal_th": 0.55}
    # castle
    # params = {"min_inliers": 140, "epsilon": 0.04, "normal_th": 0.7}

    # 250
    # params = {"min_inliers": 300, "epsilon": 0.01, "normal_th": 0.75}
    # Lucy
    # params = {"min_inliers": 400, "epsilon": 0.012, "normal_th": 0.70}
    # forbidden tower
    # params = {"min_inliers": 800, "epsilon": 0.015, "normal_th": 0.65}
    # droid
    # params = {"min_inliers": 1500, "epsilon": 0.018, "normal_th": 0.6}
    # sazabi
    # params = {"min_inliers": 1200, "epsilon": 0.022, "normal_th": 0.6}
    # temple
    # params = {"min_inliers": 1400, "epsilon": 0.022, "normal_th": 0.65}
    # castle
    params = {"min_inliers": 80, "epsilon": 0.025, "normal_th": 0.8}


    # # 500
    # params = {"min_inliers": 100, "epsilon": 0.008, "normal_th": 0.80}
    # lucy
    # params = {"min_inliers": 150, "epsilon": 0.01, "normal_th": 0.80}
    # forbidden tower
    # params = {"min_inliers": 600, "epsilon": 0.012, "normal_th": 0.80}
    # droid
    # params = {"min_inliers": 600, "epsilon": 0.014, "normal_th": 0.75}
    # sazabi
    # params = {"min_inliers": 600, "epsilon": 0.012, "normal_th": 0.70}
    # temple
    # params = {"min_inliers": 900, "epsilon": 0.016, "normal_th": 0.7}
    # castle
    # params = {"min_inliers": 30, "epsilon": 0.008, "normal_th": 0.75}

    # 900
    # params = {"min_inliers": 6, "epsilon": 0.00108, "normal_th": 0.85}

    # # # 2000
    # params = {"min_inliers": 30, "epsilon": 0.004, "normal_th": 0.90}
    # armadillo, Horse, tarbosaurus
    # params = {"min_inliers": 25, "epsilon": 0.003, "normal_th": 0.90}
    # forbidden tower
    # params = {"min_inliers": 100, "epsilon": 0.006, "normal_th": 0.82}
    # droid
    # params = {"min_inliers": 60, "epsilon": 0.0045, "normal_th": 0.85}
    # sazabi
    # params = {"min_inliers": 120, "epsilon": 0.006, "normal_th": 0.85}
    # temple
    # params = {"min_inliers": 180, "epsilon": 0.0075, "normal_th": 0.8}
    # castle
    # params = {"min_inliers": 7, "epsilon": 0.001, "normal_th": 0.9}


    # 5000
    # params = {"min_inliers": 30, "epsilon": 0.0025, "normal_th": 0.85}
    # forbidden tower
    # params = {"min_inliers": 50, "epsilon": 0.0035, "normal_th": 0.90}
    # temple
    # params = {"min_inliers": 35, "epsilon": 0.0028, "normal_th": 0.90}
    # castle
    # params = {"min_inliers": 3, "epsilon": 0.0008, "normal_th": 0.95}

    # 6000


    # # # 10000
    # params = {"min_inliers": 8, "epsilon": 0.001, "normal_th": 0.96}
    # Horse, armadillo, tarbosaurus
    # params = {"min_inliers": 7, "epsilon": 0.0008, "normal_th": 0.96}
    # forbidden tower
    # params = {"min_inliers": 15, "epsilon": 0.0012, "normal_th": 0.9}
    # droid
    # params = {"min_inliers": 7, "epsilon": 0.0008, "normal_th": 0.96}
    # sazabi
    # params = {"min_inliers": 12, "epsilon": 0.0018, "normal_th": 0.92}
    # temple
    # params = {"min_inliers": 12, "epsilon": 0.0018, "normal_th": 0.92}
    # city
    # params = {"min_inliers": 12, "epsilon": 0.0018, "normal_th": 0.92}

    # 20k
    # city
    # params = {"min_inliers": 10, "epsilon": 0.0005, "normal_th": 0.95}

    ds.detect_planes(params)
    ds.plot_plane_params()






if __name__ == '__main__':


    ds = ScalabilityDataset()

    ds.get_models(scales="40",names='castle')
    #
    # ds.sample(2000000)

    # make_coacd_comparison_dataset()