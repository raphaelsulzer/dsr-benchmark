import os, sys, subprocess, trimesh, shutil, json, time
from datetime import datetime

import pandas as pd
from scan_settings import scan_settings
from libmesh import check_mesh_contains
from tqdm import tqdm
import numpy as np
from glob import glob
from pathlib import Path
from default_dataset import DefaultDataset
import open3d as o3d
from pypsdr import psdr

DEBUG = 1



class KSR42Dataset_ori(DefaultDataset):

    def __init__(self,classes=[]):
        super().__init__()
        self.path = os.path.join(self.path, "KSR42_original")

        self.classes = classes if isinstance(classes,list) else [classes]

        if not classes:
            with open(os.path.join(self.path, "classes.lst"), 'r') as f:
                categories = f.read().split('\n')
            if '' in categories:
                categories.remove('')
            self.classes = categories

    def setup(self):

        ## make model.lst
        # for c in self.classes:
        #
        #     cpath = os.path.join(self.path,c)
        #     models = os.listdir(cpath)
        #     np.savetxt(os.path.join(cpath,'models.lst'), models, fmt="%s")

        # ## rename input point cloud
        # for model in self.model_dicts:
        #
        #     files = glob(model["path"]+"/*.vg")
        #
        #     if not len(files) == 1:
        #         print(model["path"])
        #         continue
        #     else:
        #         # pcd=o3d.io.read_point_cloud(files[0])
        #         #
        #         # os.makedirs(os.path.join(model["path"],"pointcloud"))
        #         #
        #         # if not pcd.has_normals():
        #         #     o3d.geometry.estimate_normals(pcd)
        #         #
        #         # o3d.io.write_point_cloud(os.path.join(model["path"],"pointcloud","pointcloud.ply"),pcd)
        #         #
        #         # points = np.asarray(pcd.points)
        #         # normals = np.asarray(pcd.normals)
        #         #
        #         # np.savez(os.path.join(model["path"],"pointcloud","pointcloud.npz"),points=points,normals=normals)
        #
        #         os.makedirs(os.path.join(model["path"],"planes"))
        #         shutil.copyfile(files[0], os.path.join(os.path.join(model["path"],"planes","planes.vg")))

        ## setup plane param file
        for model in self.model_dicts:


            path = os.path.join("/root/Downloads/Kinetic-Partition-3D-Benchmark/",model["class"],model["model"],"ours")
            files = glob(path+"/*_params.txt")

            if not len(files) == 1:
                print(model["path"])
                continue
            else:

                try:
                    params_dict = {}
                    with open(files[0], "r") as f:
                        params = f.readlines()

                    params_dict["min_inliers"] = int(params[3].split(":")[1])
                    params_dict["epsilon"] = float(params[4].split(":")[1])
                    if "Nearest neighbors" in params[5]:
                        params_dict["normal_th"] = float(params[6].split(":")[1])
                        params_dict["knn"] = int(params[5].split(":")[1])
                    else:
                        params_dict["normal_th"] = float(params[5].split(":")[1])
                        params_dict["knn"] = int(10)


                    pcd = o3d.io.read_point_cloud(model["pointcloud_ply"])
                    diag = np.linalg.norm(pcd.get_min_bound() - pcd.get_max_bound())

                    params_dict["epsilon"] /= diag


                    with open(os.path.join(model["plane_params"]), "w") as outfile:
                        # json_data refers to the above JSON
                        json.dump(params_dict, outfile)

                except:
                    print("problem with model {}/{}".format(model["class"],model["model"]))


    def detect_planes(self):

        for model in tqdm(self.model_dicts):

            try:

                pd = psdr(1)

                pd.load_points(model["pointcloud_ply"])

                with open(model["plane_params"],"r") as f:
                    params = json.load(f)
                params["epsilon"]= params["epsilon"]*model["bb_diagonal"]
                pd.detect(params["min_inliers"],params["epsilon"],params["normal_th"],params["knn"])
                pd.refine()
                pd.save(model["planes"])
                pd.save(model["planes_ply"])

            except:
                print("Problem with plane extraction for model {}/{}".format(model["class"],model["model"]))




    def get_models(self,list="models.lst",names=None):

        for c in self.classes:

            models = np.genfromtxt(os.path.join(self.path,c,list),dtype=str)

            for m in models:

                if m[0] == "#":
                    continue

                if names is not None:
                    if m not in names:
                        continue

                d = {}
                d["class"] = c
                d["model"] = m
                d["path"] = os.path.join(self.path,c,m)

                d["pointcloud"] = os.path.join(self.path,c,m,"pointcloud","pointcloud.npz")
                d["pointcloud_ply"] = os.path.join(self.path,c,m,"pointcloud","pointcloud.ply")
                pcd = o3d.io.read_point_cloud(d["pointcloud_ply"])
                d["bb_diagonal"] = np.linalg.norm(pcd.get_min_bound() - pcd.get_max_bound())


                d["eval"] = dict()
                d["eval"]["occ"] = None
                d["eval"]["pointcloud"] = os.path.join(self.path,c,m,"pointcloud","pointcloud.npz")

                d["mesh"] = os.path.join(self.path,c,m,"poisson","mesh.off")

                d["planes_vg"] = os.path.join(self.path,c,m,"planes","planes.vg")
                d["planes_ply"] = os.path.join(self.path,c,m,"planes","planes.ply")
                d["planes"] = os.path.join(self.path,c,m,"planes","planes.npz")
                d["plane_params"] = os.path.join(self.path,c,m,"planes","params.json")

                d["ksr"] = {}
                d["ksr"]["surface"] = os.path.join(self.path,c,m,"ksr",'{}','{}',"surface.off")
                d["ksr"]["partition"] = os.path.join(self.path,c,m,"ksr",'{}','{}',"partition.ply")

                d["abspy"] = {}
                d["abspy"]["surface"] = os.path.join(self.path,c,m,"abspy",'{}','{}',"surface.off")
                d["abspy"]["partition"] = os.path.join(self.path,c,m,"abspy",'{}','{}',"partition.ply")

                d["compod"] = {}
                d["compod"]["surface"] = os.path.join(self.path,c,m,"compod","surface.ply")
                d["compod"]["surface_simplified"] = os.path.join(self.path,c,m,"compod","surface_simplified.obj")
                d["compod"]["partition"] = os.path.join(self.path,c,m,"compod","partition.ply")
                d["compod"]["partition_pickle"] = os.path.join(self.path,c,m,"compod","partition")
                d["compod"]["in_cells"] = os.path.join(self.path,c,m,"compod","in_cells.ply")

                d["coacd"] = {}
                d["coacd"]["surface"] = os.path.join(self.path,c,m,"coacd","in_cells.ply")
                d["coacd"]["partition"] = os.path.join(self.path,c,m,"coacd","in_cells.ply")

                d["qem"] = {}
                d["qem"]["surface"] = os.path.join(self.path,c,m,"qem",'{}',"surface.off")
                d["qem"]["partition"] = os.path.join(self.path,c,m,"qem",'{}',"in_cells.ply")

                self.model_dicts.append(d)

        return self.model_dicts



    def scan(self,scan_setting="4",scanner_dir="/home/raphael/cpp/mesh-tools/build/release/scan",
             normal_method='jet', normal_neighborhood=30, normal_orient=1):

        if(len(self.model_dicts) < 1):
            print("\nERROR: run get_models() first!")
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

    def clean(self):

        for m in tqdm(self.model_dicts, ncols=50):


            try:

                os.remove(str(Path(m["mesh"]).with_suffix(".ply")))
            except:
                print(m["model"])
                # raise

    def make_poisson(self, depth=6, boundary=2):

        time_dicts = []
        for m in tqdm(self.model_dicts):
            # try:

            td = {"class":m["class"],"model":m["model"]}
            os.makedirs(os.path.dirname(m["mesh"]),exist_ok=True)
            command = [self.poisson_exe,
                       "--in", m["pointcloud_ply"],
                       "--out", m["mesh"][:-4],
                       "--depth", str(depth),
                       "--bType", str(boundary)]
            print("run command: ", *command)
            p = subprocess.Popen(command)
            # p = subprocess.Popen(command, shell=False,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            t0 = time.time()
            p.wait()
            td["poisson"] = time.time() - t0

            time_dicts.append(td)

            # poisson can only export to ply, change it to off
            mesh = o3d.io.read_triangle_mesh(m["mesh"][:-4]+".ply")
            o3d.io.write_triangle_mesh(m["mesh"],mesh)


        df = pd.DataFrame(time_dicts)
        outfile = os.path.join(self.path,"results", "compod", datetime.now().strftime("%m-%d-%Y-%H%M-%S")+"_poisson", "depth_{}.csv".format(depth))
        os.makedirs(os.path.dirname(outfile),exist_ok=True)
        df.to_csv(outfile)




    def standardize(self,padding=0.1):

        for m in tqdm(self.model_dicts):

            if os.path.isfile(os.path.splitext(m["mesh"])[0]+"_unit.off"):
               continue

            print("Standardize {} to {}".format(m["mesh"],os.path.splitext(m["mesh"])[0]+"_unit.off"))

            mesh = o3d.io.read_triangle_mesh(m["mesh"])
            center = mesh.get_axis_aligned_bounding_box().get_center()

            mesh = mesh.translate(-center)


            max_bound = np.vstack([np.abs(mesh.get_min_bound()), mesh.get_max_bound()])
            col_index = np.argmax(max_bound,axis=1)[1]
            scale = np.abs(mesh.get_min_bound()[col_index]) + np.abs(mesh.get_max_bound()[col_index])


            mesh = mesh.scale((100-padding) / scale, [0, 0, 0])

            o3d.io.write_triangle_mesh(os.path.splitext(m["mesh"])[0]+"_unit.off",mesh)



    def make_pointcloud_ply(self,n_points=100000,std_noise=0.0):

        print("Writing pointclouds for reconstruction input...\n")

        for m in tqdm(self.model_dicts):

            data = np.load(m["pointcloud"])

            ind = np.random.randint(0, data["points"].shape[0], size = (n_points,))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data["points"][ind])
            pcd.normals = o3d.utility.Vector3dVector(data["normals"][ind])
            o3d.io.write_point_cloud(m["pointcloud_ply"], pcd)


    def convert(self):

        for m in tqdm(self.model_dicts):
            pcd = o3d.io.read_point_cloud(str(Path(m["eval"]["pointcloud"]).with_suffix(".ply")))
            np.savez(str(Path(m["eval"]["pointcloud"]).with_suffix(".npz")), points=np.asarray(pcd.points), normals=np.asarray(pcd.normals))

    def convert_mesh(self):

        for m in tqdm(self.model_dicts):
            plymesh = str(Path(m["mesh"]).with_suffix(".ply"))
            mesh = o3d.io.read_triangle_mesh(plymesh)
            o3d.io.write_triangle_mesh(m["mesh"],mesh)

    def move(self):

        for model in self.model_dicts:

            os.makedirs(os.path.join(self.path,model["class"],model["model"],"compod"),exist_ok=True)

            file = os.path.join(self.path,model["class"],model["model"],"surface_simplified","{}_{}.obj".format(model["model"],model["class"]))
            if os.path.isfile(file):
                shutil.move(file,model["compod"]["surface_simplified"])
                shutil.rmtree(os.path.join(self.path,model["class"],model["model"],"surface_simplified"))

            file = os.path.join(self.path,model["class"],model["model"],"surface","{}_{}.ply".format(model["model"],model["class"]))
            if os.path.isfile(file):
                shutil.move(file,model["compod"]["surface"])
                shutil.rmtree(os.path.join(self.path,model["class"],model["model"],"surface"))


            file = os.path.join(self.path,model["class"],model["model"],"partition","partition.ply")
            if os.path.isfile(file):
                shutil.move(file,model["compod"]["partition"])
                # os.remove(os.path.join(self.path,model["class"],model["model"],"partition"))


            folder = os.path.join(self.path,model["class"],model["model"],"partition")
            if os.path.isdir(folder):
                shutil.move(folder,model["compod"]["partition_pickle"])
                # os.remove(os.path.join(self.path,model["class"],model["model"],"partition"))


            a=5



    def make_eval(self,n_points=100000,unit=False,surface=True,occ=True):

        print("Sample points on surface and in bounding box for evaluation...\n")


        if(len(self.model_dicts) < 1):
            print("\nERROR: run get_models() first!")
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
                        points_uniform = np.random.rand(n_points_uniform, 3)
                        points_uniform = points_uniform - 0.5
                        bb_diag = sqrt(3)



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

    ds = KSR42Dataset_ori()
    ds.get_models()
    ds.make_poisson(depth=9)
    # ds.setup()

    # ds.detect_planes()
