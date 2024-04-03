import os, sys, subprocess, trimesh
from tqdm import tqdm
import numpy as np
from pathlib import Path
import open3d as o3d

from libmesh import check_mesh_contains
from dsrb import DefaultDataset
from dsrb.scan_settings import scan_settings

class KSR42Dataset(DefaultDataset):

    def __init__(self,classes=[]):
        super().__init__()
        self.path = os.path.join(self.path, "KSR42_dataset")

        self.classes = classes if isinstance(classes,list) else [classes]

        if not classes:
            with open(os.path.join(self.path, "classes.lst"), 'r') as f:
                categories = f.read().split('\n')
            if '' in categories:
                categories.remove('')
            self.classes = categories

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

                d["eval"] = dict()
                d["eval"]["occ"] = os.path.join(self.path, c, m, "eval","points.npz")
                d["eval"]["pointcloud"] = os.path.join(self.path,c, m, "eval","pointcloud.npz")
                d["eval"]["polygons"] = os.path.join(self.path,c, m, "eval","polygon_samples.npz")

                d["pointcloud"] = os.path.join(self.path, c, m, "pointcloud", "{}.npz".format(m))
                d["pointcloud_ply"] = os.path.join(self.path, c, m, "pointcloud", "{}.ply".format(m))
                d["mesh"] = os.path.join(self.path,c,m,"mesh.off")
                # d["planes"] = os.path.join(self.path,c,m,"planes","planes.npz")
                # d["planes"] = os.path.join(self.path,c,m,"planes_from_params","planes.npz")

                d["planes_vg"] = os.path.join(self.path, c, m, "planes", "{}.vg".format(m))
                d["planes_ply"] = os.path.join(self.path, c, m, "planes", "{}.ply".format(m))
                d["planes"] = os.path.join(self.path, c, m, "planes", "{}.npz".format(m))
                d["plane_params"] = os.path.join(self.path, c, m, "planes", "{}.json".format(m))

                d["output"] = {}
                d["output"]["surface"] = os.path.join(self.path, c, m, "{}", "surface.ply")
                d["output"]["surface_simplified"] = os.path.join(self.path, c, m, "{}", "surface_simplified.obj")
                d["output"]["partition"] = os.path.join(self.path, c, m, "{}", "partition.ply")
                d["output"]["partition_pickle"] = os.path.join(self.path, c, m, "{}", "partition")
                d["output"]["in_cells"] = os.path.join(self.path, c, m, "{}", "in_cells.obj")
                d["output"]["settings"] = os.path.join(self.path, c, m, "{}", "settings.yaml")

                self.model_dicts.append(d)

        if not len(self.model_dicts):
            print("ERROR: no models found!")
            return None
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

        for m in self.model_dicts:

            # inp = os.path.dirname(m["pointcloud"])
            # out = os.path.join(os.path.dirname(m["pointcloud"]),"..","eval_unit")
            # shutil.copytree(inp,out)
            infile = m["qem"]["partition"].format("None").replace("qem","coacd")
            outfile = m["coacd"]["partition"]
            os.rename(infile,outfile)
            os.rmdir(os.path.dirname(infile))




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

    ds = KSR42Dataset(classes="Cities")
    ds.get_models(names="huawei")

    ds.make_poisson(depth=11)
    # ds.sample(n_points=10000000)
    # ds.detect_planes({"min_inliers": 12, "epsilon": 0.0003, "normal_th": 0.95},max_seconds=3600)


    # ds.move()

    # ds.standardize()


    # ds.make_poisson(depth=11)
    #
    # ds.convert_mesh()
    #
    #
    # # ds.standardize()

    # ds.convert()

    # ds.make_eval(n_points=5000000,occ=True)
