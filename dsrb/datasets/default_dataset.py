import os, sys, subprocess, trimesh, logging
try:
    import vedo
except:
    pass
from tqdm import tqdm
import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.colors as mcolors
from copy import deepcopy

from dsrb.scan_settings import scan_settings
from dsrb.logger import make_dsrb_logger



class DefaultDataset:

    def __init__(self,
                 logger=None,
                 verbosity=logging.INFO,
                 tqdm_enabled=True,
                 debug_export=False):

        if logger is not None:
            self.logger = logger
        else:
            self.logger = make_dsrb_logger("DATASET",level=verbosity)

        # data_dir_file = os.path.join(os.path.dirname(__file__),"..","..","CPP_DIR.txt")
        data_dir_file = "./DATA_DIR.txt"
        if not os.path.isfile(data_dir_file):
            self.logger.error("Could not find {}. Please put a file called DATA_DIR.txt with the path that points to your dataset directory in the root folder of this project.".format(data_dir_file))
            return 1
        with open(data_dir_file,"r") as f:
            data_dir = f.readline()

        # cpp_dir_file = os.path.join(os.path.dirname(__file__),"..","..","CPP_DIR.txt")
        cpp_dir_file = "./CPP_DIR.txt"
        if not os.path.isfile(cpp_dir_file):
            self.logger.error("Could not find {}. Please put a file called CPP_DIR.txt with the path that points to your cpp directory in the root folder of this project.".format(cpp_dir_file))
        else:
            with open(cpp_dir_file,"r") as f:
                cpp_dir = f.readline()


        self.path = data_dir
        self.model_dicts = []
        self.mesh_tools_dir = os.path.join(cpp_dir,"mesh-tools","build","release")
        self.poisson_exe = os.path.join(cpp_dir,"PoissonRecon","Bin","Linux","PoissonRecon")
        self.debug_export = debug_export
        self.tqdm_disabled = not tqdm_enabled

        self.splits = None



    def scan(self,scan_configuration="4"):
        self.logger.info("Scan ground truth surface...")

        scan_exe = os.path.join(self.mesh_tools_dir,"scan")
        if not os.path.isfile(scan_exe):
            self.logger.error("Could not find {}. Please install mesh-tools from here: https://github.com/raphaelsulzer/mesh-tools.git".format(scan_exe))
            return 1


        if(len(self.model_dicts) < 1):
            self.logger.error("run get_models() before scan().")
            return 1

        scan = scan_settings[scan_configuration]

        if self.splits is None:
            self.splits = ["default"]
            md = {}
            md["default"] = self.model_dicts
            self.model_dicts = md


        for s in self.splits:
            models = self.model_dicts[s]
            for model in tqdm(models, ncols=50, file=sys.stdout):

                try:
                    os.makedirs(os.path.dirname(model["scan"]),exist_ok=True)

                    command = [str(scan_exe),
                               "-i", str(model["mesh"]),
                               "-o", str(model["scan"][:-4]),
                               "--points", scan["points"],
                               "--noise", scan["noise"],
                               "--outliers", scan["outliers"],
                               "--cameras", scan["cameras"],
                               "--normal_neighborhood", "30",
                               "--normal_method", "jet",
                               "--normal_orientation", "1",
                               "--export", "all",
                               "-e","n"]
                    if self.logger.level < 20:
                        print(*command)
                        p = subprocess.Popen(command)
                    else:
                        p = subprocess.Popen(command,stdout=subprocess.DEVNULL)
                    p.wait()

                    os.makedirs(os.path.dirname(model["scan_ply"]),exist_ok=True)
                    os.rename(model["scan"].replace(".npz",".ply"),model["scan_ply"])
                except Exception as e:
                    # raise e
                    print(e)
                    print("Skipping {}/{}".format(model["class"], model["model"]))

    def sample(self,n_points=1000000,overwrite=True):

        for m in tqdm(self.model_dicts, disable=self.tqdm_disabled, file=sys.stdout):

            if os.path.isfile(m["pointcloud_ply"]) and not overwrite:
                continue
            try:
                os.makedirs(os.path.dirname(m["pointcloud"]), exist_ok=True)
                mesh = trimesh.load(m["mesh"])
                points, fid = mesh.sample(n_points, return_index=True)
                normals = mesh.face_normals[fid]

                # print('Writing points to: {}'.format(m["pointcloud"]))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.normals = o3d.utility.Vector3dVector(normals)
                o3d.io.write_point_cloud(m["pointcloud_ply"], pcd)

                np.savez_compressed(m["pointcloud"],points=points,normals=normals)

            except Exception as e:
                raise e
                print(e)
                print("Problem with {}".format(m["model"]))



    def make_pointcloud_ply(self,n_points=100000,std_noise=0.0):
        print("Writing pointclouds for reconstruction input...\n")

        for m in tqdm(self.model_dicts, file=sys.stdout):

            data = np.load(m["pointcloud"])

            ind = np.random.randint(0, data["points"].shape[0], size = (n_points,))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data["points"][ind])
            pcd.normals = o3d.utility.Vector3dVector(data["normals"][ind])
            o3d.io.write_point_cloud(m["pointcloud_ply"], pcd)



    def estim_normals(self, method='jet', neighborhood=30, orient=2):
        if (len(self.model_dicts) < 1):
            print("\nERROR: run get_models() first!")
            sys.exit(1)

        for m in tqdm(self.model_dicts, disable=self.tqdm_disabled, file=sys.stdout):
            try:
                command = [str(os.path.join(self.mesh_tools_dir, "normal")),
                           # "-w", str(self.path),
                           "-s", "ply",
                           "-i", str(m["pointcloud_ply"]),
                           "-o", str(m["pointcloud_ply"]),
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
        for m in tqdm(self.model_dicts, ncols=50, file=sys.stdout):
            try:
                os.remove(str(Path(m["mesh"]).with_suffix(".ply")))
            except:
                print(m["model"])
                # raise

    def make_poisson(self, depth=8, boundary=2, trim=None, keep_largest_component_only=False):

        for m in tqdm(self.model_dicts, ncols=50, file=sys.stdout):
            try:

                os.makedirs(os.path.dirname(m["mesh"]),exist_ok=True)
                command = [self.poisson_exe,
                           "--in", m["pointcloud_ply"],
                           "--out", m["mesh"][:-4]+".ply",
                           "--depth", str(depth),
                           "--density",
                           "--bType", str(boundary)]
                print("run command: ", *command)
                p = subprocess.Popen(command)
                # p = subprocess.Popen(command, shell=False,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                p.wait()


                if trim is not None:
                    command = [self.poisson_exe.replace("Linux/PoissonRecon","Linux/SurfaceTrimmer"),
                               "--in", m["mesh"][:-4]+".ply",
                               "--out", m["mesh"][:-4]+".ply",
                               "--trim", str(trim)]
                    print("run command: ", *command)
                    p = subprocess.Popen(command)
                    p.wait()


                if keep_largest_component_only:
                    self.keep_largest_components_only(m["mesh"][:-4]+".ply")

                # mesh = o3d.io.read_triangle_mesh(m["mesh"][:-4]+".ply")
                # o3d.io.write_triangle_mesh(m["mesh"],mesh)
                # os.remove(m["mesh"][:-4]+".ply")

            except Exception as e:
                print(e)



    def standardize(self,padding=0.1):
        """
        Standardize mesh so that maximum bounding box side length is 1.
        :return:
        """
        for m in tqdm(self.model_dicts,disable=self.tqdm_disabled, file=sys.stdout):

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


    def scale(self):
        """
        Scale mesh so that bounding box has a diagonal length of 1.
        :return:
        """

        for m in tqdm(self.model_dicts,disable=self.tqdm_disabled, file=sys.stdout):

            mesh = o3d.io.read_triangle_mesh(m["mesh"])

            bb = mesh.get_axis_aligned_bounding_box()
            mesh = mesh.translate(-bb.get_center(),relative=True)

            minb = mesh.get_min_bound()
            maxb = mesh.get_max_bound()
            diag = np.linalg.norm(minb-maxb)

            mesh = mesh.scale(1/diag,center=(0,0,0))

            outfile = os.path.join(self.path,"input_unit",os.path.split(m["mesh"])[1])
            os.makedirs(os.path.dirname(outfile),exist_ok=True)
            o3d.io.write_triangle_mesh(outfile,mesh)

    def make_eval(self,n_points=100000,unit=False,surface=True,occ=True):

        from libmesh import check_mesh_contains

        self.logger.info("Sample points on ground truth surface and in bounding box for evaluation...")

        if(len(self.model_dicts) < 1):
            self.logger.error("run get_models() first!")
            return 1
        for m in tqdm(self.model_dicts, file=sys.stdout):

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

                    np.savez_compressed(m["eval"]["pointcloud"], points=points_surface, normals=normals)

                    if self.debug_export:
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
                        points_surface = mesh.sample(n_points_surface)
                        points_surface += 0.05 * np.random.randn(n_points_surface, 3) * np.linalg.norm(min-max)
                    else:
                        points_uniform = np.random.rand(n_points_uniform, 3)
                        points_uniform = points_uniform - 0.5
                        points_surface = mesh.sample(n_points_surface)
                        points_surface += 0.05 * np.random.randn(n_points_surface, 3)

                    points = np.concatenate([points_uniform, points_surface], axis=0)

                    occupancies = check_mesh_contains(mesh, points)

                    colors = np.zeros(shape=(n_points, 3)) + [0, 0, 1]
                    colors[occupancies] = [1,0,0]

                    dtype = np.float16
                    points = points.astype(dtype)
                    occupancies = np.packbits(occupancies)

                    np.savez_compressed(m["eval"]["occ"], points=points, occupancies=occupancies)

                    if self.debug_export:
                        print('Writing points: %s' % m["eval"]["occ"])
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                        o3d.io.write_point_cloud(str(Path(m["eval"]["occ"]).with_suffix(".ply")), pcd)

            except Exception as e:
                # raise e
                print(e)
                print("Problem with {}".format(m["model"]))


    def color_mesh_by_component_trimesh(self, mesh, colors):

        meshes = []

        for me in mesh.split():
            # me.visual.face_colors = np.random.randint(100,255,size=3).tolist()
            col = colors[np.random.choice(len(colors))]
            col = mcolors.to_rgb(col)
            col = (np.array(col) * 255).astype(int)
            # me.visual.face_colors = col
            me.visual.vertex_colors = col
            meshes.append(me)

        return trimesh.util.concatenate(meshes), len(meshes)

    def color_mesh_by_component_vedo(self, mesh, colors):

        meshes = []
        ms = mesh.split()
        for me in ms:
            # me.visual.face_colors = np.random.randint(100,255,size=3).tolist()
            col = colors[np.random.choice(len(colors))]
            col = mcolors.to_rgb(col)
            col = (np.array(col) * 255).astype(int)
            # me.visual.face_colors = col
            me.pointcolors = col
            meshes.append(me)

        return vedo.merge(meshes), len(meshes)

    def recolor_in_cells(self,method,colors = "pale",backend="vedo"):

        if isinstance(colors,str):
            if colors == "saturated":
                colors = ["#ff0000", "#f4ec00", "#01ffff", "#ff7f00", "#0000ff", "#00ff01", "#6f00d8", "#ff00ff"]
            elif colors == "pale":
                colors = ["#CC99C9","#9EC1CF","#9EE09E","#FDFD97","#FEB144","#FF6663"]
            else:
                print("Don't know that color type")

        for model in tqdm(self.model_dicts, disable=self.tqdm_disabled, file=sys.stdout):

            infile = model[method]["in_cells"]

            if not os.path.isfile(infile):
                print("{} does not exist".format(infile))
                continue

            outfile = infile[:-4]+"_recolored"+infile[-4:]

            if backend == "vedo":
                mesh = vedo.load(infile)
                mesh, n = self.color_mesh_by_component_vedo(mesh,colors)
                vedo.io.write(mesh, outfile, binary=False)
            else:
                mesh = trimesh.load(infile)
                mesh, n = self.color_mesh_by_component_trimesh(mesh,colors)
                mesh.export(outfile)

    def recolor_planes(self, colors = "pale"):

        from pycompod import PlaneExporter

        # standard colors:
        # https://coolors.co/ff0000-f4ec00-01ffff-ff7f00-0000ff-00ff01-6f00d8-ff00ff
        if isinstance(colors,str):
            if colors == "saturated":
                colors = ["#ff0000", "#f4ec00", "#01ffff", "#ff7f00", "#0000ff", "#00ff01", "#6f00d8", "#ff00ff"]
            elif colors == "pale":
                colors = ["#CC99C9","#9EC1CF","#9EE09E","#FDFD97","#FEB144","#FF6663"]
            else:
                print("Don't know that color type")


        rgb_colors = []
        for col in colors:
            col = mcolors.to_rgb(col)
            col = (np.array(col) * 255).astype(int)
            rgb_colors.append(col)
        rgb_colors = np.array(rgb_colors)

        pe = PlaneExporter()
        for model in tqdm(self.model_dicts, file=sys.stdout):

            if not os.path.isfile(model["planes"]):
                print("{} not found".format(model["planes"]))
                continue

            data = np.load(model["planes"])
            data = dict(data)
            nplanes = len(data["group_parameters"])
            colors = np.random.choice(len(rgb_colors), nplanes)
            data["colors"] = rgb_colors[colors]
            plane_file = model["planes"][:-4]+"_recolored.ply"
            pt_file = model["planes"][:-4]+"_samples_recolored.ply"
            pe.save_points_and_planes_from_array(plane_filename=plane_file, point_filename=pt_file, planes_array=data)



    def detect_planes(self,params,max_seconds=-1,max_iterations=-1):

        try:
            from pypsdr import psdr # for importing the default_dataset in blender
        except:
            pass
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
            psd.save(model["planes"].replace(".npz","_detected.npz"))
            psd.save(model["planes_ply"].replace(".ply","_detected.ply"))
            # psd.set_discretization()
            psd.refine(max_seconds=max_seconds,max_iterations=max_iterations)
            psd.save(model["planes"])
            psd.save(model["planes_ply"])


            with open(model["plane_params"], "w") as outfile:
                # json_data refers to the above JSON
                json.dump(params, outfile)


    def ply2npz(self):

        for model in self.model_dicts:

            pc = o3d.io.read_point_cloud(model["pointcloud_ply"])
            np.savez_compressed(model["pointcloud"],points=np.asarray(pc.points),normals=np.asarray(pc.normals))



    def keep_largest_components_only(self,infile):

        import vedo.io

        mesh = vedo.io.load(infile)
        mesh = mesh.extract_largest_region()
        vedo.io.write(mesh,infile)
