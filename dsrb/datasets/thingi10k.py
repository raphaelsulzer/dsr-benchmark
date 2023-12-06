import os
import shutil
import time
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import numpy as np
from datetime import datetime

from dsrb import DefaultDataset

class Thingi10kDataset(DefaultDataset):

    def __init__(self):
        super().__init__()
        self.path = os.path.join(self.path, "thingi10k")
        self.model_dicts = []


    def get_models(self,list=None,names=None):

        if list is None:
            models = os.listdir(self.path)
            models = np.array(models)
            models = np.sort(models)
            models = models.astype(str)
        else:
            models = np.genfromtxt(os.path.join(self.path, list), dtype=str)

        for m in models:

            if m[0] == "#":
                continue

            if not os.path.isdir(os.path.join(self.path,m)):
                continue

            if m == "results":
                continue

            if names is not None:
                if m not in names:
                    continue

            d = {}
            d["class"] = ""
            d["model"] = m
            d["path"] = os.path.join(self.path, m)

            d["pointcloud"] = os.path.join(self.path, m, "pointcloud", "pointcloud.npz")
            d["pointcloud_ply"] = os.path.join(self.path, m, "pointcloud", "pointcloud.ply")
            # pcd = o3d.io.read_point_cloud(d["pointcloud_ply"])
            # d["bb_diagonal"] = np.linalg.norm(pcd.get_min_bound() - pcd.get_max_bound())

            d["eval"] = dict()
            d["eval"]["occ"] = os.path.join(self.path, m, "eval", "points.npz")
            d["eval"]["pointcloud"] = os.path.join(self.path, m, "eval", "pointcloud.npz")

            d["mesh"] = os.path.join(self.path, m, "mesh.off")

            d["planes_vg"] = os.path.join(self.path, m, "planes", "planes.vg")
            d["planes_ply"] = os.path.join(self.path, m, "planes", "planes.ply")
            d["planes"] = os.path.join(self.path, m, "planes", "planes.npz")
            d["plane_params"] = os.path.join(self.path, m, "planes", "params.json")



            d["output"] = {}
            d["output"]["surface"] = os.path.join(self.path, m, "{}", "surface.ply")
            d["output"]["surface_simplified"] = os.path.join(self.path, m, "{}", "surface_simplified.obj")
            d["output"]["partition"] = os.path.join(self.path, m, "{}", "partition.ply")
            d["output"]["partition_pickle"] = os.path.join(self.path, m, "{}", "partition")
            d["output"]["in_cells"] = os.path.join(self.path, m, "{}", "in_cells.ply")
            d["output"]["settings"] = os.path.join(self.path, m, "{}", "settings.yaml")


            self.model_dicts.append(d)

        return self.model_dicts


    def clean(self):

        for model in self.model_dicts:

            if os.path.isdir(os.path.join(model["path"],"compod")):
                shutil.rmtree(os.path.join(model["path"],"compod"))
            a=5


    def stl2off(self):

        models = os.listdir(os.path.join(self.path,"stl"))
        for m in tqdm(models):

            try:
                mfile = os.path.join(self.path,"stl",m)
                name = m[:-4]
                mesh = o3d.io.read_triangle_mesh(mfile)
                os.makedirs(os.path.join(self.path,name),exist_ok=True)
                o3d.io.write_triangle_mesh(os.path.join(self.path,name,"mesh.off"),mesh)

                a=5

            except:
                # raise e
                pass

    def setup_lists(self):


        small = []
        medium = []
        large = []
        onethousand = []
        for m in tqdm(self.model_dicts[:1000]):


            data = np.load(m["planes"])
            num_planes = len(data["group_parameters"])
            if num_planes <=100:
                small.append(m["model"])
            elif num_planes <= 250:
                medium.append(m["model"])
            else:
                large.append(m["model"])

            onethousand.append(m["model"])

        np.savetxt(os.path.join(self.path,"small.lst"), small, fmt="%s")
        np.savetxt(os.path.join(self.path,"medium.lst"), medium, fmt="%s")
        np.savetxt(os.path.join(self.path,"large.lst"), large, fmt="%s")
        np.savetxt(os.path.join(self.path,"1000.lst"), onethousand, fmt="%s")

    def detect_planes(self):

        try:
            from pypsdr import psdr
        except:
            pass # for importing default dataset in blender

        time_dicts = []
        for model in tqdm(self.model_dicts):

            params = {'min_inliers': 10, 'epsilon': 0.008, 'normal_th': 0.85}

            try:

                td = {"model":model["model"]}
                psd = psdr(1)

                pcd = o3d.io.read_point_cloud(model["pointcloud_ply"])
                diag =  np.linalg.norm(pcd.get_min_bound() - pcd.get_max_bound())

                psd.load_points(model["pointcloud_ply"])

                params["epsilon"] = params["epsilon"]*diag
                t0 = time.time()
                psd.detect(**params)
                td["detect"] = time.time() - t0
                t0 = time.time()
                psd.refine(max_seconds=180)
                td["refine"] = time.time() - t0

                psd.save(model["planes"])
                psd.save(model["planes_ply"])

                nplanes = np.load(model["planes"])["group_parameters"].shape[0]

                td["n_planes"] = nplanes

                time_dicts.append(td)

            except Exception as e:
                # raise e
                print("Problem with plane extraction for model {}/{}".format(model["class"],model["model"]))


        df = pd.DataFrame(time_dicts)
        outfile = os.path.join(self.path,"results", "compod", datetime.now().strftime("%m-%d-%Y-%H%M-%S")+"_plane_detection", "{}.csv".format(str(params)))
        os.makedirs(os.path.dirname(outfile),exist_ok=True)
        df.to_csv(outfile)

    def rename(self):

        for model in self.model_dicts:

            rmesh = model["mesh"].replace("mesh","mesh_repaired")
            if os.path.isfile(rmesh):
                os.remove(model["mesh"])
                os.rename(rmesh,model["mesh"])




if __name__ == '__main__':

    ds = Thingi10kDataset()
    ds.get_models()
    # ds.rename()

    ds.clean()


    # ds.make_eval()
    # ds.sample()
    # ds.detect_planes()

