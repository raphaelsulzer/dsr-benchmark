import os, sys, subprocess, trimesh
from libmesh import check_mesh_contains
from tqdm import tqdm
import numpy as np
from glob import glob
from pathlib import Path
from .default_dataset import DefaultDataset
import open3d as o3d


class RobustDataset(DefaultDataset):

    def __init__(self,path="/home/rsulzer/data/RobustLowPolyDataSet/Thingi10k", **kwargs):
        super().__init__(path, **kwargs)

        self.path = path
        self.model_dicts = []


    def get_models(self, id=None, reduce=None, params=""):

        model_dicts = []
        for i in range(100):

            d = dict()
            d["mesh"] = os.path.join(self.path, "input_repaired", "{}.off".format(i))
            if not os.path.isfile(d["mesh"]):
                continue
            d["class"] = ""
            d["model"] = str(i)
            d["eval"] = dict()
            d["eval"]["pointcloud"] = os.path.join(self.path, "eval", str(i), "pointcloud.npz")
            d["eval"]["occ"] = os.path.join(self.path, "eval", str(i), "points.npz")
            d["pointcloud"] = os.path.join(self.path, "pointcloud", "{}.ply".format(i))

            d["planes"] = os.path.join(self.path,"output","{"+params+"}","{}","planes","{}.npz".format(i))
            d["planes_ply"] = os.path.join(self.path,"output","{"+params+"}","{}","planes","{}.ply".format(i))

            d["output"] = {}
            d["output"]["surface"] = os.path.join(self.path,"output", "{"+params+"}", "{}", "surface", "{}.obj".format(i))
            d["output"]["surface_simplified"] = os.path.join(self.path, "output","{"+params+"}", "{}", "surface_simplified", "{}.obj".format(i))
            d["output"]["partition"] = os.path.join(self.path, "output","{"+params+"}", "{}", "partition", str(i) ,"partition.ply")
            d["output"]["partition_pickle"] = os.path.join(self.path, "output","{"+params+"}", "{}", "partition", str(i))
            d["output"]["in_cells"] = os.path.join(self.path, "output", "{"+params+"}", "{}", "in_cells", "{}.ply".format(i))
            d["output"]["settings"] = os.path.join(self.path, "output", "{"+params+"}", "{}", "settings", "{}.yaml".format(i))

            model_dicts.append(d)

        if id is not None:
            for model in model_dicts:
                if str(id) == model["model"]:
                    return [model]
                else:
                    continue
                print("Model {} not found".format(id))
                return []

        if reduce is not None:
            model_dicts = model_dicts[:reduce]

        self.model_dicts = model_dicts
        return model_dicts

    def input_off2obj(self):

        for m in self.model_dicts:
            mesh = o3d.io.read_triangle_mesh(m["mesh"])
            o3d.io.write_triangle_mesh(m["mesh"][:-4] + ".obj", mesh)