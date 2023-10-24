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
        self.model_dicts = self.get_models()


    def get_models(self, id=None, reduce=None, params=""):

        model_dicts = []
        for i in range(100):

            model = dict()
            model["mesh"] = os.path.join(self.path, "input_repaired", "{}.off".format(i))
            if not os.path.isfile(model["mesh"]):
                continue
            model["class"] = ""
            model["model"] = str(i)
            model["eval"] = dict()
            model["eval"]["pointcloud"] = os.path.join(self.path, "eval", str(i), "pointcloud.npz")
            model["eval"]["occ"] = os.path.join(self.path, "eval", str(i), "points.npz")
            model["pointcloud"] = os.path.join(self.path, "pointcloud", "{}.ply".format(i))
            model["planes"] = os.path.join(self.path,"output",params,"planes","{}.npz".format(i))
            model_dicts.append(model)

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