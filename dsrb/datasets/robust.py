import os
import open3d as o3d

from dsrb import DefaultDataset

class RobustDataset(DefaultDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.path = os.path.join(self.path,"RobustLowPolyDataSet","Thingi10k")
        self.model_dicts = []


    def get_models(self, names=None, reduce=None, params=""):

        model_dicts = []
        for i in range(100):

            d = dict()
            d["mesh"] = os.path.join(self.path, "input_repaired", "{}.off".format(i))
            # d["mesh"] = os.path.join(self.path, "input", "{}.obj".format(i))
            if not os.path.isfile(d["mesh"]):
                continue
            d["class"] = ""
            d["model"] = str(i)
            d["eval"] = dict()
            d["eval"]["pointcloud"] = os.path.join(self.path, "eval", str(i), "pointcloud.npz")
            d["eval"]["occ"] = os.path.join(self.path, "eval", str(i), "points.npz")
            d["pointcloud_ply"] = os.path.join(self.path, "pointcloud", "{}.ply".format(i))
            d["pointcloud"] = os.path.join(self.path, "pointcloud", "{}.npz".format(i))

            d["planes"] = os.path.join(self.path,"output","{"+params+"}","{}","planes","{}.npz".format(i))
            d["planes_ply"] = os.path.join(self.path,"output","{"+params+"}","{}","planes","{}.ply".format(i))

            d["output"] = {}
            d["output"]["surface"] = os.path.join(self.path,"output", "{"+params+"}", "{}", "surface", "{}.obj".format(i))
            d["output"]["surface_simplified"] = os.path.join(self.path, "output","{"+params+"}", "{}", "surface_simplified", "{}.obj".format(i))
            d["output"]["surface_simplified_triangulated"] = os.path.join(self.path, "output","{"+params+"}", "{}", "surface_simplified_triangulated", "{}.obj".format(i))
            d["output"]["partition"] = os.path.join(self.path, "output","{"+params+"}", "{}", "partition", str(i) ,"partition.ply")
            d["output"]["partition_pickle"] = os.path.join(self.path, "output","{"+params+"}", "{}", "partition", str(i))
            d["output"]["in_cells"] = os.path.join(self.path, "output", "{"+params+"}", "{}", "in_cells", "{}.ply".format(i))
            d["output"]["settings"] = os.path.join(self.path, "output", "{"+params+"}", "{}", "settings", "{}.yaml".format(i))

            model_dicts.append(d)

        temp = []
        if names is not None:
            for model in model_dicts:
                if str(names) == model["model"]:
                    temp.append(model)

            model_dicts = temp
        if reduce is not None:
            model_dicts = model_dicts[:reduce]

        self.model_dicts = model_dicts
        return model_dicts

    def input_off2obj(self):

        for m in self.model_dicts:
            mesh = o3d.io.read_triangle_mesh(m["mesh"])
            o3d.io.write_triangle_mesh(m["mesh"][:-4] + ".obj", mesh)


if __name__ == '__main__':

    ds = RobustDataset()
    ds.get_models()

    ds.make_eval()

    # ds.sample(n_points=4000000)
