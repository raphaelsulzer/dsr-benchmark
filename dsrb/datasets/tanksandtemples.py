import os
import numpy as np
import open3d as o3d

from dsrb import DefaultDataset


class TanksAndTemples(DefaultDataset):

    def __init__(self,classes=[]):
        super().__init__()
        self.path = os.path.join(self.path, "TanksAndTemples")

    def get_models(self,list="models.lst",names=None):

        models = np.genfromtxt(os.path.join(self.path,list),dtype=str)

        for m in models:

            if m[0] == "#":
                continue

            if names is not None:
                if m not in names:
                    continue

            c = ""

            d = {}
            d["class"] = c
            d["model"] = m
            d["path"] = os.path.join(self.path,c,m)

            d["alignment"] = os.path.join(self.path,m,"{}_trans.txt".format(m))
            d["crop"] = os.path.join(self.path,m,"{}.json".format(m))
            d["pointcloud_mvs"] = os.path.join(self.path,m,"{}_COLMAP.ply".format(m))
            d["pointcloud"] = os.path.join(self.path,m,"pointcloud.npz")
            d["pointcloud_ply"] = os.path.join(self.path,m,"pointcloud.ply")
            d["pointcloud_lidar"] = os.path.join(self.path,c,m,"pointcloud","pointcloud_voxel_001.ply")
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

            d["output"] = {}
            d["output"]["surface"] = os.path.join(self.path,c, m, "{}", "surface.ply")
            d["output"]["surface_simplified"] = os.path.join(self.path,c, m, "{}", "surface_simplified.obj")
            d["output"]["partition"] = os.path.join(self.path,c, m, "{}", "partition.ply")
            d["output"]["partition_pickle"] = os.path.join(self.path,c, m, "{}", "partition")
            d["output"]["in_cells"] = os.path.join(self.path,c, m, "{}", "in_cells.obj")
            d["output"]["settings"] = os.path.join(self.path,c, m, "{}", "settings.yaml")

            self.model_dicts.append(d)

        if not len(self.model_dicts):
            print("ERROR: no models found!")
            return None
        return self.model_dicts

    def allign_and_crop(self):

        for model in self.model_dicts:
            pc = o3d.io.read_point_cloud(model["pointcloud_mvs"])
            trans = np.loadtxt(model["alignment"])
            pc = pc.transform(trans)
            crop = o3d.visualization.read_selection_polygon_volume(model["crop"])
            pc = crop.crop_point_cloud(pc)

            o3d.io.write_point_cloud(model["pointcloud_ply"],pc)

            np.savez_compressed(model["pointcloud"], points=np.asarray(pc.points), normals=np.asarray(pc.normals))


if __name__ == '__main__':

    ds = TanksAndTemples()
    ds.get_models(names="Truck")
    # ds.allign_and_crop()
    ds.make_poisson(depth=10,keep_largest_component_only=True)

    ds.ply2npz()


