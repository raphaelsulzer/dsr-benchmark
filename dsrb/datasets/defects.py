import os, trimesh
import numpy as np
from copy import deepcopy
import open3d as o3d

from dsrb import DefaultDataset

class DefectsDataset(DefaultDataset):

    def __init__(self,classes=[]):
        super().__init__()
        self.path = os.path.join(self.path, "DefectsDataset")

        self.classes = classes if isinstance(classes,list) else [classes]
        if not classes:
            with open(os.path.join(self.path, "classes.lst"), 'r') as f:
                categories = f.read().split('\n')
            if '' in categories:
                categories.remove('')
            self.classes = categories




    def get_models(self,list="models.lst",names=None):


        for c in self.classes:

            models = np.genfromtxt(os.path.join(self.path, c, list), dtype=str)


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

                d["pointcloud"] = os.path.join(self.path,c,m,"pointcloud.npz")
                d["pointcloud_ply"] = os.path.join(self.path,c,m,"pointcloud.ply")
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
                d["output"]["surface_simplified"] = os.path.join(self.path, c, m, "{}", "surface_simplified.obj")
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


    def make_density(self, levels):

        for model in self.model_dicts:

            pc = o3d.io.read_point_cloud(model["pointcloud_ply"])
            for k in levels:

                npc = pc.random_down_sample(k/100)
                outfile = model["pointcloud_ply"].replace("/100/","/density{}/".format(str(k)))
                os.makedirs(os.path.dirname(outfile),exist_ok=True)
                o3d.io.write_point_cloud(outfile,npc)
                np.savez_compressed(model["pointcloud"],points=np.asarray(npc.points),normals=np.asarray(npc.normals))


    def make_outliers(self,levels):

        for model in self.model_dicts:

            pc = o3d.io.read_point_cloud(model["pointcloud_ply"])
            n = len(np.asarray(pc.points))
            pmin = pc.get_min_bound()
            pmax = pc.get_max_bound()

            for k in levels:
                outliers_points = np.random.uniform(pmin,pmax,(int(n*k/100),3))
                outliers_normals = np.random.uniform(pmin,pmax,(int(n*k/100),3))
                npc = deepcopy(pc)
                npc.points=o3d.utility.Vector3dVector(np.vstack((np.asarray(pc.points),outliers_points)))
                npc.normals=o3d.utility.Vector3dVector(np.vstack((np.asarray(pc.normals),outliers_normals)))
                outfile = model["pointcloud_ply"].replace("/100/","/outliers{}/".format(str(k)))
                os.makedirs(os.path.dirname(outfile),exist_ok=True)
                o3d.io.write_point_cloud(outfile,npc)
                np.savez_compressed(model["pointcloud"],points=np.asarray(npc.points),normals=np.asarray(npc.normals))


    def make_noise(self,levels):

        for model in self.model_dicts:

            pc = o3d.io.read_point_cloud(model["pointcloud_ply"])
            n = len(np.asarray(pc.points))
            pmin = pc.get_min_bound()
            pmax = pc.get_max_bound()
            diag = np.linalg.norm(pmin-pmax)

            for k in levels:
                noise_points = np.random.normal(loc=0,scale=diag*k/100,size=(n,3))
                npc = deepcopy(pc)
                npc.points=o3d.utility.Vector3dVector(np.asarray(pc.points)+noise_points)
                # opc.normals+=o3d.utility.Vector3dVector(outliers_normals)
                outfile = model["pointcloud_ply"].replace("/100/","/noise{}/".format(str(k)))
                os.makedirs(os.path.dirname(outfile),exist_ok=True)
                o3d.io.write_point_cloud(outfile,npc)
                np.savez_compressed(model["pointcloud"],points=np.asarray(npc.points),normals=np.asarray(npc.normals))

    def make_models_list(self):

        for c in self.classes:

            models = []
            ms = os.listdir(os.path.join(self.path,c))
            for model in ms:
                if os.path.isdir(os.path.join(self.path,c,model)):
                    models.append(model)

            np.savetxt(os.path.join(self.path,c,"models.lst"),models,fmt="%s")

    def keep_largest_components_only(self):

        import vedo.io

        for model in self.model_dicts:

            mesh = vedo.io.load(model["mesh"])
            mesh = mesh.extract_largest_region()

            os.rename(model["mesh"],model["mesh"].replace("mesh","mesh_ori"))

            vedo.io.write(mesh,model["mesh"].replace(".off",".ply"))

            mesh = trimesh.load_mesh(model["mesh"].replace(".off",".ply"))
            mesh.export(model["mesh"])


if __name__ == '__main__':

    ds = DefectsDataset(classes="carter")
    # ds.get_models(names="100")
    # ds.get_models(names=["outliers10","outliers20","outliers50"])

    # ds.get_models(names=["density1","density2","density5", "outliers100", "outliers150"])
    ds.get_models(names=["outliers75"])

    # ds.make_noise([0.1,0.2,0.5])
    # ds.make_outliers([1,2,5])
    # ds.make_density([50,20,10])

    # ds.make_noise([0.2,0.5,1.0])
    # ds.make_outliers([10,20,50])
    # ds.make_density([50,20,10])

    # ds.make_outliers([100,150])
    # ds.make_density([2])
    # ds.make_noise([1.5])
    # ds.make_outliers([75])
    # # # # #
    # ds.make_models_list()

    # # ds.get_models()
    ds.make_poisson(depth=9)

    # ds.keep_largest_components_only()




