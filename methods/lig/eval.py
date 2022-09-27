import numpy as np
import os, sys, trimesh
import pandas as pd
sys.path.append("/home/raphael/remote_python/benchmark")

from tqdm import tqdm

from datasets.modelnet10 import ModelNet10
from datasets.shapenet import ShapeNet
from datasets.berger import Berger
from evaluator.eval import MeshEvaluator


class Evaluator:

    def __init__(self,dataset):

        self.rpp = [8, 16, 32]
        self.ps = [0.05, 0.1, 0.2, 0.3]
        self.pts = [2048, 8192]

        self.dataset = dataset

        self.eval_dicts = []
        # df = pd.DataFrame(columns = ["class","model","sigma","lambda","iou","chamfer","normal"])

        self.evaluator = MeshEvaluator(n_points=100000)

    def evalTrain(self):

        split = "train"
        models = self.dataset.getModels(splits=[split],reduce=0.001)

        for m in tqdm(models["train"],ncols=50):
            for x in self.rpp:
                for y in self.ps:
                    for z in self.pts:

                        try:

                            # gt_mesh = trimesh.load(m["mesh"], process=False)
                            mesh_file = os.path.join(outpath,str(x),str(y),str(z),m["class"]+"_"+m["model"]+".ply")
                            mesh = trimesh.load(mesh_file, process=False)
                            pointcloud = np.load(m["pointcloud"])
                            pointcloud_tgt = pointcloud["points"]
                            normals_tgt = pointcloud["normals"]
                            points = np.load(m["occ"])
                            points_tgt = points['points']
                            occ_tgt = np.unpackbits(points['occupancies'])

                            eval_dict_mesh = self.evaluator.eval_mesh(
                                mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)

                            md = {}
                            md["class"] = m["class"]
                            md["model"] = m["model"]
                            md["rpp"] = str(x)
                            md["ps"] = str(y)
                            md["pts"] = str(z)
                            md["iou"] = eval_dict_mesh["iou"]
                            md["chamfer"] = eval_dict_mesh["chamfer-L1"]
                            md["normal"] = eval_dict_mesh["normals"]
                            md["watertight"] = eval_dict_mesh["watertight"]
                            self.eval_dicts.append(md)

                        except Exception as e:
                            raise
                            print(e)
                            print("Skipping {}/{}".format(m["class"],m["model"]))

        eval_df = pd.DataFrame(self.eval_dicts)
        # eval_df.set_index(['idx'], inplace=True)

        eval_df_ = eval_df.groupby(by=['rpp', 'ps', 'pts']).mean()

        return eval_df_


    def evalTest(self,models,outpath):

        l = '1.0'

        for m in tqdm(models, ncols=50):

            try:

                # gt_mesh = trimesh.load(m["mesh"], process=False)
                mesh_file = os.path.join(outpath,
                                         m["class"], m["model"] + ".ply")
                mesh = trimesh.load(mesh_file, process=False)
                pointcloud = np.load(m["pointcloud"])
                pointcloud_tgt = pointcloud["points"]
                normals_tgt = pointcloud["normals"]
                points = np.load(m["occ"])
                points_tgt = points['points']
                occ_tgt = np.unpackbits(points['occupancies'])

                eval_dict_mesh = self.evaluator.eval_mesh(
                    mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)

                md = {}
                md["class"] = m["class"]
                md["model"] = m["model"]
                md["iou"] = eval_dict_mesh["iou"]
                md["chamfer"] = eval_dict_mesh["chamfer-L1"]
                md["normal"] = eval_dict_mesh["normals"]
                md["boundary_edges"] = eval_dict_mesh["boundary_edges"]
                md["non-manifold_edges"] = eval_dict_mesh["non-manifold_edges"]
                md["watertight"] = eval_dict_mesh["watertight"]
                md["components"] = eval_dict_mesh["components"]
                if (not md["watertight"]):
                    print("non watertight mesh {}/{}".format(m["class"], m["model"]))

                self.eval_dicts.append(md)

            except Exception as e:
                print(e)
                print("Skipping {}/{}".format(md["class"], md["model"]))

        eval_df = pd.DataFrame(self.eval_dicts)
        eval_df.to_pickle(os.path.join(outpath, "results_all.pkl"))
        eval_df_class = eval_df.groupby(by=['class']).mean()
        eval_df_class.loc['mean'] = eval_df.mean()
        eval_df_class.to_csv(os.path.join(outpath, "results.csv"))

        return eval_df_class


# dataset = ModelNet10()
# split = "test"
# models = dataset.getModels(splits=[split],classes=["bathtub",  "bed",  "desk",  "dresser",  "nightstand",  "toilet"])[split]
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/lig/modelnet"

# dataset = ShapeNet()
# split = "test100"
# models = dataset.getModels(reduce=0.1,splits=[split],scan="4")[split]
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/lig/shapenet3000"


dataset = Berger()
models = dataset.getModels(scan_conf=["mvs4"])
outpath = "/mnt/raphael/ShapeNet_out/benchmark/lig/reconbench"
# outpath = "/mnt/raphael/reconbench_out/benchmark/lig/reconbench"


print("Eval shapes from ",outpath)
evl = Evaluator(dataset)
# ev_dict = evl.evalTrain()
ev_dict = evl.evalTest(models,outpath)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.precision',3)
print(ev_dict)
ev_dict.to_csv(os.path.join(outpath,"results.csv"))



a=5



