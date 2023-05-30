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

    def __init__(self):

        self.eval_dicts = []
        self.evaluator = MeshEvaluator(n_points=100000)

    def eval(self,models,outpath):

        for m in tqdm(models, ncols=50):

            try:

                # gt_mesh = trimesh.load(m["mesh"], process=False)
                mesh_file = os.path.join(outpath,m["class"],m["model"] + ".ply")
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
                if(not md["watertight"]):
                    print("{}/{}".format(m["class"],m["model"]))

                self.eval_dicts.append(md)

            except Exception as e:
                print(e)
                print("Skipping {}/{}".format(md["class"], md["model"]))

        eval_df = pd.DataFrame(self.eval_dicts)
        eval_df.to_pickle(os.path.join(outpath,"results_all.pkl"))
        eval_df_class = eval_df.groupby(by=['class']).mean()
        eval_df_class.loc['mean'] = eval_df.mean()
        eval_df_class.to_csv(os.path.join(outpath, "results.csv"))

        return eval_df_class



# dataset = ModelNet10()
# split = "test"
# models = dataset.getModels(splits=[split],classes=["bathtub",  "bed",  "desk",  "dresser",  "nightstand",  "toilet"])[split]
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/poco/tr/modelnet/meshes"

dataset = ShapeNet()
split = "test100"
models = dataset.getModels(splits=[split])[split]
outpath = "/mnt/raphael/ShapeNet_out/benchmark/poco/tr/shapenet10000/meshes"

# dataset = Berger()
# models = dataset.getModels(scan=["4"])
# outpath = "/mnt/raphael/ShapeNet_out/benchmark/poco/tr/reconbench/meshes"

print("Eval shapes from ",outpath)
evl = Evaluator()
# ev_dict = evl.evalTrain()
ev_dict = evl.eval(models,outpath)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.precision',3)
print(ev_dict)



a=5



