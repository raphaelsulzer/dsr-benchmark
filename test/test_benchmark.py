from dsrb.datasets import Berger, ModelNet10, ShapeNet
from dsrb.eval import MeshEvaluator
import open3d as o3d
import pandas as pd
import os
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


dataset_path = os.path.join(os.path.dirname(__file__),"reconbench")
ds = Berger(path=dataset_path)
models = ds.get_models(scan_configuration="mvs")
ds.make_eval()
ds.scan(scan_configuration="mvs",verbosity=0)

for model in models:
    # make a Poisson reconstruction with open3d
    pcd = o3d.io.read_point_cloud(model["scan_ply"])
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
    os.makedirs(os.path.dirname(model["output"]["surface"].format("poisson")),exist_ok=True)
    o3d.io.write_triangle_mesh(model["output"]["surface"].format("poisson"),mesh)


ev = MeshEvaluator()
results_full, results_class = ev.eval(models,outpath=ds.path,method="poisson")
print(results_full)




