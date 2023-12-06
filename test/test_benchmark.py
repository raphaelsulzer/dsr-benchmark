import os, argparse
import open3d as o3d
import pandas as pd

from dsrb import Berger, ModelNet10, ShapeNet
from dsrb.eval import MeshEvaluator
from dsrb.set_paths import set_paths

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def test(args):

    # note that, here we do not actually use the data in data_dir, but the small sample set included in the repo
    set_paths(DATA_DIR=args.dataset_dir,CPP_DIR=args.cpp_dir)

    # note necessary to provide if set_paths is called. here we override the path from set_paths
    dataset_path = os.path.join(os.path.dirname(__file__),"reconbench")
    ds = Berger(path=dataset_path)
    models = ds.get_models(scan_configuration="mvs")
    ds.make_eval()
    ds.scan(scan_configuration="mvs")

    for model in models:
        # make a Poisson reconstruction with open3d
        pcd = o3d.io.read_point_cloud(model["scan_ply"])
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        os.makedirs(os.path.dirname(model["output"]["surface"].format("poisson")),exist_ok=True)
        o3d.io.write_triangle_mesh(model["output"]["surface"].format("poisson"),mesh)

    ev = MeshEvaluator()
    results_full, results_class = ev.eval(models,outpath=ds.path,method="poisson")
    print(results_full)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp_dir", type=str, default= "/home/rsulzer/cpp", help="The directory in which the mesh-tools folder is located.")
    parser.add_argument("--dataset_dir", type=str, default= "/home/rsulzer/data", help="The directory in which the dataset folders are located.")
    args = parser.parse_args()

    test(args)
