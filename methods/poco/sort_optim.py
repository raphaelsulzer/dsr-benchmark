import os, sys, shutil
sys.path.append("/home/raphael/remote_python/benchmark")

from tqdm import tqdm

from datasets.modelnet10 import ModelNet10
from datasets.shapenet import ShapeNet
from datasets.berger import Berger
from evaluator.eval import MeshEvaluator



dataset = Berger()
models = dataset.getModels(type="berger")
outpath = "/mnt/raphael/reconbench_out"


for m in models:

    try:
        target = os.path.join(outpath,"sap",m["class"],m["model"],"target_pcl.ply")
        os.remove(target)
    except:
        pass

    src = os.path.join(outpath,"sap",m["class"],m["model"],"recon.ply")
    # dest = os.path.join(outpath,"sap",m["class"],m["model"]+".ply")
    dest = os.path.join(outpath,"sap",m["class"],m["model"],"vis","mesh","1600.ply")

    shutil.move(src,dest)


    a=5



