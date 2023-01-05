from glob import glob
import os

from converter import Converter
from datasets.reconstructed_dataset import learning_dataset


cc=Converter()

real = ["Ignatius","scan1","scan6","templeRing","Truck"]

learning = ["shapenet3000/02691156/d18592d9615b01bbbc0909d98a1ff2b4"]



path = "/home/rsulzer/data/benchmark"


for exp,d in learning_dataset.items():
    models = d["models"]
    for model in models:
        mpath = os.path.join(path,exp,model)
        methods = glob(mpath + "/*")
        for me in methods:
            if(os.path.splitext(me)[1]==".off"):
                cc.convert_mesh(me,"ply")