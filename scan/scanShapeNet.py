import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),"..","datasets"))
from berger import Berger
from shapenet import ShapeNet


dataset = ShapeNet()
dataset.getModels(scan=["6"],splits=["test100"])
dataset.scan("6")
# dataset.sample()


