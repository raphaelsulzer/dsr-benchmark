import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),"..","datasets"))
from berger import Berger
from shapenet import ShapeNet


dataset = ShapeNet()
dataset.getModels(scan_conf="6",splits=["test100"])
dataset.scan()
# dataset.sample()


