import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from datasets.berger import Berger
from datasets.shapenet import ShapeNet
from datasets.modelnet10 import ModelNet10


# dataset = ModelNet10()
# split="test"
# dataset.getModels(scan_conf="43",splits=[split])
# dataset.estimNormals()

# dataset = ShapeNet()
# split="test100"
# dataset.getModels(scan_conf="6",splits=[split])
# dataset.estimNormals(neighborhood=20)

dataset = Berger()
# split="test100"
dataset.getModels(scan_conf="4")
dataset.estimNormals(neighborhood=30)
