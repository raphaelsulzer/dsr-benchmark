import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),"..","datasets"))
from berger import Berger


dataset = Berger()
dataset.getModels(scan_conf=["4"])
dataset.scan("4")
# dataset.sample()


