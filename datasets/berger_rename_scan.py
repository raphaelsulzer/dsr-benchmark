import os, sys, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
import numpy as np
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from berger import Berger

dataset = Berger()
models = dataset.getModels(type="berger",scan_conf=["4"])


for m in models:

    print(m["class"],m["model"])

    infile = m["scan"]
    os.rename()

