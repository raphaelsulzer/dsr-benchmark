import numpy as np
import os, sys, subprocess, trimesh
import pandas as pd

from tqdm import tqdm

outpath = "/mnt/raphael/ShapeNet_out/benchmark/labatut/"

df = pd.read_csv(os.path.join(outpath,"results.csv"))

a=5



