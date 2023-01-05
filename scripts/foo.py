from pyntcloud import PyntCloud
from converter import Converter
import os

file = os.path.join("/home/rsulzer/data/benchmark/shapenet3000/02691156/d18592d9615b01bbbc0909d98a1ff2b4/input.npz")
# pcd = PyntCloud.from_file(file)

cc = Converter()

cc.convert_pc(file,"ply")


a=5
