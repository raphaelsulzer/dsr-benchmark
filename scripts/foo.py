from converter import Converter
import os

file = os.path.join("/home/rsulzer/data/benchmark/exp/table/0008/input.npz")
# pcd = PyntCloud.from_file(file)

cc = Converter()

cc.convert_pc(file,"ply",orient="sensor")


a=5
