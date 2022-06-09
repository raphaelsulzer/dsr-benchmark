import os, sys
import shutil

path = "/mnt/raphael/ModelNet10/"

with open(os.path.join(path, "classes.lst"), 'r') as f:
    categories = f.read().split('\n')
if '' in categories:
    categories.remove('')


for c in categories:

    cpath = os.path.join(path,c)
    shutil.rmtree(os.path.join(cpath,"eval"))
    shutil.rmtree(os.path.join(cpath,"gt"))
    shutil.rmtree(os.path.join(cpath,"mesh"))
    shutil.rmtree(os.path.join(cpath,"p2s"))
    shutil.rmtree(os.path.join(cpath,"sap"))
    shutil.rmtree(os.path.join(cpath,"scan"))



