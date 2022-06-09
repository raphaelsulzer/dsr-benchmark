import os, sys


path = "/mnt/raphael/ModelNet10/"

with open(os.path.join(path, "classes.lst"), 'r') as f:
    categories = f.read().split('\n')
if '' in categories:
    categories.remove('')


for c in categories:

    files = os.listdir(os.path.join(path,c))

    for f in files:
        if not f.isdecimal():
            continue

        outpath = os.path.join(path,c,f)
        os.makedirs(os.path.join(outpath,"eval"),exist_ok=True)
        os.makedirs(os.path.join(outpath,"p2s"),exist_ok=True)
        os.makedirs(os.path.join(outpath,"mesh"),exist_ok=True)
        os.makedirs(os.path.join(outpath,"sap"),exist_ok=True)
        os.makedirs(os.path.join(outpath,"scan"),exist_ok=True)

        try:
            os.rename(os.path.join(path,c,"eval",f,"points.npz"),os.path.join(outpath,"eval","points.npz"))
            os.rename(os.path.join(path,c,"eval",f,"pointcloud.npz"),os.path.join(outpath,"eval","pointcloud.npz"))
            os.rename(os.path.join(path,c,"p2s","05_query_dist",c+"_"+f+".off.npy"),os.path.join(outpath,"p2s","dists.npy"))
            os.rename(os.path.join(path,c,"p2s","05_query_pts",c+"_"+f+".off.npy"),os.path.join(outpath,"p2s","pts.npy"))
            os.rename(os.path.join(path,c,"sap",f,"psr.npz"),os.path.join(outpath,"sap","psr.npz"))
            os.rename(os.path.join(path,c,"scan","43",f,"scan.npz"),os.path.join(outpath,"scan","43.npz"))
            os.rename(os.path.join(path,"..","ModelNet10_watertight",c+"_"+f+".off"),os.path.join(outpath,"mesh","mesh.off"))
        except:
            print("Problem with {}/{}".format(c,f))