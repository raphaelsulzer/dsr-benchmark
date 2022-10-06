import os, sys, subprocess, shutil
import time

from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__),"..","..","datasets"))
from modelnet10 import ModelNet10
from shapenet import ShapeNet
from berger import Berger
import argparse
import glob


def runModelNet(args):

    for s in args.models.keys():
        smodels = args.models[s]
        for m in tqdm(smodels,ncols=50):

            try:
                os.makedirs(os.path.join(args.path, m["class"], m["model"]), exist_ok=True)
                outfile = os.path.join(args.path, m["class"], m["model"], "dgnn", args.scan + "_3dt.npz")
                if (os.path.isfile(outfile) and not args.overwrite):
                    print("exists!")
                    continue

                command = [str(os.path.join(args.sure_dir, "feat")),
                           "-w", str(os.path.join(args.path, m["class"], m["model"])),
                           "-i", str(os.path.join("..", "scan", args.scan, m["model"], "scan")),
                           "-o", args.scan,
                           "-g", str(os.path.join("..", "..", "..", "ModelNet10_watertight",
                                                  m["class"] + "_" + m["model"] + ".off")),
                           "-s", "npz",
                           "-e", ""]

                print(*command)
                p = subprocess.Popen(command)
                p.wait()

            except Exception as e:
                print(e)
                print("Skipping {}/{}".format(m["class"], m["model"]))

def runShapeNet(args):

    for s in args.models.keys():
        smodels = args.models[s]
        for m in tqdm(smodels,ncols=50):

            try:
                outfile = os.path.join(args.path,m["class"],m["model"],"dgnn",args.scan+"_3dt.npz")
                if(os.path.isfile(outfile) and not args.overwrite):
                    print("exists!")
                    continue

                command = [str(os.path.join(args.sure_dir,"feat")),
                           "-w", str(os.path.join(args.path,m["class"],m["model"])),
                           "-i", "scan/"+args.scan,
                           "-o", args.scan,
                           "-g", "mesh/mesh.off",
                           "-s", "npz",
                           "-e",""]
                # print(*command)
                # p = subprocess.Popen(command)
                p = subprocess.Popen(command, shell=False,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                p.wait()

            except Exception as e:
                print(e)
                print("Skipping {}/{}".format(m["class"], m["model"]))



def runReconbench(args):


    for m in tqdm(args.models,ncols=50):

        try:
            outfile = os.path.join(args.path,"dgnn",m["class"],args.scan+"_3dt.npz")
            if(os.path.isfile(outfile) and not args.overwrite):
                print("exists!")
                continue

            command = [str(os.path.join(args.sure_dir,"feat")),
                       "-w", args.path,
                       "-i", str(os.path.join("scan",m["class"],args.scan)),
                       "-o", str(os.path.join(m["class"],args.scan)),
                       "-g", str(os.path.join("mesh","1",m["class"]+".off")),
                       "-s", "npz",
                       "-e",""]
            print(*command)
            p = subprocess.Popen(command)
            p.wait()

            outpath = os.path.join(args.path,"dgnn",m["class"])
            os.makedirs(outpath,exist_ok=True)
            for f in glob.glob(os.path.join(args.path,"dgnn","4_*")):
                shutil.move(f,outpath)


        except Exception as e:
            print(e)
            print("Skipping {}/{}".format(m["class"], m["model"]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train SurfaceNet')

    parser.add_argument('-d', '--dataset', type=str, default='shapenet',
                        help='which dataset')
    parser.add_argument('-c', '--classes', type=str,  nargs='+', default=[],
                        help='which classes')
    parser.add_argument('-o', '--overwrite', type=bool, default=True,
                        help='overwrite existing files')
    parser.add_argument('-s', '--scan', type=str, default="4",
                        help='scan configuration')

    args = parser.parse_args()

    jz = 0



    if args.dataset == 'shapenet':

        if jz:
            args.path = "/linkhome/rech/genlgm01/uku93eu/scratch/ShapeNet"
            args.sure_dir = "/linkhome/rech/genlgm01/uku93eu/code"
        else:
            args.path = "/mnt/raphael/ShapeNet"
            args.sure_dir = "/home/raphael/cpp/mesh-tools/build/release"

        # ['02691156','02828884','02933112','02958343',
        # '03001627','03211117','03636649','03691459','04090263',
        # '04256520','04379243','04401088','04530566']

        dataset = ShapeNet(path=args.path)
        models = dataset.getModels(splits=["test100"])

        args.dataset = dataset
        args.models = models

        t0 = time.time()
        runShapeNet(args)

        tt = time.time() - t0
        print("models: ", len(models))
        print("full time: ", tt)
        print("average time: ", tt / len(models))

    elif args.dataset == 'modelnet':

        if jz:
            args.path = "/linkhome/rech/genlgm01/uku93eu/scratch/ModelNet10"
            args.sure_dir = "/linkhome/rech/genlgm01/uku93eu/code"
        else:
            args.path = "/mnt/raphael/ModelNet10"
            args.sure_dir = "/home/raphael/cpp/mesh-tools/build/release"

        dataset = ModelNet10(path=args.path)
        models = dataset.getModels()

        args.dataset = dataset
        args.models = models

        runModelNet(args)


    elif args.dataset == 'reconbench':


        args.path = "/mnt/raphael/reconbench"
        args.sure_dir = "/home/raphael/cpp/mesh-tools/build/release"

        dataset = Berger()
        models = dataset.getModels(scan=args.scan)

        args.dataset = dataset
        args.models = models

        runReconbench(args)


