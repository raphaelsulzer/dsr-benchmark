import os, sys, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__)))
from tqdm import tqdm

class ModelNet10:

    def __init__(self,path="/mnt/raphael/ModelNet10",
                 classes=[],
                 mesh_tools_dir="/home/raphael/cpp/mesh-tools/build/release"
                 ):


        self.path = path
        self.mesh_tools_dir = mesh_tools_dir
        self.model_dicts = {}

        with open(os.path.join(self.path, "classes.lst"), 'r') as f:
            categories = f.read().split('\n')
        if '' in categories:
            categories.remove('')
        self.classes = categories

    def getModels(self,splits=["train","test"],scan_conf="43",classes=None,reduce=None):


        if classes is not None:
            self.classes = classes


        self.scan_conf=scan_conf
        self.splits = splits
        for s in splits:
            class_list = []
            for c in self.classes:
                split_file = os.path.join(self.path, c, s + '.lst')
                with open(split_file, 'r') as f:
                    models = f.read().split('\n')
                if '' in models:
                    models.remove('')

                if reduce:
                    models = models[:int(len(models)*reduce)]

                for m in models:
                    d = {}
                    d["class"] = c
                    d["model"] = m
                    d["scan"] = os.path.join(self.path,c,m,"scan",scan_conf+".npz")
                    d["scan_ply"] = os.path.join(self.path,c,m,"scan",scan_conf+".ply")
                    d["occ"] = os.path.join(self.path,c,m,"eval","points.npz")
                    d["pointcloud"] = os.path.join(self.path,c,m,"eval","pointcloud.npz")
                    d["mesh"] = os.path.join(self.path+"_watertight",c+"_"+m+".off")
                    class_list.append(d)

            self.model_dicts[s] = class_list

        return self.model_dicts

    def getOne(self,splits=["train","test"],scan_conf="43",classes=None,id=0):

        if classes is not None:
            if isinstance(classes, list):
                self.classes = classes
            else:
                self.classes = [classes]

        if isinstance(splits, list):
            self.splits = splits
        else:
            self.splits = [splits]

        for s in self.splits:
            class_list = []
            for c in self.classes:
                split_file = os.path.join(self.path, c, s + '.lst')
                with open(split_file, 'r') as f:
                    models = f.read().split('\n')
                if '' in models:
                    models.remove('')

                if id >= len(models):
                    print("ID exceeds model dimension {}!".format(len(models)))
                    sys.exit(1)

                m = models[id]

                d = {}
                d["class"] = c
                d["model"] = m
                d["scan"] = os.path.join(self.path, c, "scan", scan_conf, m, "scan.npz")
                d["occ"] = os.path.join(self.path, c, "eval", m, "points.npz")
                d["pointcloud"] = os.path.join(self.path, c, "eval", m, "pointcloud.npz")
                d["mesh"] = os.path.join(self.path + "_watertight", c + "_" + m + ".off")
                class_list.append(d)

            self.model_dicts[s] = class_list

        return self.model_dicts

    def estimNormals(self, method='jet', neighborhood=30, orient=1):
        if (len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)

        for s in self.splits:
            models = self.model_dicts[s]
            for m in tqdm(models, ncols=50):
                try:
                    command = [str(os.path.join(self.mesh_tools_dir, "normal")),
                               "-w", str(self.path),
                               "-s", "npz",
                               "-i", str(os.path.join(m["class"], m["model"], "scan", self.scan_conf + ".npz")),
                               "-o", str(os.path.join(m["class"], m["model"], "scan", self.scan_conf)),
                               "--method", method,
                               "--neighborhood", str(neighborhood),
                               "--orient", str(orient),
                               "--overwrite", "1"]
                    print(*command)
                    p = subprocess.Popen(command)
                    p.wait()
                except Exception as e:
                    print(e)
                    print("Skipping {}/{}".format(m["class"], m["model"]))



