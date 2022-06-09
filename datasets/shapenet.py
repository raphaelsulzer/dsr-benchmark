import os, sys, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
from tqdm import tqdm

class ShapeNet:

    def __init__(self,path="/mnt/raphael/ShapeNet", classes=[]):

        self.path = path
        self.classes = classes
        self.model_dicts = {}

        if not classes:
            with open(os.path.join(self.path, "classes.lst"), 'r') as f:
                categories = f.read().split('\n')
            if '' in categories:
                categories.remove('')
            self.classes = categories

    def getModels(self,splits=["train","val","test"],scan="4",reduce=None):
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
                    d["scan"] = os.path.join(self.path,c,m,"scan",str(scan)+".npz")
                    d["occ"] = os.path.join(self.path,c,m,"eval","points.npz")
                    d["pointcloud"] = os.path.join(self.path,c,m,"eval","pointcloud.npz")
                    d["mesh"] = os.path.join(self.path,c,m,"mesh","mesh.off")
                    class_list.append(d)

            self.model_dicts[s] = class_list

        return self.model_dicts

    def getById(self,id="0000"):

        if not self.model_dicts:
            print("Call getModels() first")
            return None

        m = next((item for item in self.model_dicts if item["model"] == id), None)

        if m is not None:
            return m
        else:
            print("Model with ID {} does not exist".format(id))
            return None


    def getRandom(self, id="0000"):

        if not self.model_dicts:
            print("Call getModels() first")
            return None

        m = next((item for item in self.model_dicts if item["model"] == id), None)

        if m is not None:
            return m
        else:
            print("Model with ID {} does not exist".format(id))
            return None


    def scan(self,scan_setting="4",scanner_dir="/home/raphael/cpp/mesh-tools/build/release/scan"):

        if(len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)

        scan = scan_settings[scan_setting]

        for s in self.splits:
            models = self.model_dicts[s]
            for m in tqdm(models,ncols=50):

                os.makedirs(os.path.join(self.path,m["class"],m["model"],"scan"),exist_ok=True)

                command = [scanner_dir,
                           "-w", str(self.path),
                           "-i", str(os.path.join(m["class"],m["model"],"mesh","mesh.off")),
                           "-o", str(os.path.join(m["class"],m["model"],"scan",scan_setting)),
                           "--points", scan["points"],
                           "--noise", scan["noise"],
                           "--outliers", scan["outliers"],
                           "--cameras", scan["cameras"],
                           "--normal_method", "jet",
                           "--export", "all"]
                print(*command)
                p = subprocess.Popen(command)
                p.wait()