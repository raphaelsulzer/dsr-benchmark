import os, sys, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
from tqdm import tqdm

class ShapeNet:

    def __init__(self,path="/mnt/raphael/ShapeNet",
                 classes=[],
                 mesh_tools_dir="/home/raphael/cpp/mesh-tools/build/release"):

        self.path = path
        self.mesh_tools_dir = mesh_tools_dir
        self.classes = classes
        self.model_dicts = {}

        self.metadata = {
        "sofa": "04256520",
        "airplane": "02691156",
        "lamp":"03636649",
        "phone": "04401088",
        "vessel": "04530566",
        "speaker": "03691459",
        "chair": "03001627",
        "cabinet": "02933112",
        "table": "04379243",
        "display": "03211117",
        "car": "02958343",
        "bench": "02828884",
        "rifle": "04090263"
        }

        if not classes:
            with open(os.path.join(self.path, "classes.lst"), 'r') as f:
                categories = f.read().split('\n')
            if '' in categories:
                categories.remove('')
            self.classes = categories
        else:
            temp = []
            for c in classes:
                temp.append(self.metadata[c])
            self.classes = temp


    def getModels(self,splits=["train","val","test"],scan_conf="4",reduce=None):
        self.scan_conf = scan_conf
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
                    d["scan"] = os.path.join(self.path,c,m,"scan",str(scan_conf)+".npz")
                    d["scan_ply"] = os.path.join(self.path,c,m,"scan",str(scan_conf)+".ply")
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

    def estimNormals(self,method='jet',neighborhood=30,orient=1):
        if(len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)

        for s in self.splits:
            models = self.model_dicts[s]
            for m in tqdm(models,ncols=50):
                try:
                    command = [str(os.path.join(self.mesh_tools_dir,"normal")),
                               "-w", str(self.path),
                               "-s", "npz",
                               "-i", str(os.path.join(m["class"],m["model"],"scan",self.scan_conf+".npz")),
                               "-o", str(os.path.join(m["class"],m["model"],"scan",self.scan_conf)),
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

    def scan(self,scan_setting="4"):

        if(len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)

        scan = scan_settings[scan_setting]

        for s in self.splits:
            models = self.model_dicts[s]
            for m in tqdm(models,ncols=50):

                os.makedirs(os.path.join(self.path,m["class"],m["model"],"scan"),exist_ok=True)

                command = [str(os.path.join(self.mesh_tools_dir,"scan")),
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