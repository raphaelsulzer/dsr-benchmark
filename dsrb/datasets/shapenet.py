import os, sys, subprocess
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__)))
from scan_settings import scan_settings
from tqdm import tqdm
from default_dataset import DefaultDataset

class ShapeNet(DefaultDataset):

    def __init__(self,path="/home/rsulzer/data/ShapeNet",
                 classes=[],
                 mesh_tools_dir="/home/raphael/cpp/mesh-tools/build/release"):
        super().__init__()

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

        if(not os.path.isfile(os.path.join(self.path, "classes.lst"))):
            with open(os.path.join(self.path, "classes.lst"), 'w') as f:
                for v in list(self.metadata.values())[:-1]:
                    f.write(str(v)+"\n")
                f.write(str(list(self.metadata.values())[-1]))


        if not classes:
            with open(os.path.join(self.path, "classes.lst"), 'r') as f:
                categories = f.read().split('\n')
            if '' in categories:
                categories.remove('')
            self.classes = categories
        else:
            temp = []
            for c in classes:
                if c.isnumeric():
                    temp.append(c)
                else:
                    temp.append(self.metadata[c])
            self.classes = temp


    # def setup(self):
    #
    #     for c in self.classes:
    #
    #         models = os.listdir(os.path.join(self.path,c))
    #
    #         for m in models:
    #             # try:
    #
    #             infile = os.path.join(self.path,c,m,"mesh.off")
    #             if not os.path.isfile(infile):
    #                 continue
    #             os.makedirs(os.path.join(self.path,c,m,"mesh"),exist_ok=True)
    #             outfile = os.path.join(self.path,c,m,"mesh","mesh.off")
    #             os.rename(infile,outfile)


    def setup(self):

        """Move models from classX/4_watertight_scaled/modelX.off to classX/modelX/mesh/mesh.off"""

        for split in self.model_dicts.keys():

            for model in tqdm(self.model_dicts[split]):

                print("Process model {}/{}".format(model["class"], model["model"]))

                try:

                    infile = os.path.join(self.path,model["class"],"4_watertight_scaled","{}.off".format(model["model"]))
                    if not os.path.isfile(infile):
                        continue
                    outfile = os.path.join(self.path,model["class"],model["model"],"mesh","mesh.off")
                    if not os.path.isdir(self.path,model["class"],model["model"]):
                        continue
                    os.makedirs(os.path.join(self.path,model["class"],model["model"],"mesh"))
                    os.rename(infile,outfile)
                except Exception as e:
                    raise e
                    print("Problem with model {}/{}".format(model["class"],model["model"]))
                    print(e)


    def setup_bspnet_split(self):

        path = "/home/rsulzer/python/BSP-NET-pytorch/samples/"

        classes = os.listdir(path)
        for cl in classes:

            models = os.listdir(os.path.join(path,cl))

            np.savetxt(os.path.join("/home/rsulzer/data/ShapeNet",cl,"bspnet.lst"),models,fmt="%s")



    def get_models(self,splits=["train","val","test"],scan_conf="4",reduce=None,names=None):
        
        self.scan_conf = scan_conf
        self.splits = splits

        if names is not None:
            if not isinstance(names,list):
                names = [names]

        for s in splits:
            class_list = []
            for c in self.classes:
                split_file = os.path.join(self.path, c, s + '.lst')

                with open(split_file, 'r') as f:
                    models = f.read().split('\n')

                if '' in models:
                    models.remove('')

                if reduce:
                    models = models[:reduce]

                for m in models:

                    if names is not None:
                        if m not in names:
                            continue

                    d = {}
                    d["class"] = c
                    d["model"] = m
                    d["path"] = os.path.join(self.path,c,m)
                    d["scan"] = os.path.join(self.path,c,m,"scan",str(scan_conf)+".npz")
                    d["scan_ply"] = os.path.join(self.path,c,m,"scan",str(scan_conf)+".ply")
                    d["eval"] = dict()
                    d["eval"]["occ"] = os.path.join(self.path,c,m,"eval","points.npz")
                    d["eval"]["pointcloud"] = os.path.join(self.path,c,m,"eval","pointcloud.npz")
                    d["mesh"] = os.path.join(self.path,c,m,"mesh","mesh.off")
                    class_list.append(d)

            self.model_dicts[s] = class_list

        if len(splits) == 1:
            self.model_dicts = self.model_dicts[splits[0]]
        return self.model_dicts


    def get_by_id(self,id="0000"):

        if not self.model_dicts:
            print("Call get_models() first")
            return None

        for split in self.model_dicts:
            
            m = next((item for item in self.model_dicts[split] if item["model"] == id), None)

        if m is not None:
            return m
        else:
            print("Model with ID {} does not exist".format(id))
            return None


    def get_random(self, id="0000"):

        if not self.model_dicts:
            print("Call get_models() first")
            return None

        m = next((item for item in self.model_dicts if item["model"] == id), None)

        if m is not None:
            return m
        else:
            print("Model with ID {} does not exist".format(id))
            return None

    def estim_normals(self,method='jet',neighborhood=30,orient=1):
        if(len(self.model_dicts) < 1):
            print("\nERROR: run get_models() first!")
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
            print("\nERROR: run get_models() first!")
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

if __name__ == '__main__':


    ds = ShapeNet()

    ds.get_models(names=["a6d282a360621055614d73f24792753f"])

    ds.setup()





