import os, sys, subprocess
import numpy as np
from tqdm import tqdm
from dsrb import DefaultDataset
from dsrb.set_paths import set_paths
set_paths("/home/rsulzer/data","/home/rsulzer/cpp")

class ShapeNet(DefaultDataset):

    def __init__(self,
                 classes=[]):
        super().__init__()

        self.path = os.path.join(self.path,"ShapeNet")
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

    def clear(self):
        self.model_dicts = {}

    def setup(self):


        for split in self.model_dicts.keys():

            for model in tqdm(self.model_dicts[split]):

                # print("Process model {}/{}".format(model["class"], model["model"]))
                
                ### Move pointcloud.npz and points.npz to eval/
                try:
                    os.makedirs(os.path.join(self.path,model["class"],model["model"],"eval"))

                    infile = os.path.join(self.path,model["class"],model["model"],"pointcloud.npz")
                    outfile = os.path.join(self.path,model["class"],model["model"],"eval","pointcloud.npz")
                    os.rename(infile,outfile)

                    infile = os.path.join(self.path,model["class"],model["model"],"points.npz")
                    outfile = os.path.join(self.path,model["class"],model["model"],"eval","points.npz")
                    os.rename(infile,outfile)


                except Exception as e:
                    # raise e
                    print("Problem with model {}/{}".format(model["class"],model["model"]))
                    print(e)
                
                ### Move models from classX/4_watertight_scaled/modelX.off to classX/modelX/mesh/mesh.off
                try:
                    infile = os.path.join(self.path,model["class"],"4_watertight_scaled","{}.off".format(model["model"]))
                    if not os.path.isfile(infile):
                        continue
                    outfile = os.path.join(self.path,model["class"],model["model"],"mesh","mesh.off")
                    if not os.path.isdir(os.path.join(self.path,model["class"],model["model"])):
                        continue
                    os.makedirs(os.path.join(self.path,model["class"],model["model"],"mesh"))
                    os.rename(infile,outfile)
                except Exception as e:
                    raise e
                    print("Problem with model {}/{}".format(model["class"],model["model"]))
                    print(e)


    def make_bspnet_eval_file(self,models,outfile="/home/rsulzer/python/BSP-NET-original/evaluation/myeval.txt"):

        mlist = []

        for model in models:

            mlist.append("{}/{}".format(model["class"],model["model"]))


        np.savetxt(outfile,mlist,fmt="%s")




    def make_split(self,input_split="train",output_split="debug",reduce=3):

        for c in self.classes:

            split_file = os.path.join(self.path, c,'{}.lst'.format(input_split))
            with open(split_file, 'r') as f:
                models = f.read().split('\n')

            models = models[:reduce]
            np.savetxt(os.path.join(self.path, c,'{}.lst'.format(output_split)),models,fmt="%s")


    def get_models(self,splits=["train","val","test"],scan_configuration="4",reduce=None,names=None):
        
        self.scan_conf = scan_configuration
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
                    d["scan"] = os.path.join(self.path,c,m,"scan",str(scan_configuration)+".npz")
                    d["scan_ply"] = os.path.join(self.path,c,m,"scan",str(scan_configuration)+".ply")
                    d["eval"] = dict()
                    d["eval"]["occ"] = os.path.join(self.path,c,m,"eval","points.npz")
                    d["eval"]["pointcloud"] = os.path.join(self.path,c,m,"eval","pointcloud.npz")
                    d["mesh"] = os.path.join(self.path,c,m,"mesh","mesh.off")

                    d["output"] = {}
                    d["output"]["surface"] = os.path.join(self.path,c, m, "{}", "surface.ply")
                    d["output"]["surface_simplified"] = os.path.join(self.path,c, m, "{}", "surface_simplified.obj")
                    d["output"]["partition"] = os.path.join(self.path, c,m, "{}", "partition.ply")
                    d["output"]["partition_pickle"] = os.path.join(self.path,c, m, "{}", "partition")
                    d["output"]["in_cells"] = os.path.join(self.path,c, m, "{}", "in_cells.ply")
                    d["output"]["settings"] = os.path.join(self.path,c, m, "{}", "settings.yaml")

                    class_list.append(d)

            self.model_dicts[s] = class_list

        # if len(splits) == 1:
        #     self.model_dicts = self.model_dicts[splits[0]]
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



if __name__ == '__main__':


    ds = ShapeNet()

    # ds.get_models(names=["a691eee4545ce2fade94aad0562ac2e"])
    
    split = "bspnet_test10"
    ds.get_models(splits=[split])
    # ds.model_dicts = ds.model_dicts[split]
    #
    # ds.make_eval()

    # ds.make_split()
    ds.make_bspnet_eval_file(models=ds.model_dicts[split],outfile="/home/rsulzer/python/BSP-NET-original/evaluation/{}.txt".format(split))
    # ds.make_split(input_split="bspnet_test",output_split="bspnet_test10",reduce=10)






