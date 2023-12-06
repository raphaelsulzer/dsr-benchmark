import os
from dsrb import DefaultDataset

class ModelNet10(DefaultDataset):

    def __init__(self,
                 classes=[]):
        super().__init__()

        self.path = os.path.join(self.path,"ShapeNet")
        self.classes = classes
        self.model_dicts = {}

        with open(os.path.join(self.path, "classes.lst"), 'r') as f:
            categories = f.read().split('\n')
        if '' in categories:
            categories.remove('')
        self.classes = categories

    def get_models(self,splits=["train","test"],scan_configuration="43",reduce=None):


        self.scan_conf=scan_configuration
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
                    d["scan"] = os.path.join(self.path,c,m,"scan",self.scan_conf+".npz")
                    d["scan_ply"] = os.path.join(self.path,c,m,"scan",self.scan_conf+".ply")
                    d["occ"] = os.path.join(self.path,c,m,"eval","points.npz")
                    d["pointcloud"] = os.path.join(self.path,c,m,"eval","pointcloud.npz")
                    d["mesh"] = os.path.join(self.path+"_watertight",c+"_"+m+".off")
                    class_list.append(d)

            self.model_dicts[s] = class_list

        return self.model_dicts
