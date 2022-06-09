import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))


class ModelNet10:

    def __init__(self,path="/mnt/raphael/ModelNet10"):


        self.path = path
        self.model_dicts = {}

        with open(os.path.join(self.path, "classes.lst"), 'r') as f:
            categories = f.read().split('\n')
        if '' in categories:
            categories.remove('')
        self.classes = categories

    def getModels(self,splits=["train","test"],scan="43",classes=None,reduce=None):

        if classes is not None:
            self.classes = classes


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
                    d["scan"] = os.path.join(self.path,c,m,"scan",scan+".npz")
                    d["occ"] = os.path.join(self.path,c,m,"eval","points.npz")
                    d["pointcloud"] = os.path.join(self.path,c,m,"eval","pointcloud.npz")
                    d["mesh"] = os.path.join(self.path+"_watertight",c+"_"+m+".off")
                    class_list.append(d)

            self.model_dicts[s] = class_list

        return self.model_dicts

    def getOne(self,splits=["train","test"],scan="43",classes=None,id=0):

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
                d["scan"] = os.path.join(self.path, c, "scan", scan, m, "scan.npz")
                d["occ"] = os.path.join(self.path, c, "eval", m, "points.npz")
                d["pointcloud"] = os.path.join(self.path, c, "eval", m, "pointcloud.npz")
                d["mesh"] = os.path.join(self.path + "_watertight", c + "_" + m + ".off")
                class_list.append(d)

            self.model_dicts[s] = class_list

        return self.model_dicts


        # m = next((item for item in self.model_dicts if item["model"] == id), None)

