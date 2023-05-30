figures_dataset = {}

d={}
d["models"]=["Ignatius","scan1","scan6","templeRing","Truck"]
figures_dataset["real"] = d

d={}
# d["models"]=["Ignatius03","Ignatius04","Ignatius08","scan_1400005","berger_sensors","mvs","mesh","uniform_250000"]
d["models"]=["Ignatius"]
figures_dataset["Ignatius"] = d

d={}
d["models"]=["dc/dc",
             # "02958343/1a0c91c02ef35fbe68f60a737d94994a",
             # "02958343/1d343a64b4789983c10e9d4ee4bae4f4",
             "02691156/1bea1445065705eb37abdc1aa610476c",
                "02691156/d18592d9615b01bbbc0909d98a1ff2b4",
             "real/Truck",
             "table/0008",
             "table/0490"]
figures_dataset["exp"] = d




learning_dataset = {}

d = {}
d["models"] = ["02691156/d18592d9615b01bbbc0909d98a1ff2b4"]
d["train_dataset"] = "ShapeNet"
d["test_dataset"] = "ShapeNet"
d["scan"] = "4"
learning_dataset["shapenet3000"] = d

d={}
d["models"] = ["02691156/d18592d9615b01bbbc0909d98a1ff2b4"]
d["train_dataset"] = "ShapeNet"
d["test_dataset"] = "ShapeNet"
d["scan"] = "6"
learning_dataset["shapenet10000"] = d

d={}
d["models"] = ["bed/0585"]
d["train_dataset"] = "ShapeNet"
d["test_dataset"] = "ModelNet10"
d["scan"] = "43"
learning_dataset["modelnet"] = d

d={}
d["models"] = ["02691156/d18592d9615b01bbbc0909d98a1ff2b4"]
d["train_dataset"] = "ModelNet10"
d["test_dataset"] = "ShapeNet"
d["scan"] = "4"
learning_dataset["shapenet"] = d

d={}
d["models"] = ["daratech/daratech"]
d["train_dataset"] = "ShapeNet"
d["test_dataset"] = "reconbench"
d["scan"] = "4"
learning_dataset["reconbench"] = d

