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