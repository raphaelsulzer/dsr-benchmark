## learning
learning_methods = []

d=dict()
d["name"] = "ConvONet2D"
d["cite"] = "\cite{Peng2020}"
d["path"] = "conv_onet/2d/{}/meshes"
d["out_type"] = ".off"
learning_methods.append(d)

d=dict()
d["name"] = "ConvONet3D"
d["cite"] = "\cite{Peng2020}"
d["path"] = "conv_onet/3d/{}/meshes"
d["out_type"] = ".off"
learning_methods.append(d)

d=dict()
d["name"] = "SAP"
d["cite"] = "\cite{Peng2021SAP}"
d["path"] = "sap/tr/{}/meshes"
d["out_type"] = ".off"
learning_methods.append(d)

# d=dict()
# d["name"] = "LIG"
# d["cite"] = "\cite{lig}"
# d["path"] = "lig/{}"
# learning_methods.append(d)



d=dict()
d["name"] = "POCO"
d["cite"] = "\cite{boulch2022poco}"
d["path"] = "poco/tr/{}/meshes"
d["out_type"] = ".ply"
learning_methods.append(d)

d=dict()
d["name"] = "DGNN"
d["cite"] = "\cite{dgnn}"
d["path"] = "dgnn/tr/{}"
d["out_type"] = ".ply"
learning_methods.append(d)

d=dict()
d["name"] = "SPSR"
d["cite"] = "\cite{screened_poisson}"
d["path"] = "poisson/{}"
d["out_type"] = ".ply"
learning_methods.append(d)


d=dict()
# d["name"] = "Labatut~\etal"
# d["name"] = "Labatut $\it{et~al.}$"
d["name"] = "Labatut"
d["cite"] = "\cite{Labatut2009a}"
d["path"] = "labatut/{}"
# d["out_type"] = "_rt_2.5.ply"
d["out_type"] = "_rt_2.5.ply"
learning_methods.append(d)



## optim
optim_methods = []

d=dict()
d["name"] = "IGR"
d["cite"] = "\cite{Gropp2020}"
d["path"] = "igr"
d["out_type"] = ".ply"
optim_methods.append(d)

d=dict()
d["name"] = "LIG"
d["cite"] = "\cite{lig}"
d["path"] = "lig"
d["out_type"] = ".ply"
optim_methods.append(d)

d=dict()
d["name"] = "P2M"
d["cite"] = "\cite{point2mesh}"
d["path"] = "p2m/poisson"
d["out_type"] = ".ply"
optim_methods.append(d)


d=dict()
d["name"] = "SAP"
d["cite"] = "\cite{Peng2021SAP}"
d["path"] = "sap"
d["out_type"] = ".ply"
optim_methods.append(d)

# d=dict()
# d["name"] = "DSE"
# d["cite"] = "\cite{rakotosaona2021dse}"
# d["path"] = "dse"
# optim_methods.append(d)

d=dict()
d["name"] = "SPSR"
d["cite"] = "\cite{screened_poisson}"
d["path"] = "poisson"
d["out_type"] = ".ply"
optim_methods.append(d)

d=dict()
d["name"] = "Labatut~\etal"
d["cite"] = "\cite{Labatut2009a}"
d["path"] = "labatut"
d["out_type"] = "_rt_5.0.ply"
optim_methods.append(d)
