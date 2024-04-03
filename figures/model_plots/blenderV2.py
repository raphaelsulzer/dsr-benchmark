import os
try:
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=1090, stdoutToServer=True, stderrToServer=True)
except:
    pass
import open3d as o3d
from pyblender import BlenderRender
from importlib import reload
reload(pyblender)
from glob import glob
import math
import numpy as np

class ThisBlenderRender(BlenderRender):

    def __init__(self, out_path, settings, render_and_remove=False, renderer="CYCLES", samples=16):
        super().__init__(settings)


        self.out_path = out_path

        self.apply_color_settings("color")
        # self.apply_global_render_settings(renderer='BLENDER_WORKBENCH', samples=8)
        # self.apply_global_render_settings(renderer='CYCLES', samples=256)
        self.apply_global_render_settings(renderer=renderer, samples=samples,exposure=-0.4)

        self.render_and_remove = render_and_remove

        self.color_ours = [0.055, 0.388, 1.0]


    def render_pc(self, point_file, out_file, point_size=0.0001, with_color=False):

        # object = self.add_point_cloud(point_file, rotation=self.settings["rotation"], point_size=point_size, scale=self.settings["scale"])
        object = self.add_point_cloud_bplt_scatter(point_file, rotation=self.settings["rotation"],
                                                   point_size=point_size, scale=self.settings["scale"],
                                                   gradient_axis=self.settings["gradient_axis"])
        # bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
        if not with_color:
            self.add_color(object, self.color_ours, remove=False)
        self.add_attribute_color_cycles(object)

        self.add_camera(self.cam_file)
        self.add_light(self.light_files,location=self.settings["light_location"])
        # self.add_light(self.light_files)
        self.add_shadow_catcher()

        if self.render_and_remove:
            self.render_still(out_file)
            super().__init__(settings)


    def render_surface(self, file, out_file, color=None, show_edges=False, remove_color=False):

        object = self.add_surface(file, rotation=self.settings["rotation"],scale=self.settings["scale"])

        if color is not None:
            self.add_color(object, color, remove=remove_color)
        self.add_attribute_color_cycles(object)

        if show_edges:
            self.set_freestyle_edge(object, color=(0, 0, 0), thickness=self.settings["edge_thickness"])

        self.add_camera(self.cam_file)
        self.add_light(self.light_files,location=self.settings["light_location"])
        # self.add_light_relative_to_cam(self.settings["light_location"])
        self.add_shadow_catcher(self.settings["rotation"])

        if self.render_and_remove:
            self.render_still(out_file)
            super().__init__(settings)


def get_settings_dict(model):
    model_dict = dict()

    model_dict["default"] = dict()
    model_dict["default"]["rotation"] = [0, 0, 0]
    model_dict["default"]["scale"] = 1
    model_dict["default"]["point_size"] = 0.0008
    model_dict["default"]["light_location"] = [0,0,50]
    model_dict["default"]["gradient_axis"] = 1

    model_dict["templeRing"] = dict()
    model_dict["templeRing"]["rotation"] = [math.pi/2, 0, 0]
    model_dict["templeRing"]["scale"] = 1
    model_dict["templeRing"]["point_size"] = 0.0008
    model_dict["templeRing"]["light_location"] = [0,0,50]
    model_dict["templeRing"]["gradient_axis"] = 1

    model_dict["scan1"] = dict()
    model_dict["scan1"]["rotation"] = [-2.75,-0.15,-1.2]
    model_dict["scan1"]["scale"] = 1
    model_dict["scan1"]["point_size"] = 0.0008
    model_dict["scan1"]["light_location"] = [0,0,50]
    model_dict["scan1"]["gradient_axis"] = 1

    model_dict["Truck"] = dict()
    model_dict["Truck"]["rotation"] = [0, 0, 1.04]
    model_dict["Truck"]["scale"] = 1
    model_dict["Truck"]["point_size"] = 0.0008
    model_dict["Truck"]["light_location"] = [0,0,50]
    model_dict["Truck"]["gradient_axis"] = 1

    model_dict["d18592d9615b01bbbc0909d98a1ff2b4"] = dict()
    model_dict["d18592d9615b01bbbc0909d98a1ff2b4"]["rotation"] = [math.pi/2, 0, 0]
    model_dict["d18592d9615b01bbbc0909d98a1ff2b4"]["scale"] = 1
    model_dict["d18592d9615b01bbbc0909d98a1ff2b4"]["point_size"] = 0.0016
    model_dict["d18592d9615b01bbbc0909d98a1ff2b4"]["light_location"] = [0,0,50]
    model_dict["d18592d9615b01bbbc0909d98a1ff2b4"]["gradient_axis"] = 1

    model_dict["1bea1445065705eb37abdc1aa610476c"] = dict()
    model_dict["1bea1445065705eb37abdc1aa610476c"]["rotation"] = [math.pi/2, 0, 0]
    model_dict["1bea1445065705eb37abdc1aa610476c"]["scale"] = 1
    model_dict["1bea1445065705eb37abdc1aa610476c"]["point_size"] = 0.0016
    model_dict["1bea1445065705eb37abdc1aa610476c"]["light_location"] = [0,0,50]
    model_dict["1bea1445065705eb37abdc1aa610476c"]["gradient_axis"] = 1


    model_dict["0585"] = dict()
    model_dict["0585"]["rotation"] = [0, 0, math.pi/2]
    model_dict["0585"]["scale"] = 1
    model_dict["0585"]["point_size"] = 0.0016
    model_dict["0585"]["light_location"] = [0,0,50]
    model_dict["0585"]["gradient_axis"] = 1

    model_dict["0008"] = dict()
    model_dict["0008"]["rotation"] = [0, 0, math.pi/2]
    model_dict["0008"]["scale"] = 1
    model_dict["0008"]["point_size"] = 0.0016
    model_dict["0008"]["light_location"] = [0,0,50]
    model_dict["0008"]["gradient_axis"] = 1

    model_dict["0470"] = dict()
    model_dict["0470"]["rotation"] = [0, 0, math.pi/2]
    model_dict["0470"]["scale"] = 1
    model_dict["0470"]["point_size"] = 0.0016
    model_dict["0470"]["light_location"] = [0,0,50]
    model_dict["0470"]["gradient_axis"] = 1

    model_dict["daratech"] = dict()
    model_dict["daratech"]["rotation"] = [0, 0, math.pi/4]
    model_dict["daratech"]["scale"] = 1
    model_dict["daratech"]["point_size"] = 0.0016
    model_dict["daratech"]["light_location"] = [0,0,50]
    model_dict["daratech"]["gradient_axis"] = 1

    model_dict["dc"] = dict()
    model_dict["dc"]["rotation"] = [math.pi/2, 0, 0]
    model_dict["dc"]["scale"] = 1
    model_dict["dc"]["point_size"] = 0.0016
    model_dict["dc"]["light_location"] = [0,0,50]
    model_dict["dc"]["gradient_axis"] = 1

    if model in model_dict.keys():
        return model_dict[model]
    else:
        return model_dict["default"]


if __name__ == '__main__':


    inpath = "/home/rsulzer/data/SurveyAndBenchmark/"
    outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/"

    experiment = "real"
    cl = ""
    model = "Truck"
    # model = "scan1"
    # model = "templeRing"

    experiment = "shapenet"
    cl = "02691156"
    model = "d18592d9615b01bbbc0909d98a1ff2b4"

    # experiment = "modelnet"
    # cl = "bed"
    # model = "0585"

    # experiment = "reconbench"
    # cl = "daratech"
    # model = "daratech"

    # experiment = "exp"
    # cl = "02691156"
    # model = "1bea1445065705eb37abdc1aa610476c"

    # experiment = "exp"
    # cl = "dc"
    # model = "dc"
    #
    # experiment = "exp"
    # cl = "table"
    # model = "0008"
    #
    # experiment = "exp"
    # cl = "table"
    # model = "0470"

    settings = get_settings_dict(model)
    rr = True
    # rr = False
    br = ThisBlenderRender(out_path=outpath, settings=settings, render_and_remove=rr, samples=64)
    models = glob(os.path.join(inpath, experiment, cl, model) + "/*.ply")

    # pcd = o3d.io.read_point_cloud(os.path.join(inpath,experiment, cl, model,"input.ply"))
    # diag = np.linalg.norm(pcd.get_min_bound() - pcd.get_max_bound())
    # settings["light_location"] = np.array(settings["light_location"]) * diag

    pcd = o3d.io.read_triangle_mesh(os.path.join(inpath,experiment, cl, model,"gt.ply"))
    diag = np.linalg.norm(pcd.get_min_bound() - pcd.get_max_bound())
    settings["light_location"] = np.array(settings["light_location"]) * diag


    # br.light_files = os.path.join(outpath, experiment, cl, model, "light_file_surface.npz")
    # br.cam_file = os.path.join(outpath, experiment, cl, model, "cam_file_surface.npz")
    # br.set_camera_file_path(br.cam_file)
    # br.set_light_file_path(br.light_files)
    # br.render_pc(os.path.join(inpath,experiment, cl, model,"input.ply"),
    #                   out_file=os.path.join(outpath, experiment,cl, model, "input3000.png"), point_size=settings["point_size"],
    #                   with_color=False)
    # models = []

    for method in models:

        if os.path.basename(method) == "input.ply" or os.path.basename(method) == "full.ply":
            continue

        br.light_files = os.path.join(outpath, experiment,cl, model, "light_file_surface.npz")
        br.cam_file = os.path.join(outpath, experiment, cl,model, "cam_file_surface.npz")

        br.set_camera_file_path(br.cam_file)
        br.set_light_file_path(br.light_files)
        br.render_surface(method,
                          out_file=os.path.join(outpath, experiment,cl, model,"{}.png".format(os.path.basename(method).split(".")[0])),
                          color=br.color_ours, show_edges=False, remove_color=False)
