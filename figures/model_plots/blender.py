import math

import pydevd_pycharm
#pydevd_pycharm.settrace('localhost', port=1090, stdoutToServer=True, stderrToServer=True)
import os

import bpy
import numpy as np
from pathlib import Path

import blender_plots as bplt
from scipy.spatial.transform import Rotation

from glob import glob
from pyntcloud import PyntCloud

import sys
sys.path.append("/home/rsulzer/python/dsr-benchmark/datasets")
from reconstructed_dataset import learning_dataset, figures_dataset

from mathutils import Vector, Matrix

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
class MplColorHelper:

    """
    great code from here: https://stackoverflow.com/a/26109298
    which given two values start_val, stop_val makes a color gradient cmap in between.
    then an array passed to cmap.get_rgb gives the corresponding colors in rgb
    values of the array outside [start_val, stop_val] are simply assigned the endoint of the color gradient
    works great in combination with start_val = np.percentile(array,5) and stop_val = np.percentile(array,95)
    """
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

class RenderReal:
    
    
    def __init__(self):

        self.remove_model = True

        collection = bpy.data.collections.get("Collection")
        if collection:
            for obj in collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)

            bpy.data.collections.remove(collection)
        
        ## if collection already exists, remove it and all its objects
        collection = bpy.data.collections.get("MyCollection")
        if collection:
            for obj in collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
                
            bpy.data.collections.remove(collection)

        ## New Collection
        self.coll = bpy.data.collections.new("MyCollection")
        
        ## Add collection to scene collection
        bpy.context.scene.collection.children.link(self.coll)
        
        
        self.scene_coll = bpy.context.scene.collection
        
        
    def add_cam(self,location,orientation,resolution=(1024,1024)):
        ## make camera and link it
        camera_data = bpy.data.cameras.new("Camera")
        self.camera = bpy.data.objects.new("Camera", camera_data)
        # get camera location with C.scene.camera.location
        self.camera.location = location
        # get camera angle with C.scene.camera.matrix_world.to_euler()
        self.camera.rotation_euler = orientation
        self.coll.objects.link(self.camera)
        # change camera size
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]
        
        bpy.context.scene.camera = self.camera



    def add_light(self,location,energy=15):

        # create light datablock, set attributes
        light_data = bpy.data.lights.new(name="light", type='POINT')
        light_data.energy = energy

        # create new object with our light datablock
        self.light = bpy.data.objects.new(name="light", object_data=light_data)

        # link light object
        self.coll.objects.link(self.light)

        # make it active
        bpy.context.view_layer.objects.active = self.light

        # change location
        self.light.location = location


    def apply_global_render_settings(self,renderer='BLENDER_WORKBENCH',exposure=0.5,gamma=1.5,samples=4):

        ## some color and lighting settings
        bpy.context.scene.view_settings.view_transform = 'Standard'
        # bpy.context.space_data.shading.type = 'RENDERED'

        ## transparent background
        bpy.context.scene.render.film_transparent = False
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'

        ## use GPU for rendering
        bpy.context.scene.render.engine = renderer
        if renderer == 'CYCLES':
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # or "OPENCL"
            bpy.context.scene.cycles.device = "GPU"
            ## basically the higher the sharper the render
            bpy.context.scene.cycles.samples = samples
        else:
            bpy.context.scene.cycles.device = "CPU"
            bpy.data.scenes['Scene'].display.render_aa = str(samples)




    def render_mesh(self,file, out=None):

        ## get file and put it in scene collection
        bpy.ops.import_mesh.ply(filepath=file)
        obj = bpy.context.active_object

        if self.rotate:
            # obj.matrix_world = Matrix.Rotation(-math.pi/2,4,"Z") @ Matrix.Rotation(math.pi/2,4,"X")
            obj.matrix_world = Matrix.Rotation(self.rotate[2],4,"Z") \
                                @ Matrix.Rotation(self.rotate[1], 4, "Y") \
                                @ Matrix.Rotation(self.rotate[0],4,"X")


        # remove it from scene collection
        self.scene_coll.objects.unlink(obj)
        self.coll.objects.link(obj)

        outfile = out if out else str(Path(file).with_suffix(".png"))
        bpy.context.scene.render.filepath = outfile
        # bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
        bpy.ops.render.render(write_still=True)

        print("Mesh render saved to ",outfile)

        if self.remove_model:
            self.coll.objects.unlink(obj)

    def color_along_axis(self,points,axis=1):
        sign = np.sign(axis)
        axis = np.abs(axis)-1
        cmap = 'jet'
        return MplColorHelper(cmap, points[:,axis].min(), points[:,axis].max()).get_rgb(sign*points[:,axis])
        # cols=MplColorHelper(cmap, accuracy.min(), accuracy.max()).get_rgb(accuracy)

    def render_pc(self, file, out=None):

        """this is the one to use for point cloud rendering"""
        if os.path.splitext(file)[1] == ".ply":
            pcd = PyntCloud.from_file(file)
            points = pcd.points[["x","y","z"]].values
            normals = pcd.points[["nx","ny","nz"]].values
            if "red" in pcd.points.keys():
                colors = pcd.points[["red","green","blue"]].values
            else:
                colors=None
        elif os.path.splitext(file)[1] == ".npz":
            data = np.load(file)
            points = data["points"]
            normals = data["normals"]
            if "colors" in data.keys():
                colors = data["colors"]
            else:
                colors = None
            normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        else:
            print("ERROR: {} is not a supported file ending for point cloud rendering".format(os.path.splitext(file)[1]))

        if self.rotate:
            points = points @ Matrix.Rotation(-self.rotate[0], 3, "X") @ Matrix.Rotation(-self.rotate[1], 3, "Y") @ Matrix.Rotation(-self.rotate[2], 3, "Z")
            normals = normals @ Matrix.Rotation(-self.rotate[0], 3, "X") @ Matrix.Rotation(-self.rotate[1], 3, "Y") @ Matrix.Rotation(-self.rotate[2], 3, "Z")

        # normals to rotmat
        marker_default_orient = [0, 0, 1]
        angles = np.arccos(np.dot(normals, marker_default_orient))
        cross = np.cross(normals, marker_default_orient)
        cross = cross / np.linalg.norm(cross, axis=1)[:, np.newaxis]
        quat = np.array([cross[:, 0], cross[:, 1], cross[:, 2], angles[:]]).transpose()
        rots = Rotation.from_quat(quat).as_matrix()
        # has to be rotated 180 degrees when using cone marker
        I = np.identity(3)
        I[0, 0] = -1
        I[2, 2] = -1
        rots = rots @ I

        ## colors
        colors = self.color_along_axis(points=points,axis=self.color_axis)[:,:3]

        pc = "pc"
        bplt.Scatter(points,
                     color=colors,
                     marker_type="cones",
                     radius_bottom=1,
                     radius_top=3,
                     marker_scale=[self.marker_scale,self.marker_scale,self.marker_scale/3],
                     marker_rotation=rots,
                     randomize_rotation=False,
                     name=pc)
        # bplt.Scatter(points,
        #              color=colors,
        #              marker_type="uv_spheres",
        #              marker_scale=[self.marker_scale,self.marker_scale,self.marker_scale],
        #              marker_rotation=rots,
        #              randomize_rotation=False,
        #              name=pc)

        obj = bpy.context.scene.objects[pc]
        self.scene_coll.objects.unlink(obj)
        self.coll.objects.link(obj)

        outfile = out if out else str(Path(file).with_suffix(".png"))
        bpy.context.scene.render.filepath = outfile
        bpy.ops.render.render(write_still=True)
        print("Renderer to", outfile)

        if self.remove_model:
            self.coll.objects.unlink(obj)

    def render_settings(self,model,mode):

        self.rotate = None

        match mode:
            case "normal":
                bpy.data.scenes['Scene'].display.shading.light = 'MATCAP'
                bpy.data.scenes['Scene'].display.shading.studio_light = 'check_normal+y.exr'
                bpy.data.scenes['Scene'].display.shading.color_type = 'OBJECT'
            case "single_color":
                bpy.data.scenes['Scene'].display.shading.light = 'STUDIO'
                bpy.data.scenes['Scene'].display.shading.studio_light = 'rim.sl'
                bpy.data.scenes['Scene'].display.shading.single_color = (0.8, 0.183968, 0)
                bpy.data.scenes['Scene'].display.shading.color_type = 'SINGLE'
            case "axis":
                bpy.data.scenes['Scene'].display.shading.light = 'STUDIO'
                bpy.data.scenes['Scene'].display.shading.studio_light = 'rim.sl'
                bpy.data.scenes['Scene'].display.shading.color_type = 'VERTEX'
            case "color":
                bpy.data.scenes['Scene'].display.shading.light = 'STUDIO'
                bpy.data.scenes['Scene'].display.shading.studio_light = 'rim.sl'
                bpy.data.scenes['Scene'].display.shading.color_type = 'VERTEX'

        self.color_axis=2
        match model:
            case "dc/dc":
                self.color_axis=1
                self.rotate = [math.pi/2.2,0,0]
                self.marker_scale = 0.002
                # self.marker_scale = 0.005

                self.light_location = (0.6792212128639221, -1.417323112487793, 0.2476280927658081)
                self.camera_location = self.light_location
                self.camera_orientation = (1.3872339725494385, 0, 0.4199945032596588)
                self.resolution = (1024, 768)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "daratech/daratech":
                self.marker_scale = 0.0025
                # self.marker_scale = 0.005

                self.light_location = (1.2819530963897705, -0.40413326025009155, 0.46248066425323486)
                self.camera_location = self.light_location
                self.camera_orientation = (1.1498682498931885, 0, 1.2926580905914307)
                self.resolution = (1024, 768)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            # shapenet
            case "02958343/1a0c91c02ef35fbe68f60a737d94994a":
                self.color_axis = 1
                self.rotate = [-math.pi / 2, 0, math.pi / 2]
                self.marker_scale = 0.0025
                # self.marker_scale = 0.005

                self.light_location = (1.0699280500411987, 0.9045863747596741, -0.5949070453643799)
                self.camera_location = self.light_location
                self.camera_orientation = (-2.0126681327819824, 0, -0.8994788527488708)
                self.resolution = (1024, 768)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "02958343/1d343a64b4789983c10e9d4ee4bae4f4":
                self.color_axis = -2
                self.rotate = [-math.pi / 2, 0, math.pi / 2]
                self.marker_scale = 0.0025
                # self.marker_scale = 0.005

                self.light_location = (1.0699280500411987, 0.9045863747596741, -0.5949070453643799)
                self.camera_location = self.light_location
                self.camera_orientation = (-2.0126681327819824, 0, -0.8994788527488708)
                self.resolution = (1024, 768)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "02691156/d18592d9615b01bbbc0909d98a1ff2b4":
                self.rotate = [math.pi/2,0,-math.pi/2]
                self.marker_scale = 0.0025
                # self.marker_scale = 0.005

                self.light_location = (0.7351253628730774, -0.8849141001701355, 0.6502181887626648)
                self.camera_location = self.light_location
                self.camera_orientation = (1.0102378129959106, 0, 0.6573612093925476)
                self.resolution = (1024,768)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "02691156/1bea1445065705eb37abdc1aa610476c":
                self.rotate = [math.pi/2,0,-math.pi/2]
                self.marker_scale = 0.0025
                # self.marker_scale = 0.005

                self.light_location = (0.6752992868423462, -0.9008161425590515, 0.610461413860321)
                self.camera_location = self.light_location
                self.camera_orientation = (1.010238528251648, 0, 0.6573624014854431)
                self.resolution = (1024,768)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "table/0008":
                self.marker_scale = 0.0025
                # get with C.scene.camera.location
                self.light_location = (0.9463180899620056, 0.8051369786262512, 0.692879855632782)
                self.camera_location = self.light_location
                self.camera_orientation = (1.0241998434066772, 0, 2.2491040229797363)
                self.resolution = (1024, 768)
                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "table/0490":
                self.color_axis = 1
                self.marker_scale = 0.0025
                # get with C.scene.camera.location
                self.light_location = (0.8938982486724854, -1.7844570875167847, 1.3868521451950073)
                self.camera_location = self.light_location
                self.camera_orientation = (0.9474075436592102, 0, 0.4758685827255249)
                self.resolution = (1024, 768)
                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "bed/0585":
                self.marker_scale = 0.0025
                # get with C.scene.camera.location
                self.light_location = (1.193621039390564, -0.8836174011230469, 0.8282204866409302)
                self.camera_location = self.light_location
                self.camera_orientation = (0.9823122620582581, 0, 0.8947256803512573)
                self.resolution = (1024, 768)
                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "Ignatius":
                self.marker_scale = 0.001
                self.light_location = (1.2972263097763062, -0.8929003477096558, -0.06501338630914688)
                self.camera_location = self.light_location
                self.camera_orientation = (1.5757218599319458, 0, 0.9924624562263489)
                self.resolution = (768,1024)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "Truck":
                self.marker_scale = 0.001
                self.light_location = (0.8238871097564697, -0.6957366466522217, -0.00797953549772501)
                self.camera_location = self.light_location
                self.camera_orientation = (1.5757216215133667, 0, 0.9924633502960205)
                self.resolution = (1024,768)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "real/Truck":
                self.marker_scale = 0.001
                self.light_location = (0.8238871097564697, -0.6957366466522217, -0.00797953549772501)
                self.camera_location = self.light_location
                self.camera_orientation = (1.5757216215133667, 0, 0.9924633502960205)
                self.resolution = (1024,768)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "scan1":
                self.rotate = [math.pi/7.5,math.pi/28,0]
                self.marker_scale = 0.001
                self.light_location = (1.2899032831192017, -0.26632916927337646, -0.4500363767147064)
                self.camera_location = self.light_location
                self.camera_orientation = (-1.9657785892486572, 0, -1.788089394569397)
                self.resolution = (1024, 768)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "scan6":
                self.rotate = [0,0,0]
                self.marker_scale = 0.001
                self.light_location = (1.2812105417251587, -0.17517916858196259, -0.4000299274921417)
                self.camera_location = self.light_location
                self.camera_orientation = (-1.963143229484558, -0.45847922563552856, -1.5502852201461792)
                self.resolution = (1024, 768)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'
            case "templeRing":
                self.rotate = [math.pi/2,0,math.pi/4]
                self.marker_scale = 0.001
                self.light_location = (1.4406687021255493, -0.30377063155174255, 0.43409839272499084)
                self.camera_location = self.light_location
                self.camera_orientation = (1.2529363632202148, 0, 1.337907314300537)
                self.resolution = (768,1024)

                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'

            case "Ignatius/Ignatius":
                self.rotate = [0, 0, 0]
                self.marker_scale = 0.001
                self.light_location = (0.20547041296958923, -0.4521239399909973, 0.4659009575843811)
                self.camera_location = self.light_location
                self.camera_orientation = (1.3174161911010742, 0, 0.47584229707717896)

                self.resolution = (1024, 1024)
                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'

            case _:
                print("WARNING: apply default render settings")
                self.rotate = False
                self.marker_scale = 0.0025
                # get with C.scene.camera.location
                self.light_location = (1.2972263097763062, -0.8929003477096558, -0.06501338630914688)
                self.camera_location = self.light_location
                # get with C.scene.camera.matrix_world.to_euler()
                self.camera_orientation = (1.5757218599319458, 0, 0.9924624562263489)

                # set in blender in "output properties" -> format -> resolution
                self.resolution = (768,1024)
                bpy.context.scene.view_settings.exposure = 0.4
                bpy.context.scene.view_settings.gamma = 1.6
                bpy.context.scene.view_settings.look = 'Medium High Contrast'



if __name__ == '__main__':


    # exp = {"real":{"models":["Ignatius","scan1","scan6","templeRing","Truck"]}}
    exp = {"real":{"models":["scan6","templeRing","Truck"]}}
    exp.update(learning_dataset)
    exp.update(figures_dataset)

    # e = "shapenet10000"
    rr = RenderReal()
    rr.apply_global_render_settings(renderer='BLENDER_WORKBENCH',samples=32)
    rr.remove_model = True

    # experiments = ["shapenet","shapenet10000","shapenet3000","reconbench","modelnet"]
    # experiments = ["Ignatius"]
    for e in experiments:
        models = exp[e]["models"]
        outpath = os.path.join("/home/rsulzer/overleaf/SurveyAndBenchmark/figures_new/", e)
        for model in models:
            os.makedirs(os.path.join(outpath,model),exist_ok=True)

            # path = "/home/rsulzer/data/real_out/paper/models"
            path = os.path.join("/home/rsulzer/data/SurveyAndBenchmarK/",e)
            inpath = os.path.join(path,model)
            methods = glob(inpath+"/*.ply")

            # need to remove the light and camera once I include this into model loop
            rr.render_settings(model, "color")

            rr.add_cam(rr.camera_location, rr.camera_orientation, rr.resolution)
            rr.add_light(rr.light_location, energy=100)

            for me in methods:
                # continue
                rr.render_settings(model,me)
                method = os.path.basename(me).split('.')[0]
                print("Process {}/{}".format(model, method))
                if method == 'input' or method == 'full':
                    print("pass")
                    pass
                else:
                    outfile = os.path.join(outpath, model, method+".jpg")
                    # rr.render_mesh(me,outfile)
                    rr.render_pc(me,outfile) # pass a color

            try:
                method = "input"
                rr.render_settings(model, "axis")
                # rr.render_settings(model, method)
                outfile = os.path.join(outpath, model,method+".jpg")
                rr.render_pc(os.path.join(inpath, method+".ply"),outfile)
            except Exception as exc:
                print(exc)

            try:
                method = "input3000"
                rr.render_settings(model, "axis")
                # rr.render_settings(model, method)
                outfile = os.path.join(outpath, model,method+".jpg")
                rr.render_pc(os.path.join(inpath, method+".ply"),outfile)
            except Exception as exc:
                print(exc)
            try:
                method = "input10000"
                rr.render_settings(model, "axis")
                # rr.render_settings(model, method)
                outfile = os.path.join(outpath, model,method+".jpg")
                rr.render_pc(os.path.join(inpath, method+".ply"),outfile)
            except Exception as exc:
                print(exc)

            try:
                method = "mesh"
                rr.render_settings(model, "reconstruction")
                # rr.render_settings(model, method)
                outfile = os.path.join(outpath, model,method+".jpg")
                rr.render_mesh(os.path.join(inpath, method+".ply"),outfile)
            except Exception as exc:
                print(exc)

            if rr.remove_model:
                bpy.data.objects.remove(rr.camera, do_unlink=True)
                bpy.data.objects.remove(rr.light, do_unlink=True)


    # bpy.data.collections.remove(rr.coll)


    # rr.render_settings("ignatius", "output")
    # rr.render_full("/home/rsulzer/data/real_out/paper/models/Ignatius/full.ply")







