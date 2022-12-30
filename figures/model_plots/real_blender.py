#import pydevd_pycharm
#pydevd_pycharm.settrace('localhost', port=1090, stdoutToServer=True, stderrToServer=True)
import os

import bpy
import numpy as np
from pathlib import Path
from mathutils import Vector

import blender_plots as bplt
from scipy.spatial.transform import Rotation

from glob import glob

class RenderReal:
    
    
    def __init__(self):

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
        
        
    def add_cam(self,location,orientation):
        ## make camera and link it
        camera_data = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", camera_data)
        # get camera location with C.scene.camera.location
        camera.location = location
        # get camera angle with C.scene.camera.matrix_world.to_euler()
        camera.rotation_euler = orientation
        self.coll.objects.link(camera)
        # change camera size
        bpy.context.scene.render.resolution_x = 1024
        bpy.context.scene.render.resolution_y = 1024
        
        bpy.context.scene.camera = camera



    def add_light(self,location,energy=15):

        # create light datablock, set attributes
        light_data = bpy.data.lights.new(name="light", type='POINT')
        light_data.energy = energy

        # create new object with our light datablock
        light_object = bpy.data.objects.new(name="light", object_data=light_data)

        # link light object
        self.coll.objects.link(light_object)

        # make it active
        bpy.context.view_layer.objects.active = light_object

        # change location
        light_object.location = location

        return light_object


    def apply_render_settings(self,renderer='BLENDER_WORKBENCH',exposure=0.5,gamma=1.5,samples=4):

        ## some color and lighting settings
        bpy.context.scene.view_settings.view_transform = 'Standard'
        # bpy.context.space_data.shading.type = 'RENDERED'

        ## transparent background
        bpy.context.scene.render.film_transparent = True
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




    def render_mesh(self,file):

        ## get file and put it in scene collection
        bpy.ops.import_mesh.ply(filepath=file)
        obj = bpy.context.active_object

        # remove it from scene collection
        self.scene_coll.objects.unlink(obj)
        self.coll.objects.link(obj)


        bpy.context.scene.render.filepath = str(Path(file).with_suffix(".png"))
        # bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
        bpy.ops.render.render(write_still=True)

        print("Mesh render saved to ",str(Path(file).with_suffix(".png")))

        self.coll.objects.unlink(obj)


    def render_bplt(self, file):
        
        """this is the one to use for point cloud rendering"""
        data = np.load(file)

        # normals to rotmat
        marker_default_orient = [0,0,1]
        angles = np.arccos(np.dot(data["normals"],marker_default_orient))
        cross = np.cross(data["normals"],marker_default_orient)

        cross = cross / np.linalg.norm(cross, axis=1)[:, np.newaxis]
        quat = np.array([cross[:, 0], cross[:, 1], cross[:, 2], angles[:]]).transpose()
        rots = Rotation.from_quat(quat).as_matrix()

        # has to be rotated 180 degrees when using cone marker
        I=np.identity(3)
        I[0,0]=-1
        I[2,2]=-1
        rots=rots@I


        ## colors
        colors = data["colors"]
        colors = data["normals"][:,[1,2,0]]

        pc="pc"
        bplt.Scatter(data["points"],
                                color=colors,
                                marker_type="cones",
                                radius_bottom=1,
                                radius_top=3,
                                marker_scale=[0.001]*3,
                                marker_rotation=rots,
                                randomize_rotation=False,
                                name=pc)

        obj = bpy.context.scene.objects[pc]
        self.scene_coll.objects.unlink(obj)
        self.coll.objects.link(obj)



        bpy.context.scene.render.filepath = str(Path(file).with_suffix(".png"))
        bpy.ops.render.render(write_still=True)
        print("Renderer to", str(Path(file).with_suffix(".png")))

        self.coll.objects.unlink(obj)


def render_settings(model,mode):

    match model:

        case "ignatius":
            bpy.context.scene.view_settings.exposure = 0.4
            bpy.context.scene.view_settings.gamma = 1.6
            bpy.context.scene.view_settings.look = 'Medium High Contrast'
            bpy.data.scenes['Scene'].display.shading.color_type = 'OBJECT'
            match mode:

                case "pc":
                    bpy.data.scenes['Scene'].display.shading.light = 'MATCAP'
                    bpy.data.scenes['Scene'].display.shading.studio_light = 'check_normal+y.exr'
                case "mesh":
                    bpy.data.scenes['Scene'].display.shading.light = 'STUDIO'
                    bpy.data.scenes['Scene'].display.shading.studio_light = 'rim.sl'


if __name__ == '__main__':
    
    rr = RenderReal()
    location=(1.2647533416748047, -0.9426126480102539, -0.06860939413309097)
    rr.add_cam(location,(1.5757203102111816, 0, 0.9924602508544922))
    rr.add_light(location,energy=100)


    rr.apply_render_settings(renderer='BLENDER_WORKBENCH',samples=8)


    path = "/home/rsulzer/data/real_out/paper/models/Ignatius/"
    methods = glob(path+"*.ply")

    for me in methods:
        if os.path.basename(me).split('.')[0] == 'input':
            pass
        else:
            render_settings("ignatius", "mesh")
            rr.render_mesh(me)


    render_settings("ignatius","pc")
    rr.render_bplt("/home/rsulzer/data/real_out/paper/models/Ignatius/input.npz")







