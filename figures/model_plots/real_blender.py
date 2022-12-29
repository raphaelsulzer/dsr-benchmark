#import pydevd_pycharm
#pydevd_pycharm.settrace('localhost', port=1090, stdoutToServer=True, stderrToServer=True)

import bpy
import bmesh
import numpy as np
from pathlib import Path
from mathutils import Vector

from photogrammetry_importer.file_handlers.point_data_file_handler import PointDataFileHandler
from photogrammetry_importer.importers.point_utility import add_points_as_object_with_particle_system
from photogrammetry_importer.importers.point_utility import add_points_as_mesh_vertices

import blender_plots as bplt

from scipy.spatial.transform import Rotation
import math

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
        
        
    def make_cam(self):
        ## make camera and link it
        camera_data = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", camera_data)
        # get camera location with C.scene.camera.location
        camera.location = (0.3701401352882385, -1.5607348680496216, 0.0101625472307205)
        # get camera angle with C.scene.camera.matrix_world.to_euler()
        camera.rotation_euler = (1.533825159072876, -9.605929562894744e-07, 0.273378849029541)
        self.coll.objects.link(camera)
        # change camera size
        bpy.context.scene.render.resolution_x = 1000
        bpy.context.scene.render.resolution_y = 1000
        
        bpy.context.scene.camera = camera


    def render_mesh(self,file):

        ## get file and put it in scene collection
        bpy.ops.import_mesh.ply(filepath=file)
        obj = bpy.context.active_object

        # remove it from scene collection
        self.scene_coll.objects.unlink(obj)

        # add it to my collection
        self.coll.objects.link(obj)

        bpy.context.scene.render.filepath = str(Path(file).with_suffix(".png"))
        bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
        bpy.ops.render.render(write_still=True)

        print("Mesh render saved to ",str(Path(file).with_suffix(".png")))
        
    def render_pc(self,file):
        
        data = np.load(file)
        
        # Create and arrange mesh data
        m     = bpy.data.meshes.new('pc')
        m.from_pydata(data["points"], [], [])
        # Create mesh object and link to scene collection
        o = bpy.data.objects.new( 'pc', m )
        self.coll.objects.link(o)

        # Add minimal icosphere
        #bpy.ops.mesh.primitive_ico_sphere_add( subdivisions = 2, radius = 0.004 )
        #bpy.ops.mesh.primitive_cube_add(size=0.005)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=0, y_subdivisions=0, size=0.005)
        isobj = bpy.data.objects[ bpy.context.object.name ]
        
        self.coll.objects.link(isobj)
        self.scene_coll.objects.unlink(isobj)
        
        
        
        # Set instancing props
        for ob in [ isobj, o ]:
            ob.instance_type               = 'VERTS'

        # Set instance parenting (parent icosphere to verts)
        o.select_set(True)
        bpy.context.view_layer.objects.active = o

        bpy.ops.object.parent_set( type = 'VERTEX', keep_transform = True )
        
        # adding material and setup nodes:
        mat = bpy.data.materials.new("topo_mat")
        mat.use_nodes = True
        node = mat.node_tree.nodes.new("ShaderNodeAttribute")
        mat.node_tree.links.new(node.outputs['Color'], mat.node_tree.nodes['Principled BSDF'].inputs[0])
        node.attribute_name = "color"

        ob = bpy.context.scene.objects["pc.010"]

        # set material to the object:
        ob.data.materials.append(mat)
        
        ob.color = (1,0,0,1)
        
    
    def render_photogrammetry_addon(self, file):
        
        """Works but slow, and not sure how to change color. And no normal in the particle system"""
        
        points = PointDataFileHandler.parse_point_data_file(file, None)
        
        add_points_as_object_with_particle_system(points,self.coll,mesh_type='SPHERE',
        add_particle_color_emission=True,point_extent=0.001)

        bpy.context.scene.render.filepath = str(Path(file).with_suffix(".png"))
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.ops.render.render(write_still=True)
        
        print("Renderer to", str(Path(file).with_suffix(".png")))

    def apply_render_settings(self):

        bpy.context.scene.view_settings.view_transform = 'Standard'
        bpy.context.scene.view_settings.gamma = 0.92126
        bpy.context.scene.view_settings.exposure = 1.41732

        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # or "OPENCL"
        bpy.context.scene.cycles.device = "GPU"

        # bpy.context.preferences.addons["cycles"].preferences.get_devices()
        # print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        # for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        #     d["use"] = 1  # Using all devices, include GPU and CPU
        #     print(d["name"], d["use"])

        bpy.context.scene.cycles.samples = 4

    def add_light(self,location):

        # create light datablock, set attributes
        light_data = bpy.data.lights.new(name="light", type='POINT')
        light_data.energy = 15

        # create new object with our light datablock
        light_object = bpy.data.objects.new(name="light", object_data=light_data)

        # link light object
        self.coll.objects.link(light_object)

        # make it active
        bpy.context.view_layer.objects.active = light_object

        # change location
        light_object.location = location

        return light_object

    def render_bplt(self, file):
        
        """this is the one to use for point cloud rendering"""
        data = np.load(file)

        angles = np.arccos(np.dot(data["normals"],[0,0,1]))
        cross = np.cross(data["normals"],[0,0,1])

        cross = cross / np.linalg.norm(cross, axis=1)[:, np.newaxis]

        quat = np.array([cross[:, 0], cross[:, 1], cross[:, 2], angles[:]]).transpose()
        rots = Rotation.from_quat(quat).as_matrix()

        I=np.identity(3)
        I[0,0]=-1
        I[2,2]=-1
        rots=rots@I

        pc="pc"
        bplt.Scatter(data["points"],
                                color=data["colors"], 
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

        light = [0.956605,-0.261804,0.466202]
        self.add_light(light)


        self.apply_render_settings()

        bpy.context.scene.render.filepath = str(Path(file).with_suffix(".png"))
        bpy.ops.render.render(write_still=True)
        print("Renderer to", str(Path(file).with_suffix(".png")))
                                
        
        
        
        


if __name__ == '__main__':
    
    rr = RenderReal()
    
    rr.make_cam()
    
    
    #rr.render("/home/rsulzer/data/real_out/paper/models/Ignatius/labatut.ply")
    # rr.render_pc("/home/rsulzer/data/real_out/paper/models/Ignatius/input.npz")
    #rr.render_photogrammetry_addon(r'/home/rsulzer/data/real_out/paper/models/Ignatius/input.ply')
    rr.render_bplt("/home/rsulzer/data/real_out/paper/models/Ignatius/input.npz")
