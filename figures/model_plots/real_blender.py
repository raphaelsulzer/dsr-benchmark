import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=1090, stdoutToServer=True, stderrToServer=True)

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
        
        
    def makeCam(self):
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


    def render(self,file):

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
        
        
        
        for mate in ob.materials:
            pass

        
        
    
    
    def render_pc2(self,file):
        
        data = np.load(file)
        
        # Create and arrange mesh data
        m     = bpy.data.meshes.new('pc')
        m.from_pydata(data["points"], [], [])
        # Create mesh object and link to scene collection
        o = bpy.data.objects.new( 'pc', m )
        self.coll.objects.link(o)
        
        
        bm = bmesh.new()
        bm.from_mesh(me)
        for x, y, z, t, tau in verts:
            co = scale * Vector((x, y, z))
            v = bm.verts.new(co)
            v.normal = colorsys.hls_to_rgb(tau, t, 1)

        bm.to_mesh(me)
        
    
    def render_photogrammetry_addon(self, file):
        
        """Works but slow, and not sure how to change color. And no normal in the particle system"""
        
        points = PointDataFileHandler.parse_point_data_file(file, None)
        
        add_points_as_object_with_particle_system(points,self.coll,mesh_type='SPHERE',
        add_particle_color_emission=True,point_extent=0.001)
        
        
        bpy.context.scene.render.filepath = str(Path(file).with_suffix(".png"))
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.ops.render.render(write_still=True)
        
        print("Renderer to", str(Path(file).with_suffix(".png")))

    def get_rotation_matrix(self,axis, theta):
        """
        Find the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians.
        Credit: http://stackoverflow.com/users/190597/unutbu

        Args:
            axis (list): rotation axis of the form [x, y, z]
            theta (float): rotational angle in radians

        Returns:
            array. Rotation matrix.
        """
        a = np.cos(theta/2.0)

        b, c, d = -axis * np.expand_dims(np.sin(theta / 2.0),axis=1)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def render_bplt(self, file):
        
        
        data = np.load(file)

        angles = np.arccos(np.dot(data["normals"],[0,0,1]))
        cross = np.cross(data["normals"],[0,0,1])

        cross = cross / np.linalg.norm(cross, axis=1)[:, np.newaxis]

        #s=np.sqrt(1 - angles ** 2)
        quat = np.array([cross[:, 0], cross[:, 1], cross[:, 2], angles[:]]).transpose()
        rots = Rotation.from_quat(quat).as_matrix()


        scatter = bplt.Scatter(data["points"], 
                                color=data["colors"], 
                                marker_type="cones",
                                radius_bottom=1,
                                radius_top=3,
                                marker_scale=[0.001]*3,
                                marker_rotation=rots,
                                randomize_rotation=False)
        # TODO: need to translate the normal direction to 3d euler rotation vector
                                
        
        
        
        


if __name__ == '__main__':
    
    rr = RenderReal()
    
    rr.makeCam()
    
    
    #rr.render("/home/rsulzer/data/real_out/paper/models/Ignatius/labatut.ply")
    # rr.render_pc("/home/rsulzer/data/real_out/paper/models/Ignatius/input.npz")
    #rr.render_photogrammetry_addon(r'/home/rsulzer/data/real_out/paper/models/Ignatius/input.ply')
    rr.render_bplt("/home/rsulzer/data/real_out/paper/models/Ignatius/input.npz")
