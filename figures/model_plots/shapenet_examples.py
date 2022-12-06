import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "datasets"))
from berger import Berger
from shapenet import ShapeNet
import numpy as np
# import vedo
import open3d as o3d
o3d.visualization.webrtc_server.enable_webrtc()
from tqdm import tqdm
# classes={"airplane":0,"table":1,"car":1}


################## THIS SCRIPT CANNOT BE RUN THROUGH PYCHARM, BUT IT DOES WORK TO LAUNCH IT ON THE  REMOTE MACHINE DIRECTLY ####################


split="train"

interactive = True
size = (500,500)
setview=False

outpath = "/home/raphael/presentation/benchmark/shapenet"

render = setview
# for k,v in classes.items():
#
#     print(k,v)
#
#
#     # load data
#     dataset = ShapeNet(classes=[k])
#     dataset.getModels(scan_conf="6", splits=[split])
#     model = list(dataset.getModels().values())[0][v]

# load data
dataset = ShapeNet()
methods = ["mesh","pc4","pc6"]
# methods=["mesh"]
for method in methods:

    scan_conf="4"
    if method == "pc6":
        scan_conf="6"
    models = dataset.getModels(scan_conf=scan_conf, splits=[split], reduce=0.05)[split]

    for m in tqdm(models[:2]):

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=500,height=500, visible=render)

        if method == "pc4" or method == "pc6":
            # return
            data=np.load(m["scan"])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data["points"])
            pcd.normals = o3d.utility.Vector3dVector(data["normals"])
            recon_mesh = pcd

            # recon_mesh = o3d.io.read_point_cloud(m["scan_ply"])
            vis.get_render_option().point_size = 3

        else:
            recon_mesh = o3d.io.read_triangle_mesh(m["mesh"])
            recon_mesh.compute_vertex_normals()
            recon_mesh.vertex_colors = recon_mesh.vertex_normals

            vis.get_render_option().mesh_show_wireframe=False
            vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Default
            # vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Default
            vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color


        vis.add_geometry(recon_mesh)

        # if(not args.set_view):
        vis.get_render_option().light_on=True
        vis.get_render_option().show_coordinate_frame = render


        # view_file = os.path.join(outpath, "{}_{}_view.json".format(str(k),str(v)))
        view_file = os.path.join(outpath, "view.json")

        if(setview):
            vis.run()
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            print("write view file: ", view_file)
            o3d.io.write_pinhole_camera_parameters(view_file, param)
            ctr=None
        else:
            # load viewfile
            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters(view_file)
            ctr.convert_from_pinhole_camera_parameters(param)
            ctr.set_up([ 0.0, 1.0, 0.0 ])

            # vis.run()

        os.makedirs(os.path.join(outpath,method),exist_ok=True)
        image_file = os.path.join(outpath, method, "img_{}_{}.png".format(str(m["class"]),str(m["model"])))
        # print("save to: ", image_file)
        vis.capture_screen_image(image_file, do_render=True)
        # vis.close()

        vis.destroy_window()
        del ctr
        del vis





a=5
