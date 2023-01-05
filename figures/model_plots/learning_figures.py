import sys, os, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from methods.methods import learning_methods
import vedo
import numpy as np
import open3d as o3d
from datasets.reconstructed_dataset import learning_dataset

cam_dict = {}
cam_dict["02691156/d2e99eeecebf0c77bd46d022fd7d80aa"] = dict(pos=(0.7568, 0.8104, 1.266),
           focalPoint=(0.01676, -0.06281, 0.04315),
           viewup=(-0.1931, 0.8500, -0.4901),
           distance=1.675,
           clippingRange=(0.3643, 3.329))
cam_dict["02691156/d18592d9615b01bbbc0909d98a1ff2b4"] = dict(pos=(0.7568, 0.8104, 1.266),
           focalPoint=(0.01676, -0.06281, 0.04315),
           viewup=(-0.1931, 0.8500, -0.4901),
           distance=1.675,
           clippingRange=(0.3643, 3.329))
cam_dict["bed/0585"] = dict(pos=(-1.114, -1.352, 0.9898),
           focalPoint=(0.01841, -0.08021, 0.02212),
           viewup=(0.3313, 0.3665, 0.8694),
           distance=1.959,
           clippingRange=(0.8796, 3.534))

light_dict = {}
light_dict["02691156/d2e99eeecebf0c77bd46d022fd7d80aa"] = [0.6,0.6,0.6]
light_dict["bed/0585"] = [-0.6,-0.6,0.6]


def get_models(dataset,methods):

    for experiment,d in dataset.items():
        print("\nGet models from experiment ", experiment)
        for model in d["models"]:
            opath = os.path.join(outpath, experiment, model)
            os.makedirs(opath, exist_ok=True)

            for m in methods:
                ofile = os.path.join(opath, m["name"] + m["out_type"])
                # os.makedirs(outpath,exist_ok=True)
                expath = m["path"].format(experiment)
                cmd = ["scp", "enpc:/mnt/raphael/{}_out/benchmark/{}/{}".format(d["train_dataset"],expath, model+m["out_type"]),
                       str(ofile)]
                print(*cmd)
                p = subprocess.Popen(cmd)
                p.wait()

            if(experiment=="reconbench"):
                ## input ply
                cmd = ["scp", "enpc:/mnt/raphael/{}/scan/{}/{}.ply".format(d["test_dataset"], model.split('/')[0], d["scan"]),
                       str(os.path.join(opath, "input.ply"))]
                print(*cmd)
                p = subprocess.Popen(cmd)
                p.wait()
                ## input npz
                cmd = ["scp", "enpc:/mnt/raphael/{}/scan/{}/{}.npz".format(d["test_dataset"], model.split('/')[0], d["scan"]),
                       str(os.path.join(opath, "input.npz"))]
                print(*cmd)
                p = subprocess.Popen(cmd)
                p.wait()

                ## ground truth
                cmd = ["scp", "enpc:/mnt/raphael/{}/mesh/{}.off".format(d["test_dataset"], model.split('/')[0]),
                       str(os.path.join(opath, "gt.off"))]
                print(*cmd)
                p = subprocess.Popen(cmd)
                p.wait()

            else:
                ## input ply
                cmd = ["scp", "enpc:/mnt/raphael/{}/{}/scan/{}.ply".format(d["test_dataset"], model, d["scan"]),
                       str(os.path.join(opath, "input.ply"))]
                print(*cmd)
                p = subprocess.Popen(cmd)
                p.wait()
                ## input npz
                cmd = ["scp", "enpc:/mnt/raphael/{}/{}/scan/{}.npz".format(d["test_dataset"], model, d["scan"]),
                       str(os.path.join(opath, "input.npz"))]
                print(*cmd)
                p = subprocess.Popen(cmd)
                p.wait()

                ## ground truth
                cmd = ["scp", "enpc:/mnt/raphael/{}/{}/mesh/mesh.off".format(d["test_dataset"], model),
                       str(os.path.join(opath, "gt.off"))]
                print(*cmd)
                p = subprocess.Popen(cmd)
                p.wait()

def plot_model_o3d(method,setview):
    mesh_file = os.path.join(outpath,experiment,method["name"]+method["out_type"])

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=600,height=600, visible=True)

    if method["name"] == "input":
        # return
        recon_mesh = o3d.io.read_point_cloud(mesh_file)
        vis.get_render_option().point_size = 5
    else:
        recon_mesh = o3d.io.read_triangle_mesh(mesh_file)
        recon_mesh.compute_vertex_normals()
        recon_mesh.vertex_colors = recon_mesh.vertex_normals

        vis.get_render_option().mesh_show_wireframe=False
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Default
        # vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Default
        vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color

    if experiment=="shapenet" and method["name"] in ["ConvONet2D","ConvONet3D","POCO","SAP"]:
        R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
        recon_mesh.rotate(np.linalg.inv(R),center=(0, 0, 0))

    vis.add_geometry(recon_mesh)

    # if(not args.set_view):
    vis.get_render_option().light_on=True


    view_file = os.path.join(outpath,experiment,"viewpoint.json")

    if(setview):
        vis.run()
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        print("write view file: ", view_file)
        o3d.io.write_pinhole_camera_parameters(view_file, param)
    else:
        # load viewfile
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(view_file)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.run()

    image_file = os.path.join(outpath,experiment,method["name"]+ ".png")
    print("save to: ", image_file)
    vis.capture_screen_image(image_file, do_render=True)
    vis.close()

def plot_model(method):
    file = os.path.join(outpath,experiment,method["name"]+method["out_type"])
    data = vedo.load(file)

    if method["name"] == "input":
        data = vedo.Points(data, r=10.0, c=[113, 189, 247])
    else:
        data = vedo.Mesh(data, c=[180, 180, 180])
        data = data.computeNormals().phong()

    if experiment=="shapenet" and method["name"] in ["ConvONet2D","ConvONet3D","POCO","SAP"]:
        R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
        data.applyTransform(np.linalg.inv(R))

    image_file = os.path.join(outpath,experiment,method["name"]+ ".png")

    light = light_dict[model]

    p2 = vedo.Point(light, c='y')
    l2 = vedo.Light(p2, c='w', intensity=1)
    cam = cam_dict[model]

    interactive = True
    p = vedo.show(data, l2, size=(700, 700), camera=cam, interactive=interactive)
    # p = vedo.show(data, l2, size=(700, 700), interactive=interactive)
    vedo.io.screenshot(image_file)
    p.close()


def plot_models(methods):
    d = {}
    d["name"] = "gt"
    d["out_type"] = ".off"
    methods.append(d)
    d = {}
    d["name"] = "input"
    d["out_type"] = ".ply"
    methods.append(d)
    setview=False
    for m in methods:
        # plot_model(m)
        plot_model_o3d(m,setview)
        setview=False


outpath = "/home/rsulzer/data/benchmark"


get_models(learning_dataset,learning_methods)
# plot_models(learning_methods)
