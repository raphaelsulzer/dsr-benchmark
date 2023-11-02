import os, sys, subprocess, trimesh
from libmesh import check_mesh_contains
from tqdm import tqdm
import numpy as np
from glob import glob
import open3d as o3d

import matplotlib.pyplot as plt
plt.switch_backend('agg')  # there was some wired bug which is solved by this https://stackoverflow.com/a/67617670

try:
    from pycompod import PlaneExporter
except:
    pass # for importing default dataset in blender

DEBUG = 0


class SimpleShapes:

    def __init__(self,path="/home/rsulzer/data/simple_shapes", mesh_tools_dir="/home/rsulzer/cpp/mesh-tools/build/release"):

        self.path = path
        self.model_dicts = []
        self.mesh_tools_dir = mesh_tools_dir
        self.POISSON_EXE = "/home/rsulzer/cpp/PoissonRecon/Bin/Linux/PoissonRecon"


    def get_models(self,hint=None):

        models = np.genfromtxt(os.path.join(self.path,"models.lst"),dtype=str)
        for m in models:

            if hint is not None:
                if hint not in m:
                    continue

            d = {}
            d["class"] = ""
            d["model"] = m

            d["eval"] = dict()
            d["eval"]["occ"] = os.path.join(self.path,d["class"],m,"eval","points.npz")
            d["eval"]["pointcloud"] = os.path.join(self.path,d["class"],m,"eval","pointcloud.npz")

            d["pointcloud"] = os.path.join(self.path,d["class"],m,"pointcloud","pointcloud.npz")
            d["pointcloud_ply"] = os.path.join(self.path,d["class"],m,"pointcloud","pointcloud.ply")

            d["mesh"] = os.path.join(self.path,d["class"],m,"mesh.off")
            d["planes"] = os.path.join(self.path,d["class"],m,"planes","planes.npz")

            d["ksr"] = {}
            d["ksr"]["surface"] = os.path.join(self.path,d["class"],m,"ksr",'{}',"surface.off")
            d["ksr"]["partition"] = os.path.join(self.path,d["class"],m,"ksr",'{}',"partition.kgraph")

            d["abspy"] = {}
            d["abspy"]["surface"] = os.path.join(self.path,d["class"],m,"abspy",'{}',"surface.off")
            d["abspy"]["partition"] = os.path.join(self.path,d["class"],m,"abspy",'{}',"partition.ply")

            self.model_dicts.append(d)

        return self.model_dicts



    def scan(self,scan_setting="4",scanner_dir="/home/raphael/cpp/mesh-tools/build/release/scan",
             normal_method='jet', normal_neighborhood=30, normal_orient=1):

        if(len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)

        scan = scan_settings[scan_setting]

        for m in self.model_dicts:

            os.makedirs(os.path.join(self.path,"scan",m["class"]),exist_ok=True)

            command = [scanner_dir,
                       "-w", str(self.path),
                       "-i", "mesh/1/"+m["class"]+".off",
                       "-o", str(os.path.join("scan",m["class"],scan_setting)),
                       "--points", scan["points"],
                       "--noise", scan["noise"],
                       "--outliers", scan["outliers"],
                       "--cameras", scan["cameras"],
                       "--normal_method", "jet",
                       "--export", "all",
                       "--normal_neighborhood", normal_neighborhood,
                       "--normal_method", normal_method,
                       "--normal_orient", normal_orient]
            print(*command)
            p = subprocess.Popen(command)
            p.wait()


    def estimNormals(self, method='jet', neighborhood=30, orient=1):
        if (len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)

        for m in tqdm(self.model_dicts, ncols=50):
            try:
                command = [str(os.path.join(self.mesh_tools_dir, "normal")),
                           "-w", str(self.path),
                           "-s", "npz",
                           "-i", str(os.path.join("scan", m["model"], self.scan_conf + ".npz")),
                           "-o", str(os.path.join("scan", m["model"], self.scan_conf)),
                           "--method", method,
                           "--neighborhood", str(neighborhood),
                           "--orient", str(orient),
                           "--overwrite", "1"]
                print(*command)
                p = subprocess.Popen(command)
                p.wait()
            except Exception as e:
                print(e)
                print("Skipping {}/{}".format(m["class"], m["model"]))


    def makePoisson(self, depth=8, boundary=2):

        for m in tqdm(self.model_dicts, ncols=50):
            # try:
            command = [self.POISSON_EXE,
                       "--in", m["scan_ply"],
                       "--out", os.path.join(self.path, m["class"], m["mesh"]),
                       "--depth", str(depth),
                       "--bType", str(boundary)]
            print("run command: ", *command)
            p = subprocess.Popen(command)
            # p = subprocess.Popen(command, shell=False,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p.wait()



    def standardize(self,padding=0.1):

        for m in tqdm(self.model_dicts):

            mesh = o3d.io.read_triangle_mesh(m["mesh"])
            center = mesh.get_axis_aligned_bounding_box().get_center()

            mesh = mesh.translate(-center)


            max_bound = np.vstack([np.abs(mesh.get_min_bound()), mesh.get_max_bound()])
            col_index = np.argmax(max_bound,axis=1)[1]
            scale = np.abs(mesh.get_min_bound()[col_index]) + np.abs(mesh.get_max_bound()[col_index])


            mesh = mesh.scale((1-padding) / scale, [0, 0, 0])

            o3d.io.write_triangle_mesh(os.path.splitext(m["mesh"])[0]+"_unit.off",mesh)



    def makePointcloudPLY(self,n_points=100000,std_noise=0.0):

        print("Writing pointclouds for reconstruction input...\n")

        for m in tqdm(self.model_dicts):

            data = np.load(m["pointcloud"])

            ind = np.random.randint(0, data["points"].shape[0], size = (n_points,))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data["points"][ind])
            pcd.normals = o3d.utility.Vector3dVector(data["normals"][ind])
            o3d.io.write_point_cloud(m["pointcloud_ply"], pcd)




    def sample(self,n_points=100000):

        print("Sample points on surface and in bounding box for evaluation...\n")


        if(len(self.model_dicts) < 1):
            print("\nERROR: run getModels() first!")
            sys.exit(1)


        loc = np.zeros(3)
        scale = 1.0


        for m in tqdm(self.model_dicts):

            mesh = trimesh.load(m["mesh"])

            # surface points
            points_surface, fid = mesh.sample(n_points,return_index=True)
            normals = mesh.face_normals[fid]

            fpath = os.path.join(self.path,m["class"],m["model"],"eval")
            os.makedirs(fpath,exist_ok=True)
            filename = os.path.join(fpath,"pointcloud.npz")
            np.savez(filename, points=points_surface, normals=normals, loc=loc, scale=scale)

            if DEBUG:
                print('Writing points: %s' % filename)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_surface)
                pcd.normals = o3d.utility.Vector3dVector(normals)
                o3d.io.write_point_cloud(os.path.join(fpath,"pointcloud.ply"), pcd)



            # IoU points

            n_points_uniform = int(n_points * 0.5)
            n_points_surface = n_points - n_points_uniform

            boxsize = 1
            points_uniform = np.random.rand(n_points_uniform, 3)
            points_uniform = boxsize * (points_uniform - 0.5)
            points_surface = mesh.sample(n_points_surface)
            points_surface += 0.05 * np.random.randn(n_points_surface, 3)
            points = np.concatenate([points_uniform, points_surface], axis=0)

            occupancies = check_mesh_contains(mesh, points)

            colors = np.zeros(shape=(n_points, 3)) + [0, 0, 1]
            colors[occupancies] = [1,0,0]



            dtype = np.float16
            points = points.astype(dtype)
            occupancies = np.packbits(occupancies)

            filename = os.path.join(self.path,m["class"],m["model"],"eval","points.npz")
            np.savez(filename, points=points, occupancies=occupancies,loc=loc, scale=scale)

            if DEBUG:
                print('Writing points: %s' % filename)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(os.path.join(fpath,"points.ply"), pcd)



    def make_planes(self,model="cube", plot=False, noise=0.02):

        res = 9
        xx, yy = np.meshgrid(np.linspace(1, res, num=res + 1, endpoint=True, dtype=float),
                             np.linspace(1, res, num=res + 1, endpoint=True, dtype=float))
        xxr, yyr = np.meshgrid(np.linspace(res, 1, num=res + 1, endpoint=True, dtype=float),
                               np.linspace(res, 1, num=res + 1, endpoint=True, dtype=float))
        zz = np.zeros(shape=(res + 1, res + 1), dtype=float)
        zzr = np.ones(shape=(res + 1, res + 1), dtype=float) * 10.0

        if model == "split_cube":
            ppoints = np.array(
                [[5, 5, 5], [5.3, 5.3, 5.3], [0, 0, 0], [0, 0, 0], [0, 0, 0], [10, 10, 10], [10, 10, 10], [10, 10, 10]],
                dtype=float)
            normals = np.array(
                [[0, 0, 1], [0, 0, 1], [0, -1, 0], [-1, 0, 0], [0, 0, -1], [0, 1, 0], [1, 0, 0], [0, 0, 1]],
                dtype=float)

            zz5 = np.ones(shape=(res + 1, res + 1), dtype=float) * 5.0
            zz6 = np.ones(shape=(res + 1, res + 1), dtype=float) * 5.3

            sampled_points = np.array(
                [[xx, yy, zz5], [xx, yy, zz6],[xx, zz, yy], [zz, yy, xx], [xx, yy, zz], [xxr, zzr, yyr], [zzr, yyr, xxr], [xxr, yyr, zzr]])


        elif model == "slanted_cube":
            ppoints = np.array(
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [10, 10, 10], [10, 10, 10], [10, 10, 10], [5, 5, 5], [5, 5, 10]],
                dtype=float)
            normals = np.array(
                [[0, -1, 0], [-1, 0, 0], [0, 0, -1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0.5, 0, 0.45]],
                dtype=float)

            zz5 = np.ones(shape=(res + 1, res + 1), dtype=float) * 5.0

            d = -ppoints[-1].dot(normals[-1])
            zzs = (-normals[-1][0] * xx - normals[-1][1] * yy - d) * 1. / normals[-1][2]
            sampled_points = np.array(
                [[xx, zz, yy], [zz, yy, xx], [xx, yy, zz], [xxr, zzr, yyr], [zzr, yyr, xxr], [xxr, yyr, zzr],
                 [xx, yy, zz5],
                 [xx, yy, zzs]])

        elif model == "double_slanted_cube":
            ppoints = np.array(
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [10, 10, 10], [10, 10, 10], [10, 10, 10], [5, 5, 5], [5, 5, 10],
                 [5, 5, 10]], dtype=float)
            normals = np.array(
                [[0, -1, 0], [-1, 0, 0], [0, 0, -1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0.5, 0, 0.5],
                 [0, 0.5, 0.5]], dtype=float)

            zz5 = np.ones(shape=(res + 1, res + 1), dtype=float) * 5.0

            d = -ppoints[-2].dot(normals[-2])
            zzs = (-normals[-2][0] * xx - normals[-2][1] * yy - d) * 1. / normals[-2][2]

            d = -ppoints[-1].dot(normals[-1])
            zzs2 = (-normals[-1][0] * xx - normals[-1][1] * yy - d) * 1. / normals[-1][2]
            sampled_points = np.array(
                [[xx, zz, yy], [zz, yy, xx], [xx, yy, zz], [xxr, zzr, yyr], [zzr, yyr, xxr], [xxr, yyr, zzr],
                 [xx, yy, zz5],
                 [xx, yy, zzs], [xx, yy, zzs2]])

        elif model == "double_split_cube":
            ppoints = np.array(
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [10, 10, 10], [10, 10, 10], [10, 10, 10], [5, 5, 5], [6, 6, 10]],
                dtype=float)
            normals = np.array(
                [[0, -1, 0], [-1, 0, 0], [0, 0, -1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0.5, 0, 0]],
                dtype=float)
            zz5 = np.ones(shape=(res + 1, res + 1), dtype=float) * 5.0

            xxs, yys = np.meshgrid(np.linspace(6, res, num=res + 1, endpoint=True, dtype=float),
                                   np.linspace(1, res, num=res + 1, endpoint=True, dtype=float))
            zzs = np.zeros(shape=(res + 1, res + 1), dtype=float) + 6

            sampled_points = np.array(
                [[xx, zz, yy], [zz, yy, xx], [xx, yy, zz], [xxr, zzr, yyr], [zzr, yyr, xxr], [xxr, yyr, zzr],
                 [xx, yy, zz5],
                 [zzs, yys, xxs]])
        elif model == "complex_cube":
            ppoints = np.array(
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [10, 10, 10], [10, 10, 10], [10, 10, 10], [5, 5, 5], [6, 6, 10],
                 [4, 10, 10], [10, 12, 6]], dtype=float)
            normals = np.array(
                [[0, -1, 0], [-1, 0, 0], [0, 0, -1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0.5, 0, 0],
                 [0.5, 0, 0], [0, 0.5, 0]], dtype=float)

            xx5, yy5 = np.meshgrid(np.linspace(1, res + 2, num=res + 1, endpoint=True, dtype=float),
                                   np.linspace(1, res + 2, num=res + 1, endpoint=True, dtype=float))
            zz5 = np.ones(shape=(res + 1, res + 1), dtype=float) * 5.0

            xxs, yys = np.meshgrid(np.linspace(6, res, num=res + 1, endpoint=True, dtype=float),
                                   np.linspace(6, res, num=res + 1, endpoint=True, dtype=float))
            zzs = np.zeros(shape=(res + 1, res + 1), dtype=float) + 6

            xxs2, yys2 = np.meshgrid(np.linspace(1, 4, num=res + 1, endpoint=True, dtype=float),
                                     np.linspace(10.5, res + 2, num=res + 1, endpoint=True, dtype=float))
            zzs2 = np.zeros(shape=(res + 1, res + 1), dtype=float) + 4

            xxs3, yys3 = np.meshgrid(np.linspace(1, res - 6, num=res + 1, endpoint=True, dtype=float),
                                     np.linspace(1, 4, num=res + 1, endpoint=True, dtype=float))
            zzs3 = np.zeros(shape=(res + 1, res + 1), dtype=float) + 12

            sampled_points = np.array(
                [[xx, zz, yy], [zz, yy, xx], [xx, yy, zz], [xxr, zzr, yyr], [zzr, yyr, xxr], [xxr, yyr, zzr],
                 [xx, yy, zz5],
                 [zzs, yys, xxs], [zzs2, yys2, xxs2], [xxs3, zzs3, yys3]])
        # elif model == "double_split_cube":
        #     ppoints  = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0],[10, 10, 10],[10, 10, 10],[10, 10, 10],[5, 5, 5],[7.5, 7.5, 7.5]],dtype=float)
        #     normals = np.array([[0, 1, 0],[1, 0, 0],[0, 0, 1],[0, -1, 0],[-1, 0, 0],[0, 0, -1],[0, 0, 1],[0, 0, 1]],dtype=float)
        #
        #     zz5 = np.ones(shape=(res + 1, res + 1), dtype=float) * 5.0
        #     zz75 = np.ones(shape=(res + 1, res + 1), dtype=float) * 7.5
        #     sampled_points = np.array(
        #         [[xx, zz, yy], [zz, yy, xx], [xx, yy, zz], [xxr, zzr, yyr], [zzr, yyr, xxr], [xxr, yyr, zzr], [xx, yy, zz5], [xx, yy, zz75]])
        else:
            ppoints = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [10, 10, 10], [10, 10, 10], [10, 10, 10]], dtype=float)
            normals = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

            sampled_points = np.array(
                [[xx, zz, yy], [zz, yy, xx], [xx, yy, zz], [xxr, zzr, yyr], [zzr, yyr, xxr], [xxr, yyr, zzr]])

        # a plane is a*x+b*y+c*z+d=0
        # [a,b,c] is the normal. Thus, we have to calculate
        # d and we're set

        # plot the surface
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        points = []
        point_normals = []
        group_parameters = []
        group_colors = []
        group_num_points = []
        group_points = []
        pl = 0
        for i, (xx, yy, zz) in enumerate(sampled_points):
            col = np.random.rand(1, 3)
            group_colors.append(col * 255)

            d = -ppoints[i].dot(normals[i])
            plane = np.array([normals[i][0], normals[i][1], normals[i][2], d])
            plane += (np.random.rand(4, ) * noise)
            group_parameters.append(plane)
            ax.plot_surface(xx, yy, zz, alpha=0.2, color=col)

            p = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).transpose().astype(float)
            p += (np.random.rand(p.shape[0], 3) * noise)
            points.append(p)
            point_normals.append(np.repeat(normals[np.newaxis, i], p.shape[0], axis=0))
            group_num_points.append([p.shape[0]])
            group_points.append(np.arange(p.shape[0]) + pl)

            pl += p.shape[0]

        if plot:
            plt.show()
        else:
            plt.close()

        points = np.concatenate(points)
        normals = np.concatenate(point_normals)
        group_parameters = np.array(group_parameters)
        np.random.seed(42)
        group_colors = np.random.randint(low=100, high=255, size=(group_parameters.shape[0], 3), dtype=np.int32)
        group_num_points = np.array(group_num_points, dtype=np.int32).flatten()
        group_points = np.array(group_points, dtype=np.int32).flatten()

        ### npz file
        file = "/home/rsulzer/data/simple_shapes/{}/planes/planes.npz".format(model)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        np.savez(file,
                 points=points, normals=normals, group_parameters=group_parameters, group_colors=group_colors,
                 group_num_points=group_num_points, group_points=group_points, colors=group_colors)

        # ### pln file (simply the plane equations in a text file)
        # file = "/home/rsulzer/data/reconbench/planes/{}/1/planes.pln".format(model)
        # np.savetxt(file,group_parameters,'%.12g')

        ### ply file (all convex hulls of all planes)
        pt_file = "/home/rsulzer/data/simple_shapes/{}/planes/planes_sampled.ply".format(model)
        plane_file = "/home/rsulzer/data/simple_shapes/{}/planes/planes.ply".format(model)
        exp = PlaneExporter()
        data = np.load(file)
        exp.save_points_and_planes_from_array(plane_file, pt_file, data)

        ### points file
        os.makedirs(os.path.join(self.path,model,"pointcloud"),exist_ok=True)
        file = "/home/rsulzer/data/simple_shapes/{}/pointcloud/pointcloud.ply".format(model)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_point_cloud(file, pcd)

        np.savez_compressed(os.path.join(self.path,model,"pointcloud","pointcloud.npz"),points=points,normals=normals)




if __name__ == '__main__':

    ds = SimpleShapes()


    # model = "cube"
    model = "split_cube"
    # model = "slanted_cube"
    # model = "double_slanted_cube"
    ds.make_planes(model,plot=True,noise=0.0)














