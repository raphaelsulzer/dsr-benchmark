import os, subprocess
import copy
import open3d as o3d
import numpy as np
import configparser
import glob

from npz2ply import npz2ply

SURE_DIR = "/home/rsulzer/cpp/mesh-tools/build/release"

stddev = 0.0015

class Ignatius:

    def __init__(self,path):

        self.path = path

        self.trans = None
        self.trans_a = None
        self.trans_b = None
        self.trans_c = None


    def makeLearning(self):

        pcd_full = o3d.geometry.PointCloud()
        points = []
        sensors = []
        sfile = list(open(glob.glob(os.path.join(self.path,"is_ori","*.txt"))[0]))
        for i,file in enumerate(sorted(glob.glob(os.path.join(self.path,"is_ori","*.ply")))):
            pcd = o3d.io.read_point_cloud(file)

            parr = np.asarray(pcd.points)
            sarr = np.array(sfile[i].split()[1:],dtype=float)
            # points.append(parr)
            # sensors.append(np.expand_dims(sarr,axis=0).repeat(parr.shape[0],axis=0))
            pcd.normals = o3d.utility.Vector3dVector(np.expand_dims(sarr,axis=0).repeat(parr.shape[0],axis=0))
            pcd_full+=pcd


        # points = np.concatenate(points,axis=0)
        # sensors = np.concatenate(sensors,axis=0)

        vol = o3d.visualization.read_selection_polygon_volume(os.path.join(self.path,"Ignatius.json"))
        pcd = vol.crop_point_cloud(pcd_full)

        points = copy.deepcopy(np.asarray(pcd.points))
        sensors = copy.deepcopy(np.asarray(pcd.normals))
        colors = copy.deepcopy(np.asarray(pcd.colors))

        center = pcd.get_axis_aligned_bounding_box().get_center()
        pcd = pcd.translate(-center)

        points-=center
        sensors-=center

        max_bound = np.vstack([np.abs(pcd.get_min_bound()), pcd.get_max_bound()])
        col_index = np.argmax(max_bound, axis=1)[1]
        scale = np.abs(pcd.get_min_bound()[col_index]) + np.abs(pcd.get_max_bound()[col_index])

        pcd = pcd.scale(1 / scale, [0, 0, 0])

        points/=scale
        sensors/=scale

        os.makedirs(os.path.join(self.path,"scan"),exist_ok=True)
        np.savez(os.path.join(self.path,"scan","pointcloud.npz"),points=points,sensor_pos=sensors,colors=colors)

        o3d.io.write_point_cloud(os.path.join(self.path,"scan","pointcloud.ply"), pcd)

        # npz2ply(os.path.join(self.path, "eval", "poincloud.npz"), os.path.join(self.path, "eval", "pointcloud_npz.ply"),normal_type="sensor_vec")



    def uniform_sampling(self,mesh,stddev):
        n=250000
        uniform_points = mesh.sample(n)
        noise = stddev * np.random.randn(*uniform_points.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(uniform_points+noise)
        o3d.io.write_point_cloud(os.path.join(self.path,"uniform_"+str(n)+".ply"),pcd)

        command = [SURE_DIR+"/normal",
                   "-w", self.path,
                   "-i", "uniform_"+str(n)+".ply",
                   "--neighborhood", "30"]
        print(*command)
        p = subprocess.Popen(command)
        p.wait()

    def orientMVS(self):

        import open3d as o3d

        # pc=o3d.io.read_point_cloud(os.path.join(self.path,"openMVS","densify_file.ply"))

        data=np.load(os.path.join(self.path,"openMVS","densify_file.npz"))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(data["points"])
        pc.normals = o3d.utility.Vector3dVector(data["sensor_position"]-data["points"])
        pc.colors = o3d.utility.Vector3dVector(data["colors"]/256)
        self.pc  = pc
        # o3d.io.write_point_cloud(os.path.join(path, "mvs_sensor.ply"), pcd)

        self.trans_a=np.loadtxt(os.path.join(self.path,"Ignatius_trans.txt"))



        pc=pc.transform(self.trans_a)

        trans = np.eye(4,4)
        trans[0,3]=-0.2
        trans[1,3]=-0.03
        trans[2,3]=0.03

        self.trans_b=trans

        pc=pc.transform(self.trans_b)
        o3d.io.write_point_cloud(os.path.join(self.path,"openMVS","densify_file_trans.ply"),pc)


        a=5

    def cropMVS(self):

        # pc=o3d.io.read_point_cloud(os.path.join(self.path,"openMVS","densify_file_trans.ply"))

        vol = o3d.visualization.read_selection_polygon_volume(os.path.join(self.path,"Ignatius.json"))
        self.pc = vol.crop_point_cloud(self.pc)

        o3d.io.write_point_cloud(os.path.join(self.path,"openMVS","densify_file_trans_cropped.ply"),self.pc)

    def icpMVS(self):
        ## fully allign MVS to LiDAR

        # source = o3d.io.read_point_cloud(os.path.join(self.path,"openMVS","densify_file_trans_cropped.ply"))
        source = self.pc
        target = o3d.io.read_point_cloud(os.path.join(self.path,"Ignatius_gt.ply"))

        trans_init = np.eye(4,4)

        max_correspondence_distance = 0.1

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance, init=trans_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)

        self.trans_c = reg_p2p.transformation


        source=source.transform(self.trans_c)


        o3d.io.write_point_cloud(os.path.join(self.path,"openMVS","densify_file_final.ply"),source)

        self.trans=np.matmul(self.trans_b,self.trans_a)
        self.trans=np.matmul(self.trans_c,self.trans)
        # export trans
        np.savetxt(os.path.join(self.path,"openMVS","final_trans.txt"),self.trans)

        # draw_registration_result(source, target, reg_p2p.transformation)

    def translateMvsToCenter(self):

        data=np.load(os.path.join(self.path,"openMVS","densify_file.npz"))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(data["points"])
        pc.normals = o3d.utility.Vector3dVector(data["sensor_position"]-data["points"])
        pc.colors = o3d.utility.Vector3dVector(data["colors"]/256)

        trans=np.loadtxt(os.path.join(self.path,"openMVS","final_trans.txt"))
        pc=pc.transform(trans)

        vol = o3d.visualization.read_selection_polygon_volume(os.path.join(self.path,"Ignatius.json"))
        pc = vol.crop_point_cloud(pc)

        mesh = o3d.io.read_triangle_mesh(os.path.join(self.path,"all_poisson.off"))

        # one should actually do
        # center=mesh.get_axis_aligned_bounding_box().get_center()
        # see in translateLidarToCenter()

        center=mesh.get_center()

        mesh=mesh.translate(-center)
        pc=pc.translate(-center)

        scale=np.max(mesh.get_max_bound())

        mesh=mesh.scale(1/scale,[0,0,0])
        pc=pc.scale(1/scale,[0,0,0])


        o3d.io.write_point_cloud(os.path.join(self.path,"unit_mvs_alligned","mvs.ply"),pc)
        o3d.io.write_triangle_mesh(os.path.join(self.path,"unit_mvs_alligned","mesh.off"),mesh)

        a=5


    def translateLidarToCenter(self):

        os.makedirs(os.path.join(self.path,"unit_lidar_alligned"),exist_ok=True)


        mesh = o3d.io.read_triangle_mesh(os.path.join(self.path,"all_poisson.off"))
        center=mesh.get_axis_aligned_bounding_box().get_center()
        # center=mesh.get_center()

        mesh = mesh.translate(-center)
        ind = np.argmax(mesh.get_max_bound())
        scale = np.abs(mesh.get_min_bound()[ind]) + np.abs(mesh.get_max_bound()[ind])
        mesh=mesh.scale(1/scale,[0,0,0])

        o3d.io.write_triangle_mesh(os.path.join(self.path,"unit_lidar_alligned","mesh.off"),mesh)

        for file in glob.glob(os.path.join(self.path,"is_ori","*.ply")):
            pc = o3d.io.read_point_cloud(file)
            vol = o3d.visualization.read_selection_polygon_volume(os.path.join(self.path, "Ignatius.json"))
            pc = vol.crop_point_cloud(pc)
            pc = pc.translate(-center)
            pc = pc.scale(1 / scale, [0, 0, 0])
            o3d.io.write_point_cloud(os.path.join(self.path, "unit_lidar_alligned", os.path.basename(file)), pc)


        a=5

    def scanMVS(self):

        subfolder = "unit_lidar_alligned"

        cams=5
        # n=250000
        # n=374000
        n=1400005
        command = [SURE_DIR+"/scan",
                   "-w", self.path,
                   "-i", str(os.path.join(subfolder,"mesh.off")),
                   "-o", str(os.path.join(subfolder,"scan_"+str(n))),
                   "--cameras", str(cams),
                   "--points", str(n),
                   "--noise", str(stddev),
                   # "--outliers", str(0.01),
                   "--normal_method", "jet",
                   "--normal_orientation", "1",
                   "--normal_neighborhood", "30",
                   "--export", "all",
                   "-e","n"]
        # TODO: export this scan with sensors as normal field
        print(*command)
        p = subprocess.Popen(command)
        p.wait()

    def scanShape(self):
        # make the scans
        config = configparser.ConfigParser()
        normal_type = "4"


        conf=5
        conf_path = os.path.join(self.path,"confs","bumps_{}.cnf".format(str(conf)))
        print("Read scanning configuration from: " + conf_path)
        config.read(conf_path)
        infile = os.path.join(self.path,"mesh.mpu")
        # if (args.normal_type == 1):
        #     outfile = args.working_dir + "scans/with_normals/" + args.scene_conf + ".ply"
        # elif (args.normal_type == 4):
        #     outfile = args.working_dir + "scans/with_sensor/" + args.scene_conf + ".ply"
        outfile = os.path.join(self.path,"berger.ply")
        sensor_file = args.working_dir + "scans/" + args.scene_conf + "_sensor"
        pathdir = os.path.join(args.user_dir, args.reconbench_dir)

        command = []

        command.append(pathdir + "/" + config.get("uniform", "exec_name"))
        command.append(pathdir)
        command.append(infile)
        command.append(outfile)
        command.append(sensor_file)

        # required
        command.append(config.get("uniform", "camera_res_x"))
        command.append(config.get("uniform", "camera_res_y"))
        command.append(config.get("uniform", "scan_res"))

        # optional

        if config.has_option("uniform", "min_range"):
            command.append("min_range")
            command.append(config.get("uniform", "min_range"))

        if config.has_option("uniform", "max_range"):
            command.append("max_range")
            command.append(config.get("uniform", "max_range"))

        if config.has_option("uniform", "num_stripes"):
            command.append("num_stripes")
            command.append(config.get("uniform", "num_stripes"))

        if config.has_option("uniform", "laser_fov"):
            command.append("laser_fov")
            command.append(config.get("uniform", "laser_fov"))

        if config.has_option("uniform", "peak_threshold"):
            command.append("peak_threshold")
            command.append(config.get("uniform", "peak_threshold"))

        if config.has_option("uniform", "std_threshold"):
            command.append("std_threshold")
            command.append(config.get("uniform", "std_threshold"))

        if config.has_option("uniform", "additive_noise"):
            command.append("additive_noise")
            command.append(config.get("uniform", "additive_noise"))

        if config.has_option("uniform", "outlier_percentage"):
            command.append("outlier_percentage")
            command.append(config.get("uniform", "outlier_percentage"))

        if config.has_option("uniform", "laser_smoother"):
            command.append("laser_smoother")
            command.append(config.get("uniform", "laser_smoother"))

        if config.has_option("uniform", "registration_error"):
            command.append("registration_error")
            command.append(config.get("uniform", "registration_error"))

        # if config.has_option("uniform", "normal_type"):
        #         command.append("normal_type")
        #         command.append(config.get("uniform", "normal_type"))
        command.append("normal_type")
        command.append(str(normal_type))

        if config.has_option("uniform", "pca_knn"):
            command.append("pca_knn")
            command.append(config.get("uniform", "pca_knn"))

        if config.has_option("uniform", "random_sample_rotation"):
            command.append("random_sample_rotation")
            command.append(config.get("uniform", "random_sample_rotation"))

        print(*command)
        subprocess.check_call(command)


if __name__ == "__main__":

    path = '/home/rsulzer/data/Ignatius/'
    ign = Ignatius(path=path)

    # ign.translateLidarToCenter()

    # ign.orientMVS()
    # ign.cropMVS()
    # ign.icpMVS()

    # ign.translateToCenter()


    # ign.scanMVS()

    ign.makeLearning()








