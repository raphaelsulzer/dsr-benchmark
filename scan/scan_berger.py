import subprocess, os
import configparser
from munch import Munch

def scanShape(args):
    # make the scans
    config = configparser.ConfigParser()
    conf_path = os.path.join(args.working_dir,"confs/bumps_" + str(args.conf) + ".cnf")
    print("Read scanning configuration from: " + conf_path)
    config.read(conf_path)
    infile = os.path.join(args.working_dir,"mesh.mpu")
    if (args.normal_type == 1):
        outfile = os.path.join(args.working_dir,"berger_normals.ply")
    elif (args.normal_type == 4):
        outfile = os.path.join(args.working_dir,"berger_sensors.ply")

    sensor_file = os.path.join(args.working_dir,"sensors")
    # pathdir = os.path.join(args.user_dir, args.reconbench_dir)
    pathdir = args.program_dir

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
    command.append(str(args.normal_type))

    if config.has_option("uniform", "pca_knn"):
        command.append("pca_knn")
        command.append(config.get("uniform", "pca_knn"))

    if config.has_option("uniform", "random_sample_rotation"):
        command.append("random_sample_rotation")
        command.append(config.get("uniform", "random_sample_rotation"))

    print(*command)
    subprocess.check_call(command)

import sys
sys.path.append("/home/rsulzer/python/ksr-benchmark/")
from importTest import testvar
print(testvar)

from tqdm import tqdm

args = Munch()
args.working_dir = "/home/adminlocal/PhD/data/benchmark/scan_example"
args.program_dir = "/home/adminlocal/PhD/cpp/reconbench-CMake/build/release"
# args.working_dir = "/mnt/raphael/scan_example"
# args.program_dir = "/home/raphael/cpp/reconbench-CMake/build/release"
args.conf = "5"
args.normal_type = 4
scanShape(args)
