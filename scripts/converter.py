import open3d as o3d
import os
import argparse
import numpy as np


class Converter:

    def __init__(self):
        pass


    def run(self,args):
        if args.mode == "pc":
            self.convert_pc(args.input, args.type)
        elif args.mode == "mesh":
            self.convert_mesh(args.input, args.type)
        else:
            print("Mode must either be pc or mesh!")




    def convert_from_npz(self,input, outtype):

        data = np.load(input)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data["points"])
        if("normals" in data.keys()):
            pcd.normals = o3d.utility.Vector3dVector(data["normals"])
        if("colors" in data.keys()):
            pcd.colors = o3d.utility.Vector3dVector(data["colors"])

        o3d.io.write_point_cloud(os.path.splitext(input)[0] + "." + outtype, pcd)

        print("Pointcloud saved to ", os.path.splitext(input)[0] + "." + outtype)

    def convert_to_npz(self,input, outtype):

        pcd = o3d.io.read_point_cloud(input)
        np.savez(os.path.splitext(input)[0] + "." + outtype, points=np.asarray(pcd.points),
                 colors=np.asarray(pcd.colors), normals=np.asarray(pcd.normals))

        print("Pointcloud saved to ", os.path.splitext(input)[0] + "." + outtype)

    def convert_pc(self, input, outtype):

        if os.path.splitext(input)[1] == ".npz":
            self.convert_from_npz(input, outtype)
            return
        if outtype == "npz":
            self.convert_to_npz(input, outtype)
            return

        pcd = o3d.io.read_point_cloud(input)
        o3d.io.write_point_cloud(os.path.splitext(input)[0] + "." + outtype, pcd)

        print("Pointcloud saved to ", os.path.splitext(input)[0] + "." + outtype)

    def convert_mesh(self, input, outtype):

        pcd = o3d.io.read_triangle_mesh(input)
        o3d.io.write_triangle_mesh(os.path.splitext(input)[0] + "." + outtype, pcd)

        print("Mesh saved to ", os.path.splitext(input)[0] + "." + outtype)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pointcloud converter')

    parser.add_argument('mode', type=str,
                        help='Convert Pointcloud (pc) or Mesh (mesh)')
    parser.add_argument('input', type=str,
                        help='Input file')
    parser.add_argument('type', type=str,
                        help='Convert to type, e.g. [obj, ply, off]')
    args = parser.parse_args()

    c=Converter()
    c.run(args)






