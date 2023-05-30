import logging
import numpy as np
import trimesh
import sys, os
from scipy.spatial import cKDTree
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libmesh import check_mesh_contains
import pandas as pd
from tqdm import tqdm
import glob
import open3d as o3d
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from plyfile import PlyData

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







class MeshEvaluator:
    ''' Mesh evaluation class.

    It handles the mesh evaluation process.

    Args:
        n_points (int): number of points to be used for evaluation
    '''

    def __init__(self, n_points=100000, mesh_region_growing=None,logger=None):
        self.n_points = n_points
        self.mesh_region_growing = mesh_region_growing
        self.logger = logger if logger else logging.getLogger()
        # Maximum values for bounding box [-0.5, 0.5]^3
        # self.EMPTY_PCL_DICT = {
        #     'completeness': np.sqrt(3),
        #     'accuracy': np.sqrt(3),
        #     'completeness2': 3,
        #     'accuracy2': 3,
        #     'chamfer': 6,
        #     'hausdorff': 1000,
        #     'watertight': 0,
        #     'boundary_edges': 0,
        #     'non-manifold_edges': 0
        # }
        #
        # self.EMPTY_PCL_DICT_NORMALS = {
        #     'normals completeness': -1.,
        #     'normals accuracy': -1.,
        #     'normals': -1.,
        # }

    def distance_p2p(self,points_src, normals_src, points_tgt, normals_tgt):
        ''' Computes minimal distances of each point in points_src to points_tgt.

        Args:
            points_src (numpy array): source points
            normals_src (numpy array): source normals
            points_tgt (numpy array): target points
            normals_tgt (numpy array): target normals
        '''
        kdtree = cKDTree(points_tgt)
        dist, idx = kdtree.query(points_src) # dist is Euclidean distance, non-negative

        if normals_src is not None and normals_tgt is not None:
            normals_src = \
                normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
            normals_tgt = \
                normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

            normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
            # Handle normals that point into wrong direction gracefully
            # (mostly due to mehtod not caring about this in generation)
            normals_dot_product = np.abs(normals_dot_product)
        else:
            normals_dot_product = np.array(
                [np.nan] * points_src.shape[0], dtype=np.float32)
        return dist, normals_dot_product

    def distance_p2m(self,points, mesh):
        ''' Compute minimal distances of each point in points to mesh.

        Args:
            points (numpy array): points array
            mesh (trimesh): mesh

        '''
        _, dist, _ = trimesh.proximity.closest_point(mesh, points)
        return dist

    def get_threshold_percentage(self,dist, thresholds):
        ''' Evaluates a point cloud.

        Args:
            dist (numpy array): calculated distance
            thresholds (numpy array): threshold values for the F-score calculation
        '''
        in_threshold = [
            (dist <= t).mean() for t in thresholds
        ]
        return in_threshold

    def compute_iou(self,occ1, occ2):
        ''' Computes the Intersection over Union (IoU) value for two sets of
        occupancy values.

        Args:
            occ1 (tensor): first set of occupancy values
            occ2 (tensor): second set of occupancy values
        '''
        # occ1 = np.asarray(occ1)
        # occ2 = np.asarray(occ2)

        # Put all data in second dimension
        # Also works for 1-dimensional data
        if occ1.ndim >= 2:
            occ1 = occ1.reshape(occ1.shape[0], -1)
        if occ2.ndim >= 2:
            occ2 = occ2.reshape(occ2.shape[0], -1)

        # Convert to boolean values
        occ1 = (occ1 >= 0.5)
        occ2 = (occ2 >= 0.5)

        # Compute IOU
        area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
        area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

        iou = (area_intersect / area_union)

        return iou

    def getSurfaceComplexity(self,model,md,method):
        if method['name'] == 'ksr':
            pp = method.get("grid", "")
            if len(pp) > 0:
                pp = '{}{}{}'.format(pp[0], pp[1], pp[2])
        else:
            pp = method.get("prioritise_planes", "")


        if not method["name"] == "coacd":
            # filename = model[method["name"]]["surface"].format(method["k"],pp)
            # with open(filename, 'r') as f:
            #     f.readline()
            #     md["surf_facets"] = int(f.readline().split()[1])

            filename = os.path.splitext(model[method["name"]]["surface"].format(method["k"],pp))[0]+"_colored.ply"
            plydata = PlyData.read(filename)
            md["surf_facets"] = len(plydata["face"]['vertex_indices'])

            filename = model[method["name"]]["surface"].format(method["k"],pp)
            command = [self.mesh_region_growing, str(filename), str(os.path.dirname(filename))+"/"]
            print("Run command ", *command)
            p = subprocess.check_output(command)
            md["surf_regions"] = int(p.decode("utf-8").split(':')[1])

        if not method["name"] == "qem":
            filename = os.path.join(os.path.dirname(model[method["name"]]["partition"].format(method["k"],pp)),"in_cells.ply")
            with open(filename, 'r') as f:
                f.readline()
                f.readline()
                md["in_cells"]= int(f.readline().split(":")[-1])




    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt,
                  points_iou, occ_tgt):
        ''' Evaluates a mesh.

        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        '''
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:

            if self.n_points == "input":
                n_points = pointcloud_tgt.shape[0]
                pointcloud, idx = mesh.sample(n_points, return_index=True)
            else:
                pointcloud, idx = mesh.sample(self.n_points, return_index=True)

            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]
        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))
            components = 0

        out_dict = self.eval_pointcloud(
            pointcloud, pointcloud_tgt, normals, normals_tgt)

        # components =
        out_dict["components"] = mesh.body_count

        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            o3dmesh = mesh.as_open3d

            if(o3dmesh.is_edge_manifold(allow_boundary_edges=False)):
                out_dict['boundary_edges'] = 0
                out_dict['non-manifold_edges'] = 0
                out_dict['watertight'] = 1
                occ = check_mesh_contains(mesh, points_iou)
                out_dict['iou'] = self.compute_iou(occ, occ_tgt)
                return  out_dict


            out_dict['boundary_edges'] = np.asarray(o3dmesh.get_non_manifold_edges(allow_boundary_edges=False)).shape[0]
            out_dict['non-manifold_edges'] = np.asarray(o3dmesh.get_non_manifold_edges(allow_boundary_edges=True)).shape[0]

            out_dict['boundary_edges'] =out_dict['boundary_edges']-out_dict['non-manifold_edges']


            # boundary edges that are not in non-manifold edges

            # if(out_dict['boundary_edges'].shape[0]>0):
            #     not_in = np.invert(np.isin(out_dict['non-manifold_edges'], out_dict['boundary_edges']).all(axis=1))
            #     out_dict['boundary_edges'] = out_dict['boundary_edges'][not_in,:]

            if(out_dict['boundary_edges']>0):
                # print("non watertight mesh!")
                out_dict['watertight'] = 0
                out_dict['iou'] = 0.0
            else:
                out_dict['watertight'] = 1
                occ = check_mesh_contains(mesh, points_iou)
                out_dict['iou'] = self.compute_iou(occ, occ_tgt)

            # out_dict['boundary_edges'] = out_dict['boundary_edges'].shape[0]
            # out_dict['non-manifold_edges'] = out_dict['non-manifold_edges'].shape[0]

        else:
            print("Non valid mesh!!!\n")
            out_dict['boundary_edges'] = 0
            out_dict['non-manifold_edges'] = 0
            out_dict['watertight'] = 0
            out_dict['iou'] = 0.0

        return out_dict

    # def color_pointcloud(self, pointcloud, pointcloud_tgt,normals,normals_tgt,accuracy,completeness,percentile=(2,98)):
    def color_pointcloud(self, pointcloud, pointcloud_tgt, accuracy, completeness, percentile=(2,98)):

        # TODO: eventually it would probably be good to have the same scale per object, even when reconstruct with different methods
        # one simple way would be to concatenate all accuracies (and completeness respectively) per object and then apply
        # this function after all object have been treated. this way the accuracy and completeness percentiles are taken over all methods and
        # the color gradient will be on the same scale

        # nbins=50
        # bins=np.linspace(accuracy.min(),accuracy.max(),nbins)
        # accuracy=np.digitize(accuracy,bins)

        # look up cmaps here: https://matplotlib.org/stable/tutorials/colors/colormaps.html, use _r to reverse them
        cmap = 'hot_r'

        pcda = o3d.geometry.PointCloud()
        pcda.points = o3d.utility.Vector3dVector(pointcloud)
        # pcda.normals = o3d.utility.Vector3dVector(normals)
        cols=MplColorHelper(cmap, np.percentile(accuracy,percentile[0]), np.percentile(accuracy,percentile[1])).get_rgb(accuracy)
        # cols=MplColorHelper(cmap, accuracy.min(), accuracy.max()).get_rgb(accuracy)
        pcda.colors = o3d.utility.Vector3dVector(cols[:,:3])

        pcdc = o3d.geometry.PointCloud()
        pcdc.points = o3d.utility.Vector3dVector(pointcloud_tgt)
        # pcdc.normals = o3d.utility.Vector3dVector(normals_tgt)
        cols=MplColorHelper(cmap, np.percentile(completeness,percentile[0]), np.percentile(completeness,percentile[1])).get_rgb(completeness)
        # cols=MplColorHelper(cmap, completeness.min(), completeness.max()).get_rgb(completeness)
        pcdc.colors = o3d.utility.Vector3dVector(cols[:,:3])

        return pcda, pcdc


    def eval_pointcloud(self, pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None,
                        thresholds=np.linspace(1. / 1000, 1, 1000),color=True):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        '''
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            logger.warning('Empty pointcloud / mesh!')
            # out_dict = self.EMPTY_PCL_DICT.copy()
            # if normals is not None and normals_tgt is not None:
            #     out_dict.update(self.EMPTY_PCL_DICT_NORMALS)
            return dict()

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from the predicted point cloud
        completeness, completeness_normals = self.distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        # recall = get_threshold_percentage(completeness, thresholds)
        # completeness2 = completeness ** 2

        completeness_max = completeness.max()
        completeness_mean = completeness.mean()
        # completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are the points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = self.distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        # precision = get_threshold_percentage(accuracy, thresholds)
        # accuracy2 = accuracy ** 2

        accuracy_max = accuracy.max()
        accuracy_mean = accuracy.mean()
        # accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        # chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamfer = 0.5 * (completeness_mean + accuracy_mean)

        hausdorff = max(completeness_max,accuracy_max)

        # F-Score
        # F = [
        #     2 * precision[i] * recall[i] / (precision[i] + recall[i])
        #     for i in range(len(precision))
        # ]

        factor=1000
        out_dict = {
            'HD_gt->r (x10^3)': completeness_max*factor,
            'HD_r->gt (x10^3)': accuracy_max*factor,
            'Hausdorff (x10^3)' : hausdorff*factor,
            'CD_gt->r (x10^3)': completeness_mean*factor,
            'CD_r->gt (x10^3)': accuracy_mean*factor,
            'Chamfer (x10^3)': chamfer*factor
            # 'completeness': completeness_mean,
            # 'accuracy': accuracy_mean,
            # 'normals completeness': completeness_normals,
            # 'normals accuracy': accuracy_normals,
            # 'normals': normals_correctness,
            # 'completeness2': completeness2,
            # 'accuracy2': accuracy2,
            # 'chamfer-L2': chamferL2,
            # 'f-score': F[9],  # threshold = 1.0%
            # 'f-score-15': F[14],  # threshold = 1.5%
            # 'f-score-20': F[19],  # threshold = 2.0%
        }

        if color:
            # out = self.color_pointcloud(pointcloud,pointcloud_tgt,normals,normals_tgt,accuracy,completeness)
            out = self.color_pointcloud(pointcloud,pointcloud_tgt,accuracy,completeness)
            out_dict["accuracy_pointcloud"] = out[0]
            out_dict["completeness_pointcloud"] = out[1]

        return out_dict


    def eval(self, models, inpath="", outpath="", transform=False, method=None,export=True):

        self.eval_dicts=[]

        if not len(models):
            print("ERROR: no models to evaluate")
            return None

        for m in models:

            try:
                if method["name"] == "POCO~\cite{boulch2022poco}":
                    files = glob.glob(os.path.join(inpath,m["model"]+"*"))
                elif method["name"] == "P2S~\cite{points2surf}":
                    files = glob.glob(os.path.join(inpath, m["class"]+"_"+m["model"] + "*"))
                elif method["name"] in ["ksr","abspy","coacd","qem"]:
                    # files = glob.glob(os.path.join(outpath, m["model"], m["class"], "surface*"))
                    if method['name'] == 'ksr':
                        pp = method.get("grid", "")
                        if len(pp) > 0:
                            pp = '{}{}{}'.format(pp[0], pp[1], pp[2])
                    else:
                        pp = method.get("prioritise_planes", "")
                    files = [m[method["name"]]["surface"].format(method["k"],pp)]
                else:
                    files = glob.glob(os.path.join(inpath,m["class"],m["model"]+"*"))

                assert(len(files)>0)

                mesh_files = [file for file in files
                         if os.path.isfile(file)]

                mesh = trimesh.load(mesh_files[0], process=False)

                if(transform):
                    R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
                    mesh.vertices = np.matmul(mesh.vertices, R)

                pointcloud = np.load(m["eval"]["pointcloud"])
                pointcloud_tgt = pointcloud["points"]
                normals_tgt = pointcloud["normals"]
                points = np.load(m["eval"]["occ"])
                points_tgt = points['points']
                occ_tgt = np.unpackbits(points['occupancies'])

                eval_dict_mesh = self.eval_mesh(
                    mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)

                ## get AABB of all points to compute AABB diagonal for scaling the n_sample_points_per_area value
                ppmin = pointcloud_tgt.min(axis=0)
                ppmax = pointcloud_tgt.max(axis=0)
                diag = np.linalg.norm(ppmax - ppmin, ord=2, axis=0)
                scale = diag

                md = {}
                md["class"] = m["class"]
                md["model"] = m["model"]
                md["iou"] = eval_dict_mesh["iou"]
                md["Chamfer (x10^3)"] = eval_dict_mesh["Chamfer (x10^3)"] / scale
                md["CD_gt->r (x10^3)"] = eval_dict_mesh["CD_gt->r (x10^3)"] / scale
                md["CD_r->gt (x10^3)"] = eval_dict_mesh["CD_r->gt (x10^3)"] / scale

                md["Hausdorff (x10^3)"] = eval_dict_mesh["Hausdorff (x10^3)"] / scale
                md["HD_gt->r (x10^3)"] = eval_dict_mesh["HD_gt->r (x10^3)"] / scale
                md["HD_r->gt (x10^3)"] = eval_dict_mesh["HD_r->gt (x10^3)"] / scale

                md["components"] = eval_dict_mesh["components"]
                md["boundary_edges"] = eval_dict_mesh["boundary_edges"]
                md["non-manifold_edges"] = eval_dict_mesh["non-manifold_edges"]
                md["watertight"] = eval_dict_mesh["watertight"]

                md["surf_facets"] = mesh.faces.shape[0]
                md["in_cells"] = 0
                if(method["name"] in ["ksr","abspy","coacd","qem"]):
                    self.getSurfaceComplexity(m,md,method)

                self.eval_dicts.append(md)

                if export:
                    assert(eval_dict_mesh["accuracy_pointcloud"] is not None and eval_dict_mesh["completeness_pointcloud"] is not None)

                    if method['name'] == 'ksr':
                        pp = method.get("grid", "")
                        if len(pp) > 0:
                            pp = '{}{}{}'.format(pp[0], pp[1], pp[2])
                    else:
                        pp = method.get("prioritise_planes", "")
                    outfile = os.path.join(os.path.dirname(m[method["name"]]["surface"].format(method["k"],pp)),"accuracy_pc.ply")
                    o3d.io.write_point_cloud(outfile,eval_dict_mesh["accuracy_pointcloud"])

                    outfile = os.path.join(os.path.dirname(m[method["name"]]["surface"].format(method["k"],pp)),"completeness_pc.ply")
                    o3d.io.write_point_cloud(outfile,eval_dict_mesh["completeness_pointcloud"])


            except Exception as e:
                raise
                print("\nERROR: {}".format(e))
                print("Skipping {}/{}".format(m["class"], m["model"]))

                # return None


        eval_df_full = pd.DataFrame(self.eval_dicts)

        if method['name'] == 'ksr':
            pp = method.get("grid", "")
            if len(pp) > 0:
                pp = '{}{}{}'.format(pp[0], pp[1], pp[2])
        else:
            pp = method.get("prioritise_planes", "")
        op = os.path.join(outpath, "surface_full_{}{}{}.csv".format(method["name"],method["k"],pp))
        os.makedirs(os.path.join(outpath),exist_ok=True)
        eval_df_full.to_csv(op,float_format='%.3g')
        eval_df_class = eval_df_full.groupby(by=['class']).mean(numeric_only=True)
        eval_df_class.loc['mean'] = eval_df_full.mean(numeric_only=True)

        return eval_df_full, eval_df_class



