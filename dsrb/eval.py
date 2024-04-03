import logging, trimesh, vedo, os, sys
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
from tqdm import tqdm
import open3d as o3d

from libmesh import check_mesh_contains
from dsrb.logger import make_dsrb_logger

from pycompose import pdse

# from pymeshregiongrowing import libMRG as mrg

# silent vtk logs from vedo: https://stackoverflow.com/a/71524504
import vtk # vtk does not load if pymeshlab is already loaded. simply load pymeshlab before and there is no issue
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)

class MeshEvaluator:
    ''' Mesh evaluation class.

    It handles the mesh evaluation process.

    Args:
        n_points (int): number of points to be used for evaluation
    '''

    def __init__(self, n_points=100000, logger=None, verbosity=logging.INFO, export_colored_eval_pointclouds=False, debug_export=False):
        self.n_points = n_points

        if logger is not None:
            self.logger = logger
        else:
            self.logger = make_dsrb_logger("MESH_EVALUATOR",level=verbosity)

        self.debug_export = debug_export

        self.export_colored_eval_pointclouds = export_colored_eval_pointclouds

        if self.debug_export:
            self.logger.warning('\nDebug export activated! Turn off for faster processing.\n')


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

    def get_number_of_in_cells(self,model,md,method):

        filename = model["output"]["in_cells"].format(str(method))
        md["in_cells"] = len(vedo.load(filename).split(maxdepth=100000))


        # if os.path.isfile(filename):
        #     with open(filename, 'r') as f:
        #         f.readline()
        #         f.readline()
        #         md["in_cells"]= int(f.readline().split(":")[-1])
        # else:
        #     md["in_cells"] = -999999
        #     # self.logger.warning("{} missing for measuring surface complexity".format(filename))



    def eval_geometry(self, mesh, pointcloud_tgt, normals_tgt,
                  points_iou=None, occ_tgt=None):
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

        if points_iou is not None:
            occ = check_mesh_contains(mesh, points_iou)
            out_dict['iou'] = self.compute_iou(occ, occ_tgt)

        return out_dict

    def eval_topology(self, mesh):

        out_dict = {}
        ps = pdse(0)

        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            o3dmesh = mesh.as_open3d

            out_dict["components"] = mesh.body_count

            if(o3dmesh.is_edge_manifold(allow_boundary_edges=False)):
                out_dict['boundary_edges'] = 0
                out_dict['non-manifold_edges'] = 0
                out_dict['watertight'] = 1
                out_dict['intersection-free'] = ps.is_mesh_intersection_free(mesh.infile)

                return  out_dict

            out_dict['boundary_edges'] = np.asarray(o3dmesh.get_non_manifold_edges(allow_boundary_edges=False)).shape[0]
            out_dict['non-manifold_edges'] = np.asarray(o3dmesh.get_non_manifold_edges(allow_boundary_edges=True)).shape[0]
            out_dict['boundary_edges'] = out_dict['boundary_edges']-out_dict['non-manifold_edges']

            if(out_dict['boundary_edges']>0):
                # self.logger.warning("non watertight mesh!")
                out_dict['watertight'] = 0
                out_dict['iou'] = 0.0
            else:
                out_dict['watertight'] = 1
        else:
            self.logger.warning("Non valid mesh!\n")
            out_dict['boundary_edges'] = 0
            out_dict['non-manifold_edges'] = 0
            out_dict['watertight'] = 0
            out_dict['iou'] = 0.0

        out_dict['watertight'] = ps.is_mesh_watertight(mesh.infile)
        out_dict['intersection-free'] = ps.is_mesh_intersection_free(mesh.infile)


        return out_dict

    def color_pointcloud(self, pointcloud, pointcloud_tgt, accuracy, completeness, percentile=(2,98)):

        from fancycolor import GradientColor2D

        # TODO: eventually it would probably be good to have the same scale per object, even when reconstruct with different methods
        # one simple way would be to concatenate all accuracies (and completeness respectively) per object and then apply
        # this function after all object have been treated. this way the accuracy and completeness percentiles are taken over all methods and
        # the color gradient will be on the same scale

        # nbins=50
        # bins=np.linspace(accuracy.min(),accuracy.max(),nbins)
        # accuracy=np.digitize(accuracy,bins)

        # look up cmaps here: https://matplotlib.org/stable/tutorials/colors/colormaps.html, use _r to reverse them
        cmap = 'hot_r'

        pcdc = o3d.geometry.PointCloud()
        pcdc.points = o3d.utility.Vector3dVector(pointcloud_tgt)
        diag = np.linalg.norm(pcdc.get_min_bound() - pcdc.get_max_bound())

        pcda = o3d.geometry.PointCloud()
        pcda.points = o3d.utility.Vector3dVector(pointcloud)
        # pcda.normals = o3d.utility.Vector3dVector(normals)
        # cols=GradientColor2D(cmap, np.percentile(accuracy,percentile[0]), np.percentile(accuracy,percentile[1])).get_rgb(accuracy)
        cols = GradientColor2D(cmap,0.0001*diag,0.03*diag).get_rgb(accuracy)
        # cols=MplColorHelper(cmap, accuracy.min(), accuracy.max()).get_rgb(accuracy)
        pcda.colors = o3d.utility.Vector3dVector(cols[:,:3])

        # pcdc.normals = o3d.utility.Vector3dVector(normals_tgt)
        # cols=GradientColor2D(cmap, np.percentile(completeness,percentile[0]), np.percentile(completeness,percentile[1])).get_rgb(completeness)
        cols = GradientColor2D(cmap,0.0001*diag,0.03*diag).get_rgb(completeness)
        # cols=MplColorHelper(cmap, completeness.min(), completeness.max()).get_rgb(completeness)
        pcdc.colors = o3d.utility.Vector3dVector(cols[:,:3])

        return pcda, pcdc


    def eval_pointcloud(self,
                        pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
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
            self.logger.warning('Empty pointcloud / mesh!')
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
        accuracy, accuracy_normals = self.distance_p2p(pointcloud, normals, pointcloud_tgt, normals_tgt)
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

        factor=1
        out_dict = {
            'HD_gt->r': completeness_max*factor,
            'HD_r->gt': accuracy_max*factor,
            'Hausdorff' : hausdorff*factor,
            'CD_gt->r': completeness_mean*factor,
            'CD_r->gt': accuracy_mean*factor,
            'Chamfer': chamfer*factor,
            # 'completeness': completeness_mean,
            # 'accuracy': accuracy_mean,
            # 'normals completeness': completeness_normals,
            # 'normals accuracy': accuracy_normals,
            'Normal Consistency': normals_correctness
            # 'completeness2': completeness2,
            # 'accuracy2': accuracy2,
            # 'chamfer-L2': chamferL2,
            # 'f-score': F[9],  # threshold = 1.0%
            # 'f-score-15': F[14],  # threshold = 1.5%
            # 'f-score-20': F[19],  # threshold = 2.0%
        }

        if self.export_colored_eval_pointclouds:
            # out = self.color_pointcloud(pointcloud,pointcloud_tgt,normals,normals_tgt,accuracy,completeness)
            out = self.color_pointcloud(pointcloud,pointcloud_tgt,accuracy,completeness)
            out_dict["accuracy_pointcloud"] = out[0]
            out_dict["completeness_pointcloud"] = out[1]

        return out_dict


    def eval(self, models, method,
             outpath = None,
             eval_geometry = True,
             eval_topology = True,
             scale_with_diag = True,
             group_by_class = True,
             count_regions = False):

        if count_regions:
            from pymeshregiongrowing import libMRG as mrg

        self.eval_dicts=[]

        if not len(models):
            self.logger.error("No models to evaluate.")
            return None

        self.logger.info("Evaluate {} meshes".format(len(models)))

        for model in tqdm(models, ncols=50, file=sys.stdout):
            try:

                if not os.path.isfile(model["output"]["surface"].format(str(method))):
                    self.logger.warning("{} is not a file.".format(model["output"]["surface"].format(str(method))))
                    continue

                # mesh = trimesh.load(model["output"]["surface_simplified_triangulated"].format(str(method)), process=False)
                mesh = trimesh.load(model["output"]["surface"].format(str(method)), process=False)

                md = {}
                md["class"] = model["class"]
                md["model"] = model["model"]

                if eval_geometry:
                    pointcloud = np.load(model["eval"]["pointcloud"])
                    pointcloud_tgt = pointcloud["points"]
                    normals_tgt = pointcloud["normals"]
                    if model["eval"]["occ"] is not None:
                        points = np.load(model["eval"]["occ"])
                        points_tgt = points['points']
                        occ_tgt = np.unpackbits(points['occupancies'])
                        eval_dict_mesh = self.eval_geometry(mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)
                        md["iou"] = eval_dict_mesh["iou"]
                    else:
                        eval_dict_mesh = self.eval_geometry(mesh, pointcloud_tgt, normals_tgt)


                    ## get AABB of all points to compute AABB diagonal for scaling the n_sample_points_per_area value
                    ppmin = pointcloud_tgt.min(axis=0)
                    ppmax = pointcloud_tgt.max(axis=0)
                    diag = np.linalg.norm(ppmax - ppmin, ord=2, axis=0)


                    scale_txt = "(x10^2)"
                    if scale_with_diag:
                        scale = diag/10**2
                    else:
                        scale = 1/10**2

                    md["Chamfer {}".format(scale_txt)] = eval_dict_mesh["Chamfer"] / scale
                    md["CD_gt->r {}".format(scale_txt)] = eval_dict_mesh["CD_gt->r"] / scale
                    md["CD_r->gt {}".format(scale_txt)] = eval_dict_mesh["CD_r->gt"] / scale

                    md["Hausdorff {}".format(scale_txt)] = eval_dict_mesh["Hausdorff"] / scale
                    md["HD_gt->r {}".format(scale_txt)] = eval_dict_mesh["HD_gt->r"] / scale
                    md["HD_r->gt {}".format(scale_txt)] = eval_dict_mesh["HD_r->gt"] / scale

                    md["Normal Consistency"] = eval_dict_mesh["Normal Consistency"]

                    if self.export_colored_eval_pointclouds:
                        assert (eval_dict_mesh["accuracy_pointcloud"] is not None and eval_dict_mesh[
                            "completeness_pointcloud"] is not None)

                        outfile = os.path.join(os.path.dirname(model["output"]["surface"].format(str(method))),
                                               "accuracy_pc.ply")
                        o3d.io.write_point_cloud(outfile, eval_dict_mesh["accuracy_pointcloud"])

                        outfile = os.path.join(os.path.dirname(model["output"]["surface"].format(str(method))),
                                               "completeness_pc.ply")
                        o3d.io.write_point_cloud(outfile, eval_dict_mesh["completeness_pointcloud"])


                if eval_topology:

                    eval_dict_mesh = self.eval_topology(mesh)

                    md["components"] = eval_dict_mesh["components"]
                    md["boundary_edges"] = eval_dict_mesh["boundary_edges"]
                    md["non-manifold_edges"] = eval_dict_mesh["non-manifold_edges"]
                    md["watertight"] = eval_dict_mesh["watertight"]
                    md["intersection-free"] = eval_dict_mesh["intersection-free"]
                    md["surf_triangles"] = mesh.faces.shape[0]
                    md["surf_vertices"] = len(mesh.vertices)
                    md["surf_edges"] = len(mesh.edges)

                    if count_regions:
                        angle_th = 5.0
                        rg = mrg.mrg()
                        fname = model["output"]["surface"].format(str(method))
                        n_regions = rg.get_number_of_regions(fname,angle_th)
                        md["surf_regions"] = n_regions
                        rg.export_colored_mesh(fname[:-4]+"_regions.obj")
                        del rg

                    if os.path.isfile(model["output"]["surface_simplified"].format(str(method))):
                        simplified_mesh_file = model["output"]["surface_simplified"].format(str(method))
                    elif os.path.isfile(model["output"]["surface"].format(str(method))):
                        simplified_mesh_file = model["output"]["surface"].format(str(method))
                    else:
                        simplified_mesh_file = None

                    if simplified_mesh_file is not None:
                        # mesh = trimesh.load(model["output"]["surface_simplified"].format(str(method)), process=False, force="mesh")
                        # md["surf_simpl_triangles"] = mesh.faces.shape[0]
                        mesh = vedo.load(simplified_mesh_file)
                        self.logger.debug("Loading simplified surface for comlexity evaluation")
                        md["surf_simpl_polygons"] = len(mesh.cells())
                        # md["surf_simpl_triangles"] = mesh.faces.shape[0]
                        # md["surf_simpl_polygons"] = len(mesh.facets)
                        md["surf_simpl_vertices"] = len(mesh.vertices())
                        md["surf_simpl_edges"] = len(mesh.edges())


                    md["in_cells"] = np.nan
                    if "planes" in model:
                        if os.path.isfile(model["planes"].format(str(method))):
                            md["n_planes"] = np.load(model["planes"].format(str(method)))["group_parameters"].shape[0]

                    if "n_planes" in md.keys() and "surf_simpl_polygons" in md.keys():
                        md["plane_polygon_ratio"] = md["n_planes"]/md["surf_simpl_polygons"]

                    if "in_cells" in model["output"].keys() and os.path.isfile(model["output"]["in_cells"].format(str(method))):
                        self.get_number_of_in_cells(model,md,method)

                self.eval_dicts.append(md)




            except Exception as e:
                print(e)
                raise e
                self.logger.error("{}".format(e))
                self.logger.error("Skipping {}/{}".format(model["class"], model["model"]))

                # return None

        if not len(self.eval_dicts):
            self.logger.error("No models to evaluate found!")
            return None

        eval_df_full = pd.DataFrame(self.eval_dicts)
        eval_df_class = None

        if group_by_class:
            eval_df_class = eval_df_full.groupby(by=['class']).mean(numeric_only=True)
            eval_df_class.loc["mean"] = eval_df_full.mean(numeric_only=True)
            eval_df_full.loc["mean"] = eval_df_class.loc["mean"]
        else:
            eval_df_full.loc['mean'] = eval_df_full.mean(numeric_only=True)

        if outpath is not None:
            op = os.path.join(outpath, "{}_full.csv".format(method))
            os.makedirs(os.path.join(outpath),exist_ok=True)
            eval_df_full.to_csv(op,float_format='%.3g')

        return eval_df_full, eval_df_class
