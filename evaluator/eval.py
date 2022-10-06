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

def compute_iou(occ1, occ2):
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

# Maximum values for bounding box [-0.5, 0.5]^3
EMPTY_PCL_DICT = {
    'completeness': np.sqrt(3),
    'accuracy': np.sqrt(3),
    'completeness2': 3,
    'accuracy2': 3,
    'chamfer': 6,
    'watertight': 0,
    'boundary_edges': 0,
    'non-manifold_edges': 0
}

EMPTY_PCL_DICT_NORMALS = {
    'normals completeness': -1.,
    'normals accuracy': -1.,
    'normals': -1.,
}

logger = logging.getLogger(__name__)


class MeshEvaluator(object):
    ''' Mesh evaluation class.

    It handles the mesh evaluation process.

    Args:
        n_points (int): number of points to be used for evaluation
    '''

    def __init__(self, n_points=100000):
        self.n_points = n_points
        self.eval_dicts = []

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt,
                  points_iou, occ_tgt, remove_wall=False):
        ''' Evaluates a mesh.

        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        '''
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:

            # if(components > 1):
            #     mesh = mesh.split()[0]
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
                out_dict['iou'] = compute_iou(occ, occ_tgt)
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
                out_dict['iou'] = compute_iou(occ, occ_tgt)

            # out_dict['boundary_edges'] = out_dict['boundary_edges'].shape[0]
            # out_dict['non-manifold_edges'] = out_dict['non-manifold_edges'].shape[0]

        else:
            print("Non valid mesh!!!\n")
            out_dict['boundary_edges'] = 0
            out_dict['non-manifold_edges'] = 0
            out_dict['watertight'] = 0
            out_dict['iou'] = 0.0

        return out_dict

    def eval_pointcloud(self, pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None,
                        thresholds=np.linspace(1. / 1000, 1, 1000)):
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
            logger.warn('Empty pointcloud / mesh detected!')
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from the predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness ** 2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy ** 2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1 * 100,
            'f-score': F[9],  # threshold = 1.0%
            'f-score-15': F[14],  # threshold = 1.5%
            'f-score-20': F[19],  # threshold = 2.0%
        }

        return out_dict


    def eval(self, models, outpath, transform=False):

        for m in tqdm(models, ncols=50):

            try:

                # gt_mesh = trimesh.load(m["mesh"], process=False)
                # if os.path.exists(os.path.join(outpath, m["class"], m["model"]+".ply")):
                #     mesh_file = os.path.join(outpath, m["class"], m["model"]+".ply")
                # else:
                #     mesh_file = os.path.join(outpath, m["class"], m["model"] + ".off")
                # mesh_file = os.path.join(outpath, m["class"], m["model"])
                files = glob.glob(os.path.join(outpath,m["class"],m["model"]+"*"))
                mesh_files = [file for file in files
                         if os.path.isfile(file)]

                mesh = trimesh.load(mesh_files[0], process=False)

                if(transform):
                    R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
                    mesh.vertices = np.matmul(mesh.vertices, R)

                pointcloud = np.load(m["pointcloud"])
                pointcloud_tgt = pointcloud["points"]
                normals_tgt = pointcloud["normals"]
                points = np.load(m["occ"])
                points_tgt = points['points']
                occ_tgt = np.unpackbits(points['occupancies'])

                eval_dict_mesh = self.eval_mesh(
                    mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)

                md = {}
                md["class"] = m["class"]
                md["model"] = m["model"]
                md["iou"] = eval_dict_mesh["iou"]
                md["chamfer"] = eval_dict_mesh["chamfer-L1"]
                md["normal"] = eval_dict_mesh["normals"]
                md["components"] = eval_dict_mesh["components"]
                md["boundary_edges"] = eval_dict_mesh["boundary_edges"]
                md["non-manifold_edges"] = eval_dict_mesh["non-manifold_edges"]
                md["watertight"] = eval_dict_mesh["watertight"]
                if (not md["watertight"]):
                    print("Non watertight mesh {}/{}".format(m["class"], m["model"]))

                self.eval_dicts.append(md)

            except Exception as e:
                raise
                print(e)
                print("Skipping {}/{}".format(m["class"], m["model"]))

        eval_df = pd.DataFrame(self.eval_dicts)
        eval_df.to_pickle(os.path.join(outpath, "results_all.pkl"))
        eval_df_class = eval_df.groupby(by=['class']).mean()
        eval_df_class.loc['mean'] = eval_df.mean()
        eval_df_class.to_csv(os.path.join(outpath, "results.csv"))

        return eval_df_class




def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

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


def distance_p2m(points, mesh):
    ''' Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    '''
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold
