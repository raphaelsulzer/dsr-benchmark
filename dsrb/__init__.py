from .datasets.default_dataset import DefaultDataset
from .datasets.shapenet import ShapeNet
from .datasets.modelnet10 import ModelNet10
from .datasets.ksr42 import KSR42Dataset
from .datasets.ksr42_original import KSR42Dataset_ori
from .datasets.simpleShapes import SimpleShapes
from .datasets.scalability import ScalabilityDataset
from .datasets.berger import Berger
from .datasets.robust import RobustDataset
from .datasets.thingi10k import Thingi10kDataset
from .datasets.tanksandtemples import TanksAndTemples
from .datasets.defects import DefectsDataset

### this doesn't work because it will still try to import the file when writing from dsrb import scan_settings
# from .scan_settings import scan_settings
### this doesn't work and leads to some circular import error
# from .logger import make_dsrb_logger
# from .eval import MeshEvaluator