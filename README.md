# A Survey and Benchmark for Automatic Surface Reconstruction from Point Clouds

Data and evaluation code for the paper **A Survey and Benchmark for Automatic Surface Reconstruction from Point Clouds**.


## Data

### Berget *et al*. shapes

- The watertight meshes and point clouds can be downloaded [here](https://drive.google.com/file/d/18usEYyY0A1KqbVdbwu7QDA2rH-UNRdsj/view?usp=sharing).


### ModelNet10

- The ModelNet10 models made watertight using [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus)
can be downloaded [here on Zenodo](https://zenodo.org/record/5920479#.YflZilvMLIE).
- The ModelNet10 scans used in our paper can be downloaded
[here on Zenodo](https://zenodo.org/record/5940164#.YflZolvMLIE). The dataset also includes training and evaluation
data for ConvONet, Points2Surf, Shape As Points, POCO and DGNN.

### ShapeNet*v1* (13 class subset of [Choy *et al*.](https://arxiv.org/abs/1604.00449))

- The watertight ShapeNet models can be downloaded [here](https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/watertight.zip) (provided by the authors of [ONet](https://arxiv.org/abs/1812.03828)).
- Please open an issue if you are interested in the scans used in our paper.

[//]: # (### Synthetic Rooms Dataset)

[//]: # ()
[//]: # (- The watertight scenes can be downloaded [here]&#40;https://s3.eu-central-1.amazonaws.com/avg-projects/convolutional_occupancy_networks/data/room_watertight_mesh.zip&#41; &#40;provided by the authors of [ConvONet]&#40;https://arxiv.org/abs/2003.04618&#41;&#41;.)

[//]: # (- Please open an issue if you are interested in the scans used in our paper.)

[//]: # (- The training and evaluation data for ConvONet can be downloaded here.)

[//]: # (- The training data for Shape As Points can be downloaded here.)



### Data Loading

We provide a dataloader for the above datasets in the datasets directory.

### MVS Scanning

The MVS scanning was done using [mesh-tools](https://github.com/raphaelsulzer/mesh-tools).

```bash
./scan -w path/to/workingDir -i filenameMeshToScan --export npz
```

For creating the scans used in the paper the follwing settings were used:

```bash
--points 3000 --noise 0.005 --outliers 0.0 --cameras 10
```

### Range Scanning

The range scanning procedure, originally from Berget *et al*., can be found
[here](https://github.com/raphaelsulzer/reconbench-CMake).

## Evaluation

To evaluate your method at it to `method/methods.py`
and at your data to the corresponding experiment paths.
Then run `run_learning.py` or `run_optim.py` depending on your method.
