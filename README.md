# A Survey and Benchmark of Automatic Surface Reconstruction from Point Clouds

Data and evaluation code for the paper **A Survey and Benchmark of Automatic Surface Reconstruction from Point Clouds** ([arXiv](https://arxiv.org/abs/2301.13656)).

![alt text](teaser.png)


## Data

### Berget *et al*.

- The watertight meshes and scans can be downloaded [here](https://drive.google.com/file/d/18usEYyY0A1KqbVdbwu7QDA2rH-UNRdsj/view?usp=sharing).


### ModelNet10

- The watertight ModelNet10 models can be downloaded [here on Zenodo](https://zenodo.org/record/5920479#.YflZilvMLIE).
- The ModelNet10 scans used in our paper can be downloaded
[here on Zenodo](https://zenodo.org/record/5940164#.YflZolvMLIE). The dataset also includes training and evaluation
data for ConvONet, Points2Surf, Shape As Points, POCO and DGNN.

### ShapeNet*v1* (13 class subset of [Choy *et al*.](https://arxiv.org/abs/1604.00449))

- The watertight ShapeNet models can be downloaded [here](https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/watertight.zip) (provided by the authors of [ONet](https://arxiv.org/abs/1812.03828)).
- Please open an issue if you are interested in the scans used in our paper.
- Training and evaluation data for ShapeNet can be downloaded [here](https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip) (provided by the authors of [ONet](https://arxiv.org/abs/1812.03828)).

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


## References

If you find the code or data in this repository useful, please consider citing

```bibtex
@misc{sulzer2023dsr,
  doi = {10.48550/ARXIV.2301.13656},
  url = {https://arxiv.org/abs/2301.13656},
  author = {Sulzer, Raphael and Landrieu, Loic and Marlet, Renaud and Vallet, Bruno},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Computational Geometry (cs.CG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {A Survey and Benchmark of Automatic Surface Reconstruction from Point Clouds},
  publisher = {arXiv},
  year = {2023},
}
```