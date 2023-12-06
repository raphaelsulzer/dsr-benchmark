#!/bin/bash

# this file is adapted from https://github.com/drprojects/point_geometric_features/blob/main/install.sh

# Local variables
PROJECT_NAME=dsr
PYTHON=3.10.10

# Installation script for Anaconda3 environments
echo "#############################################"
echo "#                                           #"
echo "#        DSR Benchmark Installer            #"
echo "#                                           #"
echo "#############################################"
echo "adapted from https://github.com/drprojects/point_geometric_features/blob/main/install.sh"
echo


echo "_______________ Prerequisites _______________"
echo "  - conda"
echo


echo "____________ Pick conda install _____________"
echo
# Recover the path to conda on your machine
CONDA_DIR=`realpath ~/miniconda3`
if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
  CONDA_DIR=`realpath ~/anaconda3`
fi

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide you conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda conda found at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh
echo
echo


echo "________________ Installation _______________"
echo

# Create a conda environment from yml
conda create -y --name $PROJECT_NAME python=$PYTHON

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate ${PROJECT_NAME}

# Dependencies
conda install -y -c conda-forge mamba
mamba install -y conda-forge::trimesh conda-forge::vedo anaconda::cython
pip install open3d
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install networkx[default]
python setup_libmesh_convonet.py build_ext --inplace
pip install --no-build-isolation .