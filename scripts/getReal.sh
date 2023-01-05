#!/bin/bash

models=("Ignatius" "scan1" "scan6" "Truck" "templeRing")
#models=("Ignatius")


declare -A methods
methods=( #["co2"]="conv_onet/2d/shapenet3000/meshes/50000/\$FILE.off"
#          ["co3"]="conv_onet/3d/shapenet3000/meshes/50000/\$FILE.off"
#          ["dgnn"]="dgnn/shapenet3000/50000/\$FILE.ply"
#          ["igr"]="igr/50000/\$FILE/plots/igr_100000_\$FILE.ply"
#          ["labatut"]="labatut/shapenet3000/50000/\$FILE_rt_5.0.ply"
#          ["lig"]="lig/meshes/50000/\$FILE.ply"
#          ["p2m"]="p2m/50000/\$FILE.ply"
#          ["poco"]="poco/shapenet3000/meshes/50000/\$FILE.ply"
#          ["poisson"]="poisson/50000/\$FILE.ply"
#          ["sap"]="sap/shapenet3000/meshes/50000/\$FILE.off"
          ["sap_optim"]="sap_optim/50000/\$FILE/vis/mesh/1600.ply")


### get reconstructions
for m in "${!methods[@]}"; do
  for model in "${models[@]}"; do
    mkdir -p "/home/rsulzer/data/real_out/paper/models/$model"
    temp=${methods[$m]}
    temp=${temp/\$FILE/$model}
    temp=${temp/\$FILE/$model}
    filename=$(basename -- "$temp")
    extension="${filename##*.}"
    scp "enpc:/mnt/raphael/real_out/$temp" "/home/rsulzer/data/real_out/paper/models/$model/$m.$extension";
  done
done


#### get ground truth and input
for model in "${models[@]}"; do
  scp "enpc:/mnt/raphael/real/input/50000/$model/scan/pointcloud.npz" "/home/rsulzer/data/real_out/paper/models/$model/input.npz";
#  scp "enpc:/mnt/raphael/real/input/50000/$model/scan/pointcloud.ply" "/home/rsulzer/data/real_out/paper/models/$model/input.ply";
#  scp "enpc:/mnt/raphael/real/input/full/$model/scan/pointcloud.ply" "/home/rsulzer/data/real_out/paper/models/$model/full.ply";
done





