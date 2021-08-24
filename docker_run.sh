#!/bin/bash
docker run -it \
      --runtime=nvidia \
      --shm-size 32G \
      -e NVIDIA_VISIBLE_DEVICES=$2 \
      --name=$1 \
      -v /mnt/ssd0/jiizero/:/mnt/ssd0/jiizero/ \
      -v /mnt/hdd0/jiizero/:/mnt/hdd0/jiizero/ \
      pitlover/anaconda3:pytorch-10.0-cudnn7 \
      /bin/bash

# sh docker_run.sh jiizero 0 : jiizero 이름으로 gpu 0 번을 쓸 것

conda create --name confeval \
    python=3 scipy=1.5.0 numpy=1.18.5 scikit-learn=0.23.1 h5py=2.10.0 \
    pycocotools=2.0.1 tqdm=4.46.1 easydict=1.9 pytorch=1.5.1 \
    torchvision=0.6.1 cudatoolkit=10.2 pytorch-lightning=0.8.4 \
    -c pytorch -c conda-forge -y
conda activate confeval
