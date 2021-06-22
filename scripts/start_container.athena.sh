#!/usr/bin/env bash
HOMEDIR="/home/anotchenko"
CODE_DIR="$HOMEDIR/Projects/partseg"
SCANNET_DIR="$HOMEDIR/Datasets/scannet"
LOGS="$HOMEDIR/logs-part-segmentation"
docker run -it --rm \
        --runtime='nvidia' \
         --cpuset-cpus='20-27' \
        --gpus=all \
        -p 8855:8888 \
        --shm-size=32g \
        -v $CODE_DIR:/code \
        -v $SCANNET_DIR:/scannet:ro \
        -v $LOGS/athena:/logs \
        'anotchenko/partseg:latest'