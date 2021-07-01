#!/usr/bin/env bash
HOMEDIR="/home/alexandr-notchenko"
CODE_DIR="$HOMEDIR/Projects/partseg"
SCANNET_DIR="$HOMEDIR/Datasets/scan2cad"
LOGS="$HOMEDIR/logs-part-segmentation"
docker run -it --rm \
        --runtime='nvidia' \
        -p 8855:8888 \
        --shm-size=16g \
        -v $CODE_DIR:/code \
        -v $SCANNET_DIR:/scannet:ro \
        -v $LOGS/blackcat:/logs \
        --name='partseg' \
        'partseg:latest'