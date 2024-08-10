#!/bin/bash


DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name train-epoch-1000"
PYARGS="$PYARGS --data_path $DATA/datasets/marson_prepped.h5ad" #sciplex_prepped.h5ad marson_prepped.h5ad
PYARGS="$PYARGS --dataset marson"

PYARGS="$PYARGS --scCADE_model_checkpoint_path scCADE_model_checkpoint_path"
PYARGS="$PYARGS --GraphVCI_model_checkpoint_path GraphVCI_model_checkpoint_path"


PYARGS="$PYARGS --gpu 0" #PYARGS="$PYARGS --cpu"
PYARGS="$PYARGS --seed 0"

PYARGS="$PYARGS --max_epochs 1000"
PYARGS="$PYARGS --batch_size 128"


python main.py $PYARGS
