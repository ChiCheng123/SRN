#!/bin/bash

ROOT=..
export PYTHONPATH=$ROOT:$PYTHONPATH

python -u test.py \
  --config=$ROOT/config/config.json \
  --img_list=$ROOT/image_list.txt \
  --results_dir=results_dir \
  --resume=$ROOT/model/SRN.pth \
  --max_size=2100 \
  2>&1 | tee test.log

cd ./widerface_eval
rm -rf plot/baselines/Val/setting_int/*
matlab -nodesktop -nosplash -nojvm -r "wider_eval('../');quit;"
