#!/bin/bash

ROOT=..
export PYTHONPATH=$ROOT:$PYTHONPATH

srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 \
    --job-name=SRN \
python -u test.py \
  --config=$ROOT/config/config.json \
  --img_list=$ROOT/image_list.txt \
  --results_dir=results_dir \
  --resume=$ROOT/model/try1.pth \
  --max_size=1700 \
  2>&1 | tee test.log

cd ./widerface_eval
rm -rf plot/baselines/Val/setting_int/*
matlab -nodesktop -nosplash -nojvm -r "wider_eval('../');quit;"
