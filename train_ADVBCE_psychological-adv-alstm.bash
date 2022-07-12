#!/bin/bash

cd /workspace/ALSTM

export PYTHONPATH=.

/opt/conda/envs/psychological-adv-alstm/bin/python main.py \
--l 5 \
--u 4 \
--l2 0.3 \
--v 1 \
--la 0.01 \
--le 0.05 \
--r 0.003 \
--hi 0 \
--z 1