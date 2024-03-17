#!/bin/bash

cd /workspace/ALSTM

export PYTHONPATH=.

/opt/conda/envs/psychological-adv-alstm/bin/python main.py \
--l 5 \
--u 4 \
--l2 0.05 \
--v 1 \
--rl 0 \
--la 0.1 \
--le 0.05 \
--r le-3
