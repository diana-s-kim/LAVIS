#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2_gram/train/pretrain_stage2.yaml
