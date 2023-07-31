#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/blip2_gram/eval/caption_artemis_opt2.7b_eval.yaml
