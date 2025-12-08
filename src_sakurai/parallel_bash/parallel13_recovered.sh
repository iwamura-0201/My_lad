#!/bin/bash

# train and val
python src/main.py bert/amiya2 default.num_workers=16 dataset.train_ratio=6 dataset.sample.seq_len=128 default.device_id="cuda:1" loss.hypersphere.bias=0 loss.mask.bias=1 dataset.sample.mask_ratio=0.65 default.dir_name="paracheck01065" dataset.reverse=False 
