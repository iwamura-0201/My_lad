#!/bin/bash

# # test
python src/test.py ./paracheck01035/logbert/bert/HDFS_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
python src/test.py ./paracheck01035/logbert/lstm_uni/HDFS_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
python src/test.py ./paracheck01035/logbert/lstm_bi/HDFS_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
python src/test.py ./paracheck01035/logbert/gru_uni/HDFS_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
python src/test.py ./paracheck01035/logbert/gru_bi/HDFS_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
python src/test.py ./paracheck01035/logbert/mlp/HDFS_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"

