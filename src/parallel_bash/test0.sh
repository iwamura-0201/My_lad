#!/bin/bash
# # test

python src/test.py ./paracheck01035/logbert/bert/BGL_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:0"
python src/test.py ./paracheck01035/logbert/lstm_uni/BGL_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:0"
python src/test.py ./paracheck01035/logbert/lstm_bi/BGL_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:0"
python src/test.py ./paracheck01035/logbert/gru_uni/BGL_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:0"
python src/test.py ./paracheck01035/logbert/gru_bi/BGL_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:0"
python src/test.py ./paracheck01035/logbert/mlp/BGL_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:0"

