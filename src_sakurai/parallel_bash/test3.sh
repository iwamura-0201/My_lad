#!/bin/bash


# # test
python src/test.py ./outputs/logbert/lstm_bi/BGL_train4test6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/lstm_uni/BGL_train4test6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/gru_uni/BGL_train4test6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 

python src/test.py ./outputs/logbert/lstm_bi/HDFS_train4test6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/lstm_uni/HDFS_train4test6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/gru_uni/HDFS_train4test6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 

python src/test.py ./outputs/logbert/lstm_bi/TBird_train4test6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/lstm_uni/TBird_train4test6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/gru_uni/TBird_train4test6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 

python src/test.py ./outputs/logbert/lstm_bi/BGL_train6test4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/lstm_uni/BGL_train6test4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/gru_uni/BGL_train6test4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 

python src/test.py ./outputs/logbert/lstm_bi/HDFS_train6test4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2"
python src/test.py ./outputs/logbert/lstm_uni/HDFS_train6test4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2"
python src/test.py ./outputs/logbert/gru_uni/HDFS_train6test4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 

python src/test.py ./outputs/logbert/lstm_bi/TBird_train6test4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/lstm_uni/TBird_train6test4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/gru_uni/TBird_train6test4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 

python src/test.py ./outputs/logbert/lstm_bi/BGL_train8test2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2"
python src/test.py ./outputs/logbert/lstm_uni/BGL_train8test2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2"
python src/test.py ./outputs/logbert/gru_uni/BGL_train8test2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2"

python src/test.py ./outputs/logbert/lstm_bi/HDFS_train8test2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/lstm_uni/HDFS_train8test2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/gru_uni/HDFS_train8test2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 

python src/test.py ./outputs/logbert/lstm_bi/TBird_train8test2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/lstm_uni/TBird_train8test2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 
python src/test.py ./outputs/logbert/gru_uni/TBird_train8test2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2" 

