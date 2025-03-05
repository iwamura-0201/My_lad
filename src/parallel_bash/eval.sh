#!/bin/bash
# # test
python src/test.py ./reverse/logbert/gru_bi/BGL_test4train6/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/mlp/BGL_test4train6/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/bert/BGL_test4train6/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/gru_bi/BGL_test6train4/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/mlp/BGL_test6train4/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/bert/BGL_test6train4/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/gru_bi/BGL_test8train2/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/mlp/BGL_test8train2/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/bert/BGL_test8train2/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

# # test
python src/test.py ./reverse/logbert/gru_bi/BGL_test4train6/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/mlp/BGL_test4train6/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/bert/BGL_test4train6/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/gru_bi/BGL_test6train4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/mlp/BGL_test6train4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/bert/BGL_test6train4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/gru_bi/BGL_test8train2/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/mlp/BGL_test8train2/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/bert/BGL_test8train2/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

# # test
python src/test.py ./reverse/logbert/gru_bi/BGL_test4train6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/mlp/BGL_test4train6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/bert/BGL_test4train6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/gru_bi/BGL_test6train4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/mlp/BGL_test6train4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/bert/BGL_test6train4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/gru_bi/BGL_test8train2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
python src/test.py ./reverse/logbert/mlp/BGL_test8train2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
python src/test.py ./reverse/logbert/bert/BGL_test8train2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"


# # test
python src/test.py ./reverse/logbert/lstm_bi/BGL_test4train6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/lstm_uni/BGL_test4train6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/gru_uni/BGL_test4train6/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/lstm_bi/BGL_test6train4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/lstm_uni/BGL_test6train4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/gru_uni/BGL_test6train4/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/lstm_bi/BGL_test8train2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
python src/test.py ./reverse/logbert/lstm_uni/BGL_test8train2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
python src/test.py ./reverse/logbert/gru_uni/BGL_test8train2/seq_len_512/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"

# # test
python src/test.py ./reverse/logbert/lstm_bi/BGL_test4train6/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/lstm_uni/BGL_test4train6/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/gru_uni/BGL_test4train6/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/lstm_bi/BGL_test6train4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/lstm_uni/BGL_test6train4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/gru_uni/BGL_test6train4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/lstm_bi/BGL_test8train2/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/lstm_uni/BGL_test8train2/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/gru_uni/BGL_test8train2/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

# # test
python src/test.py ./reverse/logbert/lstm_bi/BGL_test4train6/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/lstm_uni/BGL_test4train6/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/gru_uni/BGL_test4train6/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/lstm_bi/BGL_test6train4/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/lstm_uni/BGL_test6train4/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/gru_uni/BGL_test6train4/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

python src/test.py ./reverse/logbert/lstm_bi/BGL_test8train2/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/lstm_uni/BGL_test8train2/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./reverse/logbert/gru_uni/BGL_test8train2/seq_len_32/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
