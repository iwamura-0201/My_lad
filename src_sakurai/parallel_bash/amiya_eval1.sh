#!/bin/bash
# # test


# python src/test.py ./paracheck0109065/logbert/bert/ART_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1" 
# python src/test.py ./paracheck0109065/logbert/bert/ART2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1"
# python src/test.py ./paracheck0109065/logbert/bert/ART3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
# python src/test.py ./paracheck0109065/logbert/bert/ART4_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1"

python src/test.py ./paracheck0109065/logbert/lstm_uni/WEB1_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1" 
# python src/test.py ./paracheck0109065/logbert/bert/WEB1_2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1"
python src/test.py ./paracheck0109065/logbert/lstm_uni/WEB1_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
# python src/test.py ./paracheck0109065/logbert/bert/WEB1_4_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1"

python src/test.py ./paracheck0109065/logbert/lstm_uni/WEB2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1" 
# python src/test.py ./paracheck0109065/logbert/bert/WEB2_2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1"
python src/test.py ./paracheck0109065/logbert/lstm_uni/WEB2_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
# python src/test.py ./paracheck0109065/logbert/bert/WEB2_4_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1"

python src/test.py ./paracheck0109065/logbert/lstm_uni/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1" 
# python src/test.py ./paracheck0109065/logbert/bert/WEB3_2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1"
python src/test.py ./paracheck0109065/logbert/lstm_uni/WEB3_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
# python src/test.py ./paracheck0109065/logbert/bert/WEB3_4_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1"
python src/test.py ./paracheck0109065/logbert/lstm_bi/WEB1_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1" 
python src/test.py ./paracheck0109065/logbert/lstm_bi/WEB1_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
python src/test.py ./paracheck0109065/logbert/lstm_bi/WEB2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1" 
python src/test.py ./paracheck0109065/logbert/lstm_bi/WEB2_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 
python src/test.py ./paracheck0109065/logbert/lstm_bi/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1" 
python src/test.py ./paracheck0109065/logbert/lstm_bi/WEB3_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1" 

# python src/test_m.py ./paracheck0109065/logbert/bert/ART_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/gru_uni/ART_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_uni/ART_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/gru_bi/ART_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_bi/ART_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:0" 
# python src/test_m.py ./paracheck0109065/logbert/bert/WEB1_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/gru_uni/WEB1_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_uni/WEB1_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/gru_bi/WEB1_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_bi/WEB1_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
# python src/test_m.py ./paracheck0109065/logbert/bert/WEB2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/gru_uni/WEB2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_uni/WEB2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/gru_bi/WEB2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_bi/WEB2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2"
# python src/test_m.py ./paracheck0109065/logbert/bert/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/gru_uni/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_uni/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/gru_bi/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_bi/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:3"

# python src/test_m.py ./paracheck0109065/logbert/bert/ART3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_uni/ART3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth  64 "cuda:0" 
# python src/test_m.py ./paracheck0109065/logbert/bert/WEB1_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_uni/WEB1_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"
# python src/test_m.py ./paracheck0109065/logbert/bert/WEB2_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_uni/WEB2_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:2"
# python src/test_m.py ./paracheck0109065/logbert/bert/WEB3_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth  ./paracheck0109065/logbert/lstm_uni/WEB3_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:3"