# python src/test_m.py ./paracheck0109065/logbert/bert/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/gru_uni/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_uni/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/gru_bi/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth ./paracheck0109065/logbert/lstm_bi/WEB3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:1"

python src/test.py ./paracheck0109065/logbert/lstm_uni/ART_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:0" 
# python src/test.py ./paracheck0109065/logbert/bert/ART2_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1"
python src/test.py ./paracheck0109065/logbert/lstm_uni/ART3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 64 "cuda:3" 
# python src/test.py ./paracheck0109065/logbert/bert/ART4_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 1 "cuda:1"
