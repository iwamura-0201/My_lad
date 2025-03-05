#!/bin/bash
# # test
python src/test.py ./paracheck01065/logbert/bert/ART3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 16 "cuda:0"
python src/test.py ./paracheck01065/logbert/bert/ART4_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 16 "cuda:0"
python src/test.py ./paracheck01065/logbert/bert/WEB1_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 16 "cuda:0"
python src/test.py ./paracheck01065/logbert/bert/WEB2_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 16 "cuda:0"
python src/test.py ./paracheck01065/logbert/bert/WEB3_3_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 16 "cuda:0"
python src/test.py ./paracheck01065/logbert/bert/WEB1_4_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 16 "cuda:0"
python src/test.py ./paracheck01065/logbert/bert/WEB2_4_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 16 "cuda:0"
python src/test.py ./paracheck01065/logbert/bert/WEB3_4_train6test4/seq_len_128/r_seed_31/weights/ValTotalbest.pth 16 "cuda:0"


