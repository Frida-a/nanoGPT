

# demo of training char-encoding baby GPT from scratch
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --compile=False >& logs/shakespeare_char_m.txt
# train a baby moe from scratch
python train-moe.py config/train_shakespeare_char_moe.py --compile=False >& logs/shakespeare_char_moe_m.txt

# TODO: training from scratch using 

# mxfp4
# shakespeare_char
# train from scratch
python train-moe.py config/train_shakespeare_char_moe_mxfp4.py --compile=False  >& logs/shakespeare_char_moe_mxfp4_wo_head_gate_m.txt

# train from scratch moe



## shakespeare dataset
python train.py config/train_shakespeare.py --compile=False >& logs/shakespeare_m.txt
python train-moe.py config/train_shakespeare_moe.py --compile=False >& logs/shakespeare_moe_m.txt
python train-moe.py config/train_shakespeare_moe_mxfp4.py --compile=False  >& logs/shakespeare_moe_mxfp4_wo_head_gate_m.txt
