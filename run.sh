

# demo of training char-encoding baby GPT from scratch
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --compile=False
# train a baby moe from scratch
python train-moe.py config/train_shakespeare_char_moe.py --compile=False

# TODO: training from scratch using 

