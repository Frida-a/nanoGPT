

# demo of training char-encoding baby GPT from scratch
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --compile=False
# train a baby moe from scratch
python train-moe.py config/train_shakespeare_char_moe.py --compile=False

# TODO: training from scratch using 

# mxfp4
# shakespeare_char
# train from scratch
python train-moe.py config/train_shakespeare_char_moe_mxfp4.py --compile=False

# train from scratch moe


# exper:
# whether to quantize linear layer in gate
