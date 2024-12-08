#!/bin/bash

#### Training quantized model from scratch with Int8

scratch_out_dir="out/richard/from-scratch" &&

# Train on Shakespeare character-level dataset
python data/shakespeare_char/prepare.py && 

mkdir -p "$scratch_out_dir/out-moe-shakespeare-char-8-bit" && 

echo "Created dir $scratch_out_dir/out-moe-shakespeare-char-8-bit" &&

python train-moe.py config/train_moe_shakespeare_char_8_bit.py --compile=False --out_dir="$scratch_out_dir/out-moe-shakespeare-char-8-bit" &&

echo "done."