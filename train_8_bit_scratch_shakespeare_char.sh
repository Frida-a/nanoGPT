#!/bin/bash

scratch_out_dir="out/richard/from-scratch" &&

python data/shakespeare_char/prepare.py &&

mkdir -p "$scratch_out_dir/out-shakespeare-char-8-bit" &&

echo "Created dir $scratch_out_dir/out-shakespeare-char-8-bit" &&

python train_8_bit.py config/train_shakespeare_char_8_bit.py --compile=False --out_dir="$scratch_out_dir/out-shakespeare-char-8-bit" &&

echo "done."