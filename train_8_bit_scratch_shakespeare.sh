#!/bin/bash

scratch_out_dir="out/richard/from-scratch" &&

python data/shakespeare/prepare.py &&

mkdir -p "$scratch_out_dir/out-shakespeare-8-bit" &&

echo "Created dir $scratch_out_dir/out-shakespeare-8-bit" &&

python train_8_bit.py config/train_shakespeare_8_bit.py --compile=False --out_dir="$scratch_out_dir/out-shakespeare-8-bit" &&

echo "done."